# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import threading
from typing import Any

import torch
import torch.distributed as dist

import vllm.envs as envs
from vllm.distributed import get_dp_group, get_ep_group
from vllm.forward_context import get_forward_context
from vllm.logger import init_logger
from vllm.utils.flashinfer import (
    has_flashinfer_nvlink_one_sided,
    has_flashinfer_nvlink_two_sided,
)
from vllm.utils.import_utils import has_deep_ep, has_mori, has_veloci_deepep

from .base_device_communicator import All2AllManagerBase, Cache

if has_flashinfer_nvlink_two_sided():
    from flashinfer.comm import Mapping  # type: ignore[import-not-found]
    from flashinfer.comm.mnnvl import MnnvlConfig  # type: ignore[import-not-found]
    from flashinfer.comm.trtllm_alltoall import (
        MnnvlMoe,  # type: ignore[import-not-found]
    )

if has_flashinfer_nvlink_one_sided():
    from flashinfer.comm import Mapping  # type: ignore[import-not-found]
    from flashinfer.comm.mnnvl import MnnvlConfig  # type: ignore[import-not-found]
    from flashinfer.comm.trtllm_moe_alltoall import (
        MoeAlltoAll,  # type: ignore[import-not-found]
        moe_a2a_get_workspace_size_per_rank,
    )


logger = init_logger(__name__)


class NaiveAll2AllManager(All2AllManagerBase):
    """
    A naive implementation of all2all communication.
    It uses all-reduce under the hood, which is not
    efficient at all. The main purpose is for testing and
    debugging.
    """

    def __init__(self, cpu_group, tcp_store_group=None):
        super().__init__(cpu_group, tcp_store_group)

    def naive_multicast(
        self,
        x: torch.Tensor,
        cu_tokens_across_sp_cpu: torch.Tensor,
        is_sequence_parallel: bool,
    ) -> torch.Tensor:
        assert len(x.shape) == 2
        buffer = torch.empty(
            (cu_tokens_across_sp_cpu[-1], x.size(1)), device=x.device, dtype=x.dtype
        )

        rank = self.rank if is_sequence_parallel else self.dp_rank
        world_size = self.world_size if is_sequence_parallel else self.dp_world_size

        start = 0 if rank == 0 else cu_tokens_across_sp_cpu[rank - 1]
        end = cu_tokens_across_sp_cpu[rank]
        buffer[start:end, :].copy_(x)
        for idx in range(world_size):
            start = 0 if idx == 0 else cu_tokens_across_sp_cpu[idx - 1]
            end = cu_tokens_across_sp_cpu[idx]
            get_ep_group().broadcast(buffer[start:end, :], idx)

        return buffer

    def dispatch_router_logits(
        self,
        hidden_states: torch.Tensor,
        router_logits: torch.Tensor,
        is_sequence_parallel: bool = False,
        extra_tensors: list[torch.Tensor] | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if extra_tensors is not None:
            raise NotImplementedError(
                "extra_tensors is not supported for NaiveAll2AllManager"
            )
        sp_size = self.tp_group.world_size if is_sequence_parallel else 1
        dp_metadata = get_forward_context().dp_metadata
        assert dp_metadata is not None
        cu_tokens_across_sp_cpu = dp_metadata.cu_tokens_across_sp(sp_size)

        hidden_states = self.naive_multicast(
            hidden_states, cu_tokens_across_sp_cpu, is_sequence_parallel
        )
        router_logits = self.naive_multicast(
            router_logits, cu_tokens_across_sp_cpu, is_sequence_parallel
        )

        return hidden_states, router_logits

    def dispatch(
        self,
        hidden_states: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
        is_sequence_parallel: bool = False,
        extra_tensors: list[torch.Tensor] | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if extra_tensors is not None:
            raise NotImplementedError(
                "extra_tensors is not supported for NaiveAll2AllManager"
            )
        sp_size = self.tp_group.world_size if is_sequence_parallel else 1
        dp_metadata = get_forward_context().dp_metadata
        assert dp_metadata is not None
        cu_tokens_across_sp_cpu = dp_metadata.cu_tokens_across_sp(sp_size)

        hidden_states = self.naive_multicast(
            hidden_states, cu_tokens_across_sp_cpu, is_sequence_parallel
        )
        topk_weights = self.naive_multicast(
            topk_weights, cu_tokens_across_sp_cpu, is_sequence_parallel
        )
        topk_ids = self.naive_multicast(
            topk_ids, cu_tokens_across_sp_cpu, is_sequence_parallel
        )
        return hidden_states, topk_weights, topk_ids

    def combine(
        self, hidden_states: torch.Tensor, is_sequence_parallel: bool = False
    ) -> torch.Tensor:
        ep_rank = self.rank if is_sequence_parallel else self.dp_rank

        dp_metadata = get_forward_context().dp_metadata
        assert dp_metadata is not None
        sp_size = self.tp_group.world_size if is_sequence_parallel else 1
        cu_tokens_across_sp_cpu = dp_metadata.cu_tokens_across_sp(sp_size)

        start = 0 if ep_rank == 0 else cu_tokens_across_sp_cpu[ep_rank - 1]
        end = cu_tokens_across_sp_cpu[ep_rank]

        all_hidden_states = get_ep_group().all_reduce(hidden_states)
        hidden_states = all_hidden_states[start:end, :]
        return hidden_states

    def destroy(self):
        pass


class AgRsAll2AllManager(All2AllManagerBase):
    """
    An implementation of all2all communication based on
    all-gather (dispatch) and reduce-scatter (combine).
    """

    def __init__(self, cpu_group, tcp_store_group=None):
        super().__init__(cpu_group, tcp_store_group)

    def dispatch_router_logits(
        self,
        hidden_states: torch.Tensor,
        router_logits: torch.Tensor,
        is_sequence_parallel: bool = False,
        extra_tensors: list[torch.Tensor] | None = None,
    ) -> (
        tuple[torch.Tensor, torch.Tensor]
        | tuple[torch.Tensor, torch.Tensor, list[torch.Tensor]]
    ):
        """
        Gather hidden_states and router_logits from all dp ranks.
        """
        dp_metadata = get_forward_context().dp_metadata
        assert dp_metadata is not None
        sizes = dp_metadata.get_chunk_sizes_across_dp_rank()
        assert sizes is not None
        dist_group = get_ep_group() if is_sequence_parallel else get_dp_group()
        assert sizes[dist_group.rank_in_group] == hidden_states.shape[0]

        tensors_to_gather = [hidden_states, router_logits]
        if extra_tensors is not None:
            tensors_to_gather.extend(extra_tensors)

        gathered_tensors = dist_group.all_gatherv(
            tensors_to_gather,
            dim=0,
            sizes=sizes,
        )

        if extra_tensors is not None:
            return (gathered_tensors[0], gathered_tensors[1], gathered_tensors[2:])
        return gathered_tensors[0], gathered_tensors[1]

    def dispatch(
        self,
        hidden_states: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
        is_sequence_parallel: bool = False,
        extra_tensors: list[torch.Tensor] | None = None,
    ) -> (
        tuple[torch.Tensor, torch.Tensor, torch.Tensor]
        | tuple[torch.Tensor, torch.Tensor, torch.Tensor, list[torch.Tensor]]
    ):
        """
        Gather hidden_states and router_logits from all dp ranks.
        """
        dp_metadata = get_forward_context().dp_metadata
        assert dp_metadata is not None
        sizes = dp_metadata.get_chunk_sizes_across_dp_rank()
        assert sizes is not None
        dist_group = get_ep_group() if is_sequence_parallel else get_dp_group()
        assert sizes[dist_group.rank_in_group] == hidden_states.shape[0]
        tensors_to_gather = [hidden_states, topk_weights, topk_ids]
        if extra_tensors is not None:
            tensors_to_gather.extend(extra_tensors)

        gathered_tensors = dist_group.all_gatherv(
            tensors_to_gather,
            dim=0,
            sizes=sizes,
        )

        hidden_states = gathered_tensors[0]
        topk_weights = gathered_tensors[1]
        topk_ids = gathered_tensors[2]

        if extra_tensors is None:
            return hidden_states, topk_weights, topk_ids

        return hidden_states, topk_weights, topk_ids, gathered_tensors[3:]

    def combine(
        self, hidden_states: torch.Tensor, is_sequence_parallel: bool = False
    ) -> torch.Tensor:
        """
        Reduce-scatter hidden_states across all dp ranks.
        """
        dp_metadata = get_forward_context().dp_metadata
        assert dp_metadata is not None
        sizes = dp_metadata.get_chunk_sizes_across_dp_rank()
        assert sizes is not None

        dist_group = get_ep_group() if is_sequence_parallel else get_dp_group()
        hidden_states = dist_group.reduce_scatterv(hidden_states, dim=0, sizes=sizes)
        return hidden_states

    def destroy(self):
        pass


class DeepEPAll2AllManagerBase(All2AllManagerBase):
    """
    All2All communication based on DeepEP High-Throughput kernels.
    """

    def __init__(self, cpu_group, tcp_store_group=None):
        assert has_deep_ep(), (
            "DeepEP kernels not found. Please follow https://github.com/vllm-project/vllm/blob/main/tools/ep_kernels/README.md"
            " to install DeepEP kernels."
        )  # noqa
        super().__init__(cpu_group, tcp_store_group)
        self.handle_cache = Cache()

        # This is the DeepEP default. Stick to it till we can establish
        # reasonable defaults based on profiling.
        self.num_sms = 20

    def get_handle(self, kwargs):
        raise NotImplementedError

    def dispatch_router_logits(
        self,
        hidden_states: torch.Tensor,
        router_logits: torch.Tensor,
        is_sequence_parallel: bool = False,
        extra_tensors: list[torch.Tensor] | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        raise NotImplementedError

    def dispatch(
        self,
        hidden_states: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
        is_sequence_parallel: bool = False,
        extra_tensors: list[torch.Tensor] | None = None,
    ) -> (
        tuple[torch.Tensor, torch.Tensor, torch.Tensor]
        | tuple[torch.Tensor, torch.Tensor, torch.Tensor, list[torch.Tensor]]
    ):
        raise NotImplementedError

    def combine(
        self, hidden_states: torch.Tensor, is_sequence_parallel: bool = False
    ) -> torch.Tensor:
        raise NotImplementedError

    def destroy(self):
        with self.handle_cache._lock:
            for _, handle in self.handle_cache._cache.items():
                handle.destroy()
            self.handle_cache._cache.clear()


class DeepEPHTAll2AllManager(DeepEPAll2AllManagerBase):
    """
    All2All communication based on DeepEP High-Throughput kernels.
    """

    def __init__(self, cpu_group, tcp_store_group=None):
        super().__init__(cpu_group, tcp_store_group)

    def _make_all2all_kwargs(self) -> dict[Any, Any]:
        # Defaults for internode and intranode are taken from DeepEP tests.
        num_nvl_bytes = envs.VLLM_DEEPEP_BUFFER_SIZE_MB * 1024 * 1024
        num_rdma_bytes = None
        num_qps_per_rank = None

        if self.internode and not envs.VLLM_DEEPEP_HIGH_THROUGHPUT_FORCE_INTRA_NODE:
            num_rdma_bytes = envs.VLLM_DEEPEP_BUFFER_SIZE_MB * 1024 * 1024
            num_qps_per_rank = self.num_sms // 2
        else:
            num_rdma_bytes = 0
            num_qps_per_rank = 1

        assert num_rdma_bytes is not None
        assert num_qps_per_rank is not None
        return dict(
            group=self.cpu_group,
            num_nvl_bytes=num_nvl_bytes,
            num_rdma_bytes=num_rdma_bytes,
            low_latency_mode=False,
            num_qps_per_rank=num_qps_per_rank,
            explicitly_destroy=True,
        )

    def get_handle(self, kwargs):
        assert len(kwargs) == 0, (
            "DeepEPHTAll2AllManager expects no arguments. All the required "
            "args are computed in the Manager itself."
        )

        import deep_ep  # type: ignore[import-not-found]

        buffer_kwargs = self._make_all2all_kwargs()
        logger.debug("DeepEP all2all args %s", buffer_kwargs)
        handle: deep_ep.Buffer = self.handle_cache.get_or_create(
            buffer_kwargs, deep_ep.Buffer
        )
        return handle

    def set_num_sms(self, num_sms: int):
        import deep_ep  # type: ignore[import-not-found]

        # Right now the buffers are sized for only what the kernels were
        # created with. So we can only reduce the number of SMS used
        # but not increase it.
        if num_sms > self.num_sms:
            num_sms = self.num_sms
        deep_ep.Buffer.set_num_sms(num_sms)


class DeepEPLLAll2AllManager(DeepEPAll2AllManagerBase):
    """
    All2All communication based on DeepEP Low-Latency kernels.
    """

    def __init__(self, cpu_group, tcp_store_group=None):
        super().__init__(cpu_group, tcp_store_group)

    def _make_all2all_kwargs(
        self,
        max_num_tokens_per_dp_rank: int,
        token_hidden_size: int,
        num_ep_ranks: int,
        num_global_experts: int,
        num_local_experts: int,
    ) -> dict[Any, Any]:
        """
        max_num_tokens_per_dp_rank : the maximum number of tokens a DP rank
          can dispatch all the ranks must hold the same value.
        token_hidden_size: the hidden dimension of each token.
        num_ep_ranks: the number of EP group ranks.
        num_global_experts: Number of experts in the model.
        num_local_experts: Number of experts in an EP rank.
        """
        import deep_ep  # type: ignore[import-not-found]

        # Defaults for internode and intranode are taken from DeepEP tests.
        num_nvl_bytes = envs.VLLM_DEEPEP_BUFFER_SIZE_MB * 1024 * 1024
        num_qps_per_rank = num_local_experts
        num_rdma_bytes = deep_ep.Buffer.get_low_latency_rdma_size_hint(
            num_max_dispatch_tokens_per_rank=max_num_tokens_per_dp_rank,
            hidden=token_hidden_size,
            num_ranks=num_ep_ranks,
            num_experts=num_global_experts,
        )

        assert num_rdma_bytes is not None
        return dict(
            group=self.cpu_group,
            num_nvl_bytes=num_nvl_bytes,
            num_rdma_bytes=num_rdma_bytes,
            low_latency_mode=True,
            num_qps_per_rank=num_qps_per_rank,
            allow_nvlink_for_low_latency_mode=True,
            allow_mnnvl=envs.VLLM_DEEPEP_LOW_LATENCY_USE_MNNVL,
            explicitly_destroy=True,
        )

    def get_handle(self, kwargs):
        """
        The kwargs for DeepEPLLAll2AllManager is dictated by
        _make_all2all_kwargs.
        """
        import deep_ep  # type: ignore[import-not-found]

        buffer_kwargs = self._make_all2all_kwargs(**kwargs)
        logger.debug("DeepEP all2all args %s", buffer_kwargs)
        handle: deep_ep.Buffer = self.handle_cache.get_or_create(
            buffer_kwargs, deep_ep.Buffer
        )
        return handle

    # DeepEP LL uses RDMA so no SMs are used for communication
    def max_sms_used(self) -> int | None:
        return 0


class NixlEPAll2AllManager(All2AllManagerBase):
    """
    All2All communication based on NIXL EP kernels.
    This backend supports elastic EP with dynamic rank connection/disconnection.
    """

    # (nixl_ep_buffer, ep_size)
    _buffer: tuple[Any, int] | None = None
    _lock = threading.Lock()

    def __init__(self, cpu_group, tcp_store_group=None):
        super().__init__(cpu_group, tcp_store_group)

        self.max_num_ep_ranks = envs.VLLM_NIXL_EP_MAX_NUM_RANKS

    def _init_buffer(
        self,
        max_num_tokens_per_dp_rank: int,
        token_hidden_size: int,
        num_experts_per_rank: int,
    ) -> None:
        from nixl_ep import Buffer  # type: ignore[import-not-found]

        max_num_global_experts = self.max_num_ep_ranks * num_experts_per_rank
        num_rdma_bytes = Buffer.get_rdma_size_hint(
            num_max_dispatch_tokens_per_rank=max_num_tokens_per_dp_rank,
            hidden=token_hidden_size,
            num_ranks=self.max_num_ep_ranks,
            num_experts=max_num_global_experts,
        )
        assert NixlEPAll2AllManager._buffer is None, (
            "NIXL EP buffer already initialized"
        )
        buffer = Buffer(
            rank=self.rank,
            tcp_store_group=self.tcp_store_group.store,
        )
        buffer.update_memory_buffers(
            num_ranks=self.max_num_ep_ranks,
            num_experts_per_rank=num_experts_per_rank,
            num_rdma_bytes=num_rdma_bytes,
        )
        ranks_to_connect = list(range(self.cpu_group.size()))
        buffer.connect_ranks(ranks_to_connect)
        NixlEPAll2AllManager._buffer = (buffer, self.cpu_group.size())

    def _update_buffer(self):
        assert NixlEPAll2AllManager._buffer is not None
        buffer, current_ep_size = NixlEPAll2AllManager._buffer
        current_ranks = list(range(current_ep_size))
        new_ep_size = self.cpu_group.size()
        buffer.set_tcp_store_group(self.tcp_store_group.store)
        if new_ep_size > len(current_ranks):
            ranks_to_connect = list(range(len(current_ranks), new_ep_size))
            buffer.connect_ranks(ranks_to_connect)
        else:
            ranks_to_disconnect = current_ranks[new_ep_size:]
            buffer.disconnect_ranks(ranks_to_disconnect)
        NixlEPAll2AllManager._buffer = (buffer, new_ep_size)

    def get_handle(self, kwargs):
        with NixlEPAll2AllManager._lock:
            if (
                NixlEPAll2AllManager._buffer is not None
                and NixlEPAll2AllManager._buffer[1] == self.cpu_group.size()
            ):
                return NixlEPAll2AllManager._buffer[0]

            num_experts_per_rank = (
                kwargs["num_global_experts"] // kwargs["num_ep_ranks"]
            )
            nixl_kwargs = dict(
                max_num_tokens_per_dp_rank=kwargs["max_num_tokens_per_dp_rank"],
                token_hidden_size=kwargs["token_hidden_size"],
                num_experts_per_rank=num_experts_per_rank,
            )
            if NixlEPAll2AllManager._buffer is None:
                self._init_buffer(**nixl_kwargs)
            else:
                self._update_buffer()

            assert NixlEPAll2AllManager._buffer is not None
            handle = NixlEPAll2AllManager._buffer[0]
            return handle

    def dispatch(
        self,
        hidden_states: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
        is_sequence_parallel: bool = False,
        extra_tensors: list[torch.Tensor] | None = None,
    ) -> (
        tuple[torch.Tensor, torch.Tensor, torch.Tensor]
        | tuple[torch.Tensor, torch.Tensor, torch.Tensor, list[torch.Tensor]]
    ):
        raise NotImplementedError

    def combine(
        self, hidden_states: torch.Tensor, is_sequence_parallel: bool = False
    ) -> torch.Tensor:
        raise NotImplementedError

    def destroy(self):
        # NOTE(yongji): NIXLEPAll2AllManager instance is recreated during
        # scale-up/down, so we cannot destroy the persistent buffer here.
        assert NixlEPAll2AllManager._buffer is not None
        buffer = NixlEPAll2AllManager._buffer[0]
        buffer.set_tcp_store_group(None)

    # NIXL EP uses RDMA so no SMs are used for communication
    def max_sms_used(self) -> int | None:
        return 0


class FlashInferNVLinkTwoSidedManager(All2AllManagerBase):
    """
    All2All communication based on flashinfer all2allv/two-sided NVLink kernels.
    """

    # This type lint could be removed after all of the work in
    # https://github.com/vllm-project/vllm/issues/26533 done.
    rank: int
    world_size: int

    def __init__(self, cpu_group, tcp_store_group=None):
        assert has_flashinfer_nvlink_two_sided(), (
            "flashinfer all2all module not found. Please install/check flashinfer"
        )  # noqa
        super().__init__(cpu_group, tcp_store_group)
        logger.debug(
            "Initialize for flashinfer All2All rank=%d, world size=%d",
            self.rank,
            self.world_size,
        )
        self.initialized = False
        self.alltoall_info = None

    def initialize(
        self,
        world_size: int,
        rank: int,
        gpus_per_node: int,
    ):
        """Initialize workspace"""
        if self.initialized:
            return

        self.cleanup()
        logger.debug("making map: rank=%d, world size=%d", rank, world_size)
        self.mapping = Mapping(
            world_size,
            rank,
            gpus_per_node,
            tp_size=world_size,
        )

        from vllm.distributed.device_communicators.mnnvl_compat import (
            CustomCommunicator,
        )

        dp_config = MnnvlConfig(
            comm_backend=CustomCommunicator(get_dp_group().cpu_group),
            fabric_page_size=1 << 29,  # 512MB
            allocation_granularity=0,  # Auto-detect
        )

        self.workspace_tensor = MnnvlMoe.get_moe_workspaces(self.mapping, dp_config)
        self.prepare_workspace_tensor = MnnvlMoe.get_moe_prepare_workspace(
            self.mapping, dp_config
        )

        self.world_size = world_size
        self.rank = rank
        self.gpus_per_node = gpus_per_node
        self.initialized = True

        logger.info(
            "FlashInfer All2All initialized for rank %s, size %s", rank, world_size
        )

    def ensure_alltoall_workspace_initialized(self):
        """Ensure workspace is initialized"""
        if not has_flashinfer_nvlink_two_sided():
            return False

        if self.world_size <= 1:
            return False

        if not self.initialized:
            self.initialize(
                world_size=self.world_size,
                rank=self.rank,
                gpus_per_node=torch.accelerator.device_count,
            )
        return self.initialized

    def get_handle(self, kwargs):
        return self

    def cleanup(self):
        """Clean up workspace"""
        if (
            self.initialized
            and self.workspace_tensor is not None
            and self.prepare_workspace_tensor is not None
        ):
            try:
                del self.workspace_tensor
                del self.prepare_workspace_tensor
            except Exception as e:
                logger.warning("Failed to cleanup FlashInfer workspace: %s", e)
            finally:
                self.workspace_tensor = None
                self.prepare_workspace_tensor = None
                self.mapping = None
                self.initialized = False


class FlashInferNVLinkOneSidedManager(All2AllManagerBase):
    """
    All2All communication based on FlashInfer's MoeAlltoAll/One-sided NVLink kernel.
    This is a newer kernel from trtllm that should perform better than the kernel
    used by flashinfer_nvlink_two_sided.
    """

    rank: int
    world_size: int

    def __init__(self, cpu_group):
        assert has_flashinfer_nvlink_one_sided(), (
            "flashinfer trtllm_moe_alltoall module not found. "
            "Please install/check flashinfer"
        )
        super().__init__(cpu_group)
        logger.debug(
            "Initialize FlashInfer One-sided NVLink rank=%d, world size=%d",
            self.rank,
            self.world_size,
        )
        self.initialized = False
        self.moe_alltoall: MoeAlltoAll | None = None
        self.mapping = None

    def initialize(
        self,
        max_num_tokens: int,
        top_k: int,
        num_experts: int,
        hidden_size: int,
    ):
        """Initialize the MoeAlltoAll workspace."""
        if self.initialized:
            return

        self.cleanup()
        gpus_per_node = torch.accelerator.device_count()
        logger.debug(
            "Making One-sided NVLink mapping: rank=%d, world size=%d",
            self.rank,
            self.world_size,
        )
        self.mapping = Mapping(
            self.world_size,
            self.rank,
            gpus_per_node,
            tp_size=self.world_size,
            moe_ep_size=self.world_size,
        )

        from vllm.distributed.device_communicators.mnnvl_compat import (
            CustomCommunicator,
        )

        dp_config = MnnvlConfig(
            comm_backend=CustomCommunicator(get_dp_group().cpu_group),
        )
        total_dispatch_payload_size_per_token = (
            hidden_size // 2  # nvfp4 hidden states
            + hidden_size // 16  # fp8 scaling factors
            + top_k * 4  # int32 topks ids
            + top_k * 4  # float32 topk weights
        )
        combine_payload_size_per_token = hidden_size * 2  # bf16 hidden states
        self.workspace_size = moe_a2a_get_workspace_size_per_rank(
            ep_size=self.world_size,
            max_num_tokens=max_num_tokens,
            total_dispatch_payload_size_per_token=total_dispatch_payload_size_per_token,
            combine_payload_size_per_token=combine_payload_size_per_token,
        )

        self.moe_alltoall = MoeAlltoAll(
            mapping=self.mapping,
            max_num_tokens=max_num_tokens,
            top_k=top_k,
            num_experts=num_experts,
            workspace_size_per_rank=self.workspace_size,
            mnnvl_config=dp_config,
        )

        self.gpus_per_node = gpus_per_node
        self.max_num_tokens = max_num_tokens
        self.top_k = top_k
        self.num_experts = num_experts
        self.hidden_size = hidden_size
        self.initialized = True

        logger.info(
            "FlashInfer One-sided NVLink initialized for rank %s, size %s",
            self.rank,
            self.world_size,
        )
        dist.barrier()

    def get_handle(self, kwargs):
        return self

    def cleanup(self):
        """Clean up resources."""
        if self.initialized and self.moe_alltoall is not None:
            try:
                del self.moe_alltoall
            except Exception as e:
                logger.warning(
                    "Failed to cleanup FlashInfer One-sided NVLink workspace: %s", e
                )
            finally:
                self.moe_alltoall = None
                self.mapping = None
                self.initialized = False


class MoriAll2AllManager(All2AllManagerBase):
    def __init__(self, cpu_group):
        assert has_mori(), (
            "MoRI kernels not found. Please follow https://github.com/ROCm/mori/blob/main/README.md"
            " to install MoRI kernels."
        )  # noqa
        import mori

        super().__init__(cpu_group)
        self.handle_cache = Cache()

        torch._C._distributed_c10d._register_process_group("mori", cpu_group)
        mori.shmem.shmem_torch_process_group_init("mori")

    def _make_all2all_kwargs(
        self,
        rank: int,
        num_ep_ranks: int,
        input_dtype: torch.dtype,
        quant_dtype: torch.dtype,
        token_hidden_size: int,
        scale_dim: int,
        scale_type_size: int,
        max_num_tokens_per_dp_rank: int,
        num_local_experts: int,
        num_experts_per_token: int,
    ):
        import mori  # type: ignore[import-not-found]

        from vllm.platforms.rocm import on_gfx942, on_gfx950

        assert on_gfx942() or on_gfx950(), (
            "mori currently only support arch gfx942 and gfx950"
        )

        if not self.internode:
            # single node
            kernel_type = mori.ops.EpDispatchCombineKernelType.IntraNode
            rdma_block_num = 0
            warp_num_per_block = 16
            block_num = 80
        else:
            # multi node
            kernel_type = mori.ops.EpDispatchCombineKernelType.InterNodeV1
            if on_gfx942():
                warp_num_per_block = 16
                block_num = 32
                rdma_block_num = 16
            elif on_gfx950():
                warp_num_per_block = 8
                block_num = 64
                rdma_block_num = 32
            else:
                raise NotImplementedError(
                    "mori currently only support arch gfx942 and gfx950"
                )

        return dict(
            rank=rank,
            world_size=num_ep_ranks,
            data_type=quant_dtype,
            hidden_dim=token_hidden_size,
            scale_dim=scale_dim,
            scale_type_size=scale_type_size,
            max_token_type_size=input_dtype.itemsize,
            max_num_inp_token_per_rank=max_num_tokens_per_dp_rank,
            num_experts_per_rank=num_local_experts,
            num_experts_per_token=num_experts_per_token,
            warp_num_per_block=warp_num_per_block,
            block_num=block_num,
            kernel_type=kernel_type,
            rdma_block_num=rdma_block_num,
            gpu_per_node=min(8, num_ep_ranks),
        )

    def _make_handle(self, **kwargs):
        import mori  # type: ignore[import-not-found]

        mori_config = mori.ops.EpDispatchCombineConfig(**kwargs)
        handle = mori.ops.EpDispatchCombineOp(mori_config)
        return handle

    def get_handle(self, kwargs):
        import mori  # type: ignore[import-not-found]

        mori_kwargs = self._make_all2all_kwargs(**kwargs)
        logger.debug("MoRI all2all args %s", mori_kwargs)
        handle: mori.ops.EpDispatchCombineOp = self.handle_cache.get_or_create(
            mori_kwargs, self._make_handle
        )
        return handle


class VelociDeepEPFallbackAll2AllManager(All2AllManagerBase):
    """
    True routing-based all-to-all for MoE expert parallelism.

    Each token is sent only to the EP ranks whose local experts are
    selected by the router.  Different ranks exchange different numbers
    of tokens (variable-size all-to-all).

    Communication uses ``dist.all_to_all_single`` for both count exchange
    and data exchange.  State from ``dispatch()`` (routing metadata) is
    kept in ``_state`` for use by ``combine()``.

    Note: ``dist.all_to_all_single`` may be broken on certain backends
    (e.g. XCCL sub-groups on Intel XPU).  This manager is the
    *reference* implementation; use the AGRS fallback if the backend
    has known all_to_all issues.
    """

    def __init__(self, cpu_group, tcp_store_group=None):
        super().__init__(cpu_group, tcp_store_group)
        print(f"Using velocideep fallback All2All (oneccl) manager. This may be slow...")  # noqa
        # Routing metadata carried from dispatch → combine.
        self._state: dict | None = None
        # Set via get_handle() before first dispatch.
        self._num_local_experts: int = 0

    def get_handle(self, kwargs):
        """Store model-specific config (num_experts)."""
        num_experts = kwargs.get("num_experts", 0)
        if num_experts > 0:
            self._num_local_experts = (
                num_experts // self.world_size
            )
        return self

    @staticmethod
    def _a2a_exchange(
        send_tensor: torch.Tensor,
        send_counts: list[int],
        recv_counts: list[int],
        group,
    ) -> torch.Tensor:
        """Variable-size exchange via dist.all_to_all_single."""
        total_recv = sum(recv_counts)
        total_send = sum(send_counts)
        row_shape = send_tensor.shape[1:]
        # Compute elements-per-row from tensor shape (not from numel ratio,
        # which breaks when shape[0]==0).
        cols = 1
        for s in row_shape:
            cols *= s
        if total_send == 0 and total_recv == 0:
            return torch.empty(
                [0] + list(row_shape),
                dtype=send_tensor.dtype, device=send_tensor.device,
            )
        flat_send = send_tensor.reshape(-1)
        send_splits = [c * cols for c in send_counts]
        recv_splits = [c * cols for c in recv_counts]
        flat_recv = torch.empty(
            sum(recv_splits),
            dtype=send_tensor.dtype, device=send_tensor.device,
        )
        dist.all_to_all_single(
            flat_recv, flat_send,
            output_split_sizes=recv_splits,
            input_split_sizes=send_splits,
            group=group,
        )
        return flat_recv.reshape([total_recv] + list(row_shape))

    def dispatch(
        self,
        hidden_states: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
        is_sequence_parallel: bool = False,
        extra_tensors: list[torch.Tensor] | None = None,
    ) -> (
        tuple[torch.Tensor, torch.Tensor, torch.Tensor]
        | tuple[torch.Tensor, torch.Tensor, torch.Tensor, list[torch.Tensor]]
    ):
        """Route tokens to EP ranks based on expert assignment."""
        dist_group = (
            get_ep_group() if is_sequence_parallel else get_dp_group()
        )
        world_size = dist_group.world_size
        group = dist_group.device_group
        my_rank = dist_group.rank_in_group
        num_local_experts = self._num_local_experts
        assert num_local_experts > 0, (
            "VelociDeepEPFallbackAll2AllManager: num_local_experts not set. "
            "Call get_handle({'num_experts': N}) first."
        )
        num_tokens = hidden_states.shape[0]
        # print(f"{topk_ids.shape=},{topk_ids=}")
        # --- Determine destination ranks ---
        expert_to_rank = topk_ids.clamp(min=0) // num_local_experts
        valid_mask = topk_ids >= 0

        dest_mask = torch.zeros(
            num_tokens, world_size, dtype=torch.bool,
            device=hidden_states.device,
        )
        for r in range(world_size):
            dest_mask[:, r] = (
                (expert_to_rank == r) & valid_mask
            ).any(dim=1)
        # print(f"{dest_mask.shape=},{dest_mask=}")
        # Per-rank send token indices.
        send_token_indices: list[torch.Tensor] = []
        for r in range(world_size):
            send_token_indices.append(
                dest_mask[:, r].nonzero(as_tuple=True)[0]
            )
        send_counts = [idx.shape[0] for idx in send_token_indices]
        # print(f"{send_counts}")
        # print(f"{hidden_states=}")
        # --- Exchange counts via all_gather ---
        send_counts_t = torch.tensor(
            send_counts, dtype=torch.int64, device=hidden_states.device,
        )
        # print(f"{send_counts_t=}")
        all_counts = dist_group.all_gatherv(
            send_counts_t, dim=0,
            sizes=[send_counts_t.shape[0]] * world_size,
        )
        # Shape: [world_size * world_size]. Reshape to matrix.
        all_counts = all_counts.reshape(world_size, world_size)
        # recv_count from rank r = rank r's send count to us (column my_rank).
        # print(f"{all_counts=}")
        recv_counts = all_counts[:, my_rank].tolist()

        # --- Gather tokens ordered by destination rank ---
        if sum(send_counts) > 0:
            send_indices = torch.cat(send_token_indices)
        else:
            send_indices = torch.empty(
                0, dtype=torch.long, device=hidden_states.device,
            )

        send_hs = hidden_states[send_indices]
        send_tw = topk_weights[send_indices]
        send_ti = topk_ids[send_indices]

        # print(f"{send_hs=},{send_counts=},{recv_counts=},{send_tw=},{send_ti=}")
    
        # --- Exchange data ---
        recv_hs = self._a2a_exchange(send_hs, send_counts, recv_counts, group)
        recv_tw = self._a2a_exchange(send_tw, send_counts, recv_counts, group)
        recv_ti = self._a2a_exchange(send_ti, send_counts, recv_counts, group)


        recv_extra: list[torch.Tensor] | None = None
        if extra_tensors is not None:
            recv_extra = []
            for t in extra_tensors:
                send_t = t[send_indices]
                recv_extra.append(
                    self._a2a_exchange(
                        send_t, send_counts, recv_counts, group,
                    )
                )

        # Save state for combine.
        self._state = {
            "send_counts": send_counts,
            "recv_counts": recv_counts,
            "send_token_indices": send_token_indices,
            "num_tokens": num_tokens,
            "group": group,
        }

        if recv_extra is None:
            return recv_hs, recv_tw, recv_ti
        return recv_hs, recv_tw, recv_ti, recv_extra

    def combine(
        self, hidden_states: torch.Tensor, is_sequence_parallel: bool = False
    ) -> torch.Tensor:
        """Reverse the dispatch routing and sum partial results."""
        state = self._state
        assert state is not None, (
            "VelociDeepEPFallbackAll2AllManager.combine() called without prior dispatch()"
        )
        self._state = None

        # Reverse: during dispatch we received recv_counts[r] from rank r,
        # now we send those results back.
        comb_send_counts = state["recv_counts"]
        comb_recv_counts = state["send_counts"]
        group = state["group"]
        num_tokens = state["num_tokens"]
        send_token_indices = state["send_token_indices"]

        recv_results = self._a2a_exchange(
            hidden_states.contiguous(),
            comb_send_counts, comb_recv_counts, group,
        )

        hidden_dim = hidden_states.shape[-1]
        result = torch.zeros(
            [num_tokens, hidden_dim],
            dtype=hidden_states.dtype, device=hidden_states.device,
        )
        offset = 0
        for r in range(len(comb_recv_counts)):
            count = comb_recv_counts[r]
            if count > 0:
                result.index_add_(
                    0, send_token_indices[r],
                    recv_results[offset:offset + count],
                )
                offset += count

        return result

    def destroy(self):
        pass


class VelociDeepEPFallbackAGRSManager(All2AllManagerBase):
    """
    Fallback All2All manager for VelociDeepEP (AGRS path).

    Uses allgather (dispatch) + reduce-scatter (combine), the same
    well-tested primitives as AgRsAll2AllManager.  This manager is
    used when VLLM_VELOCI_DEEPEP_USE_AGRS=1 is set.
    """

    def __init__(self, cpu_group, tcp_store_group=None):
        print(f"Using velocideep fallback AGRS (oneccl) manager. This may be slow...")
        super().__init__(cpu_group, tcp_store_group)

    def dispatch(
        self,
        hidden_states: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
        is_sequence_parallel: bool = False,
        extra_tensors: list[torch.Tensor] | None = None,
    ) -> (
        tuple[torch.Tensor, torch.Tensor, torch.Tensor]
        | tuple[torch.Tensor, torch.Tensor, torch.Tensor, list[torch.Tensor]]
    ):
        dp_metadata = get_forward_context().dp_metadata
        assert dp_metadata is not None
        sizes = dp_metadata.get_chunk_sizes_across_dp_rank()
        assert sizes is not None
        dist_group = get_ep_group() if is_sequence_parallel else get_dp_group()
        assert sizes[dist_group.rank_in_group] == hidden_states.shape[0]
        tensors = [hidden_states, topk_weights, topk_ids]
        if extra_tensors is not None:
            tensors.extend(extra_tensors)

        # for t in tensors:
        #     print(f"{t.shape=},{t.dtype=}")
        # print(f"{sizes=}")
        gathered = dist_group.all_gatherv(tensors, dim=0, sizes=sizes)

        # --- Diagnostic: test tiny all_gatherv in the same env where AGRS works ---

        my_rank = dist_group.rank_in_group
        ws = dist_group.world_size
        # Test 1: tiny tensor, variable sizes (same as AGRS uses)
        tiny = torch.tensor([my_rank * 10 + 1, my_rank * 10 + 2],
                            dtype=torch.int64, device=hidden_states.device)
        tiny_sizes = [2] * ws  # equal sizes
        tiny_result = dist_group.all_gatherv(tiny, dim=0, sizes=tiny_sizes)
        tiny_expected = torch.cat([
            torch.tensor([r * 10 + 1, r * 10 + 2],
                         dtype=torch.int64, device=hidden_states.device)
            for r in range(ws)
        ])
        ok1 = torch.equal(tiny_result, tiny_expected)
        # Test 2: tiny tensor, no sizes (triggers equal-size path)
        tiny2 = torch.tensor([my_rank * 10 + 1, my_rank * 10 + 2],
                             dtype=torch.int64, device=hidden_states.device)
        tiny_result2 = dist_group.all_gatherv(tiny2, dim=0)
        ok2 = torch.equal(tiny_result2, tiny_expected)
        # print(f"[AGRS DIAG rank={my_rank}] tiny all_gatherv with sizes: "
        #       f"{'PASS' if ok1 else 'FAIL'} result={tiny_result}")
        # print(f"[AGRS DIAG rank={my_rank}] tiny all_gatherv no sizes:   "
        #       f"{'PASS' if ok2 else 'FAIL'} result={tiny_result2}")

        hidden_states = gathered[0]
        topk_weights = gathered[1]
        topk_ids = gathered[2]
        # print(f"{topk_ids=},{topk_ids.shape=},{topk_ids.dtype=}")

        if extra_tensors is None:
            return hidden_states, topk_weights, topk_ids
        return hidden_states, topk_weights, topk_ids, gathered[3:]

    def dispatch_router_logits(
        self,
        hidden_states: torch.Tensor,
        router_logits: torch.Tensor,
        is_sequence_parallel: bool = False,
        extra_tensors: list[torch.Tensor] | None = None,
    ) -> (
        tuple[torch.Tensor, torch.Tensor]
        | tuple[torch.Tensor, torch.Tensor, list[torch.Tensor]]
    ):
        dp_metadata = get_forward_context().dp_metadata
        assert dp_metadata is not None
        sizes = dp_metadata.get_chunk_sizes_across_dp_rank()
        assert sizes is not None
        dist_group = get_ep_group() if is_sequence_parallel else get_dp_group()
        assert sizes[dist_group.rank_in_group] == hidden_states.shape[0]

        tensors = [hidden_states, router_logits]
        if extra_tensors is not None:
            tensors.extend(extra_tensors)

        gathered = dist_group.all_gatherv(tensors, dim=0, sizes=sizes)

        if extra_tensors is not None:
            return (gathered[0], gathered[1], gathered[2:])
        return gathered[0], gathered[1]

    def combine(
        self, hidden_states: torch.Tensor, is_sequence_parallel: bool = False
    ) -> torch.Tensor:
        dp_metadata = get_forward_context().dp_metadata
        assert dp_metadata is not None
        sizes = dp_metadata.get_chunk_sizes_across_dp_rank()
        assert sizes is not None

        dist_group = get_ep_group() if is_sequence_parallel else get_dp_group()
        return dist_group.reduce_scatterv(hidden_states, dim=0, sizes=sizes)

    def destroy(self):
        pass


class VelociDeepEPAll2AllManager(All2AllManagerBase):
    """
    All2All manager for VelociDeepEP using veloci_deepep.Buffer
    in low-latency mode (same API as DeepEPLLAll2AllManager).

    The Buffer is created here and passed to VelociDeepEPPrepareAndFinalize
    which drives the actual low_latency_dispatch/combine calls.

    dispatch/combine/dispatch_router_logits on this manager raise
    NotImplementedError because the PrepareAndFinalize class calls
    buffer.low_latency_dispatch/combine directly.
    """

    def __init__(self, cpu_group, tcp_store_group=None):
        assert has_veloci_deepep(), (
            "veloci_deepep package not found. "
            "Please install veloci_deepep to use the VelociDeepEP backend."
        )
        super().__init__(cpu_group, tcp_store_group)
        self.handle_cache = Cache()

    def _make_all2all_kwargs(
        self,
        max_num_tokens_per_dp_rank: int,
        token_hidden_size: int,
        num_ep_ranks: int,
        num_global_experts: int,
        num_local_experts: int,
    ) -> dict[Any, Any]:
        import veloci_deepep  # type: ignore[import-not-found]

        num_nvl_bytes = envs.VLLM_DEEPEP_BUFFER_SIZE_MB * 1024 * 1024
        num_qps_per_rank = num_local_experts
        num_rdma_bytes = veloci_deepep.Buffer.get_low_latency_rdma_size_hint(
            num_max_dispatch_tokens_per_rank=max_num_tokens_per_dp_rank,
            hidden=token_hidden_size,
            num_ranks=num_ep_ranks,
            num_experts=num_global_experts,
        )

        assert num_rdma_bytes is not None
        return dict(
            group=self.cpu_group,
            num_nvl_bytes=num_nvl_bytes,
            num_rdma_bytes=num_rdma_bytes,
            low_latency_mode=True,
            num_qps_per_rank=num_qps_per_rank,
            allow_nvlink_for_low_latency_mode=True,
            allow_mnnvl=envs.VLLM_DEEPEP_LOW_LATENCY_USE_MNNVL,
            explicitly_destroy=True,
        )

    def get_handle(self, kwargs):
        import veloci_deepep  # type: ignore[import-not-found]

        buffer_kwargs = self._make_all2all_kwargs(**kwargs)
        logger.debug("VelociDeepEP all2all args %s", buffer_kwargs)
        handle: veloci_deepep.Buffer = self.handle_cache.get_or_create(
            buffer_kwargs, veloci_deepep.Buffer
        )
        return handle

    # VelociDeepEP LL uses RDMA so no SMs are used for communication.
    def max_sms_used(self) -> int | None:
        return 0

    def dispatch(self, hidden_states, topk_weights, topk_ids,
                 is_sequence_parallel=False, extra_tensors=None):
        raise NotImplementedError(
            "VelociDeepEPAll2AllManager.dispatch() should not be called "
            "directly. VelociDeepEPPrepareAndFinalize calls "
            "buffer.low_latency_dispatch."
        )

    def dispatch_router_logits(self, hidden_states, router_logits,
                               is_sequence_parallel=False, extra_tensors=None):
        raise NotImplementedError(
            "VelociDeepEPAll2AllManager.dispatch_router_logits() should not "
            "be called directly."
        )

    def combine(self, hidden_states, is_sequence_parallel=False):
        raise NotImplementedError(
            "VelociDeepEPAll2AllManager.combine() should not be called "
            "directly. VelociDeepEPPrepareAndFinalize calls "
            "buffer.low_latency_combine."
        )

    def destroy(self):
        with self.handle_cache._lock:
            for _, handle in self.handle_cache._cache.items():
                handle.destroy()
            self.handle_cache._cache.clear()
