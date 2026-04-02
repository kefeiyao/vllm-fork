# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
VelociDeepEP Prepare/Finalize for MoE dispatch/combine.

Primary path: uses veloci_deepep.Buffer low-latency kernels
(same API as deep_ep.Buffer low-latency interface) for
sparse all-to-all via GPU-Direct Access.

Fallback path: when veloci_deepep is not installed, delegates to the
All2AllManager which uses oneCCL/NCCL dist.all_to_all collectives.
"""
from collections.abc import Callable

import torch

import vllm.model_executor.layers.fused_moe.modular_kernel as mk
from vllm import envs
from vllm.distributed import get_ep_group
from vllm.logger import init_logger
from vllm.model_executor.layers.fused_moe.config import FusedMoEQuantConfig
from vllm.model_executor.layers.fused_moe.topk_weight_and_reduce import (
    TopKWeightAndReduceContiguous,
    TopKWeightAndReduceDelegate,
)
from vllm.model_executor.layers.fused_moe.utils import (
    moe_kernel_quantize_input,
    normalize_batched_scales_shape,
)
from vllm.utils.flashinfer import nvfp4_block_scale_interleave
from vllm.utils.import_utils import has_veloci_deepep
from vllm.v1.worker.ubatching import (
    dbo_current_ubatch_id,
    dbo_enabled,
    dbo_maybe_run_recv_hook,
)

logger = init_logger(__name__)

if has_veloci_deepep():
    import veloci_deepep  # type: ignore[import-not-found]

# VelociDeepEP kernels quantize dispatch inputs in 128 element chunks.
VELOCI_QUANT_BLOCK_SIZE = 128
VELOCI_QUANT_BLOCK_SHAPE = [VELOCI_QUANT_BLOCK_SIZE, VELOCI_QUANT_BLOCK_SIZE]


def _dequant_fp8(
    expert_x_fp8: torch.Tensor, expert_x_scales: torch.Tensor
) -> torch.Tensor:
    """Return dequantized tensor in fp32."""
    assert expert_x_fp8.is_contiguous()
    expert_x_scales = expert_x_scales.contiguous()
    num_experts = expert_x_fp8.size(0)

    expert_x_fp32 = expert_x_fp8.to(torch.float32).view(
        num_experts, -1, VELOCI_QUANT_BLOCK_SIZE
    )
    expert_x_scales = expert_x_scales.view(num_experts, -1, 1)
    return (expert_x_fp32 * expert_x_scales).view(expert_x_fp8.size())


class VelociDeepEPPrepareAndFinalize(mk.FusedMoEPrepareAndFinalizeModular):
    """
    Prepare/Finalize for VelociDeepEP.

    When veloci_deepep is available:
      Uses veloci_deepep.Buffer low-latency dispatch/combine
      (same API as deep_ep.Buffer low-latency interface).
      Supports DBO via prepare_async / finalize_async with
      recv-hook based overlapping.

    When veloci_deepep is NOT available:
      Falls back to the All2AllManager's dispatch/combine which uses
      oneCCL/NCCL dist.all_to_all collectives.
    """

    # Low-latency kernels are compiled only for certain hidden sizes.
    # Keep sorted — maybe_roundup_layer_hidden_size depends on it.
    SUPPORTED_HIDDEN_SIZES = [2048, 2560, 3072, 4096, 5120, 6144, 7168, 8192]

    @staticmethod
    def maybe_roundup_layer_hidden_size(hidden_size: int) -> int:
        _supported_hs = VelociDeepEPPrepareAndFinalize.SUPPORTED_HIDDEN_SIZES
        num_supported_hs = len(_supported_hs)
        assert all(
            [
                _supported_hs[i] < _supported_hs[i + 1]
                for i in range(num_supported_hs - 1)
            ]
        )
        for x in _supported_hs:
            if x >= hidden_size:
                return x
        raise ValueError(
            f"Hidden Size {hidden_size} is greater than the "
            f"maximum supported hidden size {_supported_hs[-1]}"
        )

    def __init__(
        self,
        num_dispatchers: int,
        buffer: "veloci_deepep.Buffer | None" = None,
        max_tokens_per_rank: int = 0,
        use_fp8_dispatch: bool = False,
        is_sequence_parallel: bool = False,
        global_to_physical: torch.Tensor | None = None,
        physical_to_global: torch.Tensor | None = None,
        local_expert_global_ids: torch.Tensor | None = None,
    ) -> None:
        super().__init__()
        self.buffer = buffer
        self.use_veloci_deepep = buffer is not None
        self.max_tokens_per_rank_ = max_tokens_per_rank
        self.num_dispatchers_ = num_dispatchers
        self.use_fp8_dispatch = use_fp8_dispatch
        self.is_sequence_parallel = is_sequence_parallel

        if self.use_veloci_deepep:
            self.handles: list[tuple | None] = [None, None]

            topk_indices_dtype = self.topk_indices_dtype()

            def _maybe_cast(
                tensor: torch.Tensor | None,
            ) -> torch.Tensor | None:
                if tensor is None or topk_indices_dtype is None:
                    return tensor
                return tensor.to(dtype=topk_indices_dtype)

            self.global_to_physical = _maybe_cast(global_to_physical)
            self.physical_to_global = _maybe_cast(physical_to_global)
            self.local_expert_global_ids = _maybe_cast(
                local_expert_global_ids
            )

            self.use_ue8m0_dispatch = False

            logger.info_once(
                "VelociDeepEP: using veloci_deepep.Buffer (low-latency) "
                "for dispatch/combine"
            )
        else:
            self.global_to_physical = None
            self.physical_to_global = None
            self.local_expert_global_ids = None
            logger.info_once(
                "VelociDeepEP: veloci_deepep not available, "
                "falling back to oneCCL/NCCL all_to_all"
            )

    def post_init_setup(self, fused_experts: mk.FusedMoEExperts):
        if not self.use_veloci_deepep:
            return
        if not fused_experts.supports_packed_ue8m0_act_scales():
            return
        if self.use_fp8_dispatch:
            logger.debug_once(
                "Update VelociDeepEPPrepareFinalize to do packed ue8m0 "
                "scales dispatch."
            )
            self.use_ue8m0_dispatch = True
        else:
            logger.warning_once(
                "VelociDeepEPPrepareAndFinalize is setup to dispatch "
                "raw/unquantized activations despite "
                f"({fused_experts.__class__.__name__}) being able "
                "to support quantized activations.",
                scope="local",
            )

    @property
    def activation_format(self) -> mk.FusedMoEActivationFormat:
        if self.use_veloci_deepep:
            return mk.FusedMoEActivationFormat.BatchedExperts
        return mk.FusedMoEActivationFormat.Standard

    def max_num_tokens_per_rank(self) -> int | None:
        if self.use_veloci_deepep:
            return self.max_tokens_per_rank_
        return None

    def topk_indices_dtype(self) -> torch.dtype | None:
        return torch.int64 if self.use_veloci_deepep else None

    def num_dispatchers(self) -> int:
        return self.num_dispatchers_

    def output_is_reduced(self) -> bool:
        return self.use_veloci_deepep

    def supports_async(self) -> bool:
        return True

    # ------------------------------------------------------------------
    # Expert ID mapping helpers (primary path only)
    # ------------------------------------------------------------------

    def _map_global_to_physical_ids(
        self, topk_ids: torch.Tensor
    ) -> torch.Tensor:
        if self.global_to_physical is None:
            return topk_ids
        return self.global_to_physical[topk_ids]

    def _map_local_to_global_ids(
        self, expert_topk_ids: torch.Tensor
    ) -> torch.Tensor:
        if self.local_expert_global_ids is None:
            return expert_topk_ids
        return self.local_expert_global_ids[expert_topk_ids]

    # ------------------------------------------------------------------
    # Primary path: veloci_deepep.Buffer low-latency dispatch/combine
    # ------------------------------------------------------------------

    def _do_quant(
        self,
        x: torch.Tensor | tuple[torch.Tensor, torch.Tensor],
        a1_dtype: torch.dtype,
        quant_config: FusedMoEQuantConfig,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        if self.use_fp8_dispatch:
            block_k = (
                quant_config.block_shape[1]
                if quant_config.block_shape is not None
                else None
            )
            if block_k == VELOCI_QUANT_BLOCK_SIZE:
                # Kernels did the quantization for us.
                x, x_scales = x
                return x, x_scales

            # Dequant to get back the tokens in the datatype we dispatched.
            x_fp8, x_scales = x
            x = _dequant_fp8(x_fp8, x_scales).to(dtype=a1_dtype)

        assert isinstance(x, (torch.Tensor, tuple))
        q_dtype = quant_config.quant_dtype

        if q_dtype == "nvfp4" and envs.VLLM_DEEPEPLL_NVFP4_DISPATCH:
            assert isinstance(x, tuple)
            x_scales = x[1]
            x = x[0].permute(2, 0, 1)
            num_experts, max_tokens, hidden_dim_by_2 = x.shape
            hidden_dim = hidden_dim_by_2 * 2
        else:
            if q_dtype == "nvfp4":
                q_dtype = None
            assert isinstance(x, torch.Tensor)
            num_experts, max_tokens, hidden_dim = x.size()
            x = x.view((-1, hidden_dim))
            print(f"rank<{torch.distributed.get_rank()}>: VelociDeepEP: quantizing dispatch input with dtype={q_dtype}, x_dtype={x.dtype}, x_shape={x.shape}")
            x, x_scales = moe_kernel_quantize_input(
                x,
                quant_config.a1_scale,
                q_dtype,
                quant_config.per_act_token_quant,
                quant_config.block_shape,
            )
            x = x.view((num_experts, -1, hidden_dim))

        if q_dtype is not None and q_dtype != "nvfp4":
            assert x_scales is not None
            x_scales = normalize_batched_scales_shape(x_scales, num_experts)

        return x, x_scales

    def _veloci_prepare_async(
        self,
        a1: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
        num_experts: int,
        apply_router_weight_on_input: bool,
        quant_config: FusedMoEQuantConfig,
    ) -> tuple[Callable, mk.ReceiverType]:
        hidden_size = a1.size(1)
        assert hidden_size in self.SUPPORTED_HIDDEN_SIZES, (
            f"Hidden Size {hidden_size} not in supported list "
            f"{self.SUPPORTED_HIDDEN_SIZES}"
        )

        a2a_idx = dbo_current_ubatch_id()

        if self.use_fp8_dispatch:
            assert hidden_size % 128 == 0, (
                "VelociDeepEP kernels quantize the inputs in blocks of "
                "shape 128"
            )

        use_nvfp4 = False
        nvfp4_dispatch = (
            quant_config.quant_dtype == "nvfp4"
            and envs.VLLM_DEEPEPLL_NVFP4_DISPATCH
        )
        if nvfp4_dispatch:
            use_nvfp4 = True
        qc_a1_gscale_or_scale = (
            quant_config.a1_gscale
            if nvfp4_dispatch
            else quant_config.a1_scale
        )
        has_per_token_scales = (
            qc_a1_gscale_or_scale.numel() != 1
            if qc_a1_gscale_or_scale is not None
            else (
                quant_config.a2_scale.numel() != 1
                if quant_config.a2_scale is not None
                else False
            )
        )
        if not use_nvfp4:
            assert not has_per_token_scales, (
                "low_latency kernels doesn't support dispatching "
                "per-token scales"
            )

        if apply_router_weight_on_input:
            topk = topk_ids.size(1)
            assert topk == 1, (
                "apply_router_weight_on_input is only implemented for topk=1"
            )
            a1 = a1 * topk_weights.to(a1.dtype)

        # Dispatch
        dispatch_topk_ids = self._map_global_to_physical_ids(topk_ids)
        (
            expert_x,
            expert_num_tokens,
            handle,
            _,
            hook,
        ) = self.buffer.low_latency_dispatch(
            a1,
            dispatch_topk_ids,
            self.max_tokens_per_rank_,
            num_experts,
            use_fp8=self.use_fp8_dispatch,
            round_scale=self.use_ue8m0_dispatch,
            use_ue8m0=self.use_ue8m0_dispatch,
            **(dict(use_nvfp4=True) if use_nvfp4 else dict()),
            **(
                dict(x_global_scale=qc_a1_gscale_or_scale)
                if qc_a1_gscale_or_scale is not None
                else dict()
            ),
            async_finish=False,
            return_recv_hook=True,
        )
        self.handles[a2a_idx] = handle

        return (
            hook,
            lambda: self._veloci_receiver(
                expert_x,
                expert_num_tokens,
                quant_config.a1_scale,
                a1.dtype,
                quant_config,
            ),
        )

    def _veloci_receiver(
        self,
        expert_x: torch.Tensor | tuple[torch.Tensor, torch.Tensor],
        expert_num_tokens: torch.Tensor,
        a1_scale: torch.Tensor | None,
        a1_dtype: torch.dtype,
        quant_config: FusedMoEQuantConfig,
    ) -> mk.PrepareResultType:
        expert_x, expert_x_scale = self._do_quant(
            expert_x, a1_dtype, quant_config
        )

        expert_tokens_meta = mk.ExpertTokensMetadata(
            expert_num_tokens=expert_num_tokens,
            expert_num_tokens_cpu=None,
        )

        return expert_x, expert_x_scale, expert_tokens_meta, None, None

    def _veloci_finalize(
        self,
        output: torch.Tensor,
        fused_expert_output: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
        apply_router_weight_on_input: bool,
        weight_and_reduce_impl: mk.TopKWeightAndReduce,
        do_async: bool,
    ) -> tuple[Callable, Callable]:
        assert isinstance(
            weight_and_reduce_impl, TopKWeightAndReduceDelegate
        ), "Weight application and reduction happens in the combine kernel."

        a2a_idx = dbo_current_ubatch_id()
        do_recv_hook = dbo_enabled() or do_async
        handle = self.handles[a2a_idx]
        assert handle is not None

        combine_topk_weights = topk_weights
        if apply_router_weight_on_input:
            # Weights have already been applied.
            combine_topk_weights = torch.ones_like(topk_weights)

        combine_topk_ids = self._map_global_to_physical_ids(topk_ids)
        dbo_maybe_run_recv_hook()
        _, _, recv_hook = self.buffer.low_latency_combine(
            fused_expert_output,
            combine_topk_ids,
            combine_topk_weights,
            handle,
            async_finish=False,
            zero_copy=False,
            return_recv_hook=do_recv_hook,
            out=output,
        )

        return recv_hook, lambda: None

    # ------------------------------------------------------------------
    # Fallback path: oneCCL/NCCL all_to_all via All2AllManager
    # ------------------------------------------------------------------

    @staticmethod
    def _fallback_quantize_input(
        a1: torch.Tensor,
        quant_config: FusedMoEQuantConfig,
        defer_input_quant: bool,
    ) -> tuple[torch.Tensor, torch.Tensor | None, list[torch.Tensor] | None]:
        if defer_input_quant:
            return a1, None, None

        input_sf = (
            quant_config.a1_gscale
            if quant_config.use_nvfp4_w4a4
            else quant_config.a1_scale
        )

        a1q, a1q_scale = moe_kernel_quantize_input(
            a1,
            input_sf,
            quant_dtype=quant_config.quant_dtype,
            per_act_token_quant=quant_config.per_act_token_quant,
            block_shape=quant_config.block_shape,
            is_fp4_scale_swizzled=False,
        )

        if a1q_scale is None or a1q_scale.ndim == 0:
            return a1q, a1q_scale, None

        return a1q, None, [a1q_scale]

    @staticmethod
    def _fallback_unwrap_scales(
        scales: list[torch.Tensor],
        quant_config: FusedMoEQuantConfig,
    ) -> torch.Tensor:
        a1q_scale = scales[0]
        if (
            quant_config.quant_dtype == "nvfp4"
            and quant_config.is_nvfp4_scale_swizzled
        ):
            if a1q_scale.element_size() == 1:
                a1q_scale = a1q_scale.view(torch.uint8)
            a1q_scale = nvfp4_block_scale_interleave(a1q_scale)
        return a1q_scale

    def _fallback_do_dispatch(
        self,
        a1: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
        quant_config: FusedMoEQuantConfig,
        defer_input_quant: bool,
    ) -> Callable[[], mk.PrepareResultType]:
        a1q, a1q_scale, scales = self._fallback_quantize_input(
            a1, quant_config, defer_input_quant
        )
        res = get_ep_group().dispatch(
            a1q,
            topk_weights,
            topk_ids,
            is_sequence_parallel=self.is_sequence_parallel,
            extra_tensors=scales,
        )

        if scales is None:
            a1q, topk_weights, topk_ids = res
        else:
            a1q, topk_weights, topk_ids, scales = res
            a1q_scale = self._fallback_unwrap_scales(scales, quant_config)

        def receiver() -> mk.PrepareResultType:
            return a1q, a1q_scale, None, topk_ids, topk_weights

        return receiver

    def _fallback_finalize(
        self,
        output: torch.Tensor,
        fused_expert_output: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
        apply_router_weight_on_input: bool,
        weight_and_reduce_impl: mk.TopKWeightAndReduce,
    ) -> Callable[[], None]:
        if isinstance(weight_and_reduce_impl, TopKWeightAndReduceDelegate):
            weight_and_reduce_impl = TopKWeightAndReduceContiguous()

        out = weight_and_reduce_impl.apply(
            output=None,
            fused_expert_output=fused_expert_output,
            topk_weights=topk_weights,
            topk_ids=topk_ids,
            apply_router_weight_on_input=apply_router_weight_on_input,
        )

        combined = get_ep_group().combine(
            out, is_sequence_parallel=self.is_sequence_parallel
        )

        def receiver() -> None:
            output.copy_(combined)

        return receiver

    # ------------------------------------------------------------------
    # Unified interface: routes to veloci_deepep or fallback
    # ------------------------------------------------------------------

    def prepare_async(
        self,
        a1: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
        num_experts: int,
        expert_map: torch.Tensor | None,
        apply_router_weight_on_input: bool,
        quant_config: FusedMoEQuantConfig,
        defer_input_quant: bool = False,
    ) -> tuple[Callable, mk.ReceiverType] | mk.ReceiverType:
        if self.use_veloci_deepep:
            if defer_input_quant:
                raise NotImplementedError(
                    f"{self.__class__.__name__} does not support "
                    "defer_input_quant=True with veloci_deepep. "
                    "Please select an MoE kernel that accepts "
                    "quantized inputs."
                )
            return self._veloci_prepare_async(
                a1, topk_weights, topk_ids, num_experts,
                apply_router_weight_on_input, quant_config,
            )
        else:
            if apply_router_weight_on_input:
                assert topk_ids.size(1) == 1, (
                    "apply_router_weight_on_input is only implemented "
                    "for topk=1"
                )
                a1 = a1 * topk_weights.to(a1.dtype)
            return self._fallback_do_dispatch(
                a1, topk_weights, topk_ids, quant_config, defer_input_quant
            )

    def prepare(
        self,
        a1: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
        num_experts: int,
        expert_map: torch.Tensor | None,
        apply_router_weight_on_input: bool,
        quant_config: FusedMoEQuantConfig,
        defer_input_quant: bool = False,
    ) -> mk.PrepareResultType:
        if defer_input_quant and self.use_veloci_deepep:
            raise NotImplementedError(
                f"{self.__class__.__name__} does not support "
                "defer_input_quant=True with veloci_deepep."
            )
        ret = self.prepare_async(
            a1, topk_weights, topk_ids, num_experts, expert_map,
            apply_router_weight_on_input, quant_config, defer_input_quant,
        )
        if isinstance(ret, tuple):
            hook, receiver = ret
            hook()
        else:
            receiver = ret
        return receiver()

    def finalize_async(
        self,
        output: torch.Tensor,
        fused_expert_output: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
        apply_router_weight_on_input: bool,
        weight_and_reduce_impl: mk.TopKWeightAndReduce,
    ) -> tuple[Callable, Callable] | Callable:
        if self.use_veloci_deepep:
            return self._veloci_finalize(
                output, fused_expert_output, topk_weights, topk_ids,
                apply_router_weight_on_input, weight_and_reduce_impl,
                do_async=True,
            )
        else:
            return self._fallback_finalize(
                output, fused_expert_output, topk_weights, topk_ids,
                apply_router_weight_on_input, weight_and_reduce_impl,
            )

    def finalize(
        self,
        output: torch.Tensor,
        fused_expert_output: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
        apply_router_weight_on_input: bool,
        weight_and_reduce_impl: mk.TopKWeightAndReduce,
    ) -> None:
        if self.use_veloci_deepep:
            recv_hook, cleanup = self._veloci_finalize(
                output, fused_expert_output, topk_weights, topk_ids,
                apply_router_weight_on_input, weight_and_reduce_impl,
                do_async=False,
            )
            if recv_hook is not None:
                recv_hook()
            cleanup()
        else:
            receiver = self._fallback_finalize(
                output, fused_expert_output, topk_weights, topk_ids,
                apply_router_weight_on_input, weight_and_reduce_impl,
            )
            receiver()
