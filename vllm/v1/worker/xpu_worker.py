# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import ctypes
import gc
import os
from typing import Any

import torch

from vllm.config import VllmConfig
from vllm.logger import init_logger
from vllm.platforms import current_platform
from vllm.profiler.wrapper import TorchProfilerWrapper
from vllm.utils.mem_utils import MemorySnapshot, format_gib
from vllm.utils.torch_utils import set_random_seed
from vllm.v1.utils import report_usage_stats
from vllm.v1.worker.gpu_worker import Worker, init_worker_distributed_environment
from vllm.v1.worker.workspace import init_workspace_manager
from vllm.v1.worker.xpu_model_runner import XPUModelRunner, XPUModelRunnerV2

from .utils import request_memory

logger = init_logger(__name__)


def _parse_cpu_list(cpu_str: str) -> set[int]:
    """Parse a CPU list string like '0-15,64-79' into a set of CPU IDs."""
    cpus: set[int] = set()
    for part in cpu_str.split(","):
        part = part.strip()
        if "-" in part:
            lo, hi = part.split("-", 1)
            cpus.update(range(int(lo), int(hi) + 1))
        else:
            cpus.add(int(part))
    return cpus


def _bind_numa(local_rank: int) -> None:
    """Bind this process to CPUs and memory node specified by VLLM_NUMA_CONTROL.

    Format: semicolon-separated entries, one per local_rank (positional).
        Each entry: cpulist:memnode
    Example (4 workers, 2 per NUMA node, CPUs partitioned):
        VLLM_NUMA_CONTROL="0-15,64-79:0;16-31,80-95:0;32-47,96-111:1;48-63,112-127:1"
    """
    numa_control = os.environ.get("VLLM_NUMA_CONTROL")
    if not numa_control:
        return

    entries = numa_control.split(";")
    if local_rank >= len(entries):
        logger.warning(
            "VLLM_NUMA_CONTROL has %d entries but local_rank is %d, "
            "skipping NUMA binding",
            len(entries),
            local_rank,
        )
        return

    entry = entries[local_rank].strip()
    if not entry:
        logger.info("VLLM_NUMA_CONTROL entry for rank %d is empty, skipping",
                     local_rank)
        return

    if ":" not in entry:
        raise ValueError(
            f"Invalid VLLM_NUMA_CONTROL entry for rank {local_rank}: "
            f"'{entry}'. Expected format: 'cpulist:memnode' "
            f"(e.g. '0-15,64-79:0')")

    cpu_str, mem_str = entry.rsplit(":", 1)
    cpu_ids = _parse_cpu_list(cpu_str)
    mem_node = int(mem_str)

    # 1. Pin CPUs
    os.sched_setaffinity(0, cpu_ids)
    logger.info(
        "Rank %d: CPU affinity set to %d CPUs (node %d)",
        local_rank, len(cpu_ids), mem_node,
    )

    # 2. Pin memory node via libnuma
    try:
        libnuma = ctypes.CDLL("libnuma.so.1")
        libnuma.numa_available.restype = ctypes.c_int
        if libnuma.numa_available() < 0:
            logger.warning("libnuma reports NUMA not available, "
                           "skipping memory binding")
            return

        libnuma.numa_allocate_nodemask.restype = ctypes.c_void_p
        libnuma.numa_bitmask_setbit.argtypes = [
            ctypes.c_void_p, ctypes.c_uint]
        libnuma.numa_bitmask_setbit.restype = ctypes.c_void_p
        libnuma.numa_set_membind.argtypes = [ctypes.c_void_p]
        libnuma.numa_set_membind.restype = None
        libnuma.numa_bitmask_free.argtypes = [ctypes.c_void_p]
        libnuma.numa_bitmask_free.restype = None

        nodemask = libnuma.numa_allocate_nodemask()
        libnuma.numa_bitmask_setbit(nodemask, mem_node)
        libnuma.numa_set_membind(nodemask)
        libnuma.numa_bitmask_free(nodemask)
        logger.info(
            "Rank %d: memory bound to NUMA node %d",
            local_rank, mem_node,
        )
    except OSError:
        logger.warning(
            "libnuma.so.1 not found, skipping memory node binding. "
            "CPU affinity is still set.")



class XPUWorker(Worker):
    """A XPU worker class."""

    def __init__(
        self,
        vllm_config: VllmConfig,
        local_rank: int,
        rank: int,
        distributed_init_method: str,
        is_driver_worker: bool = False,
    ):
        super().__init__(
            vllm_config, local_rank, rank, distributed_init_method, is_driver_worker
        )
        device_config = self.device_config
        assert device_config.device_type == "xpu"
        assert current_platform.is_xpu()

        # Torch profiler. Enabled and configured through profiler_config.
        self.profiler: Any | None = None
        profiler_config = vllm_config.profiler_config
        if profiler_config.profiler == "torch":
            worker_name = f"{vllm_config.instance_id}-rank-{self.rank}"
            self.profiler = TorchProfilerWrapper(
                profiler_config,
                worker_name=worker_name,
                local_rank=self.local_rank,
                activities=["CPU", "XPU"],
            )

    def init_device(self):
        # Adjust local_rank for DP (same approach as CUDA gpu_worker).
        # Without per-device ZE_AFFINITY_MASK isolation, the XPU worker
        # sees all devices and must index by DP rank * TP/PP world size.
        parallel_config = self.parallel_config
        if parallel_config.data_parallel_size > 1:
            dp_local_rank = parallel_config.data_parallel_rank_local
            if dp_local_rank is None:
                dp_local_rank = parallel_config.data_parallel_index
            tp_pp_world_size = (
                parallel_config.pipeline_parallel_size
                * parallel_config.tensor_parallel_size
            )
            self.local_rank += dp_local_rank * tp_pp_world_size

        # Bind CPU affinity and memory node before any allocations
        _bind_numa(self.local_rank)

        device = self.device_config.device
        if (
            isinstance(device, torch.device)
            and device.type == "xpu"
            and current_platform.is_xpu()
        ):
            self.device = torch.device(f"xpu:{self.local_rank}")
            torch.accelerator.set_device_index(self.device)
            current_platform.check_if_supports_dtype(self.model_config.dtype)
            torch.accelerator.empty_cache()
            self.init_gpu_memory = torch.xpu.get_device_properties(
                self.local_rank
            ).total_memory
        else:
            raise RuntimeError(f"Not support device type: {self.device_config.device}")

        ENV_CCL_ATL_TRANSPORT = os.getenv("CCL_ATL_TRANSPORT", "ofi")
        ENV_LOCAL_WORLD_SIZE = os.getenv(
            "LOCAL_WORLD_SIZE", str(self.parallel_config.world_size)
        )
        os.environ["CCL_ATL_TRANSPORT"] = ENV_CCL_ATL_TRANSPORT
        os.environ["LOCAL_WORLD_SIZE"] = ENV_LOCAL_WORLD_SIZE
        os.environ["LOCAL_RANK"] = str(self.local_rank)

        init_worker_distributed_environment(
            self.vllm_config,
            self.rank,
            self.distributed_init_method,
            self.local_rank,
            current_platform.dist_backend,
        )

        # global all_reduce needed for overall oneccl warm up
        torch.distributed.all_reduce(torch.zeros(1).xpu())

        # Set random seed.
        set_random_seed(self.model_config.seed)

        # Now take memory snapshot after NCCL is initialized
        gc.collect()
        torch.accelerator.empty_cache()

        # take current memory snapshot
        self.init_snapshot = init_snapshot = MemorySnapshot(device=self.device)
        self.requested_memory = request_memory(init_snapshot, self.cache_config)
        logger.debug("worker init memory snapshot: %r", self.init_snapshot)
        logger.debug(
            "worker requested memory: %sGiB", format_gib(self.requested_memory)
        )

        # Initialize workspace manager
        num_ubatches = 2 if self.vllm_config.parallel_config.enable_dbo else 1
        init_workspace_manager(self.device, num_ubatches)

        # Construct the model runner
        model_runner = XPUModelRunnerV2 if self.use_v2_model_runner else XPUModelRunner
        self.model_runner = model_runner(  # type: ignore
            self.vllm_config, self.device
        )

        if self.rank == 0:
            # If usage stat is enabled, collect relevant info.
            report_usage_stats(self.vllm_config)
