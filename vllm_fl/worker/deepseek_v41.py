# SPDX-License-Identifier: Apache-2.0
"""MetaX V2 Graph memory accounting for the bounded V4.1 deployment."""

import torch

from vllm.logger import init_logger
from vllm.v1.worker.gpu.model_runner import GPUModelRunner

logger = init_logger(__name__)


def prepare_v41_graph_kernels(worker, warmup_fn) -> bool:
    """Warm all native sampler/draft paths before creating MetaX V4.1 graphs."""
    from vllm.distributed import get_ep_group
    from vllm.platforms import current_platform

    config = worker.vllm_config
    speculation = config.speculative_config
    if not (
        getattr(current_platform, "vendor_name", None) == "metax"
        and worker.use_v2_model_runner
        and config.model_config.hf_config.model_type == "deepseek_v41"
        and speculation is not None
        and speculation.method == "dspark"
        and config.parallel_config.enable_expert_parallel
        and config.parallel_config.data_parallel_size > 1
        and config.compilation_config.cudagraph_mode
    ):
        return False
    runner = worker.model_runner
    if runner.speculator is None:
        raise RuntimeError(
            "V4.1 DSpark kernel preparation requires the native speculator"
        )
    managers = (runner.cudagraph_manager, runner.speculator.query_cudagraph_manager)
    if any(manager is None or manager._graphs_captured for manager in managers):
        raise RuntimeError("V4.1 sampler preparation must precede graph capture")
    # Native dispatch selects NONE while _graphs_captured is False. Preserve
    # manager identity and configured FULL mode; capture follows this warmup.
    warmup_fn(runner, worker.execute_model, worker.sample_tokens)
    get_ep_group().barrier()
    logger.info("V4.1 sampler and draft kernels prepared before graph capture")
    return True


class DeepseekV41ModelRunner(GPUModelRunner):
    def capture_model(self, *, profile_only: bool = False) -> int:
        captured = super().capture_model(profile_only=profile_only)
        if captured <= 0:
            return captured
        # The native DSpark cost sweep leaves reclaimable eager buffers in the
        # default allocator. Charge only the remaining allocation to Graphs.
        torch.accelerator.synchronize()
        free_before = torch.accelerator.get_memory_info()[0]
        live_before = torch.cuda.memory_allocated()
        torch.accelerator.empty_cache()
        reclaimed = torch.accelerator.get_memory_info()[0] - free_before
        if (
            torch.cuda.memory_allocated() != live_before
            or not 0 <= reclaimed <= captured
        ):
            raise RuntimeError("Inconsistent MetaX Graph memory accounting")
        return captured - reclaimed
