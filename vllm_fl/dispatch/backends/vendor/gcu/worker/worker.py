# Copyright (c) 2026 BAAI. All rights reserved.
# Adapted from vLLM v0.32.0 gpu_worker.py; Apache-2.0.
"""GCU worker overrides; other platforms retain the shared WorkerFL."""

from __future__ import annotations

import gc
import os
from contextlib import AbstractContextManager, nullcontext

import torch

from vllm.config import (
    CUDAGraphMode,
)
from vllm.config.compilation import CompilationMode
from vllm.device_allocator import get_mem_allocator_instance
from vllm.distributed.parallel_state import (
    get_pp_group,
)
from vllm.logger import init_logger
from vllm.model_executor.model_loader import get_model_loader
from vllm.model_executor.warmup.kernel_warmup import kernel_warmup
from vllm.platforms import current_platform
from vllm.tracing import instrument
from vllm.utils.flashinfer import warn_if_flashinfer_kernels_missing
from vllm.utils.gc_utils import freeze_gc_heap, maybe_attach_gc_debug_callback
from vllm.utils.gpu_sync_debug import enable_gpu_sync_check
from vllm.utils.mem_constants import GiB_bytes
from vllm.utils.mem_utils import (
    MemorySnapshot,
    format_gib,
)
from vllm.utils.torch_utils import set_random_seed
from vllm.v1.worker.gpu.cudagraph_utils import has_compiled_submodule
from vllm.v1.worker.gpu.warmup import warmup_kernels
from vllm.v1.worker.gpu_worker import (
    Worker as GPUWorker,
    init_worker_distributed_environment,
)
from vllm.v1.worker.startup_plan import (
    maybe_save_startup_plan,
)
from vllm.v1.worker.utils import request_memory
from vllm.v1.worker.worker_base import CompilationTimes

from vllm_fl.dispatch.io_common import managed_inference_mode
from vllm_fl.worker.runtime import _initialize_fl_runtime

logger = init_logger(__name__)


class WorkerGCU(GPUWorker):
    def __init__(
        self,
        vllm_config,
        local_rank,
        rank,
        distributed_init_method,
        is_driver_worker=False,
    ):
        os.environ.setdefault("TRITON_CACHE_DIR", f"/tmp/triton-cache-fl-rank-{rank}")
        if (
            vllm_config.num_speculative_tokens == 1
            and vllm_config.scheduler_config.async_scheduling
        ):
            vllm_config.scheduler_config.scheduler_cls = (
                "vllm_fl.worker.scheduler_fl.AsyncSchedulerFL"
            )
        super().__init__(
            vllm_config, local_rank, rank, distributed_init_method, is_driver_worker
        )
        _initialize_fl_runtime(self.rank)

    def _make_model_runner(self):
        if not self.use_v2_model_runner:
            from vllm_fl.worker.model_runner import ModelRunnerFL

            return ModelRunnerFL(self.vllm_config, self.device)
        from .model_runner_v2 import GPUModelRunnerFLV2

        return GPUModelRunnerFLV2(self.vllm_config, self.device)

    def _maybe_get_memory_pool_context(self, tag: str) -> AbstractContextManager:
        # Backends without sleep-mode support use ordinary allocations when
        # sleep is disabled. Keep explicit sleep requests on the allocator
        # path so unsupported backends still report an error.
        if (
            not current_platform.is_sleep_mode_available()
            and not self.vllm_config.model_config.enable_sleep_mode
        ):
            return nullcontext()

        allocator = get_mem_allocator_instance()
        if tag == "weights":
            assert allocator.get_current_usage() == 0, (
                "CuMem allocator can only be used for one instance per process."
            )
        return allocator.use_memory_pool(tag=tag)

    def _scoped_allocator_max_split(self, max_split_size_mb: int):
        # GCU does not use CUDA allocator splitting settings.
        return nullcontext()

    @instrument(span_name="Init device")
    def init_device(self):
        if self.device_config.device_type == current_platform.device_type:
            # This env var set by Ray causes exceptions with graph building.
            os.environ.pop("NCCL_ASYNC_ERROR_HANDLING", None)
            parallel_config = self.parallel_config
            # vLLM slices physical IDs for non-CUDA-like platforms per DP rank.
            # A local rank already indexes into this slice. A full device list
            # (e.g. CUDA --device-ids) still requires the DP offset.
            assigned_ids = parallel_config.assigned_physical_gpu_ids
            local_world_size = parallel_config.local_world_size
            rank_slice_assigned = (
                assigned_ids is not None and len(assigned_ids) == local_world_size
            )
            if (
                not rank_slice_assigned
                and parallel_config.distributed_executor_backend
                not in ("ray", "external_launcher")
                and parallel_config.data_parallel_backend != "ray"
                and parallel_config.nnodes_within_dp == 1
            ):
                # Use local DP rank if available, otherwise use global DP rank.
                dp_local_rank = self.parallel_config.data_parallel_rank_local
                if dp_local_rank is None:
                    dp_local_rank = self.parallel_config.data_parallel_index

                self.local_rank += dp_local_rank * local_world_size

            # Publish the logical-to-physical mapping for topology queries
            # such as NIC affinity and P2P checks.
            if assigned_ids is not None:
                from vllm.platforms.interface import set_assigned_physical_gpu_ids

                set_assigned_physical_gpu_ids(assigned_ids)
                assert self.local_rank < len(assigned_ids), (
                    f"local_rank {self.local_rank} is out of bounds for "
                    f"assigned_physical_gpu_ids {assigned_ids}"
                )
                # NOTE(patch pr45026): local_world_size is derived from
                # parallel_config.nnodes, which is only set for the "mp"
                # multi-node backend. With the "ray"/"external_launcher"
                # backends nnodes stays 1, so local_world_size collapses to
                # the full world_size and this check wrongly fires on
                # cross-node deployments. assigned_physical_gpu_ids is already
                # per-node and the local_rank bound above fully validates the
                # mapping for these backends, so skip the check for them.
                if parallel_config.distributed_executor_backend not in (
                    "ray",
                    "external_launcher",
                ):
                    assert local_world_size <= len(assigned_ids), (
                        f"local_world_size ({local_world_size})"
                        " exceeds assigned_physical_gpu_ids count "
                        f"({len(assigned_ids)})"
                    )
            else:
                assert self.local_rank < torch.accelerator.device_count(), (
                    f"DP adjusted local rank {self.local_rank} is out of "
                    f"bounds for {torch.accelerator.device_count()} devices."
                )

            visible_device_index = (
                current_platform.logical_device_id_to_visible_device_id(self.local_rank)
            )
            self.device = torch.device(
                f"{current_platform.device_type}:{visible_device_index}"
            )
            current_platform.set_device(self.device)

            current_platform.check_if_supports_dtype(self.model_config.dtype)

            # Initialize the distributed environment BEFORE taking
            # memory snapshot
            # This ensures NCCL buffers are allocated before we measure
            # available memory
            init_worker_distributed_environment(
                self.vllm_config,
                self.rank,
                self.distributed_init_method,
                self.local_rank,
                current_platform.dist_backend,
            )

            if self.use_v2_model_runner:
                logger.info_once("Using V2 Model Runner")
            warn_if_flashinfer_kernels_missing()

            # Set random seed.
            set_random_seed(self.model_config.seed)

            # Now take memory snapshot after NCCL is initialized
            gc.collect()
            torch.accelerator.empty_cache()

            # Weights from external model loader process
            external_weight_memory = get_model_loader(
                self.load_config
            ).get_external_weight_memory(self.vllm_config)

            # take current memory snapshot after external weights are ready
            self.init_snapshot = init_snapshot = MemorySnapshot(device=self.device)
            self.requested_memory = request_memory(
                init_snapshot, self.cache_config, external_weight_memory
            )
            logger.debug("worker init memory snapshot: %r", self.init_snapshot)
            logger.debug(
                "worker requested memory: %sGiB", format_gib(self.requested_memory)
            )
        else:
            raise RuntimeError(f"Unsupported device type: {self.device_config.device}")

        self._init_workspace_and_model_runner()

    @instrument(span_name="Warmup (GPU)")
    def compile_or_warm_up_model(self) -> CompilationTimes:
        warmup_sizes: list[int] = []

        if (
            self.vllm_config.compilation_config.mode == CompilationMode.VLLM_COMPILE
            and has_compiled_submodule(self.model_runner.get_model())
        ):
            # warm up sizes that are not in cudagraph capture sizes,
            # but users still want to compile for better performance,
            # e.g. for the max-num-batched token size in chunked prefill.
            compile_sizes = self.vllm_config.compilation_config.compile_sizes
            warmup_sizes = compile_sizes.copy() if compile_sizes is not None else []  # type: ignore[assignment]
            cg_capture_sizes: list[int] = []

            if self.vllm_config.compilation_config.cudagraph_mode != CUDAGraphMode.NONE:
                cg_sizes = self.vllm_config.compilation_config.cudagraph_capture_sizes
                cg_capture_sizes = [] if cg_sizes is None else cg_sizes
                warmup_sizes = [x for x in warmup_sizes if x not in cg_capture_sizes]

            compile_ranges = self.vllm_config.compilation_config.get_compile_ranges()
            # For each compile_range, if none of the batch sizes
            # in warmup_sizes or cudagraph_capture_sizes are in the range,
            # add the end of the range to ensure compilation/warmup.
            all_sizes = set(cg_capture_sizes)
            all_sizes.update([x for x in warmup_sizes if isinstance(x, int)])
            for compile_range in compile_ranges:
                if not any(x in compile_range for x in all_sizes):
                    warmup_sizes.append(compile_range.end)

        # We skip EPLB here since we don't want to record dummy metrics
        for size in sorted(warmup_sizes, reverse=True):
            logger.info("Compile and warming up model for size %d", size)
            self.model_runner._dummy_run(
                size,
                skip_eplb=True,
                remove_lora=False,
                randomize_inputs=self.randomize_dummy_inputs,
            )
        self.model_runner.maybe_remove_all_loras(self.model_runner.lora_config)

        # Warmup and tune the kernels used during model execution before
        # cuda graph capture.
        try:
            kernel_warmup(self)
        except ImportError as error:
            # The empty runtime omits optional native warmup dependencies.
            logger.warning("kernel_warmup skipped: %s", error)

        if self.use_v2_model_runner:  # noqa: SIM102
            if self.vllm_config.kernel_config.enable_jit_warmup:
                # A workspace resize after capture frees what the graphs point at.
                warmup_kernels(
                    self.model_runner, self.execute_model, self.sample_tokens
                )

        cuda_graph_memory_bytes = 0
        if not self.model_config.enforce_eager:
            with self._get_cudagraph_capture_context():
                cuda_graph_memory_bytes = self.model_runner.capture_model()

        # Compare actual vs estimated CUDA graph memory (if we did profiling)
        if (
            hasattr(self, "cudagraph_memory_estimate")
            and self.cudagraph_memory_estimate > 0
        ):
            GiB = lambda b: round(b / GiB_bytes, 2)
            diff = abs(cuda_graph_memory_bytes - self.cudagraph_memory_estimate)
            logger.info(
                "CUDA graph pool memory: %s GiB (actual), %s GiB (estimated), "
                "difference: %s GiB (%.1f%%).",
                GiB(cuda_graph_memory_bytes),
                GiB(self.cudagraph_memory_estimate),
                GiB(diff),
                100 * diff / max(cuda_graph_memory_bytes, 1),
            )

        if self.cache_config.kv_cache_memory_bytes is None and hasattr(
            self, "peak_activation_memory"
        ):
            # Suggests optimal kv cache memory size if we rely on
            # memory_profiling to guess the kv cache memory size which
            # provides peak_activation_memory and a few other memory
            # consumption. `memory_profiling` does not consider
            # CUDAGraph memory size and may not utilize all gpu memory.
            # Users may want fine-grained control to specify kv cache
            # memory size.

            # empirically observed that the memory profiling may
            # slightly underestimate the memory consumption.
            # So leave a small buffer (=150MiB) to avoid OOM.
            redundancy_buffer_memory = 150 * (1 << 20)

            non_kv_cache_memory = (
                self.total_consumed
                + self.peak_activation_memory
                + cuda_graph_memory_bytes
            )
            kv_cache_memory_bytes_to_gpu_limit = (
                self.init_snapshot.free_memory
                - non_kv_cache_memory
                - redundancy_buffer_memory
            )
            kv_cache_memory_bytes_to_requested_limit = (
                int(self.requested_memory)
                - non_kv_cache_memory
                - redundancy_buffer_memory
            )

            msg = (
                f"Free memory on device "
                f"({format_gib(self.init_snapshot.free_memory)}/"
                f"{format_gib(self.init_snapshot.total_memory)} GiB) on startup. "
                f"Desired GPU memory utilization is "
                f"({self.cache_config.gpu_memory_utilization}, "
                f"{format_gib(self.requested_memory)} GiB). "
                f"Actual usage is {format_gib(self.total_consumed)} "
                f"GiB for consumed memory (weights + non-torch), "
                f"{format_gib(self.peak_activation_memory)} GiB "
                f"for peak activation, and {format_gib(cuda_graph_memory_bytes)} "
                f"GiB for CUDAGraph memory. Replace gpu_memory_utilization "
                f"config with `--kv-cache-memory="
                f"{kv_cache_memory_bytes_to_requested_limit}` "
                f"({format_gib(kv_cache_memory_bytes_to_requested_limit)} GiB) to fit "
                f"into requested memory, or `--kv-cache-memory="
                f"{kv_cache_memory_bytes_to_gpu_limit}` "
                f"({format_gib(kv_cache_memory_bytes_to_gpu_limit)} GiB) to fully "
                f"utilize gpu memory. Current kv cache memory in use is "
                f"{format_gib(self.available_kv_cache_memory_bytes)} GiB."
            )

            logger.info(msg)

            maybe_save_startup_plan(self, kv_cache_memory_bytes_to_requested_limit)

        if not self.use_v2_model_runner and get_pp_group().is_last_rank:
            # V1: Warm up sampler and preallocate memory buffer for logits and other
            # sampling related tensors of max possible shape to avoid memory
            # fragmentation issue.
            # NOTE: This is called after `capture_model` on purpose to prevent
            # memory buffers from being cleared by `torch.accelerator.empty_cache`.
            max_num_reqs = min(
                self.scheduler_config.max_num_seqs,
                self.scheduler_config.max_num_batched_tokens,
            )

            # We skip EPLB here since we don't want to record dummy metrics
            hidden_states, last_hidden_states = self.model_runner._dummy_run(
                num_tokens=max_num_reqs,
                skip_eplb=True,
                cudagraph_runtime_mode=CUDAGraphMode.NONE,
            )
            if self.model_runner.is_pooling_model:
                self.model_runner._dummy_pooler_run(hidden_states)
            else:
                self.model_runner._dummy_sampler_run(hidden_states=last_hidden_states)

        # Reset the seed to ensure that the random state is not affected by
        # the model initialization and profiling.
        set_random_seed(
            self.model_config.seed, self.parallel_config.data_parallel_index
        )

        # Eagerly trigger inductor's once-per-process lazy inits during
        # warmup (rather than on a later compile cache-miss at runtime).
        c_config = self.compilation_config
        if c_config.mode != CompilationMode.NONE and c_config.backend == "inductor":
            from vllm.compilation.compiler_interface import (
                trigger_inductor_lazy_init,
            )

            trigger_inductor_lazy_init(self.device)

        # All warmup is done — start monitoring for unexpected JIT
        # compilations that would cause latency spikes during inference.
        self._maybe_activate_jit_monitor()

        # Freeze the worker heap so the GC won't scan static objects
        # (model weights, KV caches, CUDA graphs) during inference.
        freeze_gc_heap()
        maybe_attach_gc_debug_callback()

        # Warmup / first-compile is done — activate the `VLLM_GPU_SYNC_CHECK`
        # gate so subsequent `execute_model` / `sample_tokens` calls enforce it.
        enable_gpu_sync_check()

        return CompilationTimes(
            language_model=self.compilation_config.compilation_time,
            encoder=self.compilation_config.encoder_compilation_time,
        )

    @managed_inference_mode()
    def determine_available_memory(self):
        return super().determine_available_memory()

    @managed_inference_mode()
    def sample_tokens(self, grammar_output):
        return super().sample_tokens(grammar_output)

    @managed_inference_mode()
    def execute_model(self, scheduler_output):
        return super().execute_model(scheduler_output)
