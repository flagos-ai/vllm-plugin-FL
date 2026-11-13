# Copyright (c) 2026 BAAI. All rights reserved.
"""Iluvatar worker extensions, isolated from other platforms."""

import time
from types import NoneType
from typing import Any

import torch

from vllm.distributed.kv_transfer import ensure_kv_transfer_initialized
from vllm.distributed.parallel_state import get_pp_group, get_tp_group
from vllm.platforms import current_platform
from vllm.sequence import IntermediateTensors
from vllm.tracing import instrument
from vllm.utils.gpu_sync_debug import with_gpu_sync_check
from vllm.utils.mem_utils import format_gib
from vllm.v1.core.sched.output import SchedulerOutput
from vllm.v1.kv_cache_interface import KVCacheConfig
from vllm.v1.outputs import AsyncModelRunnerOutput, ModelRunnerOutput

from vllm_fl.dispatch.backends.vendor.iluvatar.patches.worker_compat import (
    apply_import_compat,
)
from vllm_fl.dispatch.io_common import managed_inference_mode

apply_import_compat()
from vllm_fl.worker.worker import (
    AsyncIntermediateTensors,
    WorkerFL as BaseWorkerFL,
    logger,
)


class IluvatarWorkerFL(BaseWorkerFL):
    def init_device(self):
        from vllm_fl.dispatch.backends.vendor.iluvatar.iluvatar import (
            apply_worker_patches,
        )

        apply_worker_patches()
        super().init_device()
        from vllm_fl.dispatch.backends.vendor.iluvatar.pipeline import (
            install_pipeline_transport,
        )

        install_pipeline_transport(get_pp_group())

    def sleep(self, level: int = 1) -> None:
        torch.accelerator.synchronize()
        free_bytes_before_sleep = torch.accelerator.get_memory_info()[0]

        # FL: vendor scale aliases need not be registered buffers. Preserve
        # calibrated values across both sleep levels alongside upstream buffers.
        from vllm.v1.kv_cache_interface import is_quantized_kv_cache

        if is_quantized_kv_cache(self.cache_config.cache_dtype):
            self._sleep_saved_kv_scales = []
            seen = set()
            for layer in self.compilation_config.static_forward_context.values():
                for attr in ("_k_scale", "k_scale", "_v_scale", "v_scale"):
                    scale = getattr(layer, attr, None)
                    if isinstance(scale, torch.Tensor) and id(scale) not in seen:
                        seen.add(id(scale))
                        self._sleep_saved_kv_scales.append((scale, scale.cpu().clone()))

        # Save the buffers before level 2 sleep
        if level == 2:
            model = self.model_runner.model
            self._sleep_saved_buffers = {
                name: buffer.cpu().clone() for name, buffer in model.named_buffers()
            }
            draft = self.get_draft_model()
            if draft is not None:
                self._sleep_saved_draft_buffers = {
                    name: buffer.cpu().clone() for name, buffer in draft.named_buffers()
                }

        self._get_sleep_mode_backend().suspend(level)

        torch.accelerator.synchronize()
        deadline = time.monotonic() + (5.0 if current_platform.is_rocm() else 0)
        while True:
            free_bytes_after_sleep, total = torch.accelerator.get_memory_info()
            freed_bytes = free_bytes_after_sleep - free_bytes_before_sleep
            if freed_bytes >= 0 or time.monotonic() >= deadline:
                break
            time.sleep(0.1)

        used_bytes = total - free_bytes_after_sleep
        assert freed_bytes >= 0, "Memory usage increased after sleeping."
        logger.info(
            "Sleep mode freed %s GiB memory, %s GiB memory is still in use.",
            format_gib(freed_bytes),
            format_gib(used_bytes),
        )

    def wake_up(self, tags: list[str] | None = None) -> None:
        self._get_sleep_mode_backend().resume(tags)

        # Restore the buffers after level 2 sleep
        wake_weights = tags is None or "weights" in tags
        if wake_weights and len(self._sleep_saved_buffers):
            model = self.model_runner.model
            for name, buffer in model.named_buffers():
                if name in self._sleep_saved_buffers:
                    buffer.data.copy_(self._sleep_saved_buffers[name].data)
            self._sleep_saved_buffers = {}

        if wake_weights and len(self._sleep_saved_draft_buffers):
            draft = self.get_draft_model()
            if draft is not None:
                for name, buffer in draft.named_buffers():
                    if name in self._sleep_saved_draft_buffers:
                        buffer.data.copy_(self._sleep_saved_draft_buffers[name].data)
            self._sleep_saved_draft_buffers = {}

        if wake_weights:
            for scale, saved in getattr(self, "_sleep_saved_kv_scales", ()):
                scale.data.copy_(saved)
            self._sleep_saved_kv_scales = []

        if tags is None or "kv_cache" in tags:
            from vllm.v1.kv_cache_interface import is_quantized_kv_cache

            if is_quantized_kv_cache(self.cache_config.cache_dtype):
                # FL: newly allocated FP8 KV memory must be initialized. Model
                # scales are buffers restored above, not reset to 1.0 here.
                for entry in self.model_runner.kv_caches:
                    if entry is None:
                        continue
                    tensors = entry if isinstance(entry, (list, tuple)) else (entry,)
                    for tensor in tensors:
                        tensor.zero_()

        torch.accelerator.synchronize()

    def update_config(self, overrides: dict[str, Any]) -> None:
        """Apply overrides to the worker, runner and shared VllmConfig."""
        from vllm.config import update_config

        allowed_config_names = {"load_config", "model_config"}
        for config_name, config_overrides in overrides.items():
            if config_name not in allowed_config_names:
                allowed = ", ".join(sorted(allowed_config_names))
                raise ValueError(
                    f"Config override '{config_name}' is not supported. "
                    f"Supported configs: {allowed}"
                )
            config = getattr(self.vllm_config, config_name)
            new_config = update_config(config, config_overrides)
            setattr(self.vllm_config, config_name, new_config)
            setattr(self.model_runner, config_name, new_config)
            setattr(self, config_name, new_config)

    @instrument(span_name="Allocate KV cache")
    def initialize_from_config(self, kv_cache_config: KVCacheConfig) -> None:
        """Allocate GPU KV cache with the specified kv_cache_config."""

        # Update local config with adjusted num blocks after profiling,
        # so that it's available to the warmup stage.
        self.cache_config.num_gpu_blocks = kv_cache_config.num_blocks

        # Init kv cache connector here, because it requires
        # `kv_cache_config`.
        # NOTE(Kuntai): This need to be done before `initialize_kv_cache`,
        # because `initialize_kv_cache` will inject kv cache groups not
        # related to kv cache connector (e.g. kv cache sharing layers).
        ensure_kv_transfer_initialized(self.vllm_config, kv_cache_config)

        with self._maybe_get_memory_pool_context(tag="kv_cache"):
            self.model_runner.initialize_kv_cache(kv_cache_config)

        # Build KV-zero metadata outside the CuMem pool so the bookkeeping
        # GPU tensors (seg_addrs, block-id buffers) use the standard PyTorch
        # allocator and are not discarded during sleep/wake cycles.
        if kv_cache_config.needs_kv_cache_zeroing and hasattr(
            self.model_runner, "_init_kv_zero_meta"
        ):
            self.model_runner._init_kv_zero_meta()

    @managed_inference_mode()
    @with_gpu_sync_check
    def execute_model(
        self, scheduler_output: "SchedulerOutput"
    ) -> ModelRunnerOutput | AsyncModelRunnerOutput | None:
        # ensure any previous non-blocking PP sends are complete
        if self._pp_send_work:
            for handle in self._pp_send_work:
                handle.wait()
            self._pp_send_work = []

        intermediate_tensors = None
        forward_pass = scheduler_output.total_num_scheduled_tokens > 0
        if forward_pass and not get_pp_group().is_first_rank:
            tensor_dict, comm_handles, comm_postprocess = (
                get_pp_group().irecv_tensor_dict(
                    all_gather_group=get_tp_group(),
                )
            )
            assert tensor_dict is not None
            intermediate_tensors = AsyncIntermediateTensors(
                tensor_dict,
                comm_handles=comm_handles,
                comm_postprocess=comm_postprocess,
            )

        with self.annotate_profile(scheduler_output):
            output = self.model_runner.execute_model(
                scheduler_output, intermediate_tensors
            )
            if (
                self.use_v2_model_runner
                and self.model_runner.is_pooling_model
                and output is None
            ):
                output = self.model_runner.pool()  # type: ignore
            if isinstance(
                output, ModelRunnerOutput | AsyncModelRunnerOutput | NoneType
            ):
                return output

        assert isinstance(output, IntermediateTensors)
        parallel_config = self.vllm_config.parallel_config
        assert (
            parallel_config.distributed_executor_backend != "external_launcher"
            and not get_pp_group().is_last_rank
        )

        # launch non-blocking send of intermediate tensors
        handles = get_pp_group().isend_tensor_dict(
            output.tensors,
            all_gather_group=get_tp_group(),
        )
        # Metadata is reaped by GroupCoordinator; only wait on device sends.
        self._pp_send_work = handles[1:]

        if self.use_v2_model_runner and self.model_runner.is_pooling_model:
            self.model_runner.pool()

        return None
