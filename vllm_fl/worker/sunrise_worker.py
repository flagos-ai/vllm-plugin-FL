# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 BAAI. All rights reserved.
# init_device is adapted from vLLM v0.32.0/vllm/v1/worker/gpu_worker.py.
"""Sunrise worker retaining the official vLLM 0.32 worker protocol."""

from __future__ import annotations

import gc
import inspect
import os
from contextlib import nullcontext

import torch

from vllm.platforms import current_platform
from vllm.utils.gpu_sync_debug import with_gpu_sync_check
from vllm.v1.worker import gpu_worker as upstream

from vllm_fl.dispatch.backends.vendor.sunrise.patch import apply_sunrise_patches
from vllm_fl.dispatch.io_common import managed_inference_mode
from vllm_fl.worker.runtime import _initialize_fl_runtime

# Preserve the upstream worker's execution/communication implementation while
# replacing its fixed inference-mode decorator with FL's IO-aware context.
# Reapply the sync-check wrapper on our public execution entry points below.
_determine_available_memory = inspect.unwrap(upstream.Worker.determine_available_memory)
_execute_model = inspect.unwrap(upstream.Worker.execute_model)
_sample_tokens = inspect.unwrap(upstream.Worker.sample_tokens)


class SunriseWorker(upstream.Worker):
    """Use native v0.32 scheduling, KV allocation and weight-transfer contracts."""

    def __init__(self, *args, **kwargs):
        if current_platform.device_type != "ptpu":
            raise RuntimeError("SunriseWorker requires the Sunrise/PTPU platform")
        super().__init__(*args, **kwargs)
        # These allocator/stream patches must precede the first memory snapshot,
        # including in independently spawned worker processes.
        apply_sunrise_patches()
        _initialize_fl_runtime(self.rank)

    def _maybe_get_memory_pool_context(self, tag: str):
        if (
            self.model_config.enable_sleep_mode
            or self.model_config.enable_cumem_allocator
        ):
            raise NotImplementedError(
                "Sunrise does not support the CUDA CuMem allocator"
            )
        return nullcontext()

    def init_device(self):
        """The v0.32 initialization sequence with native PTPU device operations."""
        apply_sunrise_patches()
        if self.device_config.device_type != "ptpu":
            raise RuntimeError(
                f"Unsupported device type: {self.device_config.device_type}"
            )
        os.environ.pop("NCCL_ASYNC_ERROR_HANDLING", None)
        parallel_config = self.parallel_config
        if (
            parallel_config.distributed_executor_backend
            not in ("ray", "external_launcher")
            and parallel_config.data_parallel_backend != "ray"
            and parallel_config.nnodes_within_dp == 1
        ):
            dp_local_rank = parallel_config.data_parallel_rank_local
            if dp_local_rank is None:
                dp_local_rank = parallel_config.data_parallel_index
            tp_pcp_pp_world_size = (
                parallel_config.pipeline_parallel_size
                * parallel_config.prefill_context_parallel_size
                * parallel_config.tensor_parallel_size
            )
            self.local_rank += dp_local_rank * tp_pcp_pp_world_size

        assigned_physical_gpu_ids = parallel_config.assigned_physical_gpu_ids
        if assigned_physical_gpu_ids is not None:
            from vllm.platforms.interface import set_assigned_physical_gpu_ids

            set_assigned_physical_gpu_ids(assigned_physical_gpu_ids)
            assert self.local_rank < len(assigned_physical_gpu_ids), (
                f"local_rank {self.local_rank} is out of bounds for "
                f"assigned_physical_gpu_ids {assigned_physical_gpu_ids}"
            )
            if parallel_config.distributed_executor_backend not in (
                "ray",
                "external_launcher",
            ):
                assert parallel_config.local_world_size <= len(
                    assigned_physical_gpu_ids
                ), (
                    f"local_world_size ({parallel_config.local_world_size}) exceeds "
                    f"assigned_physical_gpu_ids count ({len(assigned_physical_gpu_ids)})"
                )
        else:
            device_count = current_platform.torch_device_fn.device_count()
            assert self.local_rank < device_count, (
                f"DP adjusted local rank {self.local_rank} is out of bounds "
                f"for {device_count} devices."
            )

        visible_device_index = current_platform.logical_device_id_to_visible_device_id(
            self.local_rank
        )
        self.device = torch.device(f"ptpu:{visible_device_index}")
        current_platform.set_device(self.device)
        current_platform.check_if_supports_dtype(self.model_config.dtype)

        # Torch process groups use PCCL; CommunicatorFL independently selects
        # the matched FlagCX implementation for tensor-parallel collectives.
        upstream.init_worker_distributed_environment(
            self.vllm_config,
            self.rank,
            self.distributed_init_method,
            self.local_rank,
            "pccl",
        )
        upstream.set_random_seed(self.model_config.seed)
        gc.collect()
        current_platform.torch_device_fn.empty_cache()
        external_weight_memory = upstream.get_model_loader(
            self.load_config
        ).get_external_weight_memory(self.vllm_config)
        self.init_snapshot = upstream.MemorySnapshot(device=self.device)
        self.requested_memory = upstream.request_memory(
            self.init_snapshot, self.cache_config, external_weight_memory
        )
        upstream.logger.debug(
            "Sunrise worker init memory snapshot: %r", self.init_snapshot
        )
        self._init_workspace_and_model_runner()

    def _make_model_runner(self):
        if self.use_v2_model_runner:
            from vllm_fl.worker.sunrise_v2_model_runner import SunriseV2ModelRunner

            upstream.logger.info("Using Sunrise Model Runner V2")
            return SunriseV2ModelRunner(self.vllm_config, self.device)
        from vllm_fl.worker.sunrise_model_runner import SunriseModelRunner

        return SunriseModelRunner(self.vllm_config, self.device)

    def synchronize_device(self) -> None:
        current_platform.torch_device_fn.synchronize()

    @managed_inference_mode()
    def determine_available_memory(self) -> int:
        return _determine_available_memory(self)

    @managed_inference_mode()
    @with_gpu_sync_check
    def execute_model(self, scheduler_output):
        return _execute_model(self, scheduler_output)

    @managed_inference_mode()
    @with_gpu_sync_check
    def sample_tokens(self, grammar_output):
        return _sample_tokens(self, grammar_output)
