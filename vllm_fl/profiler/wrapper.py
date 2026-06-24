# Copyright (c) 2025 BAAI. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# Adapted from vllm_ascend/worker/worker.py::_create_profiler.
"""Unified FL torch profiler wrapper.

``FLTorchProfilerWrapper`` abstracts the hardware probe and dispatches to the
appropriate profiler backend:

* On Ascend NPU hosts (``torch_npu`` importable + ``torch.npu.is_available()``),
  it drives ``torch_npu.profiler.profile()`` so ``--profiler-config`` produces
  the full ``_ascend_pt`` output (``op_statistic.csv``,
  ``step_trace_time.csv``, ...). Configuration mirrors vllm-ascend's
  ``_create_profiler`` so traces are byte-for-byte comparable across plugins.
* On CUDA / other hosts, it falls back to vLLM's ``TorchProfilerWrapper`` with
  ``activities=["CPU", "CUDA"]``, preserving the default vLLM trace format.

All ``torch_npu`` imports are lazy (inside ``_build_npu_backend``) so this
module is safe to import on non-Ascend hosts — it only fails if you actually
construct the wrapper on an NPU host without ``torch_npu`` installed.
"""

from __future__ import annotations

import torch
from typing_extensions import override

from vllm.config import ProfilerConfig
from vllm.logger import init_logger
from vllm.profiler.wrapper import WorkerProfiler

logger = init_logger(__name__)


def _is_npu_available() -> bool:
    """Probe Ascend NPU without failing on non-Ascend hosts.

    ``current_platform.device_type`` can return an empty string when vendor
    detection fails, so we go through ``torch_npu`` + ``torch.npu.is_available()``
    which directly checks the hardware.
    """
    try:
        import torch_npu  # noqa: F401
        return torch.npu.is_available()
    except ImportError:
        return False


class FLTorchProfilerWrapper(WorkerProfiler):
    """Single-interface torch profiler for FL, branching NPU vs CUDA.

    The wrapper picks its backend at construction time:

    * NPU  → ``torch_npu.profiler.profile(...)`` with the same
      ``_ExperimentalConfig`` / ``ProfilerActivity.NPU`` /
      ``tensorboard_trace_handler`` setup that vllm-ascend uses, so the
      resulting ``_ascend_pt/`` output is directly comparable.
    * CUDA → vLLM's ``TorchProfilerWrapper(activities=["CPU", "CUDA"])``,
      preserving the default vLLM trace format.

    Both backends expose ``start()`` / ``stop()`` and are compatible with
    ``torch.profiler.record_function`` for trace annotations, so the
    delegation in ``_start`` / ``_stop`` / ``annotate_context_manager`` is
    uniform.
    """

    def __init__(
        self,
        profiler_config: ProfilerConfig,
        worker_name: str,
        local_rank: int,
    ) -> None:
        super().__init__(profiler_config)
        self.local_rank = local_rank
        self.profiler_config = profiler_config

        if profiler_config.profiler != "torch":
            raise RuntimeError(
                f"Unrecognized profiler: {profiler_config.profiler}"
            )
        if not profiler_config.torch_profiler_dir:
            raise RuntimeError("torch_profiler_dir cannot be empty.")

        if _is_npu_available():
            self._backend = self._build_npu_backend(
                profiler_config, worker_name, local_rank
            )
        else:
            self._backend = self._build_cuda_backend(
                profiler_config, worker_name, local_rank
            )

    @staticmethod
    def _build_npu_backend(
        profiler_config: ProfilerConfig,
        worker_name: str,
        local_rank: int,
    ):
        # Lazy import: torch_npu is only available on Ascend hosts.
        import torch_npu.profiler  # type: ignore

        if local_rank in (None, 0):
            logger.info_once(
                "Ascend NPU profiling enabled. Traces will be saved to: %s",
                profiler_config.torch_profiler_dir,
                scope="local",
            )

        experimental_config = torch_npu.profiler._ExperimentalConfig(
            export_type=torch_npu.profiler.ExportType.Text,
            profiler_level=torch_npu.profiler.ProfilerLevel.Level1,
            msprof_tx=False,
            aic_metrics=torch_npu.profiler.AiCMetrics.PipeUtilization,
            l2_cache=False,
            op_attr=False,
            data_simplification=True,
            record_op_args=False,
            gc_detect_threshold=None,
        )

        return torch_npu.profiler.profile(
            activities=[
                torch_npu.profiler.ProfilerActivity.CPU,
                torch_npu.profiler.ProfilerActivity.NPU,
            ],
            with_stack=False,
            profile_memory=profiler_config.torch_profiler_with_memory,
            # torch_npu.profiler.with_modules is equivalent to
            # torch.profiler.with_stack but with significantly less overhead.
            with_modules=profiler_config.torch_profiler_with_stack,
            experimental_config=experimental_config,
            on_trace_ready=torch_npu.profiler.tensorboard_trace_handler(
                profiler_config.torch_profiler_dir,
                worker_name=worker_name,
            ),
        )

    @staticmethod
    def _build_cuda_backend(
        profiler_config: ProfilerConfig,
        worker_name: str,
        local_rank: int,
    ):
        # Lazy import to keep vllm.profiler import-time cost off the hot path
        # and avoid coupling this module to TorchProfilerWrapper at load time.
        from vllm.profiler.wrapper import TorchProfilerWrapper

        return TorchProfilerWrapper(
            profiler_config,
            worker_name=worker_name,
            local_rank=local_rank,
            activities=["CPU", "CUDA"],
        )

    @override
    def _start(self) -> None:
        self._backend.start()

    @override
    def _stop(self) -> None:
        self._backend.stop()

    @override
    def annotate_context_manager(self, name: str):
        # Both torch.profiler and torch_npu.profiler are source-compatible
        # with torch.profiler.record_function for NVTX-style trace annotations.
        return torch.profiler.record_function(name)
