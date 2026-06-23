# Copyright (c) 2025 BAAI. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# Adapted from vllm_ascend/worker/worker.py::_create_profiler.
"""Ascend NPU profiler wrapper.

Extends vLLM's ``WorkerProfiler`` to drive ``torch_npu.profiler.profile``
so that ``--profiler-config`` produces the full ``_ascend_pt`` output
(op_statistic.csv, step_trace_time.csv, ...) instead of CPU-only traces.

All ``torch_npu`` imports are lazy (inside ``__init__``) so this module is
safe to import on non-Ascend hosts — it only fails if you actually
instantiate the wrapper without ``torch_npu`` installed.
"""

from __future__ import annotations

import torch
from typing_extensions import override

from vllm.config import ProfilerConfig
from vllm.logger import init_logger
from vllm.profiler.wrapper import WorkerProfiler

logger = init_logger(__name__)


class AscendTorchProfilerWrapper(WorkerProfiler):
    """``WorkerProfiler`` backed by ``torch_npu.profiler.profile``.

    The NPU profiler produces ``_ascend_pt/ASCEND_PROFILER_OUTPUT/`` with
    op_statistic.csv / step_trace_time.csv when ``analyse()`` is run on the
    captured trace. Configuration mirrors vllm-ascend's ``_create_profiler``
    so output is byte-for-byte comparable across the two plugins.
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

        self.profiler = torch_npu.profiler.profile(
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

    @override
    def _start(self) -> None:
        self.profiler.start()

    @override
    def _stop(self) -> None:
        self.profiler.stop()

    @override
    def annotate_context_manager(self, name: str):
        # torch_npu is source-compatible with torch.profiler.record_function
        # for NVTX-style trace annotations.
        return torch.profiler.record_function(name)
