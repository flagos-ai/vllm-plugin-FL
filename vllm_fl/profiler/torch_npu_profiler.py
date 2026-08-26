# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
# Copyright 2023 The vLLM team.
# SPDX-License-Identifier: Apache-2.0
"""Ascend torch profiler integration owned by the FL runtime.

Adapted from vLLM-Ascend 0.20.2rc1.  This module deliberately depends only
on torch_npu and vLLM, so standalone FL installations do not import
``vllm_ascend``.
"""

from __future__ import annotations

from typing import Any

import torch_npu
from vllm.config import ProfilerConfig
from vllm.profiler.wrapper import WorkerProfiler


class TorchNPUProfilerWrapper(WorkerProfiler):
    """Wire vLLM's worker profiler lifecycle to ``torch_npu.profiler``."""

    def __init__(self, profiler_config: ProfilerConfig, trace_name: str) -> None:
        super().__init__(profiler_config)
        self.profiler: Any = self._create_profiler(
            profiler_config,
            trace_name,
        )

    @staticmethod
    def _create_profiler(
        profiler_config: ProfilerConfig,
        trace_name: str,
    ) -> Any:
        if profiler_config.profiler != "torch":
            raise RuntimeError(
                f"Unrecognized profiler: {profiler_config.profiler}"
            )
        if not profiler_config.torch_profiler_dir:
            raise RuntimeError("torch_profiler_dir cannot be empty.")

        experimental_config = torch_npu.profiler._ExperimentalConfig(
            export_type=torch_npu.profiler.ExportType.Text,
            profiler_level=torch_npu.profiler.ProfilerLevel.Level1,
            msprof_tx=False,
            aic_metrics=torch_npu.profiler.AiCMetrics.AiCoreNone,
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
            # torch_npu's with_modules is the useful low-overhead equivalent
            # for vLLM's with_stack configuration.
            with_modules=profiler_config.torch_profiler_with_stack,
            experimental_config=experimental_config,
            on_trace_ready=torch_npu.profiler.tensorboard_trace_handler(
                profiler_config.torch_profiler_dir,
                worker_name=trace_name,
            ),
        )

    def _start(self) -> None:
        self.profiler.start()

    def _stop(self) -> None:
        self.profiler.stop()

    def _profiler_step(self) -> bool:
        return True
