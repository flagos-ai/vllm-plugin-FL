# Copyright (c) 2025 BAAI. All rights reserved.
#
# Licensed under the Apache License, Version 2.0.
"""Ascend NPU profiler wrapper."""

from __future__ import annotations

import os

import torch
from typing_extensions import override

from vllm.config import ProfilerConfig
from vllm.logger import init_logger
from vllm.profiler.wrapper import WorkerProfiler

logger = init_logger(__name__)


class AscendTorchProfilerWrapper(WorkerProfiler):
    """WorkerProfiler backed by torch_npu.profiler.profile."""

    def __init__(
        self,
        profiler_config: ProfilerConfig,
        worker_name: str,
        local_rank: int,
    ) -> None:
        super().__init__(profiler_config)
        self.local_rank = local_rank
        self.worker_name = worker_name
        self.profiler_config = profiler_config
        self.profiler = None

        if profiler_config.profiler != "torch":
            raise RuntimeError(
                f"Unrecognized profiler: {profiler_config.profiler}"
            )
        if not profiler_config.torch_profiler_dir:
            raise RuntimeError("torch_profiler_dir cannot be empty.")

        import torch_npu.profiler  # type: ignore

        self.torch_npu_profiler = torch_npu.profiler
        self.experimental_config = torch_npu.profiler._ExperimentalConfig(
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

        if local_rank in (None, 0):
            logger.info_once(
                "Ascend NPU profiling enabled. Trace root: %s",
                profiler_config.torch_profiler_dir,
                scope="local",
            )

    def _case_profiler_dir(self) -> str:
        root = self.profiler_config.torch_profiler_dir
        marker = os.path.join(root, ".current_case_dir")

        try:
            case_dir = open(marker, encoding="utf-8").read().strip()
        except OSError:
            case_dir = ""

        if not case_dir:
            return root
        if not os.path.isabs(case_dir):
            case_dir = os.path.join(root, case_dir)

        os.makedirs(case_dir, exist_ok=True)
        return case_dir

    def _create_profiler(self):
        trace_dir = self._case_profiler_dir()

        if self.local_rank in (None, 0):
            logger.info(
                "Ascend NPU profiling trace dir: %s",
                trace_dir,
            )

        return self.torch_npu_profiler.profile(
            activities=[
                self.torch_npu_profiler.ProfilerActivity.CPU,
                self.torch_npu_profiler.ProfilerActivity.NPU,
            ],
            with_stack=False,
            profile_memory=self.profiler_config.torch_profiler_with_memory,
            with_modules=self.profiler_config.torch_profiler_with_stack,
            experimental_config=self.experimental_config,
            on_trace_ready=self.torch_npu_profiler.tensorboard_trace_handler(
                trace_dir,
                worker_name=self.worker_name,
            ),
        )

    @override
    def _start(self) -> None:
        self.profiler = self._create_profiler()
        self.profiler.start()

    @override
    def _stop(self) -> None:
        if self.profiler is None:
            return
        self.profiler.stop()
        self.profiler = None

    @override
    def annotate_context_manager(self, name: str):
        return torch.profiler.record_function(name)
