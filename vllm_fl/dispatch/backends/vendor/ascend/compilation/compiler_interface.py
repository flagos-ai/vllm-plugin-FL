# Copyright (c) 2026 BAAI. All rights reserved.
# Adapted from https://github.com/vllm-project/vllm-ascend/blob/main/vllm_ascend/compilation/compiler_interface.py
# Below is the original copyright:
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""
Ascend-specific compiler interface for vllm-plugin-FL.

This module provides the CompilerInterface subclass used when Ascend graph
compilation (npugraph_ex / torchair) is enabled.  By default vllm-plugin-FL
falls back to eager mode on NPU, so this class is only instantiated when the
user explicitly enables `ascend_compilation_config.enable_npugraph_ex`.
"""

from __future__ import annotations

import logging
from typing import Any, Callable

import torch

from vllm.compilation.compiler_interface import CompilerInterface
from vllm.config import VllmConfig

logger = logging.getLogger(__name__)


class AscendCompiler(CompilerInterface):
    """
    Ascend compiler interface.

    When npugraph_ex is available, this compiler delegates graph compilation to
    it.  Otherwise it raises a clear error directing users to eager mode.
    """

    def __init__(self) -> None:
        super().__init__()
        self._nge = None
        try:
            import npugraph_ex as nge
            self._nge = nge
            self._use_npugraph_ex = True
        except ImportError:
            try:
                import torchair as nge
                self._nge = nge
                self._use_npugraph_ex = False
            except ImportError as exc:
                raise ImportError(
                    "npugraph_ex or torchair is required for AscendCompiler. "
                    "Either install it or disable ascend_compilation_config."
                ) from exc

    def compute_hash(self, vllm_config: VllmConfig) -> str:
        import torch_npu
        return f"{torch_npu.__version__}_{self._use_npugraph_ex}"

    def initialize_cache(self, cache_dir: str, *args, **kwargs) -> None:
        logger.info("AscendCompiler cache dir: %s", cache_dir)

    def compile(
        self,
        graph: Callable,
        example_inputs: list[torch.Tensor],
        additional_inductor_config: dict,
        rank: int = 0,
    ) -> Any:
        raise NotImplementedError(
            "AscendCompiler.compile is not yet implemented in vllm-plugin-FL. "
            "Use backend='eager' (default on NPU) for cudagraph-only execution.")

    def load(self, path: str) -> Any:
        raise NotImplementedError(
            "AscendCompiler.load is not yet implemented in vllm-plugin-FL.")
