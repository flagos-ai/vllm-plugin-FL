# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.
# Copyright 2026 FlagOS Contributors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Ascend graph compiler integration owned by vllm-plugin-FL.

This is a deliberately small adaptation of vLLM-Ascend 0.20.2rc1's
``AscendCompiler``.  The first migration stage only connects vLLM's compiler
interface to the npugraph_ex capture backend.  Graph-fusion passes and static
kernel compilation are intentionally left for later stages.
"""

from __future__ import annotations

import copy
import importlib
from collections.abc import Callable, Mapping
from typing import Any

import torch
import torch.fx as fx
from torch._inductor.compile_fx import graph_returns_tuple, make_graph_return_tuple
from vllm.compilation.compiler_interface import CompilerInterface
from vllm.config import VllmConfig
from vllm.config.utils import Range
from vllm.logger import init_logger

logger = init_logger(__name__)

_ASCEND_COMPILATION_CONFIG_KEY = "ascend_compilation_config"


def _get_ascend_compilation_config(vllm_config: VllmConfig) -> Mapping[str, Any]:
    """Read the FL-owned Ascend compiler options from ``additional_config``."""

    additional_config = getattr(vllm_config, "additional_config", None) or {}
    if not isinstance(additional_config, Mapping):
        raise TypeError("vllm_config.additional_config must be a mapping")

    config = additional_config.get(_ASCEND_COMPILATION_CONFIG_KEY, {})
    if not isinstance(config, Mapping):
        raise TypeError(
            "additional_config['ascend_compilation_config'] must be a mapping"
        )
    return config


def _set_npu_compile_mode() -> None:
    """Disable JIT compilation as required by the capture-mode NPU backend."""

    npu = getattr(torch, "npu", None)
    set_compile_mode = getattr(npu, "set_compile_mode", None)
    if set_compile_mode is None:
        raise RuntimeError(
            "AscendCompiler requires torch.npu.set_compile_mode; ensure "
            "torch_npu is installed and imported before graph compilation"
        )
    set_compile_mode(jit_compile=False)


def _load_npugraph_ex_backend() -> Callable[..., Any]:
    """Create an npugraph_ex backend, with TorchAir compatibility fallback."""

    npugraph_ex_error: ImportError | None = None
    try:
        nge = importlib.import_module("npugraph_ex")
        try:
            config_module = importlib.import_module(
                "npugraph_ex.configs.compiler_config"
            )
            process_kwargs_options = config_module._process_kwargs_options
        except (ImportError, AttributeError):
            # Older npugraph_ex releases expose the same helper here.
            config_module = importlib.import_module(
                "npugraph_ex.configs.npugraphex_config"
            )
            process_kwargs_options = config_module._process_kwargs_options
    except ImportError as exc:
        npugraph_ex_error = exc
    else:
        _set_npu_compile_mode()
        compiler_config = nge.CompilerConfig()
        process_kwargs_options(
            compiler_config,
            {
                "options": {
                    # Execute the FX graph eagerly before ACL graph capture.
                    "force_eager": True,
                    # Match vLLM-Ascend 0.20.2rc1: reinplace can make GELU
                    # fall back to CPU on the target software stack.
                    "inplace_pass": False,
                }
            },
        )
        return nge.get_npu_backend(compiler_config=compiler_config)

    try:
        torchair = importlib.import_module("torchair")
    except ImportError as torchair_error:
        raise RuntimeError(
            "AscendCompiler is enabled, but neither the preferred "
            "'npugraph_ex' backend nor the compatible 'torchair' backend "
            "could be imported. Install a graph backend compatible with the "
            "current torch_npu/CANN stack. "
            f"npugraph_ex error: {npugraph_ex_error!r}; "
            f"torchair error: {torchair_error!r}"
        ) from torchair_error

    _set_npu_compile_mode()
    compiler_config = torchair.CompilerConfig()
    # TorchAir's reduce-overhead mode is the compatibility form of
    # npugraph_ex capture mode.  Keep its behavior aligned with the preferred
    # backend's force_eager=True and inplace_pass=False options.
    compiler_config.mode = "reduce-overhead"
    compiler_config.debug.run_eagerly = True
    compiler_config.debug.aclgraph.disable_reinplace_inplaceable_ops_pass = True
    return torchair.get_npu_backend(compiler_config=compiler_config)


def _compile_with_npugraph_ex(
    graph: fx.GraphModule,
    example_inputs: list[Any],
) -> tuple[Callable[..., Any] | None, None]:
    backend = _load_npugraph_ex_backend()

    # torch.compile requires the backend graph to return a tuple.
    if not graph_returns_tuple(graph):
        return make_graph_return_tuple(graph, example_inputs, backend), None
    return backend(graph, example_inputs), None


class AscendCompiler(CompilerInterface):
    """Compile vLLM FX graphs with FL's local Ascend graph backend."""

    name = "AscendCompiler"

    def __init__(self) -> None:
        self._vllm_config: VllmConfig | None = None

    def compute_hash(self, vllm_config: VllmConfig) -> str:
        # CompilerManager invokes this before the first compile.  Keep the
        # exact config instance so compile() reads the same additional_config
        # that participated in vLLM's cache hash.
        self._vllm_config = vllm_config
        return vllm_config.compute_hash()

    def compile(
        self,
        graph: fx.GraphModule,
        example_inputs: list[Any],
        compiler_config: dict[str, Any],
        compile_range: Range,
        key: str | None = None,
    ) -> tuple[Callable[..., Any] | None, Any | None]:
        del compiler_config, compile_range, key

        if self._vllm_config is None:
            raise RuntimeError(
                "AscendCompiler.compute_hash(vllm_config) must be called "
                "before compile()"
            )

        ascend_config = _get_ascend_compilation_config(self._vllm_config)
        if not bool(ascend_config.get("enable_npugraph_ex", False)):
            raise RuntimeError(
                "AscendCompiler was selected while "
                "additional_config['ascend_compilation_config']"
                "['enable_npugraph_ex'] is not enabled"
            )
        if bool(ascend_config.get("enable_static_kernel", False)):
            raise NotImplementedError(
                "enable_static_kernel is not supported by the initial FL "
                "AscendCompiler migration; keep it false"
            )

        logger.info(
            "Ascend npugraph_ex compilation is enabled for the current FX graph."
        )

        # Compiler backends may mutate FX graphs in-place.  Keep the graph
        # supplied by vLLM reusable across compile ranges.
        graph = copy.deepcopy(graph)

        from torch._guards import detect_fake_mode

        current_fake_mode = detect_fake_mode()
        if current_fake_mode is not None:
            example_inputs = [
                current_fake_mode.from_tensor(inp)
                if (
                    isinstance(inp, torch.Tensor)
                    and hasattr(inp, "fake_mode")
                    and inp.fake_mode is not current_fake_mode
                )
                else inp
                for inp in example_inputs
            ]

        return _compile_with_npugraph_ex(graph, example_inputs)
