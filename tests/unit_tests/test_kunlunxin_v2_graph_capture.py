# Copyright (c) 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0

import runpy
import sys
from pathlib import Path
from types import ModuleType
from unittest.mock import Mock

import pytest


@pytest.mark.parametrize("staged", [False, True])
def test_kunlunxin_replaces_v2_graph_capture_alias(monkeypatch, staged):
    parallel_state = ModuleType("vllm.distributed.parallel_state")
    parallel_state.graph_capture = Mock(side_effect=AssertionError)
    cudagraph_utils = ModuleType("vllm.v1.worker.gpu.cudagraph_utils")
    cudagraph_utils.graph_capture = parallel_state.graph_capture
    model_runner = ModuleType("vllm_fl.worker.model_runner")
    model_runner._device_graph_capture = Mock()
    for module in (parallel_state, cudagraph_utils, model_runner):
        monkeypatch.setitem(sys.modules, module.__name__, module)

    path = (
        Path(__file__).resolve().parents[2]
        / "vllm_fl/dispatch/backends/vendor/kunlunxin/patch.py"
    )
    patch_v2_graph_capture = runpy.run_path(str(path))["patch_v2_graph_capture"]
    original = parallel_state.graph_capture
    replacement = model_runner._device_graph_capture
    replacements = [] if staged else None

    patch_v2_graph_capture(replacements)

    assert parallel_state.graph_capture is original
    if staged:
        assert cudagraph_utils.graph_capture is original
        assert replacements == [
            (cudagraph_utils, "graph_capture", original, replacement, True)
        ]
    else:
        assert cudagraph_utils.graph_capture is replacement
        device = object()
        cudagraph_utils.graph_capture(device=device)
        replacement.assert_called_once_with(device=device)
