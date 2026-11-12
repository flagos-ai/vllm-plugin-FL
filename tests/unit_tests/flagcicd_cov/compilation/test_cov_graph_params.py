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

"""CPU tests for vllm_fl.compilation.graph_params — the per-shape task registry.

Covers the prepare/require/clear lifecycle and the weak-ref workspace release
against the module singleton (restored to empty after every test).

C1 tranche of the supplementary coverage suite (flagcicd_cov).
"""

from __future__ import annotations

import pytest

from . import _flaggems_shim  # noqa: F401  (must run before vllm_fl import)

import torch

import vllm_fl.compilation.graph_params as gp


@pytest.fixture(autouse=True)
def _clean_singleton():
    """Run every test against an empty registry and leave it empty after."""
    gp.clear_graph_params()
    yield
    gp.clear_graph_params()


def test_fresh_registry_starts_empty():
    """The module singleton exposes four empty per-shape maps."""
    params = gp.get_graph_params()
    assert params.events == {}
    assert params.workspaces == {}
    assert params.handles == {}
    assert params.attn_params == {}


def test_prepare_shape_initializes_all_four_maps():
    """prepare_shape seeds empty task lists and a None workspace per shape."""
    params = gp.prepare_graph_params(16)
    assert params is gp.get_graph_params()
    assert params.events[16] == []
    assert params.workspaces[16] is None
    assert params.handles[16] == []
    assert params.attn_params[16] == []


def test_prepare_shape_rejects_non_positive_sizes():
    """Zero or negative capture sizes are rejected with ValueError."""
    with pytest.raises(ValueError, match="must be positive, got 0") as excinfo:
        gp.prepare_graph_params(0)
    assert str(excinfo.value) == "Graph capture size must be positive, got 0"
    with pytest.raises(ValueError, match="must be positive, got -3") as excinfo:
        gp.prepare_graph_params(-3)
    assert str(excinfo.value) == "Graph capture size must be positive, got -3"


def test_prepare_shape_rejects_shapes_with_captured_state():
    """A re-prepare with leftover events, handles, or params is a RuntimeError."""
    params = gp.prepare_graph_params(8)
    params.events[8].append("event")
    with pytest.raises(RuntimeError, match="already contain captured state") as excinfo:
        gp.prepare_graph_params(8)
    assert "capture size 8" in str(excinfo.value)
    gp.clear_graph_params()

    params = gp.prepare_graph_params(8)
    params.handles[8].append("handle")
    with pytest.raises(RuntimeError, match="already contain captured state") as excinfo:
        gp.prepare_graph_params(8)
    assert "capture size 8" in str(excinfo.value)
    gp.clear_graph_params()

    params = gp.prepare_graph_params(8)
    params.attn_params[8].append("attn")
    with pytest.raises(RuntimeError, match="already contain captured state") as excinfo:
        gp.prepare_graph_params(8)
    assert "capture size 8" in str(excinfo.value)


def test_prepare_shape_allows_reuse_after_full_clear():
    """After clear(), the same shape can be prepared again."""
    params = gp.prepare_graph_params(8)
    params.events[8].append("event")
    gp.clear_graph_params()
    params = gp.prepare_graph_params(8)
    assert params.events[8] == []


def test_require_shape_rejects_unprepared_sizes():
    """require_shape raises for a shape that was never prepared."""
    gp.prepare_graph_params(4)
    with pytest.raises(RuntimeError, match="not prepared for capture size 9") as excinfo:
        gp.get_graph_params().require_shape(9)
    assert "capture size 9" in str(excinfo.value)


def test_require_shape_accepts_prepared_sizes():
    """require_shape returns quietly for a prepared shape."""
    gp.prepare_graph_params(4)
    result = gp.get_graph_params().require_shape(4)
    assert result is None


def test_clear_drops_every_shape():
    """clear empties all four maps of the registry."""
    params = gp.prepare_graph_params(4)
    params.events[4].append("e")
    params.workspaces[4] = torch.zeros(2)
    params.handles[4].append("h")
    params.attn_params[4].append("a")
    gp.clear_graph_params()
    assert params.events == {}
    assert params.workspaces == {}
    assert params.handles == {}
    assert params.attn_params == {}


def test_weak_ref_workspace_keeps_none_workspace():
    """A None workspace is a documented no-op for the weak-ref release."""
    gp.prepare_graph_params(8)
    gp.weak_ref_workspace(8)
    assert gp.get_graph_params().workspaces[8] is None


def test_weak_ref_workspace_requires_prepared_shape():
    """The release path validates the shape through require_shape."""
    with pytest.raises(RuntimeError, match="not prepared for capture size 5") as excinfo:
        gp.weak_ref_workspace(5)
    assert "capture size 5" in str(excinfo.value)


def test_weak_ref_workspace_keeps_workspace_object_on_cpu_builds():
    """Without the compiled vllm._C op the weak-ref falls back to the tensor.

    On the A100 image vllm._C exists and the workspace becomes a weak ref
    aliasing the same storage; on this CPU build the fallback keeps the very
    same object. Both preserve the storage address retained by the graph.
    """
    workspace = torch.arange(6.0)
    params = gp.prepare_graph_params(8)
    params.workspaces[8] = workspace
    gp.weak_ref_workspace(8)
    released = params.workspaces[8]
    assert released is workspace or released.data_ptr() == workspace.data_ptr()
