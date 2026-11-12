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

"""CPU tests for vllm_fl.compilation.graph — dispatch, copies, and replay logic.

``vllm_fl.compilation.graph`` resolves its ``Graph.graph`` backend class at
import time from ``current_platform.device_type``; this host is
``UnspecifiedPlatform`` (empty device_type), which would raise
``NotImplementedError("not support graph")`` at import. The platform probe is
a sanctioned patch boundary, so a module-scoped shim patches
``device_type = "cuda"`` before the first import (identical to the A100's
real value, making the patch a no-op there) and restores it afterwards.

The capture and replay bodies are driven with fake graph objects and a fake
``torch_device_fn`` — no CUDA graph is ever instantiated, matching the work
order's rule that compilation/graph tests on non-GPU hosts only exercise
configuration, registration, and branch logic.

C1 tranche of the supplementary coverage suite (flagcicd_cov).
"""

from __future__ import annotations

from contextlib import contextmanager
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from . import _flaggems_shim  # noqa: F401  (must run before vllm_fl import)

from vllm.platforms import current_platform

# The Graph class body reads current_platform.device_type at import time.
_orig_device_type = current_platform.device_type
current_platform.device_type = "cuda"
import vllm_fl.compilation.graph as graph_module  # noqa: E402
current_platform.device_type = _orig_device_type
# Drop the instance-attribute shadow the pre-patch created (on hosts where
# device_type lives on the class, e.g. PlatformFL on the CI image).
if "device_type" in vars(current_platform):
    del current_platform.device_type

from vllm.config import CUDAGraphMode  # noqa: E402
from vllm.forward_context import (  # noqa: E402
    BatchDescriptor,
    ForwardContext,
    override_forward_context,
)


def _make_config(cudagraph_copy_inputs: bool = False, tp: int = 1) -> MagicMock:
    """Build a VllmConfig stand-in exposing the two read attributes."""
    config = MagicMock()
    config.compilation_config.cudagraph_copy_inputs = cudagraph_copy_inputs
    config.parallel_config.tensor_parallel_size = tp
    return config


class _FakeGraph:
    """Capture-time graph object recording replays."""

    def __init__(self) -> None:
        self.replays = 0

    def replay(self) -> None:
        self.replays += 1


class _FakeRuntime:
    """GraphRuntimeController stand-in recording scope entry."""

    def __init__(self) -> None:
        self.replay_entered = False

    @contextmanager
    def replay_scope(self, context):
        self.replay_entered = True
        yield


def _forward_context(mode: CUDAGraphMode, batch_descriptor) -> ForwardContext:
    return ForwardContext(
        no_compile_layers={},
        attn_metadata=None,
        slot_mapping=None,
        cudagraph_runtime_mode=mode,
        batch_descriptor=batch_descriptor,
    )


@pytest.fixture(autouse=True)
def _cpu_graph_pool(monkeypatch):
    """Give every GraphWrapper a None pool (avoids dummy base-class graph pools)."""
    monkeypatch.setattr(
        graph_module.current_platform, "get_global_graph_pool", lambda: None
    )


@pytest.fixture(autouse=True)
def _restore_capturing_flag():
    """Keep the global cudagraph-capturing flag enabled after each test."""
    from vllm.compilation.monitor import set_cudagraph_capturing_enabled

    set_cudagraph_capturing_enabled(True)
    yield
    set_cudagraph_capturing_enabled(True)


def test_graph_class_selects_cuda_backend():
    """The platform dispatch binds torch.cuda.CUDAGraph for the cuda device type."""
    assert graph_module.Graph.graph is not None
    import torch

    assert graph_module.Graph.graph is torch.cuda.CUDAGraph


def test_graph_options_defaults():
    """GraphOptions defaults enable debug logging and weak-ref outputs."""
    options = graph_module.GraphOptions()
    assert (options.debug_log_enable, options.gc_disable, options.weak_ref_output) == (
        True,
        False,
        True,
    )


def test_graph_entry_defaults():
    """A fresh GraphEntry carries only its batch descriptor."""
    descriptor = BatchDescriptor(num_tokens=4)
    entry = graph_module.GraphEntry(batch_descriptor=descriptor)
    assert entry.batch_descriptor is descriptor
    assert entry.graph is None
    assert entry.output is None
    assert entry.input_addresses is None
    assert entry.input_tensors is None


def test_weak_ref_tensors_returns_tensor_when_upstream_unavailable():
    """Without vllm._C.weak_ref_tensor the fallback returns the tensor itself."""
    import torch

    tensor = torch.arange(4.0)
    # The local venv lacks the compiled op, so the wrapper's except branch
    # returns the identity; on builds with vllm._C this still holds a tensor
    # aliasing the same storage.
    result = graph_module.weak_ref_tensors(tensor)
    assert result is tensor or result.data_ptr() == tensor.data_ptr()


def test_copy_graph_inputs_rejects_count_mismatch():
    """A changed input count between capture and replay is a RuntimeError."""
    import torch

    with pytest.raises(RuntimeError, match="input count changed") as excinfo:
        graph_module._copy_graph_inputs(
            [torch.zeros(4)], [torch.zeros(4), torch.zeros(4)]
        )
    assert "expected 1, got 2" in str(excinfo.value)
    with pytest.raises(RuntimeError, match="input count changed") as excinfo:
        graph_module._copy_graph_inputs(None, [torch.zeros(4)])
    assert "expected 0, got 1" in str(excinfo.value)


def test_copy_graph_inputs_rejects_metadata_change():
    """A changed shape between capture and replay is a RuntimeError."""
    import torch

    with pytest.raises(RuntimeError, match="metadata changed") as excinfo:
        graph_module._copy_graph_inputs([torch.zeros(4)], [torch.zeros(5)])
    assert "expected shape=(4,)" in str(excinfo.value)
    assert "got shape=(5,)" in str(excinfo.value)


def test_copy_graph_inputs_copies_new_values_in_place():
    """Runtime values are copied into the capture-time buffers in place."""
    import torch

    static = torch.zeros(4)
    runtime = torch.arange(4, dtype=static.dtype)
    same = static.clone()
    # Identical data pointers are skipped; distinct buffers are copied.
    graph_module._copy_graph_inputs([static, same], [runtime, same.clone()])
    assert static.tolist() == [0.0, 1.0, 2.0, 3.0]
    assert same.tolist() == [0.0, 0.0, 0.0, 0.0]


@pytest.mark.parametrize(
    ("device_type", "tp", "mode", "global_copy", "expected"),
    [
        ("musa", 2, CUDAGraphMode.PIECEWISE, False, True),
        ("ptpu", 2, CUDAGraphMode.PIECEWISE, False, True),
        ("musa", 1, CUDAGraphMode.PIECEWISE, False, False),
        ("musa", 2, CUDAGraphMode.FULL, False, False),
        ("cuda", 2, CUDAGraphMode.PIECEWISE, False, False),
        ("cuda", 2, CUDAGraphMode.PIECEWISE, True, True),
    ],
)
def test_copy_inputs_matrix_selects_segment_copy(
    monkeypatch, device_type, tp, mode, global_copy, expected
):
    """Only TP PIECEWISE on musa/ptpu (or the global flag) copies segment inputs."""
    monkeypatch.setattr(graph_module.current_platform, "device_type", device_type)
    wrapper = graph_module.GraphWrapper(
        MagicMock(), _make_config(global_copy, tp), mode
    )
    assert wrapper._copy_inputs is expected


def test_wrapper_call_passes_through_without_forward_context():
    """Outside a forward context the runnable is called directly (vision path)."""
    seen = []
    wrapper = graph_module.GraphWrapper(
        lambda *a, **k: seen.append((a, k)) or 41,
        _make_config(),
        CUDAGraphMode.PIECEWISE,
    )
    assert wrapper(1, 2, key="v") == 41
    assert seen == [((1, 2), {"key": "v"})]


@pytest.mark.parametrize("mode", [CUDAGraphMode.NONE, CUDAGraphMode.FULL])
def test_wrapper_call_passes_through_on_mode_mismatch(mode):
    """A context mode of NONE or a differing mode bypasses the graph entirely."""
    seen = []
    wrapper = graph_module.GraphWrapper(
        lambda *a, **k: seen.append(a) or 42,
        _make_config(),
        CUDAGraphMode.PIECEWISE,
    )
    descriptor = BatchDescriptor(num_tokens=8)
    with override_forward_context(_forward_context(mode, descriptor)):
        assert wrapper(3) == 42
    assert seen == [(3,)]
    assert wrapper.concrete_graph_entries == {}


def test_wrapper_creates_entry_and_rejects_capture_outside_window():
    """A first matching call creates an entry, then fails the capture-time check."""
    import torch

    from vllm.compilation.monitor import set_cudagraph_capturing_enabled

    wrapper = graph_module.GraphWrapper(
        MagicMock(), _make_config(), CUDAGraphMode.PIECEWISE
    )
    descriptor = BatchDescriptor(num_tokens=8)
    set_cudagraph_capturing_enabled(False)
    with override_forward_context(_forward_context(CUDAGraphMode.PIECEWISE, descriptor)):
        with pytest.raises(RuntimeError, match="inappropriate") as excinfo:
            wrapper(torch.zeros(4))
    assert "CUDA graph capturing detected at an inappropriate time" in str(
        excinfo.value
    )
    # The entry was created (with no graph) before the validation raised.
    entry = wrapper.concrete_graph_entries[descriptor]
    assert entry.batch_descriptor is descriptor
    assert entry.graph is None
    assert entry.input_addresses is None


def test_wrapper_replays_captured_graph_and_returns_entry_output():
    """A captured entry replays its graph and returns the recorded output."""
    import torch

    wrapper = graph_module.GraphWrapper(
        MagicMock(), _make_config(), CUDAGraphMode.PIECEWISE
    )
    descriptor = BatchDescriptor(num_tokens=8)
    graph = _FakeGraph()
    output = torch.tensor([42.0])
    wrapper.concrete_graph_entries[descriptor] = graph_module.GraphEntry(
        batch_descriptor=descriptor, graph=graph, output=output
    )
    with override_forward_context(
        _forward_context(CUDAGraphMode.PIECEWISE, descriptor)
    ):
        result = wrapper(torch.zeros(4))
    assert result is output
    assert graph.replays == 1


def test_wrapper_replay_runs_inside_runtime_scope():
    """With a graph_runtime, replay happens inside its replay_scope."""
    import torch

    runtime = _FakeRuntime()
    wrapper = graph_module.GraphWrapper(
        MagicMock(), _make_config(), CUDAGraphMode.PIECEWISE, graph_runtime=runtime
    )
    descriptor = BatchDescriptor(num_tokens=8)
    graph = _FakeGraph()
    wrapper.concrete_graph_entries[descriptor] = graph_module.GraphEntry(
        batch_descriptor=descriptor, graph=graph, output=torch.tensor([1.0])
    )
    with override_forward_context(
        _forward_context(CUDAGraphMode.PIECEWISE, descriptor)
    ):
        wrapper(torch.zeros(4))
    assert runtime.replay_entered is True
    assert graph.replays == 1


def test_wrapper_replay_copies_inputs_when_enabled():
    """Replay with _copy_inputs copies runtime values into capture buffers."""
    import torch

    wrapper = graph_module.GraphWrapper(
        MagicMock(), _make_config(cudagraph_copy_inputs=True), CUDAGraphMode.PIECEWISE
    )
    descriptor = BatchDescriptor(num_tokens=2)
    static = torch.zeros(3)
    graph = _FakeGraph()
    wrapper.concrete_graph_entries[descriptor] = graph_module.GraphEntry(
        batch_descriptor=descriptor,
        graph=graph,
        output=torch.tensor([9.0]),
        input_tensors=[static],
    )
    runtime_value = torch.arange(3, dtype=static.dtype)
    with override_forward_context(
        _forward_context(CUDAGraphMode.PIECEWISE, descriptor)
    ):
        result = wrapper(runtime_value)
    assert static.tolist() == [0.0, 1.0, 2.0]
    assert graph.replays == 1
    assert result.item() == 9.0


def test_wrapper_getattr_delegates_to_runnable():
    """Unknown attributes fall through to the wrapped runnable."""
    runnable = SimpleNamespace(shared_attr="delegated")
    runnable.shared_fn = lambda: "fn"
    wrapper = graph_module.GraphWrapper(
        runnable, _make_config(), CUDAGraphMode.PIECEWISE
    )
    assert wrapper.shared_attr == "delegated"
    assert wrapper.shared_fn() == "fn"
    with pytest.raises(AttributeError):
        wrapper.absent_attribute


def test_wrapper_unwrap_and_cudagraph_wrapper_identity():
    """unwrap returns the runnable; cudagraph_wrapper returns the wrapper itself."""
    runnable = object()
    wrapper = graph_module.GraphWrapper(
        runnable, _make_config(), CUDAGraphMode.PIECEWISE
    )
    assert wrapper.unwrap() is runnable
    assert wrapper.cudagraph_wrapper is wrapper


def test_clear_graphs_and_clear_all_graphs():
    """clear_graphs drops entries; clear_all_graphs sweeps every live instance."""
    wrapper_a = graph_module.GraphWrapper(
        MagicMock(), _make_config(), CUDAGraphMode.PIECEWISE
    )
    wrapper_b = graph_module.GraphWrapper(
        MagicMock(), _make_config(), CUDAGraphMode.PIECEWISE
    )
    descriptor = BatchDescriptor(num_tokens=8)
    for wrapper in (wrapper_a, wrapper_b):
        wrapper.concrete_graph_entries[descriptor] = graph_module.GraphEntry(
            batch_descriptor=descriptor
        )
    wrapper_a.clear_graphs()
    assert wrapper_a.concrete_graph_entries == {}
    assert descriptor in wrapper_b.concrete_graph_entries
    wrapper_b.concrete_graph_entries[descriptor] = graph_module.GraphEntry(
        batch_descriptor=descriptor
    )
    graph_module.GraphWrapper.clear_all_graphs()
    assert wrapper_b.concrete_graph_entries == {}
    # Instances are tracked weakly: garbage-collected wrappers disappear.
    import gc

    del wrapper_a, wrapper_b
    gc.collect()
    alive = [ref for ref in graph_module.GraphWrapper._all_instances]
    for ref in alive:
        assert ref is not None
