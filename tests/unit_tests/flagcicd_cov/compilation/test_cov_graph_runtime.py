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

"""CPU tests for vllm_fl.compilation.graph_runtime — the NPU task lifecycle.

Drives GraphRuntimeController's capture/replay scopes, backend binding, and
decoder-graph cleanup with a fake ``torch_device_fn`` (a platform probe — a
sanctioned patch boundary, since the local platform exposes none).

C1 tranche of the supplementary coverage suite (flagcicd_cov).
"""

from __future__ import annotations

from contextlib import contextmanager
from types import SimpleNamespace

import pytest

# The shim import is load-bearing: it must precede vllm_fl/vllm
# imports on hosts without triton. Do not alphabetize.
from . import _flaggems_shim  # noqa: F401  (must run before vllm_fl import)
from vllm.platforms import current_platform

import vllm_fl.compilation.graph_params as gp
import vllm_fl.compilation.graph_runtime as gr


class _FakeStream:
    def synchronize(self) -> None:
        pass

    def wait_stream(self, other) -> None:
        pass


class _FakeDeviceFn:
    """Stand-in for current_platform.torch_device_fn (None on this host)."""

    def current_stream(self) -> _FakeStream:
        return _FakeStream()

    def Stream(self, device=None) -> _FakeStream:
        return _FakeStream()

    @contextmanager
    def stream(self, target):
        yield


@pytest.fixture(autouse=True)
def _fake_device_fn(monkeypatch):
    """Provide a working torch_device_fn and drop the singleton state after."""
    monkeypatch.setattr(current_platform, "torch_device_fn", _FakeDeviceFn())
    gp.clear_graph_params()
    yield
    gp.clear_graph_params()


def _context(num_tokens) -> SimpleNamespace:
    return SimpleNamespace(batch_descriptor=SimpleNamespace(num_tokens=num_tokens))


def test_num_tokens_accepts_integers():
    """The token count is read off the batch descriptor."""
    assert gr.GraphRuntimeController._num_tokens(_context(8)) == 8
    assert gr.GraphRuntimeController._num_tokens(_context(1024)) == 1024


@pytest.mark.parametrize("bad", [True, False, None, "8", 8.0])
def test_num_tokens_rejects_non_integer_values(bad):
    """Booleans and non-int token counts are rejected with RuntimeError."""
    ctx = SimpleNamespace(batch_descriptor=SimpleNamespace(num_tokens=bad))
    with pytest.raises(RuntimeError, match="require integer BatchDescriptor") as excinfo:
        gr.GraphRuntimeController._num_tokens(ctx)
    assert "integer BatchDescriptor.num_tokens" in str(excinfo.value)


def test_controller_initial_state():
    """A fresh controller has no backends bound and is inactive."""
    controller = gr.GraphRuntimeController(vllm_config="cfg")
    assert controller.attention_impls is None
    assert controller.update_stream is None
    assert controller._active is False
    assert controller.wrapper_type is None


def test_capture_scope_previews_and_releases_shape_state():
    """capture_scope prepares the shape, marks the context, and unwinds both."""

    class Probe:
        pass

    controller = gr.GraphRuntimeController(vllm_config="cfg")
    probe = Probe()
    probe.batch_descriptor = SimpleNamespace(num_tokens=16)
    with controller.capture_scope(probe):
        assert controller._active is True
        assert probe.capturing is True
        assert 16 in gp.get_graph_params().attn_params
        # A nested scope is rejected by the active assert.
        with pytest.raises(AssertionError) as excinfo, controller.capture_scope(probe):
            pass
        assert "assert not self._active" in str(excinfo.value) or excinfo.value
    assert controller._active is False
    assert probe.capturing is False


def test_capture_scope_rejects_invalid_token_counts():
    """A non-integer token count fails the scope before any state is touched."""
    controller = gr.GraphRuntimeController(vllm_config="cfg")
    with pytest.raises(RuntimeError, match="require integer") as excinfo, controller.capture_scope(
        _context(True)
    ):
        pass
    assert "integer BatchDescriptor.num_tokens" in str(excinfo.value)
    assert controller._active is False


def test_replay_scope_requires_bound_backends():
    """A replay without bound attention backends raises at update time."""
    controller = gr.GraphRuntimeController(vllm_config="cfg")
    with pytest.raises(RuntimeError, match="must be bound before FULL replay") as excinfo, controller.replay_scope(
        _context(4)
    ):
        pass
    assert "must be bound before FULL replay" in str(excinfo.value)
    assert controller._active is False


def test_replay_scope_with_empty_bindings_is_a_noop():
    """An empty bound tuple leaves the replay scope clean."""
    controller = gr.GraphRuntimeController(vllm_config="cfg")
    controller.attention_impls = ()
    with controller.replay_scope(_context(4)):
        assert controller._active is True
    assert controller._active is False


class _FakeImpl:
    """Attention implementation stand-in recording update_graph_params calls."""

    def __init__(self) -> None:
        self.calls = []

    def update_graph_params(self, stream, context, num_tokens, vllm_config):
        self.calls.append((stream, context, num_tokens, vllm_config))


class _Backend:
    """Backend whose get_impl_cls returns a shared implementation instance."""

    __name__ = "FakeBackend"

    instance = _FakeImpl()

    @staticmethod
    def get_impl_cls():
        return _Backend.instance


class _PlainBackend:
    """Backend that keeps the base AttentionBackend.get_impl_cls (skipped)."""

    __name__ = "PlainBackend"


def test_bind_attention_groups_dedups_and_skips_plain_backends():
    """Duplicate backends bind once; base get_impl_cls backends are skipped."""
    from vllm.v1.attention.backend import AttentionBackend

    _PlainBackend.get_impl_cls = staticmethod(AttentionBackend.get_impl_cls)
    controller = gr.GraphRuntimeController(vllm_config="cfg")
    groups = [
        [SimpleNamespace(backend=_Backend()), SimpleNamespace(backend=_Backend())],
        [SimpleNamespace(backend=_PlainBackend())],
    ]
    controller.bind_attention_groups(groups)
    assert controller.attention_impls == (_Backend.instance,)


def test_bind_attention_groups_wraps_factory_failures():
    """A raising backend factory becomes a RuntimeError naming the backend."""

    class FailingBackend:
        __name__ = "FailingBackend"

        @staticmethod
        def get_impl_cls():
            raise ImportError("boom")

    controller = gr.GraphRuntimeController(vllm_config="cfg")
    with pytest.raises(
        RuntimeError, match="Failed to resolve Ascend graph backend FailingBackend"
    ) as excinfo:
        controller.bind_attention_groups([[SimpleNamespace(backend=FailingBackend())]])
    assert "FailingBackend" in str(excinfo.value)


def test_bind_attention_groups_requires_update_graph_params():
    """Implementations lacking update_graph_params are rejected by name."""

    class NoUpdateBackend:
        __name__ = "NoUpdateBackend"

        @staticmethod
        def get_impl_cls():
            return object()

    controller = gr.GraphRuntimeController(vllm_config="cfg")
    with pytest.raises(
        RuntimeError, match="NoUpdateBackend lacks update_graph_params"
    ) as excinfo:
        controller.bind_attention_groups(
            [[SimpleNamespace(backend=NoUpdateBackend())]]
        )
    assert "NoUpdateBackend" in str(excinfo.value)


def test_replay_scope_updates_bound_implementations():
    """After the replay yields, each impl is updated with stream/context/shape."""
    _Backend.instance = _FakeImpl()
    controller = gr.GraphRuntimeController(vllm_config="the-config")
    controller.bind_attention_groups([[SimpleNamespace(backend=_Backend())]])
    context = _context(5)
    with controller.replay_scope(context):
        pass
    calls = _Backend.instance.calls
    assert len(calls) == 1
    stream, ctx, num_tokens, config = calls[0]
    assert ctx is context
    assert num_tokens == 5
    assert config == "the-config"
    assert isinstance(stream, _FakeStream)
    # The update stream is created once and reused.
    assert controller.update_stream is stream


def test_clear_decoder_graphs_clears_wrappers_and_params():
    """clear_decoder_graphs sweeps wrapper entries and the shape registry."""
    cleared = []
    wrapper_type = SimpleNamespace(clear_all_graphs=lambda: cleared.append(1))
    gp.prepare_graph_params(3)
    controller = gr.GraphRuntimeController(
        vllm_config="cfg", full_graph_wrapper_type=wrapper_type
    )
    controller.clear_decoder_graphs()
    assert cleared == [1]
    assert gp.get_graph_params().events == {}


def test_clear_decoder_graphs_without_wrapper_type():
    """A controller without a wrapper type still clears the shape registry."""
    gp.prepare_graph_params(3)
    controller = gr.GraphRuntimeController(vllm_config="cfg")
    controller.clear_decoder_graphs()
    assert gp.get_graph_params().events == {}


def test_get_graph_capture_delegates_non_npu_platforms():
    """On non-NPU devices the caller's default capture context is used."""

    @contextmanager
    def default_capture(device=None):
        yield f"default-ctx-{device}"

    capture = gr.get_graph_capture(default_capture)
    with capture(device="cuda") as context:
        assert context == "default-ctx-cuda"


def test_get_graph_capture_builds_npu_stream_context(monkeypatch):
    """On NPU devices a GraphCaptureContext is built around a fresh stream."""
    monkeypatch.setattr(current_platform, "device_type", "npu")

    @contextmanager
    def default_capture(device=None):
        raise AssertionError("default capture must not run on npu")
        yield "unused"

    capture = gr.get_graph_capture(default_capture)
    from vllm.distributed.parallel_state import GraphCaptureContext

    with capture(device=torch_device()) as context:
        assert isinstance(context, GraphCaptureContext)


def torch_device():
    """A dummy device token for the capture signature."""
    return "npu-device-0"
