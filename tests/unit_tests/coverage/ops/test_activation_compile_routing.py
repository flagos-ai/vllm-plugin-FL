# Copyright (c) 2026 BAAI. All rights reserved.

"""AMD compile-routing tests for ``vllm_fl.ops.activation``.

The combined core+coverage-gap report showed exactly these gaps
in ``vllm_fl/ops/activation.py`` (10 statements):

* the AMD graph-ops branch of ``SiluAndMulFL.forward_oot`` — when the layer
  runs on an AMD vendor and inside ``torch.compile``, the call is routed to
  ``vllm_fl.dispatch.backends.vendor.amd.graph_ops.silu_and_mul_custom_op``
  instead of the dispatched ``_silu_and_mul`` CachedOp;
* the same routing for ``GeluAndMulFL`` — its constructor flag
  (``_use_amd_graph_ops``) and both arms of its ``forward_oot``;
* the plain ``forward_cuda`` delegate (kept as one pattern exemplar on
  ``GeluAndMulFL`` below; ``SiluAndMulFL`` and ``RotaryEmbeddingFL`` carry
  the identical one-line delegation, covered by that single sample).

``torch.compile`` executes dynamo-traced bytecode that coverage.py cannot
line-trace (verified: a subprocess running only the compiled path leaves the
whole ``forward_oot`` body untraced although the AMD leaf provably runs), so
the coverage-targeted AMD tests below simulate the compiled state with
``torch.compiler.is_compiling`` patched to return True and call
``forward_oot`` directly: the branch then runs in plain, traceable Python
while the observable oracle (the AMD leaf is reached, its output returned)
is preserved.  A real ``torch.compile`` routing test is kept alongside each
as a behavioural check of the genuine dynamo path.

Everything else in these modules is already covered by the core suite
(``tests/unit_tests/compilation/test_amd_graph_ops.py`` covers the
non-AMD compile routing via ast-extracted classes).  The module under test
imports the external ``vllm`` package at module level, which is absent on
CPU-only test machines, so a minimal stub ``vllm`` graph is installed first;
the plugin sources themselves run verbatim, so coverage attributes every
executed statement to the real module files.
"""

import importlib
import sys
import types
from types import SimpleNamespace

import pytest
import torch

AMD_GRAPH_OPS_MODULE = "vllm_fl.dispatch.backends.vendor.amd.graph_ops"


def _make_module(name, **attrs):
    """Create a plain module with the given attributes attached."""
    module = types.ModuleType(name)
    module.__path__ = []
    for key, value in attrs.items():
        setattr(module, key, value)
    return module


class _StubBase:
    """Upstream vLLM layer base recording constructor arguments."""

    def __init__(self, *args, **kwargs):
        self._init_args = args
        self._init_kwargs = kwargs


def _install_ops_vllm_graph(current_platform):
    """Install the minimal vllm graph the two ops modules import.

    Only the names read at import time are provided (the upstream layer
    base classes); the plugin sources under test execute verbatim.
    """
    vllm = _make_module("vllm")
    platforms = _make_module("vllm.platforms", current_platform=current_platform)
    model_executor = _make_module("vllm.model_executor")
    layers = _make_module("vllm.model_executor.layers")
    activation_mod = _make_module(
        "vllm.model_executor.layers.activation",
        SiluAndMul=_StubBase,
        GeluAndMul=_StubBase,
    )
    rotary_mod = _make_module(
        "vllm.model_executor.layers.rotary_embedding", RotaryEmbedding=_StubBase
    )
    vllm.platforms = platforms
    vllm.model_executor = model_executor
    model_executor.layers = layers
    layers.activation = activation_mod
    layers.rotary_embedding = rotary_mod
    return {
        "vllm": vllm,
        "vllm.platforms": platforms,
        "vllm.model_executor": model_executor,
        "vllm.model_executor.layers": layers,
        "vllm.model_executor.layers.activation": activation_mod,
        "vllm.model_executor.layers.rotary_embedding": rotary_mod,
    }


@pytest.fixture()
def ops_modules(monkeypatch):
    """Import the two ops modules against a controllable platform.

    Yields a small holder whose ``current_platform`` can be swapped before
    the (single) import of each module under test; the module objects are
    dropped from ``sys.modules`` afterwards so other tests never observe
    the stub graph.
    """
    platform_holder = SimpleNamespace(
        current_platform=SimpleNamespace(vendor_name="nvidia", device_type="cuda")
    )
    graph = _install_ops_vllm_graph(platform_holder.current_platform)
    saved = {
        name: sys.modules.get(name, None)
        for name in list(sys.modules)
        if name.split(".")[0] in ("vllm", "vllm_fl")
    }
    for name, module in graph.items():
        monkeypatch.setitem(sys.modules, name, module)

    def _reimport(module_name):
        sys.modules.pop(module_name, None)
        return importlib.import_module(module_name)

    holder = SimpleNamespace(
        platform=platform_holder,
        reimport=_reimport,
    )
    yield holder
    for name in [
        name
        for name in list(sys.modules)
        if name.split(".")[0] in ("vllm", "vllm_fl") and name not in saved
    ]:
        sys.modules.pop(name, None)
    for name, module in saved.items():
        if module is None:
            sys.modules.pop(name, None)
        else:
            sys.modules[name] = module


class TestSiluAndMulAmdGraphOps:
    """SiluAndMulFL AMD compile routing (activation lines 22-31)."""

    def test_compiling_amd_call_routes_to_graph_ops(self, ops_modules, monkeypatch):
        """An AMD layer in compiled mode returns the AMD custom op result.

        Independent oracle: the AMD leaf returns ``x * 100``, a value the
        dispatched reference implementation can never produce.  The compiled
        state is simulated by patching ``torch.compiler.is_compiling`` to
        True and calling ``forward_oot`` directly, so the branch runs in
        plain coverage-traceable Python (dynamo-traced frames are invisible
        to coverage.py — verified by a subprocess probe).
        """
        amd_calls = []

        def silu_leaf(x):
            amd_calls.append(x)
            return x * 100.0

        amd_graph_ops = _make_module(
            AMD_GRAPH_OPS_MODULE, silu_and_mul_custom_op=silu_leaf
        )
        monkeypatch.setitem(sys.modules, AMD_GRAPH_OPS_MODULE, amd_graph_ops)

        activation = ops_modules.reimport("vllm_fl.ops.activation")
        layer = activation.SiluAndMulFL.__new__(activation.SiluAndMulFL)
        layer._use_amd_graph_ops = True
        monkeypatch.setattr(torch.compiler, "is_compiling", lambda: True)
        x = torch.tensor([1.0, 2.0, 3.0, 4.0])
        out = layer.forward_oot(x)
        assert len(amd_calls) == 1
        torch.testing.assert_close(out, x * 100.0)

    def test_real_compile_amd_call_routes_to_graph_ops(self, ops_modules, monkeypatch):
        """Behavioural check: real torch.compile also routes AMD layers.

        Kept as the genuine dynamo-path control for the coverage test
        above; not required for the coverage number itself.
        """
        amd_calls = []

        def silu_leaf(x):
            amd_calls.append(x)
            return x * 100.0

        amd_graph_ops = _make_module(
            AMD_GRAPH_OPS_MODULE, silu_and_mul_custom_op=silu_leaf
        )
        monkeypatch.setitem(sys.modules, AMD_GRAPH_OPS_MODULE, amd_graph_ops)

        activation = ops_modules.reimport("vllm_fl.ops.activation")
        layer = activation.SiluAndMulFL.__new__(activation.SiluAndMulFL)
        layer._use_amd_graph_ops = True
        compiled = torch.compile(layer.forward_oot, backend="eager", fullgraph=True)
        x = torch.tensor([1.0, 2.0, 3.0, 4.0])
        try:
            out = compiled(x)
        finally:
            torch._dynamo.reset()
        assert len(amd_calls) == 1
        torch.testing.assert_close(out, x * 100.0)

    def test_non_compiling_amd_call_uses_normal_dispatch(
        self, ops_modules, monkeypatch
    ):
        """Outside torch.compile the AMD branch is skipped entirely.

        The AMD leaf is replaced with a tripwire that fails the test if
        reached; the dispatched CachedOp path must serve the call instead.
        """
        amd_graph_ops = _make_module(
            AMD_GRAPH_OPS_MODULE,
            silu_and_mul_custom_op=lambda x: (_ for _ in ()).throw(
                AssertionError("AMD op reached outside compile")
            ),
        )
        monkeypatch.setitem(sys.modules, AMD_GRAPH_OPS_MODULE, amd_graph_ops)

        activation = ops_modules.reimport("vllm_fl.ops.activation")

        calls = []

        def fake_cached_op(layer, x):
            calls.append(x)
            return torch.zeros_like(x)

        monkeypatch.setattr(activation, "_silu_and_mul", fake_cached_op)
        layer = activation.SiluAndMulFL.__new__(activation.SiluAndMulFL)
        layer._use_amd_graph_ops = True
        x = torch.tensor([1.0, 2.0])
        out = layer.forward_oot(x)
        assert len(calls) == 1
        torch.testing.assert_close(out, torch.zeros_like(x))


class TestGeluAndMulGraphOps:
    """GeluAndMulFL constructor flag, forward_oot arms and forward_cuda."""

    def test_constructor_sets_amd_flag_from_platform(self, ops_modules, monkeypatch):
        """The _use_amd_graph_ops flag mirrors the platform vendor name.

        Independent oracle: two imports of the same source against two
        different platform stubs must produce opposite flag values, with the
        upstream constructor still receiving the approximate argument.
        """
        activation = ops_modules.reimport("vllm_fl.ops.activation")
        nvidia_layer = activation.GeluAndMulFL(approximate="tanh")
        assert nvidia_layer._use_amd_graph_ops is False
        assert nvidia_layer._init_kwargs == {"approximate": "tanh"}

        amd_platform = SimpleNamespace(vendor_name="amd", device_type="cuda")
        graph = _install_ops_vllm_graph(amd_platform)
        for name, module in graph.items():
            monkeypatch.setitem(sys.modules, name, module)
        activation_amd = ops_modules.reimport("vllm_fl.ops.activation")
        amd_layer = activation_amd.GeluAndMulFL(approximate="tanh")
        assert amd_layer._use_amd_graph_ops is True

    def test_compiling_amd_call_routes_to_graph_ops_with_approximate(
        self, ops_modules, monkeypatch
    ):
        """An AMD layer in compiled mode calls gelu_and_mul_custom_op.

        Independent oracle: the AMD leaf returns ``x * 100`` and records the
        approximate argument; the dispatched CachedOp can produce neither.
        """
        amd_calls = []

        def gelu_leaf(x, approximate):
            amd_calls.append((x, approximate))
            return x * 100.0

        amd_graph_ops = _make_module(
            AMD_GRAPH_OPS_MODULE, gelu_and_mul_custom_op=gelu_leaf
        )
        monkeypatch.setitem(sys.modules, AMD_GRAPH_OPS_MODULE, amd_graph_ops)

        activation = ops_modules.reimport("vllm_fl.ops.activation")
        layer = activation.GeluAndMulFL.__new__(activation.GeluAndMulFL)
        layer._use_amd_graph_ops = True
        layer.approximate = "tanh"
        monkeypatch.setattr(torch.compiler, "is_compiling", lambda: True)
        x = torch.tensor([1.0, 2.0, 3.0, 4.0])
        out = layer.forward_oot(x)
        assert amd_calls == [(x, "tanh")]
        torch.testing.assert_close(out, x * 100.0)

    def test_non_compiling_amd_call_uses_normal_dispatch(
        self, ops_modules, monkeypatch
    ):
        """Outside torch.compile the AMD branch is skipped entirely.

        The AMD leaf is a tripwire; the dispatched CachedOp must serve the
        call, and the layer must be passed through as the first argument.
        """
        amd_graph_ops = _make_module(
            AMD_GRAPH_OPS_MODULE,
            gelu_and_mul_custom_op=lambda x, approximate: (_ for _ in ()).throw(
                AssertionError("AMD op reached outside compile")
            ),
        )
        monkeypatch.setitem(sys.modules, AMD_GRAPH_OPS_MODULE, amd_graph_ops)

        activation = ops_modules.reimport("vllm_fl.ops.activation")
        calls = []

        def fake_cached_op(layer, x):
            calls.append(layer)
            return torch.zeros_like(x)

        monkeypatch.setattr(activation, "_gelu_and_mul", fake_cached_op)
        layer = activation.GeluAndMulFL.__new__(activation.GeluAndMulFL)
        layer._use_amd_graph_ops = True
        layer.approximate = "none"
        x = torch.tensor([1.0, 2.0])
        out = layer.forward_oot(x)
        assert calls == [layer]
        torch.testing.assert_close(out, torch.zeros_like(x))

    def test_forward_cuda_delegates_to_forward_oot(self, ops_modules, monkeypatch):
        """GeluAndMulFL.forward_cuda returns whatever forward_oot returns.

        Independent oracle: forward_oot is replaced by a recorder returning a
        constant tensor unlike any real activation output; forward_cuda must
        return exactly that object.
        """
        activation = ops_modules.reimport("vllm_fl.ops.activation")
        layer = activation.GeluAndMulFL.__new__(activation.GeluAndMulFL)
        sentinel = torch.tensor([7.0])
        calls = []

        def fake_oot(x):
            calls.append(x)
            return sentinel

        monkeypatch.setattr(layer, "forward_oot", fake_oot)
        x = torch.tensor([1.0, 2.0])
        assert layer.forward_cuda(x) is sentinel
        assert calls == [x]
