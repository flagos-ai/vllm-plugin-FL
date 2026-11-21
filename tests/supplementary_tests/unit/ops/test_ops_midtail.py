# Copyright (c) 2026 BAAI. All rights reserved.

"""Supplementary tests for the remaining uncovered lines of ``vllm_fl.ops``.

The combined core+supplementary coverage report shows exactly these gaps:

``vllm_fl/ops/activation.py`` (10 statements):

* the AMD graph-ops branch of ``SiluAndMulFL.forward_oot`` — when the layer
  runs on an AMD vendor and inside ``torch.compile``, the call is routed to
  ``vllm_fl.dispatch.backends.vendor.amd.graph_ops.silu_and_mul_custom_op``
  instead of the dispatched ``_silu_and_mul`` CachedOp;
* the same routing for ``GeluAndMulFL`` — its constructor flag
  (``_use_amd_graph_ops``), both arms of its ``forward_oot`` and its
  ``forward_cuda`` delegate are never executed by any suite;
* ``SiluAndMulFL.forward_cuda`` (the plain delegate at line 31).

``torch.compile`` executes dynamo-traced bytecode that coverage.py cannot
line-trace (verified: a subprocess running only the compiled path leaves the
whole ``forward_oot`` body untraced although the AMD leaf provably runs), so
the coverage-targeted AMD tests below simulate the compiled state with
``torch.compiler.is_compiling`` patched to return True and call
``forward_oot`` directly: the branch then runs in plain, traceable Python
while the observable oracle (the AMD leaf is reached, its output returned)
is preserved.  A real ``torch.compile`` routing test is kept alongside each
as a behavioural check of the genuine dynamo path.

``vllm_fl/ops/rotary_embedding.py`` (10 statements):

* the NPU init branch of ``RotaryEmbeddingFL.__init__`` — on an NPU device
  type the constructor imports and runs
  ``vllm_fl.dispatch.backends.vendor.ascend.impl.rotary.initialize_rotary_embedding_ascend``;
* the AMD graph-ops branch of ``RotaryEmbeddingFL.forward_oot`` (the
  ``rotary_embedding_custom_op`` call plus the ``copy_`` aliasing), the
  ``rotary_dim == head_size`` reshape arm and the ``forward_cuda`` delegate;
  the ``key is None`` native-path arm is likewise unexecuted by any suite;

``vllm_fl/distributed/kv_transfer/flagcx_connector.py`` (lines 74, 80, 81):

* line 74 — ``FLAGCX_PATH`` pointing at a real directory that is not yet on
  ``sys.path`` gets appended at import time;
* lines 80-81 — a failing ``plugin.interservice.flagcx_wrapper`` import
  re-raises as an ``ImportError`` with the FLAGCX_PATH guidance message.

Everything else in these modules is already covered by the core suite
(``tests/unit_tests/compilation/test_amd_graph_ops.py`` covers the
non-AMD compile routing via ast-extracted classes) and the supplementary
distributed suite; the tests below only add what is missing.  The modules
under test import the external ``vllm`` package at module level, which is
absent on CPU-only test machines, so a minimal stub ``vllm`` graph is
installed first (the ``test_custom_ops_registration.py`` pattern); the plugin
sources themselves run verbatim, so coverage attributes every executed
statement to the real module files.  The flagcx connector is re-executed from
its source file under a unique module name against the shared
``_flagcx_stubs.FlagcxStubEnv`` boundary (the ``test_flagcx_transfer_plan.py``
pattern), so its import-time environment handling runs unmodified.
"""

import importlib.util
import sys
import types
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

pytestmark = [pytest.mark.supplementary]

_ROOT = Path(__file__).resolve().parents[4]
_CONNECTOR_SOURCE = _ROOT / "vllm_fl/distributed/kv_transfer/flagcx_connector.py"

AMD_GRAPH_OPS_MODULE = "vllm_fl.dispatch.backends.vendor.amd.graph_ops"
ASCEND_ROTARY_MODULE = "vllm_fl.dispatch.backends.vendor.ascend.impl.rotary"


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

    def test_forward_cuda_delegates_to_forward_oot(self, ops_modules, monkeypatch):
        """SiluAndMulFL.forward_cuda returns whatever forward_oot returns.

        Independent oracle: forward_oot is replaced by a recorder returning a
        constant tensor unlike any real activation output; forward_cuda must
        return exactly that object.
        """
        activation = ops_modules.reimport("vllm_fl.ops.activation")
        layer = activation.SiluAndMulFL.__new__(activation.SiluAndMulFL)
        sentinel = torch.tensor([42.0])
        calls = []

        def fake_oot(x):
            calls.append(x)
            return sentinel

        monkeypatch.setattr(layer, "forward_oot", fake_oot)
        x = torch.tensor([1.0, 2.0])
        assert layer.forward_cuda(x) is sentinel
        assert calls == [x]


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


class TestRotaryEmbeddingNpuInit:
    """RotaryEmbeddingFL.__init__ NPU branch (rotary lines 37-42)."""

    def test_npu_device_type_initializes_ascend_rotary(self, ops_modules, monkeypatch):
        """An NPU platform triggers the Ascend rotary initializer exactly once.

        Independent oracle: the initializer records each invocation; the
        constructor must call it precisely one time and set
        ``_use_amd_graph_ops`` False for an ascend vendor.
        """
        init_calls = []
        ascend_rotary = _make_module(
            ASCEND_ROTARY_MODULE,
            initialize_rotary_embedding_ascend=lambda: init_calls.append(True),
        )
        monkeypatch.setitem(sys.modules, ASCEND_ROTARY_MODULE, ascend_rotary)

        npu_platform = SimpleNamespace(vendor_name="ascend", device_type="npu")
        graph = _install_ops_vllm_graph(npu_platform)
        for name, module in graph.items():
            monkeypatch.setitem(sys.modules, name, module)

        rotary = ops_modules.reimport("vllm_fl.ops.rotary_embedding")
        layer = rotary.RotaryEmbeddingFL(
            head_size=8,
            rotary_dim=4,
            max_position_embeddings=16,
            base=10000.0,
            is_neox_style=True,
            dtype=torch.float32,
        )
        assert init_calls == [True]
        assert layer._use_amd_graph_ops is False
        # The upstream constructor still ran (the stub records positional
        # args; is_neox_style is the fifth argument).
        assert layer._init_args[4] is True

    def test_non_npu_device_type_skips_ascend_initializer(
        self, ops_modules, monkeypatch
    ):
        """A CUDA platform never touches the Ascend initializer."""
        ascend_rotary = _make_module(
            ASCEND_ROTARY_MODULE,
            initialize_rotary_embedding_ascend=lambda: (_ for _ in ()).throw(
                AssertionError("Ascend initializer reached on a CUDA platform")
            ),
        )
        monkeypatch.setitem(sys.modules, ASCEND_ROTARY_MODULE, ascend_rotary)

        rotary = ops_modules.reimport("vllm_fl.ops.rotary_embedding")
        layer = rotary.RotaryEmbeddingFL(
            head_size=8,
            rotary_dim=4,
            max_position_embeddings=16,
            base=10000.0,
            is_neox_style=True,
            dtype=torch.float32,
        )
        assert layer._use_amd_graph_ops is False


class TestRotaryEmbeddingForwardOot:
    """RotaryEmbeddingFL.forward_oot AMD branch, reshapes and delegates."""

    @staticmethod
    def _make_layer(rotary, *, head_size=8, rotary_dim=4, is_neox_style=True):
        """Build a bare instance with the attributes forward_oot reads."""
        layer = rotary.RotaryEmbeddingFL.__new__(rotary.RotaryEmbeddingFL)
        layer.head_size = head_size
        layer.rotary_dim = rotary_dim
        layer.is_neox_style = is_neox_style
        layer._use_amd_graph_ops = False
        return layer

    @staticmethod
    def _make_inputs(num_tokens=2, head_size=8, rotary_dim=4):
        """Build flat (positions, query, key) inputs plus a cos/sin cache."""
        positions = torch.arange(num_tokens)
        query = torch.ones(num_tokens, 2, head_size)
        key = torch.full((num_tokens, 2, head_size), 2.0)
        cos_sin_cache = (
            torch.arange(num_tokens, dtype=torch.float32)
            .unsqueeze(1)
            .expand(num_tokens, 2 * rotary_dim)
            .contiguous()
        )
        return positions, query, key, cos_sin_cache

    def test_amd_compiling_call_routes_to_graph_ops(self, ops_modules, monkeypatch):
        """An AMD layer in compiled mode uses rotary_embedding_custom_op.

        Independent oracle: the AMD leaf returns ``q + 100`` / ``k + 200``
        and records the interleaving flag; the returned aliases must be
        copied back into the in-place buffers (the core suite's contract),
        so the outputs must equal the leaf values while remaining the very
        tensors handed in as query_rot/key_rot slices.
        """
        amd_calls = []

        def rotary_leaf(query, key, cos, sin, positions, *, rotary_interleaved):
            amd_calls.append(
                (query.shape, key.shape, cos.shape, sin.shape, rotary_interleaved)
            )
            return query + 100.0, key + 200.0

        amd_graph_ops = _make_module(
            AMD_GRAPH_OPS_MODULE, rotary_embedding_custom_op=rotary_leaf
        )
        monkeypatch.setitem(sys.modules, AMD_GRAPH_OPS_MODULE, amd_graph_ops)

        rotary = ops_modules.reimport("vllm_fl.ops.rotary_embedding")
        layer = self._make_layer(rotary)
        layer._use_amd_graph_ops = True
        positions, query, key, cos_sin_cache = self._make_inputs()
        layer._match_cos_sin_cache_dtype = lambda q: cos_sin_cache

        monkeypatch.setattr(torch.compiler, "is_compiling", lambda: True)
        q_out, k_out = layer.forward_oot(positions, query, key)
        # leaf called once with the sliced rot tensors and the right flag.
        assert len(amd_calls) == 1
        q_shape, k_shape, cos_shape, sin_shape, interleaved = amd_calls[0]
        assert q_shape == (2, 2, 4)  # num_tokens, kv, rotary_dim
        assert k_shape == (2, 2, 4)
        assert cos_shape == (2, 4)  # half of cos_sin_cache chunked on dim -1
        assert sin_shape == (2, 4)
        assert interleaved is False  # not is_neox_style
        # outputs equal leaf values (copy_ back into the rot slices)
        expected_q = torch.ones(2, 2, 8)
        expected_q[..., :4] += 100.0
        expected_k = torch.full((2, 2, 8), 2.0)
        expected_k[..., :4] += 200.0
        torch.testing.assert_close(q_out, expected_q)
        torch.testing.assert_close(k_out, expected_k)
        # and the underlying query/key buffers were mutated in place
        torch.testing.assert_close(query[..., :4], torch.ones(2, 2, 4) + 100.0)
        torch.testing.assert_close(key[..., :4], torch.full((2, 2, 4), 2.0) + 200.0)

    def test_rotary_dim_equal_head_size_uses_plain_reshape(
        self, ops_modules, monkeypatch
    ):
        """rotary_dim == head_size skips the pass-through concatenation arm.

        Independent oracle: the dispatched CachedOp writes distinguishable
        constants into the rot slices; with rotary_dim == head_size there is
        no pass slice, so the output must be exactly the reshaped rot
        buffer, and its shape must equal the input shape.
        """
        rotary = ops_modules.reimport("vllm_fl.ops.rotary_embedding")
        layer = self._make_layer(rotary, head_size=4, rotary_dim=4)
        positions, query, key, cos_sin_cache = self._make_inputs(
            head_size=4, rotary_dim=4
        )
        layer._match_cos_sin_cache_dtype = lambda q: cos_sin_cache

        def fake_cached_op(layer, q, k, cos, sin, pos, interleaved, inplace):
            q.fill_(7.0)
            k.fill_(9.0)
            return q, k

        monkeypatch.setattr(rotary, "_rotary_embedding", fake_cached_op)
        q_out, k_out = layer.forward_oot(positions, query, key)
        torch.testing.assert_close(q_out, torch.full((2, 2, 4), 7.0))
        torch.testing.assert_close(k_out, torch.full((2, 2, 4), 9.0))
        assert q_out.shape == query.shape
        assert k_out.shape == key.shape

    def test_key_none_takes_native_path(self, ops_modules, monkeypatch):
        """key=None is routed to the upstream forward_native unchanged.

        Independent oracle: forward_native is replaced by a sentinel
        recorder; the (positions, query, None) triple must arrive verbatim
        and its return value must be passed through.
        """
        rotary = ops_modules.reimport("vllm_fl.ops.rotary_embedding")
        layer = self._make_layer(rotary)
        sentinel_q = torch.tensor([5.0])
        sentinel_k = torch.tensor([6.0])
        calls = []

        def fake_native(pos, q, k):
            calls.append((pos, q, k))
            return sentinel_q, sentinel_k

        monkeypatch.setattr(layer, "forward_native", fake_native, raising=False)
        positions = torch.tensor([0, 1])
        query = torch.ones(2, 8)
        q_out, k_out = layer.forward_oot(positions, query, None)
        assert calls == [(positions, query, None)]
        assert q_out is sentinel_q
        assert k_out is sentinel_k

    def test_forward_cuda_delegates_to_forward_oot(self, ops_modules, monkeypatch):
        """RotaryEmbeddingFL.forward_cuda forwards to forward_oot verbatim.

        Independent oracle: forward_oot is replaced by a sentinel recorder;
        forward_cuda must hand over the exact (positions, query, key)
        triple and return its result unmodified.
        """
        rotary = ops_modules.reimport("vllm_fl.ops.rotary_embedding")
        layer = self._make_layer(rotary)
        sentinel = (torch.tensor([1.0]), torch.tensor([2.0]))
        calls = []

        def fake_oot(pos, q, k):
            calls.append((pos, q, k))
            return sentinel

        monkeypatch.setattr(layer, "forward_oot", fake_oot)
        positions = torch.tensor([0, 1])
        query = torch.ones(2, 8)
        key = torch.zeros(2, 8)
        assert layer.forward_cuda(positions, query, key) is sentinel
        assert calls == [(positions, query, key)]


# ---------------------------------------------------------------------------
# flagcx_connector import-time environment handling (lines 72-84)
# ---------------------------------------------------------------------------

pytest.importorskip("msgspec")
pytest.importorskip("numpy")
pytest.importorskip("zmq")


@pytest.fixture()
def load_connector(monkeypatch):
    """Yield a loader executing the connector source under a unique name.

    Uses the shared ``_flagcx_stubs.FlagcxStubEnv`` boundary (installed via
    ``monkeypatch.setitem`` so nothing leaks), minus the wrapper stub: the
    import-time wrapper import is what the tests vary.  ``sys.path`` is
    snapshotted so the ``FLAGCX_PATH`` append is rolled back afterwards.
    """
    from tests.supplementary_tests.unit.distributed._flagcx_stubs import (
        FlagcxStubEnv,
    )

    subject = "_test_ops_midtail_flagcx_connector"
    monkeypatch.setattr(sys, "path", sys.path.copy())
    monkeypatch.delenv("FLAGCX_PATH", raising=False)

    def load(install_wrapper=True, library=None):
        env = FlagcxStubEnv(monkeypatch, flagcx_library=library)
        env.install()
        if not install_wrapper:
            # Nullify the wrapper stub the env just installed: a None entry
            # makes ``from plugin.interservice.flagcx_wrapper import ...``
            # raise ImportError, which is the code path under test.
            monkeypatch.setitem(sys.modules, "plugin.interservice.flagcx_wrapper", None)
        spec = importlib.util.spec_from_file_location(subject, _CONNECTOR_SOURCE)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module

    yield load
    sys.modules.pop(subject, None)


class TestFlagcxConnectorImportPaths:
    """flagcx_connector lines 72-84: sys.path append and wrapper ImportError."""

    def test_flagcx_path_dir_appended_to_sys_path(
        self, load_connector, monkeypatch, tmp_path
    ):
        """A valid FLAGCX_PATH directory is appended to sys.path.

        Independent oracle: a fresh directory (guaranteed absent from
        sys.path) is used; after the import it must be the last sys.path
        entry.
        """
        flagcx_root = tmp_path / "flagcx-root"
        flagcx_root.mkdir()
        monkeypatch.setenv("FLAGCX_PATH", str(flagcx_root))
        load_connector()
        assert str(flagcx_root) in sys.path
        assert sys.path[-1] == str(flagcx_root)

    def test_flagcx_path_missing_dir_not_appended(self, load_connector, monkeypatch):
        """A nonexistent FLAGCX_PATH directory leaves sys.path untouched."""
        monkeypatch.setenv("FLAGCX_PATH", "/definitely/not/a/real/dir/xyz")
        before = list(sys.path)
        load_connector()
        assert sys.path == before

    def test_wrapper_import_failure_raises_guidance(self, load_connector, monkeypatch):
        """A missing FlagCX wrapper re-raises ImportError with guidance.

        Independent oracle: the error message must name FLAGCX_PATH and the
        expected wrapper module path, and the original ImportError must be
        chained as ``__cause__``.
        """
        monkeypatch.setitem(sys.modules, "plugin.interservice.flagcx_wrapper", None)
        with pytest.raises(
            ImportError, match="Cannot import FlagCX wrapper"
        ) as excinfo:
            load_connector(install_wrapper=False)
        assert "FLAGCX_PATH" in str(excinfo.value)
        assert "plugin/interservice/flagcx_wrapper.py" in str(excinfo.value)
        assert isinstance(excinfo.value.__cause__, ImportError)
