# Copyright (c) 2026 BAAI. All rights reserved.

"""NPU-construction and forward-oot tests for ``vllm_fl.ops.rotary_embedding``.

Coverage gaps in ``vllm_fl/ops/rotary_embedding.py`` (10 statements):

* the NPU init branch of ``RotaryEmbeddingFL.__init__`` — on an NPU device
  type the constructor imports and runs
  ``vllm_fl.dispatch.backends.vendor.ascend.impl.rotary.initialize_rotary_embedding_ascend``;
* the AMD graph-ops branch of ``RotaryEmbeddingFL.forward_oot`` (the
  ``rotary_embedding_custom_op`` call plus the ``copy_`` aliasing), the
  ``rotary_dim == head_size`` reshape arm and the ``key is None``
  native-path arm.

The ``forward_cuda`` delegate is the same one-line delegation kept as a
single pattern exemplar in ``test_activation_compile_routing.py``; it is not
duplicated here.  The module under test imports the external ``vllm``
package at module level, which is absent on CPU-only test machines, so a
minimal stub ``vllm`` graph is installed first; the plugin sources
themselves run verbatim, so coverage attributes every executed statement to
the real module files.
"""

import importlib
import sys
import types
from types import SimpleNamespace

import pytest
import torch

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
