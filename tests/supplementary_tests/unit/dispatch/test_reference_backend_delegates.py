# Copyright (c) 2026 BAAI. All rights reserved.

"""Supplementary tests for the remaining ``reference.py`` backend branches.

``test_reference_backend.py`` and the core ``test_builtin_ops.py`` (neither
of which may be modified) already cover the numeric contracts of
``topk_softmax``, ``moe_align_block_size``, ``moe_sum`` and
``dynamic_per_token_quant_int8``, plus the cached-True availability probe
and the silu/gelu/rms_norm delegates.  Still uncovered after those run:

* ``is_available``'s ImportError branch (the torch probe failing) and the
  second, uncached call through the memoized class attribute;
* ``rotary_embedding``'s delegate body (argument pass-through to
  ``impl.rotary.rotary_embedding_torch``, verified against an independent
  complex-exponent oracle);
* ``attention_backend``'s three enum branches (FLASHMLA_SPARSE / FLASHMLA /
  FLASH_ATTN) through a stubbed
  ``vllm.v1.attention.backends.registry.AttentionBackendEnum``;
* ``invoke_fused_moe_triton_kernel``'s pass-through to vLLM's kernel (all 18
  positional arguments plus the two keywords captured through a stubbed
  ``vllm.model_executor.layers.fused_moe.fused_moe``).

The lazy imports inside these methods resolve through ``sys.modules`` at call
time, so per-test module stubs installed via ``monkeypatch.setitem`` steer
them on both platforms (real vLLM present or not) without touching any
existing test.
"""

import math
import sys
from types import ModuleType, SimpleNamespace

import pytest
import torch

from vllm_fl.dispatch.backends.reference import ReferenceBackend

pytestmark = pytest.mark.supplementary


@pytest.fixture
def backend():
    """A fresh ReferenceBackend instance with a cleared availability cache."""
    saved = ReferenceBackend._available
    ReferenceBackend._available = None
    try:
        yield ReferenceBackend()
    finally:
        ReferenceBackend._available = saved


# ---------------------------------------------------------------------------
# is_available: the ImportError branch
# ---------------------------------------------------------------------------


class TestIsAvailableImportError:
    """The torch probe failing caches False on the class."""

    def test_import_error_caches_false(self, backend, monkeypatch):
        """A failing ``import torch`` inside the probe caches False."""

        def broken_import(name, *args, **kwargs):
            if name == "torch":
                raise ImportError("torch vanished")
            return real_import(name, *args, **kwargs)

        real_import = (
            __builtins__["__import__"]
            if isinstance(__builtins__, dict)
            else __builtins__.__import__
        )  # noqa: E501
        monkeypatch.setattr("builtins.__import__", broken_import)
        assert backend.is_available() is False
        assert ReferenceBackend._available is False

    def test_cached_result_skips_probe(self, backend, monkeypatch):
        """Once cached, the probe body (and the import) never re-runs."""
        ReferenceBackend._available = False
        # Only the probe's own ``import torch`` would prove the probe re-ran;
        # every other import is delegated to the real builtin because
        # pytest's own machinery (linecache/tokenize, plugin teardown) keeps
        # importing modules while this test runs and would otherwise crash
        # the whole session.
        real_import = (
            __builtins__["__import__"]
            if isinstance(__builtins__, dict)
            else __builtins__.__import__
        )

        def torch_only_import(name, *args, **kwargs):
            if name == "torch":
                raise AssertionError("probe re-ran")
            return real_import(name, *args, **kwargs)

        monkeypatch.setattr("builtins.__import__", torch_only_import)
        assert backend.is_available() is False


# ---------------------------------------------------------------------------
# rotary_embedding: delegate body and numeric contract
# ---------------------------------------------------------------------------


class TestRotaryEmbeddingDelegate:
    """``rotary_embedding`` forwards to the torch impl; independent oracle."""

    @staticmethod
    def _make_inputs(head_dim=8, seq_len=5, dtype=torch.float32):
        """query/key [seq, heads, dim] plus cos/sin caches and positions."""
        torch.manual_seed(1234)
        query = torch.randn(seq_len, 2, head_dim, dtype=dtype)
        key = torch.randn(seq_len, 2, head_dim, dtype=dtype)
        # Half-width caches: rotary_dim = head_dim // 2.
        cos = torch.randn(seq_len, head_dim // 2, dtype=dtype)
        sin = torch.randn(seq_len, head_dim // 2, dtype=dtype)
        positions = torch.arange(seq_len, dtype=torch.int64)
        return query, key, cos, sin, positions

    def test_delegate_returns_impl_result(self, backend):
        """The backend method returns exactly the impl's tuple (delegate body)."""
        from vllm_fl.dispatch.backends.reference.impl.rotary import (
            rotary_embedding_torch,
        )

        query, key, cos, sin, positions = self._make_inputs()
        obj = SimpleNamespace(rotary_interleaved=False)
        expected_q, expected_k = rotary_embedding_torch(
            obj, query, key, cos, sin, positions
        )
        out_q, out_k = backend.rotary_embedding(obj, query, key, cos, sin, positions)
        torch.testing.assert_close(out_q, expected_q)
        torch.testing.assert_close(out_k, expected_k)

    def test_neox_oracle_matches_complex_rotation(self, backend):
        """NeoX-style rotation equals complex multiplication by e^{i*theta}.

        Independent oracle: pair up (x_i, x_{i+d/2}) into a complex number
        and multiply by exp(i * theta_i); the real/imaginary parts are the
        rotated pair. This is a different op mix than rotate_half.
        """
        head_dim, seq_len, heads = 8, 4, 2
        torch.manual_seed(99)
        query = torch.randn(seq_len, heads, head_dim)
        key = torch.randn(seq_len, heads, head_dim)
        # Full-width caches with matching halves: theta_i = atan2(sin, cos).
        angles = torch.rand(seq_len, head_dim // 2) * 2 * math.pi
        cos = torch.cos(angles)
        sin = torch.sin(angles)

        class Obj:
            rotary_interleaved = False

        out_q, _ = backend.rotary_embedding(
            Obj(), query, key, cos, sin, torch.arange(seq_len)
        )

        for pos in range(seq_len):
            for h in range(heads):
                for i in range(head_dim // 2):
                    c = complex(
                        query[pos, h, i].item(), query[pos, h, i + head_dim // 2].item()
                    )
                    rot = c * complex(cos[pos, i].item(), sin[pos, i].item())
                    torch.testing.assert_close(
                        out_q[pos, h, i].item(), rot.real, rtol=1e-5, atol=1e-6
                    )
                    torch.testing.assert_close(
                        out_q[pos, h, i + head_dim // 2].item(),
                        rot.imag,
                        rtol=1e-5,
                        atol=1e-6,
                    )

    def test_interleaved_keyword_forwarded(self, backend):
        """``rotary_interleaved``/``inplace`` reach the impl unchanged (kwarg
        pass-through; the interleaved layout differs from NeoX so the branch
        is observable)."""
        head_dim, seq_len = 8, 3
        torch.manual_seed(7)
        query = torch.randn(seq_len, 1, head_dim)
        key = torch.randn(seq_len, 1, head_dim)
        angles = torch.rand(seq_len, head_dim // 2) * 2 * math.pi
        cos, sin = torch.cos(angles), torch.sin(angles)
        positions = torch.arange(seq_len)

        class Obj:
            rotary_interleaved = False

        neox_q, _ = backend.rotary_embedding(Obj(), query, key, cos, sin, positions)
        inter_q, _ = backend.rotary_embedding(
            Obj(), query, key, cos, sin, positions, rotary_interleaved=True
        )
        # The two layouts rotate the same pairs, so the element SETS agree
        # position-wise while the pair order differs.
        assert not torch.allclose(neox_q, inter_q)
        # Interleaved: pair (2i, 2i+1). Oracle via a direct complex product.
        for pos in range(seq_len):
            i = 0
            c = complex(query[pos, 0, 2 * i].item(), query[pos, 0, 2 * i + 1].item())
            rot = c * complex(cos[pos, i].item(), sin[pos, i].item())
            torch.testing.assert_close(
                inter_q[pos, 0, 2 * i].item(), rot.real, rtol=1e-5, atol=1e-6
            )
            torch.testing.assert_close(
                inter_q[pos, 0, 2 * i + 1].item(), rot.imag, rtol=1e-5, atol=1e-6
            )


# ---------------------------------------------------------------------------
# attention_backend: the three enum branches
# ---------------------------------------------------------------------------


class TestAttentionBackendEnum:
    """``attention_backend`` maps (use_mla, use_sparse) onto the enum paths."""

    @staticmethod
    def _install_registry_stub(monkeypatch):
        """A minimal AttentionBackendEnum double with the three members."""
        vllm = ModuleType("vllm")
        vllm.__path__ = []
        backends = ModuleType("vllm.v1.attention.backends")
        backends.__path__ = []
        registry = ModuleType("vllm.v1.attention.backends.registry")

        class _Entry:
            def __init__(self, path):
                self.path = path

            def get_path(self):
                return self.path

        class AttentionBackendEnum:
            FLASH_ATTN = _Entry("flash-attn-path")
            FLASHMLA = _Entry("flashmla-path")
            FLASHMLA_SPARSE = _Entry("flashmla-sparse-path")

        registry.AttentionBackendEnum = AttentionBackendEnum
        vllm.v1 = ModuleType("vllm.v1")
        vllm.v1.__path__ = []
        vllm.v1.attention = backends
        backends.registry = registry
        monkeypatch.setitem(sys.modules, "vllm", vllm)
        monkeypatch.setitem(sys.modules, "vllm.v1", vllm.v1)
        monkeypatch.setitem(sys.modules, "vllm.v1.attention", backends)
        monkeypatch.setitem(sys.modules, "vllm.v1.attention.backends", backends)
        monkeypatch.setitem(
            sys.modules, "vllm.v1.attention.backends.registry", registry
        )
        return registry

    @pytest.mark.parametrize(
        ("use_mla", "use_sparse", "expected"),
        [
            (True, True, "flashmla-sparse-path"),
            (True, False, "flashmla-path"),
            (False, False, "flash-attn-path"),
        ],
    )
    def test_enum_branches(self, backend, monkeypatch, use_mla, use_sparse, expected):
        """The (mla, sparse) flag pair selects the matching enum member."""
        self._install_registry_stub(monkeypatch)
        assert (
            backend.attention_backend(use_mla=use_mla, use_sparse=use_sparse)
            == expected
        )

    def test_stub_get_path_was_used(self, backend, monkeypatch):
        """The path comes from the enum member's get_path(), not a literal."""
        registry = self._install_registry_stub(monkeypatch)
        result = backend.attention_backend(use_mla=True, use_sparse=True)
        assert result == registry.AttentionBackendEnum.FLASHMLA_SPARSE.get_path()

    def test_defaults_are_false(self, backend, monkeypatch):
        """Without flags the plain FLASH_ATTN path is returned."""
        self._install_registry_stub(monkeypatch)
        assert backend.attention_backend() == "flash-attn-path"


# ---------------------------------------------------------------------------
# invoke_fused_moe_triton_kernel: the vLLM pass-through
# ---------------------------------------------------------------------------


class TestInvokeFusedMoeTritonKernel:
    """The delegate forwards every argument to vLLM's kernel module."""

    @staticmethod
    def _install_fused_moe_stub(monkeypatch):
        """Stub vllm.model_executor.layers.fused_moe.fused_moe with a capture."""
        captured = {}

        def fake_invoke(*args, **kwargs):
            captured["args"] = args
            captured["kwargs"] = kwargs
            return "kernel-result"

        parts = [
            "vllm",
            "model_executor",
            "layers",
            "fused_moe",
            "fused_moe",
        ]
        modules = {}
        for length in range(1, len(parts) + 1):
            name = ".".join(parts[:length])
            module = ModuleType(name)
            module.__path__ = []
            modules[name] = module
        for length in range(1, len(parts)):
            setattr(
                modules[".".join(parts[:length])],
                parts[length],
                modules[".".join(parts[: length + 1])],
            )
        modules[
            "vllm.model_executor.layers.fused_moe.fused_moe"
        ].invoke_fused_moe_triton_kernel = fake_invoke
        for name, module in modules.items():
            monkeypatch.setitem(sys.modules, name, module)
        return captured

    def test_forwards_all_positional_and_keyword_args(self, backend, monkeypatch):
        """All 18 positionals plus block_shape/B_bias reach the kernel."""
        captured = self._install_fused_moe_stub(monkeypatch)
        backend.invoke_fused_moe_triton_kernel(
            "A",
            "B",
            "C",
            "A_scale",
            "B_scale",
            "topk_weights",
            "sorted_token_ids",
            "expert_ids",
            "num_tokens_post_padded",
            True,  # mul_routed_weight
            8,  # top_k
            "config",
            "compute_type",
            True,  # use_fp8_w8a8
            False,  # use_int8_w8a8
            False,  # use_int8_w8a16
            False,  # use_int4_w4a16
            True,  # per_channel_quant
            block_shape=[128, 128],
            B_bias="bias-tensor",
        )
        args = captured["args"]
        assert args == (
            "A",
            "B",
            "C",
            "A_scale",
            "B_scale",
            "topk_weights",
            "sorted_token_ids",
            "expert_ids",
            "num_tokens_post_padded",
            True,
            8,
            "config",
            "compute_type",
            True,
            False,
            False,
            False,
            True,
        )
        assert captured["kwargs"] == {
            "block_shape": [128, 128],
            "B_bias": "bias-tensor",
        }

    def test_default_keywords_are_none(self, backend, monkeypatch):
        """Omitting block_shape/B_bias still forwards them as None."""
        captured = self._install_fused_moe_stub(monkeypatch)
        backend.invoke_fused_moe_triton_kernel(*(f"arg{i}" for i in range(18)))
        assert captured["args"] == tuple(f"arg{i}" for i in range(18))
        assert captured["kwargs"] == {"block_shape": None, "B_bias": None}
