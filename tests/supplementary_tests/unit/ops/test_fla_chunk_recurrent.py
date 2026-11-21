# Copyright (c) 2026 BAAI. All rights reserved.

"""Supplementary tests for vllm_fl.ops.fla chunk / fused_recurrent CustomOps.

``vllm_fl.ops.fla``'s package __init__ imports chunk.py, which needs vllm at
import time.  Like ``test_fla_utils.py``, the stub vllm graph below only
mocks the external vllm boundary; the op classes under test run unmodified.
``forward_native`` is then driven end-to-end with the FLA kernel boundary
replaced by a recorded fake, so the op contracts (validation, default
scale, cu_seqlens handling, beta defaulting, flag forwarding) are asserted
without a GPU.
"""

import os
import sys
import types
from contextlib import nullcontext

import pytest
import torch

# chunk.py picks its implementation at import time: without flag_gems the
# flaggems branch must be off, which USE_FLAGGEMS=0 guarantees.
os.environ.setdefault("USE_FLAGGEMS", "0")


@pytest.fixture(autouse=True)
def _patch_cuda_device_ctx(monkeypatch):
    """Neutralize torch.cuda.device on CPU-only builds.

    input_guard enters ``torch.cuda.device(tensor.device.index)``; a CPU
    tensor has index None and torch raises TypeError inside
    ``_exchange_device``.  ``torch.cuda.device`` is an external boundary
    (a torch side effect the guard must trigger), so it is mocked to a
    nullcontext here, exactly as ``test_fla_utils.py`` does.
    """
    monkeypatch.setattr(torch.cuda, "device", lambda idx: nullcontext())


def _install_vllm_stubs():
    """Create the minimal vllm module graph vllm_fl.ops.fla imports."""
    vllm = types.ModuleType("vllm")
    vllm.__path__ = []
    third_party = types.ModuleType("vllm.third_party")
    third_party.__path__ = []
    fla = types.ModuleType("vllm.third_party.flash_linear_attention")
    fla.__path__ = []
    fla_ops = types.ModuleType("vllm.third_party.flash_linear_attention.ops")
    fla_ops.__path__ = []
    l2norm = types.ModuleType("vllm.third_party.flash_linear_attention.ops.l2norm")
    l2norm.l2norm_fwd = lambda x: x
    fla_chunk = types.ModuleType("vllm.third_party.flash_linear_attention.ops.chunk")
    fla_chunk.chunk_gated_delta_rule_fwd = lambda *a, **k: None
    fla_rec = types.ModuleType(
        "vllm.third_party.flash_linear_attention.ops.fused_recurrent"
    )
    fla_rec.fused_recurrent_gated_delta_rule_fwd = lambda *a, **k: None
    me = types.ModuleType("vllm.model_executor")
    me.__path__ = []
    custom_op = types.ModuleType("vllm.model_executor.custom_op")

    class CustomOp:
        """Stub of vLLM's CustomOp with the decorators the ops apply."""

        @classmethod
        def register(cls, name):
            def decorator(obj):
                return obj

            return decorator

    custom_op.CustomOp = CustomOp
    vllm.third_party = third_party
    vllm.model_executor = me
    third_party.flash_linear_attention = fla
    fla.ops = fla_ops
    fla_ops.l2norm = l2norm
    fla_ops.chunk = fla_chunk
    fla_ops.fused_recurrent = fla_rec
    me.custom_op = custom_op
    for name, mod in {
        "vllm": vllm,
        "vllm.third_party": third_party,
        "vllm.third_party.flash_linear_attention": fla,
        "vllm.third_party.flash_linear_attention.ops": fla_ops,
        "vllm.third_party.flash_linear_attention.ops.l2norm": l2norm,
        "vllm.third_party.flash_linear_attention.ops.chunk": fla_chunk,
        "vllm.third_party.flash_linear_attention.ops.fused_recurrent": fla_rec,
        "vllm.model_executor": me,
        "vllm.model_executor.custom_op": custom_op,
    }.items():
        sys.modules[name] = mod


_STUB_ENTRIES = [
    "vllm",
    "vllm.third_party",
    "vllm.third_party.flash_linear_attention",
    "vllm.third_party.flash_linear_attention.ops",
    "vllm.third_party.flash_linear_attention.ops.l2norm",
    "vllm.third_party.flash_linear_attention.ops.chunk",
    "vllm.third_party.flash_linear_attention.ops.fused_recurrent",
    "vllm.model_executor",
    "vllm.model_executor.custom_op",
]
_SAVED = {name: sys.modules.get(name) for name in _STUB_ENTRIES}

# The op classes' CustomOp base reads the current vLLM config during
# ``__init__``, which the real vllm only provides inside a
# ``set_current_vllm_config`` context. The stub graph's bare CustomOp keeps the
# op contract testable on any host, so the stubs are ALWAYS swapped in here,
# the two modules under test are imported through the real package (so
# chunk.py's relative ``from .utils import input_guard`` resolves), and the
# pre-existing sys.modules entries — real vllm included — are restored
# afterwards. This makes the file order-independent: no real-vllm import
# earlier in the session can change how these imports resolve, and the swap
# leaves no trace for later files.
_install_vllm_stubs()
try:
    import vllm_fl.ops.fla.chunk as chunk_mod  # noqa: E402
    import vllm_fl.ops.fla.fused_recurrent as rec_mod  # noqa: E402
finally:
    for name in _STUB_ENTRIES:
        if _SAVED[name] is None:
            sys.modules.pop(name, None)
        else:
            sys.modules[name] = _SAVED[name]


def _make_qkvgbeta(B, T, H, K, V, dtype=torch.bfloat16, seed=0):
    """Deterministic small gated-delta-rule inputs."""
    g = torch.Generator().manual_seed(seed)
    q = torch.randn(B, T, H, K, generator=g, dtype=torch.float32).to(dtype)
    k = torch.randn(B, T, H, K, generator=g, dtype=torch.float32).to(dtype)
    v = torch.randn(B, T, H, V, generator=g, dtype=torch.float32).to(dtype)
    gating = -torch.rand(B, T, H, generator=g, dtype=torch.float32).to(dtype)
    beta = torch.rand(B, T, H, generator=g, dtype=torch.float32).to(dtype)
    return q, k, v, gating, beta


@pytest.fixture
def chunk_calls(monkeypatch):
    """Record every chunk kernel invocation and return the recorder."""
    calls = []

    def fake_fwd(**kwargs):
        """Boundary double: echoes inputs so contracts are assertable."""
        calls.append(kwargs)
        return (
            kwargs["g"],
            kwargs["q"].clone(),
            kwargs["q"].clone(),
            None,
            None,
            None,
            None,
        )

    monkeypatch.setattr(
        chunk_mod,
        "_chunk_impl",
        types.SimpleNamespace(chunk_gated_delta_rule_fwd=fake_fwd),
    )
    return calls


@pytest.fixture
def rec_calls(monkeypatch):
    """Record every fused_recurrent kernel invocation."""
    calls = []

    def fake_fwd(**kwargs):
        """Boundary double: returns output and final-state placeholders."""
        calls.append(kwargs)
        return kwargs["q"].clone(), kwargs["initial_state"]

    monkeypatch.setattr(
        rec_mod,
        "fused_recurrent_gated_delta_rule_fwd",
        fake_fwd,
    )
    return calls


class TestChunkOpForwardNative:
    """ChunkGatedDeltaRuleOp.forward_native contract with a fake kernel."""

    @pytest.mark.supplementary
    def test_defaults_stored(self):
        """Default construction stores the documented defaults."""
        op = chunk_mod.ChunkGatedDeltaRuleOp()
        assert op.output_final_state is False
        assert op.use_qk_l2norm_in_kernel is False
        assert op.suppress_level == 0

    @pytest.mark.supplementary
    def test_env_suppress_level(self, monkeypatch):
        """GDN_RECOMPUTE_SUPPRESS_LEVEL is read at construction time."""
        monkeypatch.setenv("GDN_RECOMPUTE_SUPPRESS_LEVEL", "3")
        op = chunk_mod.ChunkGatedDeltaRuleOp()
        assert op.suppress_level == 3

    @pytest.mark.supplementary
    def test_forward_native_basic(self, chunk_calls):
        """Defaults flow to the kernel: scale=1/sqrt(K), no cu_seqlens."""
        op = chunk_mod.ChunkGatedDeltaRuleOp()
        q, k, v, g, beta = _make_qkvgbeta(1, 4, 2, 8, 4, seed=1)
        o, final = op.forward_native(q, k, v, g, beta)
        # Independent oracle for the default scale: k.shape[-1] ** -0.5.
        assert chunk_calls[0]["scale"] == pytest.approx(8**-0.5)
        assert chunk_calls[0]["output_final_state"] is False
        assert chunk_calls[0]["cu_seqlens"] is None
        # Boundary double echoes q as the output token tensor.
        torch.testing.assert_close(o, q)
        assert final is None

    @pytest.mark.supplementary
    def test_forward_native_explicit_scale_forwarded(self, chunk_calls):
        """An explicit scale is forwarded verbatim to the kernel."""
        op = chunk_mod.ChunkGatedDeltaRuleOp()
        q, k, v, g, beta = _make_qkvgbeta(2, 3, 2, 8, 4, seed=2)
        op.forward_native(q, k, v, g, beta, scale=0.25)
        assert chunk_calls[0]["scale"] == 0.25

    @pytest.mark.supplementary
    def test_forward_native_float32_rejected(self):
        """float32 inputs are rejected by the documented assertion."""
        op = chunk_mod.ChunkGatedDeltaRuleOp()
        q, k, v, g, beta = _make_qkvgbeta(1, 2, 2, 4, 2, dtype=torch.float32)
        with pytest.raises(AssertionError, match="float32"):
            op.forward_native(q, k, v, g, beta)

    @pytest.mark.supplementary
    def test_forward_native_beta_rank_rejected(self):
        """beta must be rank-3 [B, T, H]."""
        op = chunk_mod.ChunkGatedDeltaRuleOp()
        q, k, v, g, beta = _make_qkvgbeta(1, 2, 2, 4, 2)
        with pytest.raises(AssertionError, match="beta"):
            op.forward_native(q, k, v, g, beta[0])

    @pytest.mark.supplementary
    def test_forward_native_headfirst_warns(self, chunk_calls, recwarn):
        """seq_len < num_heads triggers the format-mismatch warning."""
        op = chunk_mod.ChunkGatedDeltaRuleOp()
        q, k, v, g, beta = _make_qkvgbeta(1, 2, 4, 4, 2)  # T=2 < H=4
        op.forward_native(q, k, v, g, beta)
        assert any("format mismatch" in str(w.message) for w in recwarn.list)

    @pytest.mark.supplementary
    def test_forward_native_cu_seqlens_batch_must_be_one(self):
        """cu_seqlens with batch > 1 is rejected."""
        op = chunk_mod.ChunkGatedDeltaRuleOp()
        q, k, v, g, beta = _make_qkvgbeta(2, 2, 2, 4, 2)
        cu = torch.tensor([0, 2, 4], dtype=torch.long)
        with pytest.raises(ValueError, match="batch size is expected to be 1"):
            op.forward_native(q, k, v, g, beta, cu_seqlens=cu)

    @pytest.mark.supplementary
    def test_forward_native_cu_seqlens_state_count_checked(self):
        """initial_state count must match len(cu_seqlens) - 1."""
        op = chunk_mod.ChunkGatedDeltaRuleOp()
        q, k, v, g, beta = _make_qkvgbeta(1, 4, 2, 4, 2)
        cu = torch.tensor([0, 2, 4], dtype=torch.long)
        bad_state = torch.zeros(3, 2, 4, 2, dtype=q.dtype)  # 3 != 2
        with pytest.raises(ValueError, match="initial states"):
            op.forward_native(q, k, v, g, beta, cu_seqlens=cu, initial_state=bad_state)

    @pytest.mark.supplementary
    def test_forward_native_cu_seqlens_valid_state(self, chunk_calls):
        """A valid cu_seqlens/initial_state pair flows to the kernel."""
        op = chunk_mod.ChunkGatedDeltaRuleOp()
        q, k, v, g, beta = _make_qkvgbeta(1, 4, 2, 4, 2)
        cu = torch.tensor([0, 2, 4], dtype=torch.long)
        state = torch.zeros(2, 2, 4, 2, dtype=q.dtype)
        op.forward_native(q, k, v, g, beta, cu_seqlens=cu, initial_state=state)
        assert chunk_calls[0]["cu_seqlens"] is cu
        assert chunk_calls[0]["initial_state"] is state

    @pytest.mark.supplementary
    def test_forward_native_l2norm_applied_when_flagged(self, monkeypatch):
        """use_qk_l2norm_in_kernel=True normalizes q and k inside the Function."""
        op = chunk_mod.ChunkGatedDeltaRuleOp(use_qk_l2norm_in_kernel=True)
        q, k, v, g, beta = _make_qkvgbeta(1, 3, 2, 4, 2)
        seen = []
        monkeypatch.setattr(
            chunk_mod,
            "l2norm_fwd",
            lambda t: (
                seen.append(t.shape),
                t / (t.float().norm(dim=-1, keepdim=True) + 1e-12).to(t.dtype),
            )[1],
        )
        monkeypatch.setattr(
            chunk_mod,
            "_chunk_impl",
            types.SimpleNamespace(
                chunk_gated_delta_rule_fwd=lambda **kw: (
                    kw["g"],
                    kw["q"],
                    kw["q"],
                    None,
                    None,
                    None,
                    None,
                )
            ),
        )
        op.forward_native(q, k, v, g, beta)
        # Both q (shape [B,T,H,K]) and k (same shape) were normalized.
        assert len(seen) == 2


class TestFusedRecurrentOpForwardNative:
    """FusedRecurrentGatedDeltaRuleOp.forward_native contract."""

    @pytest.mark.supplementary
    def test_defaults_stored(self):
        """Default construction stores the documented defaults."""
        op = rec_mod.FusedRecurrentGatedDeltaRuleOp()
        assert op.inplace_final_state is False
        assert op.use_qk_l2norm_in_kernel is False

    @pytest.mark.supplementary
    def test_forward_native_basic(self, rec_calls):
        """Defaults flow to the kernel: scale=1/sqrt(K), beta forwarded."""
        op = rec_mod.FusedRecurrentGatedDeltaRuleOp()
        q, k, v, g, beta = _make_qkvgbeta(1, 4, 2, 8, 4, seed=3)
        state = torch.zeros(1, 2, 8, 4, dtype=q.dtype)
        o, final = op.forward_native(q, k, v, g, beta, initial_state=state)
        assert rec_calls[0]["scale"] == pytest.approx(8**-0.5)
        assert rec_calls[0]["initial_state"] is state
        assert rec_calls[0]["inplace_final_state"] is False
        # Boundary double echoes q / the initial state back.
        torch.testing.assert_close(o, q)
        assert final is state or torch.equal(final, state)

    @pytest.mark.supplementary
    def test_forward_native_explicit_scale_positive_required(self):
        """An explicit scale must be positive (asserted contract)."""
        op = rec_mod.FusedRecurrentGatedDeltaRuleOp()
        q, k, v, g, beta = _make_qkvgbeta(1, 2, 2, 4, 2)
        with pytest.raises(AssertionError, match="positive"):
            op.forward_native(q, k, v, g, beta, scale=-1.0)

    @pytest.mark.supplementary
    def test_forward_native_cu_seqlens_batch_must_be_one(self):
        """cu_seqlens with batch > 1 is rejected."""
        op = rec_mod.FusedRecurrentGatedDeltaRuleOp()
        q, k, v, g, beta = _make_qkvgbeta(3, 2, 2, 4, 2)
        cu = torch.tensor([0, 2, 4], dtype=torch.long)
        with pytest.raises(ValueError, match="batch size is expected to be 1"):
            op.forward_native(q, k, v, g, beta, cu_seqlens=cu)

    @pytest.mark.supplementary
    def test_forward_native_beta_none_defaults_to_ones(self, rec_calls):
        """beta=None is replaced by all-ones of q's [B, T, H] shape."""
        op = rec_mod.FusedRecurrentGatedDeltaRuleOp()
        q, k, v, g, beta = _make_qkvgbeta(2, 3, 5, 4, 2, seed=4)
        op.forward_native(q, k, v, g, None)
        sent = rec_calls[0]["beta"]
        assert sent.shape == (2, 3, 5)
        assert torch.all(sent == 1)

    @pytest.mark.supplementary
    def test_forward_native_g_and_beta_made_contiguous(self, rec_calls):
        """Non-contiguous g/beta tensors are compacted before the kernel."""
        op = rec_mod.FusedRecurrentGatedDeltaRuleOp()
        q, k, v, g, beta = _make_qkvgbeta(1, 4, 2, 4, 2, seed=5)
        # Transposed views of the same values: same data, non-contiguous layout.
        g_t = g.transpose(1, 2).contiguous().transpose(1, 2)
        beta_t = beta.transpose(1, 2).contiguous().transpose(1, 2)
        assert not g_t.is_contiguous()
        assert not beta_t.is_contiguous()
        op.forward_native(q, k, v, g_t, beta_t)
        assert rec_calls[0]["g"].is_contiguous()
        assert rec_calls[0]["beta"].is_contiguous()

    @pytest.mark.supplementary
    def test_forward_native_flags_forwarded(self, rec_calls):
        """inplace_final_state and l2norm flags are forwarded to the kernel."""
        op = rec_mod.FusedRecurrentGatedDeltaRuleOp(
            inplace_final_state=True, use_qk_l2norm_in_kernel=True
        )
        q, k, v, g, beta = _make_qkvgbeta(1, 2, 2, 4, 2)
        op.forward_native(q, k, v, g, beta)
        assert rec_calls[0]["inplace_final_state"] is True
        assert rec_calls[0]["use_qk_l2norm_in_kernel"] is True

    @pytest.mark.supplementary
    def test_forward_native_indices_and_tokens_forwarded(self, rec_calls):
        """ssm_state_indices / num_accepted_tokens pass through unchanged."""
        op = rec_mod.FusedRecurrentGatedDeltaRuleOp()
        q, k, v, g, beta = _make_qkvgbeta(1, 2, 2, 4, 2)
        ssm = torch.tensor([0, 1])
        nat = torch.tensor([1, 1])
        op.forward_native(
            q, k, v, g, beta, ssm_state_indices=ssm, num_accepted_tokens=nat
        )
        assert rec_calls[0]["ssm_state_indices"] is ssm
        assert rec_calls[0]["num_accepted_tokens"] is nat
