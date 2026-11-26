# Copyright (c) 2026 BAAI. All rights reserved.

"""CPU tests for the DeepSeek-V4 Ascend attention method bodies.

``test_deepseek_v4_policy.py`` loads ``vllm_fl/models/deepseek_v4.py`` under
stubbed ``vllm.*`` modules but stops at the constructor: the bodies of
``forward``, ``_run_parallel_input_projections``, ``_fused_qnorm_rope_kv_insert``,
``_compress_kv_bf16``, ``_prepare_and_attn``, ``_o_proj`` and ``forward_mqa``
are the statements still missing from the combined coverage report. This
module reuses the same stub-graph loader and then invokes those methods as
unbound functions on ``SimpleNamespace`` doubles — the technique of the
torch_npu-gated ``tests/unit_tests/models/test_deepseek_v4.py`` — with
hand-derived numeric oracles so the real production lines run on CPU:

* ``forward`` — the fused qr/kv split, per-half RMSNorm and the hand-off to
  ``_prepare_and_attn_fn``/``_o_proj`` (padded buffer shape, local-head
  slice, llama_4_scaling dropped);
* ``_run_parallel_input_projections`` — the FP32 compressor gate GEMM;
* ``_fused_qnorm_rope_kv_insert`` — weightless q RMSNorm, rope passthrough
  and SWA cache row storage through the slot mapping;
* ``_compress_kv_bf16`` — packed state rows (both overlap layouts), the
  softmax-weighted compression reference for ratios 4 and 128, boundary
  gating, invalid-block masking and the cos/sin rope tail rotation;
* ``_prepare_and_attn`` — wq_b expansion, the C4A topk candidate fill and
  the compressor hand-off;
* ``_o_proj`` — inverse rope then the two-stage output projection;
* ``forward_mqa`` — SWA-only and compressed attention against an
  independent softmax oracle, window cropping, invalid-block masking,
  per-request block tables and the attn_sink stabilization.

The final tests run only where a real vLLM install exists (the WSL run);
they construct the layer through the real upstream constructor and check a
full ``forward`` against a from-scratch reference.

Why this file exists alongside ``tests/unit_tests/models/test_deepseek_v4.py``:
the unit-tree version is gated by ``pytest.importorskip("torch_npu")``, so on
any CI host without an Ascend runtime it collects zero tests. This file needs
no torch_npu (the stub-graph loader stands in for vllm/flaggems_vllm), so its
method-body tests are the only coverage of the real ``forward``/``forward_mqa``
lines on NPU-less machines; the two ported real-vLLM tests at the end (skipped
where vLLM is absent) extend that to the real upstream constructor.
"""

import contextlib
import importlib.util
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest
import torch

_SOURCE = Path(__file__).resolve().parents[3] / "vllm_fl/models/deepseek_v4.py"
_MODULE_NAME = "supp_deepseek_v4_more_under_test"



class _UpstreamBase:
    """Recording stand-in for upstream DeepseekV4Attention."""

    def __init__(self, *args, **kwargs):
        self.init_args = args
        self.init_kwargs = kwargs


class _HCHeadOp:
    """Stand-in for vllm.model_executor.layers.mhc.HCHeadOp."""


class _DeepseekV4SparseMLAMetadataBuilder:
    """Stand-in for the upstream sparse-MLA metadata builder."""


class _XpuModelBase:
    """Stand-in for the upstream XPU causal-LM base class."""


def _install_stubs(monkeypatch):
    """Install the fake vllm/flaggems_vllm tree (same shape as the policy tests)."""
    modules = {}

    def leaf(name, **attrs):
        parts = name.split(".")
        for length in range(1, len(parts) + 1):
            partial = ".".join(parts[:length])
            if partial not in modules:
                module = ModuleType(partial)
                module.__path__ = []
                modules[partial] = module
        for length in range(1, len(parts)):
            setattr(
                modules[".".join(parts[:length])],
                parts[length],
                modules[".".join(parts[: length + 1])],
            )
        for key, value in attrs.items():
            setattr(modules[name], key, value)
        return modules[name]

    leaf("vllm.forward_context", get_forward_context=lambda: SimpleNamespace())
    leaf("vllm.models.deepseek_v4.attention", DeepseekV4Attention=_UpstreamBase)
    leaf(
        "vllm.models.deepseek_v4.sparse_mla",
        DeepseekV4FlashMLAMetadata=type("DeepseekV4FlashMLAMetadata", (), {}),
        DeepseekV4SparseMLABackend=type("DeepseekV4SparseMLABackend", (), {}),
        DeepseekV4SparseMLAMetadataBuilder=_DeepseekV4SparseMLAMetadataBuilder,
    )
    leaf("vllm.platforms", current_platform=SimpleNamespace(device_type="npu"))
    leaf(
        "vllm.v1.attention.backends.mla.sparse_swa",
        DeepseekSparseSWABackend=type("DeepseekSparseSWABackend", (), {}),
        DeepseekSparseSWAMetadata=type("DeepseekSparseSWAMetadata", (), {}),
        DeepseekSparseSWAMetadataBuilder=type(
            "DeepseekSparseSWAMetadataBuilder", (), {}
        ),
    )
    leaf("vllm.model_executor.layers.mhc", HCHeadOp=_HCHeadOp)
    leaf("flaggems_vllm", hc_head_fused_kernel=lambda *args, **kwargs: None)
    xpu_model = SimpleNamespace(
        DeepseekV4ForCausalLM=_XpuModelBase,
        DeepseekV4XPUAttention=object,
    )
    leaf("vllm.models.deepseek_v4.xpu", model=xpu_model)

    for name, module in modules.items():
        monkeypatch.setitem(sys.modules, name, module)
    return SimpleNamespace(xpu_model=xpu_model)


@pytest.fixture
def deepseek_mod(monkeypatch):
    """Load a fresh copy of the real deepseek_v4.py under stub modules."""
    _install_stubs(monkeypatch)
    monkeypatch.setattr(torch, "npu", SimpleNamespace(Event=object), raising=False)
    monkeypatch.delattr(_HCHeadOp, "forward_native", raising=False)
    monkeypatch.delattr(_HCHeadOp, "_vllm_fl_ascend", raising=False)

    spec = importlib.util.spec_from_file_location(_MODULE_NAME, _SOURCE)
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, _MODULE_NAME, module)
    spec.loader.exec_module(module)
    return module


def _set_attn_metadata(module, attn_metadata):
    """Point the loaded module's get_forward_context at fixed metadata."""
    module.get_forward_context = lambda: SimpleNamespace(attn_metadata=attn_metadata)


class _ScaleRotary:
    """Deterministic rope stand-in scaling q and kv by fixed factors."""

    def __init__(self, q_factor=1.0, kv_factor=1.0):
        self.q_factor = q_factor
        self.kv_factor = kv_factor
        self.calls = []

    def forward_native(self, positions, q, kv, inverse=False):
        assert inverse is False
        self.calls.append((positions, q, kv))
        return q * self.q_factor, None if kv is None else kv * self.kv_factor


# ---------------------------------------------------------------------------
# DeepseekV4FLAttention.forward
# ---------------------------------------------------------------------------


def test_forward_rmsnorms_split_qr_and_kv(deepseek_mod):
    """forward splits the fused projection and RMSNorms both halves.

    Independent oracle: with qr_kv row [3, 4, 6, 8], q_lora_rank 2 and
    head_dim 2, qr = [3, 4] and kv = [6, 8]; each half is scaled by
    rsqrt(mean(x^2) + eps) and then by its norm weight, so
    qr_out = [3, 4] * rsqrt(12.5 + eps) * [1, 0.5] and
    kv_out = [6, 8] * rsqrt(50 + eps) * [2, 1]. llama_4_scaling is accepted
    but dropped, and the compressor/indexer outputs pass through verbatim.
    """
    eps = 1e-6
    qr_kv = torch.tensor([[3.0, 4.0, 6.0, 8.0]])
    captured = {}

    def fake_projections(hidden_states):
        captured["hidden_states"] = hidden_states
        return qr_kv, "kv-score", None, None

    def fake_prepare(*args):
        captured["prepare_args"] = args

    attention = SimpleNamespace(
        padded_heads=2,
        head_dim=2,
        q_lora_rank=2,
        eps=eps,
        q_norm=SimpleNamespace(weight=torch.tensor([1.0, 0.5])),
        kv_norm=SimpleNamespace(weight=torch.tensor([2.0, 1.0])),
        _run_parallel_input_projections=fake_projections,
        _prepare_and_attn_fn=fake_prepare,
        _o_proj=lambda o, positions: o,
        n_local_heads=2,
    )
    hidden_states = torch.randn(1, 7)
    positions = torch.tensor([0])

    out = deepseek_mod.DeepseekV4FLAttention.forward(
        attention, positions, hidden_states, llama_4_scaling=torch.ones(1)
    )

    qr_expected = (
        torch.tensor([[3.0, 4.0]])
        * torch.rsqrt(torch.tensor(12.5 + eps))
        * torch.tensor([1.0, 0.5])
    )
    kv_expected = (
        torch.tensor([[6.0, 8.0]])
        * torch.rsqrt(torch.tensor(50.0 + eps))
        * torch.tensor([2.0, 1.0])
    )
    args = captured["prepare_args"]
    torch.testing.assert_close(args[1], qr_expected)
    torch.testing.assert_close(args[2], kv_expected)
    assert args[3] == "kv-score"
    assert args[4] is None and args[5] is None
    assert captured["hidden_states"] is hidden_states
    assert args[0] is hidden_states
    assert args[6] is positions
    assert out.shape == (1, 2, 2)


def test_forward_allocates_o_padded_and_slices_for_o_proj(deepseek_mod):
    """o_padded is [tokens, padded_heads, head_dim]; _o_proj sees local heads.

    Independent oracle: with 2 tokens, padded_heads 5 and n_local_heads 3,
    the buffer handed to _prepare_and_attn_fn is (2, 5, 4) while _o_proj
    receives the (2, 3, 4) local-head slice; the fill performed inside the
    prepare stub must be observable in that slice, and forward must return
    whatever _o_proj returns.
    """
    seen = {}

    def fake_prepare(hidden_states, qr, kv, kv_score, iks, iw, positions, o_padded):
        seen["o_padded"] = o_padded
        o_padded.fill_(2.0)

    def fake_o_proj(o, positions):
        seen["o_proj_shape"] = tuple(o.shape)
        seen["o_proj_filled"] = bool((o == 2.0).all())
        return "o-proj-result"

    attention = SimpleNamespace(
        padded_heads=5,
        head_dim=4,
        q_lora_rank=3,
        eps=1e-6,
        q_norm=SimpleNamespace(weight=torch.ones(3)),
        kv_norm=SimpleNamespace(weight=torch.ones(4)),
        _run_parallel_input_projections=lambda hs: (
            torch.zeros(2, 7),
            torch.zeros(2, 4),
            None,
            None,
        ),
        _prepare_and_attn_fn=fake_prepare,
        _o_proj=fake_o_proj,
        n_local_heads=3,
    )
    hidden_states = torch.randn(2, 6)

    out = deepseek_mod.DeepseekV4FLAttention.forward(
        attention, torch.zeros(2, dtype=torch.long), hidden_states
    )

    assert seen["o_padded"].shape == (2, 5, 4)
    assert seen["o_padded"].dtype == hidden_states.dtype
    assert seen["o_padded"].device == hidden_states.device
    assert seen["o_proj_shape"] == (2, 3, 4)
    assert seen["o_proj_filled"] is True
    assert out == "o-proj-result"


# ---------------------------------------------------------------------------
# DeepseekV4FLAttention._run_parallel_input_projections
# ---------------------------------------------------------------------------


def test_input_projections_without_compressor_returns_none_scores(deepseek_mod):
    """Without a compressor layer, kv_score and indexer outputs are None.

    The fused WQA/WKV GEMM output must be returned verbatim.
    """
    gemm_result = torch.randn(3, 6)
    seen = {}

    def fake_gemm(hidden_states):
        seen["hidden_states"] = hidden_states
        return gemm_result

    attention = SimpleNamespace(_fused_wqa_wkv_gemm=fake_gemm, compressor=None)
    hidden_states = torch.randn(3, 5)

    qr_kv, kv_score, indexer_kv_score, indexer_weights = (
        deepseek_mod.DeepseekV4FLAttention._run_parallel_input_projections(
            attention, hidden_states
        )
    )

    assert qr_kv is gemm_result
    assert kv_score is None
    assert indexer_kv_score is None
    assert indexer_weights is None
    assert seen["hidden_states"] is hidden_states


def test_input_projections_compressor_gemm_runs_in_fp32(deepseek_mod):
    """With a compressor, kv_score is the fp32 product with the gate weight.

    Independent oracle: the einsum formulation of the same product. The
    bf16 operands make the fp32 cast observable — computing the same
    product in bf16 would not match the einsum-in-fp32 reference.
    """
    weight = torch.randn(4, 5, dtype=torch.bfloat16) * 0.3
    hidden_states = torch.randn(3, 5, dtype=torch.bfloat16)
    attention = SimpleNamespace(
        _fused_wqa_wkv_gemm=lambda hs: torch.zeros(3, 2),
        compressor=SimpleNamespace(fused_wkv_wgate=SimpleNamespace(weight=weight)),
    )

    qr_kv, kv_score, _, _ = (
        deepseek_mod.DeepseekV4FLAttention._run_parallel_input_projections(
            attention, hidden_states
        )
    )

    expected = torch.einsum("td,od->to", hidden_states.float(), weight.float())
    torch.testing.assert_close(kv_score, expected)
    assert kv_score.dtype == torch.float32
    assert qr_kv.shape == (3, 2)


# ---------------------------------------------------------------------------
# DeepseekV4FLAttention._fused_qnorm_rope_kv_insert
# ---------------------------------------------------------------------------


def test_fused_qnorm_rope_kv_insert_passthrough_without_dict_metadata(deepseek_mod):
    """Non-dict attn metadata leaves q untouched (no norm, no cache write)."""
    attention = SimpleNamespace(
        swa_cache_layer=SimpleNamespace(
            prefix="layer.swa_cache", kv_cache=torch.zeros(1, 1, 1)
        )
    )
    q = torch.randn(2, 3, 4)

    returned = deepseek_mod.DeepseekV4FLAttention._fused_qnorm_rope_kv_insert(
        attention, q, torch.randn(2, 4), torch.zeros(2, dtype=torch.long), object()
    )

    assert returned is q


def test_fused_qnorm_rope_kv_insert_norms_q_and_stores_rotated_kv(deepseek_mod):
    """q is weightless-RMSNormed, roped, and rotated kv lands in the SWA cache.

    Independent oracle: with q row [2, 2, 2] the normalization factor is
    rsqrt(mean(4) + eps); the fake rope scales q by 3 and kv by 5, so the
    returned q is the normalized value times 3 while cache rows hold kv
    times 5 — exactly the slots marked valid, none of the masked ones.
    """
    eps = 1e-6
    head_dim = 3
    block_size = 2
    kv_cache = torch.zeros(2, block_size, head_dim)
    rotary = _ScaleRotary(q_factor=3.0, kv_factor=5.0)
    q = torch.full((3, 1, head_dim), 2.0)
    kv = torch.stack(
        [torch.arange(head_dim, dtype=torch.float32) + float(i) for i in range(3)]
    )
    positions = torch.tensor([0, 1, 2])
    metadata = SimpleNamespace(
        slot_mapping=torch.tensor([0, -1, 2]), block_size=block_size
    )
    attention = SimpleNamespace(
        swa_cache_layer=SimpleNamespace(prefix="layer.swa_cache", kv_cache=kv_cache),
        eps=eps,
        rotary_emb=rotary,
    )

    returned = deepseek_mod.DeepseekV4FLAttention._fused_qnorm_rope_kv_insert(
        attention, q, kv, positions, {"layer.swa_cache": metadata}
    )

    norm = torch.rsqrt(torch.tensor(4.0 + eps))
    torch.testing.assert_close(returned, torch.full((3, 1, head_dim), 2.0) * norm * 3)
    # Slot 0 -> page 0 offset 0; slot 2 -> page 1 offset 0; slot -1 dropped.
    torch.testing.assert_close(kv_cache[0, 0], (kv[0] * 5))
    torch.testing.assert_close(kv_cache[1, 0], (kv[2] * 5))
    assert torch.count_nonzero(kv_cache[0, 1]) == 0
    assert torch.count_nonzero(kv_cache[1, 1]) == 0
    assert rotary.calls[0][0] is positions
    # rope receives the *normalized* q.
    torch.testing.assert_close(rotary.calls[0][1], q * norm)


def test_fused_qnorm_rope_kv_insert_all_slots_invalid_writes_nothing(deepseek_mod):
    """When every slot is negative the cache write is skipped entirely."""
    kv_cache = torch.zeros(1, 2, 3)
    rotary = _ScaleRotary()
    attention = SimpleNamespace(
        swa_cache_layer=SimpleNamespace(prefix="layer.swa_cache", kv_cache=kv_cache),
        eps=1e-6,
        rotary_emb=rotary,
    )
    metadata = SimpleNamespace(slot_mapping=torch.tensor([-1, -1]), block_size=2)

    deepseek_mod.DeepseekV4FLAttention._fused_qnorm_rope_kv_insert(
        attention,
        torch.randn(2, 1, 3),
        torch.randn(2, 3),
        torch.tensor([0, 1]),
        {"layer.swa_cache": metadata},
    )

    assert torch.count_nonzero(kv_cache) == 0
    assert rotary.calls  # rope still ran on the normalized q


# ---------------------------------------------------------------------------
# DeepseekV4FLAttention._compress_kv_bf16
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "ratio,num_tokens,state_block_size",
    [(4, 8, 4), (128, 128, 128), (128, 128, 8)],
)
def test_compress_kv_bf16_matches_reference(
    deepseek_mod, ratio, num_tokens, state_block_size
):
    """The compressed key matches a hand-rolled softmax-weighted reference.

    Oracle ported from the torch_npu-gated core test: for every compressed
    boundary the window rows are gathered (offset by head_dim once past the
    overlap for ratio 4), softmaxed over the gate scores plus the
    position-dependent ape, RMSNormed and scaled by the compressor norm —
    with cos=1/sin=0 the rope tail rotation is the identity.
    """
    head_dim = 4
    coff = 1 + (ratio == 4)
    state_width = coff * head_dim
    generator = torch.Generator().manual_seed(17)
    kv_score = torch.randn(num_tokens, 2 * state_width, generator=generator) * 0.1
    ape = torch.randn(ratio, state_width, generator=generator) * 0.03
    norm_weight = torch.tensor([1.0, 0.8, 1.2, 0.9])
    positions = torch.arange(num_tokens)
    num_state_blocks = (num_tokens + state_block_size - 1) // state_block_size
    state_block_table = torch.arange(num_state_blocks).flip(0)
    state_slots = (
        state_block_table[positions // state_block_size] * state_block_size
        + positions % state_block_size
    )
    state_cache = torch.zeros(
        (num_state_blocks, state_block_size, 2 * state_width),
        dtype=torch.float32,
    )
    compressed_cache = torch.zeros((1, 128 // ratio, head_dim), dtype=torch.bfloat16)
    compressed_slots = torch.full((num_tokens,), -1, dtype=torch.long)
    boundaries = list(range(ratio - 1, num_tokens, ratio))
    for slot, position in enumerate(boundaries):
        compressed_slots[position] = slot
    state_metadata = SimpleNamespace(
        slot_mapping=state_slots,
        token_to_req_indices=torch.zeros(num_tokens, dtype=torch.int32),
        block_table=state_block_table.unsqueeze(0),
        block_size=state_block_size,
    )
    compressed_metadata = SimpleNamespace(slot_mapping=compressed_slots, block_size=128)
    cos_sin = torch.cat(
        (torch.ones((num_tokens, 1)), torch.zeros((num_tokens, 1))), dim=1
    )
    compressor = SimpleNamespace(
        ape=ape,
        state_cache=SimpleNamespace(
            prefix="layer.compressor.state_cache", kv_cache=state_cache
        ),
        norm=SimpleNamespace(weight=norm_weight),
    )
    attention = SimpleNamespace(
        prefix="layer",
        compressor=compressor,
        compress_ratio=ratio,
        head_dim=head_dim,
        rope_head_dim=2,
        eps=1e-6,
        rotary_emb=SimpleNamespace(cos_sin_cache=cos_sin),
        kv_cache=compressed_cache,
    )
    _set_attn_metadata(
        deepseek_mod,
        {
            "layer.compressor.state_cache": state_metadata,
            "layer": compressed_metadata,
        },
    )

    deepseek_mod.DeepseekV4FLAttention._compress_kv_bf16(attention, kv_score, positions)

    for slot, position in enumerate(boundaries):
        start = position - coff * ratio + 1
        keys = []
        scores = []
        for token in range(max(0, start), position + 1):
            head_offset = head_dim if ratio == 4 and token - start >= ratio else 0
            keys.append(kv_score[token, head_offset : head_offset + head_dim])
            scores.append(
                kv_score[
                    token,
                    state_width + head_offset : state_width + head_offset + head_dim,
                ]
                + ape[token % ratio, head_offset : head_offset + head_dim]
            )
        key_rows = torch.stack(keys)
        score_rows = torch.stack(scores)
        value = (torch.softmax(score_rows, dim=0) * key_rows).sum(dim=0)
        value = value * torch.rsqrt(value.square().mean() + attention.eps)
        value = (value * norm_weight).to(torch.bfloat16)
        torch.testing.assert_close(
            compressed_cache.view(-1, head_dim)[slot], value, rtol=0.02, atol=0.02
        )


def test_compress_kv_bf16_returns_early_without_dict_metadata(deepseek_mod):
    """A non-dict attn metadata makes the compression a no-op."""
    state_cache = torch.zeros(1, 4, 8)
    attention = SimpleNamespace(
        compressor=SimpleNamespace(
            ape=torch.zeros(4, 4),
            state_cache=SimpleNamespace(
                prefix="layer.compressor.state_cache", kv_cache=state_cache
            ),
            norm=SimpleNamespace(weight=torch.ones(4)),
        ),
        compress_ratio=4,
        head_dim=4,
    )
    _set_attn_metadata(deepseek_mod, object())

    deepseek_mod.DeepseekV4FLAttention._compress_kv_bf16(
        attention, torch.randn(4, 16), torch.arange(4)
    )

    assert torch.count_nonzero(state_cache) == 0


def test_compress_kv_bf16_drops_invalid_state_slots(deepseek_mod):
    """state rows whose slots are negative are filtered before the write."""
    ratio = 4
    head_dim = 4
    state_width = 2 * head_dim
    num_tokens = 4
    kv_score = torch.randn(num_tokens, 2 * state_width)
    positions = torch.arange(num_tokens)
    state_cache = torch.zeros(1, 4, 2 * state_width)
    state_metadata = SimpleNamespace(
        slot_mapping=torch.tensor([0, -1, 2, -1]),
        token_to_req_indices=torch.zeros(num_tokens, dtype=torch.int32),
        block_table=torch.tensor([[0]]),
        block_size=4,
    )
    compressed_metadata = SimpleNamespace(
        slot_mapping=torch.full((num_tokens,), -1, dtype=torch.long), block_size=128
    )
    compressor = SimpleNamespace(
        ape=torch.zeros(ratio, state_width),
        state_cache=SimpleNamespace(
            prefix="layer.compressor.state_cache", kv_cache=state_cache
        ),
        norm=SimpleNamespace(weight=torch.ones(head_dim)),
    )
    attention = SimpleNamespace(
        prefix="layer",
        compressor=compressor,
        compress_ratio=ratio,
        head_dim=head_dim,
        rope_head_dim=2,
        eps=1e-6,
        rotary_emb=SimpleNamespace(
            cos_sin_cache=torch.cat(
                (torch.ones((num_tokens, 1)), torch.zeros((num_tokens, 1))), dim=1
            )
        ),
        kv_cache=torch.zeros(1, 32, head_dim, dtype=torch.bfloat16),
    )
    _set_attn_metadata(
        deepseek_mod,
        {
            "layer.compressor.state_cache": state_metadata,
            "layer": compressed_metadata,
        },
    )

    deepseek_mod.DeepseekV4FLAttention._compress_kv_bf16(attention, kv_score, positions)

    # Only slots 0 and 2 stored: packed rows are cat(kv, score+ape).
    ape = compressor.ape
    expected_0 = torch.cat(
        (kv_score[0, :state_width], kv_score[0, state_width:] + ape[0])
    )
    expected_2 = torch.cat(
        (kv_score[2, :state_width], kv_score[2, state_width:] + ape[2])
    )
    torch.testing.assert_close(state_cache[0, 0], expected_0)
    torch.testing.assert_close(state_cache[0, 2], expected_2)
    assert torch.count_nonzero(state_cache[0, 1]) == 0
    assert torch.count_nonzero(state_cache[0, 3]) == 0


def test_compress_kv_bf16_skips_compression_without_boundary(deepseek_mod):
    """No position with (pos + 1) % ratio == 0 means no compressed rows.

    Independent oracle: with ratio 4 and positions [0, 1, 2], every
    (pos + 1) is below 4, so the active mask is empty, the state rows are
    still written, but the compressed cache stays untouched.
    """
    ratio = 4
    head_dim = 4
    state_width = 2 * head_dim
    num_tokens = 3
    kv_score = torch.randn(num_tokens, 2 * state_width)
    positions = torch.arange(num_tokens)
    state_cache = torch.zeros(1, 4, 2 * state_width)
    compressed_cache = torch.zeros(1, 32, head_dim, dtype=torch.bfloat16)
    state_metadata = SimpleNamespace(
        slot_mapping=torch.arange(num_tokens),
        token_to_req_indices=torch.zeros(num_tokens, dtype=torch.int32),
        block_table=torch.tensor([[0]]),
        block_size=4,
    )
    compressed_metadata = SimpleNamespace(
        slot_mapping=torch.arange(num_tokens), block_size=128
    )
    compressor = SimpleNamespace(
        ape=torch.zeros(ratio, state_width),
        state_cache=SimpleNamespace(
            prefix="layer.compressor.state_cache", kv_cache=state_cache
        ),
        norm=SimpleNamespace(weight=torch.ones(head_dim)),
    )
    attention = SimpleNamespace(
        prefix="layer",
        compressor=compressor,
        compress_ratio=ratio,
        head_dim=head_dim,
        rope_head_dim=2,
        eps=1e-6,
        rotary_emb=SimpleNamespace(
            cos_sin_cache=torch.cat(
                (torch.ones((num_tokens, 1)), torch.zeros((num_tokens, 1))), dim=1
            )
        ),
        kv_cache=compressed_cache,
    )
    _set_attn_metadata(
        deepseek_mod,
        {
            "layer.compressor.state_cache": state_metadata,
            "layer": compressed_metadata,
        },
    )

    deepseek_mod.DeepseekV4FLAttention._compress_kv_bf16(attention, kv_score, positions)

    assert torch.count_nonzero(state_cache) > 0
    assert torch.count_nonzero(compressed_cache) == 0


def test_compress_kv_bf16_requires_compressor_when_called(deepseek_mod):
    """The method asserts a compressor exists before touching the caches."""
    attention = SimpleNamespace(compressor=None, compress_ratio=4)
    _set_attn_metadata(
        deepseek_mod,
        {
            "layer.compressor.state_cache": SimpleNamespace(),
            "layer": SimpleNamespace(),
        },
    )
    with pytest.raises(AssertionError):
        deepseek_mod.DeepseekV4FLAttention._compress_kv_bf16(
            attention, torch.zeros(1, 8), torch.zeros(1)
        )


# ---------------------------------------------------------------------------
# DeepseekV4FLAttention._prepare_and_attn
# ---------------------------------------------------------------------------


def test_prepare_and_attn_runs_pipeline_without_compressor(deepseek_mod):
    """Without a compressor, q flows straight from wq_b through mqa.

    Independent oracle: qr [tokens, 2] with wq_b weight [heads*head_dim, 2]
    must reshape to [tokens, heads, head_dim]; _compress_kv_bf16 must not
    run and the C4A topk fill must not run for compress_ratio 0.
    """
    calls = {}

    def fake_wq_b(qr):
        calls["wq_b"] = qr
        # maps [tokens, 2] -> [tokens, 3] so the view is [tokens, 1, 3]
        return qr @ torch.tensor([[1.0, 0.0, 1.0], [0.0, 1.0, 1.0]])

    class _Rotary:
        def forward_native(self, positions, q, kv):
            return q * 2.0, kv * 3.0 if kv is not None else None

    swa_cache = torch.zeros(1, 4, 2)
    attention = SimpleNamespace(
        prefix="layer",
        wq_b=fake_wq_b,
        _fused_qnorm_rope_kv_insert=lambda q, kv, positions, attn_metadata: (
            deepseek_mod.DeepseekV4FLAttention._fused_qnorm_rope_kv_insert(
                attention, q, kv, positions, attn_metadata
            )
        ),
        rotary_emb=_Rotary(),
        eps=1e-6,
        swa_cache_layer=SimpleNamespace(prefix="layer.swa_cache", kv_cache=swa_cache),
        compressor=None,
        compress_ratio=0,
        n_local_heads=1,
        head_dim=3,
        forward_mqa=lambda q, kv, positions, output: calls.setdefault(
            "mqa", (q, kv, positions, output)
        ),
    )
    metadata = {
        "layer.swa_cache": SimpleNamespace(slot_mapping=torch.tensor([0]), block_size=4)
    }
    _set_attn_metadata(deepseek_mod, metadata)

    qr = torch.tensor([[1.0, 2.0]])
    kv = torch.tensor([[3.0, 4.0]])
    positions = torch.tensor([0])
    o_padded = torch.zeros(1, 1, 2)

    deepseek_mod.DeepseekV4FLAttention._prepare_and_attn(
        attention, "hs", qr, kv, None, None, None, positions, o_padded
    )

    mqa_q, mqa_kv, mqa_positions, mqa_output = calls["mqa"]
    # wq_b maps [1,2] -> [1, 3] via [[1,0,1],[0,1,1]], viewed [1, 1, 3],
    # then qnorm (rsqrt(mean(q^2)+eps) with q=[1,2,3]... wait: [1*1+2*0,
    # 1*0+2*1, 1*1+2*1] = [1, 2, 3]) and rope q*2.
    torch.testing.assert_close(
        mqa_q,
        torch.tensor([[[2.0, 4.0, 6.0]]])
        * torch.rsqrt(torch.tensor(14.0 / 3.0 + 1e-6)),
    )
    torch.testing.assert_close(mqa_kv, kv)
    assert mqa_positions is positions
    assert mqa_output is o_padded
    assert torch.count_nonzero(swa_cache[0, 0]) > 0  # rotated kv stored


def test_prepare_and_attn_runs_compressor_and_c4a_topk_fill(deepseek_mod):
    """With ratio 4 the compressor runs and the topk buffer is filled.

    Independent oracle: for positions [1, 3, 4, 7] the candidate count is
    (pos + 1) // 4 = [0, 1, 1, 2], so the buffer row must hold candidate
    ids [0..topk_tokens-1] where valid and -1 beyond the count.
    """
    calls = {}

    attention = SimpleNamespace(
        prefix="layer",
        wq_b=lambda qr: qr,
        _fused_qnorm_rope_kv_insert=lambda q, kv, positions, attn_metadata: (
            deepseek_mod.DeepseekV4FLAttention._fused_qnorm_rope_kv_insert(
                attention, q, kv, positions, attn_metadata
            )
        ),
        rotary_emb=_ScaleRotary(),
        eps=1e-6,
        swa_cache_layer=SimpleNamespace(
            prefix="layer.swa_cache", kv_cache=torch.zeros(1, 4, 2)
        ),
        compressor=SimpleNamespace(mark="compressor"),
        compress_ratio=4,
        n_local_heads=1,
        head_dim=2,
        indexer=SimpleNamespace(topk_tokens=8),
        topk_indices_buffer=torch.zeros(4, 8, dtype=torch.int32),
        forward_mqa=lambda q, kv, positions, output: calls.setdefault("mqa", (q, kv)),
    )
    metadata = {
        "layer.swa_cache": SimpleNamespace(
            slot_mapping=torch.tensor([0, 1, 2, 3]), block_size=4
        ),
        "layer.compressor.state_cache": SimpleNamespace(
            slot_mapping=torch.tensor([0, 1, 2, 3]),
            block_size=4,
            token_to_req_indices=torch.zeros(4, dtype=torch.int32),
            block_table=torch.tensor([[0]]),
        ),
        "layer": SimpleNamespace(
            slot_mapping=torch.tensor([0, 1, 2, 3]), block_size=16
        ),
    }
    _set_attn_metadata(deepseek_mod, metadata)

    def fake_compress(kv_score, positions):
        calls["compress"] = (kv_score, positions)

    attention._compress_kv_bf16 = fake_compress
    q = torch.ones(4, 1, 2)
    deepseek_mod.DeepseekV4FLAttention._prepare_and_attn(
        attention,
        "hs",
        torch.ones(4, 2),
        torch.ones(4, 2),
        "kv-score",
        None,
        None,
        torch.tensor([1, 3, 4, 7]),
        torch.zeros(4, 1, 2),
    )

    assert calls["compress"][0] == "kv-score"
    torch.testing.assert_close(calls["compress"][1], torch.tensor([1, 3, 4, 7]))
    torch.testing.assert_close(
        calls["mqa"][0], q * torch.rsqrt(torch.tensor(1.0 + 1e-6))
    )
    expected = torch.tensor(
        [
            [-1, -1, -1, -1, -1, -1, -1, -1],
            [0, -1, -1, -1, -1, -1, -1, -1],
            [0, -1, -1, -1, -1, -1, -1, -1],
            [0, 1, -1, -1, -1, -1, -1, -1],
        ],
        dtype=torch.int32,
    )
    torch.testing.assert_close(attention.topk_indices_buffer, expected)


def test_prepare_and_attn_skips_topk_fill_for_ratio_128(deepseek_mod):
    """Only ratio 4 fills the topk buffer; ratio 128 leaves it untouched."""
    buffer = torch.full((2, 8), 7, dtype=torch.int32)
    attention = SimpleNamespace(
        prefix="layer",
        wq_b=lambda qr: qr,
        _fused_qnorm_rope_kv_insert=lambda q, kv, positions, attn_metadata: (
            deepseek_mod.DeepseekV4FLAttention._fused_qnorm_rope_kv_insert(
                attention, q, kv, positions, attn_metadata
            )
        ),
        rotary_emb=_ScaleRotary(),
        eps=1e-6,
        swa_cache_layer=SimpleNamespace(
            prefix="layer.swa_cache", kv_cache=torch.zeros(1, 4, 2)
        ),
        compressor=None,
        compress_ratio=128,
        n_local_heads=1,
        head_dim=2,
        forward_mqa=lambda q, kv, positions, output: None,
    )
    attention.topk_indices_buffer = buffer
    _set_attn_metadata(
        deepseek_mod,
        {
            "layer.swa_cache": SimpleNamespace(
                slot_mapping=torch.tensor([0, 1]), block_size=4
            )
        },
    )

    deepseek_mod.DeepseekV4FLAttention._prepare_and_attn(
        attention,
        "hs",
        torch.ones(2, 2),
        torch.ones(2, 2),
        None,
        None,
        None,
        torch.tensor([127, 255]),
        torch.zeros(2, 1, 2),
    )

    assert torch.equal(
        attention.topk_indices_buffer, torch.full((2, 8), 7, dtype=torch.int32)
    )


# ---------------------------------------------------------------------------
# DeepseekV4FLAttention._o_proj
# ---------------------------------------------------------------------------


def test_o_proj_inverse_rope_then_two_stage_projection(deepseek_mod):
    """_o_proj inverse-ropes o, then applies wo_a and wo_b in order.

    Independent oracle: the fake rope returns q * 4; wo_a and wo_b are
    recording lambdas whose product must equal the returned tensor.
    """
    calls = []

    class _InverseRotary:
        def forward_native(self, positions, o, kv, inverse=False):
            assert inverse is True
            assert kv is None
            calls.append(("rope", positions))
            return o * 4.0, None

    def fake_wo_a(flat):
        calls.append(("wo_a", flat))
        return flat * 0.5

    def fake_wo_b(z):
        calls.append(("wo_b", z))
        return z + 1.0

    attention = SimpleNamespace(
        rotary_emb=_InverseRotary(), wo_a=fake_wo_a, wo_b=fake_wo_b, n_local_groups=1
    )
    o = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    positions = torch.tensor([0, 1])

    out = deepseek_mod.DeepseekV4FLAttention._o_proj(attention, o, positions)

    torch.testing.assert_close(out, o * 4.0 * 0.5 + 1.0)
    assert [name for name, _ in calls] == ["rope", "wo_a", "wo_b"]
    torch.testing.assert_close(calls[1][1], o * 4.0)
    torch.testing.assert_close(calls[2][1], o * 4.0 * 0.5)


def test_o_proj_rejects_multiple_local_groups(deepseek_mod):
    """More than one local group raises before any projection runs."""
    calls = []

    class _Rotary:
        def forward_native(self, positions, o, kv, inverse=False):
            calls.append("rope")
            return o, None

    attention = SimpleNamespace(
        rotary_emb=_Rotary(), wo_a=lambda x: x, wo_b=lambda x: x, n_local_groups=2
    )

    with pytest.raises(ValueError, match="one group per TP rank"):
        deepseek_mod.DeepseekV4FLAttention._o_proj(
            attention, torch.zeros(1, 4), torch.zeros(1, dtype=torch.long)
        )
    assert calls == ["rope"]  # the inverse rope still ran before the check


# ---------------------------------------------------------------------------
# DeepseekV4FLAttention.forward_mqa
# ---------------------------------------------------------------------------


def _swa_double(
    ratio=0,
    head_dim=4,
    window_size=128,
    sink_value=-float("inf"),
    swa_cache=None,
    compressed_cache=None,
):
    """Build the SimpleNamespace attention double forward_mqa reads."""
    swa_block = swa_cache.shape[1] if swa_cache is not None else 128
    return SimpleNamespace(
        prefix="layer",
        swa_cache_layer=SimpleNamespace(prefix="layer.swa_cache", kv_cache=swa_cache),
        kv_cache=compressed_cache,
        compress_ratio=ratio,
        max_model_len=128,
        window_size=window_size,
        head_dim=head_dim,
        n_local_heads=1,
        scale=head_dim**-0.5,
        attn_sink=torch.tensor([sink_value]),
        topk_indices_buffer=None,
        _swa_block=swa_block,
    )


def test_forward_mqa_zeroes_output_without_dict_metadata(deepseek_mod):
    """Non-dict metadata makes forward_mqa a zero-fill of the output."""
    output = torch.ones(2, 1, 4)
    attention = _swa_double(swa_cache=torch.zeros(1, 128, 4))
    _set_attn_metadata(deepseek_mod, object())

    deepseek_mod.DeepseekV4FLAttention.forward_mqa(
        attention, torch.zeros(2, 1, 4), torch.zeros(2, 4), torch.tensor([0, 1]), output
    )

    assert torch.count_nonzero(output) == 0


def test_forward_mqa_swa_only_matches_softmax_reference(deepseek_mod):
    """SWA-only attention over a full window matches a softmax oracle.

    Independent oracle: keys are the SWA cache rows [0..position], scores
    are q.k/sqrt(d) and the output is softmax(scores) @ keys. With
    attn_sink=-inf the sink term drops out of both numerator and
    denominator, so the naive softmax reference must hold exactly.
    """
    head_dim = 4
    swa_cache = torch.zeros(1, 128, head_dim)
    for i in range(16):
        swa_cache[0, i] = torch.arange(head_dim, dtype=torch.float32) * (i + 1) * 0.1
    attention = _swa_double(swa_cache=swa_cache)
    positions = torch.tensor([15])
    q = torch.ones(1, 1, head_dim)
    swa_metadata = SimpleNamespace(
        token_to_req_indices=torch.tensor([0], dtype=torch.int32),
        block_size=128,
        block_table=torch.tensor([[0]]),
    )
    _set_attn_metadata(deepseek_mod, {"layer.swa_cache": swa_metadata})
    output = torch.empty_like(q)

    deepseek_mod.DeepseekV4FLAttention.forward_mqa(
        attention, q, torch.zeros(1, head_dim), positions, output
    )

    keys = swa_cache[0, :16].float()
    logits = (q[0, 0].float() @ keys.T) * attention.scale
    expected = torch.softmax(logits, dim=-1) @ keys
    torch.testing.assert_close(
        output[0, 0], expected.to(output.dtype), rtol=0.01, atol=0.01
    )


def test_forward_mqa_crops_to_sliding_window(deepseek_mod):
    """Only the last window_size SWA rows take part in the softmax.

    Independent oracle: with position 63 and window 8, the reference
    softmax must use keys [56..63]; keys outside the window must not
    change the output even when they differ from the in-window ones.
    """
    head_dim = 4
    window_size = 8
    swa_cache = torch.zeros(1, 128, head_dim)
    for i in range(64):
        swa_cache[0, i] = torch.arange(head_dim, dtype=torch.float32) * (i + 1)
    attention = _swa_double(window_size=window_size, swa_cache=swa_cache)
    positions = torch.tensor([63])
    q = torch.ones(1, 1, head_dim)
    swa_metadata = SimpleNamespace(
        token_to_req_indices=torch.tensor([0], dtype=torch.int32),
        block_size=128,
        block_table=torch.tensor([[0]]),
    )
    _set_attn_metadata(deepseek_mod, {"layer.swa_cache": swa_metadata})
    output = torch.empty_like(q)

    deepseek_mod.DeepseekV4FLAttention.forward_mqa(
        attention, q, torch.zeros(1, head_dim), positions, output
    )

    keys = swa_cache[0, 56:64].float()
    logits = (q[0, 0].float() @ keys.T) * attention.scale
    expected = torch.softmax(logits, dim=-1) @ keys
    torch.testing.assert_close(
        output[0, 0], expected.to(output.dtype), rtol=0.01, atol=0.01
    )


def test_forward_mqa_masks_rows_before_window_start(deepseek_mod):
    """History rows at negative positions are masked out of the softmax.

    Independent oracle: with position 2 and window 8 only keys [0..2] are
    valid; the -inf-masked rows must contribute exactly zero weight, so
    the reference softmax over the valid keys alone must match.
    """
    head_dim = 4
    swa_cache = torch.zeros(1, 8, head_dim)
    for i in range(8):
        swa_cache[0, i] = torch.arange(head_dim, dtype=torch.float32) * (i + 1)
    attention = _swa_double(window_size=8, swa_cache=swa_cache)
    positions = torch.tensor([2])
    q = torch.ones(1, 1, head_dim)
    swa_metadata = SimpleNamespace(
        token_to_req_indices=torch.tensor([0], dtype=torch.int32),
        block_size=8,
        block_table=torch.tensor([[0]]),
    )
    _set_attn_metadata(deepseek_mod, {"layer.swa_cache": swa_metadata})
    output = torch.empty_like(q)

    deepseek_mod.DeepseekV4FLAttention.forward_mqa(
        attention, q, torch.zeros(1, head_dim), positions, output
    )

    keys = swa_cache[0, :3].float()
    logits = (q[0, 0].float() @ keys.T) * attention.scale
    expected = torch.softmax(logits, dim=-1) @ keys
    torch.testing.assert_close(
        output[0, 0], expected.to(output.dtype), rtol=0.01, atol=0.01
    )


def test_forward_mqa_skips_unallocated_blocks(deepseek_mod):
    """block_table entries of -1 mark unallocated pages and are masked.

    Independent oracle: with block table [[-1]] every SWA row is invalid;
    with a finite attn_sink of 0 the output must be exactly zero (all
    weight drained into the sink), not NaN.
    """
    head_dim = 4
    swa_cache = torch.ones(1, 8, head_dim)
    attention = _swa_double(window_size=8, sink_value=0.0, swa_cache=swa_cache)
    positions = torch.tensor([5])
    q = torch.ones(1, 1, head_dim)
    swa_metadata = SimpleNamespace(
        token_to_req_indices=torch.tensor([0], dtype=torch.int32),
        block_size=8,
        block_table=torch.tensor([[-1]]),
    )
    _set_attn_metadata(deepseek_mod, {"layer.swa_cache": swa_metadata})
    output = torch.empty_like(q)

    deepseek_mod.DeepseekV4FLAttention.forward_mqa(
        attention, q, torch.zeros(1, head_dim), positions, output
    )

    assert torch.count_nonzero(output) == 0
    assert torch.isfinite(output).all()


@pytest.mark.parametrize("ratio,position", [(4, 3), (128, 127)])
def test_forward_mqa_compressed_and_swa_share_softmax(deepseek_mod, ratio, position):
    """Compressed candidates and SWA rows share one softmax denominator.

    Oracle ported from the torch_npu-gated core test: the compressed
    cache row 0 holds all-ones and the SWA rows hold [0..position]; the
    reference concatenates both key sets, softmaxes and compares.
    """
    head_dim = 4
    swa_cache = torch.zeros((1, 128, head_dim), dtype=torch.bfloat16)
    compressed_cache = torch.zeros((1, 128 // ratio, head_dim), dtype=torch.bfloat16)
    compressed_cache[0, 0, :] = 1
    attention = _swa_double(
        ratio=ratio, swa_cache=swa_cache, compressed_cache=compressed_cache
    )
    attention.topk_indices_buffer = torch.arange(32, dtype=torch.int32).unsqueeze(0)
    q = torch.ones((1, 1, head_dim), dtype=torch.bfloat16)
    positions = torch.tensor([position])
    swa_metadata = SimpleNamespace(
        token_to_req_indices=torch.tensor([0], dtype=torch.int32),
        block_size=128,
        block_table=torch.tensor([[0]]),
    )
    compressed_metadata = SimpleNamespace(
        block_size=128, block_table=torch.tensor([[0]])
    )
    _set_attn_metadata(
        deepseek_mod,
        {"layer.swa_cache": swa_metadata, "layer": compressed_metadata},
    )
    output = torch.empty_like(q)

    deepseek_mod.DeepseekV4FLAttention.forward_mqa(attention, q, q, positions, output)

    keys = torch.cat((compressed_cache[0, :1], swa_cache[0, : position + 1]))
    logits = (q[0, 0].float() @ keys.float().T) * attention.scale
    expected = (torch.softmax(logits, dim=-1) @ keys.float()).to(q.dtype)
    torch.testing.assert_close(output[0, 0], expected, rtol=0.01, atol=0.01)
    assert output[0, 0, 0] > 0


def test_forward_mqa_c128a_uses_all_candidates(deepseek_mod):
    """C128A expands candidates 0..(pos+1)//ratio-1 from the block table.

    Independent oracle: with ratio 128, position 127 and a compressed
    cache holding a single candidate row, the reference is a softmax over
    [candidate, swa keys] — the candidate mask is
    candidates < (positions + 1) // ratio.
    """
    head_dim = 4
    ratio = 128
    swa_cache = torch.zeros((1, 128, head_dim), dtype=torch.bfloat16)
    for i in range(128):
        swa_cache[0, i] = 0.01 * i
    compressed_cache = torch.zeros((1, 1, head_dim), dtype=torch.bfloat16)
    compressed_cache[0, 0] = 1.5
    attention = _swa_double(
        ratio=ratio, swa_cache=swa_cache, compressed_cache=compressed_cache
    )
    q = torch.ones((1, 1, head_dim), dtype=torch.bfloat16)
    positions = torch.tensor([127])
    swa_metadata = SimpleNamespace(
        token_to_req_indices=torch.tensor([0], dtype=torch.int32),
        block_size=128,
        block_table=torch.tensor([[0]]),
    )
    compressed_metadata = SimpleNamespace(
        block_size=128, block_table=torch.tensor([[0]])
    )
    _set_attn_metadata(
        deepseek_mod,
        {"layer.swa_cache": swa_metadata, "layer": compressed_metadata},
    )
    output = torch.empty_like(q)

    deepseek_mod.DeepseekV4FLAttention.forward_mqa(attention, q, q, positions, output)

    keys = torch.cat((compressed_cache[0, :1], swa_cache[0, :128]))
    logits = (q[0, 0].float() @ keys.float().T) * attention.scale
    expected = (torch.softmax(logits, dim=-1) @ keys.float()).to(q.dtype)
    torch.testing.assert_close(output[0, 0], expected, rtol=0.01, atol=0.01)


def test_forward_mqa_batches_requests_via_token_to_req(deepseek_mod):
    """token_to_req_indices picks each token's block-table row.

    Independent oracle: two tokens of different requests; request 0 has
    keys [k..] in page 0, request 1 has different keys in page 1. Each
    token's output must equal the softmax oracle over its own page.
    """
    head_dim = 4
    swa_cache = torch.zeros(2, 8, head_dim)
    for i in range(8):
        swa_cache[0, i] = torch.arange(head_dim, dtype=torch.float32)
        swa_cache[1, i] = torch.arange(head_dim, dtype=torch.float32) * -0.5
    attention = _swa_double(window_size=8, swa_cache=swa_cache)
    q = torch.ones(2, 1, head_dim)
    positions = torch.tensor([4, 4])
    swa_metadata = SimpleNamespace(
        token_to_req_indices=torch.tensor([0, 1], dtype=torch.int32),
        block_size=8,
        block_table=torch.tensor([[0], [1]]),
    )
    _set_attn_metadata(deepseek_mod, {"layer.swa_cache": swa_metadata})
    output = torch.empty_like(q)

    deepseek_mod.DeepseekV4FLAttention.forward_mqa(
        attention, q, torch.zeros(2, head_dim), positions, output
    )

    for token, page in enumerate((0, 1)):
        keys = swa_cache[page, :5].float()
        logits = (q[token, 0].float() @ keys.T) * attention.scale
        expected = torch.softmax(logits, dim=-1) @ keys
        torch.testing.assert_close(
            output[token, 0], expected.to(output.dtype), rtol=0.01, atol=0.01
        )


def test_forward_mqa_finite_sink_beats_all_masked_rows(deepseek_mod):
    """A finite sink keeps fully-masked rows finite (zero), not NaN.

    Independent oracle: plain torch.softmax([-inf, -inf]) is NaN — the FL
    formula subtracts the max against the sink and adds exp(sink - max)
    to the denominator, so all-masked rows resolve to exact zeros.
    """
    naive = torch.softmax(torch.tensor([[-float("inf"), -float("inf")]]), -1)
    assert bool(torch.isnan(naive).all())
    head_dim = 4
    swa_cache = torch.ones(1, 8, head_dim)
    attention = _swa_double(window_size=1, sink_value=0.0, swa_cache=swa_cache)
    # window 1 and position 0: only key [0] valid; make it invalid via page -1.
    swa_metadata = SimpleNamespace(
        token_to_req_indices=torch.tensor([0], dtype=torch.int32),
        block_size=8,
        block_table=torch.tensor([[-1]]),
    )
    _set_attn_metadata(deepseek_mod, {"layer.swa_cache": swa_metadata})
    q = torch.ones(1, 1, head_dim)
    output = torch.empty_like(q)

    deepseek_mod.DeepseekV4FLAttention.forward_mqa(
        attention, q, torch.zeros(1, head_dim), torch.tensor([0]), output
    )

    assert torch.isfinite(output).all()
    assert torch.count_nonzero(output) == 0


def test_forward_mqa_c4a_topk_buffer_respects_validity(deepseek_mod):
    """C4A candidate -1 entries and beyond-count ids are masked.

    Independent oracle: buffer row [-1, 1, 0] with position 7 admits
    candidates 1 and 0 (both < (7 + 1) // 4 = 2); -1 is masked. The
    reference softmax runs over [candidate rows in buffer order, swa].
    """
    head_dim = 4
    ratio = 4
    swa_cache = torch.zeros((1, 128, head_dim), dtype=torch.bfloat16)
    for i in range(8):
        swa_cache[0, i] = 0.1 * i
    compressed_cache = torch.zeros((1, 32, head_dim), dtype=torch.bfloat16)
    for i in range(3):
        compressed_cache[0, i] = 1.0 + i
    attention = _swa_double(
        ratio=ratio, swa_cache=swa_cache, compressed_cache=compressed_cache
    )
    attention.topk_indices_buffer = torch.tensor([[-1, 1, 0]], dtype=torch.int32)
    q = torch.ones((1, 1, head_dim), dtype=torch.bfloat16)
    positions = torch.tensor([7])
    swa_metadata = SimpleNamespace(
        token_to_req_indices=torch.tensor([0], dtype=torch.int32),
        block_size=128,
        block_table=torch.tensor([[0]]),
    )
    compressed_metadata = SimpleNamespace(
        block_size=128, block_table=torch.tensor([[0]])
    )
    _set_attn_metadata(
        deepseek_mod,
        {"layer.swa_cache": swa_metadata, "layer": compressed_metadata},
    )
    output = torch.empty_like(q)

    deepseek_mod.DeepseekV4FLAttention.forward_mqa(attention, q, q, positions, output)

    keys = torch.cat((compressed_cache[0, [1, 0]], swa_cache[0, :8]))
    logits = (q[0, 0].float() @ keys.float().T) * attention.scale
    expected = (torch.softmax(logits, dim=-1) @ keys.float()).to(q.dtype)
    torch.testing.assert_close(output[0, 0], expected, rtol=0.01, atol=0.01)


# ---------------------------------------------------------------------------
# Real vLLM construction (runs where vLLM is installed, e.g. the WSL CPU env)
# ---------------------------------------------------------------------------

try:
    import vllm  # noqa: F401

    _HAS_REAL_VLLM = True
except ImportError:  # pragma: no cover - depends on the host install
    _HAS_REAL_VLLM = False


def _real_vllm_config(
    compress_ratios,
    hidden_size,
    q_lora_rank,
    head_dim,
    qk_rope_head_dim,
    index_n_heads=None,
    index_head_dim=None,
):
    """Build the FakeVC shell validated against vLLM 0.32 CPU construction."""
    from vllm.config import VllmConfig

    hf_config = SimpleNamespace(
        compress_ratios=compress_ratios,
        num_hidden_layers=1,
        sliding_window=128,
        index_topk=64,
        hidden_size=hidden_size,
        num_attention_heads=2,
        q_lora_rank=q_lora_rank,
        o_lora_rank=q_lora_rank,
        head_dim=head_dim,
        qk_rope_head_dim=qk_rope_head_dim,
        o_groups=1,
        rms_norm_eps=1e-6,
        max_position_embeddings=128,
        rope_parameters={
            "rope_type": "default",
            "rope_theta": 10000.0,
            "factor": 1.0,
            "original_max_position_embeddings": 128,
        },
        rope_theta=10000.0,
        compress_rope_theta=10000.0,
        model_type="deepseek_v4",
        architectures=["DeepseekV4ForCausalLM"],
    )
    if index_n_heads is not None:
        hf_config.index_n_heads = index_n_heads
        hf_config.index_head_dim = index_head_dim
    base = VllmConfig()
    shell = SimpleNamespace()
    for attr in (a for a in dir(base) if not a.startswith("_")):
        with contextlib.suppress(Exception):  # best-effort attribute copy
            setattr(shell, attr, getattr(base, attr))
    shell.model_config = SimpleNamespace(
        hf_config=hf_config,
        max_model_len=128,
        is_multimodal_model=False,
        is_moe=False,
        dtype="bfloat16",
    )
    shell.use_v2_model_runner = False
    shell.attention_config.sparse_mla_prefill_workspace_factor = 1
    return shell


def _init_gloo_dist():
    """One-process gloo distributed group (idempotent within a session)."""
    import tempfile

    import torch.distributed as torch_dist

    import vllm.distributed as dist

    if torch_dist.is_initialized():
        return
    dist.init_distributed_environment(
        backend="gloo",
        world_size=1,
        rank=0,
        distributed_init_method="file://" + tempfile.mktemp(),
        local_rank=0,
    )
    dist.initialize_model_parallel(1, 1)


@pytest.fixture
def swa_layer():
    """Construct a real SWA-only DeepseekV4FLAttention on CPU."""
    pytest.importorskip("vllm", reason="real vLLM install required (WSL CPU env)")
    import torch.nn.functional as F

    import vllm.config.vllm as vv

    from vllm_fl.models import deepseek_v4 as real_module

    vc = _real_vllm_config([0], 32, 8, 16, 8)
    previous_config = vv._current_vllm_config
    vv._current_vllm_config = vc
    had_npu = hasattr(torch, "npu")
    original_npu = getattr(torch, "npu", None)

    class _InstantiableEvent:
        """Upstream __init__ instantiates the swapped Event class."""

    torch.npu = SimpleNamespace(Event=_InstantiableEvent)
    try:
        _init_gloo_dist()
        torch.manual_seed(0)
        attn = real_module.DeepseekV4FLAttention(
            prefix="supp.swa.layers.0", vllm_config=vc
        )
    finally:
        vv._current_vllm_config = previous_config
        if had_npu:
            torch.npu = original_npu
        else:
            del torch.npu
    with torch.no_grad():
        attn.attn_sink.fill_(0.0)
        for p in (attn.fused_wqa_wkv, attn.wq_b, attn.wo_a, attn.wo_b):
            p.weight.copy_(torch.randn_like(p.weight) * 0.05)
        attn.q_norm.weight.fill_(1.0)
        attn.kv_norm.weight.fill_(1.0)
    for _, mod in attn.named_modules():
        if type(mod).__name__ in (
            "MergedColumnParallelLinear",
            "ColumnParallelLinear",
            "RowParallelLinear",
        ):
            mod.cpu_linear = F.linear
    attn._prepare_and_attn_fn = attn._prepare_and_attn
    bs = attn.swa_cache_layer.block_size
    attn.swa_cache_layer.kv_cache = torch.zeros(2, bs, 16)
    return attn, vc


@pytest.mark.skipif(
    not _HAS_REAL_VLLM, reason="real vLLM install required (WSL CPU env)"
)
def test_real_swa_forward_matches_from_scratch_reference(swa_layer):
    """A real forward equals a from-scratch eager reference computation.

    Independent oracle, derived without the module under test's helpers:
    split the fused projection, RMSNorm both halves, wq_b-project q,
    normalize q per head, rope q/kv, causal softmax over the rotated kv
    rows (with the finite sink in the denominator), inverse-rope the
    context, then wo_a/wo_b.
    """
    from vllm.forward_context import set_forward_context

    attn, vc = swa_layer
    eps = 1e-6
    hs = torch.randn(4, 32, dtype=torch.float32)
    positions = torch.arange(4)
    bs = attn.swa_cache_layer.block_size
    swa_meta = SimpleNamespace(
        slot_mapping=torch.arange(4),
        block_size=bs,
        block_table=torch.tensor([[0, 0]]),
        token_to_req_indices=torch.zeros(4, dtype=torch.long),
    )

    with set_forward_context({"supp.swa.layers.0.swa_cache": swa_meta}, vc):
        out = attn.forward(positions, hs)

    assert out.shape == (4, 32)
    assert bool(torch.isfinite(out).all())

    weight = attn.fused_wqa_wkv.weight
    qr_kv = hs @ weight.T
    qr, kv = qr_kv[:, :8], qr_kv[:, 8:]
    qr = qr * torch.rsqrt(qr.pow(2).mean(-1, keepdim=True) + eps)
    kv = kv * torch.rsqrt(kv.pow(2).mean(-1, keepdim=True) + eps)
    q = (qr @ attn.wq_b.weight.T).view(4, 2, 16)
    q = q * torch.rsqrt(q.float().pow(2).mean(-1, keepdim=True) + eps).to(q.dtype)
    q_rot, kv_rot = attn.rotary_emb.forward_native(positions, q, kv.unsqueeze(1))
    kv_flat = kv_rot.squeeze(1)
    stored = attn.swa_cache_layer.kv_cache[0, :4]
    torch.testing.assert_close(stored.float(), kv_flat.float(), rtol=0.01, atol=1e-3)

    scores = torch.einsum("thd,sd->ths", q_rot.float(), kv_flat.float()) * (16**-0.5)
    valid = torch.tril(torch.ones(4, 4, dtype=torch.bool))
    scores = scores.masked_fill(~valid[:, None, :], -float("inf"))
    sink = torch.zeros(2)
    max_score = torch.maximum(scores.amax(-1, keepdim=True), sink[None, :, None])
    w = torch.exp(scores - max_score) * valid[:, None, :]
    denom = w.sum(-1, keepdim=True) + torch.exp(sink[None, :, None] - max_score)
    ctx = torch.einsum("ths,sd->thd", w, kv_flat.float()) / denom
    o_inv, _ = attn.rotary_emb.forward_native(positions, ctx, None, inverse=True)
    z = o_inv.flatten(1) @ attn.wo_a.weight.T
    reference = z @ attn.wo_b.weight.T

    torch.testing.assert_close(out, reference, rtol=0.01, atol=2e-2)


@pytest.mark.skipif(
    not _HAS_REAL_VLLM, reason="real vLLM install required (WSL CPU env)"
)
def test_real_swa_rope_round_trip_is_identity(swa_layer):
    """The scaling rotary rope and its inverse cancel exactly."""
    attn, _ = swa_layer
    q = torch.randn(3, 2, 16)
    positions = torch.tensor([0, 5, 17])
    forward, _ = attn.rotary_emb.forward_native(positions, q, None)
    restored, _ = attn.rotary_emb.forward_native(positions, forward, None, inverse=True)
    torch.testing.assert_close(restored, q, rtol=1e-4, atol=1e-4)
