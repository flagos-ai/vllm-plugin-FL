# SPDX-License-Identifier: Apache-2.0
"""Residual DeepSeek-V4 FL coverage: C4A/C128A compression and dispatch.

These paths were not reached by w8b (which stops at the non-dict early
return) or w17 (import/init only).  The real source file is executed with the
upstream vLLM packages supplied as explicit stubs; every tensor operation and
the ``_compress_kv_bf16`` / ``_prepare_and_attn`` logic under test is the real
implementation.  No weights or accelerators are touched.
"""

from __future__ import annotations

import importlib.util
import os
import sys
import types
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

SOURCE_REL = "vllm_fl/models/deepseek_v4.py"


def _source_path() -> Path:
    candidates = []
    override = os.environ.get("VLLM_FL_SOURCE_ROOT")
    if override:
        candidates.append(Path(override))
    candidates.extend(
        [Path("/tmp/flsrc"), Path.cwd(), Path(__file__).resolve().parents[2]]
    )
    for root in candidates:
        target = root / SOURCE_REL
        if target.is_file():
            return target
    raise FileNotFoundError("could not locate " + SOURCE_REL)


def _ensure_pkg(monkeypatch, name):
    existing = sys.modules.get(name)
    if isinstance(existing, types.ModuleType):
        return existing
    module = types.ModuleType(name)
    module.__path__ = []
    monkeypatch.setitem(sys.modules, name, module)
    if "." in name:
        parent_name, leaf = name.rsplit(".", 1)
        parent = _ensure_pkg(monkeypatch, parent_name)
        monkeypatch.setattr(parent, leaf, module, raising=False)
    return module


def _install(monkeypatch, name, **attrs):
    module = types.ModuleType(name)
    for key, value in attrs.items():
        setattr(module, key, value)
    if "." in name:
        parent_name, leaf = name.rsplit(".", 1)
        parent = _ensure_pkg(monkeypatch, parent_name)
        monkeypatch.setattr(parent, leaf, module, raising=False)
    monkeypatch.setitem(sys.modules, name, module)
    return module


@pytest.fixture()
def dsv4(monkeypatch):
    class _BaseAttention:
        def __init__(self, *args, **kwargs):
            pass

    class _SWABuilder:
        pass

    class _SWABackend:
        pass

    class _MLABackend:
        pass

    _ensure_pkg(monkeypatch, "vllm.models.deepseek_v4.xpu")
    _install(
        monkeypatch,
        "vllm.models.deepseek_v4.xpu.model",
        DeepseekV4ForCausalLM=object,
        DeepseekV4XPUAttention="ORIGINAL",
    )
    _install(
        monkeypatch,
        "vllm.models.deepseek_v4.attention",
        DeepseekV4Attention=_BaseAttention,
    )
    _install(
        monkeypatch,
        "vllm.models.deepseek_v4.sparse_mla",
        DeepseekV4FlashMLAMetadata=object,
        DeepseekV4SparseMLABackend=_MLABackend,
        DeepseekV4SparseMLAMetadataBuilder=_SWABuilder,
    )
    _install(
        monkeypatch,
        "vllm.v1.attention.backends.mla.sparse_swa",
        DeepseekSparseSWABackend=_SWABackend,
        DeepseekSparseSWAMetadata=object,
        DeepseekSparseSWAMetadataBuilder=_SWABuilder,
    )
    _install(
        monkeypatch,
        "vllm.forward_context",
        get_forward_context=lambda: SimpleNamespace(attn_metadata=None),
    )
    _install(
        monkeypatch,
        "vllm.platforms",
        current_platform=SimpleNamespace(device_type="cuda"),
    )

    spec = importlib.util.spec_from_file_location("_w25_dsv4", _source_path())
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _build(cls, **attrs):
    obj = object.__new__(cls)
    for name, value in attrs.items():
        setattr(obj, name, value)
    return obj


def _context(monkeypatch, module, attn_metadata):
    monkeypatch.setattr(
        module,
        "get_forward_context",
        lambda: SimpleNamespace(attn_metadata=attn_metadata),
    )


def _reference_compressed(attn, state_cache, state_meta, active_positions, req_ids):
    """Independent per-position re-computation of the compressed key."""
    ratio = attn.compress_ratio
    overlap = ratio == 4
    coff = 1 + int(overlap)
    head_dim = attn.head_dim
    state_width = coff * head_dim
    window = coff * ratio
    bs = state_meta.block_size
    block_table = state_meta.block_table
    norm_w = attn.compressor.norm.weight.float()
    eps = attn.eps
    outputs = []
    for pos, req in zip(active_positions.tolist(), req_ids.tolist()):
        kv_rows = []
        score_rows = []
        for offset in range(window):
            hp = pos - window + 1 + offset
            if hp < 0:
                kv_rows.append(torch.zeros(head_dim))
                score_rows.append(torch.full((head_dim,), float("-inf")))
                continue
            block = int(block_table[req, hp // bs])
            if block < 0:
                kv_rows.append(torch.zeros(head_dim))
                score_rows.append(torch.full((head_dim,), float("-inf")))
                continue
            row = state_cache[block, hp % bs]
            if overlap and offset >= ratio:
                cols = torch.arange(head_dim) + head_dim
            else:
                cols = torch.arange(head_dim)
            kv_rows.append(row[:state_width][cols])
            score_rows.append(row[state_width:][cols])
        weights = torch.softmax(torch.stack(score_rows), dim=0)
        comp = (weights * torch.stack(kv_rows)).sum(dim=0)
        comp = (
            comp
            * torch.rsqrt(comp.square().mean(dim=-1, keepdim=True) + eps)
            * norm_w
        )
        outputs.append(comp)
    return outputs


def _compress_state(monkeypatch, module, *, ratio, positions):
    head_dim = 2
    rope_dim = 2
    overlap = ratio == 4
    coff = 1 + int(overlap)
    state_width = coff * head_dim
    window = coff * ratio

    num_tokens = positions.shape[0]
    state_block_size = 4 if overlap else 8
    state_blocks = 2
    state_cache = torch.zeros(
        state_blocks, state_block_size, 2 * state_width, dtype=torch.float32
    )
    state_slots = torch.arange(num_tokens, dtype=torch.long)
    block_table = torch.tensor(
        [[0, -1]] if overlap else [[0, 1]], dtype=torch.long
    )
    state_meta = SimpleNamespace(
        slot_mapping=state_slots,
        block_size=state_block_size,
        block_table=block_table,
        token_to_req_indices=torch.zeros(num_tokens, dtype=torch.long),
    )

    compressed_block_size = 2 * ratio if overlap else 2 * ratio
    comp_blocks = 2
    kv_cache = torch.zeros(
        comp_blocks, compressed_block_size // ratio, head_dim, dtype=torch.float32
    )
    comp_meta = SimpleNamespace(
        slot_mapping=torch.arange(num_tokens, dtype=torch.long),
        block_size=compressed_block_size,
    )

    kv_score = torch.arange(
        num_tokens * 2 * state_width, dtype=torch.float32
    ).reshape(num_tokens, 2 * state_width) * 0.1
    ape = (
        torch.arange(ratio * state_width, dtype=torch.float32).reshape(
            ratio, state_width
        )
        * 0.01
    )
    norm_w = torch.tensor([1.5, 0.5], dtype=torch.float32)
    cos_sin_cache = torch.zeros(32, rope_dim, dtype=torch.float32)
    cos_sin_cache[:, 0] = 1.0

    compressor = SimpleNamespace(
        state_cache=SimpleNamespace(prefix="state", kv_cache=state_cache),
        ape=ape,
        norm=SimpleNamespace(weight=norm_w),
    )
    attn = _build(
        module.DeepseekV4FLAttention,
        prefix="comp",
        compress_ratio=ratio,
        head_dim=head_dim,
        rope_head_dim=rope_dim,
        eps=1e-6,
        compressor=compressor,
        kv_cache=kv_cache,
        rotary_emb=SimpleNamespace(cos_sin_cache=cos_sin_cache),
    )
    _context(
        monkeypatch,
        module,
        {"state": state_meta, "comp": comp_meta},
    )
    return SimpleNamespace(
        attn=attn,
        kv_score=kv_score,
        state_cache=state_cache,
        state_meta=state_meta,
        kv_cache=kv_cache,
        comp_meta=comp_meta,
        window=window,
    )


def test_compress_c4a_overlap_writes_state_and_compressed(dsv4, monkeypatch):
    positions = torch.tensor([0, 3, 6, 7], dtype=torch.long)
    ctx = _compress_state(monkeypatch, dsv4, ratio=4, positions=positions)
    attn = ctx.attn

    attn._compress_kv_bf16(ctx.kv_score, positions)

    # State cache rows hold kv | (score + ape) for every valid slot.
    state_width = 2 * attn.head_dim  # coff(2) * head_dim
    expected_state = torch.cat(
        (
            ctx.kv_score[:, :state_width],
            ctx.kv_score[:, state_width:]
            + ctx.attn.compressor.ape[positions.remainder(attn.compress_ratio)],
        ),
        dim=-1,
    ).to(ctx.state_cache.dtype)
    for slot in range(positions.shape[0]):
        block, row = divmod(slot, ctx.state_meta.block_size)
        torch.testing.assert_close(ctx.state_cache[block, row], expected_state[slot])

    # Only positions 3 and 7 satisfy (pos + 1) % 4 == 0.
    active = ((positions + 1).remainder(4) == 0).nonzero().flatten()
    active_positions = positions[active]
    req_ids = ctx.state_meta.token_to_req_indices[active]
    expected = _reference_compressed(
        attn, ctx.state_cache, ctx.state_meta, active_positions, req_ids
    )
    block_size = ctx.comp_meta.block_size // attn.compress_ratio
    slots = ctx.comp_meta.slot_mapping[active].tolist()
    for slot, want in zip(slots, expected):
        block, row = divmod(slot, block_size)
        torch.testing.assert_close(ctx.kv_cache[block, row], want, atol=1e-5, rtol=1e-4)
        assert torch.isfinite(ctx.kv_cache[block, row]).all()

    # Inactive compressed slots stay untouched.
    touched = {
        divmod(slot, block_size) for slot in slots
    }
    for block in range(ctx.kv_cache.shape[0]):
        for row in range(ctx.kv_cache.shape[1]):
            if (block, row) not in touched:
                assert torch.count_nonzero(ctx.kv_cache[block, row]) == 0


def test_compress_c128a_non_overlap_uses_planar_channels(dsv4, monkeypatch):
    positions = torch.tensor([6, 7], dtype=torch.long)
    ctx = _compress_state(monkeypatch, dsv4, ratio=8, positions=positions)

    ctx.attn._compress_kv_bf16(ctx.kv_score, positions)

    state_width = ctx.attn.head_dim  # coff(1) * head_dim
    expected_state = torch.cat(
        (
            ctx.kv_score[:, :state_width],
            ctx.kv_score[:, state_width:]
            + ctx.attn.compressor.ape[positions.remainder(8)],
        ),
        dim=-1,
    ).to(ctx.state_cache.dtype)
    torch.testing.assert_close(
        ctx.state_cache[0, 0], expected_state[0]
    )
    torch.testing.assert_close(
        ctx.state_cache[0, 1], expected_state[1]
    )

    # Only position 7 is divisible into a compressed candidate.
    active = ((positions + 1).remainder(8) == 0).nonzero().flatten()
    assert active.tolist() == [1]
    expected = _reference_compressed(
        ctx.attn,
        ctx.state_cache,
        ctx.state_meta,
        positions[active],
        ctx.state_meta.token_to_req_indices[active],
    )[0]
    block_size = ctx.comp_meta.block_size // 8
    slot = int(ctx.comp_meta.slot_mapping[active][0])
    block, row = divmod(slot, block_size)
    torch.testing.assert_close(ctx.kv_cache[block, row], expected, atol=1e-5, rtol=1e-4)
    # The other compressed row was never written.
    other = (0, 0) if (block, row) != (0, 0) else (0, 1)
    assert torch.count_nonzero(ctx.kv_cache[other]) == 0


def test_compress_all_invalid_slots_short_circuits(dsv4, monkeypatch):
    positions = torch.tensor([6, 7], dtype=torch.long)
    ctx = _compress_state(monkeypatch, dsv4, ratio=8, positions=positions)
    ctx.state_meta.slot_mapping = torch.tensor([-1, -1], dtype=torch.long)
    ctx.comp_meta.slot_mapping = torch.tensor([-1, -1], dtype=torch.long)
    _context(
        monkeypatch,
        dsv4,
        {"state": ctx.state_meta, "comp": ctx.comp_meta},
    )

    ctx.attn._compress_kv_bf16(ctx.kv_score, positions)

    # valid_state.any() is False, so no state row is written.
    assert torch.count_nonzero(ctx.state_cache) == 0
    # No compressed candidate is active, so nothing is stored.
    assert torch.count_nonzero(ctx.kv_cache) == 0


def _prepare_attn(dsv4, *, compress_ratio, num_tokens=3):
    n_local_heads, head_dim = 1, 2
    recorded = {}

    def fake_insert(q, kv, positions, metadata):
        return q

    def fake_compress(kv_score, positions):
        recorded["kv_score"] = kv_score
        recorded["positions"] = positions

    def fake_mqa(q, kv, positions, out):
        recorded["q"] = q
        recorded["mqa_out"] = out

    attn = _build(
        dsv4.DeepseekV4FLAttention,
        n_local_heads=n_local_heads,
        head_dim=head_dim,
        compress_ratio=compress_ratio,
        compressor=SimpleNamespace(),
        wq_b=lambda x: torch.arange(
            x.shape[0] * n_local_heads * head_dim, dtype=torch.float32
        ).reshape(x.shape[0], n_local_heads * head_dim),
        indexer=SimpleNamespace(topk_tokens=3),
        topk_indices_buffer=torch.full(
            (num_tokens, 3), 77, dtype=torch.int32
        ),
    )
    attn._fused_qnorm_rope_kv_insert = fake_insert
    attn._compress_kv_bf16 = fake_compress
    attn.forward_mqa = fake_mqa
    return attn, recorded


def test_prepare_dispatches_compression_and_c4a_indices(dsv4, monkeypatch):
    attn, recorded = _prepare_attn(dsv4, compress_ratio=4)
    _context(monkeypatch, dsv4, {})
    kv_score = torch.arange(6, dtype=torch.float32).reshape(3, 2)
    positions = torch.tensor([0, 3, 7], dtype=torch.long)
    out = torch.empty(3, 1, 2)

    attn._prepare_and_attn(
        torch.randn(3, 4),
        torch.randn(3, 2),
        torch.randn(3, 2),
        kv_score,
        None,
        None,
        positions,
        out,
    )

    # Compression runs before MQA with the projection output.
    assert recorded["kv_score"] is kv_score
    assert recorded["positions"] is positions
    assert recorded["mqa_out"] is out
    buffer = attn.topk_indices_buffer
    assert buffer[0].tolist() == [-1, -1, -1]
    assert buffer[1].tolist() == [0, -1, -1]
    assert buffer[2].tolist() == [0, 1, -1]


def test_prepare_compressor_without_c4a_skips_index_buffer(dsv4, monkeypatch):
    attn, recorded = _prepare_attn(dsv4, compress_ratio=1)
    _context(monkeypatch, dsv4, {})
    kv_score = torch.arange(6, dtype=torch.float32).reshape(3, 2)
    positions = torch.tensor([0, 1, 2], dtype=torch.long)

    attn._prepare_and_attn(
        torch.randn(3, 4),
        torch.randn(3, 2),
        torch.randn(3, 2),
        kv_score,
        None,
        None,
        positions,
        torch.empty(3, 1, 2),
    )

    assert recorded["kv_score"] is kv_score
    assert recorded["positions"] is positions
    # Ratio 1 does not populate the C4A topk buffer.
    assert torch.all(attn.topk_indices_buffer == 77)
