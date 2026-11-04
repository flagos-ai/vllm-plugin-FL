# Copyright (c) 2026 BAAI. All rights reserved.

"""Correctness checks for the Hygon FlashMLA DeepSeek-V4 sparse attention.

Run with::

    pytest -q -s tests/unit_tests/dispatch/test_hygon_sparse_attn_flash_mla.py

Covers:
- the FlashMLA decode adapter (SWA-only and SWA + compressed-topk) against a
  PyTorch reference and against vLLM's Triton ragged kernel,
- the FlashMLA prefill adapter on the bf16 gathered workspace,
- the dense C4A topk mapping kernel,
- patch routing plus the ``VLLM_FL_SPARSE_ATTN_IMPL=triton`` escape hatch.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

_HEAD_DIM = 512
_NOPE_DIM = 448
_SCALE = _HEAD_DIM**-0.5


@pytest.fixture(scope="module")
def flash_impl():
    if not torch.cuda.is_available():
        pytest.skip("Hygon/HIP GPU is required")
    from vllm_fl.dispatch.backends.vendor.hygon.impl.attention import (
        sparse_attn_flash_mla as impl,
    )

    if impl._resolve_flash_mla() is None:
        pytest.skip("Hygon FlashMLA wheel is required")
    return impl


@pytest.fixture(scope="module")
def triton_ops():
    from vllm.v1.attention.ops import rocm_aiter_mla_sparse

    return rocm_aiter_mla_sparse


def _randn(*shape: int, dtype=torch.bfloat16) -> torch.Tensor:
    return torch.randn(*shape, device="cuda", dtype=dtype)


def _make_ocp_fp8_ds_mla_cache(
    rows: torch.Tensor,
    block_size: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Encode rows in the physical 576B data + 8B scale page layout (scale=1)."""
    assert rows.shape[-1] == _HEAD_DIM and rows.dtype == torch.bfloat16
    num_rows = rows.shape[0]
    num_blocks = (num_rows + block_size - 1) // block_size
    cache = torch.zeros(
        num_blocks, block_size, 584, device=rows.device, dtype=torch.uint8
    )
    nope_fp8 = rows[:, :_NOPE_DIM].to(torch.float8_e4m3fn)
    dequantized = torch.cat(
        (nope_fp8.to(torch.bfloat16), rows[:, _NOPE_DIM:]), dim=-1
    )
    for row in range(num_rows):
        block_idx, position = divmod(row, block_size)
        block = cache[block_idx].view(-1)
        data_offset = position * 576
        block[data_offset : data_offset + _NOPE_DIM].copy_(
            nope_fp8[row].view(torch.uint8)
        )
        block[data_offset + _NOPE_DIM : data_offset + 576].view(torch.bfloat16).copy_(
            rows[row, _NOPE_DIM:]
        )
        scale_offset = block_size * 576 + position * 8
        block[scale_offset : scale_offset + 7] = 127
    return cache, dequantized


def _sparse_attention_reference(
    q: torch.Tensor,
    keys_per_query: list[torch.Tensor],
    scale: float,
    attn_sink: torch.Tensor | None,
) -> torch.Tensor:
    output = torch.empty_like(q, dtype=torch.bfloat16)
    for row, keys in enumerate(keys_per_query):
        q_row = q[row].float()
        keys_f = keys.float()
        scores = (q_row @ keys_f.transpose(0, 1)) * scale
        if attn_sink is None:
            probs = torch.softmax(scores, dim=-1)
        else:
            sink = attn_sink[: q.shape[1]].float().unsqueeze(-1)
            probs = torch.softmax(torch.cat((scores, sink), dim=-1), dim=-1)[
                :, : keys.shape[0]
            ]
        output[row] = (probs @ keys_f).to(torch.bfloat16)
    return output


def _assert_close(name: str, actual: torch.Tensor, expected: torch.Tensor, atol=0.04):
    diff = (actual.float() - expected.float()).abs()
    assert not torch.isnan(actual).any(), f"{name}: NaN output"
    assert not torch.isinf(actual).any(), f"{name}: Inf output"
    max_abs = diff.max().item()
    mean_abs = diff.mean().item()
    print(
        f"{name}: shape={tuple(actual.shape)} max_abs={max_abs:.6g} "
        f"mean_abs={mean_abs:.6g}"
    )
    assert torch.allclose(actual.float(), expected.float(), atol=atol, rtol=0.02), (
        f"{name}: max_abs={max_abs} mean_abs={mean_abs}"
    )


def _run_triton_decode(triton_ops, **kw):
    out = torch.empty_like(kw["q"])
    triton_ops.rocm_sparse_attn_decode(
        q=kw["q"],
        kv_cache=kw["kv_cache"],
        swa_k_cache=kw["swa_k_cache"],
        swa_only=kw["swa_only"],
        topk_indices=kw["topk_indices"],
        topk_lens=kw["topk_lens"],
        swa_indices=kw["swa_indices"],
        swa_lens=kw["swa_lens"],
        swa_ragged_indices=None,
        swa_ragged_indptr=None,
        topk_ragged_indices=None,
        topk_ragged_indptr=None,
        attn_sink=kw["attn_sink"],
        scale=kw["scale"],
        head_dim=_HEAD_DIM,
        nope_head_dim=_NOPE_DIM,
        rope_head_dim=64,
        output=out,
    )
    return out


@pytest.mark.parametrize("with_extra", [False, True])
@pytest.mark.parametrize("with_sink", [False, True])
def test_decode_flash_mla_matches_torch_and_triton(
    flash_impl, triton_ops, with_extra, with_sink
):
    torch.manual_seed(70 + int(with_extra) * 2 + int(with_sink))
    batch, heads = 4, 8
    # Unaligned widths exercise the 128-alignment padding path.
    main_rows = _randn(256, _HEAD_DIM) * 0.125
    extra_rows = _randn(128, _HEAD_DIM) * 0.125
    main_cache, main_dequant = _make_ocp_fp8_ds_mla_cache(main_rows, 64)
    extra_cache, extra_dequant = _make_ocp_fp8_ds_mla_cache(extra_rows, 64)
    q = _randn(batch, heads, _HEAD_DIM) * 0.125

    main_width, extra_width = 96, 80
    main_lens = torch.tensor([96, 64, 48, 96], device="cuda", dtype=torch.int32)
    extra_lens = torch.tensor([80, 40, 20, 80], device="cuda", dtype=torch.int32)
    main_indices = torch.full((batch, main_width), -1, device="cuda", dtype=torch.int32)
    extra_indices = torch.full(
        (batch, extra_width), -1, device="cuda", dtype=torch.int32
    )
    for row in range(batch):
        main_indices[row, : int(main_lens[row])] = torch.randperm(
            main_rows.shape[0], device="cuda"
        )[: int(main_lens[row])].to(torch.int32)
        extra_indices[row, : int(extra_lens[row])] = torch.randperm(
            extra_rows.shape[0], device="cuda"
        )[: int(extra_lens[row])].to(torch.int32)

    attn_sink = (
        torch.randn(heads, device="cuda", dtype=torch.float32) * 0.1 if with_sink else None
    )
    kwargs = dict(
        q=q,
        kv_cache=extra_cache if with_extra else None,
        swa_k_cache=main_cache,
        swa_only=not with_extra,
        topk_indices=extra_indices if with_extra else None,
        topk_lens=extra_lens if with_extra else None,
        swa_indices=main_indices,
        swa_lens=main_lens,
        attn_sink=attn_sink,
        scale=_SCALE,
    )

    actual = torch.empty_like(q)
    flash_impl.sparse_attn_decode_flash_mla(
        head_dim=_HEAD_DIM,
        nope_head_dim=_NOPE_DIM,
        rope_head_dim=64,
        output=actual,
        **kwargs,
    )

    keys_per_query = []
    for row in range(batch):
        keys = main_dequant[
            main_indices[row, : int(main_lens[row])].long()
        ]
        if with_extra:
            keys = torch.cat(
                (keys, extra_dequant[extra_indices[row, : int(extra_lens[row])].long()]),
                dim=0,
            )
        keys_per_query.append(keys)
    expected = _sparse_attention_reference(q, keys_per_query, _SCALE, attn_sink)
    _assert_close(f"flash_mla/decode(extra={with_extra},sink={with_sink})", actual, expected)
    _assert_close(
        f"flash_mla_vs_triton/decode(extra={with_extra},sink={with_sink})",
        actual,
        _run_triton_decode(triton_ops, **kwargs),
    )


@pytest.mark.parametrize("with_sink", [False, True])
def test_prefill_flash_mla_matches_torch_and_triton(
    flash_impl, triton_ops, with_sink
):
    torch.manual_seed(303)
    tokens, heads, num_kv, topk = 128, 8, 1024, 640
    kv_rows = _randn(num_kv, _HEAD_DIM) * 0.125
    q = _randn(tokens, heads, _HEAD_DIM) * 0.125
    indices = torch.stack(
        [torch.randperm(num_kv, device="cuda")[:topk] for _ in range(tokens)]
    ).to(torch.int32)
    lens = torch.full((tokens,), topk, device="cuda", dtype=torch.int32)
    lens[::3] = topk - 128  # ragged lengths
    attn_sink = (
        torch.randn(heads, device="cuda", dtype=torch.float32) * 0.1 if with_sink else None
    )

    actual = torch.empty_like(q)
    flash_impl.sparse_attn_prefill_flash_mla(
        q=q,
        kv=kv_rows.view(num_kv, 1, _HEAD_DIM),
        indices=indices,
        topk_length=lens,
        scale=_SCALE,
        head_dim=_HEAD_DIM,
        nope_head_dim=_NOPE_DIM,
        rope_head_dim=64,
        attn_sink=attn_sink,
        output=actual,
    )

    keys_per_query = [
        kv_rows[indices[t, : int(lens[t])].long()] for t in range(tokens)
    ]
    expected = _sparse_attention_reference(q, keys_per_query, _SCALE, attn_sink)
    _assert_close(f"flash_mla/prefill(sink={with_sink})", actual, expected)

    triton_out = torch.empty_like(q)
    triton_ops.rocm_sparse_attn_prefill(
        q=q,
        kv=kv_rows.view(num_kv, 1, _HEAD_DIM),
        indices=indices,
        topk_length=lens,
        scale=_SCALE,
        head_dim=_HEAD_DIM,
        nope_head_dim=_NOPE_DIM,
        rope_head_dim=64,
        attn_sink=attn_sink,
        output=triton_out,
    )
    _assert_close(f"flash_mla_vs_triton/prefill(sink={with_sink})", actual, triton_out)


def test_compute_global_topk_indices_and_lens(flash_impl):
    torch.manual_seed(404)
    num_tokens, topk, block_size, num_reqs = 32, 512, 64, 4
    num_blocks_per_req = 8
    local = torch.randint(0, num_blocks_per_req * block_size, (num_tokens, topk), device="cuda", dtype=torch.int32)
    # Inject invalid entries (-1) like the indexer's padding.
    local[:, -7:] = -1
    token_to_req = torch.arange(num_tokens, device="cuda", dtype=torch.int32) % num_reqs
    block_table = torch.randint(
        0, 64, (num_reqs, num_blocks_per_req), device="cuda", dtype=torch.int32
    )
    is_valid = torch.ones(num_tokens, device="cuda", dtype=torch.bool)
    is_valid[::5] = False

    global_idx, lens = flash_impl.compute_global_topk_indices_and_lens(
        local, token_to_req, block_table, block_size, is_valid
    )

    for t in range(num_tokens):
        if not bool(is_valid[t]):
            assert int(lens[t]) == 0
            assert bool((global_idx[t] == -1).all())
            continue
        expected = []
        for k in range(topk):
            entry = int(local[t, k])
            if entry < 0:
                expected.append(-1)
                continue
            blk, off = divmod(entry, block_size)
            expected.append(int(block_table[int(token_to_req[t]), blk]) * block_size + off)
        assert int(lens[t]) == sum(e != -1 for e in expected)
        assert global_idx[t].tolist() == expected


def test_patch_routes_decode_and_env_switch(flash_impl, monkeypatch):
    from vllm_fl.dispatch.backends.vendor.hygon.patches.models import deepseek_v4 as patches

    patches.patch_sparse_attn_flash_mla()

    import vllm.models.deepseek_v4.amd.rocm as rocm_module

    assert getattr(rocm_module.rocm_sparse_attn_prefill, "_vllm_fl_hygon", False)
    assert getattr(rocm_module.DeepseekV4ROCMAiterMLAAttention._forward_decode, "_vllm_fl_hygon", False)

    torch.manual_seed(505)
    batch, heads = 2, 8
    main_rows = _randn(128, _HEAD_DIM) * 0.125
    main_cache, main_dequant = _make_ocp_fp8_ds_mla_cache(main_rows, 64)
    q = _randn(batch, heads, _HEAD_DIM) * 0.125
    main_width = 64
    main_indices = torch.stack(
        [torch.randperm(main_rows.shape[0], device="cuda")[:main_width] for _ in range(batch)]
    ).to(torch.int32)
    main_lens = torch.full((batch,), main_width, device="cuda", dtype=torch.int32)

    layer = SimpleNamespace(
        compress_ratio=1,
        swa_cache_layer=SimpleNamespace(kv_cache=main_cache),
        topk_indices_buffer=None,
        attn_sink=None,
        scale=_SCALE,
        head_dim=_HEAD_DIM,
        nope_head_dim=_NOPE_DIM,
        rope_head_dim=64,
    )
    swa_metadata = SimpleNamespace(
        num_decodes=batch,
        num_decode_tokens=batch,
        is_valid_token=None,
        decode_swa_indices=main_indices.view(batch, 1, main_width),
        decode_swa_lens=main_lens,
        decode_swa_ragged_indices=None,
        decode_swa_ragged_indptr=None,
    )

    calls = {"adapter": 0}
    original_adapter = flash_impl.sparse_attn_decode_flash_mla

    def _record_adapter(**kwargs):
        calls["adapter"] += 1
        original_adapter(**kwargs)

    monkeypatch.setattr(flash_impl, "sparse_attn_decode_flash_mla", _record_adapter)

    output = torch.empty_like(q)
    rocm_module.DeepseekV4ROCMAiterMLAAttention._forward_decode(
        layer, q, None, swa_metadata, None, True, output
    )
    assert calls["adapter"] == 1, "flash_mla decode adapter must be routed by default"

    keys_per_query = [main_dequant[main_indices[b].long()] for b in range(batch)]
    expected = _sparse_attention_reference(q, keys_per_query, _SCALE, None)
    _assert_close("patched_route/decode", output, expected)

    monkeypatch.setenv("VLLM_FL_SPARSE_ATTN_IMPL", "triton")
    try:
        output2 = torch.empty_like(q)
        rocm_module.DeepseekV4ROCMAiterMLAAttention._forward_decode(
            layer, q, None, swa_metadata, None, True, output2
        )
        assert calls["adapter"] == 1, "triton mode must bypass the flash_mla adapter"
        _assert_close("triton_fallback/decode", output2, expected)
    finally:
        monkeypatch.delenv("VLLM_FL_SPARSE_ATTN_IMPL")


def test_decode_adapter_tolerates_unaligned_cache(flash_impl):
    """FlashMLA does not require 576B base alignment (verified empirically)."""
    torch.manual_seed(606)
    rows = _randn(128, _HEAD_DIM) * 0.125
    cache, dequant = _make_ocp_fp8_ds_mla_cache(rows, 64)
    flat = cache.view(-1)
    offset = 8 if flat.data_ptr() % 576 == 0 else 0  # force misalignment when possible
    if offset:
        cache = flat[offset : offset + flat.numel()].view_as(cache)
    print(f"cache data_ptr % 576 = {cache.data_ptr() % 576}")

    q = _randn(2, 8, _HEAD_DIM) * 0.125
    width = 64
    indices = torch.stack(
        [torch.randperm(rows.shape[0], device="cuda")[:width] for _ in range(2)]
    ).to(torch.int32)
    lens = torch.full((2,), width, device="cuda", dtype=torch.int32)

    actual = torch.empty_like(q)
    flash_impl.sparse_attn_decode_flash_mla(
        q=q,
        kv_cache=None,
        swa_k_cache=cache,
        swa_only=True,
        topk_indices=None,
        topk_lens=None,
        swa_indices=indices,
        swa_lens=lens,
        attn_sink=None,
        scale=_SCALE,
        head_dim=_HEAD_DIM,
        nope_head_dim=_NOPE_DIM,
        rope_head_dim=64,
        output=actual,
    )
    expected = _sparse_attention_reference(
        q, [dequant[indices[b].long()] for b in range(2)], _SCALE, None
    )
    _assert_close("flash_mla/decode(unaligned cache)", actual, expected)
