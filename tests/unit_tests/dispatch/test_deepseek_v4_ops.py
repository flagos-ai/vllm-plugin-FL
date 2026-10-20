# Copyright (c) 2026 BAAI. All rights reserved.

"""Tests for DeepSeek-V4 operator dispatch."""

from contextlib import nullcontext
from unittest.mock import Mock

import torch

from vllm_fl.dispatch import CachedOp
from vllm_fl.dispatch.backends.reference.impl.deepseek_v4 import (
    deepseek_v4_hc_head_torch,
    deepseek_v4_int8_scaled_mm_torch,
    deepseek_v4_inv_rope_quant_int8_torch,
    deepseek_v4_mhc_post_torch,
)
from vllm_fl.dispatch.types import BackendImplKind
from vllm_fl.ops import deepseek_v4_int8_woa

DSV4_OPS = {
    "deepseek_v4_inv_rope_quant_int8",
    "deepseek_v4_inv_rope_quant_fp8",
    "deepseek_v4_int8_scaled_mm",
    "deepseek_v4_mhc_pre",
    "deepseek_v4_mhc_fused_post_pre",
    "deepseek_v4_mhc_post",
    "deepseek_v4_hc_head",
    "deepseek_v4_fused_q_kv_rmsnorm",
    "deepseek_v4_fused_q_kv_rmsnorm_int8",
    "deepseek_v4_qnorm_rope_kv_quant_insert",
    "deepseek_v4_qnorm_rope_kv_bf16_insert",
    "deepseek_v4_qnorm_rope_kv_fp8_insert",
    "deepseek_v4_compute_global_topk_indices_and_lens",
    "deepseek_v4_flash_mla_with_kvcache",
    "deepseek_v4_dequantize_and_gather_k_cache",
    "deepseek_v4_combine_topk_swa_indices",
    "deepseek_v4_flash_mla_sparse_fwd",
    "deepseek_v4_fused_indexer_q_rope_quant",
    "deepseek_v4_fused_indexer_q_rope_quant_int8",
    "deepseek_v4_compress_int8_indexer_k_cache",
    "deepseek_v4_int8_mqa_logits",
    "deepseek_v4_int8_paged_mqa_logits",
}

DSV4_VENDOR_ONLY_OPS = {
    "deepseek_v4_mhc_pre_int8",
    "deepseek_v4_mhc_fused_post_pre_int8",
}

SPARSE_INDEXER_OPS = {
    "indexer_k_quant_and_cache",
    "cp_gather_indexer_k_quant_cache",
    "top_k_per_row_prefill",
    "top_k_per_row_decode",
    "pack_seq_triton",
    "unpack_seq_triton",
}


def test_model_frontends_only_register_cached_ops_during_import():
    from vllm_fl.ops import deepseek_v4 as deepseek_v4_ops, sparse_attn_indexer

    assert set(deepseek_v4_ops._OPS) == {
        op_name.removeprefix("deepseek_v4_")
        for op_name in DSV4_OPS | DSV4_VENDOR_ONLY_OPS
        if op_name != "deepseek_v4_inv_rope_quant_int8"
    }
    assert all(isinstance(op, CachedOp) for op in deepseek_v4_ops._OPS.values())
    assert isinstance(
        deepseek_v4_int8_woa._dispatch_inv_rope_quant_int8,
        CachedOp,
    )
    assert all(
        isinstance(op, CachedOp)
        for op in (
            sparse_attn_indexer._indexer_k_quant_and_cache,
            sparse_attn_indexer._cp_gather_indexer_k_quant_cache,
            sparse_attn_indexer._top_k_per_row_prefill,
            sparse_attn_indexer._pack_seq_triton,
            sparse_attn_indexer._top_k_per_row_decode,
            sparse_attn_indexer._unpack_seq_triton,
        )
    )


def test_reference_inv_rope_quant_int8():
    o = torch.tensor(
        [[[1, 2, 3, 4], [-1, -2, 5, 6]]],
        dtype=torch.bfloat16,
    )
    positions = torch.tensor([0], dtype=torch.int32)
    cos_sin_cache = torch.tensor([[0, 1]], dtype=torch.float32)

    quantized, scales = deepseek_v4_inv_rope_quant_int8_torch(
        o,
        positions,
        cos_sin_cache,
        n_groups=1,
        heads_per_group=2,
        nope_dim=2,
        rope_dim=2,
    )

    expected = torch.tensor(
        [[[21, 42, 85, -64, -21, -42, 127, -106]]],
        dtype=torch.int8,
    )
    assert torch.equal(quantized, expected)
    torch.testing.assert_close(
        scales,
        torch.tensor([[[6 / 127]]], dtype=torch.float32),
    )


def test_frontend_dispatches_through_cached_op(monkeypatch):
    expected = (Mock(), Mock())
    dispatch = Mock(return_value=expected)
    monkeypatch.setattr(
        deepseek_v4_int8_woa,
        "_dispatch_inv_rope_quant_int8",
        dispatch,
    )
    args = (
        Mock(),
        Mock(),
        Mock(),
        2,
        4,
        64,
        64,
    )

    actual = deepseek_v4_int8_woa.fused_inv_rope_quant_int8(*args)

    assert actual is expected
    dispatch.assert_called_once_with(*args)


def test_all_backends_register_deepseek_v4_op(monkeypatch):
    from vllm_fl.dispatch.backends.flaggems import register_ops as flaggems_ops
    from vllm_fl.dispatch.backends.reference import register_ops as reference_ops
    from vllm_fl.dispatch.backends.vendor.cuda import register_ops as cuda_ops

    registered = []

    class Registry:
        def register_many(self, impls):
            registered.extend(impls)

    monkeypatch.setattr(
        flaggems_ops,
        "use_flaggems_op",
        lambda op_name: op_name == deepseek_v4_int8_woa.DSV4_INV_ROPE_QUANT_INT8_OP,
    )
    registry = Registry()
    flaggems_ops.register_builtins(registry)
    cuda_ops.register_builtins(registry)
    reference_ops.register_builtins(registry)

    implementations = [
        impl
        for impl in registered
        if impl.op_name == deepseek_v4_int8_woa.DSV4_INV_ROPE_QUANT_INT8_OP
    ]
    assert {impl.impl_id for impl in implementations} == {
        "default.flagos",
        "vendor.cuda",
        "reference.torch",
    }
    assert {impl.kind for impl in implementations} == {
        BackendImplKind.DEFAULT,
        BackendImplKind.VENDOR,
        BackendImplKind.REFERENCE,
    }


def test_reference_scaled_mm_and_mhc_ops():
    x_q = torch.tensor([[1, -2]], dtype=torch.int8)
    weight = torch.tensor([[3, 4], [5, 6]], dtype=torch.int8)
    actual = deepseek_v4_int8_scaled_mm_torch(
        x_q,
        weight,
        torch.tensor([[0.5]]),
        torch.tensor([0.25, 0.5]),
        torch.float32,
    )
    torch.testing.assert_close(actual, torch.tensor([[-0.875, -2.0]]))

    residual = torch.tensor([[[1, 2], [3, 4]]], dtype=torch.bfloat16)
    layer = torch.tensor([[2, -1]], dtype=torch.bfloat16)
    post = torch.tensor([[[0.5], [1.0]]], dtype=torch.float32)
    comb = torch.eye(2, dtype=torch.float32).unsqueeze(0)
    torch.testing.assert_close(
        deepseek_v4_mhc_post_torch(layer, residual, post, comb),
        torch.tensor([[[2, 1.5], [5, 3]]], dtype=torch.bfloat16),
    )

    fn = torch.zeros((2, 4), dtype=torch.float32)
    head = deepseek_v4_hc_head_torch(
        residual,
        fn,
        torch.ones(1),
        torch.zeros(2),
        1e-6,
        0.0,
    )
    torch.testing.assert_close(head, residual.float().mean(dim=1).to(torch.bfloat16))


def test_all_backends_register_all_deepseek_v4_ops(monkeypatch):
    from vllm_fl.dispatch.backends.flaggems import register_ops as flaggems_ops
    from vllm_fl.dispatch.backends.reference import register_ops as reference_ops
    from vllm_fl.dispatch.backends.vendor.cuda import register_ops as cuda_ops

    registered = []

    class Registry:
        def register_many(self, impls):
            registered.extend(impls)

    monkeypatch.setattr(
        flaggems_ops,
        "use_flaggems_op",
        lambda op_name: op_name in DSV4_OPS | SPARSE_INDEXER_OPS,
    )
    registry = Registry()
    flaggems_ops.register_builtins(registry)
    cuda_ops.register_builtins(registry)
    reference_ops.register_builtins(registry)

    for op_name in DSV4_OPS:
        implementations = [impl for impl in registered if impl.op_name == op_name]
        assert {impl.impl_id for impl in implementations} == {
            "default.flagos",
            "vendor.cuda",
            "reference.torch",
        }

    for op_name in DSV4_VENDOR_ONLY_OPS:
        implementations = [impl for impl in registered if impl.op_name == op_name]
        assert {impl.impl_id for impl in implementations} == {"vendor.cuda"}

    for op_name in SPARSE_INDEXER_OPS:
        implementations = [impl for impl in registered if impl.op_name == op_name]
        assert {impl.impl_id for impl in implementations} == {
            "default.flagos",
            "vendor.cuda",
            "reference.torch",
        }


def test_flaggems_fused_post_pre_composes_fused_mhc_ops(monkeypatch):
    import flag_gems
    from flag_gems import fused

    from vllm_fl.dispatch.backends.flaggems.impl import deepseek_v4

    x = torch.zeros((2, 8), dtype=torch.bfloat16)
    residual = torch.zeros((2, 4, 8), dtype=torch.bfloat16)
    post_mix = torch.zeros((2, 4, 1), dtype=torch.float32)
    comb_mix = torch.zeros((2, 4, 4), dtype=torch.float32)
    fn = torch.zeros((24, 32), dtype=torch.float32)
    hc_scale = torch.zeros(3, dtype=torch.float32)
    hc_base = torch.zeros(24, dtype=torch.float32)
    norm_weight = torch.ones(8, dtype=torch.bfloat16)

    residual_cur = residual + 1
    post_cur = post_mix + 2
    comb_cur = comb_mix + 2
    layer_input = x + 2

    mhc_post = Mock(return_value=residual_cur)
    mhc_pre = Mock(return_value=(post_cur, comb_cur, layer_input))
    rms_norm = Mock(return_value=layer_input + 3)
    monkeypatch.setattr(
        deepseek_v4.torch_device_fn, "device", lambda _device: nullcontext()
    )
    monkeypatch.setattr(fused, "mhc_post", mhc_post)
    monkeypatch.setattr(fused, "mhc_pre", mhc_pre)
    monkeypatch.setattr(flag_gems, "rms_norm", rms_norm)

    actual = deepseek_v4._flaggems_mhc_fused_post_pre_impl(
        x,
        residual,
        post_mix,
        comb_mix,
        fn,
        hc_scale,
        hc_base,
        1e-6,
        1e-6,
        1e-6,
        2.0,
        20,
        1,
        1,
        norm_weight,
        1e-5,
    )

    assert actual[0] is residual_cur
    assert actual[1] is post_cur
    assert actual[2] is comb_cur
    torch.testing.assert_close(actual[3], layer_input + 3)
    mhc_post.assert_called_once_with(x, residual, post_mix, comb_mix)
    mhc_pre.assert_called_once_with(
        residual_cur,
        fn,
        hc_scale,
        hc_base,
        1e-6,
        1e-6,
        1e-6,
        2.0,
        20,
        1,
    )
    rms_norm.assert_called_once_with(
        layer_input,
        tuple(norm_weight.shape),
        norm_weight,
        1e-5,
    )


def test_sparse_indexer_overrides_upstream_cuda_entrypoint():
    from vllm.model_executor.layers.sparse_attn_indexer import SparseAttnIndexer

    from vllm_fl.ops.sparse_attn_indexer import SparseAttnIndexerFL

    assert SparseAttnIndexerFL.forward_cuda is not SparseAttnIndexer.forward_cuda
