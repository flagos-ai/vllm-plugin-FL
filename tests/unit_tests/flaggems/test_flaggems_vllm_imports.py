# Copyright (c) 2026 BAAI. All rights reserved.

"""
Test cases for flaggems_vllm import substitution.

Verifies that all 'from flaggems_vllm.ops.xxx import yyy' used in the codebase
are importable and return callable objects.

Run: python3 tests/unit_tests/flaggems/test_flaggems_vllm_imports.py
"""


def test_silu_and_mul():
    from flaggems_vllm.ops.silu_and_mul import silu_and_mul

    assert callable(silu_and_mul)


def test_gelu_and_mul():
    from flaggems_vllm.ops.gelu_and_mul import gelu_and_mul

    assert callable(gelu_and_mul)


def test_silu_and_mul_with_clamp():
    from flaggems_vllm.ops.silu_and_mul_with_clamp import silu_and_mul_with_clamp_kernel

    assert callable(silu_and_mul_with_clamp_kernel)


def test_flash_attn_varlen_func():
    from flaggems_vllm.ops.attention import flash_attn_varlen_func

    assert callable(flash_attn_varlen_func)


def test_reshape_and_cache_flash():
    from flaggems_vllm.ops.reshape_and_cache_flash import reshape_and_cache_flash

    assert callable(reshape_and_cache_flash)


def test_flash_mla():
    from flaggems_vllm.ops.flash_mla import flash_mla

    assert callable(flash_mla)


def test_moe_align_block_size_triton():
    from flaggems_vllm.ops.moe_align_block_size import moe_align_block_size_triton

    assert callable(moe_align_block_size_triton)


def test_topk_softmax():
    from flaggems_vllm.ops.topk_softmax import topk_softmax

    assert callable(topk_softmax)


def test_topk_softplus_sqrt():
    from flaggems_vllm.ops.topk_softplus_sqrt import topk_softplus_sqrt

    assert callable(topk_softplus_sqrt)


def test_invoke_fused_moe_triton_kernel():
    from flaggems_vllm.ops.fused_moe import invoke_fused_moe_triton_kernel

    assert callable(invoke_fused_moe_triton_kernel)


def test_grouped_topk():
    from flaggems_vllm.ops.grouped_topk import grouped_topk

    assert callable(grouped_topk)


def test_moe_sum():
    from flaggems_vllm.ops.moe_sum import moe_sum

    assert callable(moe_sum)


def test_router_gemm():
    from flaggems_vllm.ops.router_gemm import router_gemm

    assert callable(router_gemm)


def test_mhc_pre():
    from flaggems_vllm.ops.mhc.mhc_pre import mhc_pre

    assert callable(mhc_pre)


def test_mhc_post():
    from flaggems_vllm.ops.mhc.mhc_post import mhc_post

    assert callable(mhc_post)


def test_hc_head_fused_kernel():
    from flaggems_vllm.ops.mhc.hc_head_fused_kernel import hc_head_fused_kernel

    assert callable(hc_head_fused_kernel)


def test_rms_norm():
    # rms_norm_forward is not available in flaggems_vllm,
    # always falls back to flag_gems.rms_norm_forward
    from flag_gems import rms_norm_forward

    assert callable(rms_norm_forward)


def test_rope():
    from flaggems_vllm.ops.rotary_embedding import apply_rotary_pos_emb

    assert callable(apply_rotary_pos_emb)


def test_fused_experts_impl():
    import flaggems_vllm

    assert callable(flaggems_vllm.fused_experts_impl)


if __name__ == "__main__":
    tests = [v for k, v in globals().items() if k.startswith("test_")]
    passed = 0
    failed = 0
    for t in tests:
        try:
            t()
            print(f"✅ {t.__name__}")
            passed += 1
        except Exception as e:
            print(f"❌ {t.__name__}: {e}")
            failed += 1
    print(f"\n{passed} passed, {failed} failed")
