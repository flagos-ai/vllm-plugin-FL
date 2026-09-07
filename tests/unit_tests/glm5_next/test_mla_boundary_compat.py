# SPDX-License-Identifier: Apache-2.0

import sys
from types import ModuleType

import pytest
import torch

from vllm_fl.patches import glm5_next_v024 as glm5_patch

_install_mla_boundary_compat_ops = glm5_patch._install_mla_boundary_compat_ops


def _fake_ops(*, cache_impl):
    module = ModuleType("fake_vllm_custom_ops")

    def concat_mla_q(q_nope, q_pe, output):
        output.copy_(torch.cat((q_nope, q_pe), dim=-1))

    module.concat_mla_q = concat_mla_q
    module.concat_and_cache_mla = cache_impl
    return module


def test_mla_boundary_compat_keeps_vendor_fast_paths(monkeypatch) -> None:
    monkeypatch.setattr(glm5_patch, "_has_vllm_cache_op", lambda name: False)
    calls = {"cache": 0, "q": 0}

    def vendor_cache(kv_c, k_pe, cache, slots, cache_dtype, scale):
        del k_pe, slots, cache_dtype, scale
        calls["cache"] += 1
        cache.view(-1, cache.shape[-1])[: kv_c.shape[0]].copy_(kv_c)

    ops = _fake_ops(cache_impl=vendor_cache)
    original_q = ops.concat_mla_q

    def counted_q(*args):
        calls["q"] += 1
        return original_q(*args)

    ops.concat_mla_q = counted_q
    assert _install_mla_boundary_compat_ops(ops)
    assert not _install_mla_boundary_compat_ops(ops)

    q_nope = torch.randn(2, 3, 4)
    q_pe_empty = torch.empty(2, 3, 0)
    q_out = torch.empty_like(q_nope)
    ops.concat_mla_q(q_nope, q_pe_empty, q_out)
    torch.testing.assert_close(q_out, q_nope)
    assert calls["q"] == 0

    q_pe = torch.randn(2, 3, 2)
    q_out_with_pe = torch.empty(2, 3, 6)
    ops.concat_mla_q(q_nope, q_pe, q_out_with_pe)
    torch.testing.assert_close(q_out_with_pe, torch.cat((q_nope, q_pe), -1))
    assert calls["q"] == 1

    kv_c = torch.randn(3, 4)
    cache = torch.zeros(2, 4, 4)
    ops.concat_and_cache_mla(
        kv_c,
        torch.empty(3, 0),
        cache,
        torch.arange(3),
        "auto",
        torch.ones(1),
    )
    assert calls["cache"] == 1
    torch.testing.assert_close(cache.view(-1, 4)[:3], kv_c)


def test_mla_cache_missing_vendor_op_uses_bf16_slot_fallback(
    monkeypatch,
) -> None:
    monkeypatch.setattr(glm5_patch, "_has_vllm_cache_op", lambda name: False)
    calls = {"cache": 0}

    def missing_vendor_cache(*args, **kwargs):
        del args, kwargs
        calls["cache"] += 1
        raise AttributeError(
            "'_OpNamespace' '_C_cache_ops' object has no attribute "
            "'concat_and_cache_mla'"
        )

    # Make the test independent of whether its host environment has FlagGems.
    monkeypatch.setitem(sys.modules, "flag_gems.fused.concat_and_cache_mla", None)
    ops = _fake_ops(cache_impl=missing_vendor_cache)
    _install_mla_boundary_compat_ops(ops)

    kv_c = torch.arange(12, dtype=torch.float32).view(4, 3)
    slots = torch.tensor([0, 3, -1, 7])
    cache = torch.zeros(2, 4, 3)
    empty_pe = torch.empty(4, 0)
    scale = torch.ones(1)

    ops.concat_and_cache_mla(kv_c, empty_pe, cache, slots, "bfloat16", scale)
    torch.testing.assert_close(cache.view(-1, 3)[slots[:2]], kv_c[:2])
    torch.testing.assert_close(cache.view(-1, 3)[slots[3:]], kv_c[3:])

    # The missing native ABI is remembered, so subsequent calls do not pay
    # for another exception before taking the fallback.
    cache.zero_()
    ops.concat_and_cache_mla(kv_c, empty_pe, cache, slots, "auto", scale)
    assert calls["cache"] == 1

    with pytest.raises(NotImplementedError, match="only supports BF16"):
        ops.concat_and_cache_mla(kv_c, empty_pe, cache, slots, "fp8_ds_mla", scale)


def test_mla_cache_does_not_hide_unrelated_vendor_errors(monkeypatch) -> None:
    monkeypatch.setattr(glm5_patch, "_has_vllm_cache_op", lambda name: False)

    def broken_vendor_cache(*args, **kwargs):
        del args, kwargs
        raise AttributeError("vendor metadata is missing")

    ops = _fake_ops(cache_impl=broken_vendor_cache)
    _install_mla_boundary_compat_ops(ops)

    with pytest.raises(AttributeError, match="vendor metadata"):
        ops.concat_and_cache_mla(
            torch.randn(1, 3),
            torch.empty(1, 0),
            torch.zeros(1, 1, 3),
            torch.zeros(1, dtype=torch.int64),
            "auto",
            torch.ones(1),
        )


def test_mla_cache_wrapper_is_not_installed_when_vendor_abi_exists(
    monkeypatch,
) -> None:
    monkeypatch.setattr(glm5_patch, "_has_vllm_cache_op", lambda name: True)

    def vendor_cache(*args, **kwargs):
        del args, kwargs

    ops = _fake_ops(cache_impl=vendor_cache)
    _install_mla_boundary_compat_ops(ops)

    assert ops.concat_and_cache_mla is vendor_cache


def test_explicit_flaggems_rejects_accidental_vendor_attention(
    monkeypatch,
) -> None:
    monkeypatch.setattr(glm5_patch, "get_glm5_provider", lambda: "flaggems")

    def vendor_cache(*args, **kwargs):
        del args, kwargs

    ops = _fake_ops(cache_impl=vendor_cache)
    _install_mla_boundary_compat_ops(ops)

    with pytest.raises(RuntimeError, match="did not select FlagGemsSparseMLABackend"):
        ops.concat_mla_q(
            torch.randn(1, 2, 3),
            torch.empty(1, 2, 0),
            torch.empty(1, 2, 3),
        )
