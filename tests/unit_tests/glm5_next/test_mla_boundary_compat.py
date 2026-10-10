# SPDX-License-Identifier: Apache-2.0
"""Initialize MLA compatibility from native capabilities, before state writes.

Public library operators are explicitly stubbed here; numerical correctness is
covered by the operator-library tests. These tests exercise real patch binding.
"""

from types import ModuleType

import pytest
import torch

from vllm_fl.kernels.glm5_next import indexer_backend
from vllm_fl.patches import glm5_next_runtime as glm5_patch

_install_mla_boundary_compat_ops = glm5_patch._install_mla_boundary_compat_ops


@pytest.fixture(autouse=True)
def clean_worker_patch_records(monkeypatch):
    from vllm_fl.activation import reset_activation_for_tests

    reset_activation_for_tests()
    glm5_patch._concat_handles_zero_rope.cache_clear()
    monkeypatch.setattr(glm5_patch.current_platform, "device_type", "cpu")
    yield
    reset_activation_for_tests()
    glm5_patch._concat_handles_zero_rope.cache_clear()


def _fake_ops(*, cache_impl, query_impl=None):
    module = ModuleType("fake_vllm_custom_ops")

    def concat_mla_q(q_nope, q_pe, output):
        output.copy_(torch.cat((q_nope, q_pe), dim=-1))

    module.concat_mla_q = query_impl or concat_mla_q
    module.concat_and_cache_mla = cache_impl
    return module


def _public_ops(monkeypatch, *, query=None, writer=None):
    ops = {"concat_mla_q": query, "concat_and_cache_mla": writer}
    requested = []

    def load(module, name):
        assert module == name
        requested.append(name)
        return ops[name]

    monkeypatch.setattr(indexer_backend, "_load_flaggems_op", load)
    return requested


def test_supported_native_query_and_cache_keep_identity(monkeypatch):
    monkeypatch.setattr(glm5_patch, "_has_vllm_cache_op", lambda name: True)
    calls = []

    def native_query(q, rope, out):
        calls.append("query")
        out.copy_(torch.cat((q, rope), dim=-1))

    def native_cache(*args):
        calls.append("cache")

    ops = _fake_ops(cache_impl=native_cache, query_impl=native_query)
    requested = _public_ops(monkeypatch)
    assert not _install_mla_boundary_compat_ops(ops)
    assert not _install_mla_boundary_compat_ops(ops)
    assert ops.concat_mla_q is native_query
    assert ops.concat_and_cache_mla is native_cache
    assert calls == ["query"]  # stable capability is probed once
    assert not requested
    q = torch.randn(2, 3, 4)
    for rope_width in (0, 2):
        rope = torch.randn(2, 3, rope_width)
        out = torch.empty(2, 3, 4 + rope_width)
        ops.concat_mla_q(q, rope, out)
        torch.testing.assert_close(out, torch.cat((q, rope), -1))
    assert calls == ["query", "query", "query"]


def test_zero_rope_guard_preserves_positive_rope_native_path(monkeypatch):
    monkeypatch.setattr(glm5_patch, "_has_vllm_cache_op", lambda name: True)
    calls = []

    def native_query(q, rope, out):
        calls.append("native")
        if rope.shape[-1]:
            out.copy_(torch.cat((q, rope), -1))

    def public_query(q, rope, out):
        calls.append("public")
        out.copy_(torch.cat((q, rope), -1))

    ops = _fake_ops(cache_impl=lambda *args: None, query_impl=native_query)
    requested = _public_ops(monkeypatch, query=public_query)
    assert _install_mla_boundary_compat_ops(ops)
    assert not _install_mla_boundary_compat_ops(ops)
    assert requested == ["concat_mla_q"]
    assert calls == ["native"]
    q = torch.randn(2, 3, 4)
    # Start with the valid native condition, then change only the RoPE width.
    for rope_width, implementation in ((2, "native"), (0, "public")):
        rope = torch.randn(2, 3, rope_width)
        out = torch.empty(2, 3, 4 + rope_width)
        before = len(calls)
        ops.concat_mla_q(q, rope, out)
        assert calls[before:] == [implementation]
        torch.testing.assert_close(out, torch.cat((q, rope), -1))


def test_missing_cache_abi_selects_public_writer_before_native_call(monkeypatch):
    monkeypatch.setattr(glm5_patch, "_has_vllm_cache_op", lambda name: False)
    calls = []

    def missing_native(*args):
        calls.append("native")
        pytest.fail("missing ABI was executed")

    def writer(kv, rope, cache, slots, cache_dtype, scale):
        calls.append(cache_dtype)
        assert rope.shape[-1] == 0
        valid = slots >= 0
        cache.view(-1, kv.shape[-1])[slots[valid]] = kv[valid]

    ops = _fake_ops(cache_impl=missing_native)
    requested = _public_ops(monkeypatch, writer=writer)
    assert _install_mla_boundary_compat_ops(ops)
    assert not _install_mla_boundary_compat_ops(ops)
    assert requested == ["concat_and_cache_mla"]
    kv = torch.arange(12, dtype=torch.float32).view(4, 3)
    slots = torch.tensor([0, 3, -1, 7])
    for dtype in ("bfloat16", "auto"):
        cache = torch.zeros(2, 4, 3)
        ops.concat_and_cache_mla(
            kv, torch.empty(4, 0), cache, slots, dtype, torch.ones(1)
        )
        torch.testing.assert_close(cache.view(-1, 3)[slots[slots >= 0]], kv[slots >= 0])
    assert calls == ["auto", "auto"]


@pytest.mark.parametrize(
    "error_type", [RuntimeError, NotImplementedError, torch.OutOfMemoryError]
)
def test_writer_failure_after_mutation_is_not_retried(monkeypatch, error_type):
    monkeypatch.setattr(glm5_patch, "_has_vllm_cache_op", lambda name: False)
    calls = []
    error = error_type("writer failed after writing")

    def writer(kv, rope, cache, slots, cache_dtype, scale):
        calls.append("public")
        cache.view(-1, kv.shape[-1])[slots] = 1
        raise error

    ops = _fake_ops(cache_impl=lambda *args: calls.append("native"))
    _public_ops(monkeypatch, writer=writer)
    _install_mla_boundary_compat_ops(ops)
    cache = torch.zeros(2, 4, 3)
    with pytest.raises(error_type) as caught:
        ops.concat_and_cache_mla(
            torch.zeros(2, 3),
            torch.empty(2, 0),
            cache,
            torch.tensor([0, 1]),
            "auto",
            torch.ones(1),
        )
    assert caught.value is error
    assert calls == ["public"]
    assert torch.equal(cache.view(-1, 3)[:2], torch.ones(2, 3))


def test_unrelated_native_error_propagates_when_abi_exists(monkeypatch):
    monkeypatch.setattr(glm5_patch, "_has_vllm_cache_op", lambda name: True)
    calls = []

    def broken_native(*args):
        calls.append("native")
        raise AttributeError("vendor metadata is missing")

    ops = _fake_ops(cache_impl=broken_native)
    requested = _public_ops(monkeypatch)
    assert not _install_mla_boundary_compat_ops(ops)
    with pytest.raises(AttributeError, match="vendor metadata"):
        ops.concat_and_cache_mla(
            torch.ones(1, 3),
            torch.empty(1, 0),
            torch.zeros(1, 1, 3),
            torch.zeros(1, dtype=torch.int64),
            "auto",
            torch.ones(1),
        )
    assert calls == ["native"]
    assert not requested


def test_missing_public_writer_does_not_partially_install_query_patch(monkeypatch):
    monkeypatch.setattr(glm5_patch, "_has_vllm_cache_op", lambda name: False)

    def no_zero_rope(q, rope, out):
        if rope.shape[-1]:
            out.copy_(torch.cat((q, rope), -1))

    ops = _fake_ops(cache_impl=lambda *args: None, query_impl=no_zero_rope)
    query, cache = ops.concat_mla_q, ops.concat_and_cache_mla
    _public_ops(monkeypatch, query=lambda q, rope, out: out.copy_(q))
    with pytest.raises(RuntimeError, match="requires FlagGems-vllm"):
        _install_mla_boundary_compat_ops(ops)
    assert ops.concat_mla_q is query
    assert ops.concat_and_cache_mla is cache


@pytest.mark.parametrize(
    "guard",
    [
        "rope_dim must be 64, got 0",
        "concat_mla_q, /workspace/csrc/libtorch_stable/cache_kernels.cu:1563, rope_dim must be 64, got 0",
    ],
)
def test_zero_rope_native_shape_guard_selects_public_boundary(monkeypatch, guard):
    monkeypatch.setattr(glm5_patch, "_has_vllm_cache_op", lambda name: True)
    calls = []

    def native_query(q, rope, out):
        calls.append("native")
        if not rope.shape[-1]:
            raise RuntimeError(guard)
        out.copy_(torch.cat((q, rope), -1))

    def public_query(q, rope, out):
        calls.append("public")
        out.copy_(torch.cat((q, rope), -1))

    ops = _fake_ops(cache_impl=lambda *args: None, query_impl=native_query)
    requested = _public_ops(monkeypatch, query=public_query)
    assert _install_mla_boundary_compat_ops(ops)
    assert not _install_mla_boundary_compat_ops(ops)
    assert requested == ["concat_mla_q"]
    assert calls == ["native"]
    q = torch.ones(1, 1, 512, dtype=torch.bfloat16)
    out = torch.empty_like(q)
    ops.concat_mla_q(q, torch.empty(1, 1, 0, dtype=q.dtype), out)
    assert calls == ["native", "public"]
    torch.testing.assert_close(out, q)


@pytest.mark.parametrize(
    "error",
    [
        RuntimeError("kernel launch failed"),
        torch.OutOfMemoryError("CUDA out of memory"),
        RuntimeError("rope_dim must be 64, got 8"),
    ],
)
def test_native_probe_propagates_unrecognized_execution_error(monkeypatch, error):
    monkeypatch.setattr(glm5_patch, "_has_vllm_cache_op", lambda name: True)
    calls = []

    def broken_query(q, rope, out):
        calls.append("native")
        raise error

    ops = _fake_ops(cache_impl=lambda *args: None, query_impl=broken_query)
    original = ops.concat_mla_q
    requested = _public_ops(monkeypatch)
    with pytest.raises(type(error)) as caught:
        _install_mla_boundary_compat_ops(ops)
    assert caught.value is error
    assert ops.concat_mla_q is original
    assert calls == ["native"]
    assert not requested
