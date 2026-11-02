# SPDX-License-Identifier: Apache-2.0
"""Mla contracts and regression coverage."""

import itertools
from types import ModuleType, SimpleNamespace

import pytest
import torch

from vllm.config import VllmConfig
from vllm.forward_context import set_forward_context
from vllm.v1.attention.backend import AttentionCGSupport
from vllm.v1.attention.backends.mla.indexer import (
    DeepSeekV32IndexerDecodeMetadata,
    DeepseekV32IndexerMetadata,
)
from vllm.v1.kv_cache_interface import MLAAttentionSpec, UniformTypeKVCacheSpecs

from vllm_fl.attention import sparse_mla_metadata
from vllm_fl.attention.sparse_mla_metadata import convert_sparse_indices
from vllm_fl.dispatch import capabilities
from vllm_fl.dispatch.backends.flaggems.impl import mla as mla_sparse
from vllm_fl.dispatch.manager import OpManager
from vllm_fl.dispatch.policy import SelectionPolicy, policy_context
from vllm_fl.models.glm5 import indexer_backend, runtime as glm5_patch
from vllm_fl.models.glm5.sparse_indexer import (
    sparse_attn_indexer_kpool,
)

_install_mla_boundary_compat_ops = glm5_patch._install_mla_boundary_compat_ops


@pytest.fixture(autouse=False)
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


@pytest.mark.usefixtures("clean_worker_patch_records")
def test_missing_query_abi_does_not_initialize_a_device(monkeypatch):
    monkeypatch.setattr(glm5_patch, "_has_vllm_cache_op", lambda name: False)
    monkeypatch.setattr(
        torch, "ones", lambda *a, **kw: pytest.fail("ABI probe allocated a tensor")
    )
    assert not glm5_patch._concat_handles_zero_rope(
        lambda *a: pytest.fail("missing native query was executed")
    )


@pytest.mark.usefixtures("clean_worker_patch_records")
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


@pytest.mark.usefixtures("clean_worker_patch_records")
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


@pytest.mark.usefixtures("clean_worker_patch_records")
def test_missing_cache_abi_selects_public_writer_before_native_call(monkeypatch):
    monkeypatch.setattr(
        glm5_patch, "_has_vllm_cache_op", lambda name: name == "concat_mla_q"
    )
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


@pytest.mark.usefixtures("clean_worker_patch_records")
@pytest.mark.parametrize(
    "error_type", [RuntimeError, NotImplementedError, torch.OutOfMemoryError]
)
def test_writer_failure_after_mutation_is_not_retried(monkeypatch, error_type):
    monkeypatch.setattr(
        glm5_patch, "_has_vllm_cache_op", lambda name: name == "concat_mla_q"
    )
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


@pytest.mark.usefixtures("clean_worker_patch_records")
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


@pytest.mark.usefixtures("clean_worker_patch_records")
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


@pytest.mark.usefixtures("clean_worker_patch_records")
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


@pytest.mark.usefixtures("clean_worker_patch_records")
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


NATIVE_ENTRIES = [
    (
        "per_token_group_quant_fp8",
        "per_token_group_quant_fp8",
        "vllm.model_executor.layers.quantization.utils.fp8_utils",
        "per_token_group_quant_fp8",
    ),
    (
        "indexer_k_quant_and_cache",
        "indexer_k_quant_and_cache",
        "vllm._custom_ops",
        "indexer_k_quant_and_cache",
    ),
    (
        "cp_gather_indexer_k_quant_cache",
        "cp_gather_indexer_k_quant_cache",
        "vllm._custom_ops",
        "cp_gather_indexer_k_quant_cache",
    ),
    (
        "mqa_logits",
        "fp8_fp4_mqa_logits",
        "vllm.utils.deep_gemm",
        "fp8_fp4_mqa_logits",
    ),
    (
        "paged_mqa_logits",
        "fp8_fp4_paged_mqa_logits",
        "vllm.utils.deep_gemm",
        "fp8_fp4_paged_mqa_logits",
    ),
    (
        "topk_prefill",
        "top_k_per_row_prefill",
        "torch.ops._C",
        "top_k_per_row_prefill",
    ),
    (
        "pack_seq",
        "pack_seq_triton",
        "vllm.v1.attention.ops.common",
        "pack_seq_triton",
    ),
    (
        "unpack_seq",
        "unpack_seq_triton",
        "vllm.v1.attention.ops.common",
        "unpack_seq_triton",
    ),
]


@pytest.fixture
def backend_factory(monkeypatch):
    # Native ABI contracts model an FP8-capable NVIDIA device, independently
    # of the runner. Real BF16/device execution is covered below separately.
    monkeypatch.setattr(indexer_backend.current_platform, "supports_fp8", lambda: True)
    monkeypatch.setattr(indexer_backend, "use_nvidia_reference", lambda: True)
    monkeypatch.delenv("VLLM_FL_FLAGOS_WHITELIST", raising=False)
    monkeypatch.delenv("VLLM_FL_FLAGOS_BLACKLIST", raising=False)

    def create(public):
        backend = indexer_backend.Glm5NextIndexerBackend()
        backend._manager = OpManager(register_builtins=False)
        monkeypatch.setattr(backend, "_flag", lambda *args: public)
        return backend

    return create


def _install_native_modules(monkeypatch, modules):
    original = indexer_backend.importlib.import_module

    def load(name, *args, **kwargs):
        if name in modules:
            return modules[name]
        return original(name, *args, **kwargs)

    monkeypatch.setattr(indexer_backend.importlib, "import_module", load)
    if "torch.ops._C" in modules:
        monkeypatch.setattr(torch.ops, "_C", modules["torch.ops._C"])


def _invoke(backend, method, *, preflight=False):
    # Numerical boundaries are stubs; availability and policy are the subject.
    if method == "topk_prefill":
        args = (None,) * 8
    elif method in {"pack_seq", "unpack_seq"}:
        args = (torch.zeros(2, 2), torch.ones(2, dtype=torch.int32))
    else:
        args = (torch.zeros(2, 128),)
    return getattr(backend, method)(*args, _preflight=preflight)


def _vendor_only(op):
    return SelectionPolicy.from_dict(per_op_order={op: ["vendor:cuda"]})


@pytest.mark.parametrize("method,op,module_name,symbol", NATIVE_ENTRIES)
@pytest.mark.parametrize("invalid", ["missing", "not_callable"])
def test_each_native_entry_is_checked_before_execution(
    backend_factory, monkeypatch, method, op, module_name, symbol, invalid
):
    calls = []
    module = ModuleType(module_name)

    def native(*args, **kwargs):
        calls.append("native")
        return "native"

    def public(*args, **kwargs):
        calls.append("library")
        return "library"

    setattr(module, symbol, native)
    _install_native_modules(monkeypatch, {module_name: module})
    with policy_context(_vendor_only(op)):
        valid = backend_factory(public)
        assert _invoke(valid, method, preflight=True)["selected"] == "glm5.cuda"
        assert calls == []
        assert _invoke(valid, method) == "native"
    assert calls == ["native"]

    # Break only the public native symbol before constructing a fresh adapter.
    if invalid == "missing":
        delattr(module, symbol)
    else:
        setattr(module, symbol, object())
    missing = backend_factory(public)
    with policy_context(SelectionPolicy(prefer="vendor")):
        assert _invoke(missing, method, preflight=True)["selected"] == "glm5.flaggems"
        assert calls == ["native"]
        assert _invoke(missing, method) == "library"
    assert calls == ["native", "library"]
    with policy_context(_vendor_only(op)), pytest.raises(RuntimeError, match=op):
        _invoke(missing, method, preflight=True)
    assert calls == ["native", "library"]


@pytest.mark.parametrize("method,op,module_name,symbol", NATIVE_ENTRIES)
@pytest.mark.parametrize(
    "error_type", [NotImplementedError, RuntimeError, torch.OutOfMemoryError]
)
def test_native_execution_error_preserves_object_without_retry(
    backend_factory, monkeypatch, method, op, module_name, symbol, error_type
):
    calls = []
    failure = [None]
    error = error_type("selected native execution failed")
    module = ModuleType(module_name)

    def native(*args, **kwargs):
        calls.append("native")
        if failure[0] is not None:
            raise failure[0]
        return "native"

    def public(*args, **kwargs):
        calls.append("library")
        return "library"

    setattr(module, symbol, native)
    _install_native_modules(monkeypatch, {module_name: module})
    backend = backend_factory(public)
    with policy_context(SelectionPolicy(prefer="vendor")):
        assert _invoke(backend, method) == "native"
        failure[0] = error
        with pytest.raises(error_type) as caught:
            _invoke(backend, method)
    assert caught.value is error
    assert calls == ["native", "native"]
    assert backend._manager.get_failed_impls(op) == {}


def _topk_boundaries(monkeypatch, calls, failure=None):
    native = ModuleType("torch.ops._C")
    workspace = ModuleType("vllm.v1.worker.workspace")

    def ordinary(*args):
        calls.append("ordinary")
        return "ordinary"

    def persistent(*args):
        calls.append("persistent")
        if failure is not None and failure[0] is not None:
            raise failure[0]
        return "persistent"

    def allocate(*args):
        calls.append("allocate")
        return (torch.empty(1, dtype=torch.uint8),)

    def workspace_manager():
        calls.append("workspace_manager")
        return SimpleNamespace(get_simultaneous=allocate)

    native.top_k_per_row_decode = ordinary
    native.persistent_topk = persistent
    workspace.current_workspace_manager = workspace_manager
    _install_native_modules(
        monkeypatch, {"torch.ops._C": native, workspace.__name__: workspace}
    )
    monkeypatch.setattr(indexer_backend.current_platform, "is_cuda", lambda: True)
    return native, workspace


def _decode(backend, *, preflight=False):
    return backend.topk_decode(
        torch.zeros(1, 2050),
        1,
        torch.tensor([[2050]], dtype=torch.int32),
        torch.empty(1, 512, dtype=torch.int32),
        1,
        2050,
        1,
        512,
        max_seq_len=2050,
        _preflight=preflight,
    )


@pytest.mark.parametrize("missing", ["ordinary", "persistent", "workspace_manager"])
def test_decode_native_guards_precede_workspace_allocation(
    backend_factory, monkeypatch, missing
):
    calls = []
    native, workspace = _topk_boundaries(monkeypatch, calls)

    def public(*args):
        calls.append("library")
        return "library"

    with policy_context(_vendor_only("top_k_per_row_decode")):
        valid = backend_factory(public)
        assert _decode(valid, preflight=True)["selected"] == "glm5.cuda"
        assert calls == []
        assert _decode(valid) == "persistent"
    assert calls == ["workspace_manager", "allocate", "persistent"]

    if missing == "workspace_manager":
        del workspace.current_workspace_manager
    else:
        delattr(
            native,
            "top_k_per_row_decode" if missing == "ordinary" else "persistent_topk",
        )
    calls.clear()
    guarded = backend_factory(public)
    if missing == "ordinary":
        with policy_context(SelectionPolicy(prefer="vendor")):
            assert _decode(guarded, preflight=True)["selected"] == "glm5.flaggems"
            assert calls == []
            assert _decode(guarded) == "library"
        assert calls == ["library"]
        with (
            policy_context(_vendor_only("top_k_per_row_decode")),
            pytest.raises(RuntimeError, match="top_k_per_row_decode"),
        ):
            _decode(guarded, preflight=True)
        assert calls == ["library"]
    else:
        with policy_context(_vendor_only("top_k_per_row_decode")):
            assert _decode(guarded, preflight=True)["selected"] == "glm5.cuda"
            assert calls == []
            assert _decode(guarded) == "ordinary"
        assert calls == ["ordinary"]


@pytest.mark.parametrize(
    "error_type", [NotImplementedError, RuntimeError, torch.OutOfMemoryError]
)
def test_persistent_execution_error_never_retries_ordinary_or_library(
    backend_factory, monkeypatch, error_type
):
    calls = []
    failure = [None]
    _topk_boundaries(monkeypatch, calls, failure)
    error = error_type("persistent native execution failed")

    def public(*args):
        calls.append("library")
        return "library"

    backend = backend_factory(public)
    with policy_context(SelectionPolicy(prefer="vendor")):
        assert _decode(backend) == "persistent"
        calls.clear()
        failure[0] = error
        with pytest.raises(error_type) as caught:
            _decode(backend)
    assert caught.value is error
    assert calls == ["workspace_manager", "allocate", "persistent"]
    assert backend._manager.get_failed_impls("top_k_per_row_decode") == {}


def _make_impl(topk_tokens: int = 16) -> "mla_sparse.FlagGemsSparseMLAImpl":
    return mla_sparse.FlagGemsSparseMLAImpl(
        num_heads=2,
        head_size=8,
        scale=0.5,
        num_kv_heads=1,
        alibi_slopes=None,
        sliding_window=None,
        kv_cache_dtype="auto",
        logits_soft_cap=None,
        attn_type="decoder",
        kv_sharing_target_layer_name=None,
        topk_indices_buffer=torch.zeros(topk_tokens, dtype=torch.int32),
        indexer=None,
        kv_lora_rank=8,
    )


def test_sparse_mla_is_an_eager_boundary():
    spec = SimpleNamespace(head_size=576)
    for cls in (
        mla_sparse.FlagGemsSparseMLABackend,
        mla_sparse.FlagGemsSparseMLAMetadataBuilder,
    ):
        assert cls._cudagraph_support is AttentionCGSupport.NEVER
        assert cls.get_cudagraph_support(None, spec) is AttentionCGSupport.NEVER


def test_missing_library_fails_before_execution(monkeypatch):
    monkeypatch.setattr(capabilities, "get_flaggems_vllm_operator", lambda *a: None)
    with pytest.raises(RuntimeError, match="requires FlagGems-vllm"):
        _make_impl()


def test_failure_after_cache_write_propagates_without_second_impl(monkeypatch):
    cache_writes: list[int] = []

    def writer(kv_c, k_pe, kv_cache, slots, *, kv_cache_dtype, scale):
        cache_writes.append(1)
        kv_cache.view(-1, kv_cache.shape[-1])[slots] = 1.0
        raise RuntimeError("kernel failed after writing")

    def fake_flag_op(name):
        if name == "concat_and_cache_mla":
            return writer
        return lambda *a, **k: None

    monkeypatch.setattr(capabilities, "get_flaggems_vllm_operator", fake_flag_op)
    impl = _make_impl()

    kv_cache = torch.zeros(4, 2, 1, 12, dtype=torch.bfloat16)
    kv_c = torch.zeros(3, 8, dtype=torch.bfloat16)
    k_pe = torch.zeros(3, 1, 4, dtype=torch.bfloat16)
    slots = torch.tensor([0, 2, 5], dtype=torch.int64)

    with pytest.raises(RuntimeError, match="after writing"):
        impl.do_kv_cache_update(
            kv_c, k_pe, kv_cache, slots, "auto", torch.ones(1, dtype=torch.float32)
        )
    assert cache_writes == [1]


@pytest.mark.gpu
@pytest.mark.parametrize(
    "pool,max_tokens", [(1, 4096), (4, 4096), (4, 4095), (4, 4097)]
)
@pytest.mark.parametrize("force_bf16", [False, True])
def test_real_decode_uses_pool_capacity_and_preserves_token_selection(
    monkeypatch, pool, max_tokens, force_bf16
):
    from vllm_fl.models.glm5 import sparse_indexer

    use_fp8 = indexer_backend.current_platform.supports_fp8() and not force_bf16
    monkeypatch.setattr(
        indexer_backend.current_platform, "supports_fp8", lambda: use_fp8
    )
    backend = indexer_backend.Glm5NextIndexerBackend()
    monkeypatch.setattr(sparse_indexer, "INDEXER_BACKEND", backend)
    page, dim, heads, batch = 64, 128, 32, 4
    capacity = (max_tokens + pool - 1) // pool
    # Actual K-pool indexer: 512 logical tokens per block, translated into
    # 64-pool kernel pages, including the final partially used logical block.
    pages = (max_tokens + 511) // 512 * (512 // pool // page)
    backing = torch.full((pages, 16896), 193, dtype=torch.uint8, device="cuda")
    cache = torch.as_strided(backing, (pages, page, dim + 4), (16896, dim + 4, 1))

    # Build the wire format on CPU; vendor Torch need not implement FP8 casts
    # to test the real GPU byte decoder, paged logits and token selection.
    def fp8_input(shape):
        encoded = torch.ones(shape).to(torch.float8_e4m3fn)
        wire = encoded.view(torch.uint8).to("cuda")
        return wire.view(torch.float8_e4m3fn), encoded.float().to("cuda")

    scales = (torch.arange(pages * page, device="cuda") + 1).float() / 128
    if use_fp8:
        keys, reference_keys = fp8_input((pages, page, dim))
        backing[:, : page * dim].copy_(keys.view(torch.uint8).reshape(pages, -1))
        backing[:, page * dim : page * (dim + 4)].copy_(
            scales.view(pages, page).view(torch.uint8)
        )
    else:
        backing = torch.full((pages, 8448), -13, dtype=torch.bfloat16, device="cuda")
        cache = torch.as_strided(backing, (pages, page, dim), (8448, dim, 1))
        # Distinct BF16 values avoid ambiguous TopK ties at the cutoff.
        distinct = (
            (torch.arange(pages * page, dtype=torch.int16) + 0x3800)
            .view(torch.bfloat16)
            .to("cuda")
        )
        cache.copy_(distinct.view(pages, page, 1).expand(-1, -1, dim))
        reference_keys = cache.float()
        scales.fill_(1)
    before = backing.clone()
    table = torch.stack(
        [torch.arange(pages, device="cuda").roll(row) for row in range(batch)]
    ).int()
    lengths = torch.tensor(
        [0, 1, 65, max_tokens // pool], dtype=torch.int32, device="cuda"
    )
    tails = [1, 1, 1, max_tokens % pool] if pool > 1 else [0] * batch
    positions = lengths.long() * pool - 1
    positions += torch.tensor(tails, device="cuda")
    if use_fp8:
        q, reference_q = fp8_input((batch, heads, dim))
    else:
        q = torch.ones(batch, heads, dim, device="cuda", dtype=torch.bfloat16)
        reference_q = q.float()
    weights = torch.ones(batch, heads, device="cuda")
    decode = DeepSeekV32IndexerDecodeMetadata(
        block_table=table,
        seq_lens=lengths[:, None],
        decode_lens=torch.ones(batch, dtype=torch.int32, device="cuda"),
        requires_padding=False,
        schedule_metadata=torch.empty(0, dtype=torch.int32, device="cuda"),
    )
    metadata = DeepseekV32IndexerMetadata(
        seq_lens=lengths,
        max_seq_len=max_tokens,
        slot_mapping=torch.full((batch,), -1, dtype=torch.int64, device="cuda"),
        num_decodes=batch,
        num_decode_tokens=batch,
        num_prefills=0,
        num_prefill_tokens=0,
        decode=decode,
    )
    topk_tokens = 512 * pool
    width = (topk_tokens + pool - 1 + 127) // 128 * 128
    output = torch.full((batch, width), 123, dtype=torch.int32, device="cuda")
    reference_logits = torch.zeros(batch, capacity, device="cuda")
    expected_indices = []
    for row, length in enumerate(lengths.tolist()):
        page_ids = table[row].long()
        row_keys = reference_keys[page_ids].reshape(-1, dim)[:length]
        row_scales = scales.view(pages, page)[page_ids].reshape(-1)[:length]
        values = (
            torch.relu((reference_q[row] @ row_keys.T) * row_scales)
            * weights[row, :, None]
        ).sum(0)
        reference_logits[row, :length] = values
        pools = torch.topk(values, min(512, length)).indices.tolist()
        tokens = [index * pool + offset for index in pools for offset in range(pool)]
        tokens.extend(length * pool + offset for offset in range(tails[row]))
        expected_indices.append(sorted(tokens))

    calls = []
    actual_paged = backend.paged_mqa_logits

    def observe_paged(*args, **kwargs):
        logits = actual_paged(*args, **kwargs)
        assert kwargs["max_model_len"] == capacity
        assert args[4] is table
        assert args[3].data_ptr() == decode.seq_lens.data_ptr()
        assert args[3].shape == decode.seq_lens.shape
        assert logits.shape == (batch, capacity)
        torch.testing.assert_close(logits, reference_logits, atol=2e-5, rtol=2e-5)
        calls.append(logits)
        return logits

    # A forwarding observer around the real public resolver/kernel; no
    # numerical, dependency, metadata or top-k implementation is replaced.
    monkeypatch.setattr(backend, "paged_mqa_logits", observe_paged)
    with set_forward_context({"indexer": metadata}, VllmConfig()):
        result = sparse_attn_indexer_kpool(
            torch.empty(batch, 1, device="cuda"),
            "indexer",
            cache,
            q,
            None,
            torch.empty(batch, dim, device="cuda"),
            weights,
            128,
            None,
            topk_tokens,
            dim,
            max_tokens,
            batch,
            output,
            True,
            index_kpool=pool,
            positions=positions,
        )
    assert len(calls) == 1
    assert result is output
    for row in range(batch):
        actual = result[row][result[row] >= 0].tolist()
        assert sorted(actual) == expected_indices[row]
    assert torch.equal(backing, before)


def test_sparse_physical_conversion_delegates_without_tensor_materialization(
    monkeypatch,
):
    args = (object(), object(), object(), 64)
    expected = (object(), object())
    calls = []

    def convert(*received):
        calls.append(received)
        return expected

    monkeypatch.setattr(sparse_mla_metadata, "convert_sparse_indices", convert)
    assert mla_sparse._convert_request_to_physical_indices(*args) is expected
    assert calls == [args]


def test_sparse_lookup_uses_public_capabilities_not_indexer(monkeypatch):
    expected = object()
    calls = []

    def lookup(name):
        calls.append(name)
        return lambda: expected

    monkeypatch.setattr(capabilities, "get_flaggems_vllm_operator", lookup)
    monkeypatch.setattr(
        indexer_backend,
        "_load_flaggems_op",
        lambda *args: pytest.fail("MLA must not import the indexer adapter"),
    )
    binding = mla_sparse._resolve_sparse_mla_kernel("concat_and_cache_mla")
    assert binding.preflight()[0].fn() is expected
    assert calls == ["concat_and_cache_mla"]


def test_sparse_graph_query_resolves_only_attention(monkeypatch):
    calls = []
    binding = SimpleNamespace(
        graph_capabilities={"test": True},
        preflight=lambda: [SimpleNamespace(impl_id="test")],
    )

    def resolve(name, *, required=True):
        calls.append((name, required))
        return binding

    monkeypatch.setattr(mla_sparse, "_resolve_sparse_mla_kernel", resolve)
    assert mla_sparse.sparse_mla_cudagraph_support(512)[0] is (
        AttentionCGSupport.UNIFORM_SINGLE_TOKEN_DECODE
    )
    assert calls == [("flash_mla_sparse_fwd", False)]
    assert mla_sparse.sparse_mla_cudagraph_support(576)[0] is AttentionCGSupport.NEVER
    assert len(calls) == 1


def test_sparse_runtime_resolves_both_required_bindings(monkeypatch):
    calls = []

    def resolve(name, *, required=True):
        calls.append((name, required))
        return name

    monkeypatch.setattr(mla_sparse, "_resolve_sparse_mla_kernel", resolve)
    assert mla_sparse._resolve_sparse_mla_kernels() == (
        "concat_and_cache_mla",
        "flash_mla_sparse_fwd",
    )
    assert calls == [("concat_and_cache_mla", True), ("flash_mla_sparse_fwd", True)]


@pytest.mark.parametrize("head_size", [512, 576])
@pytest.mark.parametrize("capable", [False, True])
def test_sparse_mla_decode_graph_requires_operator_capability(
    monkeypatch, head_size, capable
):
    op = SimpleNamespace(supports_single_token_decode_graph=capable)
    monkeypatch.setattr(capabilities, "get_flaggems_vllm_operator", lambda *args: op)
    expected = (
        AttentionCGSupport.UNIFORM_SINGLE_TOKEN_DECODE
        if head_size == 512 and capable
        else AttentionCGSupport.NEVER
    )
    spec = SimpleNamespace(head_size=head_size)
    for cls in (
        mla_sparse.FlagGemsSparseMLABackend,
        mla_sparse.FlagGemsSparseMLAMetadataBuilder,
    ):
        assert cls.get_cudagraph_support(None, spec) is expected


def test_sparse_mla_missing_graph_capability_preserves_eager(monkeypatch):
    monkeypatch.setattr(capabilities, "get_flaggems_vllm_operator", lambda *args: None)
    assert mla_sparse.sparse_mla_cudagraph_support(512)[0] is AttentionCGSupport.NEVER


@pytest.mark.parametrize("capable", [False, True])
@pytest.mark.parametrize(
    "case,supported",
    [
        ("main", True),
        ("with_indexer", True),
        ("two_main", True),
        ("unsupported_head", False),
        ("mixed_heads", False),
        ("index_only", False),
        ("empty", False),
        ("non_mla", False),
    ],
)
def test_sparse_mla_grouped_graph_support_checks_own_specs(
    monkeypatch, capable, case, supported
):
    def spec(head=512, ratio=1, dtype=torch.bfloat16):
        return MLAAttentionSpec(
            block_size=512,
            num_kv_heads=1,
            head_size=head,
            dtype=dtype,
            compress_ratio=ratio,
            indexes_kv_by_block_stride=True,
        )

    main, index = spec(), spec(132, 4, torch.uint8)
    entries = {
        "main": {"a": main},
        "with_indexer": {"a": main, "i": index},
        "two_main": {"a": main, "b": spec()},
        "unsupported_head": {"a": spec(576), "i": index},
        "mixed_heads": {"a": main, "b": spec(576), "i": index},
        "index_only": {"i": index},
        "empty": {},
        "non_mla": {"a": SimpleNamespace(head_size=512)},
    }[case]
    group = UniformTypeKVCacheSpecs(block_size=512, kv_cache_specs=entries)
    monkeypatch.setattr(
        capabilities,
        "get_flaggems_vllm_operator",
        lambda *args: SimpleNamespace(supports_single_token_decode_graph=capable),
    )
    expected = (
        AttentionCGSupport.UNIFORM_SINGLE_TOKEN_DECODE
        if capable and supported
        else AttentionCGSupport.NEVER
    )
    for cls in (
        mla_sparse.FlagGemsSparseMLABackend,
        mla_sparse.FlagGemsSparseMLAMetadataBuilder,
    ):
        assert cls.get_cudagraph_support(None, group) is expected
        assert cls._cudagraph_support is AttentionCGSupport.NEVER


@pytest.mark.parametrize("kind", ["flagos", "vendor"])
@pytest.mark.parametrize("fail", [False, True])
def test_sparse_mla_small_heads_use_selected_provider_without_padding(
    monkeypatch, kind, fail
):
    import flaggems_vllm

    from vllm_fl.dispatch import manager as dispatch_manager
    from vllm_fl.dispatch.types import BackendImplKind, OpImpl

    calls = []
    error = torch.OutOfMemoryError("selected provider failure")

    def kernel(label):
        def call(q, *args, **kwargs):
            calls.append((label, tuple(q.shape)))
            if fail:
                raise error
            return q.clone(), None, None

        call.supported_query_heads = (4, 64, 128)
        call.supports_single_token_decode_graph = True
        return call

    public, native = kernel("flagos"), kernel("vendor")
    platform = OpManager(register_builtins=False)
    platform.registry.register_impl(
        OpImpl(
            "flash_mla_sparse_fwd",
            "vendor.test",
            BackendImplKind.VENDOR,
            native,
            vendor="test",
        )
    )
    monkeypatch.setattr(dispatch_manager, "get_default_manager", lambda: platform)
    monkeypatch.setattr(
        capabilities,
        "get_flaggems_vllm_operator",
        lambda name: public if name == "flash_mla_sparse_fwd" else lambda *a, **k: None,
    )
    monkeypatch.setattr(flaggems_vllm, "contiguous_copy", lambda x: x.contiguous())
    # This CPU test isolates provider routing and error propagation, not the
    # device-only conversion kernel (covered by real-device cache/op tests).
    monkeypatch.setattr(
        sparse_mla_metadata,
        "convert_sparse_indices",
        lambda *args: (
            torch.tensor([[0], [64]], dtype=torch.int32),
            torch.ones(2, dtype=torch.int32),
        ),
    )

    def no_head_expansion(q, heads):
        assert q.shape[1] == heads == 4
        return q.contiguous()

    monkeypatch.setattr(flaggems_vllm, "pad_attention_heads", no_head_expansion)
    policy = SelectionPolicy.from_dict(per_op_order={"flash_mla_sparse_fwd": [kind]})
    with policy_context(policy):
        impl = _make_impl()
        impl.topk_indices_buffer = torch.zeros((2, 1), dtype=torch.int32)
        meta = mla_sparse.FlagGemsSparseMLAMetadata(
            num_reqs=2,
            max_query_len=1,
            max_seq_len=1,
            num_actual_tokens=2,
            query_start_loc=torch.tensor([0, 1, 2], dtype=torch.int32),
            slot_mapping=torch.tensor([0, 64]),
            block_table=torch.tensor([[0], [1]], dtype=torch.int32),
            req_id_per_token=torch.tensor([0, 1], dtype=torch.int32),
        )
        q = torch.randn(2, 4, 8).bfloat16()
        cache = torch.zeros(2, 64, 8, dtype=torch.bfloat16)
        if fail:
            with pytest.raises(torch.OutOfMemoryError) as caught:
                impl.forward_mqa(q, cache, meta, None)
            assert caught.value is error
        else:
            output, lse = impl.forward_mqa(q, cache, meta, None)
            torch.testing.assert_close(output, q)
            assert lse is None
        assert calls == [(kind, (2, 4, 8))]
        assert impl._sparse_attn.manager.get_failed_impls("flash_mla_sparse_fwd") == {}


def test_sparse_mla_graph_contract_follows_selected_vendor(monkeypatch):
    from vllm_fl.dispatch import manager as dispatch_manager
    from vllm_fl.dispatch.types import BackendImplKind, OpImpl

    public = lambda *a, **k: None
    public.supports_single_token_decode_graph = False
    native = lambda *a, **k: None
    native.supports_single_token_decode_graph = True
    platform = OpManager(register_builtins=False)
    platform.registry.register_impl(
        OpImpl(
            "flash_mla_sparse_fwd",
            "vendor.test",
            BackendImplKind.VENDOR,
            native,
            vendor="test",
        )
    )
    monkeypatch.setattr(dispatch_manager, "get_default_manager", lambda: platform)
    monkeypatch.setattr(capabilities, "get_flaggems_vllm_operator", lambda *a: public)
    for name, expected in [
        ("vendor", AttentionCGSupport.UNIFORM_SINGLE_TOKEN_DECODE),
        ("flagos", AttentionCGSupport.NEVER),
    ]:
        with policy_context(
            SelectionPolicy.from_dict(per_op_order={"flash_mla_sparse_fwd": [name]})
        ):
            assert mla_sparse.sparse_mla_cudagraph_support(512)[0] is expected


def _sparse_mapping_reference(requests, table, indices, block_size):
    """Independent row oracle: filter, map, preserve order and duplicates."""
    requests, table, indices = requests.cpu(), table.cpu(), indices.cpu()
    out = torch.full(indices.shape, -1, dtype=torch.int32)
    lengths = torch.zeros(indices.shape[0], dtype=torch.int32)
    for row in range(indices.shape[0]):
        req = int(requests[row])
        values = []
        for token in indices[row].tolist():
            if token < 0 or token // block_size >= table.shape[1]:
                continue
            block = int(table[req, token // block_size])
            if block >= 0:
                values.append(block * block_size + token % block_size)
        if values:
            out[row, : len(values)] = torch.tensor(values, dtype=torch.int64).to(
                torch.int32
            )
        lengths[row] = len(values)
    return out, lengths


def _sparse_mapping_inputs(
    rows, width, dtype=torch.int32, strided=False, block_size=64
):
    g = torch.Generator().manual_seed(42)
    table = torch.randint(0, 128, (4, 67), generator=g, dtype=dtype)
    table[:, 1::3] = -1
    table[0, 0] = 1 << 27  # Address math must widen before multiplication.
    requests = (torch.arange(rows, dtype=dtype) % 4) - 4
    indices = torch.randint(
        -5, 70 * block_size, (rows, width), generator=g, dtype=dtype
    )
    if width:
        indices[:, 0] = 0
    if width > 3:
        indices[:, 1:4] = torch.tensor([block_size, 0, 2 * block_size], dtype=dtype)
    values = [requests.cuda(), table.cuda(), indices.cuda()]
    if strided:
        result = []
        for value in values:
            shape = list(value.shape)
            shape[-1] *= 2
            backing = torch.empty(shape, dtype=value.dtype, device=value.device)
            view = backing[..., ::2]
            view.copy_(value)
            result.append(view)
        values = result
    return values


@pytest.mark.parametrize("rows", [0, 1, 8])
@pytest.mark.parametrize("width", [0, 1, 3, 17, 1023, 1024, 1025, 2048, 8195])
@pytest.mark.parametrize("dtype", [torch.int32, torch.int64])
@pytest.mark.parametrize("strided", [False, True])
@pytest.mark.gpu
def test_sparse_mapping(rows, width, dtype, strided):
    args = _sparse_mapping_inputs(rows, width, dtype, strided)
    before = [a.clone() for a in args]
    actual = convert_sparse_indices(*args, 64)
    expected = _sparse_mapping_reference(*args, 64)
    # Report reduction/count errors separately from compacted-store errors.
    torch.testing.assert_close(actual[1].cpu(), expected[1], rtol=0, atol=0)
    for a, e in zip(actual, expected):
        torch.testing.assert_close(a.cpu(), e, rtol=0, atol=0)
        assert a.dtype == torch.int32 and a.is_contiguous()
    for a, b in zip(args, before):
        torch.testing.assert_close(a, b, rtol=0, atol=0)


@pytest.mark.parametrize(
    "dtypes", list(itertools.product([torch.int32, torch.int64], repeat=3))
)
@pytest.mark.parametrize("block_size", [1, 3, 64, 128])
@pytest.mark.gpu
def test_mixed_dtypes_and_page_sizes(dtypes, block_size):
    args = _sparse_mapping_inputs(8, 2051, block_size=block_size)
    args = [a.to(d) for a, d in zip(args, dtypes)]
    for a, e in zip(
        convert_sparse_indices(*args, block_size),
        _sparse_mapping_reference(*args, block_size),
    ):
        torch.testing.assert_close(a.cpu(), e, rtol=0, atol=0)


@pytest.mark.gpu
def test_no_pages_and_all_invalid():
    req = torch.zeros(8, dtype=torch.int32, device="cuda")
    indices = torch.zeros((8, 1025), dtype=torch.int32, device="cuda")
    for table in [
        torch.empty((1, 0), dtype=torch.int32, device="cuda"),
        torch.full((1, 8), -1, dtype=torch.int32, device="cuda"),
    ]:
        out, lengths = convert_sparse_indices(req, table, indices, 64)
        assert (out == -1).all().item() and (lengths == 0).all().item()


@pytest.mark.parametrize("request_id", [0, -1])
@pytest.mark.parametrize("dtype", [torch.int32, torch.int64])
@pytest.mark.gpu
def test_compaction_masks_literal_with_offset_inputs(request_id, dtype):
    # Sentinel guards expose tail-lane reads and writes beyond logical views.
    req = torch.tensor([737, request_id, 737], dtype=dtype, device="cuda")[1:2]
    pages = torch.tensor([737, 10, -1, 12, 737], dtype=dtype, device="cuda")[1:4][
        None, :
    ]
    tokens = torch.tensor(
        [737, -1, 0, 64, 128, 1, 192, 737], dtype=dtype, device="cuda"
    )[1:7][None, :]
    before = [x.clone() for x in (req, pages, tokens)]
    out, lengths = convert_sparse_indices(req, pages, tokens, 64)
    assert lengths.cpu().tolist() == [3]
    assert out.cpu().tolist() == [[640, 768, 641, -1, -1, -1]]
    assert out.is_contiguous() and out.untyped_storage().nbytes() == out.numel() * 4
    for value, original in zip((req, pages, tokens), before):
        torch.testing.assert_close(value, original, rtol=0, atol=0)


@pytest.mark.parametrize("width", [17, 2048, 8195])
@pytest.mark.gpu
def test_dynamic_graph_two_replays(width):
    args = _sparse_mapping_inputs(8, width)
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(3):
            convert_sparse_indices(*args, 64)
    torch.cuda.current_stream().wait_stream(stream)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        actual = convert_sparse_indices(*args, 64)
    for value in [-1, 128]:
        args[2][:, 0] = value
        args[1][:, 0] = 3 if value == -1 else -1
        for tensor in actual:
            tensor.fill_(-737)
        graph.replay()
        for a, e in zip(actual, _sparse_mapping_reference(*args, 64)):
            torch.testing.assert_close(a.cpu(), e, rtol=0, atol=0)


@pytest.mark.gpu
def test_unsupported_contracts():
    req, table, indices = _sparse_mapping_inputs(1, 3)
    with pytest.raises(ValueError):
        convert_sparse_indices(req, table, indices, 0)
    with pytest.raises(ValueError):
        convert_sparse_indices(req[:, None], table, indices, 64)
    with pytest.raises(ValueError):
        convert_sparse_indices(req[:0], table, indices, 64)
    with pytest.raises(NotImplementedError):
        convert_sparse_indices(req, table, indices.float(), 64)
    with pytest.raises(ValueError):
        convert_sparse_indices(req.cpu(), table, indices, 64)
