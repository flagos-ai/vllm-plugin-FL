# SPDX-License-Identifier: Apache-2.0
"""Native API guards use real dispatch and explicit numerical boundary stubs."""

from types import ModuleType, SimpleNamespace

import pytest
import torch

from vllm_fl.dispatch.manager import OpManager
from vllm_fl.dispatch.policy import SelectionPolicy, policy_context
from vllm_fl.kernels.glm5_next import indexer_backend

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
