# Copyright (c) 2026 BAAI. All rights reserved.
"""CPU ownership controls for the exact MUSA output reconstruction function.

The vendor interface is a CPU double; these tests do not validate native MUSA
capture or read an allocation after its real owner has released it.
"""

import ast
import builtins
import gc
import weakref
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
import torch


class _Tensor:
    def __init__(self, payload, *, device="musa", flags=None):
        self.payload = payload
        self.device = SimpleNamespace(type=device, index=0)
        self.flags = flags or {}

    @property
    def layout(self):
        return self.flags.get("layout", self.payload.layout)

    @property
    def dtype(self):
        return self.payload.dtype

    @property
    def requires_grad(self):
        return self.flags.get("requires_grad", self.payload.requires_grad)

    @property
    def is_quantized(self):
        return self.flags.get("is_quantized", self.payload.is_quantized)

    def is_conj(self):
        return self.flags.get("is_conj", self.payload.is_conj())

    def is_neg(self):
        return self.flags.get("is_neg", self.payload.is_neg())

    def untyped_storage(self):
        return self.payload.untyped_storage()

    def size(self):
        return self.payload.size()

    def stride(self):
        return self.payload.stride()

    def storage_offset(self):
        return self.payload.storage_offset()


class _Intermediate:
    def __init__(self, tensors):
        self.tensors = tensors


@pytest.fixture
def contract():
    source = Path(__file__).resolve().parents[3] / "vllm_fl/compilation/graph.py"
    module = ast.parse(source.read_text(encoding="utf-8"), filename=str(source))
    names = {"_weak_ref_musa_tensors", "weak_ref_tensors"}
    musa_source = source.with_name("musa_graph.py")
    musa_module = ast.parse(
        musa_source.read_text(encoding="utf-8"), filename=str(musa_source)
    )
    funcs = [
        n
        for n in [*module.body, *musa_module.body]
        if isinstance(n, ast.FunctionDef) and n.name in names
    ]
    assert {n.name for n in funcs} == names
    state = SimpleNamespace(
        storage_calls=[], native_calls=[], errors={}, platform="musa"
    )

    def storage(data_ptr, device, size):
        state.storage_calls.append((data_ptr, device, size))
        if "storage" in state.errors:
            raise state.errors["storage"]
        assert device.type == "musa"
        return torch._C._construct_storage_from_data_pointer(
            data_ptr, torch.device("cpu"), size
        )

    def native(metadata, borrowed):
        state.native_calls.append((metadata, borrowed))
        if "native" in state.errors:
            raise state.errors["native"]
        assert set(metadata) == {"size", "stride", "dtype", "device", "storage_offset"}
        assert metadata["device"].type == "musa"
        payload = torch.empty(0, dtype=metadata["dtype"]).set_(
            borrowed, metadata["storage_offset"], metadata["size"], metadata["stride"]
        )
        return _Tensor(payload)

    vendor = SimpleNamespace(
        _MUSAC=SimpleNamespace(_construct_MUSA_Tensor_From_Storage_And_Metadata=native)
    )
    platform = SimpleNamespace(device_type="musa")
    sentinel = object()

    def importer(name, globals=None, locals=None, fromlist=(), level=0):
        if name == "torch_musa":
            return vendor
        if name == "vllm_fl.compilation.musa_graph":
            assert tuple(fromlist) == ("_weak_ref_musa_tensors",)
            return SimpleNamespace(
                _weak_ref_musa_tensors=namespace["_weak_ref_musa_tensors"]
            )
        if name == "vllm.sequence":
            return SimpleNamespace(IntermediateTensors=_Intermediate)
        if name == "vllm.utils.torch_utils":
            return SimpleNamespace(weak_ref_tensors=lambda value: ("upstream", value))
        if name.startswith(("vllm", "torch_musa")):
            raise AssertionError("Unexpected target import")
        return builtins.__import__(name, globals, locals, fromlist, level)

    namespace = {
        "__builtins__": dict(vars(builtins), __import__=importer),
        "Any": Any,
        "torch": SimpleNamespace(
            Tensor=_Tensor,
            strided=torch.strided,
            _C=SimpleNamespace(_construct_storage_from_data_pointer=storage),
        ),
        "current_platform": platform,
        "_weak_ref_npu_tensors": lambda value: ("npu", value),
    }
    exec(
        compile(ast.Module(body=funcs, type_ignores=[]), str(source), "exec"), namespace
    )
    state.convert = namespace["_weak_ref_musa_tensors"]
    state.route = namespace["weak_ref_tensors"]
    state.vendor = vendor
    state.platform = platform
    state.sentinel = sentinel
    return state


@pytest.mark.parametrize(
    "dtype",
    [
        torch.float16,
        torch.bfloat16,
        torch.float32,
        torch.float64,
        torch.int32,
        torch.int64,
        torch.bool,
        torch.complex64,
    ],
)
@pytest.mark.parametrize("kind", ["offset", "transpose", "broadcast", "empty_view"])
def test_native_metadata_and_exact_storage_preserved(contract, dtype, kind):
    owner = torch.arange(96, dtype=torch.float32).to(dtype)
    if kind == "offset":
        payload = owner.as_strided((3, 4), (12, 2), 5)
    elif kind == "transpose":
        payload = owner.view(8, 12).t()
    elif kind == "broadcast":
        payload = owner.as_strided((3, 4), (0, 1), 7)
    else:
        payload = owner.as_strided((0, 4), (12, 2), 5)
    original = _Tensor(payload)
    result = contract.convert(original).payload
    assert torch.equal(result, payload)
    assert result.dtype == payload.dtype
    assert result.size() == payload.size()
    assert result.stride() == payload.stride()
    assert result.storage_offset() == payload.storage_offset()
    assert result.data_ptr() == payload.data_ptr()
    assert result.untyped_storage().data_ptr() == payload.untyped_storage().data_ptr()
    assert result.untyped_storage().nbytes() == payload.untyped_storage().nbytes()
    assert result.untyped_storage()._cdata != payload.untyped_storage()._cdata
    assert not result.untyped_storage().resizable()
    assert len(contract.storage_calls) == len(contract.native_calls) == 1


def test_recursive_tree_preserves_tensor_identity_and_view_aliases(contract):
    owner = torch.arange(32, dtype=torch.float32)
    a = _Tensor(owner[2:14])
    b = _Tensor(owner[5:17])
    tree = [a, (a, {"view": b, "other": _Intermediate({"again": a})}), None, "text"]
    result = contract.convert(tree)
    assert result[0] is result[1][0] is result[1][1]["other"].tensors["again"]
    assert result[1][1]["view"] is not result[0]
    assert torch._C._is_alias_of(result[0].payload, result[1][1]["view"].payload)
    assert len(contract.storage_calls) == 1
    assert len(contract.native_calls) == 2
    assert result[2:] == [None, "text"]


def test_output_descriptor_does_not_keep_original_allocation_alive(contract):
    original = _Tensor(torch.arange(24, dtype=torch.float32))
    owner_ref = weakref.ref(original.payload)
    storage_ref = original.untyped_storage()._weak_ref()
    result = contract.convert(original)
    del original
    gc.collect()
    assert owner_ref() is None
    assert torch.UntypedStorage._expired(storage_ref)
    # No dereference of result after owner destruction: graph-pool lifetime
    # must be provided by the real native graph, not by this descriptor.
    del result
    torch.UntypedStorage._free_weak_ref(storage_ref)


def test_external_captured_input_keeps_memory_owned(contract):
    original = _Tensor(torch.arange(24, dtype=torch.float32))
    storage_ref = original.untyped_storage()._weak_ref()
    inputs = [original]
    output = contract.convert(original)
    del original
    gc.collect()
    assert not torch.UntypedStorage._expired(storage_ref)
    assert torch.equal(output.payload, inputs[0].payload)
    del output
    inputs.clear()
    gc.collect()
    assert torch.UntypedStorage._expired(storage_ref)
    torch.UntypedStorage._free_weak_ref(storage_ref)


@pytest.mark.parametrize(
    "flag", ["requires_grad", "is_quantized", "is_conj", "is_neg", "layout"]
)
def test_unsupported_tensor_metadata_is_rejected(contract, flag):
    item = _Tensor(
        torch.ones(4), flags={flag: torch.sparse_coo if flag == "layout" else True}
    )
    with pytest.raises(ValueError, match="plain strided inference"):
        contract.convert(item)
    assert contract.storage_calls == contract.native_calls == []


def test_empty_storage_does_not_require_vendor_constructor(contract):
    original = _Tensor(torch.empty(0))
    del contract.vendor._MUSAC
    assert contract.convert(original) is original
    assert contract.storage_calls == contract.native_calls == []


def test_nonempty_null_storage_is_rejected(contract):
    original = _Tensor(torch.ones(1))
    original.untyped_storage = lambda: SimpleNamespace(
        nbytes=lambda: 4, data_ptr=lambda: 0
    )
    with pytest.raises(ValueError, match="null address"):
        contract.convert(original)
    assert contract.storage_calls == contract.native_calls == []


def test_missing_vendor_interface_fails_instead_of_retaining_output(contract):
    del contract.vendor._MUSAC
    with pytest.raises(RuntimeError, match="metadata tensor constructor"):
        contract.convert(_Tensor(torch.ones(1)))
    assert contract.storage_calls == []


@pytest.mark.parametrize("where", ["native", "storage"])
def test_constructor_exception_identity_is_preserved(contract, where):
    error = RuntimeError("constructor failure")
    contract.errors[where] = error
    with pytest.raises(RuntimeError) as caught:
        contract.convert(_Tensor(torch.ones(1)))
    assert caught.value is error


@pytest.mark.parametrize("device", ["cpu", "cuda", "npu"])
def test_non_musa_tensor_unchanged_even_on_musa_platform(contract, device):
    item = _Tensor(torch.ones(2, requires_grad=True), device=device)
    assert contract.route(item) is item
    assert contract.storage_calls == contract.native_calls == []


@pytest.mark.parametrize("platform", ["cuda", "npu", "cpu"])
def test_non_musa_platform_delegation_unchanged(contract, platform):
    contract.platform.device_type = platform
    value = contract.sentinel
    assert contract.route(value) == ("npu" if platform == "npu" else "upstream", value)
    assert contract.storage_calls == contract.native_calls == []
