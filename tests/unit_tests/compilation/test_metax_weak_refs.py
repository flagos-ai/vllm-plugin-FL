# Copyright (c) 2026 BAAI. All rights reserved.
"""Non-owning graph outputs without optional MetaX native extensions."""

import ast
import gc
import sys
import weakref
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import Mock

import pytest
import torch

SOURCE = Path(__file__).resolve().parents[3] / "vllm_fl/compilation/graph.py"


def _load(torch_module=torch):
    names = {
        "_borrow_metax_tensor_storage",
        "_weak_ref_metax_tensor",
        "_weak_ref_metax_tensors",
        "weak_ref_tensors",
    }
    tree = ast.parse(SOURCE.read_text(encoding="utf-8"))
    nodes = [node for node in tree.body if getattr(node, "name", None) in names]
    future = ast.ImportFrom(
        module="__future__", names=[ast.alias(name="annotations")], level=0
    )
    module = ModuleType("metax_weak_refs_under_test")
    module.torch = torch_module
    module.current_platform = SimpleNamespace(vendor_name="metax", device_type="cuda")
    code = ast.fix_missing_locations(ast.Module(body=[future, *nodes], type_ignores=[]))
    exec(compile(code, str(SOURCE), "exec"), module.__dict__)
    return module


@pytest.mark.parametrize(
    ("dtype", "conjugate", "negative", "requires_grad"),
    [
        (torch.float32, False, False, False),
        (torch.bfloat16, False, False, False),
        (torch.complex64, True, False, False),
        (torch.float32, False, True, False),
        (torch.complex64, True, True, True),
    ],
)
def test_borrowed_storage_preserves_views_and_releases_original_owners(
    dtype, conjugate, negative, requires_grad
):
    module = _load()
    owner = torch.arange(48, dtype=torch.float32).to(dtype)
    if requires_grad:
        owner.requires_grad_()
    view = owner.reshape(6, 8)[1:5, 1::2]
    if conjugate:
        view = view.conj()
    if negative:
        view = view._neg_view()
    storage = view.untyped_storage()
    expected = view.detach().clone()
    alias = module._borrow_metax_tensor_storage(view)
    empty = module._borrow_metax_tensor_storage(view[:0])

    assert alias is not view and alias._base is None
    assert alias.data_ptr() == view.data_ptr()
    assert alias.shape == view.shape and alias.stride() == view.stride()
    assert alias.storage_offset() == view.storage_offset() == 9
    assert alias.dtype == view.dtype and alias.device == view.device
    assert alias.requires_grad == requires_grad and alias.grad_fn is None
    assert alias.is_conj() == view.is_conj() == conjugate
    assert alias.is_neg() == view.is_neg() == negative
    assert alias.untyped_storage()._cdata != storage._cdata
    assert alias.untyped_storage().nbytes() == storage.nbytes()
    assert not alias.untyped_storage().resizable()
    assert empty.shape == view[:0].shape and empty.stride() == view.stride()
    assert empty.storage_offset() == 9
    torch.testing.assert_close(alias, expected)

    owner_ref, view_ref, storage_ref = (
        weakref.ref(owner),
        weakref.ref(view),
        weakref.ref(storage),
    )
    del owner, view, storage
    gc.collect()
    assert owner_ref() is None and view_ref() is None and storage_ref() is None
    # alias/empty now contain dangling graph-pool pointers. Their metadata is
    # non-owning; do not dereference them after the real owner has been freed.


def test_borrowed_alias_destruction_does_not_free_owning_tensor():
    module = _load()
    owner = torch.arange(32, dtype=torch.float32)
    view = owner[3::2]
    alias = module._borrow_metax_tensor_storage(view)
    del alias
    gc.collect()
    torch.testing.assert_close(view, torch.arange(3, 32, 2, dtype=torch.float32))


def test_generic_reconstruction_only_creates_zero_element_tensor(monkeypatch):
    module = _load()
    owner = torch.ones(16)
    original_empty = torch.empty
    allocations = []

    def empty(size, **kwargs):
        allocations.append(size)
        assert size == 0
        return original_empty(size, **kwargs)

    monkeypatch.setattr(torch, "empty", empty)
    result = module._borrow_metax_tensor_storage(owner[1::3])
    assert allocations == [0]
    assert result.untyped_storage().data_ptr() == owner.untyped_storage().data_ptr()


def test_cuda_metadata_constructor_avoids_even_zero_element_constructor():
    storage = SimpleNamespace(data_ptr=lambda: 123456, nbytes=lambda: 96)
    value = SimpleNamespace(
        device=torch.device("cuda:2"),
        dtype=torch.float32,
        shape=torch.Size([2, 3]),
        untyped_storage=lambda: storage,
        storage_offset=lambda: 4,
        stride=lambda: (6, 2),
        is_conj=lambda: False,
        is_neg=lambda: False,
        requires_grad=False,
    )
    borrowed, result = object(), Mock()
    construct_storage = Mock(return_value=borrowed)
    construct_tensor = Mock(return_value=result)
    torch_stub = SimpleNamespace(
        _C=SimpleNamespace(
            _construct_storage_from_data_pointer=construct_storage,
            _construct_CUDA_Tensor_From_Storage_And_Metadata=construct_tensor,
        ),
        empty=Mock(side_effect=AssertionError("unexpected tensor allocation")),
    )
    module = _load(torch_stub)
    assert module._borrow_metax_tensor_storage(value) is result
    construct_storage.assert_called_once_with(123456, torch.device("cuda:2"), 96)
    metadata, actual_storage = construct_tensor.call_args.args
    assert actual_storage is borrowed
    assert metadata["storage_offset"] == 4 and metadata["stride"] == (6, 2)
    assert metadata["device"] == torch.device("cuda:2")
    torch_stub.empty.assert_not_called()


@pytest.mark.parametrize("namespace", ["_C", "vllm_fl", None])
def test_native_cuda_kernel_takes_precedence_but_schema_does_not(namespace):
    value = SimpleNamespace(device=SimpleNamespace(type="cuda"))
    native_result, borrowed_result = object(), object()
    native = Mock(return_value=native_result)
    probe = Mock(side_effect=lambda name, key: name == f"{namespace}::weak_ref_tensor")
    torch_stub = SimpleNamespace(
        _C=SimpleNamespace(_dispatch_has_kernel_for_dispatch_key=probe),
        ops=SimpleNamespace(
            _C=SimpleNamespace(weak_ref_tensor=native),
            vllm_fl=SimpleNamespace(weak_ref_tensor=native),
        ),
    )
    module = _load(torch_stub)
    module._borrow_metax_tensor_storage = Mock(return_value=borrowed_result)
    result = module._weak_ref_metax_tensor(value)
    if namespace:
        assert result is native_result
        native.assert_called_once_with(value)
        module._borrow_metax_tensor_storage.assert_not_called()
    else:
        assert result is borrowed_result
        native.assert_not_called()
        module._borrow_metax_tensor_storage.assert_called_once_with(value)
    assert all(call.args[1] == "CUDA" for call in probe.call_args_list)


def test_metax_nested_outputs_include_intermediate_tensors(monkeypatch):
    class IntermediateTensors:
        def __init__(self, tensors):
            self.tensors = tensors

    sequence = ModuleType("vllm.sequence")
    sequence.IntermediateTensors = IntermediateTensors
    monkeypatch.setitem(sys.modules, sequence.__name__, sequence)
    module = _load()
    replacements = {}

    def borrow(value):
        replacements[id(value)] = object()
        return replacements[id(value)]

    module._weak_ref_metax_tensor = borrow
    first, second, third, empty = (
        torch.ones(1),
        torch.ones(2),
        torch.ones(3),
        torch.empty(0),
    )
    value = {
        "list": [first, (second, empty)],
        "intermediate": IntermediateTensors({"hidden": third}),
        "metadata": None,
    }
    result = module.weak_ref_tensors(value)
    assert result["list"][0] is replacements[id(first)]
    assert result["list"][1] == (replacements[id(second)], replacements[id(empty)])
    assert result["intermediate"].tensors["hidden"] is replacements[id(third)]
    assert result["metadata"] is None


def test_metax_keeps_cpu_outputs_and_propagates_reconstruction_errors(monkeypatch):
    module = _load()
    tensor = torch.ones(2)
    assert module.weak_ref_tensors(tensor) is tensor
    module._weak_ref_metax_tensor = Mock(
        side_effect=RuntimeError("cannot borrow storage")
    )
    with pytest.raises(RuntimeError, match="cannot borrow storage"):
        module.weak_ref_tensors(tensor)


@pytest.mark.parametrize("device_type", ["npu", "gcu"])
def test_native_device_weak_refs_take_precedence_over_metax_vendor(device_type):
    module = _load()
    module.current_platform.device_type = device_type
    module._weak_ref_metax_tensors = Mock(
        side_effect=AssertionError("MetaX storage aliases require the CUDA device type")
    )
    value, expected = object(), object()
    native = Mock(return_value=expected)
    setattr(module, f"_weak_ref_{device_type}_tensors", native)
    assert module.weak_ref_tensors(value) is expected
    native.assert_called_once_with(value)
    module._weak_ref_metax_tensors.assert_not_called()


@pytest.mark.parametrize("primitive", [None, object()])
def test_missing_storage_constructor_fails_instead_of_retaining_output(
    monkeypatch, primitive
):
    module = _load()
    monkeypatch.setattr(torch._C, "_construct_storage_from_data_pointer", primitive)
    with pytest.raises(RuntimeError, match="MetaX graph weak references require"):
        module._borrow_metax_tensor_storage(torch.ones(1))
