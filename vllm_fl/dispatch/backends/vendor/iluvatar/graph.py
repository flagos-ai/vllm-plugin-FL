# Copyright (c) 2026 BAAI. All rights reserved.

from typing import Any

import torch


def _weak_ref_iluvatar_tensor(value: torch.Tensor) -> torch.Tensor:
    if value.device.type != "cuda":
        return value

    construct_storage = getattr(torch._C, "_construct_storage_from_data_pointer", None)
    construct_tensor = getattr(
        torch._C, "_construct_CUDA_Tensor_From_Storage_And_Metadata", None
    )
    if not callable(construct_storage) or not callable(construct_tensor):
        raise RuntimeError(
            "Iluvatar graph weak references require PyTorch CUDA storage and "
            "tensor construction primitives. Disable graph mode or update PyTorch."
        )
    # Follow PyTorch cudagraph_trees output reconstruction. The captured graph
    # owns the pool; these output handles must not keep allocations live across
    # captures of different shapes. Preserve views relative to the storage base.
    storage = value.untyped_storage()
    metadata = {
        "nbytes": storage.nbytes(),
        "data_ptr": storage.data_ptr(),
        "size": value.shape,
        "stride": value.stride(),
        "dtype": value.dtype,
        "device": value.device,
        "storage_offset": value.storage_offset(),
    }
    borrowed_storage = construct_storage(
        metadata["data_ptr"], metadata["device"], metadata["nbytes"]
    )
    result = construct_tensor(metadata, borrowed_storage)
    if value.is_conj():
        torch._C._set_conj(result, True)
    if value.is_neg():
        torch._C._set_neg(result, True)
    return result


def _weak_ref_iluvatar_tensors(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        return _weak_ref_iluvatar_tensor(value)
    if isinstance(value, list):
        return [_weak_ref_iluvatar_tensors(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_weak_ref_iluvatar_tensors(item) for item in value)
    if isinstance(value, dict):
        return {key: _weak_ref_iluvatar_tensors(item) for key, item in value.items()}

    from vllm.sequence import IntermediateTensors

    if isinstance(value, IntermediateTensors):
        return IntermediateTensors(
            {
                key: _weak_ref_iluvatar_tensors(item)
                for key, item in value.tensors.items()
            }
        )
    return value
