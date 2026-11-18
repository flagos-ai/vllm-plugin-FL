# Copyright (c) 2026 BAAI. All rights reserved.

"""Capabilities of the installed public torch-npu operators."""

import torch

from vllm_fl.dispatch.capabilities import has_device_kernel


def has_npu_operator(name: str, *parameters: str) -> bool:
    import torch_npu

    if not callable(getattr(torch_npu, name, None)):
        return False
    try:
        schema = getattr(torch.ops.npu, name).default._schema
    except AttributeError:
        return False
    return {arg.name for arg in schema.arguments}.issuperset(
        parameters
    ) and has_device_kernel(f"npu::{name}", "npu")
