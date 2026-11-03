# SPDX-License-Identifier: Apache-2.0
# 2026 - Modified by MetaX Integrated Circuits (Shanghai) Co., Ltd. All Rights Reserved.

# Adapted for vLLM 0.32 from MetaX-MACA/vLLM-metax @ 440ec38e9508.

import importlib
import os
import sys

import torch


def replace_loaded_vllm_aliases(module_name: str, name: str, replacement) -> None:
    """Replace the defining export and same-object aliases already imported."""
    module = importlib.import_module(module_name)
    original = getattr(module, name)
    for loaded_name, loaded_module in list(sys.modules.items()):
        if loaded_name.startswith("vllm.") and loaded_module is not None:
            if vars(loaded_module).get(name) is original:
                setattr(loaded_module, name, replacement)


def find_mccl_library() -> str:
    """Find MCCL, honoring the vendor override and the native vLLM override."""
    so_file = os.getenv("VLLM_MCCL_SO_PATH") or os.getenv("VLLM_NCCL_SO_PATH")
    if so_file:
        return so_file
    if torch.version.cuda is not None:
        return "libmccl.so"
    raise ValueError("MCCL only supports MACA backends.")


replace_loaded_vllm_aliases("vllm.utils.nccl", "find_nccl_library", find_mccl_library)
