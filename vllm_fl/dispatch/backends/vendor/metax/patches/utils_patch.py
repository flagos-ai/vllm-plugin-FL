# SPDX-License-Identifier: Apache-2.0
# 2026 - Modified by MetaX Integrated Circuits (Shanghai) Co., Ltd. All Rights Reserved.

import vllm
import vllm.utils.nccl

import torch


def find_mccl_library() -> str:
    """
    We either use the library file specified by the `VLLM_NCCL_SO_PATH`
    environment variable, or we find the library file brought by PyTorch.
    After importing `torch`, `libnccl.so.2` or `librccl.so.1` can be
    found by `ctypes` automatically.
    """
    so_file = None

    # manually load the nccl library
    if torch.version.cuda is not None:
        so_file = "libmccl.so"
    else:
        raise ValueError("MCCL only supports MACA backends.")
    return so_file


from vllm_fl.patches.kimi_k3.profile import is_kimi_k3_profile

if not is_kimi_k3_profile():
    vllm.utils.nccl.find_nccl_library = find_mccl_library
