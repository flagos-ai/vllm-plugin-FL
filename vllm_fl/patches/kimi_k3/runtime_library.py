# SPDX-License-Identifier: Apache-2.0
# 2026 - Modified by MetaX Integrated Circuits (Shanghai) Co., Ltd.
"""Locate the permitted MCCL device-runtime library without importing kernels."""

import os


def find_mccl_library() -> str:
    return os.environ.get("VLLM_NCCL_SO_PATH") or "libmccl.so"
