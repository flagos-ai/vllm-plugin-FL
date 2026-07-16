# SPDX-License-Identifier: Apache-2.0
# 2026 - Modified by MetaX Integrated Circuits (Shanghai) Co., Ltd. All Rights Reserved.

# --------------------------------------------------------------------------------
# Hotfix: torch 2.8+metax is missing several torch.accelerator memory APIs that
# were added in PyTorch 2.9+. Patch them to the equivalent torch.cuda calls.
# Reference: https://github.com/MetaX-MACA/vLLM-metax/blob/releases/v0.21.0/vllm_metax/patch/torch_fix/fix_standalone_compile.py
# TODO: remove when MetaX ships torch >= 2.9 with full accelerator API support.
# --------------------------------------------------------------------------------

import torch

_MISSING_APIS = {
    "empty_cache": torch.cuda.empty_cache,
    "memory_stats": torch.cuda.memory_stats,
    "memory_reserved": torch.cuda.memory_reserved,
    "memory_allocated": torch.cuda.memory_allocated,
    "reset_peak_memory_stats": torch.cuda.reset_peak_memory_stats,
    "max_memory_allocated": torch.cuda.max_memory_allocated,
}

for _name, _impl in _MISSING_APIS.items():
    if not hasattr(torch.accelerator, _name):
        setattr(torch.accelerator, _name, _impl)
