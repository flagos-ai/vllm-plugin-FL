# SPDX-License-Identifier: Apache-2.0
# 2026 - Modified by MetaX Integrated Circuits (Shanghai) Co., Ltd. All Rights Reserved.

# -----------------------------------------------------
# Note: torch 2.8+metax does not have torch.accelerator.empty_cache
#       (added in PyTorch 2.10). Patch it to use torch.cuda.empty_cache.
# _____________________________________________________

import torch


if not hasattr(torch.accelerator, "empty_cache"):
    torch.accelerator.empty_cache = torch.cuda.empty_cache
