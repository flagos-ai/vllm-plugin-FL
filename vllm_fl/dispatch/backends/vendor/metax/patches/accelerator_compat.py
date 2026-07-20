# Copyright 2026 FlagOS Contributors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# SPDX-License-Identifier: Apache-2.0
# 2026 - Modified by MetaX Integrated Circuits (Shanghai) Co., Ltd. All Rights Reserved.

# -----------------------------------------------------
# Note: torch 2.8+metax does not have torch.accelerator.empty_cache
#       (added in PyTorch 2.10). Patch it to use torch.cuda.empty_cache.
# _____________________________________________________

import torch


if not hasattr(torch.accelerator, "empty_cache"):
    torch.accelerator.empty_cache = torch.cuda.empty_cache
