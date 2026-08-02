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
from __future__ import annotations

import torch
import torch.nn.functional as F


def silu_and_mul_maca(obj, x: torch.Tensor) -> torch.Tensor:
    """SiLU activation followed by element-wise multiplication."""
    d = x.shape[-1] // 2
    op = getattr(torch.ops._C, "silu_and_mul", None)
    if op is not None:
        out = torch.empty(*x.shape[:-1], d, dtype=x.dtype, device=x.device)
        op(out, x)
        return out

    x1, x2 = x[..., :d], x[..., d:]
    return F.silu(x1) * x2


def gelu_and_mul_maca(obj, x: torch.Tensor) -> torch.Tensor:
    """GELU activation followed by element-wise multiplication."""
    d = x.shape[-1] // 2
    approximate = getattr(obj, "approximate", "none") if obj is not None else "none"

    if approximate == "tanh":
        op = getattr(torch.ops._C, "gelu_tanh_and_mul", None)
    else:
        op = getattr(torch.ops._C, "gelu_and_mul", None)

    if op is not None:
        out = torch.empty(*x.shape[:-1], d, dtype=x.dtype, device=x.device)
        op(out, x)
        return out

    x1, x2 = x[..., :d], x[..., d:]
    return F.gelu(x1, approximate=approximate) * x2
