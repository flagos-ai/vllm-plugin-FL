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
"""Ascend-specific quantization methods and kernels."""

from .linear import AscendModelSlimLinearMethod
from .moe import AscendModelSlimW8A8DynamicMoEMethod
from .w8a8 import (
    AscendW8A8DynamicLinearScheme,
    AscendW8A8StaticLinearScheme,
    get_w8a8_linear_scheme,
)

__all__ = [
    "AscendModelSlimLinearMethod",
    "AscendModelSlimW8A8DynamicMoEMethod",
    "AscendW8A8DynamicLinearScheme",
    "AscendW8A8StaticLinearScheme",
    "get_w8a8_linear_scheme",
]
