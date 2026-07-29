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
"""Fixed kernel entry points used by the WNA16 quantization adapters.

Implementations live in this package and are called directly. They are not
registered with vllm-fl's general operator dispatch.
"""

from .gemm import is_wna16_gemm_available, wna16_gemm
from .moe import is_wna16_moe_available, wna16_moe

__all__ = [
    "is_wna16_gemm_available",
    "is_wna16_moe_available",
    "wna16_gemm",
    "wna16_moe",
]
