# Copyright 2026 FlagOS Contributors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""W8A8 adapters for dynamic per-token INT8 inference."""

from .reference import (
    dynamic_per_token_quant_int8,
    w8a8_linear_reference,
)

__all__ = [
    "dynamic_per_token_quant_int8",
    "w8a8_linear_reference",
]
