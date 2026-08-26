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

from vllm.model_executor.layers.fla.ops import fused_recurrent as _fused_recurrent


class _PackedDecodeKernel:
    def __init__(self, kernel):
        self._kernel = kernel

    def __getitem__(self, grid):
        launch = self._kernel[grid]

        def tuned_launch(**kwargs):
            if (
                kwargs["mixed_qkv"].shape[0] >= 8
                and kwargs["H"] == 8
                and kwargs["HV"] == 16
                and kwargs["K"] == 128
                and kwargs["V"] == 128
                and kwargs["BV"] == 32
            ):
                kwargs["num_warps"] = 4
            return launch(**kwargs)

        return tuned_launch


_fused_recurrent.fused_recurrent_gated_delta_rule_packed_decode_kernel = (
    _PackedDecodeKernel(
        _fused_recurrent.fused_recurrent_gated_delta_rule_packed_decode_kernel
    )
)
