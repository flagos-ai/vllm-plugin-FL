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
"""fl opt-A7: pin decode (3D) unified-attention launch config on Iluvatar.

Offline sweep on BI-V150 (batch=64, GQA 16q/2kv, d=128, block=16, bf16):

  stock launch (w4 s3) : 6464us @kv=16896, 1938us @kv=4608
  pinned w1 s1         : 3524us @kv=16896, 1099us @kv=4608  -> 1.8x/1.76x

TILE_SIZE=16 and NUM_PAR_SOFTMAX_SEGMENTS=16 are already optimal
(tile 32/64/128 and segs 4-64 all lose or tie). num_warps/num_stages
change scheduling only: output is bit-exact vs stock (max_diff = 0.0).
Pin applies to the 3D (flash-decoding) path only; the 2D path keeps
stock launch parameters.

A/B switches:
  VLLM_FL_DISABLE_DECODE_PIN=1  -> keep stock launch config
  VLLM_FL_DECODE_NUM_WARPS / VLLM_FL_DECODE_NUM_STAGES -> override pin
"""
import os

from vllm.logger import init_logger
from vllm.v1.attention.ops import triton_unified_attention as _tua

logger = init_logger(__name__)

_DISABLED = os.environ.get("VLLM_FL_DISABLE_DECODE_PIN", "0") == "1"
_NUM_WARPS = int(os.environ.get("VLLM_FL_DECODE_NUM_WARPS", "1"))
_NUM_STAGES = int(os.environ.get("VLLM_FL_DECODE_NUM_STAGES", "1"))


def apply() -> None:
    if _DISABLED:
        return
    kern = _tua.kernel_unified_attention

    class _PinnedLaunch:
        def __getitem__(self, grid):
            fn = kern[grid]

            def _call(**kw):
                if kw.get("IS_3D"):
                    kw["num_warps"] = _NUM_WARPS
                    kw["num_stages"] = _NUM_STAGES
                return fn(**kw)

            return _call

    _tua.kernel_unified_attention = _PinnedLaunch()
    logger.info(
        "fl iluvatar: decode unified-attention pinned to num_warps=%d "
        "num_stages=%d (3D path only, bit-exact, sweep-verified ~1.8x)",
        _NUM_WARPS,
        _NUM_STAGES,
    )
