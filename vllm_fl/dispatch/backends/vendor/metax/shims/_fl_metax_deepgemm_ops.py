"""Replace vLLM's DeepGEMM paged-MQA helpers with FlagGems Triton ones.

HY4-Flash's compressed indexer calls
``vllm.utils.deep_gemm.get_paged_mqa_logits_metadata`` and
``fp8_fp4_paged_mqa_logits``.  The empty wheel has no DeepGEMM; FlagGems ships
Triton equivalents with the same semantics.  The varlen ``indices`` argument
(capability 10 only) is accepted and ignored, matching metax's non-varlen path.

Kill switch: ``FL_METAX_GEMS_DEEPGEMM=0``.
"""

import os
import sys

from ._fl_metax_gate import metax_active

_SWITCH = "FL_METAX_GEMS_DEEPGEMM"


def _metadata(context_lens, block_size, num_sms, indices=None):
    from flag_gems.ops.get_paged_mqa_logits_metadata import (
        get_paged_mqa_logits_metadata,
    )

    return get_paged_mqa_logits_metadata(context_lens, block_size, num_sms)


def _logits(*args, indices=None, **kwargs):
    from flaggems_vllm.ops.fp8_fp4_paged_mqa_logits import (
        fp8_fp4_paged_mqa_logits,
    )

    return fp8_fp4_paged_mqa_logits(*args, **kwargs)


def patch() -> bool:
    if not metax_active():
        return False
    value = os.environ.get(_SWITCH, "1").strip().lower()
    if value in ("", "0", "false", "no", "off"):
        return False

    import vllm.utils.deep_gemm as deep_gemm

    deep_gemm.get_paged_mqa_logits_metadata = _metadata
    deep_gemm.fp8_fp4_paged_mqa_logits = _logits

    # Patch consumers that already did ``from ... import name``.
    for name in (
        "vllm_fl.models.hy_v4_flash.nvidia.compressed_indexer",
        "vllm.models.hy_v4.nvidia.compressed_indexer",
    ):
        mod = sys.modules.get(name)
        if mod is not None:
            if hasattr(mod, "get_paged_mqa_logits_metadata"):
                mod.get_paged_mqa_logits_metadata = _metadata
            if hasattr(mod, "fp8_fp4_paged_mqa_logits"):
                mod.fp8_fp4_paged_mqa_logits = _logits
    print("[vllm_fl][metax] DeepGEMM paged-MQA -> FlagGems Triton")
    return True


patch()
