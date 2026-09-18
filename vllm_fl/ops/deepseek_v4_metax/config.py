# SPDX-License-Identifier: Apache-2.0
"""CPU-only configuration checks; importing this module never initializes a GPU."""

import os

ENV = "VLLM_FL_DSV4_METAX_OPTIMIZATIONS"


def enabled():
    return os.getenv(ENV, "0") == "1"


def validate_shape(*, vendor, tp, pp, dp, ep, hidden, heads, layers, cache):
    if (vendor, tp, pp, dp, ep, hidden, heads, layers) != (
        "metax",
        8,
        1,
        1,
        False,
        4096,
        64,
        43,
    ) or not str(cache).startswith("fp8"):
        raise ValueError(
            "DSV4 MetaX optimizations require C550/TP8/PP1/DP1, no EP, "
            "Flash H4096/HQ64/L43 and FP8 KV cache"
        )


# These are launch choices, not compiler-cache entries or per-rank tuning maps.
DEFAULTS = {
    "VLLM_FL_METAX_DECODE_HQ16": "1",
    "VLLM_FL_METAX_DECODE_HALFD_QKFOLD": "1",
    "VLLM_FL_METAX_DECODE_MODEL1_HALFD": "1",
    "VLLM_FL_METAX_DECODE_SPLIT_TOPK": "1",
    "VLLM_FL_METAX_DECODE_SPLIT_TOPK_S": "4",
    "VLLM_FL_METAX_DECODE_SPLIT_TOPK_BK": "16",
    "VLLM_FL_METAX_DECODE_SPLIT_TOPK_BH": "16",
    "VLLM_FL_METAX_DECODE_SPLIT_TOPK_WARPS": "4",
    "VLLM_FL_METAX_DECODE_SPLIT_TOPK_STAGES": "1",
    "VLLM_FL_METAX_DECODE_SPLIT_TOPK_MAX_B": "64",
    "VLLM_FL_METAX_GRAPHSAFE_DECODE_INDEXER": "1",
    "VLLM_FL_METAX_PAGED_INDEXER_LOGITS": "1",
    "VLLM_FL_METAX_PREFILL_HEAD_PACKING": "1",
    "VLLM_FL_METAX_PREFILL_HALFD_SERIAL": "1",
    "VLLM_FL_METAX_PREFILL_QKFOLD_BLOCKDIAG": "1",
    "VLLM_FL_METAX_PREFILL_ATTN_STAGED": "1",
    "VLLM_FL_METAX_PREFILL_INDEXER_ROWSHARD": "1",
    "VLLM_FL_METAX_PREFILL_NATIVE_TOPK": "0",
    "VLLM_FL_METAX_INDEXER_M84_MODE": "on",
    "VLLM_FL_METAX_INDEXER_M84_DEGREE": "4",
    "VLLM_FL_METAX_MHC_PREFILL_SPLIT": "1",
    "VLLM_FL_METAX_MHC_PRE_SMALLN": "1",
    "VLLM_FL_METAX_MHC_PRE_SMALLN_BK": "64",
    "VLLM_FL_METAX_MHC_PRE_SMALLN_BM": "16",
    "VLLM_FL_METAX_MHC_PRE_SMALLN_SPLITS": "8",
    "VLLM_FL_METAX_MHC_PRE_SMALLN_WARPS": "4",
    "VLLM_FL_METAX_MHC_PRE_SMALLN_STAGES": "1",
}


def configure(vllm_config, vendor):
    parallel = vllm_config.parallel_config
    hf = vllm_config.model_config.hf_config
    validate_shape(
        vendor=vendor,
        tp=parallel.tensor_parallel_size,
        pp=parallel.pipeline_parallel_size,
        dp=parallel.data_parallel_size,
        ep=parallel.enable_expert_parallel,
        hidden=hf.hidden_size,
        heads=hf.num_attention_heads,
        layers=hf.num_hidden_layers,
        cache=vllm_config.cache_config.cache_dtype,
    )
    if vllm_config.speculative_config is not None:
        raise ValueError("The optimized DSV4 path has not been validated with MTP")
    if int(os.getenv("VLLM_FL_METAX_INDEXER_M84_DEGREE", "4")) != 4:
        raise ValueError("Only M/4 passed model top-k validation; M/8 is not supported")
    for key, value in DEFAULTS.items():
        os.environ.setdefault(key, value)
    # Dispatch can initialize before model construction. Require the user to
    # set these exclusions before startup instead of silently changing them late.
    excluded = set(os.getenv("VLLM_FL_FLAGOS_BLACKLIST", "").split(","))
    if not {"topk", "masked_fill", "masked_fill_"}.issubset(excluded):
        raise ValueError(
            "Set VLLM_FL_FLAGOS_BLACKLIST=topk,masked_fill,masked_fill_ before startup"
        )
    if vllm_config.attention_config.use_fp4_indexer_cache:
        raise ValueError("DSV4 MetaX optimized indexer requires the FP8 cache path")
