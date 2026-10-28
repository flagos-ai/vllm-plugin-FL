# Copyright (c) 2026 BAAI. All rights reserved.

"""Runtime initialization shared by the NVIDIA and multi-vendor workers."""

from vllm.logger import init_logger

import vllm_fl.envs as fl_envs
from vllm_fl.attention.utils import patch_mm_encoder_attention
from vllm_fl.utils import get_flag_gems_whitelist_blacklist

logger = init_logger(__name__)


def _initialize_fl_runtime(rank: int) -> None:
    """Install FL dispatch hooks before model/operator construction."""
    from vllm_fl.ops.custom_ops import register_oot_ops

    # CustomOp caches its selected forward method in __init__, so the
    # multimodal attention patch must run before model construction.
    patch_mm_encoder_attention()
    register_oot_ops()

    if not fl_envs.USE_FLAGGEMS:
        return

    import flag_gems

    whitelist, blacklist = get_flag_gems_whitelist_blacklist()
    should_record = rank == 0
    common = {
        "record": should_record,
        "once": True,
        "path": fl_envs.FLAGGEMS_ENABLE_OPLIST_PATH,
    }
    if whitelist:
        logger.info("[FlagGems] Enable only the following ops: %s", whitelist)
        flag_gems.only_enable(include=whitelist, **common)
    elif blacklist:
        logger.info("[FlagGems] Disable the following ops: %s", blacklist)
        _enable_except(flag_gems, blacklist, common)
    else:
        logger.info("[FlagGems] Enable all ops")
        flag_gems.enable(**common)


def _enable_except(flag_gems, blacklist: list[str], common: dict) -> None:
    """Enable every FlagGems op except *blacklist*.

    ``flag_gems.enable(unused=...)`` is not honoured for every operator in
    FlagGems 5.4.0 — blacklisted ATen overloads such as ``pow.Scalar`` still get
    their CUDA kernel replaced, which breaks backends whose Triton compiler
    rejects the generated code. Deriving an explicit whitelist from the full
    operator list keeps the exclusion effective, and falls back to ``unused=``
    when the operator list cannot be read.
    """
    excluded = set(blacklist)
    try:
        all_ops = [entry[0] for entry in flag_gems._FULL_CONFIG]
    except Exception as e:  # pragma: no cover - depends on FlagGems internals
        logger.warning(
            "[FlagGems] Could not read the operator list (%s); falling back to "
            "enable(unused=...), which may not exclude every operator.",
            e,
        )
        flag_gems.enable(unused=blacklist, **common)
        return

    include = [op for op in all_ops if op not in excluded]
    missed = excluded.difference(all_ops)
    if missed:
        logger.warning(
            "[FlagGems] Blacklisted ops not present in the operator list: %s",
            sorted(missed),
        )
    logger.info(
        "[FlagGems] Enabling %d of %d ops (excluding %d blacklisted).",
        len(include),
        len(all_ops),
        len(all_ops) - len(include),
    )
    flag_gems.only_enable(include=include, **common)
