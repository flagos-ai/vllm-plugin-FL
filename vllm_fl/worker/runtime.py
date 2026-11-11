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
        flag_gems.enable(unused=blacklist, **common)
    else:
        logger.info("[FlagGems] Enable all ops")
        flag_gems.enable(**common)
