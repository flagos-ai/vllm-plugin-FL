# Copyright (c) 2026 BAAI. All rights reserved.

"""Runtime initialization shared by the NVIDIA and multi-vendor workers."""

from vllm.logger import init_logger

import vllm_fl.envs as fl_envs
from vllm_fl.attention.utils import patch_mm_encoder_attention
from vllm_fl.utils import get_flag_gems_whitelist_blacklist

logger = init_logger(__name__)


def get_fl_model_runner_cls():
    from importlib.metadata import version

    from packaging.version import Version

    if Version(version("vllm")) >= Version("0.32.0"):
        from vllm_fl.worker.native_model_runner import NativeModelRunnerFL

        return NativeModelRunnerFL

    from vllm_fl.worker.model_runner import ModelRunnerFL

    return ModelRunnerFL


def is_residual_scattered_for_sp(vllm_config, num_input_tokens: int) -> bool:
    """Keep the FL pipeline tensor-layout contract across vLLM versions."""
    from vllm.v1.worker import utils as worker_utils

    native = getattr(worker_utils, "is_residual_scattered_for_sp", None)
    if native is not None:
        return native(vllm_config, num_input_tokens)

    # vLLM 0.32 removed this helper; FL still prepares legacy PP transfers.
    compilation = vllm_config.compilation_config
    if not compilation.pass_config.enable_sp:
        return False
    tp = vllm_config.parallel_config.tensor_parallel_size
    if tp == 1:
        return False
    assert compilation.use_inductor_graph_partition or not compilation.splitting_ops, (
        "Sequence parallelism requires full-graph compilation"
    )
    assert num_input_tokens % tp == 0, (
        "Sequence parallelism requires TP-aligned token counts"
    )
    return True


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
