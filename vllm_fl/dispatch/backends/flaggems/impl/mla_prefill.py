# Copyright (c) 2026 BAAI. All rights reserved.
"""FlagGems implementation of the MLA prefill dispatch contract."""


def mla_prefill_flaggems(**kwargs):
    import vllm_fl.envs as fl_envs

    if fl_envs.VLLM_FL_USE_FLAGGEMS_VLLM:
        from flaggems_vllm import flash_attn_varlen_func
    else:
        from flag_gems import flash_attn_varlen_func

    return flash_attn_varlen_func(**kwargs)
