# Copyright (c) 2026 BAAI. All rights reserved.
"""FlagGems implementation of the MLA prefill dispatch contract."""


def mla_prefill_flaggems(**kwargs):
    from flag_gems import flash_attn_varlen_func

    return flash_attn_varlen_func(**kwargs)
