# Copyright (c) 2026 BAAI. All rights reserved.
"""Allow performance runs to skip rank-0-only FlagGems DEBUG recording."""

import functools
import logging
import os


def patch_flaggems_recording():
    import flag_gems

    original = flag_gems.setup_flaggems_logging
    if getattr(original, "_hygon_record_control", False):
        return

    @functools.wraps(original)
    def setup(path=None, record=True, once=False):
        if os.getenv("VLLM_FL_HYGON_RECORD_OPLIST", "1") == "0":
            # once=True filters after LogRecord creation and findCaller().
            # Disabling DEBUG here also skips that per-op Python work.
            logging.getLogger("flag_gems").setLevel(logging.INFO)
            return original(path=path, record=False, once=once)
        return original(path=path, record=record, once=once)

    setup._hygon_record_control = True
    flag_gems.setup_flaggems_logging = setup
