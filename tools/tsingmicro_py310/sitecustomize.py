# Copyright (c) 2026 BAAI. All rights reserved.
"""Explicit, experimental Python 3.10 bootstrap for TXDA vLLM 0.32.

Enable with FL_TXDA_PY310_COMPAT=1, GEMS_VENDOR=tsingmicro, and this directory
on PYTHONPATH before starting Python. Inherited environment variables also
activate the bootstrap in spawned Python workers. This does not change vLLM's
declared Python requirement or establish general Python 3.10 support.
"""

import os
import sys


def _enable_txda_py310_compat() -> None:
    if (
        os.environ.get("FL_TXDA_PY310_COMPAT") != "1"
        or os.environ.get("GEMS_VENDOR") != "tsingmicro"
        or sys.version_info[:2] != (3, 10)
    ):
        return

    import datetime
    import typing

    import typing_extensions

    for name in ("Self", "Never", "NotRequired", "Required", "assert_never"):
        if not hasattr(typing, name):
            setattr(typing, name, getattr(typing_extensions, name))
    if not hasattr(datetime, "UTC"):
        datetime.UTC = datetime.timezone.utc


_enable_txda_py310_compat()
