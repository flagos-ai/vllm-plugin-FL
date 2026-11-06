# SPDX-License-Identifier: Apache-2.0
"""Opt-in worker bootstrap. Only serve.py adds this directory to PYTHONPATH."""

import os
import traceback

try:
    from vllm_fl.patches.deepseek_v4_metax.bootstrap import initialize

    initialize()
except BaseException:
    traceback.print_exc()
    os._exit(70)
