# Copyright (c) 2026 BAAI. All rights reserved.

import subprocess
import sys


def test_non_attention_operators_do_not_import_attention_backends():
    # Use a fresh interpreter: other tests may intentionally load attention.
    code = """
import importlib.abc
import sys

class RejectAttention(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname in {
            'vllm_fl.dispatch.backends.flaggems.impl.attention',
            'vllm_fl.dispatch.backends.flaggems.impl.mla',
            'vllm_fl.dispatch.backends.flaggems.impl.custom_attention',
        }:
            raise AssertionError(f'Unrelated attention backend imported: {fullname}')

sys.meta_path.insert(0, RejectAttention())
from vllm_fl.dispatch.backends.flaggems.impl.fused_moe import topk_softmax_flaggems
from vllm_fl.dispatch.backends.flaggems.impl.quantization import (
    dynamic_per_token_quant_int8_flaggems_vllm,
)
assert callable(topk_softmax_flaggems)
assert callable(dynamic_per_token_quant_int8_flaggems_vllm)
"""
    result = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, timeout=90
    )
    assert result.returncode == 0, result.stdout + result.stderr
