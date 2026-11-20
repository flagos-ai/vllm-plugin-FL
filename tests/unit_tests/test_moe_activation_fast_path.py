# SPDX-License-Identifier: Apache-2.0
"""FlagGems' fused fast path only supports silu; other activations must fall
back to the step-by-step path (NemotronH uses relu2_no_mul, which aborts the
engine inside flag_gems.fused_experts_impl)."""

import importlib.util
from pathlib import Path
from types import SimpleNamespace

_SPEC = importlib.util.spec_from_file_location(
    "activation_support_under_test",
    Path(__file__).resolve().parents[2] / "vllm_fl/ops/fused_moe/activation_support.py",
)
_MODULE = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_MODULE)
flaggems_fast_path_supported = _MODULE.flaggems_fast_path_supported


def test_silu_uses_fast_path():
    assert flaggems_fast_path_supported(SimpleNamespace(value="silu"))
    assert flaggems_fast_path_supported("silu")


def test_non_silu_activations_skip_fast_path():
    for value in ("relu2_no_mul", "gelu", "silu_and_mul"):
        assert not flaggems_fast_path_supported(SimpleNamespace(value=value))
        assert not flaggems_fast_path_supported(value)
