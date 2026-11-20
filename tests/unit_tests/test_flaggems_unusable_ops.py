# SPDX-License-Identifier: Apache-2.0
"""FlagGems' aten::clone kernel asserts a CUDA device; CPU clones during engine
init (torch._dynamo.utils.preserve_rng_state) aborted Ling/Ring-mini with
"Expected a cuda device, but got: cpu". clone must stay on stock torch in every
enable mode."""

import importlib.util
from pathlib import Path

_SPEC = importlib.util.spec_from_file_location(
    "flaggems_ops_under_test",
    Path(__file__).resolve().parents[2] / "vllm_fl/patches/flaggems_ops.py",
)
_MODULE = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_MODULE)
exclude_unusable = _MODULE.exclude_unusable
with_unusable = _MODULE.with_unusable
FLAGGEMS_UNUSABLE_OPS = _MODULE.FLAGGEMS_UNUSABLE_OPS


def test_clone_is_always_excluded():
    assert FLAGGEMS_UNUSABLE_OPS == ("clone",)


def test_blacklist_gets_clone_appended_deduplicated():
    assert with_unusable(["clone"]) == ["clone"]
    assert with_unusable(["rms_norm"]) == ["rms_norm", "clone"]
    assert with_unusable(None, ["silu"]) == ["silu", "clone"]


def test_whitelist_drops_clone():
    assert exclude_unusable(["clone", "rms_norm"]) == ["rms_norm"]
    assert exclude_unusable(["rms_norm"]) == ["rms_norm"]
    assert exclude_unusable(None) == []
