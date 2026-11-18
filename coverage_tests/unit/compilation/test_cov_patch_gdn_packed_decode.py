# Copyright 2026 FlagOS Contributors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""CPU tests for the packed-GDN FP32-beta kernel replacement.

The Triton kernel is only executable on an accelerator, so the module's
source-marker contract (which is what the installer uses to decide whether to
patch) is asserted directly: the replacement kernel must carry the FP32-beta
marker and must not contain the vulnerable dtype-rounding expression, while a
kernel whose source still contains it is reported as needing the patch.
"""

from __future__ import annotations

import inspect
import sys
import types
from types import SimpleNamespace

import vllm_fl.patches.gdn_packed_decode as patch_module


def _vulnerable_source_kernel(*args):
    # tl.sigmoid(b_val).to(b.dtype.element_ty).to(tl.float32)
    return None


def _clean_source_kernel(*args):
    return None


def test_replacement_kernel_is_marked_and_drops_dtype_rounding():
    """The shipped kernel keeps sigmoid(b) in FP32 and is self-identifying."""
    kernel = (
        patch_module._fused_recurrent_gated_delta_rule_packed_decode_kernel_fp32_beta
    )
    assert getattr(kernel, "_fl_fp32_beta", False) is True

    source = inspect.getsource(kernel.fn)
    assert patch_module._VULNERABLE_BETA_EXPRESSION not in source
    assert "beta_val = tl.sigmoid(b_val)" in source
    assert patch_module._kernel_needs_beta_patch(kernel) is False


def test_kernel_needs_patch_detects_vulnerable_source():
    """A kernel whose source still rounds beta to the input dtype needs patching."""
    assert patch_module._kernel_needs_beta_patch(_vulnerable_source_kernel) is True


def test_kernel_needs_patch_accepts_clean_source():
    """An upstream-fixed kernel that lacks the marker is left alone."""
    assert patch_module._kernel_needs_beta_patch(_clean_source_kernel) is False


def test_kernel_needs_patch_inspects_wrapped_fn_attribute():
    """A JIT wrapper exposing ``.fn`` is inspected through that function."""
    wrapper = SimpleNamespace(fn=_vulnerable_source_kernel)
    assert patch_module._kernel_needs_beta_patch(wrapper) is True


def test_kernel_needs_patch_assumes_fix_when_source_unavailable():
    """A stripped wheel cannot be inspected, so the correction is applied."""
    assert patch_module._kernel_needs_beta_patch(len) is True


def _fake_target(monkeypatch, kernel):
    module = types.ModuleType(patch_module._TARGET_MODULE)
    module.fused_recurrent_gated_delta_rule_packed_decode_kernel = kernel
    monkeypatch.setitem(sys.modules, patch_module._TARGET_MODULE, module)
    return module


def test_installer_replaces_vulnerable_kernel_and_is_idempotent(monkeypatch):
    """The target symbol is swapped once; the swap is self-detecting."""
    target = _fake_target(monkeypatch, _vulnerable_source_kernel)
    attr = "fused_recurrent_gated_delta_rule_packed_decode_kernel"

    assert patch_module.patch_vllm_packed_gdn_beta() is True
    assert getattr(target, attr) is (
        patch_module._fused_recurrent_gated_delta_rule_packed_decode_kernel_fp32_beta
    )
    assert patch_module.patch_vllm_packed_gdn_beta() is False
    assert getattr(target, attr) is (
        patch_module._fused_recurrent_gated_delta_rule_packed_decode_kernel_fp32_beta
    )


def test_installer_leaves_already_marked_kernel(monkeypatch):
    """A kernel carrying the marker is treated as already corrected."""
    existing = SimpleNamespace(_fl_fp32_beta=True)
    target = _fake_target(monkeypatch, existing)

    assert patch_module.patch_vllm_packed_gdn_beta() is False
    assert target.fused_recurrent_gated_delta_rule_packed_decode_kernel is existing


def test_installer_leaves_upstream_fixed_kernel(monkeypatch):
    """A marker-less kernel without the vulnerable expression is not replaced."""
    target = _fake_target(monkeypatch, _clean_source_kernel)

    assert patch_module.patch_vllm_packed_gdn_beta() is False
    assert target.fused_recurrent_gated_delta_rule_packed_decode_kernel is (
        _clean_source_kernel
    )


def test_installer_noops_when_symbol_is_missing(monkeypatch):
    """An AttributeError on the target symbol is a quiet no-op."""
    module = types.ModuleType(patch_module._TARGET_MODULE)
    monkeypatch.setitem(sys.modules, patch_module._TARGET_MODULE, module)
    assert patch_module.patch_vllm_packed_gdn_beta() is False


def test_installer_noops_when_module_cannot_be_imported(monkeypatch):
    """An ImportError from the target module is a quiet no-op."""

    def explode(name):
        raise ImportError(name)

    monkeypatch.setattr(patch_module.importlib, "import_module", explode)
    assert patch_module.patch_vllm_packed_gdn_beta() is False
