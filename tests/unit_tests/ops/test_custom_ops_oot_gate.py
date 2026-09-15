# Copyright (c) 2026 BAAI. All rights reserved.

"""register_oot_ops must apply vendor patches regardless of the OOT gate.

The vendor appliers monkey-patch upstream vllm functions that in-tree
(non-OOT) code paths also go through, so applying them only on the OOT path
leaves those paths unpatched -- the regression this file guards.
"""

import pytest

import vllm_fl.ops.custom_ops as custom_ops

_KUNLUNXIN = "vllm_fl.dispatch.backends.vendor.kunlunxin.patch.apply_kunlunxin_patches"
_ASCEND = "vllm_fl.dispatch.backends.vendor.ascend.patch.apply_ascend_patches"
_SUNRISE = "vllm_fl.dispatch.backends.vendor.sunrise.patch.apply_sunrise_patches"


class _FakePlatform:
    def __init__(self, vendor_name, device_type):
        self.vendor_name = vendor_name
        self.device_type = device_type


@pytest.fixture
def applied(monkeypatch):
    """Record which patch appliers run, stubbing every side effect."""
    calls = []

    for target, label in (
        (_KUNLUNXIN, "kunlunxin"),
        (_ASCEND, "ascend"),
        (_SUNRISE, "sunrise"),
    ):
        monkeypatch.setattr(target, lambda label=label: calls.append(label))

    monkeypatch.setattr(
        custom_ops, "_patch_unquantized_moe_oracle", lambda: calls.append("oracle")
    )
    monkeypatch.setattr(
        custom_ops, "_patch_fused_moe_factory", lambda: calls.append("factory")
    )
    return calls


def _configure(monkeypatch, *, oot_enabled, vendor_name, device_type):
    monkeypatch.setattr(
        "vllm.platforms.current_platform", _FakePlatform(vendor_name, device_type)
    )
    monkeypatch.setattr("vllm_fl.utils.is_oot_enabled", lambda: oot_enabled)
    monkeypatch.setattr("vllm_fl.utils.get_oot_blacklist", lambda: [])
    monkeypatch.setattr("vllm_fl.utils.get_oot_whitelist", lambda: None)
    monkeypatch.setattr("vllm_fl.utils.use_flaggems_op", lambda op_name: True)


def test_vendor_patches_applied_with_oot_disabled(monkeypatch, applied):
    """The kunlunxin patches must land even when OOT registration is off."""
    _configure(
        monkeypatch, oot_enabled=False, vendor_name="kunlunxin", device_type="cuda"
    )

    custom_ops.register_oot_ops()

    assert "kunlunxin" in applied
    # With OOT off, the in-tree FusedMoE is what is meant to run.
    assert "factory" not in applied


def test_vendor_patches_applied_once_with_oot_enabled(monkeypatch, applied):
    """Hoisted appliers run once, not once per registered op."""
    monkeypatch.setattr(custom_ops, "OOT_OPS", {})
    _configure(
        monkeypatch, oot_enabled=True, vendor_name="kunlunxin", device_type="cuda"
    )

    custom_ops.register_oot_ops()

    assert applied.count("kunlunxin") == 1
    assert "factory" in applied


def test_no_vendor_patch_on_unrelated_platform(monkeypatch, applied):
    _configure(monkeypatch, oot_enabled=False, vendor_name="nvidia", device_type="cuda")

    custom_ops.register_oot_ops()

    assert applied == ["oracle"]
