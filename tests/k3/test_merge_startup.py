# Copyright (c) 2026 BAAI. All rights reserved.

"""Regression guards for merging the native and open MetaX startup routes."""

from unittest.mock import Mock

import pytest


def prepare_registration(monkeypatch, *, k3, open_runtime):
    import vllm_fl as plugin
    from vllm_fl.patches import metax_open
    from vllm_fl.utils import DeviceInfo

    monkeypatch.setattr(plugin, "is_kimi_k3_profile", lambda: k3)
    monkeypatch.setattr(plugin, "verify_dependencies", Mock())
    monkeypatch.setattr(plugin, "_register_in_progress", False)
    monkeypatch.setattr(plugin, "_arm_cpu_platform", lambda: None)
    monkeypatch.setattr(plugin, "_init_vendor_device", Mock())
    monkeypatch.setattr(plugin, "_get_op_config", Mock())
    monkeypatch.setattr(plugin, "_force_adopt_platform", Mock())
    monkeypatch.setattr(plugin, "_patch_flash_attn_import", Mock())
    monkeypatch.setattr(DeviceInfo, "vendor_name", property(lambda _: "metax"))
    monkeypatch.setattr(metax_open, "uses_open_metax_runtime", lambda: open_runtime)
    monkeypatch.delenv("FLAGCX_PATH", raising=False)
    return plugin


@pytest.mark.parametrize("k3", [False, True])
def test_required_open_runtime_failure_is_not_swallowed(monkeypatch, k3):
    plugin = prepare_registration(monkeypatch, k3=k3, open_runtime=True)
    failure = Mock(side_effect=RuntimeError("required open ABI failed"))
    monkeypatch.setattr(plugin, "_patch_custom_ops", failure)
    with pytest.raises(RuntimeError, match="required open ABI failed"):
        plugin.register()
    failure.assert_called_once()
    assert plugin._register_in_progress is False
    plugin._patch_flash_attn_import.assert_not_called()


def test_legacy_optional_hook_failure_keeps_upstream_registration(monkeypatch):
    plugin = prepare_registration(monkeypatch, k3=False, open_runtime=False)

    def optional_hook():
        raise ValueError("optional")

    monkeypatch.setattr(plugin, "_patch_custom_ops", optional_hook)
    assert plugin.register() == "vllm_fl.platform.PlatformFL"
    assert plugin._register_in_progress is False
    plugin._patch_flash_attn_import.assert_called_once()


def test_reentrant_registration_keeps_the_platform_qualname(monkeypatch):
    plugin = prepare_registration(monkeypatch, k3=False, open_runtime=False)
    monkeypatch.setattr(plugin, "_register_in_progress", True)
    assert plugin.register() == "vllm_fl.platform.PlatformFL"
    plugin._init_vendor_device.assert_not_called()


def test_open_runtime_does_not_retry_legacy_kernel_patches(monkeypatch):
    from vllm_fl.dispatch.backends.vendor.metax import patches

    monkeypatch.setattr(patches, "uses_open_metax_runtime", lambda: True)
    importer = Mock(side_effect=AssertionError("legacy patch must remain inactive"))
    monkeypatch.setattr(patches.importlib, "import_module", importer)
    assert patches._import_patches() == {}
    assert patches.ensure_patches() == {}
    importer.assert_not_called()
