# SPDX-License-Identifier: Apache-2.0
"""Contract tests for optional T-Head native-extension initialization."""

from __future__ import annotations

import pytest

from vllm_fl.dispatch.backends.vendor.thead import bootstrap
from vllm_fl.dispatch.backends.vendor.thead.impl import native_extensions


def test_native_library_directory_requires_absolute_override(monkeypatch):
    monkeypatch.setenv("VLLM_FL_THEAD_NATIVE_LIB_DIR", "relative/libs")

    with pytest.raises(ValueError, match="must be an absolute path"):
        native_extensions.load_all_native_extensions()


def test_missing_bundle_reports_configured_directory(tmp_path, monkeypatch):
    monkeypatch.setenv("VLLM_FL_THEAD_NATIVE_LIB_DIR", str(tmp_path))

    with pytest.raises(native_extensions.NativeExtensionBundleMissingError) as error:
        native_extensions.load_all_native_extensions()

    assert {path.parent for path in error.value.missing_paths} == {tmp_path}
    assert {path.name for path in error.value.missing_paths} == set(
        native_extensions._FILES.values()
    )


def test_missing_bundle_warns_once_and_uses_fallback(tmp_path, monkeypatch):
    monkeypatch.setenv("PPU_SDK", "/opt/ppu")
    monkeypatch.setenv("VLLM_FL_THEAD_NATIVE_LIB_DIR", str(tmp_path))
    monkeypatch.setattr(bootstrap, "_WARNED_MISSING_BUNDLES", set())
    messages = []

    def record_warning(message, *args):
        messages.append(message % args)

    monkeypatch.setattr(bootstrap.logger, "warning", record_warning)

    assert bootstrap.initialize_native_extensions() is False
    assert bootstrap.initialize_native_extensions() is False

    assert len(messages) == 1
    assert "T-Head native extensions are unavailable" in messages[0]
    assert str(tmp_path) in messages[0]


def test_non_missing_load_failure_is_not_suppressed(monkeypatch):
    def fail_load():
        raise OSError("incompatible ABI")

    monkeypatch.setenv("PPU_SDK", "/opt/ppu")
    monkeypatch.setattr(
        native_extensions,
        "load_all_native_extensions",
        fail_load,
    )

    with pytest.raises(OSError, match="incompatible ABI"):
        bootstrap.initialize_native_extensions()
