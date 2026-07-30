# Copyright (c) 2026 BAAI. All rights reserved.

from types import SimpleNamespace

import pytest

from vllm_fl.quantization import compressed_tensors
from vllm_fl.quantization.compressed_tensors import (
    CompatibilityReport,
    WNA16Scheme,
    inspect_vllm_compressed_tensors_api,
    register_compressed_tensors_oot,
    validate_compressed_tensors_wna16_config,
)
from vllm_fl.quantization.marlin import is_marlin_moe_platform
from vllm_fl.quantization.wna16 import moe as moe_adapter


def _config():
    return {
        "quant_method": "compressed-tensors",
        "format": "pack-quantized",
        "quantization_status": "compressed",
        "config_groups": {
            "w4a16_g32": {
                "targets": ["re:^model\\..*\\.mlp\\..*$"],
                "weights": {
                    "num_bits": 4,
                    "type": "int",
                    "strategy": "group",
                    "group_size": 32,
                    "symmetric": True,
                    "dynamic": False,
                },
            }
        },
        "ignore": [],
    }


def test_accepts_standard_w4a16_group_config():
    schemes = validate_compressed_tensors_wna16_config(_config())
    assert schemes == [
        WNA16Scheme(
            num_bits=4,
            group_size=32,
            symmetric=True,
            strategy="group",
            has_activation_quantization=False,
        )
    ]


def test_rejects_algorithm_specific_or_nonstandard_format():
    config = _config()
    config["quant_method"] = "custom-int4"
    with pytest.raises(ValueError, match="compressed-tensors"):
        validate_compressed_tensors_wna16_config(config)


def test_rejects_activation_quantization_for_wna16():
    config = _config()
    config["config_groups"]["w4a16_g32"]["input_activations"] = {"num_bits": 8}
    with pytest.raises(ValueError, match="weight-only"):
        validate_compressed_tensors_wna16_config(config)


@pytest.mark.parametrize(
    ("is_cuda", "expected"),
    [(True, True), (False, False)],
)
def test_marlin_moe_requires_nvidia_cuda(is_cuda, expected):
    class FakePlatform:
        def is_cuda(self):
            return is_cuda

    assert is_marlin_moe_platform(FakePlatform()) is expected


def test_local_moe_adapter_is_not_installed_without_kernel(monkeypatch):
    monkeypatch.setattr(
        moe_adapter.kernels,
        "is_wna16_moe_available",
        lambda: False,
    )
    assert moe_adapter.install_fl_wna16_moe_method() is False


def test_local_moe_adapter_replaces_only_the_upstream_wna16_method(
    monkeypatch,
):
    class UpstreamMethod:
        pass

    module = SimpleNamespace(
        CompressedTensorsWNA16MoEMethod=UpstreamMethod,
    )
    monkeypatch.setattr(
        moe_adapter.kernels,
        "is_wna16_moe_available",
        lambda: True,
    )
    monkeypatch.setattr(moe_adapter, "import_module", lambda name: module)
    assert moe_adapter.install_fl_wna16_moe_method() is True
    assert issubclass(
        module.CompressedTensorsWNA16MoEMethod,
        UpstreamMethod,
    )
    assert module.CompressedTensorsWNA16MoEMethod._vllm_fl_local_wna16_moe


def test_compatibility_probe_accepts_refactored_scheme_classes(monkeypatch):
    """Class integration points remain valid when method ownership changes."""

    class UpstreamClass:
        pass

    def fake_import(module_name):
        if module_name.endswith("compressed_tensors_wNa16"):
            raise ImportError("historical module name is unavailable")
        return SimpleNamespace(
            CompressedTensorsWNA16=UpstreamClass,
            CompressedTensorsWNA16MoEMethod=UpstreamClass,
        )

    monkeypatch.setattr(compressed_tensors, "import_module", fake_import)
    report = inspect_vllm_compressed_tensors_api()

    assert report.linear_wna16 is True
    assert report.moe_wna16 is True
    assert report.details == ()


def test_moe_registration_does_not_require_linear_scheme(monkeypatch):
    report = CompatibilityReport(
        vllm_version="test",
        linear_wna16=False,
        moe_wna16=True,
    )
    calls = []

    monkeypatch.setattr(
        compressed_tensors,
        "inspect_vllm_compressed_tensors_api",
        lambda: report,
    )
    monkeypatch.setattr(
        "vllm_fl.quantization.wna16.kernels.is_wna16_moe_available",
        lambda: True,
    )
    monkeypatch.setattr("vllm_fl.utils.is_oot_enabled", lambda: True)
    monkeypatch.setattr(
        "vllm_fl.quantization.wna16.moe.install_fl_wna16_moe_method",
        lambda: calls.append("install") or True,
    )
    monkeypatch.setattr(
        "vllm_fl.quantization.marlin.configure_wna16_moe_backend",
        lambda: calls.append("configure") or "plugin-local",
    )

    assert register_compressed_tensors_oot() is report
    assert calls == ["install", "configure"]


def test_moe_install_failure_leaves_upstream_selection_unchanged(monkeypatch):
    report = CompatibilityReport(
        vllm_version="test",
        linear_wna16=True,
        moe_wna16=True,
    )
    calls = []

    monkeypatch.setattr(
        compressed_tensors,
        "inspect_vllm_compressed_tensors_api",
        lambda: report,
    )
    monkeypatch.setattr(
        "vllm_fl.quantization.wna16.kernels.is_wna16_moe_available",
        lambda: True,
    )
    monkeypatch.setattr("vllm_fl.utils.is_oot_enabled", lambda: True)

    def fail_install():
        calls.append("install")
        raise ImportError("changed upstream API")

    monkeypatch.setattr(
        "vllm_fl.quantization.wna16.moe.install_fl_wna16_moe_method",
        fail_install,
    )
    monkeypatch.setattr(
        "vllm_fl.quantization.marlin.configure_wna16_moe_backend",
        lambda: calls.append("configure") or "plugin-local",
    )

    assert register_compressed_tensors_oot() is report
    assert calls == ["install"]
