# Copyright (c) 2026 BAAI. All rights reserved.

from types import SimpleNamespace

import pytest

from vllm_fl.quantization.compressed_tensors import (
    WNA16Scheme,
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
