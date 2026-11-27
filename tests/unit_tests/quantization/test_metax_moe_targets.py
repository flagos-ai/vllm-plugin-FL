# Copyright (c) 2026 BAAI. All rights reserved.

"""CPU checks for legacy MoE target compatibility; no weight/kernel execution."""

import copy
import importlib.util
from pathlib import Path

import pytest

from vllm.model_executor.layers.fused_moe import RoutedExperts
from vllm.model_executor.layers.quantization.compressed_tensors.compressed_tensors import (
    CompressedTensorsConfig,
)


@pytest.fixture
def target_patch(monkeypatch):
    original = CompressedTensorsConfig._add_fused_moe_to_target_scheme_map
    monkeypatch.setattr(
        CompressedTensorsConfig, "_add_fused_moe_to_target_scheme_map", original
    )
    # Test this installer without applying unrelated vendor patches to pytest.
    path = (
        Path(__file__).resolve().parents[3]
        / "vllm_fl/dispatch/backends/vendor/metax/patches/compressed_tensors_moe_targets.py"
    )
    spec = importlib.util.spec_from_file_location("metax_moe_target_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    for target, name, replacement in module.prepare():
        setattr(target, name, replacement)
    return module


def quant_config(targets, ignore=()):
    groups = {}
    for target, bits in targets.items():
        groups[target] = {
            "targets": [target],
            "weights": {
                "num_bits": bits,
                "type": "int",
                "strategy": "channel",
                "symmetric": True,
            },
            "input_activations": {
                "num_bits": 8,
                "type": "int",
                "strategy": "token",
                "symmetric": True,
                "dynamic": True,
            },
        }
    raw = {
        "quant_method": "compressed-tensors",
        "format": "int-quantized",
        "config_groups": groups,
        "ignore": list(ignore),
    }
    return CompressedTensorsConfig.from_config(copy.deepcopy(raw))


@pytest.mark.parametrize(
    ("targets", "expected"),
    [
        ({"Linear": 8, "FusedMoE": 4}, 4),
        ({"FusedMoE": 4}, 4),
        ({"Linear": 8}, 8),
        ({"Linear": 8, "FusedMoE": 4, "RoutedExperts": 8}, 8),
    ],
)
def test_expert_target_precedence(target_patch, targets, expected):
    config = quant_config(targets)
    config._add_fused_moe_to_target_scheme_map()
    assert config.target_scheme_map["RoutedExperts"]["weights"].num_bits == expected
    # Real class-based matching; this fixture has no parameters or backend.
    experts = object.__new__(RoutedExperts)
    assert (
        config.get_scheme_dict(experts, "model.layers.0.experts.w1")["weights"].num_bits
        == expected
    )
    if "Linear" in targets:
        assert config.target_scheme_map["Linear"]["weights"].num_bits == 8


def test_missing_target_stays_unquantized(target_patch):
    config = quant_config({})
    config._add_fused_moe_to_target_scheme_map()
    assert config.target_scheme_map == {}


def test_repeated_install_and_rule_application_are_idempotent(target_patch):
    installed = CompressedTensorsConfig._add_fused_moe_to_target_scheme_map
    assert target_patch.prepare() == []
    assert CompressedTensorsConfig._add_fused_moe_to_target_scheme_map is installed
    config = quant_config({"Linear": 8, "FusedMoE": 4})
    config._add_fused_moe_to_target_scheme_map()
    first = config.target_scheme_map["RoutedExperts"]
    config._add_fused_moe_to_target_scheme_map()
    assert config.target_scheme_map["RoutedExperts"] is first
    assert first is config.target_scheme_map["FusedMoE"]


def test_ignore_and_layer_specific_rules_keep_priority(target_patch):
    config = quant_config(
        {"Linear": 8, "FusedMoE": 4}, ignore=["model.layers.0.experts.0.w1"]
    )
    config.target_scheme_map["re:.*layers.1.experts.*"] = config.target_scheme_map[
        "Linear"
    ]
    config._add_fused_moe_to_target_scheme_map()
    experts = object.__new__(RoutedExperts)
    assert config.get_scheme_dict(experts, "model.layers.0.experts.0.w1") is None
    assert (
        config.get_scheme_dict(experts, "model.layers.1.experts.0.w1")[
            "weights"
        ].num_bits
        == 8
    )
    assert (
        config.get_scheme_dict(experts, "model.layers.2.experts.0.w1")[
            "weights"
        ].num_bits
        == 4
    )
