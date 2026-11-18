# SPDX-License-Identifier: Apache-2.0
"""Explicit target-only W8A8 metadata contract; no conversion or dequantization."""


def validate_w8a8_contract(config):
    if (
        config.get("quant_method") != "compressed-tensors"
        or config.get("format") != "int-quantized"
    ):
        raise ValueError("Flash W8A8 requires compressed-tensors int-quantized weights")
    groups = config.get("config_groups", {})
    if not groups:
        raise ValueError("Missing W8A8 groups")
    expected_weight = {
        "num_bits": 8,
        "type": "int",
        "strategy": "channel",
        "symmetric": True,
        "dynamic": False,
    }
    expected_input = {
        "num_bits": 8,
        "type": "int",
        "strategy": "token",
        "symmetric": True,
        "dynamic": True,
    }
    for name, group in groups.items():
        for field, expected in (
            ("weights", expected_weight),
            ("input_activations", expected_input),
        ):
            spec = group.get(field, {})
            if any(spec.get(k) != v for k, v in expected.items()):
                raise ValueError(f"Unsupported Flash W8A8 {name}.{field}: {spec}")
        if group.get("output_activations") is not None:
            raise ValueError("Flash W8A8 output quantization is not validated")
    return None
