# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0
"""Select the MetaX method through the existing compressed-tensors W4A8 arm."""


def prepare():
    from vllm.model_executor.layers.quantization.compressed_tensors.compressed_tensors_moe import (
        compressed_tensors_moe_w4a8_int8 as method_module,
    )

    from vllm_fl.quantization.metax_w4a8 import MetaXCompressedTensorsW4A8Method

    # Resolve the required native symbol before any of the group is installed.
    if not callable(getattr(method_module, "CompressedTensorsW4A8Int8MoEMethod", None)):
        raise RuntimeError("native compressed-tensors W4A8 method is unavailable")
    return [
        (
            method_module,
            "CompressedTensorsW4A8Int8MoEMethod",
            MetaXCompressedTensorsW4A8Method,
        )
    ]
