# Copyright (c) 2026 BAAI. All rights reserved.

from unittest.mock import MagicMock, patch

from vllm_fl.ops.fused_moe import layer


def _runner_with_quant_method(quant_method):
    runner = MagicMock()
    runner._quant_method = quant_method
    runner.moe_config = MagicMock()
    return runner


def test_fused_moe_fl_replaces_unquantized_method():
    quant_method = MagicMock(spec=layer.UnquantizedFusedMoEMethod)
    runner = _runner_with_quant_method(quant_method)
    replacement = MagicMock()

    with (
        patch.object(layer, "_OrigFusedMoE", return_value=runner),
        patch.object(
            layer,
            "UnquantizedFusedMoEMethodFL",
            return_value=replacement,
        ) as replacement_cls,
        patch.object(layer, "replace_router_with_fl") as replace_router,
    ):
        result = layer.FusedMoEFL(test_arg=True)

    assert result is runner
    replacement_cls.assert_called_once_with(runner.moe_config)
    runner._replace_quant_method.assert_called_once_with(replacement)
    replace_router.assert_called_once_with()


def test_fused_moe_fl_preserves_quantized_method():
    quant_method = object()
    runner = _runner_with_quant_method(quant_method)

    with (
        patch.object(layer, "_OrigFusedMoE", return_value=runner),
        patch.object(layer, "UnquantizedFusedMoEMethodFL") as replacement_cls,
        patch.object(layer, "replace_router_with_fl") as replace_router,
        patch.object(layer.logger, "info_once") as info_once,
    ):
        result = layer.FusedMoEFL()

    assert result is runner
    replacement_cls.assert_not_called()
    runner._replace_quant_method.assert_not_called()
    replace_router.assert_called_once_with()
    info_once.assert_called_once_with(
        "Preserving upstream quantized MoE method %s in FusedMoEFL.",
        "object",
    )
