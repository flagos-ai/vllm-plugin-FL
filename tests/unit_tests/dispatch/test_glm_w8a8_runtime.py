"""Regression guards for the integrated source selection and dynamic INT8."""
import os
from pathlib import Path
from types import SimpleNamespace

import torch


def test_dynamic_scales_do_not_select_weight_only():
    from vllm_fl.ops.fused_moe.layer import CompressedTensorsW8A8Int8MoEMethodFL
    layer = SimpleNamespace(
        w13_weight_scale=torch.ones(2, 4, 1),
        w2_weight_scale=torch.ones(2, 8, 1),
        w13_input_scale=None, w2_input_scale=None,
    )
    method = object.__new__(CompressedTensorsW8A8Int8MoEMethodFL)
    config = method.get_fused_moe_quant_config(layer)
    assert config.use_int8_w8a8
    assert not config.use_int8_w8a16
    assert config.quant_dtype == torch.int8
    assert config.per_act_token_quant


def test_imports_use_integrated_sources():
    import flag_gems
    import vllm_fl
    plugin = Path(__file__).resolve().parents[3]
    assert Path(vllm_fl.__file__).resolve().is_relative_to(plugin)
    assert Path(flag_gems.__file__).resolve().is_relative_to(
        Path(os.environ["GLM52_FLAGGEMS_ROOT"]))
