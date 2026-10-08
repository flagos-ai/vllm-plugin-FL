# Copyright (c) 2026 BAAI. All rights reserved.

"""Unit tests for the HY4 indexer WK pairing loader."""

from types import SimpleNamespace

import pytest
import torch

from vllm.model_executor.models.utils import PPMissingLayer

from vllm_fl.model_loader.hy_v4_indexer import (
    FP8_FORMAT,
    MXFP8_FORMAT,
    IndexerWKLoader,
    dequantize_fp8_wk,
    dequantize_mxfp8_wk,
    normalize_quant_format,
    quant_metadata_from_quant_config,
)
from vllm_fl.models import hy_v4
from vllm_fl.models.hy_v4 import HYV4ForCausalLM

_FP8_BLOCK = 32
_MXFP8_GROUP = 32


class _StubParam:
    """Records every ``weight_loader`` call without a real vLLM layer."""

    def __init__(self, rows: int, cols: int) -> None:
        self.data = torch.zeros(rows, cols, dtype=torch.bfloat16)
        self.loads: list[tuple[int, torch.Tensor]] = []

    def weight_loader(self, param, weight: torch.Tensor, shard_id: int) -> None:
        param.loads.append((shard_id, weight.detach().clone()))
        with torch.no_grad():
            param.data[: weight.shape[0]].copy_(weight)


def _fused_name(layer_prefix: str) -> str:
    return f"{layer_prefix}.wk_weights_proj.weight"


def _params_for(*layers: str, rows: int = 128, cols: int = 64) -> dict:
    return {_fused_name(layer): _StubParam(rows, cols) for layer in layers}


def _mxfp8_weight(rows: int, cols: int) -> torch.Tensor:
    torch.manual_seed(rows * 100 + cols)
    return torch.randn(rows, cols).to(torch.float8_e4m3fn)


def _mxfp8_scale(rows: int, cols: int) -> torch.Tensor:
    return (
        torch.arange(rows * (cols // _MXFP8_GROUP), dtype=torch.int64)
        .remainder(5)
        .reshape(rows, cols // _MXFP8_GROUP)
        .add(125)
        .to(torch.uint8)
    )


def _fp8_weight(rows: int, cols: int) -> torch.Tensor:
    torch.manual_seed(rows * 1000 + cols)
    return torch.randn(rows, cols).to(torch.float8_e4m3fn)


def _fp8_scale(rows: int, cols: int) -> torch.Tensor:
    groups = (rows // _FP8_BLOCK, cols // _FP8_BLOCK)
    return torch.linspace(0.25, 2.0, groups[0] * groups[1]).reshape(groups)


def _reference_mxfp8(weight: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    scales = torch.exp2(scale.to(torch.int16).float() - 127.0)
    scales = scales.repeat_interleave(_MXFP8_GROUP, dim=-1)
    return (weight.float() * scales).to(torch.bfloat16)


def _reference_fp8(weight: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    scales = (
        scale.float()
        .repeat_interleave(_FP8_BLOCK, dim=0)
        .repeat_interleave(_FP8_BLOCK, dim=1)
    )
    return (weight.float() * scales).to(torch.bfloat16)


def test_mxfp8_weight_before_scale_matches_reference():
    layer = "model.layers.0.self_attn.indexer"
    params = _params_for(layer, rows=64, cols=64)
    loader = IndexerWKLoader(params, set())

    weight = _mxfp8_weight(64, 64)
    scale = _mxfp8_scale(64, 64)
    assert loader.consume(f"{layer}.wk.weight", weight) is True
    assert loader.consume(f"{layer}.wk.weight_scale", scale) is True
    assert loader.finish() == {_fused_name(layer)}

    param = params[_fused_name(layer)]
    assert [shard for shard, _ in param.loads] == [0]
    torch.testing.assert_close(param.loads[0][1], _reference_mxfp8(weight, scale))


def test_mxfp8_scale_before_weight_matches_reference():
    layer = "model.layers.1.self_attn.indexer"
    params = _params_for(layer, rows=64, cols=64)
    loader = IndexerWKLoader(params, set())

    weight = _mxfp8_weight(64, 64)
    scale = _mxfp8_scale(64, 64)
    assert loader.consume(f"{layer}.wk.weight_scale", scale) is True
    assert loader.consume(f"{layer}.wk.weight", weight) is True
    assert loader.finish() == {_fused_name(layer)}

    param = params[_fused_name(layer)]
    assert [shard for shard, _ in param.loads] == [0]
    torch.testing.assert_close(param.loads[0][1], _reference_mxfp8(weight, scale))


def test_fp8_weight_before_scale_matches_reference():
    layer = "model.layers.2.self_attn.indexer"
    params = _params_for(layer, rows=64, cols=64)
    loader = IndexerWKLoader(params, set())

    weight = _fp8_weight(64, 64)
    scale = _fp8_scale(64, 64)
    assert loader.consume(f"{layer}.wk.weight", weight) is True
    assert loader.consume(f"{layer}.wk.weight_scale_inv", scale) is True
    assert loader.finish() == {_fused_name(layer)}

    param = params[_fused_name(layer)]
    assert [shard for shard, _ in param.loads] == [0]
    torch.testing.assert_close(param.loads[0][1], _reference_fp8(weight, scale))


def test_fp8_scale_before_weight_matches_reference():
    layer = "model.layers.3.self_attn.indexer"
    params = _params_for(layer, rows=64, cols=64)
    loader = IndexerWKLoader(params, set())

    weight = _fp8_weight(64, 64)
    scale = _fp8_scale(64, 64)
    assert loader.consume(f"{layer}.wk.weight_scale_inv", scale) is True
    assert loader.consume(f"{layer}.wk.weight", weight) is True
    assert loader.finish() == {_fused_name(layer)}

    param = params[_fused_name(layer)]
    assert [shard for shard, _ in param.loads] == [0]
    torch.testing.assert_close(param.loads[0][1], _reference_fp8(weight, scale))


def test_interleaved_layers_are_paired_independently():
    layers = [
        "model.layers.0.self_attn.indexer",
        "model.layers.1.self_attn.indexer",
    ]
    params = _params_for(*layers, rows=64, cols=64)
    loader = IndexerWKLoader(params, set())

    weights = {layer: _mxfp8_weight(64, 64) for layer in layers}
    scales = {layer: _mxfp8_scale(64, 64) for layer in layers}

    assert loader.consume(f"{layers[0]}.wk.weight", weights[layers[0]]) is True
    assert loader.consume(f"{layers[1]}.wk.weight", weights[layers[1]]) is True
    assert loader.consume(f"{layers[1]}.wk.weight_scale", scales[layers[1]]) is True
    assert loader.consume(f"{layers[0]}.wk.weight_scale", scales[layers[0]]) is True
    assert loader.finish() == {_fused_name(layer) for layer in layers}

    for layer in layers:
        param = params[_fused_name(layer)]
        assert [shard for shard, _ in param.loads] == [0]
        torch.testing.assert_close(
            param.loads[0][1],
            _reference_mxfp8(weights[layer], scales[layer]),
        )


def test_format_is_chosen_per_layer_not_from_one_dtype():
    fp8_layer = "model.layers.0.self_attn.indexer"
    mxfp8_layer = "model.layers.1.self_attn.indexer"
    params = _params_for(fp8_layer, mxfp8_layer, rows=64, cols=64)
    loader = IndexerWKLoader(params, set())

    fp8_weight, fp8_scale = _fp8_weight(64, 64), _fp8_scale(64, 64)
    mxfp8_weight, mxfp8_scale = _mxfp8_weight(64, 64), _mxfp8_scale(64, 64)

    loader.consume(f"{fp8_layer}.wk.weight", fp8_weight)
    loader.consume(f"{mxfp8_layer}.wk.weight", mxfp8_weight)
    loader.consume(f"{fp8_layer}.wk.weight_scale_inv", fp8_scale)
    loader.consume(f"{mxfp8_layer}.wk.weight_scale", mxfp8_scale)
    assert loader.finish() == {_fused_name(fp8_layer), _fused_name(mxfp8_layer)}

    torch.testing.assert_close(
        params[_fused_name(fp8_layer)].loads[0][1],
        _reference_fp8(fp8_weight, fp8_scale),
    )
    torch.testing.assert_close(
        params[_fused_name(mxfp8_layer)].loads[0][1],
        _reference_mxfp8(mxfp8_weight, mxfp8_scale),
    )


def test_pp_missing_layers_are_skipped_before_caching():
    loader = IndexerWKLoader({}, {"model.layers.1."})
    assert (
        loader.consume(
            "model.layers.1.self_attn.indexer.wk.weight",
            _mxfp8_weight(64, 64),
        )
        is True
    )
    assert (
        loader.consume(
            "model.layers.1.self_attn.indexer.wk.weight_scale",
            _mxfp8_scale(64, 64),
        )
        is True
    )
    assert loader.finish() == set()


def test_pp_missing_boundary_does_not_match_longer_layer_index():
    layer = "model.layers.10.self_attn.indexer"
    params = _params_for(layer, rows=64, cols=64)
    loader = IndexerWKLoader(params, {"model.layers.1"})

    weight, scale = _mxfp8_weight(64, 64), _mxfp8_scale(64, 64)
    loader.consume(f"{layer}.wk.weight", weight)
    loader.consume(f"{layer}.wk.weight_scale", scale)

    assert loader.finish() == {_fused_name(layer)}
    torch.testing.assert_close(
        params[_fused_name(layer)].loads[0][1],
        _reference_mxfp8(weight, scale),
    )


def test_local_layer_without_target_parameter_raises():
    loader = IndexerWKLoader({}, set())
    loader.consume("model.layers.0.self_attn.indexer.wk.weight", _mxfp8_weight(64, 64))
    with pytest.raises(ValueError, match="missing for local layer"):
        loader.consume(
            "model.layers.0.self_attn.indexer.wk.weight_scale",
            _mxfp8_scale(64, 64),
        )


def test_missing_weight_is_reported_at_finish():
    layer = "model.layers.0.self_attn.indexer"
    loader = IndexerWKLoader(_params_for(layer), set())
    loader.consume(f"{layer}.wk.weight_scale", _mxfp8_scale(64, 64))
    with pytest.raises(ValueError, match="missing weight"):
        loader.finish()


def test_missing_scale_is_reported_at_finish():
    layer = "model.layers.0.self_attn.indexer"
    loader = IndexerWKLoader(_params_for(layer), set())
    loader.consume(f"{layer}.wk.weight", _mxfp8_weight(64, 64))
    with pytest.raises(ValueError, match="missing scale"):
        loader.finish()


def test_both_scale_kinds_are_reported_as_conflict():
    layer = "model.layers.0.self_attn.indexer"
    loader = IndexerWKLoader(_params_for(layer, rows=64, cols=64), set())
    loader.consume(f"{layer}.wk.weight_scale_inv", _fp8_scale(64, 64))
    loader.consume(f"{layer}.wk.weight_scale", _mxfp8_scale(64, 64))
    loader.consume(f"{layer}.wk.weight", _mxfp8_weight(64, 64))
    with pytest.raises(ValueError, match="conflicting scales"):
        loader.finish()


def test_duplicate_input_is_reported_at_finish():
    layer = "model.layers.0.self_attn.indexer"
    loader = IndexerWKLoader(_params_for(layer, rows=64, cols=64), set())
    weight = _mxfp8_weight(64, 64)
    loader.consume(f"{layer}.wk.weight", weight)
    loader.consume(f"{layer}.wk.weight", weight)
    loader.consume(f"{layer}.wk.weight_scale", _mxfp8_scale(64, 64))
    with pytest.raises(ValueError, match="duplicate input"):
        loader.finish()


def test_conflicting_scale_after_load_raises():
    layer = "model.layers.0.self_attn.indexer"
    loader = IndexerWKLoader(_params_for(layer, rows=64, cols=64), set())
    loader.consume(f"{layer}.wk.weight", _mxfp8_weight(64, 64))
    loader.consume(f"{layer}.wk.weight_scale", _mxfp8_scale(64, 64))
    with pytest.raises(ValueError, match="already loaded"):
        loader.consume(f"{layer}.wk.weight_scale_inv", _fp8_scale(64, 64))


def test_duplicate_weight_after_load_raises():
    layer = "model.layers.0.self_attn.indexer"
    loader = IndexerWKLoader(_params_for(layer, rows=64, cols=64), set())
    weight = _mxfp8_weight(64, 64)
    loader.consume(f"{layer}.wk.weight", weight)
    loader.consume(f"{layer}.wk.weight_scale", _mxfp8_scale(64, 64))
    with pytest.raises(ValueError, match="already loaded"):
        loader.consume(f"{layer}.wk.weight", weight)


def test_wrong_scale_dtype_raises():
    layer = "model.layers.0.self_attn.indexer"
    loader = IndexerWKLoader(_params_for(layer), set())
    with pytest.raises(ValueError, match="torch.uint8"):
        loader.consume(f"{layer}.wk.weight_scale", torch.ones(4, 2))
    with pytest.raises(ValueError, match="floating dtype"):
        loader.consume(
            f"{layer}.wk.weight_scale_inv",
            torch.ones(4, 2, dtype=torch.uint8),
        )
    with pytest.raises(ValueError, match="unsupported dtype"):
        loader.consume(f"{layer}.wk.weight", torch.ones(4, 64, dtype=torch.int8))


def test_fp8_non_divisible_scale_shape_raises():
    layer = "model.layers.0.self_attn.indexer"
    loader = IndexerWKLoader(_params_for(layer, rows=64, cols=64), set())
    loader.consume(f"{layer}.wk.weight", _fp8_weight(64, 64))
    with pytest.raises(ValueError, match="do not divide"):
        loader.consume(
            f"{layer}.wk.weight_scale_inv", torch.ones(2, 3, dtype=torch.float32)
        )


def test_fp8_mismatched_row_grouping_raises():
    layer = "model.layers.0.self_attn.indexer"
    loader = IndexerWKLoader(_params_for(layer, rows=64, cols=64), set())
    loader.consume(f"{layer}.wk.weight", _fp8_weight(64, 64))
    with pytest.raises(ValueError, match="block grouping"):
        loader.consume(
            f"{layer}.wk.weight_scale_inv", torch.ones(1, 2, dtype=torch.float32)
        )


def test_mxfp8_non_divisible_scale_shape_raises():
    layer = "model.layers.0.self_attn.indexer"
    loader = IndexerWKLoader(_params_for(layer, rows=4, cols=65), set())
    loader.consume(f"{layer}.wk.weight", _mxfp8_weight(4, 65))
    with pytest.raises(ValueError, match="group size"):
        loader.consume(f"{layer}.wk.weight_scale", torch.ones(4, 2, dtype=torch.uint8))


def test_quant_metadata_conflict_raises():
    layer = "model.layers.0.self_attn.indexer"
    loader = IndexerWKLoader(
        _params_for(layer, rows=64, cols=64),
        set(),
        quant_metadata="mxfp8",
    )
    loader.consume(f"{layer}.wk.weight", _fp8_weight(64, 64))
    with pytest.raises(ValueError, match="format conflict"):
        loader.consume(f"{layer}.wk.weight_scale_inv", _fp8_scale(64, 64))


def test_quant_metadata_mapping_validates_each_layer():
    fp8_layer = "model.layers.0.self_attn.indexer"
    mxfp8_layer = "model.layers.1.self_attn.indexer"
    params = _params_for(fp8_layer, mxfp8_layer, rows=64, cols=64)
    loader = IndexerWKLoader(
        params,
        set(),
        quant_metadata={fp8_layer: "fp8", mxfp8_layer: "MXFP8"},
    )
    loader.consume(f"{fp8_layer}.wk.weight", _fp8_weight(64, 64))
    loader.consume(f"{fp8_layer}.wk.weight_scale_inv", _fp8_scale(64, 64))
    loader.consume(f"{mxfp8_layer}.wk.weight", _mxfp8_weight(64, 64))
    loader.consume(f"{mxfp8_layer}.wk.weight_scale", _mxfp8_scale(64, 64))
    assert loader.finish() == {
        _fused_name(fp8_layer),
        _fused_name(mxfp8_layer),
    }


def test_bf16_and_other_weights_fall_through():
    layer = "model.layers.0.self_attn.indexer"
    loader = IndexerWKLoader(_params_for(layer), set())
    assert (
        loader.consume(f"{layer}.wk.weight", torch.zeros(4, 64, dtype=torch.bfloat16))
        is False
    )
    assert loader.consume(f"{layer}.weights_proj.weight", torch.zeros(2, 64)) is False
    assert (
        loader.consume("model.layers.0.mlp.gate_proj.weight", torch.zeros(4, 64))
        is False
    )
    assert loader.finish() == set()


def test_quant_format_helpers():
    assert normalize_quant_format("MXFP8") == MXFP8_FORMAT
    assert normalize_quant_format("modelopt_mxfp8") == MXFP8_FORMAT
    assert normalize_quant_format("fp8") == FP8_FORMAT
    assert normalize_quant_format("modelopt") is None
    assert normalize_quant_format(None) is None

    class _QuantConfig:
        def __init__(self, name):
            self._name = name

        def get_name(self):
            return self._name

    assert (
        quant_metadata_from_quant_config(_QuantConfig("modelopt_mxfp8")) == MXFP8_FORMAT
    )
    assert quant_metadata_from_quant_config(_QuantConfig("fp8")) == FP8_FORMAT
    assert quant_metadata_from_quant_config(_QuantConfig("modelopt")) is None
    assert quant_metadata_from_quant_config(None) is None


def test_dequantize_helpers_are_self_consistent():
    weight = _mxfp8_weight(4, 64)
    scale = _mxfp8_scale(4, 64)
    torch.testing.assert_close(
        dequantize_mxfp8_wk(weight, scale), _reference_mxfp8(weight, scale)
    )

    fp8_weight = _fp8_weight(64, 64)
    fp8_scale = _fp8_scale(64, 64)
    torch.testing.assert_close(
        dequantize_fp8_wk(fp8_weight, fp8_scale),
        _reference_fp8(fp8_weight, fp8_scale),
    )


class _IntegrationModel:
    """Minimal carrier so the real ``load_weights`` body can run on CPU."""


def _make_integration_model(params, named_modules=None, quant_config=None):
    model = _IntegrationModel()
    model.config = SimpleNamespace(
        num_attention_heads=4,
        num_hidden_layers=1,
        n_routed_experts=2,
    )
    model.quant_config = quant_config
    model.named_parameters = lambda recurse=True: iter(params.items())
    modules = [("", model)] if named_modules is None else named_modules
    model.named_modules = lambda recurse=True: iter(modules)
    model._packed_expert_target = HYV4ForCausalLM._packed_expert_target
    model._load_all_experts = lambda *args, **kwargs: (_ for _ in ()).throw(
        AssertionError("packed expert path must not run")
    )
    return model


def _patch_cpu_parallelism(monkeypatch):
    monkeypatch.setattr(hy_v4, "get_tensor_model_parallel_rank", lambda: 0)
    monkeypatch.setattr(hy_v4, "get_tensor_model_parallel_world_size", lambda: 1)


def test_load_weights_mxfp8_integration(monkeypatch):
    from vllm.model_executor.models import utils as vllm_model_utils

    monkeypatch.setattr(vllm_model_utils, "_model_to_pp_missing_layer_names", {})
    _patch_cpu_parallelism(monkeypatch)

    layer = "model.layers.0.self_attn.indexer"
    fused = _fused_name(layer)
    params = {fused: _StubParam(6, 32)}
    model = _make_integration_model(params)

    weight = torch.full((4, 32), 2.0, dtype=torch.float8_e4m3fn)
    scale = torch.tensor([[127], [128], [126], [127]], dtype=torch.uint8)
    proj = torch.full((2, 32), 5.0, dtype=torch.bfloat16)

    loaded = HYV4ForCausalLM.load_weights(
        model,
        [
            (f"{layer}.wk.weight_scale", scale),
            (f"{layer}.weights_proj.weight", proj),
            (f"{layer}.wk.weight", weight),
        ],
    )

    assert loaded == {fused}
    param = params[fused]
    shards = [shard for shard, _ in param.loads]
    assert sorted(shards) == [0, 1]
    wk_weight = next(value for shard, value in param.loads if shard == 0)
    proj_weight = next(value for shard, value in param.loads if shard == 1)
    torch.testing.assert_close(wk_weight, _reference_mxfp8(weight, scale))
    torch.testing.assert_close(proj_weight, proj)


def test_load_weights_fp8_integration(monkeypatch):
    from vllm.model_executor.models import utils as vllm_model_utils

    monkeypatch.setattr(vllm_model_utils, "_model_to_pp_missing_layer_names", {})
    _patch_cpu_parallelism(monkeypatch)

    layer = "model.layers.0.self_attn.indexer"
    fused = _fused_name(layer)
    params = {fused: _StubParam(34, 64)}
    model = _make_integration_model(params)

    weight = _fp8_weight(32, 64)
    scale = _fp8_scale(32, 64)
    proj = torch.full((2, 64), 7.0, dtype=torch.bfloat16)

    loaded = HYV4ForCausalLM.load_weights(
        model,
        [
            (f"{layer}.wk.weight", weight),
            (f"{layer}.wk.weight_scale_inv", scale),
            (f"{layer}.weights_proj.weight", proj),
        ],
    )

    assert loaded == {fused}
    param = params[fused]
    shards = [shard for shard, _ in param.loads]
    assert sorted(shards) == [0, 1]
    wk_weight = next(value for shard, value in param.loads if shard == 0)
    torch.testing.assert_close(wk_weight, _reference_fp8(weight, scale))


def test_load_weights_skips_pp_missing_indexer(monkeypatch):
    from vllm.model_executor.models import utils as vllm_model_utils

    monkeypatch.setattr(vllm_model_utils, "_model_to_pp_missing_layer_names", {})
    _patch_cpu_parallelism(monkeypatch)

    local_layer = "model.layers.0.self_attn.indexer"
    missing_layer = "model.layers.1.self_attn.indexer"
    fused = _fused_name(local_layer)
    params = {fused: _StubParam(6, 32)}
    model = _make_integration_model(params)
    model.named_modules = lambda recurse=True: iter(
        [("", model), ("model.layers.1", PPMissingLayer())]
    )

    local_weight = torch.full((4, 32), 3.0, dtype=torch.float8_e4m3fn)
    local_scale = torch.tensor([[127], [127], [127], [127]], dtype=torch.uint8)
    missing_weight = torch.full((4, 32), 9.0, dtype=torch.float8_e4m3fn)
    missing_scale = torch.tensor([[127], [127], [127], [127]], dtype=torch.uint8)

    loaded = HYV4ForCausalLM.load_weights(
        model,
        [
            (f"{missing_layer}.wk.weight", missing_weight),
            (f"{missing_layer}.wk.weight_scale", missing_scale),
            (f"{local_layer}.wk.weight", local_weight),
            (f"{local_layer}.wk.weight_scale", local_scale),
            (f"{local_layer}.weights_proj.weight", torch.ones(2, 32)),
        ],
    )

    assert loaded == {fused}
    param = params[fused]
    assert [shard for shard, _ in param.loads] == [0, 1]
    torch.testing.assert_close(
        param.loads[0][1], _reference_mxfp8(local_weight, local_scale)
    )


@pytest.mark.parametrize(
    "field,dtype",
    [
        ("weight", torch.int8),
        ("weight_scale", torch.float32),
        ("weight_scale_inv", torch.uint8),
    ],
)
def test_pp_missing_fields_are_skipped_before_dtype_validation(field, dtype):
    loader = IndexerWKLoader({}, ["model.layers.1"])
    name = f"model.layers.1.self_attn.indexer.wk.{field}"
    assert loader.consume(name, torch.zeros(2, 32, dtype=dtype))
    assert loader.finish() == set()
    with pytest.raises(ValueError, match="dtype|torch.uint8"):
        loader.consume(
            name.replace("layers.1.", "layers.10."), torch.zeros(2, 32, dtype=dtype)
        )


@pytest.mark.parametrize("first", ["plain", "pending", "quantized"])
def test_plain_wk_cannot_overwrite_or_mix_with_another_weight(first):
    layer = "model.layers.0.self_attn.indexer"
    loader = IndexerWKLoader(_params_for(layer, rows=32, cols=64))
    name = f"{layer}.wk.weight"
    if first == "plain":
        assert not loader.consume(name, torch.ones(32, 64, dtype=torch.bfloat16))
    else:
        loader.consume(name, _fp8_weight(32, 64))
        if first == "quantized":
            loader.consume(f"{layer}.wk.weight_scale_inv", _fp8_scale(32, 64))
    with pytest.raises(ValueError, match="model.layers.0.self_attn.indexer"):
        loader.consume(name, torch.zeros(32, 64, dtype=torch.bfloat16))


@pytest.mark.parametrize("field", ["weight", "weight_scale_inv"])
def test_quantized_input_after_plain_wk_is_rejected(field):
    layer = "model.layers.0.self_attn.indexer"
    loader = IndexerWKLoader(_params_for(layer))
    assert not loader.consume(f"{layer}.wk.weight", torch.zeros(32, 64))
    tensor = _fp8_weight(32, 64) if field == "weight" else _fp8_scale(32, 64)
    with pytest.raises(ValueError, match="already loaded"):
        loader.consume(f"{layer}.wk.{field}", tensor)


@pytest.mark.parametrize("shape", [(), (32,), (1, 32, 32), (0, 32), (32, 0)])
@pytest.mark.parametrize("fmt", [FP8_FORMAT, MXFP8_FORMAT])
def test_dequantize_rejects_non_matrix_or_empty_weights(shape, fmt):
    weight = torch.empty(shape, dtype=torch.float8_e4m3fn)
    if fmt == MXFP8_FORMAT:
        scale_shape = (*shape[:-1], 1) if shape else ()
        scale = torch.full(scale_shape, 127, dtype=torch.uint8)
        dequantize = dequantize_mxfp8_wk
    else:
        scale = torch.ones((1,) * len(shape))
        dequantize = dequantize_fp8_wk
    with pytest.raises(ValueError, match="2D|non-empty"):
        dequantize(weight, scale)


def test_load_weights_rejects_plain_duplicate_without_overwriting(monkeypatch):
    _patch_cpu_parallelism(monkeypatch)
    layer = "model.layers.0.self_attn.indexer"
    params = _params_for(layer, rows=32, cols=64)
    weight = _fp8_weight(32, 64)
    scale = _fp8_scale(32, 64)
    with pytest.raises(ValueError, match="already loaded"):
        HYV4ForCausalLM.load_weights(
            _make_integration_model(params),
            [
                (f"{layer}.wk.weight", weight),
                (f"{layer}.wk.weight_scale_inv", scale),
                (f"{layer}.wk.weight", torch.zeros(32, 64, dtype=torch.bfloat16)),
            ],
        )
    param = params[_fused_name(layer)]
    assert len(param.loads) == 1
    torch.testing.assert_close(param.data, _reference_fp8(weight, scale))


@pytest.mark.parametrize("missing_shard", [0, 1])
def test_load_weights_rejects_missing_stacked_component(monkeypatch, missing_shard):
    _patch_cpu_parallelism(monkeypatch)
    name = "model.layers.0.mlp.gate_up_proj.weight"
    param = _StubParam(8, 32)
    model = _make_integration_model({name: param})
    source = "up_proj" if missing_shard == 0 else "gate_proj"
    with pytest.raises(ValueError, match=f"component coverage.*shard={missing_shard}"):
        HYV4ForCausalLM.load_weights(
            model, [(name.replace("gate_up_proj", source), torch.ones(4, 32))]
        )


def test_load_weights_rejects_missing_weights_projection(monkeypatch):
    _patch_cpu_parallelism(monkeypatch)
    layer = "model.layers.0.self_attn.indexer"
    model = _make_integration_model({_fused_name(layer): _StubParam(6, 32)})
    with pytest.raises(ValueError, match="component coverage.*shard=1"):
        HYV4ForCausalLM.load_weights(
            model,
            [
                (f"{layer}.wk.weight", torch.ones(4, 32, dtype=torch.float8_e4m3fn)),
                (
                    f"{layer}.wk.weight_scale",
                    torch.full((4, 1), 127, dtype=torch.uint8),
                ),
            ],
        )


@pytest.mark.parametrize(
    "omitted", [None, (1, "up_proj", "weight"), (1, "up_proj", "weight_scale")]
)
def test_split_expert_component_coverage(monkeypatch, omitted):
    _patch_cpu_parallelism(monkeypatch)
    prefix = "model.layers.0.mlp.experts"
    names = [
        f"{prefix}.routed_experts.{target}_{kind}"
        for target in ("w13", "w2")
        for kind in ("weight", "weight_scale")
    ]
    params = {name: torch.nn.Parameter(torch.zeros(2, 4, 4)) for name in names}
    seen = []

    def load(param, tensor, name, *, shard_id, expert_id, return_success):
        if expert_id != 1:
            return False
        seen.append((name, expert_id, shard_id))
        return True

    for param in params.values():
        param.weight_loader = load
    model = _make_integration_model(params)
    # Only global expert 1 is local; omitted remote expert 0 is legitimate.
    owner = SimpleNamespace(expert_map=torch.tensor([-1, 0]))
    model.named_modules = lambda: iter(
        [("", model), (f"{prefix}.routed_experts", owner)]
    )
    weights = [
        (f"{prefix}.{expert}.{proj}.{kind}", torch.ones(4, 4))
        for expert in (1,)
        for proj in ("gate_proj", "up_proj", "down_proj")
        for kind in ("weight", "weight_scale")
        if (expert, proj, kind) != omitted
    ]
    if omitted is None:
        assert HYV4ForCausalLM.load_weights(model, weights) == set(params)
        assert len(seen) == 6
    else:
        with pytest.raises(ValueError, match="component coverage.*expert=1 shard=w3"):
            HYV4ForCausalLM.load_weights(model, weights)


@pytest.mark.parametrize(
    "projection,sizes,sources",
    [
        ("mlp.gate_up_proj", [4, 4], ("gate_proj", "up_proj")),
        ("self_attn.indexer.wk_weights_proj", [4, 2], ("wk", "weights_proj")),
    ],
)
def test_complete_fused_safetensors_equals_split_loading(
    monkeypatch, tmp_path, projection, sizes, sources
):
    from safetensors.torch import load_file, save_file

    import vllm.model_executor.parameter as parameters
    from vllm.model_executor.layers.linear import MergedColumnParallelLinear

    _patch_cpu_parallelism(monkeypatch)
    # Actual vLLM parameter/loader, with the indexer's replicated TP contract.
    monkeypatch.setattr(parameters, "get_tensor_model_parallel_rank", lambda: 0)
    monkeypatch.setattr(parameters, "get_tensor_model_parallel_world_size", lambda: 1)
    name = f"model.layers.0.{projection}.weight"
    fused = torch.arange(sum(sizes) * 8, dtype=torch.float32).reshape(sum(sizes), 8)
    loaded = []
    for full in (False, True):
        with torch.device("cpu"):
            layer = MergedColumnParallelLinear(
                8, sizes, bias=False, disable_tp=True, params_dtype=torch.float32
            )
        param = layer.weight
        model = _make_integration_model({name: param})
        if full:
            # The enclosing model's TP can differ from this replicated layer.
            monkeypatch.setattr(
                hy_v4, "get_tensor_model_parallel_world_size", lambda: 2
            )
            tensors = {name: fused}
        else:
            tensors = {
                name.replace(projection.split(".")[-1], source): part.contiguous()
                for source, part in zip(sources, fused.split(sizes))
            }
        path = tmp_path / f"{full}.safetensors"
        save_file(tensors, path)
        assert HYV4ForCausalLM.load_weights(model, load_file(path).items()) == {name}
        loaded.append(param.detach().clone())
    torch.testing.assert_close(loaded[0], fused, rtol=0, atol=0)
    torch.testing.assert_close(loaded[1], fused, rtol=0, atol=0)


@pytest.mark.parametrize("rows", [4, 9])
def test_complete_fused_bad_shape_rejected_before_write(monkeypatch, rows):
    _patch_cpu_parallelism(monkeypatch)
    name = "model.layers.0.mlp.gate_up_proj.weight"
    param = torch.nn.Parameter(torch.full((8, 4), -12345.0))
    param.weight_loader = lambda *args: pytest.fail(
        "shape must be checked before writing"
    )
    with pytest.raises(ValueError, match="complete fused tensor shape mismatch"):
        HYV4ForCausalLM.load_weights(
            _make_integration_model({name: param}), [(name, torch.ones(rows, 4))]
        )
    assert torch.all(param == -12345)


def test_complete_fused_global_tensor_shards_to_local_tp_rank(monkeypatch):
    import vllm.model_executor.layers.linear as linear
    import vllm.model_executor.parameter as parameters

    for module in (hy_v4, linear, parameters):
        monkeypatch.setattr(module, "get_tensor_model_parallel_rank", lambda: 1)
        monkeypatch.setattr(module, "get_tensor_model_parallel_world_size", lambda: 2)
    with torch.device("cpu"):
        layer = linear.MergedColumnParallelLinear(
            4, [8, 8], bias=False, params_dtype=torch.float32
        )
    name = "model.layers.0.mlp.gate_up_proj.weight"
    fused = torch.arange(64, dtype=torch.float32).reshape(16, 4)
    assert HYV4ForCausalLM.load_weights(
        _make_integration_model({name: layer.weight}), [(name, fused)]
    ) == {name}
    torch.testing.assert_close(
        layer.weight, torch.cat([fused[4:8], fused[12:16]]), rtol=0, atol=0
    )
