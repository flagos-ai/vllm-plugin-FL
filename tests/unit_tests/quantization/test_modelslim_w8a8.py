import sys
from types import ModuleType

import pytest
import torch

import vllm.model_executor.parameter as vllm_parameter

from vllm_fl.quantization.modelslim_w8a8 import (
    ModelSlimW8A8Config,
    _ModelSlimW8A8StaticLinearScheme,
)


@pytest.fixture(autouse=True)
def _single_rank_tensor_parallel(monkeypatch):
    monkeypatch.setattr(
        vllm_parameter,
        "get_tensor_model_parallel_rank",
        lambda: 0,
    )
    monkeypatch.setattr(
        vllm_parameter,
        "get_tensor_model_parallel_world_size",
        lambda: 1,
    )


def _deepseek_v4_config():
    return ModelSlimW8A8Config(
        {
            "layers.0.attn.wq_a.weight": "W8A8_DYNAMIC",
            "layers.0.attn.wkv.weight": "W8A8_DYNAMIC",
            "layers.0.attn.wq_b.weight": "W8A8_DYNAMIC",
            "layers.0.attn.wo_a.weight": "FLOAT",
            "layers.0.ffn.experts.0.w1.weight": "W8A8_DYNAMIC",
            "layers.0.ffn.experts.0.w2.weight": "W8A8_DYNAMIC",
            "layers.0.ffn.experts.0.w3.weight": "W8A8_DYNAMIC",
        }
    )


def _glm_config(root: str) -> ModelSlimW8A8Config:
    return ModelSlimW8A8Config(
        {
            f"{root}layers.0.self_attn.q_a_proj.weight": "W8A8",
            f"{root}layers.0.self_attn.kv_a_proj_with_mqa.weight": "W8A8",
            f"{root}layers.0.self_attn.q_b_proj.weight": "W8A8",
            f"{root}layers.0.self_attn.o_proj.weight": "W8A8",
            f"{root}layers.0.self_attn.indexer.wq_b.weight": "W8A8",
            f"{root}layers.0.mlp.experts.0.gate_proj.weight": "W8A8_DYNAMIC",
            f"{root}layers.0.mlp.experts.0.up_proj.weight": "W8A8_DYNAMIC",
            f"{root}layers.0.mlp.experts.0.down_proj.weight": "W8A8_DYNAMIC",
            f"{root}layers.0.mlp.shared_experts.gate_proj.weight": "W8A8_DYNAMIC",
            f"{root}layers.0.mlp.shared_experts.up_proj.weight": "W8A8_DYNAMIC",
            f"{root}layers.0.mlp.shared_experts.down_proj.weight": "W8A8_DYNAMIC",
        }
    )


def test_maps_deepseek_v4_packed_attention_names():
    config = _deepseek_v4_config()
    assert config._is_dynamic_w8a8("model.layers.0.attn.fused_wqa_wkv")
    assert config._is_dynamic_w8a8("model.layers.0.attn.wq_b")
    assert not config._is_dynamic_w8a8("model.layers.0.attn.wo_a")


def test_maps_deepseek_v4_routed_experts():
    assert _deepseek_v4_config()._is_dynamic_w8a8("model.layers.0.ffn.experts")


def test_rejects_mixed_packed_quantization():
    config = _deepseek_v4_config()
    config.description["layers.0.attn.wkv.weight"] = "FLOAT"
    with pytest.raises(ValueError, match="mixes"):
        config._is_dynamic_w8a8("model.layers.0.attn.fused_wqa_wkv")


@pytest.mark.parametrize("description_root", ["", "model."])
@pytest.mark.parametrize("runtime_root", ["", "model."])
def test_maps_glm_static_attention_with_optional_model_root(
    description_root: str,
    runtime_root: str,
):
    config = _glm_config(description_root)
    prefix = f"{runtime_root}layers.0.self_attn"

    assert config._quant_kind(f"{prefix}.fused_qkv_a_proj") == "W8A8"
    assert config._quant_kind(f"{prefix}.q_a_proj") == "W8A8"
    assert config._quant_kind(f"{prefix}.kv_a_proj_with_mqa") == "W8A8"
    assert config._quant_kind(f"{prefix}.q_b_proj") == "W8A8"
    assert config._quant_kind(f"{prefix}.o_proj") == "W8A8"
    assert config._quant_kind(f"{prefix}.indexer.wq_b") == "W8A8"


@pytest.mark.parametrize("description_root", ["", "model."])
@pytest.mark.parametrize("runtime_root", ["", "model."])
def test_maps_glm_dynamic_routed_and_shared_experts(
    description_root: str,
    runtime_root: str,
):
    config = _glm_config(description_root)
    prefix = f"{runtime_root}layers.0.mlp"

    assert config._quant_kind(f"{prefix}.experts") == "W8A8_DYNAMIC"
    assert config._quant_kind(f"{prefix}.shared_experts.gate_up_proj") == "W8A8_DYNAMIC"
    assert config._quant_kind(f"{prefix}.shared_experts.down_proj") == "W8A8_DYNAMIC"


def test_rejects_mixed_glm_fused_qkv_a_quantization():
    config = _glm_config("")
    config.description["layers.0.self_attn.kv_a_proj_with_mqa.weight"] = "FLOAT"

    with pytest.raises(ValueError, match="mixes"):
        config._quant_kind("model.layers.0.self_attn.fused_qkv_a_proj")


def test_rejects_incomplete_glm_fused_qkv_a_quantization():
    config = _glm_config("")
    del config.description["layers.0.self_attn.kv_a_proj_with_mqa.weight"]

    with pytest.raises(ValueError, match="missing quantization entries"):
        config._quant_kind("model.layers.0.self_attn.fused_qkv_a_proj")


def _create_static_layer() -> tuple[_ModelSlimW8A8StaticLinearScheme, torch.nn.Module]:
    scheme = _ModelSlimW8A8StaticLinearScheme()
    layer = torch.nn.Module()
    scheme.create_weights(
        layer,
        output_partition_sizes=[3, 5],
        input_size_per_partition=4,
        params_dtype=torch.bfloat16,
        weight_loader=lambda *args, **kwargs: None,
    )
    return scheme, layer


def test_static_scheme_rejects_unverified_fp16_activation_path():
    scheme = _ModelSlimW8A8StaticLinearScheme()
    layer = torch.nn.Module()

    with pytest.raises(ValueError, match="requires bfloat16"):
        scheme.create_weights(
            layer,
            output_partition_sizes=[8],
            input_size_per_partition=4,
            params_dtype=torch.float16,
            weight_loader=lambda *args, **kwargs: None,
        )


def test_static_scheme_registers_exact_modelslim_parameters():
    _, layer = _create_static_layer()
    parameters = dict(layer.named_parameters())

    assert set(parameters) == {
        "weight",
        "input_scale",
        "input_offset",
        "deq_scale",
        "quant_bias",
    }
    assert parameters["weight"].shape == (8, 4)
    assert parameters["weight"].dtype == torch.int8
    assert parameters["input_scale"].shape == (2,)
    assert parameters["input_scale"].dtype == torch.float32
    assert parameters["input_offset"].shape == (2,)
    assert parameters["input_offset"].dtype == torch.int8
    assert parameters["deq_scale"].shape == (8,)
    assert parameters["deq_scale"].dtype == torch.float32
    assert parameters["quant_bias"].shape == (8,)
    assert parameters["quant_bias"].dtype == torch.int32
    assert all(not parameter.requires_grad for parameter in parameters.values())
    assert not hasattr(layer, "weight_scale")
    assert not hasattr(layer, "weight_offset")


def test_static_scheme_transposes_weight_and_expands_partition_quantizers():
    scheme, layer = _create_static_layer()
    checkpoint_weight = torch.arange(32, dtype=torch.int8).reshape(8, 4)
    layer.weight.data.copy_(checkpoint_weight)
    layer.input_scale.data.copy_(torch.tensor([0.25, 0.5]))
    # The checkpoint stores float32 offsets; loading casts them to the native
    # torch_npu quantizer's int8 parameter contract.
    layer.input_offset.data.copy_(torch.tensor([1.0, -2.0], dtype=torch.float32))

    scheme.process_weights_after_loading(layer)

    assert layer.weight.shape == (4, 8)
    assert layer.weight.is_contiguous()
    assert torch.equal(layer.weight, checkpoint_weight.t().contiguous())
    assert layer.aclnn_input_scale_reciprocal.dtype == torch.bfloat16
    assert layer.aclnn_input_offset.dtype == torch.bfloat16
    assert torch.equal(
        layer.aclnn_input_scale_reciprocal,
        torch.tensor([[4.0] * 4, [2.0] * 4], dtype=torch.bfloat16),
    )
    assert torch.equal(
        layer.aclnn_input_offset,
        torch.tensor([[1.0] * 4, [-2.0] * 4], dtype=torch.bfloat16),
    )


def test_static_scheme_quantizes_each_fused_partition_and_calls_torch_npu(
    monkeypatch,
):
    scheme, layer = _create_static_layer()
    layer.weight.data.copy_(torch.arange(32, dtype=torch.int8).reshape(8, 4))
    layer.input_scale.data.copy_(torch.tensor([0.25, 0.5]))
    layer.input_offset.data.copy_(torch.tensor([1.0, -2.0], dtype=torch.float32))
    layer.deq_scale.data.copy_(torch.arange(1, 9, dtype=torch.float32))
    layer.quant_bias.data.copy_(torch.arange(8, dtype=torch.int32))
    scheme.process_weights_after_loading(layer)

    quantize_calls = []
    matmul_calls = []
    torch_npu = ModuleType("torch_npu")

    def fake_quantize(x, scale, offset, dtype, axis, sqrt_mode):
        quantize_calls.append(
            (x, scale.clone(), offset.clone(), dtype, axis, sqrt_mode)
        )
        return torch.full_like(x, len(quantize_calls), dtype=torch.int8)

    def fake_quant_matmul(
        x,
        weight,
        deq_scale,
        *,
        bias,
        output_dtype,
    ):
        matmul_calls.append(
            (x, weight.clone(), deq_scale.clone(), bias.clone(), output_dtype)
        )
        return torch.full(
            (x.shape[0], weight.shape[1]),
            len(matmul_calls),
            dtype=output_dtype,
        )

    torch_npu.npu_quantize = fake_quantize
    torch_npu.npu_quant_matmul = fake_quant_matmul
    monkeypatch.setitem(sys.modules, "torch_npu", torch_npu)

    x = torch.ones((2, 4), dtype=torch.bfloat16)
    output = scheme.apply_weights(layer, x, bias=None)

    assert len(quantize_calls) == 2
    assert [call[1].tolist() for call in quantize_calls] == [[4.0] * 4, [2.0] * 4]
    assert [call[2].tolist() for call in quantize_calls] == [
        [1.0] * 4,
        [-2.0] * 4,
    ]
    assert all(call[3:] == (torch.qint8, -1, False) for call in quantize_calls)

    assert len(matmul_calls) == 2
    assert torch.equal(matmul_calls[0][0], torch.ones_like(x, dtype=torch.int8))
    assert torch.equal(matmul_calls[1][0], torch.full_like(x, 2, dtype=torch.int8))
    assert matmul_calls[0][1].shape == (4, 3)
    assert matmul_calls[1][1].shape == (4, 5)
    expected_weight = torch.arange(32, dtype=torch.int8).reshape(8, 4).t()
    assert torch.equal(matmul_calls[0][1], expected_weight[:, :3])
    assert torch.equal(matmul_calls[1][1], expected_weight[:, 3:])
    assert matmul_calls[0][2].tolist() == [1.0, 2.0, 3.0]
    assert matmul_calls[1][2].tolist() == [4.0, 5.0, 6.0, 7.0, 8.0]
    assert matmul_calls[0][3].tolist() == [0, 1, 2]
    assert matmul_calls[1][3].tolist() == [3, 4, 5, 6, 7]
    assert all(call[4] == torch.bfloat16 for call in matmul_calls)
    assert torch.equal(
        output,
        torch.tensor(
            [[1, 1, 1, 2, 2, 2, 2, 2], [1, 1, 1, 2, 2, 2, 2, 2]],
            dtype=torch.bfloat16,
        ),
    )
