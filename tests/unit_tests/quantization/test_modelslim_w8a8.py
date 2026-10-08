import sys
from types import ModuleType

import pytest
import torch

import vllm.model_executor.parameter as vllm_parameter

import vllm_fl.quantization.modelslim_w8a8 as modelslim_w8a8
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


def _loaded_static_layer(
    checkpoint_weight: torch.Tensor,
    partition_widths: list[int],
    input_scale: torch.Tensor,
    input_offset: torch.Tensor,
    deq_scale: torch.Tensor,
    quant_bias: torch.Tensor,
    layer: torch.nn.Module | None = None,
) -> tuple[_ModelSlimW8A8StaticLinearScheme, torch.nn.Module]:
    scheme = _ModelSlimW8A8StaticLinearScheme()
    layer = torch.nn.Module() if layer is None else layer
    scheme.create_weights(
        layer,
        output_partition_sizes=partition_widths,
        input_size_per_partition=checkpoint_weight.shape[1],
        params_dtype=torch.bfloat16,
        weight_loader=lambda *args, **kwargs: None,
    )
    layer.weight.data.copy_(checkpoint_weight)
    layer.input_scale.data.copy_(input_scale)
    layer.input_offset.data.copy_(input_offset)
    layer.deq_scale.data.copy_(deq_scale)
    layer.quant_bias.data.copy_(quant_bias)
    scheme.process_weights_after_loading(layer)
    return scheme, layer


def _static_modelslim_reference(
    x: torch.Tensor,
    checkpoint_weight: torch.Tensor,
    partition_widths: list[int],
    input_scale: torch.Tensor,
    input_offset: torch.Tensor,
    deq_scale: torch.Tensor,
    quant_bias: torch.Tensor,
    bias: torch.Tensor | None,
    *,
    tp_rank: int = 0,
) -> torch.Tensor:
    """Reference from checkpoint values, without prepared tensors or NPU ops.

    AscendQuantV3 computes round(x / scale + offset), and int32 matmul bias
    is added before the channel dequantization scale. Row-parallel shards
    apply both checkpoint and floating-point bias on TP rank zero only.
    """
    output_offset = 0
    outputs = []
    for partition, width in enumerate(partition_widths):
        output_slice = slice(output_offset, output_offset + width)
        quantized = (
            (x.float() / input_scale[partition] + input_offset[partition])
            .round()
            .clamp(-128, 127)
            .to(torch.int32)
        )
        accumulator = quantized @ checkpoint_weight[output_slice].to(torch.int32).T
        if tp_rank == 0:
            accumulator = accumulator + quant_bias[output_slice]
        output = (accumulator.float() * deq_scale[output_slice].float()).to(
            torch.bfloat16
        )
        if bias is not None and tp_rank == 0:
            output = output + bias[output_slice]
        outputs.append(output)
        output_offset += width
    return torch.cat(outputs, dim=-1)


def _install_numerical_torch_npu_stubs(monkeypatch):
    """Execute the documented quantize/matmul equations on CPU tensors."""
    torch_npu = ModuleType("torch_npu")

    def quantize(x, scale, offset, dtype, axis, sqrt_mode):
        assert (dtype, axis, sqrt_mode) == (torch.qint8, -1, False)
        return (
            (x.float() * scale.float() + offset.float())
            .round()
            .clamp(-128, 127)
            .to(torch.int8)
        )

    def quant_matmul(x, weight, deq_scale, *, bias, output_dtype):
        accumulator = x.to(torch.int32) @ weight.to(torch.int32)
        if bias is not None:
            accumulator = accumulator + bias
        return (accumulator.float() * deq_scale.float()).to(output_dtype)

    torch_npu.npu_quantize = quantize
    torch_npu.npu_quant_matmul = quant_matmul
    monkeypatch.setitem(sys.modules, "torch_npu", torch_npu)


def test_static_scheme_matches_independent_fused_partition_reference(monkeypatch):
    _install_numerical_torch_npu_stubs(monkeypatch)
    x = torch.tensor(
        [[1.0625, -0.8125, 0.4375, 1.6875], [-1.1875, 0.3125, 1.9375, -0.5625]],
        dtype=torch.bfloat16,
    )
    checkpoint_weight = torch.tensor(
        [
            [2, -3, 1, 4],
            [-1, 2, -4, 3],
            [3, 1, -2, -1],
            [1, -2, 3, 2],
            [-4, 1, 2, -3],
            [2, 3, -1, -2],
            [-3, -2, 1, 4],
            [4, -1, -3, 2],
        ],
        dtype=torch.int8,
    )
    widths = [3, 5]
    input_scale = torch.tensor([0.5, 0.25])
    input_offset = torch.tensor([1, -2], dtype=torch.int8)
    deq_scale = torch.tensor([0.0625, 0.125, 0.25, 0.5, 1, 0.03125, 0.25, 0.125])
    quant_bias = torch.tensor([3, -2, 1, 4, -3, 2, -1, 5], dtype=torch.int32)
    bias = torch.tensor(
        [0.125, -0.25, 0.5, 1, -1, 0.75, -0.125, 0.25],
        dtype=torch.bfloat16,
    )

    scheme, layer = _loaded_static_layer(
        checkpoint_weight, widths, input_scale, input_offset, deq_scale, quant_bias
    )
    actual = scheme.apply_weights(layer, x, bias)
    expected = _static_modelslim_reference(
        x,
        checkpoint_weight,
        widths,
        input_scale,
        input_offset,
        deq_scale,
        quant_bias,
        bias,
    )

    assert torch.equal(actual, expected)
    assert torch.count_nonzero(actual) == actual.numel()
    assert not torch.equal(actual[:, :3], actual[:, 3:6])


def test_static_scheme_row_parallel_bias_is_applied_once_before_tp_reduction(
    monkeypatch,
):
    _install_numerical_torch_npu_stubs(monkeypatch)

    class FakeRowParallelLinear(torch.nn.Module):
        pass

    monkeypatch.setattr(modelslim_w8a8, "RowParallelLinear", FakeRowParallelLinear)
    x = torch.tensor(
        [[1.0625, -0.8125, 0.4375, 1.6875, -1.1875, 0.3125, 1.9375, -0.5625]],
        dtype=torch.bfloat16,
    )
    checkpoint_weight = torch.tensor(
        [
            [2, -3, 1, 4, 3, -1, 2, 1],
            [-1, 2, -4, 3, 1, -2, 4, -3],
            [3, 1, -2, -1, -1, 4, -3, 2],
            [1, -2, 3, 2, 2, -3, 1, -4],
        ],
        dtype=torch.int8,
    )
    input_scales = [torch.tensor([0.5]), torch.tensor([0.25])]
    input_offsets = [
        torch.tensor([1], dtype=torch.int8),
        torch.tensor([-2], dtype=torch.int8),
    ]
    deq_scale = torch.tensor([0.0625, 0.125, 0.25, 0.5])
    quant_bias = torch.tensor([3, -2, 1, 4], dtype=torch.int32)
    bias = torch.tensor([0.125, -0.25, 0.5, 1], dtype=torch.bfloat16)

    local_outputs = []
    reference_outputs = []
    for rank in (0, 1):
        shard = slice(rank * 4, (rank + 1) * 4)
        layer = FakeRowParallelLinear()
        layer.tp_rank = rank
        scheme, layer = _loaded_static_layer(
            checkpoint_weight[:, shard],
            [4],
            input_scales[rank],
            input_offsets[rank],
            deq_scale,
            quant_bias,
            layer,
        )
        local_outputs.append(scheme.apply_weights(layer, x[:, shard], bias))
        reference_outputs.append(
            _static_modelslim_reference(
                x[:, shard],
                checkpoint_weight[:, shard],
                [4],
                input_scales[rank],
                input_offsets[rank],
                deq_scale,
                quant_bias,
                bias,
                tp_rank=rank,
            )
        )

    reduced = local_outputs[0] + local_outputs[1]
    expected = reference_outputs[0] + reference_outputs[1]
    assert torch.equal(local_outputs[0], reference_outputs[0])
    assert torch.equal(local_outputs[1], reference_outputs[1])
    assert torch.equal(reduced, expected)
    assert not torch.equal(reduced, expected + bias)


@pytest.mark.gpu
def test_real_ascend_static_modelslim_fused_partitions_match_cpu_reference():
    pytest.importorskip("torch_npu")
    if not torch.npu.is_available():
        pytest.skip("requires an Ascend NPU")

    # Use aligned K/N dimensions required by the actual quantized matmul.
    x = (((torch.arange(256).reshape(2, 128) % 17) - 8).float() / 16 + 0.015625).to(
        torch.bfloat16
    )
    checkpoint_weight = (((torch.arange(128 * 128).reshape(128, 128) * 3) % 11) - 5).to(
        torch.int8
    )
    widths = [64, 64]
    input_scale = torch.tensor([0.25, 0.5], dtype=torch.float32)
    input_offset = torch.tensor([1, -2], dtype=torch.int8)
    deq_scale = (0.03125 * (2.0 ** (torch.arange(128) % 4))).to(torch.float32)
    quant_bias = ((torch.arange(128) % 7) - 3).to(torch.int32)
    bias = (((torch.arange(128) % 5) - 2).float() / 16).to(torch.bfloat16)

    scheme, layer = _loaded_static_layer(
        checkpoint_weight, widths, input_scale, input_offset, deq_scale, quant_bias
    )
    expected = _static_modelslim_reference(
        x,
        checkpoint_weight,
        widths,
        input_scale,
        input_offset,
        deq_scale,
        quant_bias,
        bias,
    )
    layer = layer.to("npu:0")
    actual = scheme.apply_weights(layer, x.to("npu:0"), bias.to("npu:0"))

    torch.testing.assert_close(
        actual.cpu().float(), expected.float(), rtol=0.03, atol=0.25
    )


@pytest.mark.gpu
def test_real_ascend_static_modelslim_tp_shard_sum_matches_cpu_reference(
    monkeypatch,
):
    pytest.importorskip("torch_npu")
    if not torch.npu.is_available():
        pytest.skip("requires an Ascend NPU")

    class FakeRowParallelLinear(torch.nn.Module):
        pass

    monkeypatch.setattr(modelslim_w8a8, "RowParallelLinear", FakeRowParallelLinear)
    x = (((torch.arange(512).reshape(2, 256) % 19) - 9).float() / 16 + 0.015625).to(
        torch.bfloat16
    )
    checkpoint_weight = (((torch.arange(128 * 256).reshape(128, 256) * 5) % 13) - 6).to(
        torch.int8
    )
    input_scales = [torch.tensor([0.25]), torch.tensor([0.5])]
    input_offsets = [
        torch.tensor([1], dtype=torch.int8),
        torch.tensor([-2], dtype=torch.int8),
    ]
    deq_scale = (0.03125 * (2.0 ** (torch.arange(128) % 4))).float()
    quant_bias = ((torch.arange(128) % 7) - 3).to(torch.int32)
    bias = (((torch.arange(128) % 5) - 2).float() / 16).to(torch.bfloat16)

    local_outputs = []
    reference_outputs = []
    for rank in (0, 1):
        shard = slice(rank * 128, (rank + 1) * 128)
        layer = FakeRowParallelLinear()
        layer.tp_rank = rank
        scheme, layer = _loaded_static_layer(
            checkpoint_weight[:, shard],
            [128],
            input_scales[rank],
            input_offsets[rank],
            deq_scale,
            quant_bias,
            layer,
        )
        reference_outputs.append(
            _static_modelslim_reference(
                x[:, shard],
                checkpoint_weight[:, shard],
                [128],
                input_scales[rank],
                input_offsets[rank],
                deq_scale,
                quant_bias,
                bias,
                tp_rank=rank,
            )
        )
        layer = layer.to("npu:0")
        local_outputs.append(
            scheme.apply_weights(layer, x[:, shard].to("npu:0"), bias.to("npu:0")).cpu()
        )

    # A real TP all-reduce sums these local BF16 outputs; this checks its
    # numerical inputs and verifies that neither bias was added on rank one.
    actual_reduced = local_outputs[0] + local_outputs[1]
    expected_reduced = reference_outputs[0] + reference_outputs[1]
    for actual, expected in zip(local_outputs, reference_outputs, strict=True):
        torch.testing.assert_close(
            actual.float(), expected.float(), rtol=0.03, atol=0.25
        )
    torch.testing.assert_close(
        actual_reduced.float(), expected_reduced.float(), rtol=0.03, atol=0.25
    )
