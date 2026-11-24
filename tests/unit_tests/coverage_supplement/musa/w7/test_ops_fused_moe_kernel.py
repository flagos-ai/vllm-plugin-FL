# Coverage tests: vllm_fl/ops/fused_moe/fused_moe.py (CPU only).

from types import SimpleNamespace

import pytest
import torch

from vllm_fl.ops.fused_moe import fused_moe as fm


def _patch_kernels(monkeypatch):
    monkeypatch.setattr(fm, "_get_config_dtype_str", lambda **kwargs: "fp16")
    monkeypatch.setattr(fm, "_get_config_quant_dtype", lambda **kwargs: None)
    monkeypatch.setattr(
        fm, "try_get_optimal_moe_config", lambda *args, **kwargs: {"BLOCK_SIZE_M": 16}
    )
    monkeypatch.setattr(
        fm, "_moe_align_block_size", lambda *args, **kwargs: (None, None, None)
    )
    monkeypatch.setattr(
        fm, "_invoke_fused_moe_triton_kernel", lambda *args, **kwargs: None
    )
    monkeypatch.setattr(fm, "moe_kernel_quantize_input", lambda A, **kwargs: (A, None))
    monkeypatch.setattr(fm, "_moe_sum", lambda tensor, output: None)
    monkeypatch.setattr(
        fm, "_silu_and_mul", lambda _out, x: x.new_empty((x.shape[0], x.shape[1] // 2))
    )
    monkeypatch.setattr(
        fm, "_gelu_and_mul", lambda _out, x: x.new_empty((x.shape[0], x.shape[1] // 2))
    )


def _args(**overrides):
    hidden = torch.zeros((2, 4), dtype=torch.float32)
    values = {
        "hidden_states": hidden,
        "w1": torch.zeros((2, 8, 4), dtype=torch.float32),
        "w2": torch.zeros((2, 4, 4), dtype=torch.float32),
        "topk_weights": torch.ones((2, 1), dtype=torch.float32),
        "topk_ids": torch.zeros((2, 1), dtype=torch.int64),
    }
    values.update(overrides)
    return values


@pytest.mark.parametrize("activation", ["silu", "gelu", "silu_no_mul", "gelu_no_mul"])
def test_fused_experts_impl_activation_variants(monkeypatch, activation):
    _patch_kernels(monkeypatch)
    output = fm.fused_experts_impl(**_args(), activation=activation)
    assert output.shape == (2, 4)


def test_fused_experts_impl_accepts_activation_enum_like(monkeypatch):
    _patch_kernels(monkeypatch)
    output = fm.fused_experts_impl(
        **_args(), activation=SimpleNamespace(value="silu")
    )
    assert output.shape == (2, 4)


def test_fused_experts_impl_rejects_unknown_activation(monkeypatch):
    _patch_kernels(monkeypatch)
    with pytest.raises(ValueError, match="Unsupported FusedMoe activation"):
        fm.fused_experts_impl(**_args(), activation="mystery")


def test_fused_experts_impl_inplace_returns_input(monkeypatch):
    _patch_kernels(monkeypatch)
    args = _args()
    output = fm.fused_experts_impl(**args, inplace=True)
    assert output is args["hidden_states"]


def test_fused_experts_impl_int4_weight_layout(monkeypatch):
    _patch_kernels(monkeypatch)
    args = _args(
        hidden_states=torch.zeros((2, 8), dtype=torch.float32),
        w1=torch.zeros((2, 4, 4), dtype=torch.float32),
        w2=torch.zeros((2, 8, 2), dtype=torch.float32),
    )
    output = fm.fused_experts_impl(**args, use_int4_w4a16=True)
    assert output.shape == (2, 8)


def test_fused_experts_impl_chunking_second_chunk(monkeypatch):
    _patch_kernels(monkeypatch)
    tokens = 65537
    args = _args(
        hidden_states=torch.zeros((tokens, 4), dtype=torch.float32),
        topk_weights=torch.ones((tokens, 1), dtype=torch.float32),
        topk_ids=torch.zeros((tokens, 1), dtype=torch.int64),
    )
    output = fm.fused_experts_impl(**args)
    assert output.shape == (tokens, 4)


def test_fused_experts_impl_validation_errors():
    with pytest.raises(AssertionError, match="topk shape mismatch"):
        fm.fused_experts_impl(
            **_args(topk_ids=torch.zeros((2, 2), dtype=torch.int64))
        )
    with pytest.raises(AssertionError, match="contiguous"):
        fm.fused_experts_impl(
            **_args(hidden_states=torch.zeros((4, 2), dtype=torch.float32).t())
        )
    with pytest.raises(AssertionError, match="Stride of last dimension"):
        fm.fused_experts_impl(
            **_args(w1=torch.zeros((2, 4, 8), dtype=torch.float32).transpose(1, 2))
        )
    with pytest.raises(AssertionError):
        fm.fused_experts_impl(
            **_args(hidden_states=torch.zeros((2, 4), dtype=torch.int32))
        )
    with pytest.raises(AssertionError, match="Hidden size mismatch"):
        fm.fused_experts_impl(
            **_args(w1=torch.zeros((2, 8, 5), dtype=torch.float32))
        )
    with pytest.raises(AssertionError, match="Hidden size mismatch"):
        fm.fused_experts_impl(
            **_args(
                hidden_states=torch.zeros((2, 8), dtype=torch.float32),
                w1=torch.zeros((2, 4, 5), dtype=torch.float32),
            ),
            use_int4_w4a16=True,
        )


def test_fused_experts_wrapper_uses_default_quant_config(monkeypatch):
    captured = {}

    def fake_impl(**kwargs):
        captured.update(kwargs)
        return "result"

    monkeypatch.setattr(fm, "fused_experts_impl", fake_impl)
    assert (
        fm.fused_experts(
            hidden_states=torch.zeros(1, 4),
            w1=torch.zeros(2, 8, 4),
            w2=torch.zeros(2, 4, 4),
            topk_weights=torch.ones(1, 1),
            topk_ids=torch.zeros(1, 1, dtype=torch.int64),
        )
        == "result"
    )
    default = fm.FUSED_MOE_UNQUANTIZED_CONFIG
    assert captured["use_fp8_w8a8"] == default.use_fp8_w8a8
    assert captured["per_channel_quant"] == default.per_act_token_quant
    assert captured["block_shape"] == default.block_shape


def test_fused_experts_wrapper_forwards_custom_quant_config(monkeypatch):
    captured = {}
    monkeypatch.setattr(
        fm, "fused_experts_impl", lambda **kwargs: captured.update(kwargs) or "ok"
    )
    quant_config = SimpleNamespace(
        use_fp8_w8a8=True,
        use_int8_w8a8=True,
        use_int8_w8a16=False,
        use_int4_w4a16=False,
        per_act_token_quant=True,
        w1_scale="w1s",
        w2_scale="w2s",
        w1_zp="w1z",
        w2_zp="w2z",
        a1_scale="a1s",
        a2_scale="a2s",
        block_shape=[128, 128],
        w1_bias="w1b",
        w2_bias="w2b",
    )
    result = fm.fused_experts(
        hidden_states=torch.zeros(1, 4),
        w1=torch.zeros(2, 8, 4),
        w2=torch.zeros(2, 4, 4),
        topk_weights=torch.ones(1, 1),
        topk_ids=torch.zeros(1, 1, dtype=torch.int64),
        activation="gelu",
        quant_config=quant_config,
        apply_router_weight_on_input=True,
        global_num_experts=4,
        expert_map="map",
    )
    assert result == "ok"
    assert captured["use_fp8_w8a8"] is True
    assert captured["per_channel_quant"] is True
    assert captured["block_shape"] == [128, 128]
    assert captured["w1_bias"] == "w1b"
    assert captured["expert_map"] == "map"
    assert captured["activation"] == "gelu"
