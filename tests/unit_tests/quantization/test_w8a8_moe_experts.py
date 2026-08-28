# Copyright 2026 FlagOS Contributors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from types import SimpleNamespace

import pytest
import torch

from vllm.model_executor.layers.fused_moe.activation import MoEActivation

from vllm_fl.quantization.w8a8 import moe_experts


def _apply_arguments():
    hidden_states = torch.ones((2, 4), dtype=torch.bfloat16)
    return {
        "output": torch.empty_like(hidden_states),
        "hidden_states": hidden_states,
        "w1": torch.ones((2, 8, 4), dtype=torch.int8),
        "w2": torch.ones((2, 4, 4), dtype=torch.int8),
        "topk_weights": torch.ones((2, 1), dtype=torch.float32),
        "topk_ids": torch.zeros((2, 1), dtype=torch.int64),
        "activation": MoEActivation.SILU,
        "global_num_experts": 2,
        "expert_map": None,
        "a1q_scale": None,
        "a2_scale": None,
        "workspace13": torch.empty(0),
        "workspace2": torch.empty(0),
        "expert_tokens_meta": None,
        "apply_router_weight_on_input": False,
    }


def _quant_config():
    return SimpleNamespace(
        use_int8_w8a8=True,
        per_act_token_quant=True,
        block_shape=None,
        w1_scale=torch.ones((2, 8, 1), dtype=torch.float32),
        w2_scale=torch.ones((2, 4, 1), dtype=torch.float32),
        w1_bias=None,
        w2_bias=None,
    )


@pytest.mark.parametrize(
    "experts_cls",
    [
        moe_experts.FlagGemsW8A8Experts,
        moe_experts.VllmFunctionalW8A8Experts,
    ],
)
def test_functional_experts_defer_activation_quantization(experts_cls):
    instance = SimpleNamespace()
    assert experts_cls.expects_unquantized_inputs.fget(instance) is True


def test_vllm_functional_experts_changes_the_triton_input_contract():
    instance = SimpleNamespace(_lora_context=None)

    assert moe_experts.TritonExperts.expects_unquantized_inputs.fget(instance) is False
    assert (
        moe_experts.VllmFunctionalW8A8Experts.expects_unquantized_inputs.fget(instance)
        is True
    )


def test_vllm_functional_experts_call_only_native_vllm(monkeypatch):
    calls = []

    def fake_vllm_fused_experts(**kwargs):
        calls.append(kwargs)
        return torch.full_like(kwargs["hidden_states"], 2)

    monkeypatch.setattr(
        moe_experts,
        "_vllm_fused_experts",
        fake_vllm_fused_experts,
    )
    monkeypatch.setattr(
        moe_experts,
        "_flaggems_fused_experts_impl",
        lambda **kwargs: pytest.fail("NVIDIA must not dispatch through FlagGems"),
    )
    quant_config = _quant_config()
    arguments = _apply_arguments()

    moe_experts.VllmFunctionalW8A8Experts.apply(
        SimpleNamespace(quant_config=quant_config),
        **arguments,
    )

    assert calls[0]["hidden_states"].dtype == torch.bfloat16
    assert calls[0]["quant_config"] is quant_config
    assert calls[0]["activation"] is MoEActivation.SILU
    assert "inplace" not in calls[0]
    assert torch.equal(
        arguments["output"],
        torch.full_like(arguments["output"], 2),
    )


def test_functional_experts_calls_flaggems_with_exact_w8a8_contract(monkeypatch):
    calls = []

    def fake_fused_experts_impl(**kwargs):
        calls.append(kwargs)
        return torch.full_like(kwargs["hidden_states"], 3)

    monkeypatch.setattr(
        moe_experts,
        "_flaggems_fused_experts_impl",
        fake_fused_experts_impl,
    )
    quant_config = _quant_config()
    instance = SimpleNamespace(quant_config=quant_config)
    arguments = _apply_arguments()

    moe_experts.FlagGemsW8A8Experts.apply(instance, **arguments)

    assert calls[0]["hidden_states"].dtype == torch.bfloat16
    assert calls[0]["hidden_states"].shape == (2, 4)
    assert calls[0]["w1"].shape == (2, 8, 4)
    assert calls[0]["w2"].shape == (2, 4, 4)
    assert calls[0]["w1_scale"] is quant_config.w1_scale
    assert calls[0]["w2_scale"] is quant_config.w2_scale
    assert calls[0]["w1_scale"].shape == (2, 8, 1)
    assert calls[0]["w2_scale"].shape == (2, 4, 1)
    assert calls[0]["a1_scale"] is None
    assert calls[0]["a2_scale"] is None
    assert calls[0]["use_int8_w8a8"] is True
    assert calls[0]["per_channel_quant"] is True
    assert calls[0]["activation"] == MoEActivation.SILU.value
    assert calls[0]["inplace"] is False
    assert torch.equal(
        arguments["output"],
        torch.full_like(arguments["output"], 3),
    )


def test_functional_experts_rejects_prequantized_input(monkeypatch):
    monkeypatch.setattr(
        moe_experts,
        "_flaggems_fused_experts_impl",
        lambda **kwargs: pytest.fail("FlagGems fused_experts must not run"),
    )
    instance = SimpleNamespace(
        quant_config=_quant_config(),
    )
    arguments = _apply_arguments()
    arguments["hidden_states"] = torch.ones((2, 4), dtype=torch.int8)
    arguments["a1q_scale"] = torch.ones((2, 1), dtype=torch.float32)

    with pytest.raises(ValueError, match="quantized before"):
        moe_experts.FlagGemsW8A8Experts.apply(instance, **arguments)


def test_functional_experts_rejects_activation_not_supported_by_flaggems(
    monkeypatch,
):
    monkeypatch.setattr(
        moe_experts,
        "_flaggems_fused_experts_impl",
        lambda **kwargs: pytest.fail("FlagGems fused_experts must not run"),
    )
    arguments = _apply_arguments()
    arguments["activation"] = MoEActivation.GELU

    with pytest.raises(NotImplementedError, match="only silu"):
        moe_experts.FlagGemsW8A8Experts.apply(
            SimpleNamespace(quant_config=_quant_config()),
            **arguments,
        )


@pytest.mark.parametrize(
    ("scale_name", "bad_scale"),
    [
        ("w1_scale", torch.ones((2, 1, 8), dtype=torch.float32)),
        ("w2_scale", torch.ones((2, 1, 4), dtype=torch.float32)),
        ("w1_scale", torch.ones((2, 8, 1), dtype=torch.bfloat16)),
    ],
)
def test_functional_experts_rejects_incompatible_weight_scale(
    monkeypatch,
    scale_name,
    bad_scale,
):
    monkeypatch.setattr(
        moe_experts,
        "_flaggems_fused_experts_impl",
        lambda **kwargs: pytest.fail("FlagGems fused_experts must not run"),
    )
    quant_config = _quant_config()
    setattr(quant_config, scale_name, bad_scale)
    instance = SimpleNamespace(quant_config=quant_config)

    with pytest.raises((TypeError, ValueError), match="weight scales|scale must"):
        moe_experts.FlagGemsW8A8Experts.apply(instance, **_apply_arguments())
