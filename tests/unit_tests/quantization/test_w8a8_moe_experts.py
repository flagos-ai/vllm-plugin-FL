# Copyright 2026 FlagOS Contributors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
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


def test_functional_experts_defer_activation_quantization():
    instance = SimpleNamespace()
    assert (
        moe_experts.TritonW8A8Experts.expects_unquantized_inputs.fget(instance) is True
    )


@pytest.mark.parametrize("supports_inplace", [True, False])
def test_functional_experts_calls_vllm_with_float_input(
    monkeypatch,
    supports_inplace,
):
    calls = []

    if supports_inplace:

        def fake_fused_experts(
            hidden_states,
            w1,
            w2,
            topk_weights,
            topk_ids,
            inplace,
            activation,
            apply_router_weight_on_input,
            global_num_experts,
            expert_map,
            quant_config,
        ):
            calls.append(
                {
                    "hidden_states": hidden_states,
                    "inplace": inplace,
                    "quant_config": quant_config,
                }
            )
            return torch.full_like(hidden_states, 3)

    else:

        def fake_fused_experts(
            hidden_states,
            w1,
            w2,
            topk_weights,
            topk_ids,
            activation,
            apply_router_weight_on_input,
            global_num_experts,
            expert_map,
            quant_config,
        ):
            calls.append(
                {
                    "hidden_states": hidden_states,
                    "quant_config": quant_config,
                }
            )
            return torch.full_like(hidden_states, 3)

    monkeypatch.setattr(moe_experts, "fused_experts", fake_fused_experts)
    monkeypatch.setattr(
        moe_experts,
        "_FUSED_EXPERTS_HAS_INPLACE",
        supports_inplace,
    )
    quant_config = SimpleNamespace(use_int8_w8a8=True)
    instance = SimpleNamespace(quant_config=quant_config)
    arguments = _apply_arguments()

    moe_experts.TritonW8A8Experts.apply(instance, **arguments)

    assert calls[0]["hidden_states"].dtype == torch.bfloat16
    assert calls[0]["quant_config"] is quant_config
    if supports_inplace:
        assert calls[0]["inplace"] is False
    assert torch.equal(
        arguments["output"],
        torch.full_like(arguments["output"], 3),
    )


def test_functional_experts_rejects_prequantized_input(monkeypatch):
    monkeypatch.setattr(
        moe_experts,
        "fused_experts",
        lambda **kwargs: pytest.fail("fused_experts must not run"),
    )
    instance = SimpleNamespace(
        quant_config=SimpleNamespace(use_int8_w8a8=True),
    )
    arguments = _apply_arguments()
    arguments["hidden_states"] = torch.ones((2, 4), dtype=torch.int8)
    arguments["a1q_scale"] = torch.ones((2, 1), dtype=torch.float32)

    with pytest.raises(ValueError, match="quantized before"):
        moe_experts.TritonW8A8Experts.apply(instance, **arguments)
