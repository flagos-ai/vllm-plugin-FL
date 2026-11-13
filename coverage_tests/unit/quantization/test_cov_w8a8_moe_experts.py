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

"""CPU tests for the W8A8 MoE expert adapters.

The vLLM Triton expert base class and the FlagGems ``fused_experts_impl``
kernel are recorders from ``_w8a8_fakes``.  The tests assert the exact
keyword contract each adapter marshals (dtypes, shapes, scales, bias and
routing flags) and re-derive the copied output from the recorded result.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from ._w8a8_fakes import (
    MoEActivation,
    QuantConfig,
    build_moe_experts_environment,
    import_under_fakes,
)

MODULE = "vllm_fl.quantization.w8a8.moe_experts"

_EXC = {
    "value": ValueError,
    "not_impl": NotImplementedError,
    "type": TypeError,
    "rows": ValueError,
}


@pytest.fixture
def experts():
    env = build_moe_experts_environment()
    with import_under_fakes(MODULE, env.modules, purge=(MODULE,)) as module:
        # ``import vllm_fl.envs as fl_envs`` resolves through the real parent
        # package's attribute when that module is already loaded, bypassing the
        # injected module.  Rebind the module global to the controlled stand-in.
        module.fl_envs = env.fl_envs
        yield module, env


def _moe_config(**attrs):
    defaults = {
        "swiglu_limit": None,
        "swiglu_alpha": None,
        "swiglu_beta": None,
    }
    defaults.update(attrs)
    return SimpleNamespace(**defaults)


def _apply_args(
    module,
    *,
    hidden=None,
    w1=None,
    w2=None,
    topk_weights=None,
    topk_ids=None,
    activation=None,
    a1q_scale=None,
    a2_scale=None,
):
    hidden = torch.randn(3, 8) if hidden is None else hidden
    w1 = torch.randint(-8, 8, (2, 8, 8), dtype=torch.int8) if w1 is None else w1
    w2 = torch.randint(-8, 8, (2, 8, 4), dtype=torch.int8) if w2 is None else w2
    topk_weights = torch.ones(3, 2) if topk_weights is None else topk_weights
    topk_ids = (
        torch.tensor([[0, 1], [1, 0], [0, 0]], dtype=torch.int64)
        if topk_ids is None
        else topk_ids
    )
    return {
        "output": torch.zeros_like(hidden),
        "hidden_states": hidden,
        "w1": w1,
        "w2": w2,
        "topk_weights": topk_weights,
        "topk_ids": topk_ids,
        "activation": MoEActivation.SILU if activation is None else activation,
        "global_num_experts": 2,
        "expert_map": None,
        "a1q_scale": a1q_scale,
        "a2_scale": a2_scale,
        "workspace13": torch.empty(0),
        "workspace2": torch.empty(0),
        "expert_tokens_meta": None,
        "apply_router_weight_on_input": True,
    }


def _quant_config(**changes):
    base = {
        "use_int8_w8a8": True,
        "per_act_token_quant": True,
        "block_shape": None,
        "w1_scale": torch.ones(2, 8, 1),
        "w2_scale": torch.ones(2, 8, 1),
        "w1_bias": None,
        "w2_bias": None,
    }
    base.update(changes)
    return QuantConfig(**base)


def test_constructor_copies_swiglu_params_without_mutating_source(experts):
    module, env = experts
    source = _quant_config()
    cls = module.VllmFunctionalW8A8Experts
    instance = cls(
        _moe_config(swiglu_limit=7.0, swiglu_alpha=0.5, swiglu_beta=1.25),
        source,
    )
    assert instance._use_modular is True
    forwarded = env.experts_init_calls[0][1]
    assert forwarded is not source
    assert forwarded.gemm1_clamp_limit == 7.0
    assert forwarded.gemm1_alpha == 0.5
    assert forwarded.gemm1_beta == 1.25
    assert source.gemm1_clamp_limit is None


def test_constructor_without_swiglu_params_keeps_config(experts):
    module, env = experts
    source = _quant_config()
    instance = module.VllmFunctionalW8A8Experts(_moe_config(), source)
    assert instance._use_modular is False
    assert env.experts_init_calls[0][1] is source
    assert instance.expects_unquantized_inputs is True


@pytest.mark.parametrize(
    "moe_attrs",
    [
        {"swiglu_limit": 7.0},
        {"swiglu_alpha": 2.0},
        {"swiglu_beta": 1.0},
    ],
)
def test_vllm_functional_apply_routes_modular_configs_to_super(experts, moe_attrs):
    module, env = experts
    instance = module.VllmFunctionalW8A8Experts(
        _moe_config(**moe_attrs), _quant_config()
    )
    kwargs = _apply_args(module)
    instance.apply(**kwargs)
    assert len(env.super_apply_calls) == 1
    assert env.vllm_fused_experts_calls == []
    forwarded = env.super_apply_calls[0]
    assert forwarded["hidden_states"] is kwargs["hidden_states"]
    assert forwarded["activation"] is MoEActivation.SILU


def test_vllm_functional_apply_routes_non_silu_to_super(experts):
    module, env = experts
    instance = module.VllmFunctionalW8A8Experts(_moe_config(), _quant_config())
    kwargs = _apply_args(module, activation=MoEActivation.GELU)
    instance.apply(**kwargs)
    assert len(env.super_apply_calls) == 1
    assert env.vllm_fused_experts_calls == []


def test_vllm_functional_apply_calls_fused_experts_and_copies(experts):
    module, env = experts
    instance = module.VllmFunctionalW8A8Experts(_moe_config(), _quant_config())
    kwargs = _apply_args(module)
    env.vllm_result = torch.full_like(kwargs["output"], 2.5)
    instance.apply(**kwargs)

    assert kwargs["output"].eq(2.5).all()
    call = env.vllm_fused_experts_calls[0]
    assert call["hidden_states"] is kwargs["hidden_states"]
    assert call["w1"] is kwargs["w1"]
    assert call["w2"] is kwargs["w2"]
    assert call["topk_weights"] is kwargs["topk_weights"]
    assert call["topk_ids"] is kwargs["topk_ids"]
    assert call["activation"] is MoEActivation.SILU
    assert call["apply_router_weight_on_input"] is True
    assert call["global_num_experts"] == 2
    assert call["expert_map"] is None
    assert call["quant_config"] is instance.quant_config


@pytest.mark.parametrize(
    ("changes", "a1q", "a2q", "dtype", "lora", "exc_key", "match"),
    [
        (
            {"use_int8_w8a8": False},
            None,
            None,
            torch.float16,
            None,
            "value",
            "INT8 W8A8 quant config",
        ),
        (
            {"per_act_token_quant": False},
            None,
            None,
            torch.float16,
            None,
            "value",
            "dynamic per-token",
        ),
        (
            {"block_shape": (128, 128)},
            None,
            None,
            torch.float16,
            None,
            "value",
            "dynamic per-token",
        ),
        (
            {},
            None,
            None,
            torch.float16,
            object(),
            "not_impl",
            "does not support LoRA",
        ),
        (
            {},
            torch.ones(3, 1),
            None,
            torch.float16,
            None,
            "value",
            "quantized before",
        ),
        (
            {},
            None,
            torch.ones(3, 1),
            torch.float16,
            None,
            "value",
            "quantized before",
        ),
        (
            {},
            None,
            None,
            torch.int8,
            None,
            "type",
            "floating-point hidden states",
        ),
    ],
)
def test_vllm_functional_apply_rejects_invalid_contract(
    experts, changes, a1q, a2q, dtype, lora, exc_key, match
):
    module, _ = experts
    instance = module.VllmFunctionalW8A8Experts(_moe_config(), _quant_config(**changes))
    instance._lora_context = lora
    hidden = torch.randn(3, 8).to(dtype)
    kwargs = _apply_args(module, hidden=hidden, a1q_scale=a1q, a2_scale=a2q)
    with pytest.raises(_EXC[exc_key], match=match):
        instance.apply(**kwargs)


def test_flag_gems_apply_marshals_full_w8a8_contract(experts):
    module, env = experts
    w1_scale = torch.rand(2, 8, 1)
    w2_scale = torch.rand(2, 8, 1)
    w1_bias = torch.rand(2, 8)
    w2_bias = torch.rand(2, 8)
    quant_config = _quant_config(
        w1_scale=w1_scale, w2_scale=w2_scale, w1_bias=w1_bias, w2_bias=w2_bias
    )
    instance = module.FlagGemsW8A8Experts(_moe_config(), quant_config)
    assert instance.expects_unquantized_inputs is True

    kwargs = _apply_args(module)
    env.flaggems_result = torch.full_like(kwargs["output"], -1.5)
    instance.apply(**kwargs)

    assert kwargs["output"].eq(-1.5).all()
    call = env.flaggems_calls[0]
    assert call["hidden_states"] is kwargs["hidden_states"]
    assert call["w1"] is kwargs["w1"]
    assert call["w2"] is kwargs["w2"]
    assert call["topk_weights"] is kwargs["topk_weights"]
    assert call["topk_ids"] is kwargs["topk_ids"]
    assert call["inplace"] is False
    assert call["activation"] == "silu"
    assert call["apply_router_weight_on_input"] is True
    assert call["use_fp8_w8a8"] is False
    assert call["use_int8_w8a8"] is True
    assert call["use_int8_w8a16"] is False
    assert call["use_int4_w4a16"] is False
    assert call["per_channel_quant"] is True
    assert call["global_num_experts"] == 2
    assert call["expert_map"] is None
    assert call["w1_scale"] is w1_scale
    assert call["w2_scale"] is w2_scale
    assert call["a1_scale"] is None
    assert call["a2_scale"] is None
    assert call["block_shape"] is None
    assert call["w1_bias"] is w1_bias
    assert call["w2_bias"] is w2_bias


@pytest.mark.parametrize(
    ("changes", "a1q", "dtype", "lora", "exc_key", "match"),
    [
        (
            {"use_int8_w8a8": False},
            None,
            torch.float16,
            None,
            "value",
            "INT8 W8A8 quant config",
        ),
        (
            {"per_act_token_quant": False},
            None,
            torch.float16,
            None,
            "value",
            "dynamic per-token",
        ),
        (
            {"block_shape": (128, 128)},
            None,
            torch.float16,
            None,
            "value",
            "dynamic per-token",
        ),
        (
            {},
            None,
            torch.float16,
            object(),
            "not_impl",
            "does not support LoRA",
        ),
        (
            {},
            torch.ones(3, 1),
            torch.float16,
            None,
            "value",
            "quantized before",
        ),
        (
            {},
            None,
            torch.int8,
            None,
            "type",
            "floating-point hidden states",
        ),
    ],
)
def test_flag_gems_apply_rejects_invalid_contract(
    experts, changes, a1q, dtype, lora, exc_key, match
):
    module, _ = experts
    instance = module.FlagGemsW8A8Experts(_moe_config(), _quant_config(**changes))
    instance._lora_context = lora
    kwargs = _apply_args(module, hidden=torch.randn(3, 8).to(dtype), a1q_scale=a1q)
    with pytest.raises(_EXC[exc_key], match=match):
        instance.apply(**kwargs)


def _contract_call(module, **overrides):
    hidden = torch.randn(3, 8)
    w1 = torch.randint(-4, 4, (2, 8, 8), dtype=torch.int8)
    w2 = torch.randint(-4, 4, (2, 8, 4), dtype=torch.int8)
    topk_weights = torch.ones(3, 2)
    topk_ids = torch.tensor([[0, 1], [1, 0], [0, 0]], dtype=torch.int64)
    values = {
        "hidden_states": hidden,
        "w1": w1,
        "w2": w2,
        "topk_weights": topk_weights,
        "topk_ids": topk_ids,
        "activation": MoEActivation.SILU,
        "w1_scale": torch.ones(2, 8, 1),
        "w2_scale": torch.ones(2, 8, 1),
        "w1_bias": None,
        "w2_bias": None,
    }
    values.update(overrides)
    return module, values


def test_validate_w8a8_contract_accepts_valid_inputs(experts):
    module, _ = experts
    _, values = _contract_call(module)
    result = module._validate_w8a8_contract(**values)
    # The validator is a pure check: valid inputs return None and are not
    # mutated (shapes and dtypes of the contract tensors are unchanged).
    assert result is None
    assert values["w1"].shape == (2, 8, 8)
    assert values["w2"].shape == (2, 8, 4)
    assert values["w1_scale"].shape == (2, 8, 1)


@pytest.mark.parametrize(
    ("mutation", "exc_key", "match"),
    [
        (
            lambda v: v.update(activation=MoEActivation.GELU),
            "not_impl",
            "only silu",
        ),
        (
            lambda v: v.update(hidden_states=torch.randn(2, 3, 8)),
            "value",
            "contiguous",
        ),
        (
            lambda v: v.update(hidden_states=torch.randn(8, 3).t()),
            "value",
            "contiguous",
        ),
        (
            lambda v: v.update(w1=torch.zeros(8, 8, dtype=torch.int8)),
            "value",
            "3D expert weights",
        ),
        (
            lambda v: v.update(w1=v["w1"].float()),
            "type",
            "must be int8",
        ),
        (
            lambda v: v.update(w2=torch.zeros(3, 8, 4, dtype=torch.int8)),
            "value",
            "weight shapes are incompatible",
        ),
        (
            lambda v: v.update(w2=torch.zeros(2, 8, 5, dtype=torch.int8)),
            "value",
            "gate/up and down-projection shapes",
        ),
        (
            lambda v: v.update(w1=v["w1"].transpose(1, 2)),
            "value",
            "contiguous in K",
        ),
        (
            lambda v: v.update(topk_weights=torch.ones(3, 3)),
            "value",
            "matching \\[M, top_k\\] routing",
        ),
        (
            lambda v: (
                v.update(topk_weights=torch.ones(4, 2)),
                v.update(topk_ids=torch.zeros(4, 2, dtype=torch.int64)),
            ),
            "rows",
            "routing and activation rows",
        ),
        (
            lambda v: v.update(topk_weights=torch.ones(2, 3).t()),
            "value",
            "routing must be contiguous",
        ),
        (
            lambda v: v.update(w1_scale=None),
            "value",
            "w1_scale must have shape",
        ),
        (
            lambda v: v.update(w1_scale=torch.ones(2, 8)),
            "value",
            "w1_scale must have shape",
        ),
        (
            lambda v: v.update(w1_scale=torch.ones(2, 8, 1, dtype=torch.float64)),
            "type",
            "float32",
        ),
        (
            lambda v: v.update(w1_scale=torch.ones(2, 8, 2)[..., :1]),
            "value",
            "must be contiguous",
        ),
        (
            lambda v: v.update(w2_scale=None),
            "value",
            "w2_scale must have shape",
        ),
        (
            lambda v: v.update(w1_scale=torch.ones(2, 8, 1, device="meta")),
            "value",
            "share a device",
        ),
        (
            lambda v: v.update(w1_bias=torch.ones(2, 7)),
            "value",
            "w1_bias must have shape",
        ),
        (
            lambda v: v.update(w2_bias=torch.ones(2, 7)),
            "value",
            "w2_bias must have shape",
        ),
    ],
)
def test_validate_w8a8_contract_rejects_invalid_inputs(
    experts, mutation, exc_key, match
):
    module, _ = experts
    _, values = _contract_call(module)
    mutation(values)
    with pytest.raises(_EXC[exc_key], match=match):
        module._validate_w8a8_contract(**values)


def test_flaggems_impl_selects_backend_by_environment(experts):
    module, env = experts
    env.fl_envs.__dict__["VLLM_FL_USE_FLAGGEMS_VLLM"] = True
    try:
        module._flaggems_fused_experts_impl(marker=1)
        assert env.flaggems_vllm_calls == [{"marker": 1}]
        assert env.flaggems_calls == []
    finally:
        env.fl_envs.__dict__["VLLM_FL_USE_FLAGGEMS_VLLM"] = False
    module._flaggems_fused_experts_impl(marker=2)
    assert env.flaggems_calls == [{"marker": 2}]
