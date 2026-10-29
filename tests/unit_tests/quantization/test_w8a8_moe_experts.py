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

import sys
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch

from vllm.model_executor.layers.fused_moe.activation import MoEActivation

from vllm_fl.ops.fused_moe import fused_moe as pipeline
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
        gemm1_alpha=None,
        gemm1_beta=None,
        gemm1_clamp_limit=None,
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


@pytest.mark.parametrize(
    "parameters",
    [{}, {"swiglu_limit": None, "swiglu_alpha": None, "swiglu_beta": None}],
)
def test_native_fallback_preserves_original_config_without_activation_parameters(
    monkeypatch,
    parameters,
):
    original = _quant_config()
    config = SimpleNamespace(**parameters)
    calls = []

    def init(self, moe_config, quant_config):
        calls.append((moe_config, quant_config))

    monkeypatch.setattr(moe_experts.TritonExperts, "__init__", init)
    instance = moe_experts.VllmFunctionalW8A8Experts(config, original)
    assert calls[0][0] is config
    assert calls[0][1] is original
    assert instance._use_modular is False


def test_modular_fallback_copies_activation_parameters(monkeypatch):
    from vllm.model_executor.layers.fused_moe.config import int8_w8a8_moe_quant_config

    original = int8_w8a8_moe_quant_config(
        w1_scale=torch.ones((2, 8, 1)),
        w2_scale=torch.ones((2, 4, 1)),
        a1_scale=None,
        a2_scale=None,
        per_act_token_quant=True,
    )
    config = SimpleNamespace(swiglu_limit=7.0, swiglu_alpha=1.702, swiglu_beta=1.0)
    calls = []

    def init(self, moe_config, quant_config):
        calls.append((moe_config, quant_config))

    monkeypatch.setattr(moe_experts.TritonExperts, "__init__", init)
    moe_experts.VllmFunctionalW8A8Experts(config, original)
    forwarded = calls[0][1]
    assert calls[0][0] is config
    assert forwarded is not original
    assert forwarded.gemm1_clamp_limit == 7.0
    assert forwarded.gemm1_alpha == 1.702
    assert forwarded.gemm1_beta == 1.0
    assert forwarded.use_int8_w8a8 and forwarded.per_act_token_quant
    assert original.gemm1_clamp_limit is None
    assert original.gemm1_alpha is None
    assert original.gemm1_beta is None


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
        SimpleNamespace(quant_config=quant_config, _use_modular=False),
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
    instance = moe_experts.FlagGemsW8A8Experts(
        SimpleNamespace(activation=MoEActivation.SILU), quant_config
    )
    arguments = _apply_arguments()

    instance.apply(**arguments)

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


@pytest.mark.parametrize(
    "experts_type",
    [moe_experts.FlagGemsW8A8Experts, moe_experts.VllmFunctionalW8A8Experts],
)
@pytest.mark.parametrize(
    "activation", [MoEActivation.SILU, MoEActivation.SWIGLUOAI_UNINTERLEAVE]
)
def test_functional_experts_rejects_prequantized_input(
    monkeypatch, experts_type, activation
):
    monkeypatch.setattr(
        moe_experts,
        "_flaggems_fused_experts_impl",
        lambda **kwargs: pytest.fail("FlagGems fused_experts must not run"),
    )
    instance = SimpleNamespace(
        quant_config=_quant_config(),
    )
    arguments = _apply_arguments()
    arguments["activation"] = activation
    arguments["hidden_states"] = torch.ones((2, 4), dtype=torch.int8)
    arguments["a1q_scale"] = torch.ones((2, 1), dtype=torch.float32)

    with pytest.raises(ValueError, match="quantized before"):
        experts_type.apply(instance, **arguments)


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


def _split_moe_config(**kwargs):
    return SimpleNamespace(
        activation=MoEActivation.SWIGLUOAI_UNINTERLEAVE,
        **{"swiglu_alpha": 1.702, "swiglu_beta": 1.0, "swiglu_limit": 7.0, **kwargs},
    )


def test_oai_params_are_local_to_selected_experts_and_keep_quant_config():
    q = _quant_config()
    before = vars(q).copy()
    cfg1 = _split_moe_config()
    cfg2 = _split_moe_config(swiglu_alpha=0.7, swiglu_beta=-2.0, swiglu_limit=3.5)
    first = moe_experts.FlagGemsW8A8Experts(cfg1, q)
    second = moe_experts.FlagGemsW8A8Experts(cfg2, q)
    assert first.quant_config is q and second.quant_config is q
    assert first.moe_config is cfg1 and second.moe_config is cfg2
    assert first._activation_kwargs == {
        "gemm1_alpha": 1.702,
        "gemm1_beta": 1.0,
        "gemm1_clamp_limit": 7.0,
    }
    assert second._activation_kwargs == {
        "gemm1_alpha": 0.7,
        "gemm1_beta": -2.0,
        "gemm1_clamp_limit": 3.5,
    }
    assert first._activation_kwargs is not second._activation_kwargs
    assert vars(q).keys() == before.keys()
    assert all((vars(q)[key] is value for key, value in before.items()))
    assert q.use_int8_w8a8 and q.per_act_token_quant
    assert q.gemm1_alpha is q.gemm1_beta is q.gemm1_clamp_limit is None


@pytest.mark.parametrize(
    "field,value",
    [
        ("swiglu_alpha", None),
        ("swiglu_beta", float("nan")),
        ("swiglu_limit", 0),
        ("swiglu_limit", float("inf")),
    ],
)
def test_bad_oai_metadata_rejected_in_selected_experts(field, value):
    with pytest.raises(ValueError):
        moe_experts.FlagGemsW8A8Experts(
            _split_moe_config(**{field: value}), _quant_config()
        )


@pytest.mark.parametrize("field", ["use_int8_w8a8", "per_act_token_quant"])
def test_split_oai_requires_dynamic_w8a8_only_in_selected_experts(field):
    q = _quant_config()
    setattr(q, field, False)
    with pytest.raises(ValueError, match="dynamic-token INT8 adapter"):
        moe_experts.FlagGemsW8A8Experts(_split_moe_config(), q)


def test_split_oai_uses_plugin_quantized_pipeline(monkeypatch):
    calls = []

    def fused(*pos, **kwargs):
        calls.append((pos, kwargs))
        return torch.ones_like(kwargs["hidden_states"])

    monkeypatch.setattr(pipeline, "fused_experts_impl", fused)
    monkeypatch.setattr(
        moe_experts,
        "_flaggems_fused_experts_impl",
        lambda **kwargs: pytest.fail("split OAI must not enter SiLU-only FlagGems"),
    )
    q = _quant_config()
    args = _apply_arguments()
    args["activation"] = MoEActivation.SWIGLUOAI_UNINTERLEAVE
    instance = moe_experts.FlagGemsW8A8Experts(_split_moe_config(), q)
    instance.apply(**args)
    assert len(calls) == 1
    assert calls[0][1]["hidden_states"] is args["hidden_states"]
    assert calls[0][1]["activation"] == "swigluoai_uninterleave"
    assert calls[0][1]["use_int8_w8a8"] is True
    assert calls[0][1]["per_channel_quant"] is True
    assert calls[0][1]["gemm1_alpha"] == 1.702
    assert calls[0][1]["gemm1_beta"] == 1.0
    assert calls[0][1]["gemm1_clamp_limit"] == 7.0
    assert torch.equal(args["output"], torch.ones_like(args["output"]))
    assert q.gemm1_alpha is q.gemm1_beta is q.gemm1_clamp_limit is None
    instance._activation_kwargs.pop("gemm1_beta")
    with pytest.raises(ValueError, match="explicit gate parameters"):
        moe_experts.FlagGemsW8A8Experts.apply(instance, **args)


@pytest.fixture
def oai_chain(monkeypatch):
    def run(*, enabled=True, allowed=True, operator=None, per_channel=True):
        from vllm_fl.ops.fused_moe import activation

        monkeypatch.setenv("VLLM_FL_USE_FLAGGEMS_VLLM", "1" if enabled else "0")
        monkeypatch.setattr(activation, "use_flaggems_op", lambda op: allowed)
        monkeypatch.setattr(
            pipeline,
            "try_get_optimal_moe_config",
            lambda *args, **kwargs: {"BLOCK_SIZE_M": 4},
        )
        monkeypatch.setattr(
            pipeline, "_get_config_quant_dtype", lambda **kwargs: torch.int8
        )
        monkeypatch.setattr(pipeline, "_get_config_dtype_str", lambda **kwargs: "int8")
        quantize = Mock(
            side_effect=lambda **kwargs: (
                kwargs["A"].to(torch.int8),
                torch.ones(kwargs["A"].shape[0], 1),
            )
        )
        monkeypatch.setattr(pipeline, "moe_kernel_quantize_input", quantize)
        monkeypatch.setattr(
            pipeline,
            "_moe_align_block_size",
            lambda *args, **kwargs: (None, None, None),
        )
        gemms = []

        def gemm(a, b, output, *args, **kwargs):
            gemms.append((a, args))
            output.fill_(1)

        monkeypatch.setattr(pipeline, "_invoke_fused_moe_triton_kernel", gemm)
        monkeypatch.setattr(pipeline, "_moe_sum", lambda x, out: out.copy_(x.sum(1)))
        activation_op = Mock(side_effect=lambda kind, out, x, **kwargs: out.fill_(1))
        monkeypatch.setattr(activation, "apply_moe_activation", activation_op)
        module = SimpleNamespace()
        if operator is not None:
            module.try_swigluoai_uninterleave_quant_int8 = operator
        monkeypatch.setitem(sys.modules, "flaggems_vllm", module)
        output = pipeline.fused_experts_impl(
            torch.ones(2, 2),
            torch.ones(2, 4, 2, dtype=torch.int8),
            torch.ones(2, 2, 2, dtype=torch.int8),
            torch.ones(2, 1),
            torch.zeros(2, 1, dtype=torch.int32),
            activation="swigluoai_uninterleave",
            use_int8_w8a8=True,
            per_channel_quant=per_channel,
            gemm1_alpha=1.702,
            gemm1_beta=1.0,
            gemm1_clamp_limit=7.0,
        )
        assert output.shape == (2, 2) and torch.equal(output, torch.ones_like(output))
        return (quantize, activation_op, gemms)

    return run


@pytest.mark.parametrize(
    "options", [{"enabled": False}, {"allowed": False}, {"per_channel": False}]
)
def test_original_chain_is_preserved_when_fusion_is_not_selected(oai_chain, options):
    operator = Mock()
    quantize, activation, gemms = oai_chain(operator=operator, **options)
    operator.assert_not_called()
    assert quantize.call_count == 2 and activation.call_count == 1 and (len(gemms) == 2)
    assert activation.call_args.kwargs == {
        "clamp_limit": 7.0,
        "alpha": 1.702,
        "beta": 1.0,
    }


@pytest.mark.parametrize("available", [False, True])
def test_missing_or_unsupported_fusion_keeps_original_chain(oai_chain, available):
    operator = Mock(return_value=None) if available else None
    quantize, activation, _ = oai_chain(operator=operator)
    assert quantize.call_count == 2 and activation.call_count == 1


def test_existing_fusion_result_goes_directly_to_gemm2(oai_chain):
    q, scale = (torch.full((2, 2), 3, dtype=torch.int8), torch.ones(2, 1))
    operator = Mock(return_value=(q, scale))
    quantize, activation, gemms = oai_chain(operator=operator)
    assert quantize.call_count == 1 and activation.call_count == 0
    assert gemms[1][0] is q and gemms[1][1][0] is scale
    assert operator.call_args.kwargs == {
        "clamp_limit": 7.0,
        "alpha": 1.702,
        "beta": 1.0,
    }


def test_selected_execution_error_is_not_hidden(oai_chain):
    with pytest.raises(RuntimeError, match="kernel failed"):
        oai_chain(operator=Mock(side_effect=RuntimeError("kernel failed")))


@pytest.mark.parametrize("switch", [None, "0", "1"])
def test_fused_experts_library_choice(monkeypatch, switch):
    import flag_gems

    if switch is None:
        monkeypatch.delenv("VLLM_FL_USE_FLAGGEMS_VLLM", raising=False)
    else:
        monkeypatch.setenv("VLLM_FL_USE_FLAGGEMS_VLLM", switch)
    fg, fv = Mock(), Mock()
    monkeypatch.setattr(flag_gems, "fused_experts_impl", fg, raising=False)
    monkeypatch.setitem(
        sys.modules, "flaggems_vllm", SimpleNamespace(fused_experts_impl=fv)
    )
    moe_experts._flaggems_fused_experts_impl(hidden_states=None)
    (fv if switch == "1" else fg).assert_called_once_with(hidden_states=None)
    (fg if switch == "1" else fv).assert_not_called()


@pytest.mark.parametrize(
    "activation,parameters",
    [
        (MoEActivation.GELU, {}),
        (
            MoEActivation.SWIGLUOAI_UNINTERLEAVE,
            {"swiglu_limit": 7.0, "swiglu_alpha": 1.702, "swiglu_beta": 1.0},
        ),
        (MoEActivation.SILU, {"swiglu_limit": 7.0}),
        (MoEActivation.SILU, {"swiglu_alpha": 1.702, "swiglu_beta": 1.0}),
    ],
)
def test_native_experts_delegate_configurable_activation(
    monkeypatch, activation, parameters
):
    calls = []

    def init(self, moe_config, quant_config):
        self.quant_config = quant_config

    monkeypatch.setattr(moe_experts.TritonExperts, "__init__", init)
    monkeypatch.setattr(
        moe_experts.TritonExperts,
        "apply",
        lambda self, **kwargs: calls.append(kwargs),
    )
    monkeypatch.setattr(
        moe_experts,
        "_vllm_fused_experts",
        lambda **kwargs: pytest.fail(
            "Configurable activation must use modular experts"
        ),
    )
    from vllm.model_executor.layers.fused_moe.config import int8_w8a8_moe_quant_config

    quant_config = int8_w8a8_moe_quant_config(
        w1_scale=torch.ones((2, 8, 1)),
        w2_scale=torch.ones((2, 4, 1)),
        a1_scale=None,
        a2_scale=None,
        per_act_token_quant=True,
    )
    instance = moe_experts.VllmFunctionalW8A8Experts(
        SimpleNamespace(**parameters),
        quant_config,
    )
    arguments = _apply_arguments()
    arguments["activation"] = activation
    instance.apply(**arguments)
    assert len(calls) == 1
    assert calls[0].keys() == arguments.keys()
    assert all(calls[0][name] is value for name, value in arguments.items())
