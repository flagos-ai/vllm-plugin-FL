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

"""CPU tests for the unquantized fused-MoE expert entry points.

``vllm_fl.ops.fused_moe.fused_moe`` validates the token/weight shapes, asks
upstream for the launch configuration, then drives the alignment, expert and
activation kernels through ``CachedOp`` dispatch.  The kernels are the
FlagGems/triton boundary and are replaced here by explicit two-phase
recorders that compute the documented expert math over the recorded launch
parameters.  Every numeric expectation is derived in the test from plain
torch over the same weights, ids and router weights.

The image's upstream ``_get_config_quant_dtype`` rejects the keyword the
plugin passes (SOURCE_BUGS S6): the helper binding is therefore replaced
with the documented three-flag signature so the reachable normal branches can
be exercised.  The failure itself is not asserted anywhere.
"""

from __future__ import annotations

import pytest
import torch

from . import _imports  # noqa: F401  (installs CPU-only import placeholders)
from vllm_fl.ops.fused_moe import fused_moe as Mod

F = torch.nn.functional


class _Kernels:
    """Two-phase fake for the fused-MoE kernel boundary."""

    def __init__(self, topk):
        self.topk = topk
        self.phase = "w1"
        self.align_calls = []
        self.kernel_calls = []
        self.quant_calls = []
        self.mul_calls = []
        self.sum_calls = []
        self.config_calls = []
        self.dtype_calls = []
        self.quant_dtype_calls = []

    # -- external helpers -------------------------------------------------
    def dtype_str(self, **kwargs):
        self.dtype_calls.append(kwargs)
        return kwargs.get("dtype", torch.float32)

    def quant_dtype(self, **kwargs):
        self.quant_dtype_calls.append(kwargs)
        return kwargs.get("result")

    def optimal_config(self, *args, **kwargs):
        self.config_calls.append((args, kwargs))
        return {"BLOCK_SIZE_M": 1}

    def quantize(self, **kwargs):
        self.quant_calls.append(kwargs)
        return kwargs["A"], kwargs["A_scale"]

    # -- kernels ----------------------------------------------------------
    def align(self, topk_ids, block_size, global_experts, expert_map, **_):
        self.align_calls.append(
            {
                "topk_ids": topk_ids,
                "block_size": block_size,
                "global_experts": global_experts,
                "expert_map": expert_map,
            }
        )
        assert block_size == 1
        flat = topk_ids.reshape(-1)
        order = torch.argsort(flat, stable=True)
        experts = flat[order].to(torch.int32)
        if expert_map is not None:
            experts = expert_map[experts]
        entries = order.to(torch.int32)
        return entries, experts, torch.tensor(entries.numel(), dtype=torch.int32)

    def invoke(
        self,
        activation,
        weights,
        output,
        act_scale,
        weight_scale,
        router_weights,
        sorted_ids,
        expert_ids,
        num_post,
        apply_router_weight_on_input,
        top_k_num,
        config,
        **kwargs,
    ):
        self.kernel_calls.append(
            {
                "A": activation,
                "B": weights,
                "C": output,
                "act_scale": act_scale,
                "weight_scale": weight_scale,
                "router_weights": router_weights,
                "sorted_ids": sorted_ids,
                "expert_ids": expert_ids,
                "num_post": num_post,
                "apply_router_weight_on_input": apply_router_weight_on_input,
                "top_k_num": top_k_num,
                "config": config,
                **kwargs,
            }
        )
        phase, self.phase = self.phase, "w2"
        bias = kwargs.get("B_bias")
        for position in range(sorted_ids.numel()):
            entry = int(sorted_ids[position])
            expert = int(expert_ids[position])
            token, slot = divmod(entry, self.topk)
            row = token if phase == "w1" else entry
            weight = weights[expert].float()
            if phase == "w1" and kwargs.get("use_int4_w4a16"):
                # The fake boundary unpacks the packed 4-bit width by
                # duplicating each stored nibble; the reference does the same.
                weight = weight.repeat_interleave(2, dim=-1)
            out = activation[row].float() @ weight.T
            if apply_router_weight_on_input:
                out = out * router_weights[token, slot].float()
            if bias is not None:
                out = out + bias[expert].float()
            output[token, slot].copy_(out)
        return output

    def silu_and_mul(self, out, x):
        self.mul_calls.append(("silu", tuple(x.shape)))
        half = x.shape[-1] // 2
        return F.silu(x[:, :half]) * x[:, half:]

    def gelu_and_mul(self, out, x):
        self.mul_calls.append(("gelu", tuple(x.shape)))
        half = x.shape[-1] // 2
        return F.gelu(x[:, :half]) * x[:, half:]

    def moe_sum(self, src, out):
        self.sum_calls.append((tuple(src.shape), tuple(out.shape)))
        out.copy_(src.sum(dim=1))
        return out


def _install(monkeypatch, topk=2, quant_dtype=None, dtype_str="float16"):
    kernels = _Kernels(topk)
    kernels.quant_dtype = lambda **kwargs: (
        kernels.quant_dtype_calls.append(kwargs) or quant_dtype
    )
    kernels.dtype_str = lambda **kwargs: kernels.dtype_calls.append(kwargs) or dtype_str
    monkeypatch.setattr(Mod, "_get_config_dtype_str", kernels.dtype_str)
    monkeypatch.setattr(Mod, "_get_config_quant_dtype", kernels.quant_dtype)
    monkeypatch.setattr(Mod, "try_get_optimal_moe_config", kernels.optimal_config)
    monkeypatch.setattr(Mod, "moe_kernel_quantize_input", kernels.quantize)
    monkeypatch.setattr(Mod, "_moe_align_block_size", kernels.align)
    monkeypatch.setattr(Mod, "_invoke_fused_moe_triton_kernel", kernels.invoke)
    monkeypatch.setattr(Mod, "_silu_and_mul", kernels.silu_and_mul)
    monkeypatch.setattr(Mod, "_gelu_and_mul", kernels.gelu_and_mul)
    monkeypatch.setattr(Mod, "_moe_sum", kernels.moe_sum)
    monkeypatch.setattr(
        Mod,
        "FUSED_MOE_UNQUANTIZED_CONFIG",
        _quant_config(),
    )
    return kernels


def _quant_config(**overrides):
    values = {
        "use_fp8_w8a8": False,
        "use_int8_w8a8": False,
        "use_int8_w8a16": False,
        "use_int4_w4a16": False,
        "per_act_token_quant": False,
        "w1_scale": None,
        "w2_scale": None,
        "w1_zp": None,
        "w2_zp": None,
        "a1_scale": None,
        "a2_scale": None,
        "block_shape": None,
        "w1_bias": None,
        "w2_bias": None,
    }
    values.update(overrides)
    return type("QuantConfig", (), values)()


def _inputs(activation="silu", dtype=torch.float32, seed=0):
    torch.manual_seed(seed)
    tokens, hidden, out_width, experts = 2, 4, 8, 2
    states = torch.randn(tokens, hidden, dtype=torch.float32).to(dtype)
    w1 = torch.randn(experts, out_width, hidden, dtype=torch.float32).to(dtype)
    w2_width = out_width if activation.endswith("_no_mul") else out_width // 2
    w2 = torch.randn(experts, hidden, w2_width, dtype=torch.float32).to(dtype)
    weights = torch.tensor([[0.25, 0.75], [0.875, 0.125]], dtype=torch.float32)
    ids = torch.tensor([[1, 0], [0, 1]], dtype=torch.int32)
    return states, w1, w2, weights, ids


def _activation_reference(pre, activation):
    if activation == "silu":
        half = pre.numel() // 2
        return F.silu(pre[:half]) * pre[half:]
    if activation == "gelu":
        half = pre.numel() // 2
        return F.gelu(pre[:half]) * pre[half:]
    if activation == "silu_no_mul":
        return F.silu(pre)
    return F.gelu(pre)


def _reference(
    hidden,
    w1,
    w2,
    weights,
    ids,
    activation,
    packed=False,
    w1_bias=None,
    w2_bias=None,
    weight_on_input=False,
):
    tokens = hidden.shape[0]
    topk = ids.shape[1]
    out = torch.zeros(tokens, w2.shape[1], dtype=hidden.dtype, device=hidden.device)
    for token in range(tokens):
        acc = torch.zeros(w2.shape[1], dtype=torch.float32)
        for slot in range(topk):
            expert = int(ids[token, slot])
            router_weight = weights[token, slot].float()
            weight = w1[expert].float()
            if packed:
                weight = weight.repeat_interleave(2, dim=-1)
            pre = hidden[token].float() @ weight.T
            if weight_on_input:
                pre = pre * router_weight
            if w1_bias is not None:
                pre = pre + w1_bias[expert].float()
            value = _activation_reference(pre, activation)
            projected = value @ w2[expert].float().T
            if not weight_on_input:
                projected = projected * router_weight
            if w2_bias is not None:
                projected = projected + w2_bias[expert].float()
            acc += projected
        out[token].copy_(acc)
    return out


@pytest.mark.parametrize("activation", ["silu", "gelu", "silu_no_mul", "gelu_no_mul"])
def test_fused_experts_activation_paths_match_torch_reference(monkeypatch, activation):
    """Each activation dispatch produces the plain-torch SwiGLU/GELU result."""
    kernels = _install(monkeypatch)
    hidden, w1, w2, weights, ids = _inputs(activation)
    result = Mod.fused_experts(hidden, w1, w2, weights, ids, activation=activation)
    expected = _reference(hidden, w1, w2, weights, ids, activation)
    torch.testing.assert_close(result, expected, rtol=1e-5, atol=1e-5)
    assert kernels.mul_calls == (
        [("silu", (4, 8))]
        if activation == "silu"
        else [("gelu", (4, 8))]
        if activation == "gelu"
        else []
    )
    assert kernels.sum_calls == [((2, 2, 4), (2, 4))]
    # The second kernel receives the activation width actually produced.
    second = kernels.kernel_calls[1]
    assert second["A"].shape == (4, w2.shape[-1])
    assert second["apply_router_weight_on_input"] is True
    assert second["top_k_num"] == 1
    # Alignment uses the optimal BLOCK_SIZE_M and all global experts.
    align = kernels.align_calls[0]
    assert align["block_size"] == 1
    assert align["global_experts"] == 2
    assert align["expert_map"] is None
    # The quantize helper was handed the config's null scale fields.
    assert kernels.quant_calls[0]["quant_dtype"] is None
    assert kernels.quant_calls[0]["per_act_token_quant"] is False


def test_fused_experts_enum_activation_and_inplace_result_alias(monkeypatch):
    """A MoEActivation-like enum is normalized; inplace returns the input."""
    kernels = _install(monkeypatch)
    hidden, w1, w2, weights, ids = _inputs("silu")
    snapshot_reference = _reference(hidden.clone(), w1, w2, weights, ids, "silu")
    enum_like = type("Act", (), {"value": "silu"})()
    result = Mod.fused_experts_impl(
        hidden,
        w1,
        w2,
        weights,
        ids,
        inplace=True,
        activation=enum_like,
        use_fp8_w8a8=False,
        use_int8_w8a8=False,
        use_int8_w8a16=False,
        use_int4_w4a16=False,
    )
    assert result is hidden
    torch.testing.assert_close(result, snapshot_reference, rtol=1e-5, atol=1e-5)
    assert kernels.mul_calls == [("silu", (4, 8))]


def test_fused_experts_routes_quant_config_and_expert_map(monkeypatch):
    """A custom quant config's flags, scales, biases and map reach the kernels."""
    scale1 = torch.ones(1)
    scale2 = torch.full((1,), 2.0)
    a_scale = torch.full((2, 1), 0.5)
    a2_scale = torch.full((4, 1), 0.25)
    w1_bias = torch.arange(16).view(2, 8).float() / 8
    w2_bias = torch.arange(8).view(2, 4).float() / 8
    config = _quant_config(
        use_int8_w8a8=True,
        per_act_token_quant=True,
        w1_scale=scale1,
        w2_scale=scale2,
        a1_scale=a_scale,
        a2_scale=a2_scale,
        block_shape=[1, 2],
        w1_bias=w1_bias,
        w2_bias=w2_bias,
    )
    kernels = _install(monkeypatch, quant_dtype=torch.int8, dtype_str="int8_w8a8")
    hidden, w1, w2, weights, ids = _inputs("silu")
    expert_map = torch.tensor([0, 1], dtype=torch.int32)
    result = Mod.fused_experts(
        hidden,
        w1,
        w2,
        weights,
        ids,
        quant_config=config,
        global_num_experts=4,
        expert_map=expert_map,
    )
    torch.testing.assert_close(
        result,
        _reference(
            hidden,
            w1,
            w2,
            weights,
            ids,
            "silu",
            w1_bias=w1_bias,
            w2_bias=w2_bias,
        ),
        rtol=1e-5,
        atol=1e-5,
    )
    # The config dtype probe received the documented flags.
    dtype_call = kernels.dtype_calls[0]
    assert dtype_call["use_fp8_w8a8"] is False
    assert dtype_call["use_int8_w8a16"] is False
    assert dtype_call["use_int4_w4a16"] is False
    assert dtype_call["ocp_mx_scheme"] is None
    # The quant-dtype probe received the three flags and nothing else.
    assert kernels.quant_dtype_calls == [
        {"use_fp8_w8a8": False, "use_int8_w8a8": True, "ocp_mx_scheme": None}
    ]
    quant = kernels.quant_calls[0]
    assert quant["quant_dtype"] is torch.int8
    assert quant["per_act_token_quant"] is True
    assert quant["block_shape"] == [1, 2]
    assert quant["A_scale"] is a_scale
    assert kernels.quant_calls[1]["A_scale"] is a2_scale
    first, second = kernels.kernel_calls
    assert first["weight_scale"] is scale1
    assert second["weight_scale"] is scale2
    assert first["B_bias"] is w1_bias
    assert second["B_bias"] is w2_bias
    assert first["use_int8_w8a8"] is True
    assert first["per_channel_quant"] is True
    assert first["block_shape"] == [1, 2]
    assert kernels.align_calls[0]["global_experts"] == 4
    assert kernels.align_calls[0]["expert_map"] is expert_map
    # Stable expert sort keeps token order inside each expert group.
    assert first["sorted_ids"].tolist() == [1, 2, 0, 3]


def test_fused_experts_router_weight_on_input_flag_is_inverted_for_w2(monkeypatch):
    """Scalar router weights are applied in exactly one of the two kernels."""
    kernels = _install(monkeypatch)
    hidden, w1, w2, weights, ids = _inputs("silu")
    result = Mod.fused_experts(
        hidden, w1, w2, weights, ids, apply_router_weight_on_input=True
    )
    first, second = kernels.kernel_calls
    assert first["apply_router_weight_on_input"] is True
    assert second["apply_router_weight_on_input"] is False
    torch.testing.assert_close(
        result,
        _reference(hidden, w1, w2, weights, ids, "silu", weight_on_input=True),
        rtol=1e-5,
        atol=1e-5,
    )


@pytest.mark.parametrize(
    ("dtype", "compute_name"),
    [
        (torch.bfloat16, "bfloat16"),
        (torch.float16, "float16"),
        (torch.float32, "float32"),
    ],
)
def test_kernel_compute_type_tracks_hidden_dtype(monkeypatch, dtype, compute_name):
    """Each supported hidden dtype selects the matching triton compute type."""
    kernels = _install(monkeypatch)
    hidden, w1, w2, weights, ids = _inputs("silu", dtype=dtype)
    Mod.fused_experts(hidden, w1, w2, weights, ids)
    expected = getattr(Mod.tl, compute_name)
    assert all(call["compute_type"] is expected for call in kernels.kernel_calls)


def test_fused_experts_rejects_mismatched_hidden_width():
    """The first validation rejects an activation/weight width mismatch."""
    hidden = torch.zeros(2, 8)
    w1 = torch.zeros(2, 8, 4)
    w2 = torch.zeros(2, 4, 4)
    with pytest.raises(AssertionError, match="Hidden size mismatch"):
        Mod.fused_experts_impl(hidden, w1, w2, torch.zeros(2, 1), torch.zeros(2, 1))


def test_fused_experts_int4_checks_half_width_and_rejects_odd_match():
    """int4 weights are packed two-per-element and the check uses k//2."""
    hidden = torch.zeros(2, 4)
    w1 = torch.zeros(2, 8, 4)  # 4 == 4//2? no: 4 != 2
    w2 = torch.zeros(2, 4, 4)
    with pytest.raises(AssertionError, match="Hidden size mismatch"):
        Mod.fused_experts_impl(
            hidden,
            w1,
            w2,
            torch.zeros(2, 1),
            torch.zeros(2, 1),
            use_int4_w4a16=True,
        )


def test_fused_experts_accepts_packed_int4_width(monkeypatch):
    """The int4 path checks hidden/2 == packed width and runs the kernels."""
    kernels = _install(monkeypatch)
    hidden = torch.randn(2, 4)
    w1 = torch.randn(2, 2, 2)
    w2 = torch.randn(2, 4, 2)
    weights = torch.tensor([[0.5, 0.5], [0.25, 0.75]])
    ids = torch.tensor([[0, 1], [1, 0]], dtype=torch.int32)
    result = Mod.fused_experts_impl(
        hidden,
        w1,
        w2,
        weights,
        ids,
        activation="silu_no_mul",
        use_int4_w4a16=True,
    )
    expected = _reference(hidden, w1, w2, weights, ids, "silu_no_mul", packed=True)
    torch.testing.assert_close(result, expected, rtol=1e-5, atol=1e-5)
    assert all(call["use_int4_w4a16"] is True for call in kernels.kernel_calls)


def test_fused_experts_rejects_topk_shape_and_layout_violations():
    """topk shape, contiguity, last-stride and dtype are all validated."""
    hidden = torch.zeros(2, 4)
    w1 = torch.zeros(2, 8, 4)
    w2 = torch.zeros(2, 4, 4)
    valid_ids = torch.zeros(2, 1)
    valid_weights = torch.zeros(2, 1)
    with pytest.raises(AssertionError, match="topk shape mismatch"):
        Mod.fused_experts_impl(hidden, w1, w2, valid_weights, torch.zeros(2, 2))
    with pytest.raises(AssertionError, match="must be contiguous"):
        Mod.fused_experts_impl(torch.zeros(4, 4).T, w1, w2, valid_weights, valid_ids)
    with pytest.raises(AssertionError, match="Stride of last dimension"):
        Mod.fused_experts_impl(
            hidden, torch.zeros(2, 4, 8).transpose(1, 2), w2, valid_weights, valid_ids
        )
    with pytest.raises(AssertionError, match="Stride of last dimension"):
        Mod.fused_experts_impl(
            hidden, w1, torch.zeros(2, 4, 4).transpose(1, 2), valid_weights, valid_ids
        )
    with pytest.raises(AssertionError):
        Mod.fused_experts_impl(
            hidden.double(),
            w1.double(),
            w2.double(),
            valid_weights,
            valid_ids,
        )


def test_fused_experts_rejects_unknown_activation(monkeypatch):
    """An unknown activation name raises after the first expert kernel ran."""
    _install(monkeypatch)
    hidden, w1, w2, weights, ids = _inputs("silu")
    with pytest.raises(ValueError, match="Unsupported FusedMoe activation: relu"):
        Mod.fused_experts(hidden, w1, w2, weights, ids, activation="relu")


def test_fused_experts_wrapper_defaults_to_unquantized_config(monkeypatch):
    """A None quant_config uses the module's unquantized sentinel."""
    kernels = _install(monkeypatch)
    hidden, w1, w2, weights, ids = _inputs("silu")
    Mod.fused_experts(hidden, w1, w2, weights, ids)
    config = Mod.FUSED_MOE_UNQUANTIZED_CONFIG
    assert config.use_fp8_w8a8 is False
    assert config.use_int8_w8a8 is False
    assert config.per_act_token_quant is False
    assert kernels.quant_calls[0]["quant_dtype"] is None
