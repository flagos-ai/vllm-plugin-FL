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

"""Contracts of the GLM5-Next MoE, KDA and decoder building blocks.

Every test drives the host-side logic of ``vllm_fl/models/glm5/model.py`` with
inert doubles for the vLLM layers, conv/KDA kernels and mHC operators.  The
expected values are derived independently in the test (plain tensor math or
explicit argument records), so no assertion reuses the implementation under
test.  These paths are CPU-constructible; kernels are never launched here.
"""

from types import SimpleNamespace

import pytest
import torch
from torch import nn

from . import _import_shim  # noqa: F401  (idempotent; no-op on CI)
from vllm_fl.models.glm5 import model as glm5_model
from vllm_fl.models.glm5.config import Glm5NextTextConfig


def _text_config(**overrides):
    kwargs = dict(
        hidden_size=8,
        intermediate_size=16,
        moe_intermediate_size=16,
        n_routed_experts=4,
        n_shared_experts=1,
        num_experts_per_tok=2,
        norm_topk_prob=True,
        routed_scaling_factor=2.5,
        n_group=2,
        topk_group=1,
        scoring_func="sigmoid",
        topk_method="noaux_tc",
        moe_router_dtype="float32",
        swiglu_limit=10.0,
        hidden_act="silu",
    )
    kwargs.update(overrides)
    return Glm5NextTextConfig(**kwargs)


class _RecordingModule(nn.Module):
    """``nn.Module`` double that records the constructor arguments."""

    def __init__(self, *args, **kwargs):
        super().__init__()
        self.args = args
        self.kwargs = kwargs


class _RecordingGate(_RecordingModule):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.out_dtype = kwargs.get("out_dtype")
        self.calls = []

    def forward(self, x):
        self.calls.append(x)
        return x.new_zeros((x.shape[0], self.args[1])), None


class _RecordingExperts(_RecordingModule):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.is_internal_router = False
        self.calls = []

    def forward(self, hidden_states, router_logits):
        self.calls.append((hidden_states, router_logits))
        return hidden_states


class _RecordingMLP(_RecordingModule):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)


def _install_moe_doubles(monkeypatch, ep_size=1, ep_rank=0):
    monkeypatch.setattr(glm5_model, "Glm5NextMLP", _RecordingMLP)
    monkeypatch.setattr(glm5_model.fused_moe_layers, "GateLinear", _RecordingGate)
    monkeypatch.setattr(glm5_model.fused_moe_layers, "FusedMoE", _RecordingExperts)
    monkeypatch.setattr(glm5_model, "get_tensor_model_parallel_world_size", lambda: 1)
    monkeypatch.setattr(glm5_model, "get_tensor_model_parallel_rank", lambda: 0)
    ep_group = SimpleNamespace(size=lambda: ep_size)
    monkeypatch.setattr(
        glm5_model,
        "get_ep_group",
        lambda: SimpleNamespace(device_group=ep_group, rank_in_group=ep_rank),
    )


def _parallel_config(**overrides):
    kwargs = dict(
        use_sequence_parallel_moe=False,
        enable_eplb=False,
        eplb_config=SimpleNamespace(num_redundant_experts=0),
    )
    kwargs.update(overrides)
    return SimpleNamespace(**kwargs)


class TestGlm5NextMoEConstruction:
    """Router, shared expert, EPLB arithmetic and FusedMoE wiring."""

    def _build(self, monkeypatch, config=None, parallel=None, **kwargs):
        _install_moe_doubles(monkeypatch)
        config = config or _text_config()
        parallel = parallel or _parallel_config()
        return glm5_model.Glm5NextMoE(
            config,
            parallel,
            prefix="model.layers.0.mlp",
            apply_routed_scale_to_output=kwargs.get(
                "apply_routed_scale_to_output", False
            ),
        )

    def test_eplb_arithmetic_and_gate_bias(self, monkeypatch):
        """Local physical experts are the full logical set on one EP rank."""
        config = _text_config()
        parallel = _parallel_config(
            enable_eplb=True,
            eplb_config=SimpleNamespace(num_redundant_experts=2),
        )
        moe = self._build(monkeypatch, config=config, parallel=parallel)

        assert moe.tp_size == 1 and moe.tp_rank == 0
        assert moe.ep_size == 1 and moe.ep_rank == 0
        assert moe.routed_scaling_factor == 2.5
        assert moe.n_routed_experts == 4
        assert moe.n_shared_experts == 1
        assert moe.n_logical_experts == 4
        assert moe.n_redundant_experts == 2
        assert moe.n_physical_experts == 6
        assert moe.n_local_physical_experts == 6
        assert moe.physical_expert_start == 0
        assert moe.physical_expert_end == 6

        assert moe.gate.args == (8, 4)
        assert moe.gate.kwargs["out_dtype"] is torch.float32
        bias = moe.gate.e_score_correction_bias
        assert isinstance(bias, nn.Parameter)
        assert bias.shape == (4,) and bias.dtype == torch.float32

        shared = moe.shared_experts
        assert isinstance(shared, _RecordingMLP)
        assert shared.kwargs["hidden_size"] == 8
        assert shared.kwargs["intermediate_size"] == 16
        assert shared.kwargs["reduce_results"] is False
        assert shared.kwargs["swiglu_limit"] == 10.0
        assert shared.kwargs["prefix"] == "model.layers.0.mlp.shared_experts"

        experts = moe.experts
        assert experts.kwargs["num_experts"] == 4
        assert experts.kwargs["top_k"] == 2
        assert experts.kwargs["hidden_size"] == 8
        assert experts.kwargs["intermediate_size"] == 16
        assert experts.kwargs["renormalize"] is True
        assert experts.kwargs["use_grouped_topk"] is True
        assert experts.kwargs["num_expert_group"] == 2
        assert experts.kwargs["topk_group"] == 1
        assert experts.kwargs["scoring_func"] == "sigmoid"
        assert experts.kwargs["routed_scaling_factor"] == 2.5
        assert experts.kwargs["apply_routed_scale_to_output"] is False
        assert experts.kwargs["e_score_correction_bias"] is bias
        assert experts.kwargs["enable_eplb"] is True
        assert experts.kwargs["num_redundant_experts"] == 2
        assert experts.kwargs["router_logits_dtype"] is torch.float32
        assert experts.kwargs["swiglu_limit"] == 10.0
        assert experts.kwargs["gate"] is moe.gate
        assert experts.kwargs["shared_experts"] is shared

    def test_non_noaux_router_has_no_bias(self, monkeypatch):
        """topk_method outside noaux_tc disables the correction bias."""
        config = _text_config(topk_method="greedy")
        moe = self._build(monkeypatch, config=config)
        assert moe.gate.e_score_correction_bias is None
        assert isinstance(moe.shared_experts, _RecordingMLP)
        assert moe.experts.kwargs["shared_experts"] is moe.shared_experts
        assert moe.experts.kwargs["e_score_correction_bias"] is None

    @pytest.mark.parametrize("n_shared_experts", [1, 2])
    def test_shared_expert_count_scales_intermediate_width(
        self, monkeypatch, n_shared_experts
    ):
        """The shared expert owns moe_intermediate_size * count columns."""
        config = _text_config(n_shared_experts=n_shared_experts)
        moe = self._build(monkeypatch, config=config)
        assert moe.n_shared_experts == n_shared_experts
        shared = moe.shared_experts
        assert isinstance(shared, _RecordingMLP)
        assert shared.kwargs["intermediate_size"] == 16 * n_shared_experts
        assert shared.kwargs["hidden_size"] == 8
        assert shared.kwargs["reduce_results"] is False
        assert moe.experts.kwargs["shared_experts"] is shared

    def test_apply_routed_scale_flag_reaches_fused_moe(self, monkeypatch):
        """The routed-scale flag is forwarded verbatim."""
        _install_moe_doubles(monkeypatch)
        moe = glm5_model.Glm5NextMoE(
            _text_config(),
            _parallel_config(),
            prefix="p",
            apply_routed_scale_to_output=True,
        )
        assert moe.experts.kwargs["apply_routed_scale_to_output"] is True

    def test_unsupported_activation_and_missing_clamp_raise(self, monkeypatch):
        """The reference SwiGLU contract rejects gelu and an absent clamp."""
        _install_moe_doubles(monkeypatch)
        with pytest.raises(ValueError, match="Only silu is supported"):
            glm5_model.Glm5NextMoE(
                _text_config(hidden_act="gelu"), _parallel_config(), prefix="p"
            )
        with pytest.raises(ValueError, match="finite swiglu_limit"):
            glm5_model.Glm5NextMoE(
                _text_config(swiglu_limit=None), _parallel_config(), prefix="p"
            )


class TestGlm5NextMoEForward:
    """Token reshaping, router selection and the sequence-parallel gather."""

    def _moe(self, monkeypatch, parallel=None, internal_router=False):
        _install_moe_doubles(monkeypatch)
        moe = glm5_model.Glm5NextMoE(
            _text_config(), parallel or _parallel_config(), prefix="p"
        )
        moe.experts.is_internal_router = internal_router
        return moe

    def test_internal_router_feeds_hidden_states(self, monkeypatch):
        """Internal routers receive the hidden states as router logits."""
        moe = self._moe(monkeypatch, internal_router=True)
        hidden = torch.arange(24, dtype=torch.float32).view(6, 4)
        out = moe(hidden, already_sequence_parallel=True)
        assert out.shape == (6, 4)
        assert torch.equal(out, hidden)
        _, router_logits = moe.experts.calls[-1]
        assert torch.equal(router_logits, hidden)

    def test_external_gate_supplies_router_logits(self, monkeypatch):
        """A non-internal router calls the gate first."""
        moe = self._moe(monkeypatch)
        hidden = torch.ones(3, 4)
        logits = torch.arange(12, dtype=torch.float32).view(3, 4)

        class _Gate(nn.Module):
            def __init__(self):
                super().__init__()
                self.calls = []

            def forward(self, x):
                self.calls.append(x)
                return logits, None

        moe.gate = _Gate()
        out = moe(hidden, already_sequence_parallel=True)
        assert torch.equal(out, hidden)
        assert torch.equal(moe.gate.calls[-1], hidden)
        assert torch.equal(moe.experts.calls[-1][1], logits)

    def test_sequence_parallel_chunk_and_gather_trim_padding(self, monkeypatch):
        """Chunked tokens are gathered back and trimmed to the token count."""
        calls = []
        gathered = torch.arange(28, dtype=torch.float32).view(7, 4)

        def fake_chunk(x):
            calls.append(("chunk", tuple(x.shape)))
            return x[:3]

        def fake_all_gather(x, dim):
            calls.append(("gather", tuple(x.shape), dim))
            return gathered

        monkeypatch.setattr(glm5_model, "sequence_parallel_chunk", fake_chunk)
        monkeypatch.setattr(
            glm5_model, "tensor_model_parallel_all_gather", fake_all_gather
        )
        moe = self._moe(
            monkeypatch, parallel=_parallel_config(use_sequence_parallel_moe=True)
        )
        hidden = torch.zeros(6, 4)
        out = moe(hidden)
        assert calls == [("chunk", (6, 4)), ("gather", (3, 4), 0)]
        # num_tokens = 6, so the seventh gathered row is padding.
        assert torch.equal(out, gathered[:6])
        assert out.shape == (6, 4)

    def test_already_sequence_parallel_skips_chunk_and_gather(self, monkeypatch):
        """Pre-chunked tokens are passed through untouched."""
        calls = []
        monkeypatch.setattr(
            glm5_model,
            "sequence_parallel_chunk",
            lambda x: calls.append("chunk") or x,
        )
        monkeypatch.setattr(
            glm5_model,
            "tensor_model_parallel_all_gather",
            lambda *a: calls.append("gather"),
        )
        moe = self._moe(
            monkeypatch,
            parallel=_parallel_config(use_sequence_parallel_moe=True),
            internal_router=True,
        )
        hidden = torch.ones(2, 4)
        out = moe(hidden, already_sequence_parallel=True)
        assert calls == []
        assert out.shape == (2, 4)


class TestGlm5NextLinearAttentionInit:
    """KDA bounds and the checkpoint A_log loader adapter."""

    @staticmethod
    def _build(monkeypatch, config, dim_first=True):
        recorded = {}

        def fake_base_init(self, config, vllm_config, prefix=""):
            del vllm_config, prefix
            self.kda_config_seen = config.linear_attn_config

            def old_loader(param, weight):
                recorded["param"] = param
                recorded["weight"] = weight
                return "loaded"

            self.A_log = SimpleNamespace(weight_loader=old_loader)

        monkeypatch.setattr(
            glm5_model.KimiGatedDeltaNetAttention, "__init__", fake_base_init
        )
        monkeypatch.setattr(glm5_model, "is_conv_state_dim_first", lambda: dim_first)
        attn = glm5_model.Glm5NextLinearAttention(config, None, prefix="p")
        return attn, recorded

    def test_default_and_custom_gate_bounds(self, monkeypatch):
        """gate_lower_bound defaults to -5.0 and honours the checkpoint."""
        config = Glm5NextTextConfig(
            linear_attn_config={
                "num_heads": 2,
                "head_dim": 4,
                "short_conv_kernel_size": 3,
            }
        )
        attn, _ = self._build(monkeypatch, config)
        assert attn.kda_lower_bound == -5.0
        assert attn._conv_state_dim_first is True

        config = Glm5NextTextConfig(
            linear_attn_config={
                "num_heads": 2,
                "head_dim": 4,
                "short_conv_kernel_size": 3,
                "gate_lower_bound": -3.5,
            }
        )
        attn, _ = self._build(monkeypatch, config, dim_first=False)
        assert attn.kda_lower_bound == -3.5
        assert attn._conv_state_dim_first is False

    def test_one_dimensional_a_log_is_reshaped_before_delegating(self, monkeypatch):
        """Checkpoint [num_heads] logs become [1, 1, heads, 1] runtime params."""
        config = Glm5NextTextConfig(
            linear_attn_config={
                "num_heads": 2,
                "head_dim": 4,
                "short_conv_kernel_size": 3,
            }
        )
        attn, recorded = self._build(monkeypatch, config)
        weight = torch.arange(3, dtype=torch.float32)
        assert attn.A_log.weight_loader("param", weight) == "loaded"
        reshaped = recorded["weight"]
        assert reshaped.shape == (1, 1, 3, 1)
        assert torch.equal(reshaped.flatten(), weight)

    def test_already_shaped_a_log_passes_through(self, monkeypatch):
        """Runtime-shaped tensors are not reshaped again."""
        config = Glm5NextTextConfig(
            linear_attn_config={
                "num_heads": 2,
                "head_dim": 4,
                "short_conv_kernel_size": 3,
            }
        )
        attn, recorded = self._build(monkeypatch, config)
        weight = torch.arange(6, dtype=torch.float32).view(1, 1, 2, 3)
        attn.A_log.weight_loader("param", weight)
        assert recorded["weight"] is weight


class TestGlm5NextDecoderLayerConstruction:
    """Layer assembly for the KDA/MLA and dense/sparse/mHC combinations."""

    @staticmethod
    def _config(**overrides):
        values = dict(
            num_hidden_layers=2,
            rms_norm_eps=1e-6,
            mhc=True,
            is_kda_layer=lambda idx: idx == 0,
            hidden_size=8,
            num_attention_heads=4,
            qk_nope_head_dim=4,
            qk_rope_head_dim=0,
            v_head_dim=4,
            q_lora_rank=4,
            kv_lora_rank=6,
            max_position_embeddings=64,
            mlp_layer_types=["sparse", "dense"],
            hidden_act="silu",
            intermediate_size=16,
            swiglu_limit=10.0,
            mhc_num_residual_streams=3,
            hc_eps=1e-6,
            mhc_sinkhorn_iterations=5,
            mhc_post_mult_value=2.0,
        )
        values.update(overrides)
        return SimpleNamespace(**values)

    @staticmethod
    def _vllm_config(config):
        return SimpleNamespace(
            model_config=SimpleNamespace(hf_text_config=config),
            cache_config=object(),
            quant_config=None,
            parallel_config=SimpleNamespace(),
        )

    def _build(self, monkeypatch, config, layer_idx=0):
        monkeypatch.setattr(glm5_model, "Glm5NextLinearAttention", _RecordingModule)
        monkeypatch.setattr(glm5_model, "Glm5NextMLAAttention", _RecordingModule)
        monkeypatch.setattr(glm5_model, "Glm5NextMoE", _RecordingModule)
        monkeypatch.setattr(glm5_model, "Glm5NextMLP", _RecordingModule)
        monkeypatch.setattr(glm5_model, "RMSNorm", _RecordingModule)
        monkeypatch.setattr(glm5_model, "MHCPreOp", _RecordingModule)
        monkeypatch.setattr(glm5_model, "MHCPostOp", _RecordingModule)
        monkeypatch.setattr(glm5_model, "MHCFusedPostPreOp", _RecordingModule)
        buffer = torch.empty(4, 8, dtype=torch.int32)
        return glm5_model.Glm5NextDecoderLayer(
            self._vllm_config(config),
            layer_idx,
            f"model.layers.{layer_idx}",
            buffer,
        )

    def test_kda_layer_with_sparse_mlp_registers_mhc_parameters(self, monkeypatch):
        """KDA attention, sparse MLP and the mHC parameter shapes line up."""
        config = self._config()
        layer = self._build(monkeypatch, config, layer_idx=0)

        assert layer.layer_idx == 0
        assert layer.num_hidden_layers == 2
        assert layer.rms_norm_eps == 1e-6
        assert layer.mhc is True
        assert isinstance(layer.self_attn, _RecordingModule)
        assert layer.self_attn.args[0] is config
        assert layer.self_attn.kwargs["prefix"] == "model.layers.0.self_attn"
        assert isinstance(layer.mlp, _RecordingModule)
        assert layer.mlp.kwargs["prefix"] == "model.layers.0.mlp"

        n = 3
        d_model = n * 8
        mix_hc = (2 + n) * n
        assert layer.n == n
        assert layer.hc_eps == 1e-6
        assert layer.mhc_sinkhorn_iterations == 5
        assert layer.mhc_post_mult_value == 2.0
        expected = {
            "hc_attn_fn": (mix_hc, d_model),
            "hc_attn_base": (mix_hc,),
            "hc_attn_scale": (3,),
            "hc_ffn_fn": (mix_hc, d_model),
            "hc_ffn_base": (mix_hc,),
            "hc_ffn_scale": (3,),
        }
        for name, shape in expected.items():
            param = getattr(layer, name)
            assert isinstance(param, nn.Parameter)
            assert param.shape == shape
            assert param.dtype == torch.float32
        assert layer.hc_attn_fn_broadcast is None
        assert isinstance(layer.mhc_pre_op, _RecordingModule)
        assert isinstance(layer.mhc_post_op, _RecordingModule)
        assert isinstance(layer.mhc_fused_post_pre_op, _RecordingModule)

    def test_mla_layer_with_dense_mlp_forwards_projection_config(self, monkeypatch):
        """MLA layers receive the full projection geometry from the config."""
        config = self._config()
        layer = self._build(monkeypatch, config, layer_idx=1)

        assert isinstance(layer.self_attn, _RecordingModule)
        kwargs = layer.self_attn.kwargs
        assert kwargs["hidden_size"] == 8
        assert kwargs["num_heads"] == 4
        assert kwargs["qk_nope_head_dim"] == 4
        assert kwargs["qk_rope_head_dim"] == 0
        assert kwargs["v_head_dim"] == 4
        assert kwargs["q_lora_rank"] == 4
        assert kwargs["kv_lora_rank"] == 6
        assert kwargs["max_position_embeddings"] == 64
        assert kwargs["quant_config"] is None
        assert kwargs["prefix"] == "model.layers.1.self_attn"
        assert kwargs["topk_indices_buffer"].shape == (4, 8)
        assert isinstance(layer.mlp, _RecordingModule)
        assert layer.mlp.kwargs["intermediate_size"] == 16
        assert layer.mlp.kwargs["swiglu_limit"] == 10.0

    def test_mhc_disabled_omits_stream_parameters(self, monkeypatch):
        """A non-mHC layer has no stream parameters or mHC operators."""
        config = self._config(mhc=False)
        layer = self._build(monkeypatch, config)
        assert layer.mhc is False
        for name in (
            "hc_attn_fn",
            "hc_attn_base",
            "hc_attn_scale",
            "hc_ffn_fn",
            "hc_ffn_base",
            "hc_ffn_scale",
            "mhc_pre_op",
        ):
            assert not hasattr(layer, name)


class _NormChain:
    """Residual-aware RMSNorm double: (x + 1, residual + x)."""

    def __init__(self):
        self.calls = []

    def __call__(self, x, residual=None):
        self.calls.append((x, residual))
        if residual is None:
            return x + 1
        return x - 1, residual + x


class _PlainNorm:
    """RMSNorm double adding a fixed offset."""

    def __init__(self, offset=1):
        self.offset = offset

    def __call__(self, x):
        return x + self.offset


class _MLPDouble(nn.Module):
    def __init__(self):
        super().__init__()
        self.calls = []

    def forward(self, x):
        self.calls.append(x)
        return x + 10


def _decoder_forward_layer(**overrides):
    layer = glm5_model.Glm5NextDecoderLayer.__new__(glm5_model.Glm5NextDecoderLayer)
    nn.Module.__init__(layer)
    layer.layer_idx = overrides.get("layer_idx", 1)
    layer.num_hidden_layers = overrides.get("num_hidden_layers", 2)
    layer.mhc = overrides.get("mhc", True)
    layer.rms_norm_eps = 1e-6
    layer.n = 2
    layer.hc_eps = 1e-6
    layer.mhc_sinkhorn_iterations = 4
    layer.mhc_post_mult_value = 2.0
    layer.hc_attn_fn = torch.zeros(8, 4)
    layer.hc_attn_scale = torch.zeros(3)
    layer.hc_attn_base = torch.zeros(8)
    layer.hc_ffn_fn = torch.ones(8, 4)
    layer.hc_ffn_scale = torch.ones(3)
    layer.hc_ffn_base = torch.ones(8)
    layer.hc_attn_fn_broadcast = overrides.get("broadcast")
    layer.input_layernorm = _PlainNorm(1)
    layer.post_attention_layernorm = _NormChain()
    layer.mlp = _MLPDouble()
    layer._attention = lambda positions, hidden: hidden * 2
    events = []
    layer.events = events

    class _Pre:
        def __call__(self, *args, **kwargs):
            events.append(("hc_pre", args, kwargs))
            return "post", "comb", args[0] * 3

    class _Broadcast:
        def __call__(self, residual, norm):
            events.append(("broadcast", residual, norm))
            return "bres", "bpost", "bcomb", residual * 4

    class _Fused:
        def __call__(self, *args):
            events.append(("fused", args))
            x, residual, post, comb, fn = args[:5]
            return residual, post, comb, x * 5

    class _Post:
        def __call__(self, x, residual, post, comb):
            events.append(("post", x, residual, post, comb))
            return x + residual

    layer._hc_pre = _Pre()
    layer._hc_pre_broadcast = _Broadcast()
    layer._hc_fused_post_pre = _Fused()
    layer.mhc_post_op = _Post()
    return layer


class TestGlm5NextDecoderLayerForward:
    """Routing of the plain, pre/post and first-layer broadcast mHC paths."""

    def test_non_mhc_path_uses_norm_attention_norm_mlp(self):
        """Non-mHC layers reduce to a standard pre-norm residual block."""
        layer = _decoder_forward_layer()
        layer.mhc = False
        hidden = torch.ones(2, 4)
        positions = torch.zeros(2, dtype=torch.long)

        out, residual, post, comb = glm5_model.Glm5NextDecoderLayer.forward(
            layer, positions, hidden, None, None, None
        )
        # input norm: +1 -> attention: x2 = (h+1)*2
        attn_in = (hidden + 1) * 2
        # post norm returns (x-1, residual+x) where x is the attention output.
        normed = attn_in - 1
        expected_residual = hidden + attn_in
        expected = expected_residual + (normed + 10)
        assert torch.equal(out, expected)
        assert torch.equal(residual, expected_residual)
        assert post is None and comb is None
        assert layer.post_attention_layernorm.calls[0][1] is hidden

    def test_first_layer_pre_expands_residual_streams(self):
        """Layer 0 broadcasts the hidden state into mHC streams before pre."""
        layer = _decoder_forward_layer(layer_idx=0, broadcast=None)
        hidden = torch.arange(8, dtype=torch.float32).view(2, 4)
        positions = torch.zeros(2, dtype=torch.long)

        out, residual, post, comb = glm5_model.Glm5NextDecoderLayer.forward(
            layer, positions, hidden, None, None, None
        )
        pre_event = layer.events[0]
        assert pre_event[0] == "hc_pre"
        expanded = pre_event[1][0]
        assert expanded.shape == (2, 2, 4)
        # Every stream starts as a copy of the hidden state.
        assert torch.equal(expanded[:, 0], hidden)
        assert torch.equal(expanded[:, 1], hidden)
        assert torch.equal(residual, expanded)
        # pre -> *3, attention -> *2, fused post-pre -> *5, MLP -> +10.
        assert torch.equal(out, expanded * 30 + 10)
        assert post == "post" and comb == "comb"

    def test_later_layer_keeps_residual_unexpanded(self):
        """Layers after the first reuse the existing stream tensor."""
        layer = _decoder_forward_layer(layer_idx=1, num_hidden_layers=3)
        streams = torch.arange(16, dtype=torch.float32).view(2, 2, 4)
        positions = torch.zeros(2, dtype=torch.long)
        out, residual, _, _ = glm5_model.Glm5NextDecoderLayer.forward(
            layer, positions, streams, None, None, None
        )
        pre_event = layer.events[0]
        assert pre_event[1][0] is streams
        assert torch.equal(residual, streams)
        assert out.shape == (2, 2, 4)

    def test_broadcast_fast_path_is_selected_for_first_layer(self, monkeypatch):
        """A prepared broadcast weight routes layer 0 through the fused pre."""
        monkeypatch.setattr(glm5_model, "mhc_pre_broadcast_tilelang", object())
        layer = _decoder_forward_layer(layer_idx=0, broadcast=torch.zeros(8, 4))
        hidden = torch.ones(2, 4)
        positions = torch.zeros(2, dtype=torch.long)
        out, residual, _, _ = glm5_model.Glm5NextDecoderLayer.forward(
            layer, positions, hidden, None, None, None
        )
        assert layer.events[0][0] == "broadcast"
        assert layer.events[0][1] is hidden
        assert residual == "bres"
        # broadcast -> *4, attention -> *2, fused post-pre -> *5, MLP -> +10.
        assert torch.equal(out, hidden * 40 + 10)

    def test_post_state_routes_through_fused_post_pre(self):
        """A carried post/comb pair enters the fused post-pre helper."""
        layer = _decoder_forward_layer(layer_idx=1, num_hidden_layers=3)
        hidden = torch.ones(2, 2, 4)
        residual = torch.full((2, 2, 4), 2.0)
        post = "carried_post"
        comb = "carried_comb"
        positions = torch.zeros(2, dtype=torch.long)
        out, new_residual, new_post, new_comb = glm5_model.Glm5NextDecoderLayer.forward(
            layer, positions, hidden, residual, post, comb
        )
        first_fused = layer.events[0]
        assert first_fused[0] == "fused"
        # The attention-side fused call uses hc_attn_fn.
        assert first_fused[1][4] is layer.hc_attn_fn
        assert first_fused[1][2] == post and first_fused[1][3] == comb
        # Then attention, a second fused call with hc_ffn_fn, and the MLP.
        second_fused = layer.events[1]
        assert second_fused[0] == "fused"
        assert second_fused[1][4] is layer.hc_ffn_fn
        assert torch.equal(new_residual, residual)
        # hidden * 5 (pre) * 2 (attention) * 5 (post-pre) + 10 (MLP).
        assert torch.equal(out, hidden * 50 + 10)

    def test_last_layer_contracts_streams_with_mhc_post(self):
        """The final layer applies mhc_post and averages the streams."""
        layer = _decoder_forward_layer(layer_idx=1, num_hidden_layers=2)
        hidden = torch.ones(2, 2, 4)
        positions = torch.zeros(2, dtype=torch.long)
        out, residual, post, comb = glm5_model.Glm5NextDecoderLayer.forward(
            layer, positions, hidden, None, None, None
        )
        post_event = layer.events[-1]
        assert post_event[0] == "post"
        expected = post_event[1] + post_event[2]
        assert torch.equal(out, expected.mean(dim=1))
        assert residual is None and post is None and comb is None


class _TupleLinear(nn.Module):
    """Linear returning the ``(output, bias)`` pair vLLM layers expose."""

    def __init__(self, output_size, input_size, seed):
        super().__init__()
        generator = torch.Generator().manual_seed(seed)
        self.weight = nn.Parameter(
            torch.randn(output_size, input_size, generator=generator)
        )

    def forward(self, x):
        return torch.nn.functional.linear(x, self.weight), None


class _RecordingIndexerOp:
    def __init__(self):
        self.calls = []

    def __call__(self, hidden_states, q, k, weights, **kwargs):
        self.calls.append(
            {
                "hidden": hidden_states,
                "q": q,
                "k": k,
                "weights": weights,
                **kwargs,
            }
        )
        return "logits"


def _indexer_double(n_head, head_dim=128, hidden_size=16, seed=0):
    indexer = SimpleNamespace(
        head_dim=head_dim,
        quant_block_size=128,
        scale_fmt="ue8m0",
        n_head=n_head,
        softmax_scale=head_dim**-0.5,
        wq_b=_TupleLinear(n_head * head_dim, 8, seed + 1),
        wk_weights_proj=_TupleLinear(head_dim + n_head, hidden_size, seed + 2),
        k_norm=lambda x: x / x.norm(dim=-1, keepdim=True),
        indexer_op=_RecordingIndexerOp(),
        index_kpool_compress_ape=torch.zeros(4, head_dim),
        index_kpool=4,
    )
    gate = torch.randn(
        head_dim, hidden_size, generator=torch.Generator().manual_seed(seed + 3)
    )
    indexer.index_kpool_compress_gate = gate
    return indexer


class TestIndexerForwardNope:
    """Zero-RoPE indexer projection, weight caching and head padding."""

    def test_matches_independent_projection_reference(self, monkeypatch):
        """q/k/weights/gate_score come from the documented projection math."""
        captured = []

        def fake_quant(q):
            captured.append(q.clone())
            return q.clone(), q.new_ones((q.shape[0], 1), dtype=torch.float32)

        monkeypatch.setattr(glm5_model, "fwht128_quant_fp8", fake_quant)
        indexer = _indexer_double(n_head=2)
        hidden = torch.arange(32, dtype=torch.float32).view(2, 16)
        qr = torch.arange(16, dtype=torch.float32).view(2, 8)

        out = glm5_model._indexer_forward_nope(
            indexer,
            hidden,
            qr,
            positions=torch.zeros(2, dtype=torch.long),
            rotary_emb=None,
        )
        assert out == "logits"
        call = indexer.indexer_op.calls[-1]

        q_ref = torch.nn.functional.linear(qr, indexer.wq_b.weight).view(2, 2, 128)
        assert torch.equal(captured[0], q_ref.reshape(-1, 128))
        kw_ref = torch.nn.functional.linear(hidden, indexer.wk_weights_proj.weight)
        k_ref = kw_ref[:, :128] / kw_ref[:, :128].norm(dim=-1, keepdim=True)
        assert torch.equal(call["k"], k_ref)
        # Weights use the second half of the fused projection in FP32 and the
        # unit query scale produced above.
        wp = indexer.wk_weights_proj.weight.data[128:, :].t().contiguous().float()
        weights_ref = (
            (hidden.float() @ wp) * 1.0 * indexer.softmax_scale * indexer.n_head**-0.5
        )
        assert torch.equal(call["weights"][:, :2], weights_ref)
        assert torch.equal(
            call["gate_score"],
            torch.nn.functional.linear(hidden, indexer.index_kpool_compress_gate),
        )
        assert call["compress_ape"] is indexer.index_kpool_compress_ape
        assert call["index_kpool"] == 4

    def test_few_heads_are_zero_padded_to_32(self, monkeypatch):
        """Sub-32 head counts are padded with zeros for the sparse kernel."""
        monkeypatch.setattr(
            glm5_model,
            "fwht128_quant_fp8",
            lambda q: (
                q.clone(),
                q.new_ones((q.shape[0], 1), dtype=torch.float32),
            ),
        )
        indexer = _indexer_double(n_head=2)
        hidden = torch.ones(2, 16)
        qr = torch.ones(2, 8)
        glm5_model._indexer_forward_nope(
            indexer, hidden, qr, torch.zeros(2, dtype=torch.long), None
        )
        call = indexer.indexer_op.calls[-1]
        assert call["q"].shape == (2, 32, 128)
        assert call["weights"].shape == (2, 32)
        assert torch.count_nonzero(call["q"][:, 2:]) == 0
        assert torch.count_nonzero(call["weights"][:, 2:]) == 0

    def test_full_head_count_is_not_padded(self, monkeypatch):
        """n_head == 32 bypasses the padding branch."""
        monkeypatch.setattr(
            glm5_model,
            "fwht128_quant_fp8",
            lambda q: (
                q.clone(),
                q.new_ones((q.shape[0], 1), dtype=torch.float32),
            ),
        )
        indexer = _indexer_double(n_head=32)
        hidden = torch.ones(2, 16)
        qr = torch.ones(2, 8)
        glm5_model._indexer_forward_nope(
            indexer, hidden, qr, torch.zeros(2, dtype=torch.long), None
        )
        call = indexer.indexer_op.calls[-1]
        assert call["q"].shape == (2, 32, 128)
        assert call["weights"].shape == (2, 32)

    def test_weights_projection_is_cached_after_first_call(self, monkeypatch):
        """The transposed FP32 weight cache is built once and reused."""
        monkeypatch.setattr(
            glm5_model,
            "fwht128_quant_fp8",
            lambda q: (
                q.clone(),
                q.new_ones((q.shape[0], 1), dtype=torch.float32),
            ),
        )
        indexer = _indexer_double(n_head=2)
        hidden = torch.ones(2, 16)
        qr = torch.ones(2, 8)
        for _ in range(2):
            glm5_model._indexer_forward_nope(
                indexer, hidden, qr, torch.zeros(2, dtype=torch.long), None
            )
        cache = indexer._wp_fp32
        assert cache.shape == (16, 2) and cache.dtype == torch.float32
        with torch.no_grad():
            indexer.wk_weights_proj.weight.data.zero_()
        glm5_model._indexer_forward_nope(
            indexer, hidden, qr, torch.zeros(2, dtype=torch.long), None
        )
        assert indexer._wp_fp32 is cache
        # The cached weights still drive the computation; q_scale is unit, so
        # the remaining factors are applied in the source's own order.
        expected = torch.nn.functional.linear(hidden, cache.t())
        expected = expected * indexer.softmax_scale * indexer.n_head**-0.5
        assert torch.equal(indexer.indexer_op.calls[-1]["weights"][:, :2], expected)

    @pytest.mark.parametrize(
        "overrides",
        [
            {"head_dim": 64},
            {"quant_block_size": 64},
            {"scale_fmt": "e4m3"},
        ],
    )
    def test_layout_guards_reject_unsupported_indexers(self, monkeypatch, overrides):
        """The zero-RoPE path pins the FP8 quantizer layout it replaces."""
        monkeypatch.setattr(
            glm5_model,
            "fwht128_quant_fp8",
            lambda q: (
                q.clone(),
                q.new_ones((q.shape[0], 1), dtype=torch.float32),
            ),
        )
        indexer = _indexer_double(n_head=2)
        for key, value in overrides.items():
            setattr(indexer, key, value)
        with pytest.raises(AssertionError):
            glm5_model._indexer_forward_nope(
                indexer,
                torch.ones(2, 16),
                torch.ones(2, 8),
                torch.zeros(2, dtype=torch.long),
                None,
            )


class _FakeGDNAttentionMetadata:
    """Named type replacing vLLM's metadata dataclass in the isinstance guard."""


class _KpoolRecorder:
    def __init__(self):
        self.calls = []

    def _kpool(self, name, *args):
        self.calls.append((name, args))
        if name == "gather_state_rows":
            return "initial_state"
        if name == "copy_to":
            source, destination = args
            destination.copy_(source)
        return None


def _gate_attn_state():
    """A KDA layer double with the cache/conv weights the gate path reads."""
    conv_state = torch.zeros(2, 6, 2)  # (entries, 3 * dim, width - 1)
    recurrent_state = torch.zeros(4, 2, 2)
    attn = SimpleNamespace(
        prefix="model.layers.0.self_attn",
        kv_cache=(conv_state, recurrent_state),
        _conv_state_dim_first=True,
        head_dim=2,
        A_log="A_log",
        dt_bias="dt_bias",
        kda_lower_bound=-4.0,
    )
    for name in ("q_conv1d", "k_conv1d", "v_conv1d"):
        setattr(
            attn,
            name,
            SimpleNamespace(weight=torch.zeros(2, 1, 3), bias=torch.zeros(2)),
        )
    return attn, conv_state, recurrent_state


def _metadata(**overrides):
    meta = _FakeGDNAttentionMetadata()
    values = dict(
        has_initial_state=torch.tensor([True, False]),
        non_spec_query_start_loc=torch.tensor([0, 2, 4]),
        non_spec_state_indices_tensor=torch.tensor([0, 1]),
        num_actual_tokens=4,
        num_prefills=1,
        num_decodes=0,
    )
    values.update(overrides)
    for key, value in values.items():
        setattr(meta, key, value)
    return meta


class TestForwardReferenceSafeGate:
    """Prefill/decode routing of the plugin's KDA gate replacement."""

    def _install(self, monkeypatch, meta, attn, conv_ops_calls):
        recorder = _KpoolRecorder()
        monkeypatch.setattr(
            glm5_model, "GDNAttentionMetadata", _FakeGDNAttentionMetadata
        )
        monkeypatch.setattr(
            glm5_model,
            "get_forward_context",
            lambda: SimpleNamespace(attn_metadata={attn.prefix: meta}),
        )
        monkeypatch.setattr(glm5_model, "INDEXER_BACKEND", recorder)

        def conv_fn(x, weight, bias, **kwargs):
            conv_ops_calls.append(("fn", x, weight, bias, kwargs))
            return x

        def conv_update(x, conv_state, weight, bias, **kwargs):
            conv_ops_calls.append(("update", x, conv_state, weight, bias, kwargs))
            return x

        monkeypatch.setattr(glm5_model.conv_ops, "causal_conv1d_fn", conv_fn)
        monkeypatch.setattr(glm5_model.conv_ops, "causal_conv1d_update", conv_update)
        return recorder

    def test_prefill_gathers_state_runs_chunk_and_scatters(self, monkeypatch):
        """Prefill zeroes new states, gathers, runs chunk KDA and stores."""
        attn, conv_state, recurrent_state = _gate_attn_state()
        meta = _metadata()
        conv_calls = []
        recorder = self._install(monkeypatch, meta, attn, conv_calls)
        chunk_calls = []

        def fake_chunk(**kwargs):
            chunk_calls.append(kwargs)
            return torch.ones(1, 4, 2, 2), torch.full((2, 2, 2), 7.0)

        monkeypatch.setattr(glm5_model, "chunk_kda_with_safe_gate", fake_chunk)
        q = torch.arange(32, dtype=torch.float32).view(8, 4)
        k = q + 100
        v = q + 200
        g1 = torch.arange(32, dtype=torch.float32).view(1, 8, 2, 2)
        beta = torch.zeros(1, 8, 2)
        out = torch.zeros(1, 8, 2, 2)

        glm5_model.Glm5NextLinearAttention._forward_reference_safe_gate(
            attn, q, k, v, g1, beta, out
        )

        # q/k/v are truncated to num_actual_tokens and transposed for the
        # conv kernel; the original width dimension is moved to the front.
        for call in conv_calls:
            assert call[0] == "fn"
            assert call[1].shape == (4, 4)
            assert call[4]["activation"] == "silu"
            assert call[4]["cache_indices"] is meta.non_spec_state_indices_tensor
            assert call[4]["query_start_loc"] is meta.non_spec_query_start_loc
            assert call[4]["metadata"] is meta
            assert call[2].shape == (2, 3)
        assert [c[4]["conv_states"].shape for c in conv_calls] == [
            (2, 2, 2),
            (2, 2, 2),
            (2, 2, 2),
        ]

        chunk = chunk_calls[0]
        assert chunk["q"].shape == (1, 4, 2, 2)
        assert chunk["raw_g"].shape == (1, 4, 2, 2)
        assert chunk["beta"].shape == (1, 4, 2)
        assert chunk["initial_state"] == "initial_state"
        assert chunk["cu_seqlens"] is meta.non_spec_query_start_loc
        assert chunk["lower_bound"] == -4.0
        assert chunk["use_qk_l2norm_in_kernel"] is True
        assert chunk["output_final_state"] is True

        names = [name for name, _ in recorder.calls]
        assert names == [
            "zero_state_rows",
            "gather_state_rows",
            "scatter_state_rows",
            "copy_to",
        ]
        zero_rows = recorder.calls[0][1][1]
        assert zero_rows.tolist() == [1]
        scatter = recorder.calls[2][1]
        assert torch.equal(scatter[2], torch.full((2, 2, 2), 7.0))
        # The chunk output was copied into the caller's attention buffer.
        assert torch.equal(out[0, :4], torch.ones(4, 2, 2))

    def test_decode_runs_gate_and_recurrent_kda(self, monkeypatch):
        """Decode applies the bounded gate then the recurrent kernel."""
        attn, conv_state, recurrent_state = _gate_attn_state()
        meta = _metadata(
            has_initial_state=None,
            non_spec_state_indices_tensor=torch.tensor([0, 1, 2, 3]),
            num_prefills=0,
            num_decodes=2,
        )
        conv_calls = []
        self._install(monkeypatch, meta, attn, conv_calls)
        gate_calls = []
        recurrent_calls = []

        def fake_gate(g, a_log, head_dim, **kwargs):
            gate_calls.append((g, a_log, head_dim, kwargs))
            return g

        def fake_recurrent(**kwargs):
            recurrent_calls.append(kwargs)
            return torch.ones(1, 4, 2, 2), "last_state"

        monkeypatch.setattr(glm5_model, "fused_safe_kda_gate", fake_gate)
        monkeypatch.setattr(glm5_model, "fused_recurrent_kda", fake_recurrent)
        q = torch.arange(32, dtype=torch.float32).view(8, 4)
        g1 = torch.arange(32, dtype=torch.float32).view(1, 8, 2, 2)
        out = torch.zeros(1, 8, 2, 2)

        glm5_model.Glm5NextLinearAttention._forward_reference_safe_gate(
            attn, q, q, q, g1, torch.zeros(1, 8, 2), out
        )

        assert [call[0] for call in conv_calls] == ["update"] * 3
        for call in conv_calls:
            assert call[5]["validate_data"] is True
            assert call[5]["conv_state_indices"].tolist() == [0, 1, 2, 3]
        gate = gate_calls[0]
        assert gate[0].shape == (4, 4)
        assert torch.equal(gate[0], g1[:, :4].reshape(4, 4))
        assert gate[1] == "A_log" and gate[2] == 2
        assert gate[3]["g_bias"] == "dt_bias"
        assert gate[3]["lower_bound"] == -4.0

        recurrent = recurrent_calls[0]
        assert recurrent["g"].shape == (1, 4, 4)
        assert recurrent["initial_state"] is recurrent_state
        assert recurrent["cu_seqlens"].tolist() == [0, 2, 4]
        assert recurrent["ssm_state_indices"] is meta.non_spec_state_indices_tensor
        assert torch.equal(out[0, :4], torch.ones(4, 2, 2))

    def test_dim_second_conv_state_is_transposed(self, monkeypatch):
        """A sequence-first conv cache is transposed before chunking."""
        attn, conv_state, _ = _gate_attn_state()
        attn._conv_state_dim_first = False
        attn.kv_cache = (torch.zeros(2, 2, 6), torch.zeros(4, 2, 2))
        meta = _metadata()
        conv_calls = []
        self._install(monkeypatch, meta, attn, conv_calls)
        monkeypatch.setattr(
            glm5_model,
            "chunk_kda_with_safe_gate",
            lambda **kwargs: (torch.ones(1, 4, 2, 2), torch.zeros(2, 2, 2)),
        )
        q = torch.zeros(8, 4)
        glm5_model.Glm5NextLinearAttention._forward_reference_safe_gate(
            attn,
            q,
            q,
            q,
            torch.zeros(1, 8, 2, 2),
            torch.zeros(1, 8, 2),
            torch.zeros(1, 8, 2, 2),
        )
        expected = attn.kv_cache[0].transpose(-1, -2).chunk(3, dim=-2)[0]
        assert torch.equal(conv_calls[0][4]["conv_states"], expected)

    def test_missing_metadata_is_a_no_op(self, monkeypatch):
        """V1 profile runs without attention metadata return immediately."""
        attn, _, _ = _gate_attn_state()
        calls = []
        monkeypatch.setattr(
            glm5_model,
            "get_forward_context",
            lambda: SimpleNamespace(attn_metadata=None),
        )
        monkeypatch.setattr(
            glm5_model.conv_ops,
            "causal_conv1d_fn",
            lambda *a, **k: calls.append("conv"),
        )
        glm5_model.Glm5NextLinearAttention._forward_reference_safe_gate(
            attn,
            torch.zeros(1, 4),
            torch.zeros(1, 4),
            torch.zeros(1, 4),
            torch.zeros(1, 1, 2, 2),
            torch.zeros(1, 1, 2),
            torch.zeros(1, 1, 2, 2),
        )
        assert calls == []

    def test_alias_points_at_the_reference_implementation(self):
        """The class ``_forward`` hook is the plugin-owned gate path."""
        assert (
            glm5_model.Glm5NextLinearAttention._forward
            is glm5_model.Glm5NextLinearAttention._forward_reference_safe_gate
        )
