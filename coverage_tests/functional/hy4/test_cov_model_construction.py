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

"""Construction and wiring contracts of the HY4 backbone.

``HYV4DenseMLP`` and ``HYV4DecoderLayer`` are built against a single-rank TP
group so the vLLM tensor-parallel linear layers exercise their real sharding
metadata; their numerics are compared with plain-PyTorch references.  The
model/causal-LM wiring is driven with recording stand-ins at the external
builder boundaries (``HYV4Model``/``ParallelLMHead``/``HYV4Attention``), with
the constructor arguments asserted independently.  Expert metadata methods are
checked on minimal host objects carrying hand-written expert counts.
"""

from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F
from torch import nn

from ..._distributed import single_rank_gloo
from . import _import_shim  # noqa: F401  (idempotent; no-op on CI)
from ._construction_helpers import set_current_vllm_config
from vllm_fl.models.hy4 import model as hy_model
from vllm_fl.models.hy4.config import HYV4Config
from vllm_fl.models.hy4.model import (
    HYV4DecoderLayer,
    HYV4DenseMLP,
    HYV4ForCausalLM,
    HYV4HyperConnection,
    HYV4HyperLayer,
    HYV4MixtureOfExperts,
    HYV4MoE,
)

RTOL = 1e-4
ATOL = 1e-4


def _runtime_config():
    """Config used only to let vLLM CustomOps resolve their implementation."""
    return type(
        "VC",
        (),
        {
            "compilation_config": type(
                "CC",
                (),
                {
                    "mode": "EAGER",
                    "backend": "eager",
                    "static_forward_context": {},
                    "custom_ops": ["none"],
                    "enabled_custom_ops": set(),
                    "disabled_custom_ops": set(),
                },
            )(),
            "kernel_config": type(
                "KC",
                (),
                {"linear_backend": "auto", "enable_jit_warmup": False},
            )(),
            "model_config": type("MC", (), {"dtype": torch.bfloat16})(),
            "quant_config": None,
        },
    )()


@pytest.fixture()
def single_rank_tp():
    """Start a fresh one-rank gloo TP group; restore the previous one."""
    with single_rank_gloo("tp"):
        yield


def _tiny_config(**overrides):
    """A dimension-shrunk HYV4Config that still validates."""
    overrides.setdefault("vocab_size", 16)
    overrides.setdefault("hidden_size", 8)
    overrides.setdefault("intermediate_size", 16)
    overrides.setdefault("num_hidden_layers", 2)
    overrides.setdefault("qk_nope_head_dim", 4)
    overrides.setdefault("qk_rope_head_dim", 4)
    overrides.setdefault("qk_head_dim", 8)
    overrides.setdefault("v_head_dim", 4)
    overrides.setdefault("q_lora_rank", 4)
    overrides.setdefault("kv_lora_rank", 6)
    overrides.setdefault("n_routed_experts", 2)
    overrides.setdefault("moe_intermediate_size", 16)
    overrides.setdefault("hc_mult", 2)
    overrides.setdefault("indexer_types", ["full", "full"])
    return HYV4Config(**overrides)


def _rms_norm_reference(x, weight, eps):
    """Independent RMSNorm: x * rsqrt(mean(x^2) + eps) * weight."""
    variance = x.float().square().mean(dim=-1, keepdim=True)
    return (x.float() * torch.rsqrt(variance + eps) * weight.float()).to(x.dtype)


def _hyper_read_reference(module, streams):
    """Sigmoid-gated collapse written out independently of the module."""
    flat = streams.flatten(1).float()
    norm = torch.rsqrt(flat.square().mean(dim=-1, keepdim=True) + module.normalize_eps)
    gates = F.linear(flat, module.hc_fn) * norm
    read, write = gates.split(module.num_streams, dim=-1)
    read = torch.sigmoid(read * module.hc_scale[0] + module.hc_base[: module.num_streams])
    write = module.magnitude * torch.sigmoid(
        write * module.hc_scale[1] + module.hc_base[module.num_streams :]
    )
    collapsed = (
        (read + module.hc_eps).unsqueeze(-1) * streams.float()
    ).sum(dim=1)
    return collapsed, write + module.hc_eps


def _hyper_write_reference(streams, delta, write):
    return (
        streams.float() + write.float().unsqueeze(-1) * delta.float().unsqueeze(1)
    )


class _AttentionStub:
    """Records construction args and returns a deterministic branch delta."""

    def __init__(self, *args):
        self.args = args
        self.calls = []

    def __call__(self, positions, hidden):
        self.calls.append((positions, hidden))
        return hidden + 1.0


class _MlpStub:
    def __init__(self, *args):
        self.args = args
        self.calls = []

    def __call__(self, hidden):
        self.calls.append(hidden)
        return hidden * 0.5


class TestHyperLayerWrapper:
    """The checkpoint-compatible wrapper owns one hyper-connection module."""

    def test_constructs_pre_branch_connection(self):
        """HYV4HyperLayer builds a single pre-branch connection."""
        config = _tiny_config()
        layer = HYV4HyperLayer(config)
        assert isinstance(layer.hc_pre, HYV4HyperConnection)
        assert layer.hc_pre.num_streams == config.hc_mult
        assert layer.hc_pre.magnitude == float(config.hc_magnitude)
        assert layer.hc_pre.hc_eps == float(config.hc_eps)
        assert layer.hc_pre.hc_fn.shape == (2 * config.hc_mult, config.hc_mult * 8)


class TestDenseMlp:
    """Tensor-parallel SwiGLU wiring and forward numerics."""

    def _build(self):
        with set_current_vllm_config(_runtime_config()):
            module = HYV4DenseMLP(
                hidden_size=8,
                intermediate_size=16,
                hidden_act="silu",
                prefix="model.layers.0.mlp",
            )
        torch.manual_seed(11)
        with torch.no_grad():
            module.gate_up_proj.weight.copy_(torch.randn(32, 8))
            module.down_proj.weight.copy_(torch.randn(8, 16))
        return module

    @pytest.mark.gpu
    def test_forward_matches_reference(self, single_rank_tp):
        """Gate/up merge + SwiGLU + down projection match a plain reference."""
        module = self._build()
        hidden = torch.randn(5, 8)
        out = module(hidden)

        gate_up = module.gate_up_proj.weight
        gate = hidden @ gate_up[:16].T
        up = hidden @ gate_up[16:].T
        expected = (F.silu(gate) * up) @ module.down_proj.weight.T
        assert out.shape == (5, 8)
        torch.testing.assert_close(out, expected, rtol=RTOL, atol=ATOL)

    @pytest.mark.gpu
    def test_gate_up_and_down_shapes(self, single_rank_tp):
        """The merged gate/up projection has 2x intermediate rows."""
        module = self._build()
        assert module.gate_up_proj.weight.shape == (32, 8)
        assert module.down_proj.weight.shape == (8, 16)
        assert module.gate_up_proj.output_sizes == [16, 16]

    @pytest.mark.gpu
    def test_unsupported_activation_is_rejected(self, single_rank_tp):
        """Only silu is accepted; the failure names the offending activation."""
        with (
            set_current_vllm_config(_runtime_config()),
            pytest.raises(ValueError, match="Unsupported activation: gelu"),
        ):
            HYV4DenseMLP(8, 16, "gelu")


class TestDecoderLayerConstruction:
    """MLP branch selection and layer-index derivation from the prefix."""

    def _vllm_config(self, config):
        return type(
            "VC",
            (),
            {
                "model_config": type("MC", (), {"hf_config": config})(),
                "quant_config": None,
            },
        )()

    @pytest.mark.gpu
    def test_dense_layer_builds_mlp_and_norms(self, single_rank_tp, monkeypatch):
        """A dense layer builds the dense MLP, both norms and both hyper layers."""
        config = _tiny_config(mlp_layer_types=["dense", "dense"])
        monkeypatch.setattr(hy_model, "HYV4Attention", _AttentionStub)
        with set_current_vllm_config(_runtime_config()):
            layer = HYV4DecoderLayer(
                self._vllm_config(config), "model.layers.1", None, None
            )

        assert isinstance(layer.self_attn, _AttentionStub)
        assert isinstance(layer.mlp, HYV4DenseMLP)
        assert isinstance(layer.hc_attn_layer, HYV4HyperLayer)
        assert isinstance(layer.hc_mlp_layer, HYV4HyperLayer)
        assert layer.mlp.gate_up_proj.weight.shape == (
            2 * config.intermediate_size,
            8,
        )
        assert layer.input_layernorm.weight.shape == (config.hidden_size,)
        assert layer.post_attention_layernorm.weight.shape == (config.hidden_size,)
        # Prefix-derived layer index and projection prefixes.
        assert layer.self_attn.args[2] == 1
        assert layer.self_attn.args[3] == "model.layers.1.self_attn"
        assert layer.mlp.gate_up_proj.output_sizes == [config.intermediate_size] * 2

    def test_sparse_layer_selects_moe(self, monkeypatch):
        """A sparse MLP slot constructs HYV4MoE with the layer config."""
        config = _tiny_config(mlp_layer_types=["sparse", "dense"])
        captured = {}

        class _MoEStub:
            def __init__(self, *args):
                captured["args"] = args

        monkeypatch.setattr(hy_model, "HYV4Attention", _AttentionStub)
        monkeypatch.setattr(hy_model, "HYV4MoE", _MoEStub)
        vllm_config = self._vllm_config(config)
        with set_current_vllm_config(_runtime_config()):
            layer = HYV4DecoderLayer(vllm_config, "model.layers.0", None, None)

        assert isinstance(layer.mlp, _MoEStub)
        assert captured["args"] == (config, vllm_config, "model.layers.0.mlp")


class TestDecoderLayerForward:
    """Residual-stream plumbing between hyper connections and branches."""

    def test_forward_matches_independent_reference(self):
        """Read -> norm -> branch -> write is applied twice per layer."""
        config = _tiny_config()
        layer = object.__new__(HYV4DecoderLayer)
        nn.Module.__init__(layer)
        torch.manual_seed(5)
        layer.hc_attn_layer = HYV4HyperLayer(config)
        layer.hc_mlp_layer = HYV4HyperLayer(config)
        layer.self_attn = _AttentionStub()
        layer.mlp = _MlpStub()
        with set_current_vllm_config(_runtime_config()):
            layer.input_layernorm = hy_model.RMSNorm(
                8, eps=config.rms_norm_eps
            )
            layer.post_attention_layernorm = hy_model.RMSNorm(
                8, eps=config.rms_norm_eps
            )
        with torch.no_grad():
            layer.input_layernorm.weight.copy_(torch.randn(8))
            layer.post_attention_layernorm.weight.copy_(torch.randn(8))
        for connection in (layer.hc_attn_layer.hc_pre, layer.hc_mlp_layer.hc_pre):
            with torch.no_grad():
                torch.nn.init.normal_(connection.hc_fn, std=0.5)
                torch.nn.init.normal_(connection.hc_base, std=0.5)
                torch.nn.init.normal_(connection.hc_scale, std=0.5)

        streams = torch.randn(3, config.hc_mult, 8)
        positions = torch.arange(3)
        out = layer(positions, streams)

        hidden, write = _hyper_read_reference(layer.hc_attn_layer.hc_pre, streams)
        hidden = _rms_norm_reference(
            hidden, layer.input_layernorm.weight, config.rms_norm_eps
        )
        attn_out = hidden + 1.0
        streams_1 = _hyper_write_reference(streams, attn_out, write)

        hidden, write = _hyper_read_reference(layer.hc_mlp_layer.hc_pre, streams_1)
        hidden = _rms_norm_reference(
            hidden, layer.post_attention_layernorm.weight, config.rms_norm_eps
        )
        mlp_out = hidden * 0.5
        expected = _hyper_write_reference(streams_1, mlp_out, write)

        assert out.shape == streams.shape
        torch.testing.assert_close(out, expected, rtol=1e-5, atol=1e-5)
        assert len(layer.self_attn.calls) == 1
        assert len(layer.mlp.calls) == 1


class TestMoeForward:
    """Routed + shared combination, including the sequence-parallel gather."""

    class _FakeGate:
        def __init__(self, logits):
            self.logits = logits
            self.calls = []

        def __call__(self, hidden):
            self.calls.append(hidden)
            return self.logits, None

    class _FakeExperts:
        is_internal_router = False

        def __init__(self, offset):
            self.offset = offset
            self.calls = []

        def __call__(self, *, hidden_states, router_logits):
            self.calls.append((hidden_states, router_logits))
            return hidden_states + self.offset

    class _FakeShared:
        def __init__(self, offset):
            self.offset = offset

        def __call__(self, hidden):
            return hidden + self.offset

    def _bare(self, internal_router=False, sequence_parallel=False):
        moe = object.__new__(HYV4MoE)
        nn.Module.__init__(moe)
        moe.is_sequence_parallel = sequence_parallel
        logits = torch.zeros(4, 2)
        moe.gate = self._FakeGate(logits)
        moe.experts = self._FakeExperts(1.0)
        moe.experts.is_internal_router = internal_router
        moe.shared_experts = self._FakeShared(2.0)
        return moe

    def test_forward_combines_gate_routed_and_shared_in_fp32(self):
        """Gate logits feed the routed experts; the sum is FP32 then recast."""
        moe = self._bare()
        hidden = torch.randn(4, 8, dtype=torch.bfloat16)
        out = HYV4MoE.forward(moe, hidden)

        expected = (hidden.float() + 1.0) + (hidden.float() + 2.0)
        assert torch.equal(moe.gate.calls[0], hidden)
        assert moe.experts.calls[0][1] is moe.gate.logits
        assert out.dtype == torch.bfloat16
        torch.testing.assert_close(out.float(), expected, rtol=1e-2, atol=1e-2)

    def test_forward_internal_router_reuses_hidden_states(self):
        """An internal router consumes hidden_states instead of the gate."""
        moe = self._bare(internal_router=True)
        hidden = torch.randn(2, 3, 8)
        out = HYV4MoE.forward(moe, hidden)

        assert moe.gate.calls == []
        assert torch.equal(
            moe.experts.calls[0][1], hidden.reshape(-1, hidden.shape[-1])
        )
        expected = (hidden + 1.0) + (hidden + 2.0)
        torch.testing.assert_close(out, expected, rtol=0, atol=0)

    def test_forward_sequence_parallel_chunks_and_gathers(self, monkeypatch):
        """SP chunks tokens, then truncates the gathered combination."""
        moe = self._bare(sequence_parallel=True)
        original = torch.randn(5, 8)
        peer = torch.randn(3, 8)
        assert_events = []

        def _chunk(hidden):
            assert_events.append(("chunk", hidden))
            return hidden[:3]

        def _gather(hidden, dim):
            assert_events.append(("gather", hidden, dim))
            return torch.cat([hidden, peer], dim=0)

        monkeypatch.setattr(hy_model, "sequence_parallel_chunk", _chunk)
        monkeypatch.setattr(hy_model, "tensor_model_parallel_all_gather", _gather)

        out = HYV4MoE.forward(moe, original)

        local = original[:3]
        expected_local = (local + 1.0) + (local + 2.0)
        expected = torch.cat([expected_local, peer], dim=0)[:5]
        assert [event[0] for event in assert_events] == ["chunk", "gather"]
        assert assert_events[1][2] == 0
        torch.testing.assert_close(out, expected, rtol=0, atol=0)


def _bare_moe(n_logical=4, n_physical=6, n_local=3, n_shared=2, n_redundant=2):
    """A minimal HYV4MoE instance carrying only the metadata vLLM reads."""
    moe = object.__new__(HYV4MoE)
    nn.Module.__init__(moe)
    moe.n_logical_experts = n_logical
    moe.n_physical_experts = n_physical
    moe.n_local_physical_experts = n_local
    moe.n_routed_experts = n_logical
    moe.n_shared_experts = n_shared
    moe.n_redundant_experts = n_redundant
    updates = []
    moe.experts = type(
        "E",
        (),
        {
            "routed_experts": type(
                "R", (), {"update_expert_map": lambda self: updates.append(True)}
            )()
        },
    )()
    moe._updates = updates
    return moe


class TestMixtureOfExpertsMetadata:
    """Expert metadata forwarded to vLLM's EPLB infrastructure."""

    def test_extract_none_zeroes_every_count(self):
        """A dense-only model reports zero experts in every bucket."""
        host = HYV4MixtureOfExperts()
        host.num_moe_layers = 7
        host.extract_moe_parameters(None)
        assert host.num_moe_layers == 0
        for name in (
            "num_expert_groups",
            "num_logical_experts",
            "num_physical_experts",
            "num_local_physical_experts",
            "num_routed_experts",
            "num_shared_experts",
            "num_redundant_experts",
        ):
            assert getattr(host, name) == 0

    def test_extract_copies_example_moe_counts(self):
        """Counts are copied verbatim from the first MoE layer."""
        host = HYV4MixtureOfExperts()
        moe = _bare_moe(4, 6, 3, 2, 2)
        host.extract_moe_parameters(moe)
        assert host.num_logical_experts == 4
        assert host.num_physical_experts == 6
        assert host.num_local_physical_experts == 3
        assert host.num_routed_experts == 4
        assert host.num_shared_experts == 2
        assert host.num_redundant_experts == 2

    def test_update_rejects_changed_local_count(self):
        """A changed local count fails instead of silently re-sharding."""
        host = HYV4MixtureOfExperts()
        host.extract_moe_parameters(_bare_moe(n_local=3))
        host.moe_mlp_layers = []
        with pytest.raises(ValueError, match="local expert count changed"):
            host.update_physical_experts_metadata(8, 4)

    def test_update_propagates_counts_and_refreshes_maps(self):
        """A successful update rewrites every MoE layer and its expert map."""
        host = HYV4MixtureOfExperts()
        moe = _bare_moe(4, 6, 3, 2, 2)
        host.extract_moe_parameters(moe)
        host.moe_mlp_layers = [moe]

        host.update_physical_experts_metadata(7, 3)

        assert host.num_physical_experts == 7
        assert host.num_redundant_experts == 3
        assert moe.n_physical_experts == 7
        assert moe.n_redundant_experts == 3
        assert moe._updates == [True]


class _ApplyRecorder:
    def __init__(self, logits):
        self.logits = logits
        self.calls = []

    def apply(self, layer, hidden, bias=None):
        self.calls.append((layer, hidden, bias))
        return self.logits


class TestLogitsProcessorGetLogits:
    """The FP32 language head casts activations and slices padding away."""

    def _processor(self, gather):
        processor = object.__new__(hy_model.HYV4LogitsProcessor)
        nn.Module.__init__(processor)
        processor.org_vocab_size = 6
        processor._gather_logits = gather
        return processor

    def test_casts_hidden_to_weight_dtype_and_slices_vocab(self):
        """Hidden states are cast to the head dtype before the quant method."""
        logits = torch.arange(40, dtype=torch.float32).reshape(4, 10)
        apply = _ApplyRecorder(logits)
        weight = torch.zeros(8, 4, dtype=torch.bfloat16)
        lm_head = type("H", (), {"weight": weight, "quant_method": apply})()
        hidden = torch.randn(4, 4, dtype=torch.float32)
        bias = torch.zeros(8)
        processor = self._processor(lambda value: value)

        out = processor._get_logits(hidden, lm_head, bias)

        assert apply.calls[0][0] is lm_head
        assert apply.calls[0][1].dtype == torch.bfloat16
        assert apply.calls[0][2] is bias
        torch.testing.assert_close(out, logits[:, :6])

    def test_none_gather_returns_none(self):
        """A TP rank that gathered nothing propagates None."""
        apply = _ApplyRecorder(torch.zeros(2, 4))
        lm_head = type(
            "H", (), {"weight": torch.zeros(4, 4), "quant_method": apply}
        )()
        processor = self._processor(lambda value: None)
        assert processor._get_logits(torch.zeros(2, 4), lm_head, None) is None


class _FakeModel(nn.Module):
    def __init__(self, *, vllm_config, prefix=""):
        super().__init__()
        self.captured = (vllm_config, prefix)
        self.layers = [
            hy_model.PPMissingLayer(),
            type("L", (), {"mlp": _bare_moe()})(),
            type("L", (), {"mlp": object()})(),
        ]
        self.make_empty_intermediate_tensors = lambda *args, **kwargs: "empty"


class _FakeLMHead:
    def __init__(self, num_embeddings, embedding_dim, *, params_dtype, quant_config,
                 prefix):
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.params_dtype = params_dtype
        self.quant_config = quant_config
        self.prefix = prefix


class _FakeLogitsProcessor:
    def __init__(self, vocab_size):
        self.vocab_size = vocab_size


def _lm_config(config, dtype=torch.bfloat16):
    return type(
        "VC",
        (),
        {
            "model_config": type(
                "MC", (), {"hf_config": config, "dtype": dtype}
            )(),
            "quant_config": None,
        },
    )()


class _RankGroup:
    def __init__(self, first=True, last=True):
        self.is_first_rank = first
        self.is_last_rank = last


class TestForCausalLMWiring:
    """Embedding/LM-head wiring and expert-layer discovery."""

    def _build(self, monkeypatch, config, group, prefix=""):
        captured = {}
        monkeypatch.setattr(hy_model, "HYV4Model", _FakeModel)
        monkeypatch.setattr(hy_model, "ParallelLMHead", _FakeLMHead)
        monkeypatch.setattr(hy_model, "HYV4LogitsProcessor", _FakeLogitsProcessor)
        monkeypatch.setattr(hy_model, "get_pp_group", lambda: group)
        model = HYV4ForCausalLM(vllm_config=_lm_config(config), prefix=prefix)
        captured["lm_head"] = model.lm_head
        captured["model"] = model.model
        captured["logits"] = model.logits_processor
        return model, captured

    def test_fp32_lm_head_and_model_prefix(self, monkeypatch):
        """enable_lm_head_fp32 selects FP32 weights and the model prefix."""
        config = _tiny_config(enable_lm_head_fp32=True)
        model, captured = self._build(
            monkeypatch, config, _RankGroup(), prefix="plugin"
        )
        head = captured["lm_head"]
        assert head.params_dtype == torch.float32
        assert (head.num_embeddings, head.embedding_dim) == (16, 8)
        assert head.prefix == "plugin.lm_head"
        assert captured["model"].captured[1] == "plugin.model"
        assert captured["logits"].vocab_size == 16
        assert model.make_empty_intermediate_tensors is not None
        # One real MoE layer discovered; the PPMissingLayer and dense layer
        # are ignored.
        assert model.num_moe_layers == 1
        assert len(model.moe_mlp_layers) == 1
        assert model.moe_layers == [
            model.moe_mlp_layers[0].experts.routed_experts
        ]

    def test_bf16_lm_head_follows_model_dtype(self, monkeypatch):
        """Without the FP32 flag the head uses the configured model dtype."""
        config = _tiny_config(enable_lm_head_fp32=False)
        _, captured = self._build(
            monkeypatch, config, _RankGroup(), prefix=""
        )
        assert captured["lm_head"].params_dtype == torch.bfloat16
        assert captured["lm_head"].prefix == "lm_head"
        assert captured["model"].captured[1] == "model"

    def test_non_last_rank_uses_missing_lm_head(self, monkeypatch):
        """Non-final pipeline ranks hold a PPMissingLayer LM head."""
        config = _tiny_config()
        _, captured = self._build(
            monkeypatch, config, _RankGroup(first=True, last=False)
        )
        assert isinstance(captured["lm_head"], hy_model.PPMissingLayer)

    def test_moe_metadata_extracted_from_discovered_layer(self, monkeypatch):
        """Discovery feeds the example MoE into extract_moe_parameters."""
        config = _tiny_config()
        model, _ = self._build(monkeypatch, config, _RankGroup())
        assert model.num_logical_experts == 4
        assert model.num_physical_experts == 6
        assert model.num_local_physical_experts == 3
        assert model.num_redundant_experts == 2


class TestExpertParamsMapping:
    """Canonical and legacy routed-expert checkpoint name mappings."""

    def test_uses_vllm_helper_when_available(self, monkeypatch):
        """The canonical helper receives the expert counts and projection names."""
        observed = []

        def _helper(model, **kwargs):
            observed.append(kwargs)
            return [("p", "w", 0, "w1")]

        monkeypatch.setattr(
            hy_model, "fused_moe_make_expert_params_mapping", _helper
        )
        model = SimpleNamespace(
            config=SimpleNamespace(n_routed_experts=4), n_redundant_experts=2
        )
        assert hy_model._make_hyv4_expert_params_mapping(model) == [
            ("p", "w", 0, "w1")
        ]
        assert observed == [
            {
                "ckpt_gate_proj_name": "gate_proj",
                "ckpt_down_proj_name": "down_proj",
                "ckpt_up_proj_name": "up_proj",
                "num_experts": 4,
                "num_redundant_experts": 2,
            }
        ]

    def test_legacy_fallback_builds_three_rows_per_expert(self, monkeypatch):
        """Without the helper each expert maps gate/up to w13 and down to w2."""
        monkeypatch.setattr(
            hy_model, "fused_moe_make_expert_params_mapping", None
        )
        model = SimpleNamespace(config=SimpleNamespace(n_routed_experts=2))
        expected = []
        for expert in range(2):
            expected.extend(
                [
                    (
                        "experts.routed_experts.w13_",
                        f"experts.{expert}.gate_proj.",
                        expert,
                        "w1",
                    ),
                    (
                        "experts.routed_experts.w2_",
                        f"experts.{expert}.down_proj.",
                        expert,
                        "w2",
                    ),
                    (
                        "experts.routed_experts.w13_",
                        f"experts.{expert}.up_proj.",
                        expert,
                        "w3",
                    ),
                ]
            )
        assert hy_model._make_hyv4_expert_params_mapping(model) == expected

    def test_legacy_fallback_rejects_eplb(self, monkeypatch):
        """Redundant experts require the canonical vLLM helper."""
        monkeypatch.setattr(
            hy_model, "fused_moe_make_expert_params_mapping", None
        )
        model = SimpleNamespace(
            config=SimpleNamespace(n_routed_experts=2), n_redundant_experts=1
        )
        with pytest.raises(RuntimeError, match="EPLB loading requires"):
            hy_model._make_hyv4_expert_params_mapping(model)


class TestAttentionWrapper:
    """HYV4Attention forwards every model-config field to the MLA parent."""

    def test_attention_subclass_maps_config_arguments(self, monkeypatch):
        """The wrapper passes hidden/head dimensions, cache and plan through."""
        captured = {}

        def _fake_init(self, **kwargs):
            captured.update(kwargs)

        monkeypatch.setattr(
            hy_model.HYV4MLAAttention, "__init__", _fake_init
        )
        config = _tiny_config()
        cache_config = SimpleNamespace(cache_dtype="auto")
        vllm_config = SimpleNamespace(
            cache_config=cache_config, quant_config=None
        )
        plan = object()
        buffer = torch.zeros(1, dtype=torch.int32)

        hy_model.HYV4Attention(
            vllm_config,
            config,
            3,
            "model.layers.3.self_attn",
            buffer,
            plan,
        )

        assert captured == {
            "vllm_config": vllm_config,
            "config": config,
            "hidden_size": config.hidden_size,
            "num_heads": config.num_attention_heads,
            "qk_nope_head_dim": config.qk_nope_head_dim,
            "qk_rope_head_dim": config.qk_rope_head_dim,
            "v_head_dim": config.v_head_dim,
            "q_lora_rank": config.q_lora_rank,
            "kv_lora_rank": config.kv_lora_rank,
            "max_position_embeddings": config.max_position_embeddings,
            "cache_config": cache_config,
            "quant_config": None,
            "prefix": "model.layers.3.self_attn",
            "topk_indices_buffer": buffer,
            "layer_idx": 3,
            "runtime_plan": plan,
        }
