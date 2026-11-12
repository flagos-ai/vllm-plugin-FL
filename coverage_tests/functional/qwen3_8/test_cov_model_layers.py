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

"""Qwen3.8-Flash-Next decoder-layer and backbone construction/forward paths.

The layer constructor selects the attention owner, MoE-vs-dense MLP and both
hyper-connection branches from config; every upstream layer is replaced with a
recording double so only the plugin's own selection logic is observed.  The
backbone forward is exercised with callable layer doubles and an independent
model of the delayed-combine/injection schedule it must thread through.
"""

from types import SimpleNamespace

import pytest
import torch
from torch import nn

from vllm.config import set_current_vllm_config
from vllm.config.compilation import CompilationConfig, CompilationMode

from . import _import_shim  # noqa: F401  (idempotent; no-op on CI)
from vllm_fl.models.qwen3_8_flash_next.gpu import model as qwen_model


class _RecordingModule(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()
        self.args = args
        self.kwargs = kwargs


def _text_config(**overrides):
    values = dict(
        hidden_size=8,
        vocab_size=32,
        num_hidden_layers=1,
        intermediate_size=16,
        hidden_act="silu",
        hc_count=2,
        hc_lowrank=4,
        rms_norm_eps=1e-6,
        ple_layer_ids=[],
        mlp_only_layers=[],
        num_experts=0,
        decoder_sparse_step=1,
        layer_types=["full_attention"],
    )
    values.update(overrides)
    return SimpleNamespace(**values)


def _vllm_config(config, *, prefix="layers.0", sequence_parallel=False):
    return SimpleNamespace(
        parallel_config=SimpleNamespace(use_sequence_parallel_moe=sequence_parallel),
        model_config=SimpleNamespace(hf_text_config=config),
        cache_config=SimpleNamespace(),
        quant_config=None,
        speculative_config=None,
        scheduler_config=SimpleNamespace(max_num_batched_tokens=16),
        compilation_config=_compilation_config(),
    )


def _compilation_config():
    return CompilationConfig(mode=CompilationMode.NONE)


class _HyperConnection(nn.Module):
    def __init__(self, config, *, layer_idx=None, role="", **kwargs):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.role = role
        self.kwargs = kwargs


def _install_decoder_doubles(
    monkeypatch,
    *,
    gdn_accepts_reduce=True,
    gdn_forward_has_output=False,
):
    created = SimpleNamespace(
        gdn=None,
        non_qsa=None,
        qsa=None,
        moe=None,
        dense=None,
        ple=None,
        hyper=[],
    )

    class _GdnBase(nn.Module):
        def __init__(self, config, **kwargs):
            super().__init__()
            self.config = config
            self.kwargs = kwargs
            self.out_proj = SimpleNamespace(reduce_results=True)
            created.gdn = self

    if gdn_accepts_reduce:

        class FakeGdn(_GdnBase):
            def __init__(
                self,
                config,
                *,
                vllm_config,
                prefix,
                gqa_interleaved_layout,
                reduce_results,
            ):
                super().__init__(config)
                self.kwargs = dict(
                    vllm_config=vllm_config,
                    prefix=prefix,
                    gqa_interleaved_layout=gqa_interleaved_layout,
                    reduce_results=reduce_results,
                )

    else:

        class FakeGdn(_GdnBase):
            def __init__(
                self,
                config,
                *,
                vllm_config,
                prefix,
                gqa_interleaved_layout,
            ):
                super().__init__(config)
                self.kwargs = dict(
                    vllm_config=vllm_config,
                    prefix=prefix,
                    gqa_interleaved_layout=gqa_interleaved_layout,
                )

    if gdn_forward_has_output:
        FakeGdn.forward = lambda self, hidden_states, output: output
    else:
        FakeGdn.forward = lambda self, hidden_states: hidden_states

    class FakeAttention(_RecordingModule):
        def __init__(self, config, **kwargs):
            super().__init__(config, **kwargs)
            created.non_qsa = self

    class FakeQsa(_RecordingModule):
        def __init__(self, **kwargs):
            super().__init__(**kwargs)
            created.qsa = self

    class FakeMoe(_RecordingModule):
        def __init__(self, **kwargs):
            super().__init__(**kwargs)
            created.moe = self

    class FakeMlp(_RecordingModule):
        def __init__(self, **kwargs):
            super().__init__(**kwargs)
            created.dense = self

    class FakePle(_RecordingModule):
        def __init__(self, config, **kwargs):
            super().__init__(config, **kwargs)
            created.ple = self

    def fake_hyper(config, *, layer_idx=None, role="", **kwargs):
        module = _HyperConnection(
            config, layer_idx=layer_idx, role=role, **kwargs
        )
        created.hyper.append(module)
        return module

    monkeypatch.setattr(qwen_model, "QwenGatedDeltaNetAttention", FakeGdn)
    monkeypatch.setattr(qwen_model, "Qwen3NextAttention", FakeAttention)
    monkeypatch.setattr(qwen_model, "Qwen3_8FlashNextQSAAttention", FakeQsa)
    monkeypatch.setattr(qwen_model, "Qwen3_8FlashNextSparseMoeBlock", FakeMoe)
    monkeypatch.setattr(qwen_model, "Qwen3NextMLP", FakeMlp)
    monkeypatch.setattr(qwen_model, "Qwen3_8FlashNextPLELayer", FakePle)
    monkeypatch.setattr(
        qwen_model,
        "HyperConnectionConfig",
        lambda **kwargs: SimpleNamespace(**kwargs),
    )
    monkeypatch.setattr(qwen_model, "GatedResidualSimple", fake_hyper)
    return created


class TestDecoderLayerConstruction:
    def test_linear_attention_passes_reduce_results_when_supported(
        self, monkeypatch
    ):
        created = _install_decoder_doubles(monkeypatch)
        config = _text_config()
        layer = qwen_model.Qwen3_8FlashNextDecoderLayer(
            _vllm_config(config), "linear_attention", prefix="layers.0"
        )
        assert created.gdn.kwargs["prefix"] == "layers.0.linear_attn"
        assert created.gdn.kwargs["gqa_interleaved_layout"] is False
        assert created.gdn.kwargs["reduce_results"] is False
        assert created.gdn.out_proj.reduce_results is True
        assert layer._gdn_requires_output_buffer is False
        assert layer.ple is None

    def test_linear_attention_disables_internal_reduction_without_the_kwarg(
        self, monkeypatch
    ):
        created = _install_decoder_doubles(
            monkeypatch, gdn_accepts_reduce=False, gdn_forward_has_output=True
        )
        config = _text_config()
        layer = qwen_model.Qwen3_8FlashNextDecoderLayer(
            _vllm_config(config), "linear_attention", prefix="layers.0"
        )
        assert "reduce_results" not in created.gdn.kwargs
        assert created.gdn.out_proj.reduce_results is False
        assert layer._gdn_requires_output_buffer is True

    def test_full_attention_without_an_indexer_uses_the_stock_qwen_attention(
        self, monkeypatch
    ):
        created = _install_decoder_doubles(monkeypatch)
        config = _text_config()
        qwen_model.Qwen3_8FlashNextDecoderLayer(
            _vllm_config(config), "full_attention", prefix="layers.0"
        )
        assert created.qsa is None
        kwargs = created.non_qsa.kwargs
        assert kwargs["prefix"] == "layers.0.self_attn"
        assert kwargs["reduce_results"] is False
        assert kwargs["model_config"] is not None
        assert kwargs["cache_config"] is not None
        assert kwargs["quant_config"] is None

    def test_full_attention_with_an_indexer_builds_the_qsa_owner(
        self, monkeypatch
    ):
        created = _install_decoder_doubles(monkeypatch)
        config = _text_config(indexer_n_heads=4)
        layer = qwen_model.Qwen3_8FlashNextDecoderLayer(
            _vllm_config(config), "full_attention", prefix="layers.3"
        )
        assert created.non_qsa is None
        assert layer.layer_idx == 3
        kwargs = created.qsa.kwargs
        assert kwargs["layer_id"] == 3
        assert kwargs["reduce_results"] is False
        assert kwargs["prefix"] == "layers.3.self_attn"

    def test_invalid_layer_type_is_rejected(self, monkeypatch):
        _install_decoder_doubles(monkeypatch)
        with pytest.raises(ValueError, match="Invalid layer_type"):
            qwen_model.Qwen3_8FlashNextDecoderLayer(
                _vllm_config(_text_config()), "sliding_attention", prefix="layers.0"
            )

    def test_ple_layer_is_attached_for_member_layer_indices(self, monkeypatch):
        _install_decoder_doubles(monkeypatch)
        config = _text_config(ple_layer_ids=[1, 3])
        layer = qwen_model.Qwen3_8FlashNextDecoderLayer(
            _vllm_config(config), "full_attention", prefix="layers.0"
        )
        assert layer.ple is not None
        assert layer.ple.kwargs["layer_idx"] == 0
        assert layer.ple.kwargs["ple_dense_layer_id"] == 0
        assert layer.ple.kwargs["prefix"] == "layers.0.ple"
        # Layer index 1 is the second (not member) layer.
        assert (
            qwen_model.Qwen3_8FlashNextDecoderLayer(
                _vllm_config(config), "full_attention", prefix="layers.1"
            ).ple
            is None
        )

    def test_force_disable_ple_skips_the_embedding_layer(self, monkeypatch):
        _install_decoder_doubles(monkeypatch)
        config = _text_config(ple_layer_ids=[1])
        layer = qwen_model.Qwen3_8FlashNextDecoderLayer(
            _vllm_config(config),
            "full_attention",
            prefix="layers.0",
            force_disable_ple=True,
        )
        assert layer.ple is None

    def test_moe_and_dense_mlp_selection_follows_sparse_step(self, monkeypatch):
        created = _install_decoder_doubles(monkeypatch)
        config = _text_config(num_experts=2, decoder_sparse_step=2)
        qwen_model.Qwen3_8FlashNextDecoderLayer(
            _vllm_config(config), "full_attention", prefix="layers.1"
        )
        # absolute layer 2 % 2 == 0 -> sparse MoE.
        assert created.moe is not None
        assert created.moe.kwargs["prefix"] == "layers.1.mlp"

        created = _install_decoder_doubles(monkeypatch)
        qwen_model.Qwen3_8FlashNextDecoderLayer(
            _vllm_config(config), "full_attention", prefix="layers.0"
        )
        # absolute layer 1 % 2 != 0 -> dense MLP.
        assert created.moe is None
        assert created.dense.kwargs["hidden_size"] == 8
        assert created.dense.kwargs["intermediate_size"] == 16
        assert created.dense.kwargs["reduce_results"] is False

    def test_mlp_only_layers_never_become_moe(self, monkeypatch):
        created = _install_decoder_doubles(monkeypatch)
        config = _text_config(num_experts=4, decoder_sparse_step=1, mlp_only_layers=[0])
        qwen_model.Qwen3_8FlashNextDecoderLayer(
            _vllm_config(config), "full_attention", prefix="layers.0"
        )
        assert created.moe is None
        assert created.dense is not None

    def test_both_hyper_connections_are_attached_in_role_order(self, monkeypatch):
        created = _install_decoder_doubles(monkeypatch)
        config = _text_config()
        layer = qwen_model.Qwen3_8FlashNextDecoderLayer(
            _vllm_config(config), "full_attention", prefix="layers.2"
        )
        assert [module.role for module in created.hyper] == ["attn", "mlp"]
        assert [module.layer_idx for module in created.hyper] == [2, 2]
        assert layer.attn_hyper_connection is created.hyper[0]
        assert layer.mlp_hyper_connection is created.hyper[1]
        assert created.hyper[0].config.hc_count == 2
        assert created.hyper[0].config.hidden_size == 8


class _Mixer:
    def __init__(self, *, pending_value=100, mix_value=10, injection="injection"):
        self.pending_value = pending_value
        self.mix_value = mix_value
        self.injection = injection
        self.mix_calls = []
        self.pending_calls = []

    def combine_and_mix(self, hidden_states, prev_block_output, prev_injection):
        self.mix_calls.append((hidden_states, prev_block_output, prev_injection))
        return (
            hidden_states + 1,
            hidden_states + self.mix_value,
            self.injection,
        )

    def combine_pending(self, hidden_states, block_output, injection_logits):
        self.pending_calls.append((hidden_states, block_output, injection_logits))
        return hidden_states + self.pending_value


class _Attention:
    def __init__(self, value=2, output_buffer=False):
        self.value = value
        self.output_buffer = output_buffer
        self.calls = []
        self.buffers = []

    def __call__(self, *args, **kwargs):
        self.calls.append((args, kwargs))
        hidden = args[0] if args else kwargs["hidden_states"]
        if self.output_buffer:
            output = args[1]
            output.fill_(self.value)
            self.buffers.append(output)
            return None
        return torch.full_like(hidden, self.value)


class _Mlp:
    def __init__(self, value=3, requires_tp_all_reduce=True):
        self.value = value
        self.requires_tp_all_reduce = requires_tp_all_reduce
        self.calls = []

    def __call__(self, hidden_states):
        self.calls.append(hidden_states)
        return torch.full_like(hidden_states, self.value)


def _forward_layer(
    *,
    layer_type="full_attention",
    ple=None,
    gdn_buffer=False,
    mlp_requires_reduce=True,
):
    layer = object.__new__(qwen_model.Qwen3_8FlashNextDecoderLayer)
    nn.Module.__init__(layer)
    layer.layer_type = layer_type
    layer.ple = ple
    layer.attn_hyper_connection = _Mixer()
    layer.mlp_hyper_connection = _Mixer(mix_value=20, injection="next-injection")
    layer.self_attn = _Attention(value=2)
    layer.linear_attn = _Attention(value=4, output_buffer=gdn_buffer)
    layer._gdn_requires_output_buffer = gdn_buffer
    layer.mlp = _Mlp(value=3, requires_tp_all_reduce=mlp_requires_reduce)
    return layer


class TestDecoderLayerForward:
    def test_legacy_residual_tensor_is_rejected(self):
        layer = _forward_layer()
        with pytest.raises(ValueError, match="prev_block_output/prev_injection"):
            layer(torch.zeros(2, 16), residual=torch.zeros(2, 16))

    @pytest.mark.parametrize(
        ("kwargs", "message"),
        [
            ({"query_start_loc": torch.tensor([0, 2])}, "input_ids"),
            (
                {
                    "input_ids": torch.tensor([1, 2]),
                    "ngram_context": torch.zeros(1, 2),
                },
                "query_start_loc",
            ),
            (
                {
                    "input_ids": torch.tensor([1, 2]),
                    "query_start_loc": torch.tensor([0, 2]),
                },
                "ngram_context",
            ),
        ],
    )
    def test_ple_requires_every_context_input(self, kwargs, message):
        layer = _forward_layer(ple=lambda *args: torch.zeros(2, 16))

        def call(**inputs):
            layer.forward(torch.zeros(2, 16), **inputs)

        with pytest.raises(ValueError, match=message):
            call(**kwargs)

    def test_ple_materializes_a_pending_combine_before_adding(self, monkeypatch):
        monkeypatch.setattr(
            qwen_model, "get_tensor_model_parallel_world_size", lambda: 1
        )
        ple_calls = []

        def ple(hidden, input_ids, query_start_loc, ngram_context):
            ple_calls.append((hidden, input_ids, query_start_loc, ngram_context))
            return torch.full_like(hidden, 5)

        layer = _forward_layer(ple=ple)
        hidden = torch.zeros(2, 16)
        result, block, injection = layer(
            hidden,
            prev_block_output=torch.ones(2, 8),
            prev_injection=torch.ones(2, 2),
            positions=torch.zeros(2),
            input_ids=torch.tensor([1, 2]),
            query_start_loc=torch.tensor([0, 2]),
            ngram_context=torch.zeros(1, 2),
        )
        assert layer.attn_hyper_connection.pending_calls[0][1].shape == (2, 8)
        assert ple_calls[0][0].tolist() == torch.full((2, 16), 100.0).tolist()
        # The PLE add happens after materialization, then both HC boundaries
        # add their per-branch mix contribution.
        assert result.tolist() == torch.full((2, 16), 107.0).tolist()
        assert block.tolist() == torch.full((2, 16), 3.0).tolist()
        assert injection == "next-injection"
        assert layer.attn_hyper_connection.mix_calls[0][1] is None
        assert layer.attn_hyper_connection.mix_calls[0][2] is None

    def test_linear_attention_writes_into_the_caller_buffer(self, monkeypatch):
        monkeypatch.setattr(qwen_model, "get_tensor_model_parallel_world_size", lambda: 1)
        layer = _forward_layer(layer_type="linear_attention", gdn_buffer=True)
        result, block, _ = layer(torch.zeros(2, 16), positions=torch.zeros(2))
        assert len(layer.linear_attn.buffers) == 1
        assert layer.linear_attn.buffers[0].tolist() == torch.full((2, 16), 4.0).tolist()
        # The HC boundary consumes the filled buffer as its block output.
        assert (
            layer.mlp_hyper_connection.mix_calls[0][1].tolist()
            == torch.full((2, 16), 4.0).tolist()
        )
        assert result.tolist() == torch.full((2, 16), 2.0).tolist()
        assert block.tolist() == torch.full((2, 16), 3.0).tolist()

    def test_linear_attention_without_a_buffer_returns_the_projection(
        self, monkeypatch
    ):
        monkeypatch.setattr(qwen_model, "get_tensor_model_parallel_world_size", lambda: 1)
        layer = _forward_layer(layer_type="linear_attention", gdn_buffer=False)
        layer(torch.zeros(2, 16), positions=torch.zeros(2))
        args, kwargs = layer.linear_attn.calls[0]
        assert args == ()
        assert kwargs["hidden_states"].tolist() == torch.full((2, 16), 10.0).tolist()

    def test_full_attention_receives_positions(self, monkeypatch):
        monkeypatch.setattr(qwen_model, "get_tensor_model_parallel_world_size", lambda: 1)
        layer = _forward_layer()
        positions = torch.tensor([3, 4])
        layer(torch.zeros(2, 16), positions=positions)
        _, kwargs = layer.self_attn.calls[0]
        assert kwargs["positions"] is positions
        assert kwargs["hidden_states"].tolist() == torch.full((2, 16), 10.0).tolist()
        assert (
            layer.mlp_hyper_connection.mix_calls[0][1].tolist()
            == torch.full((2, 16), 2.0).tolist()
        )

    def test_unknown_layer_type_fails_at_forward(self, monkeypatch):
        monkeypatch.setattr(qwen_model, "get_tensor_model_parallel_world_size", lambda: 1)
        layer = _forward_layer(layer_type="unknown")
        with pytest.raises(ValueError, match="Invalid layer_type"):
            layer(torch.zeros(2, 16), positions=torch.zeros(2))

    def test_tp_reduction_applies_to_attention_and_reducible_mlp(self, monkeypatch):
        reduced = []

        def fake_all_reduce(tensor):
            reduced.append(tensor.clone())
            return tensor * 2

        monkeypatch.setattr(qwen_model, "get_tensor_model_parallel_world_size", lambda: 2)
        monkeypatch.setattr(qwen_model, "tensor_model_parallel_all_reduce", fake_all_reduce)
        layer = _forward_layer()
        result, block, _ = layer(torch.zeros(2, 16), positions=torch.zeros(2))
        assert len(reduced) == 2
        assert result.tolist() == torch.full((2, 16), 2.0).tolist()
        assert block.tolist() == torch.full((2, 16), 6.0).tolist()

    def test_moe_layers_skip_the_second_reduction(self, monkeypatch):
        reduced = []

        def fake_all_reduce(tensor):
            reduced.append(tensor.clone())
            return tensor * 2

        monkeypatch.setattr(qwen_model, "get_tensor_model_parallel_world_size", lambda: 2)
        monkeypatch.setattr(qwen_model, "tensor_model_parallel_all_reduce", fake_all_reduce)
        layer = _forward_layer(mlp_requires_reduce=False)
        _, block, _ = layer(torch.zeros(2, 16), positions=torch.zeros(2))
        assert len(reduced) == 1
        assert block.tolist() == torch.full((2, 16), 3.0).tolist()


class _FakeDecoderLayer(_RecordingModule):
    def __init__(self, vllm_config, *, layer_type, prefix):
        super().__init__(vllm_config, layer_type=layer_type, prefix=prefix)


class TestModelConstruction:
    def _build(self, monkeypatch, config, *, pp_last=True, spec=None, layers=None):
        created = SimpleNamespace(embed=None, layers=None, layer_calls=[], hyper=[])
        calls = {"make_layers": []}

        class FakeEmbedding(nn.Module):
            def __init__(self, vocab_size, hidden_size):
                super().__init__()
                self.vocab_size = vocab_size
                self.hidden_size = hidden_size
                created.embed = self

            def forward(self, input_ids):
                return torch.zeros(input_ids.shape[0], self.hidden_size)

        def fake_make_layers(num_hidden_layers, layer_fn, prefix):
            calls["make_layers"].append((num_hidden_layers, prefix))
            modules = nn.ModuleList()
            for layer_idx in range(num_hidden_layers):
                module = layer_fn(f"{prefix}.{layer_idx}")
                created.layer_calls.append(module)
                modules.append(module)
            created.layers = modules
            return 0, num_hidden_layers, modules

        def fake_hyper(config, *, use_combine=True, role="", **kwargs):
            module = SimpleNamespace(config=config, use_combine=use_combine, role=role)
            created.hyper.append(module)
            return module

        monkeypatch.setattr(qwen_model, "with_platform_quantization", lambda c: c)
        monkeypatch.setattr(qwen_model, "VocabParallelEmbedding", FakeEmbedding)
        monkeypatch.setattr(qwen_model, "make_layers", fake_make_layers)
        monkeypatch.setattr(
            qwen_model, "Qwen3_8FlashNextDecoderLayer", _FakeDecoderLayer
        )
        monkeypatch.setattr(qwen_model, "GatedResidualSimple", fake_hyper)
        monkeypatch.setattr(
            qwen_model,
            "HyperConnectionConfig",
            lambda **kwargs: SimpleNamespace(**kwargs),
        )
        monkeypatch.setattr(
            qwen_model,
            "get_pp_group",
            lambda: SimpleNamespace(is_last_rank=pp_last, is_first_rank=True),
        )
        vllm_config = SimpleNamespace(
            model_config=SimpleNamespace(
                hf_text_config=config, dtype=torch.bfloat16
            ),
            parallel_config=SimpleNamespace(
                eplb_config=SimpleNamespace(num_redundant_experts=3)
            ),
            scheduler_config=SimpleNamespace(max_num_batched_tokens=16),
            speculative_config=spec,
            compilation_config=_compilation_config(),
        )
        with set_current_vllm_config(vllm_config):
            model = qwen_model.Qwen3_8FlashNextModel(
                vllm_config=vllm_config, prefix="backbone"
            )
        return model, created, calls

    def test_last_rank_builds_embedding_layers_and_final_mixer(self, monkeypatch):
        config = _text_config(
            layer_types=["full_attention", "linear_attention", "full_attention"],
            indexer_n_heads=4,
            num_hidden_layers=3,
        )
        model, created, calls = self._build(monkeypatch, config)
        assert model.vocab_size == 32
        assert created.embed.vocab_size == 32
        assert created.embed.hidden_size == 8
        assert calls["make_layers"] == [(3, "backbone.layers")]
        assert [layer.kwargs["layer_type"] for layer in created.layer_calls] == [
            "full_attention",
            "linear_attention",
            "full_attention",
        ]
        assert [layer.kwargs["prefix"] for layer in created.layer_calls] == [
            "backbone.layers.0",
            "backbone.layers.1",
            "backbone.layers.2",
        ]
        assert model._qsa_layer_ids == frozenset({0, 2})
        assert model.num_redundant_experts == 3
        assert model.hyper_connection_mixer.role == "final"
        assert model.hyper_connection_mixer.use_combine is False

    def test_indexer_free_config_has_no_qsa_layers(self, monkeypatch):
        config = _text_config(
            layer_types=["full_attention", "full_attention"],
            indexer_n_heads=None,
            num_hidden_layers=2,
        )
        model, _, _ = self._build(monkeypatch, config)
        assert model._qsa_layer_ids == frozenset()

    def test_non_last_rank_skips_the_mixer(self, monkeypatch):
        config = _text_config(num_hidden_layers=1)
        model, created, _ = self._build(monkeypatch, config, pp_last=False)
        assert model.hyper_connection_mixer is None
        assert created.hyper == []

    def test_mtp_speculation_allocates_the_multi_stream_buffer(self, monkeypatch):
        config = _text_config(num_hidden_layers=1)
        spec = SimpleNamespace(method="mtp")
        model, _, _ = self._build(monkeypatch, config, spec=spec)
        assert model._mtp_hidden_buffer.shape == (16, 16)
        assert model._mtp_hidden_buffer.dtype is torch.bfloat16

    @pytest.mark.parametrize("spec", [None, SimpleNamespace(method="eagle")])
    def test_other_speculation_modes_allocate_no_buffer(self, monkeypatch, spec):
        config = _text_config(num_hidden_layers=1)
        model, _, _ = self._build(monkeypatch, config, spec=spec)
        assert model._mtp_hidden_buffer is None

    def test_empty_intermediate_tensors_use_the_multi_stream_width(self, monkeypatch):
        config = _text_config(num_hidden_layers=1)
        model, _, _ = self._build(monkeypatch, config)
        tensors = model.make_empty_intermediate_tensors(2, torch.float32, "cpu")
        assert tensors["hidden_states"].shape == (2, 16)
        assert tensors["hidden_states"].dtype is torch.float32


class _BackboneLayer(nn.Module):
    def __init__(self, index, *, pending_mixer=None):
        super().__init__()
        self.index = index
        self.calls = []
        self.mlp_hyper_connection = pending_mixer or _Mixer()

    def forward(self, **kwargs):
        self.calls.append(kwargs)
        hidden = kwargs["hidden_states"]
        block = torch.full((hidden.shape[0], 4), float(self.index))
        injection = torch.full((hidden.shape[0], 2), float(self.index))
        return hidden + 1, block, injection


class _FinalMixer:
    def __init__(self):
        self.calls = []

    def combine_and_mix(self, hidden_states, block_output, injection):
        self.calls.append((hidden_states, block_output, injection))
        return hidden_states, hidden_states * 2, None


def _backbone(
    *,
    first=True,
    last=True,
    layers=None,
    hc_count=2,
    hidden_size=4,
    mtp_buffer=None,
):
    model = object.__new__(qwen_model.Qwen3_8FlashNextModel)
    nn.Module.__init__(model)
    model.config = SimpleNamespace(hc_count=hc_count, hidden_size=hidden_size)
    model.embed_tokens = lambda input_ids: torch.full(
        (input_ids.shape[0], hidden_size), 7.0
    )
    model.layers = nn.ModuleList(layers or [])
    model.start_layer = 0
    model.end_layer = len(model.layers)
    model.hyper_connection_mixer = _FinalMixer() if last else None
    model._mtp_hidden_buffer = mtp_buffer
    return model


class TestModelForward:
    def _patch_pp(self, monkeypatch, *, first=True, last=True):
        monkeypatch.setattr(
            qwen_model,
            "get_pp_group",
            lambda: SimpleNamespace(is_first_rank=first, is_last_rank=last),
        )

    def test_first_rank_embeds_and_repeats_the_hidden_state(self, monkeypatch):
        self._patch_pp(monkeypatch)
        layer = _BackboneLayer(0)
        model = _backbone(layers=[layer])
        out = model.forward(
            torch.tensor([1, 2, 3]), torch.tensor([0, 1, 2])
        )
        hidden = layer.calls[0]["hidden_states"]
        assert hidden.shape == (3, 8)
        assert hidden.tolist() == torch.full((3, 8), 7.0).tolist()
        assert layer.calls[0]["positions"].tolist() == [0, 1, 2]
        assert out.tolist() == torch.full((3, 8), 16.0).tolist()

    def test_inputs_embeds_bypass_the_embedding_table(self, monkeypatch):
        self._patch_pp(monkeypatch)
        layer = _BackboneLayer(0)
        model = _backbone(layers=[layer])
        embeds = torch.arange(16, dtype=torch.float32).view(2, 8)
        model.forward(None, torch.zeros(2), inputs_embeds=embeds)
        assert torch.equal(
            layer.calls[0]["hidden_states"], embeds.repeat(1, 2)
        )

    def test_missing_inputs_raises_on_the_first_rank(self, monkeypatch):
        self._patch_pp(monkeypatch)
        model = _backbone(layers=[])
        with pytest.raises(ValueError, match="input_ids or inputs_embeds"):
            model.forward(None, torch.zeros(1))

    def test_non_first_rank_requires_and_reads_intermediate_tensors(
        self, monkeypatch
    ):
        from vllm.sequence import IntermediateTensors

        self._patch_pp(monkeypatch, first=False)
        layer = _BackboneLayer(0)
        model = _backbone(layers=[layer])
        with pytest.raises(ValueError, match="intermediate tensors"):
            model.forward(None, torch.zeros(2))
        hidden = torch.full((2, 8), 3.0)
        model.forward(
            None, torch.zeros(2), intermediate_tensors=IntermediateTensors(
                {"hidden_states": hidden}
            )
        )
        assert torch.equal(layer.calls[0]["hidden_states"], hidden)

    def test_pipeline_boundary_materializes_the_pending_mlp_block(self, monkeypatch):
        from vllm.sequence import IntermediateTensors

        self._patch_pp(monkeypatch, first=True, last=False)
        layer = _BackboneLayer(0)
        model = _backbone(layers=[layer])
        out = model.forward(torch.tensor([1, 2]), torch.zeros(2))
        assert isinstance(out, IntermediateTensors)
        # Initial hidden 7 + 1 = 8; pending combine adds 100.
        assert out["hidden_states"].tolist() == torch.full((2, 8), 108.0).tolist()
        assert layer.mlp_hyper_connection.pending_calls[0][2].tolist() == [
            [0.0, 0.0],
            [0.0, 0.0],
        ]

    def test_pending_state_is_threaded_between_layers(self, monkeypatch):
        self._patch_pp(monkeypatch)
        first = _BackboneLayer(0)
        second = _BackboneLayer(1)
        model = _backbone(layers=[first, second])
        model.forward(torch.tensor([1, 2]), torch.zeros(2))
        assert second.calls[0]["prev_block_output"].tolist() == [[0.0] * 4] * 2
        assert second.calls[0]["prev_injection"].tolist() == [[0.0] * 2] * 2
        assert first.calls[0]["prev_block_output"] is None

    def test_layer_slice_honours_pipeline_sharding(self, monkeypatch):
        self._patch_pp(monkeypatch)
        first = _BackboneLayer(0)
        middle = _BackboneLayer(1)
        last = _BackboneLayer(2)
        model = _backbone(layers=[first, middle, last])
        model.start_layer = 1
        model.end_layer = 2
        model.forward(torch.tensor([1, 2]), torch.zeros(2))
        assert first.calls == []
        assert len(middle.calls) == 1
        assert last.calls == []

    def test_deepstack_embeddings_are_expanded_and_added(self, monkeypatch):
        self._patch_pp(monkeypatch, first=True, last=False)
        layer = _BackboneLayer(0)
        model = _backbone(layers=[layer])
        deepstack = {"deepstack_input_embeds_0": torch.ones(2, 4)}
        out = model.forward(
            torch.tensor([1, 2]),
            torch.zeros(2),
            deepstack_input_embeds=deepstack,
        )
        # hidden becomes 7, expanded to 8; deepstack [2, 8] of ones is added;
        # then the pending MLP block is materialized (+100).
        assert out["hidden_states"].tolist() == torch.full((2, 8), 109.0).tolist()

    def test_mtp_buffer_captures_the_pre_mixer_hidden_state(self, monkeypatch):
        self._patch_pp(monkeypatch)
        layer = _BackboneLayer(0)
        buffer = torch.zeros(8, 8)
        model = _backbone(layers=[layer], mtp_buffer=buffer)
        out = model.forward(torch.tensor([1, 2]), torch.zeros(2))
        expected = torch.cat((torch.full((2, 4), 8.0), torch.full((2, 4), 8.0)), dim=-1)
        assert buffer[:2].tolist() == expected.tolist()
        assert buffer[2:].tolist() == torch.zeros(6, 8).tolist()
        assert out.tolist() == torch.full((2, 8), 16.0).tolist()
