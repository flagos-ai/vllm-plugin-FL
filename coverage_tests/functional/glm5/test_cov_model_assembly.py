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

"""Assembly contracts of the GLM5-Next MLA attention, model and wrapper.

The top-level constructors pull in heavy vLLM layers, so each test replaces
only that upstream boundary with a recording double and asserts the plugin's
own wiring: kpool cache replacement, zero-RoPE forward rebinding, pipeline
rank branching, the kpool top-k buffer geometry, and the first-layer mHC
broadcast weight finalization.  Expected broadcast weights are recomputed
from the stream-group definition with plain tensor math.
"""

import sys
import types
from types import SimpleNamespace

import pytest
import torch
from torch import nn

from vllm.config import CompilationMode, set_current_vllm_config
from vllm.model_executor.models.utils import PPMissingLayer

from . import _import_shim  # noqa: F401  (idempotent; no-op on CI)
from vllm_fl.models.glm5 import model as glm5_model, runtime as glm5_runtime
from vllm_fl.models.glm5.config import Glm5NextTextConfig


class _RecordingModule(nn.Module):
    """``nn.Module`` double that records constructor arguments."""

    def __init__(self, *args, **kwargs):
        super().__init__()
        self.args = args
        self.kwargs = kwargs


def _model_config(rope_dim=0, compress=True):
    return SimpleNamespace(
        hidden_size=8,
        num_attention_heads=4,
        qk_nope_head_dim=4,
        qk_rope_head_dim=rope_dim,
        v_head_dim=4,
        q_lora_rank=4,
        kv_lora_rank=6,
        max_position_embeddings=64,
        index_kpool=4,
        index_kpool_compress=compress,
    )


def _indexer_double():
    indexer = SimpleNamespace(
        head_dim=128,
        quant_block_size=128,
        scale_fmt="ue8m0",
        topk_tokens=8,
        max_model_len=64,
        max_total_seq_len=128,
        topk_indices_buffer=torch.zeros(1, dtype=torch.int32),
        prefix="model.layers.1.self_attn.indexer",
        k_cache=SimpleNamespace(prefix="model.layers.1.self_attn.indexer.k_cache"),
    )
    indexer.forward = lambda *args, **kwargs: "stock-forward"
    return indexer


def _install_mla_doubles(monkeypatch, indexer, current_platform=None):
    state = {"indexer": indexer}

    def fake_base_init(self, *args, **kwargs):
        config = kwargs["config"]
        vllm_config = kwargs["vllm_config"]
        self.qk_rope_head_dim = config.qk_rope_head_dim
        self.mla_attn = SimpleNamespace(
            rotary_emb="rope", indexer_rope_emb="indexer_rope"
        )
        self.indexer = state["indexer"]
        if self.indexer is not None:
            static_ctx = vllm_config.compilation_config.static_forward_context
            static_ctx[self.indexer.k_cache.prefix] = self.indexer.k_cache

    monkeypatch.setattr(glm5_model.DeepseekV2MLAAttention, "__init__", fake_base_init)
    if current_platform is None:
        current_platform = SimpleNamespace(
            is_cuda=lambda: False, supports_fp8=lambda: False
        )
    monkeypatch.setattr(glm5_model, "current_platform", current_platform)
    monkeypatch.setattr(glm5_model, "Glm5NextIndexerCache", _RecordingModule)
    monkeypatch.setattr(glm5_model, "Glm5NextTailCache", _RecordingModule)
    monkeypatch.setattr(glm5_model, "SparseAttnIndexerKpool", _RecordingModule)
    return state


def _build_mla(monkeypatch, config=None, indexer="default", current_platform=None):
    config = config or _model_config()
    if indexer == "default":
        indexer = _indexer_double()
    state = _install_mla_doubles(monkeypatch, indexer, current_platform)
    vllm_config = SimpleNamespace(
        compilation_config=SimpleNamespace(static_forward_context={})
    )
    attn = glm5_model.Glm5NextMLAAttention(
        vllm_config=vllm_config,
        config=config,
        hidden_size=8,
        num_heads=4,
        qk_nope_head_dim=4,
        qk_rope_head_dim=config.qk_rope_head_dim,
        v_head_dim=4,
        q_lora_rank=4,
        kv_lora_rank=6,
        max_position_embeddings=64,
        cache_config=object(),
        quant_config=None,
        prefix="model.layers.1.self_attn",
        topk_indices_buffer=torch.zeros(2, dtype=torch.int32),
    )
    return attn, vllm_config, state


class TestGlm5NextMLAAttentionConstruction:
    """kpool cache replacement and the zero-RoPE forward fast path."""

    def test_kpool_caches_replace_the_stock_indexer_cache(self, monkeypatch):
        """The per-token cache becomes a kpool cache plus a tail cache."""
        attn, vllm_config, _ = _build_mla(monkeypatch)
        indexer = attn.indexer
        static_ctx = vllm_config.compilation_config.static_forward_context
        old_prefix = "model.layers.1.self_attn.indexer.k_cache"
        assert old_prefix not in static_ctx

        assert indexer.index_kpool == 4
        ape = indexer.index_kpool_compress_ape
        assert isinstance(ape, nn.Parameter)
        assert ape.shape == (4, 128)
        assert ape.dtype == torch.float32
        assert torch.count_nonzero(ape) == 0
        gate = indexer.index_kpool_compress_gate
        assert isinstance(gate, nn.Parameter)
        assert gate.shape == (128, 8)
        assert gate.dtype == torch.bfloat16

        cache = indexer.k_cache
        assert isinstance(cache, _RecordingModule)
        assert cache.kwargs["head_dim"] == 128
        assert cache.kwargs["dtype"] is torch.bfloat16
        assert cache.kwargs["prefix"] == old_prefix
        assert cache.kwargs["index_kpool"] == 4

        tail = indexer.tail_cache
        assert tail.kwargs["head_dim"] == 128
        assert tail.kwargs["dtype"] is torch.bfloat16
        assert tail.kwargs["prefix"] == ("model.layers.1.self_attn.indexer.tail_cache")

        op = indexer.indexer_op
        assert op.args[0] is cache
        assert op.args[1:7] == (128, "ue8m0", 8, 128, 64, 128)
        assert op.args[7] is indexer.topk_indices_buffer
        assert op.kwargs["tail_cache"] is tail

    def test_fp8_cache_widens_rows_with_scale_bytes(self, monkeypatch):
        """A fp8-capable platform allocates uint8 rows scaled by 4 bytes."""
        platform = SimpleNamespace(is_cuda=lambda: False, supports_fp8=lambda: True)
        attn, _, _ = _build_mla(monkeypatch, current_platform=platform)
        cache = attn.indexer.k_cache
        # 128 fp8 values + one fp32 scale per 128-value block.
        assert cache.kwargs["head_dim"] == 128 + 128 // 128 * 4
        assert cache.kwargs["dtype"] is torch.uint8

    def test_disabled_compression_keeps_the_stock_cache(self, monkeypatch):
        """index_kpool_compress=False leaves the base indexer untouched."""
        attn, vllm_config, _ = _build_mla(
            monkeypatch, config=_model_config(compress=False)
        )
        indexer = attn.indexer
        static_ctx = vllm_config.compilation_config.static_forward_context
        assert static_ctx[indexer.k_cache.prefix] is indexer.k_cache
        assert not hasattr(indexer, "index_kpool")
        assert not hasattr(indexer, "indexer_op")

    def test_absent_indexer_is_tolerated(self, monkeypatch):
        """No indexer means no kpool replacement and no error."""
        attn, _, _ = _build_mla(monkeypatch, indexer=None)
        assert attn.indexer is None
        assert attn.mla_attn.rotary_emb is None

    def test_zero_rope_rebinds_indexer_forward(self, monkeypatch):
        """Zero RoPE drops both rotary embeddings and swaps the forward."""
        attn, _, _ = _build_mla(monkeypatch, config=_model_config(rope_dim=0))
        assert attn.mla_attn.rotary_emb is None
        assert attn.mla_attn.indexer_rope_emb is None
        assert attn.indexer.forward.__func__ is glm5_model._indexer_forward_nope

    def test_positive_rope_keeps_native_forward(self, monkeypatch):
        """A RoPE checkpoint keeps the stock indexer forward and embeddings."""
        attn, _, _ = _build_mla(monkeypatch, config=_model_config(rope_dim=64))
        assert attn.mla_attn.rotary_emb == "rope"
        assert attn.mla_attn.indexer_rope_emb == "indexer_rope"
        assert attn.indexer.forward() == "stock-forward"

    @pytest.mark.gpu
    def test_cuda_construction_patches_are_scoped(self, monkeypatch):
        """CUDA construction exposes DeepGEMM/prefill hooks only until return."""
        from vllm.model_executor.layers import sparse_attn_indexer
        from vllm.model_executor.layers.attention import mla_attention

        from vllm_fl.dispatch.backends.flaggems.impl.mla import (
            FlagGemsMLAPrefillBackend,
        )

        pristine_has_deep_gemm = sparse_attn_indexer.has_deep_gemm
        pristine_prefill = mla_attention.get_mla_prefill_backend
        observed = {}

        def checking_base_init(self, *args, **kwargs):
            observed["deep_gemm"] = sparse_attn_indexer.has_deep_gemm()
            observed["prefill"] = mla_attention.get_mla_prefill_backend(None)
            # Mimic the minimum state the plugin constructor reads.
            self.qk_rope_head_dim = kwargs["config"].qk_rope_head_dim
            self.mla_attn = SimpleNamespace(
                rotary_emb="rope", indexer_rope_emb="indexer_rope"
            )
            self.indexer = None

        monkeypatch.setattr(
            glm5_model.DeepseekV2MLAAttention,
            "__init__",
            checking_base_init,
        )
        monkeypatch.setattr(
            glm5_model,
            "current_platform",
            SimpleNamespace(is_cuda=lambda: True, supports_fp8=lambda: False),
        )
        monkeypatch.setattr(glm5_model, "use_nvidia_reference", lambda: False)
        monkeypatch.setattr(glm5_model, "Glm5NextIndexerCache", _RecordingModule)
        monkeypatch.setattr(glm5_model, "Glm5NextTailCache", _RecordingModule)
        monkeypatch.setattr(glm5_model, "SparseAttnIndexerKpool", _RecordingModule)
        vllm_config = SimpleNamespace(
            compilation_config=SimpleNamespace(static_forward_context={})
        )
        glm5_model.Glm5NextMLAAttention(
            vllm_config=vllm_config,
            config=_model_config(),
            cache_config=object(),
            prefix="model.layers.1.self_attn",
        )
        assert observed["deep_gemm"] is True
        assert observed["prefill"] is FlagGemsMLAPrefillBackend
        # temporary_patches restores the pristine module attributes.
        assert sparse_attn_indexer.has_deep_gemm is pristine_has_deep_gemm
        assert mla_attention.get_mla_prefill_backend is pristine_prefill


class _RecordingEmbedding(nn.Module):
    def __init__(self, vocab_size, hidden_size, quant_config=None, prefix=""):
        super().__init__()
        self.args = (vocab_size, hidden_size, quant_config, prefix)
        self.weight = nn.Parameter(torch.zeros(vocab_size, hidden_size))


class _RecordingLayer(nn.Module):
    def __init__(self, vllm_config, layer_idx, prefix, buffer):
        super().__init__()
        self.vllm_config = vllm_config
        self.layer_idx = layer_idx
        self.prefix = prefix
        self.buffer = buffer


def _model_namespace(pp_first=True, pp_last=True, world_size=2, num_batched=16):
    config = Glm5NextTextConfig(
        vocab_size=32,
        hidden_size=8,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=4,
        rms_norm_eps=1e-6,
        index_topk=8,
        index_kpool=4,
        layer_types=["linear_attention", "deepseek_sparse_attention"],
        mlp_layer_types=["dense", "sparse"],
    )
    ns = SimpleNamespace(
        model_config=SimpleNamespace(hf_text_config=config),
        scheduler_config=SimpleNamespace(max_num_batched_tokens=num_batched),
        parallel_config=SimpleNamespace(
            eplb_config=SimpleNamespace(num_redundant_experts=3)
        ),
        compilation_config=SimpleNamespace(
            mode=CompilationMode.NONE, static_forward_context={}
        ),
        quant_config=None,
    )
    pp = SimpleNamespace(is_first_rank=pp_first, is_last_rank=pp_last)
    return ns, config, pp


def _install_model_doubles(monkeypatch, pp, world_size=2):
    monkeypatch.setattr(glm5_model, "get_pp_group", lambda: pp)
    monkeypatch.setattr(
        glm5_model, "get_tensor_model_parallel_world_size", lambda: world_size
    )
    monkeypatch.setattr(
        glm5_model,
        "current_platform",
        SimpleNamespace(device_type="cpu"),
    )
    monkeypatch.setattr(glm5_model, "VocabParallelEmbedding", _RecordingEmbedding)
    monkeypatch.setattr(glm5_model, "RMSNorm", _RecordingModule)
    monkeypatch.setattr(glm5_model, "Glm5NextDecoderLayer", _RecordingLayer)
    made = {}

    def fake_make_layers(num_hidden_layers, layer_fn, prefix):
        made["args"] = (num_hidden_layers, prefix)
        layers = nn.ModuleList(
            [layer_fn(prefix=f"{prefix}.{idx}") for idx in range(num_hidden_layers)]
        )
        return 0, num_hidden_layers, layers

    monkeypatch.setattr(glm5_model, "make_layers", fake_make_layers)
    return made


class TestGlm5NextModelConstruction:
    """Pipeline-rank branching, kpool top-k geometry and layer assembly."""

    def test_first_and_last_rank_build_embedding_norm_and_layers(self, monkeypatch):
        """The top-k buffer width is aligned to 128 for sparse MLA."""
        ns, config, pp = _model_namespace()
        made = _install_model_doubles(monkeypatch, pp)
        with set_current_vllm_config(ns):
            model = glm5_model.Glm5NextModel(vllm_config=ns, prefix="model")

        assert model.config is config
        assert model.vocab_size == 32
        assert model.use_mha is False
        assert model.num_redundant_experts == 3
        assert model.start_layer == 0 and model.end_layer == 2
        assert made["args"] == (2, "model.layers")
        assert len(model.layers) == 2
        for idx, layer in enumerate(model.layers):
            assert isinstance(layer, _RecordingLayer)
            assert layer.layer_idx == idx
            assert layer.prefix == f"model.layers.{idx}"
        # 8 top-k + 4 kpool slots widen to the next multiple of 128.
        tensor = model.layers[0].buffer
        assert tensor.shape == (16, 128)
        assert tensor.dtype == torch.int32
        assert all(layer.buffer is tensor for layer in model.layers)

        embed = model.embed_tokens
        assert isinstance(embed, _RecordingEmbedding)
        assert embed.args == (32, 8, None, "model.embed_tokens")
        assert isinstance(model.norm, _RecordingModule)
        assert model.norm.args == (8,)
        assert model.norm.kwargs == {"eps": 1e-6}
        assert callable(model.make_empty_intermediate_tensors)
        # The compile decorator must not engage compilation for the test.
        assert model.do_not_compile is True

    def test_middle_rank_uses_pipeline_placeholders(self, monkeypatch):
        """A middle pipeline rank holds PPMissingLayer for embed and norm."""
        ns, _, pp = _model_namespace(pp_first=False, pp_last=False)
        _install_model_doubles(monkeypatch, pp)
        with set_current_vllm_config(ns):
            model = glm5_model.Glm5NextModel(vllm_config=ns, prefix="model")
        assert isinstance(model.embed_tokens, PPMissingLayer)
        assert isinstance(model.norm, PPMissingLayer)

    def test_head_count_must_divide_the_tensor_parallel_world(self, monkeypatch):
        """Head partitioning is validated at construction time."""
        ns, _, pp = _model_namespace(world_size=3)
        _install_model_doubles(monkeypatch, pp, world_size=3)
        with (
            set_current_vllm_config(ns),
            pytest.raises(AssertionError),
        ):
            glm5_model.Glm5NextModel(vllm_config=ns, prefix="model")


class _EmbeddingDouble(nn.Module):
    def __init__(self):
        super().__init__()
        self.calls = []

    def forward(self, input_ids):
        self.calls.append(input_ids.clone())
        return input_ids.float().unsqueeze(-1) + 1


class _FinalNorm(nn.Module):
    def forward(self, hidden_states):
        return hidden_states + 100


class _ChainLayer(nn.Module):
    def __init__(self, increment):
        super().__init__()
        self.increment = increment
        self.calls = []

    def forward(self, positions, hidden_states, residual, post, comb):
        self.calls.append((positions, hidden_states, residual, post, comb))
        residual_out = (
            residual + self.increment
            if isinstance(residual, torch.Tensor)
            else residual
        )
        return hidden_states + self.increment, residual_out, post, comb


def _forward_model(first=True, last=True):
    model = glm5_model.Glm5NextModel.__new__(glm5_model.Glm5NextModel)
    nn.Module.__init__(model)
    model.start_layer = 0
    model.end_layer = 2
    model.embed_tokens = _EmbeddingDouble()
    model.layers = nn.ModuleList([_ChainLayer(1), _ChainLayer(10)])
    model.norm = _FinalNorm()
    return model


class TestGlm5NextModelForward:
    """Embedding/intermediate selection and pipeline output branching."""

    def test_first_rank_embeds_then_runs_all_layers(self, monkeypatch):
        """First rank embeds ids; the last rank applies the final norm."""
        model = _forward_model()
        monkeypatch.setattr(
            glm5_model,
            "get_pp_group",
            lambda: SimpleNamespace(is_first_rank=True, is_last_rank=True),
        )
        input_ids = torch.tensor([1, 2, 3])
        positions = torch.arange(3)
        out = glm5_model.Glm5NextModel.forward(model, input_ids, positions, None)
        embedded = input_ids.float().unsqueeze(-1) + 1
        # Two layers add 1 and 10, then the final norm adds 100.
        expected = embedded + 11 + 100
        assert torch.equal(out, expected)
        assert torch.equal(model.embed_tokens.calls[0], input_ids)
        assert torch.equal(model.layers[1].calls[0][1], embedded + 1)

    def test_inputs_embeds_bypass_the_embedding(self, monkeypatch):
        """Precomputed embeddings skip the embedding lookup."""
        model = _forward_model()
        monkeypatch.setattr(
            glm5_model,
            "get_pp_group",
            lambda: SimpleNamespace(is_first_rank=True, is_last_rank=True),
        )
        inputs_embeds = torch.ones(3, 1)
        out = glm5_model.Glm5NextModel.forward(
            model, None, torch.arange(3), None, inputs_embeds=inputs_embeds
        )
        assert model.embed_tokens.calls == []
        assert torch.equal(out, inputs_embeds + 11 + 100)

    def test_non_first_rank_reads_intermediate_tensors(self, monkeypatch):
        """Middle ranks consume hidden states and residual from the pipe."""
        model = _forward_model(first=False, last=False)
        monkeypatch.setattr(
            glm5_model,
            "get_pp_group",
            lambda: SimpleNamespace(is_first_rank=False, is_last_rank=False),
        )
        hidden = torch.ones(3, 1)
        residual = torch.zeros(3, 1)
        intermediate = glm5_model.IntermediateTensors(
            {"hidden_states": hidden, "residual": residual}
        )
        positions = torch.arange(3)
        out = glm5_model.Glm5NextModel.forward(model, None, positions, intermediate)
        assert model.embed_tokens.calls == []
        assert isinstance(out, glm5_model.IntermediateTensors)
        assert torch.equal(out["hidden_states"], hidden + 11)
        assert torch.equal(out["residual"], residual + 11)

    def test_non_first_rank_requires_intermediate_tensors(self, monkeypatch):
        """A missing pipeline input is rejected rather than silently zeroed."""
        model = _forward_model(first=False, last=False)
        monkeypatch.setattr(
            glm5_model,
            "get_pp_group",
            lambda: SimpleNamespace(is_first_rank=False, is_last_rank=False),
        )
        with pytest.raises(AssertionError):
            glm5_model.Glm5NextModel.forward(model, None, torch.arange(3), None)


class TestFinalizeMhcBroadcastWeights:
    """First-layer stream-summed projection and its capability gates."""

    @staticmethod
    def _model(hidden_size=2, streams=3):
        layer = glm5_model.Glm5NextDecoderLayer.__new__(glm5_model.Glm5NextDecoderLayer)
        nn.Module.__init__(layer)
        layer.layer_idx = 0
        layer.n = streams
        mix_hc = (2 + streams) * streams
        weight = torch.arange(mix_hc * streams * hidden_size, dtype=torch.float32).view(
            mix_hc, streams * hidden_size
        )
        layer.hc_attn_fn = nn.Parameter(weight.clone())
        layer.hc_attn_fn_broadcast = None
        model = glm5_model.Glm5NextModel.__new__(glm5_model.Glm5NextModel)
        nn.Module.__init__(model)
        model.layers = nn.ModuleList([layer])
        model.start_layer = 0
        model.end_layer = 1
        model.config = SimpleNamespace(hidden_size=hidden_size)
        return model, layer, weight

    @staticmethod
    def _patch_probes(
        monkeypatch, first_rank=True, nvidia=True, deep_gemm_supported=True
    ):
        monkeypatch.setattr(
            glm5_model,
            "get_pp_group",
            lambda: SimpleNamespace(is_first_rank=first_rank),
        )
        monkeypatch.setattr(glm5_model, "use_nvidia_reference", lambda: nvidia)
        from vllm.utils import deep_gemm as deep_gemm_module

        monkeypatch.setattr(
            deep_gemm_module,
            "is_deep_gemm_supported",
            lambda: deep_gemm_supported,
        )

    def test_supported_path_sums_stream_weights(self, monkeypatch):
        """The broadcast weight is the per-stream sum of hc_attn_fn rows."""
        model, layer, weight = self._model(hidden_size=2, streams=3)
        self._patch_probes(monkeypatch)
        monkeypatch.setattr(glm5_model, "mhc_pre_broadcast_tilelang", None)
        model.finalize_mhc_broadcast_weights()

        # hc_attn_fn rows are [stream, hidden] groups; summing the stream
        # component of each output row yields the broadcast weight.
        expected = weight.view(-1, 3, 2).sum(dim=1)
        broadcast = layer.hc_attn_fn_broadcast
        assert broadcast.shape == (15, 2)
        assert torch.equal(broadcast, expected)
        assert broadcast.is_contiguous()
        assert broadcast.requires_grad is False

    def test_missing_tilelang_still_builds_the_materialized_weight(self, monkeypatch):
        """Without TileLang the weight is prepared but no preflight runs."""
        model, layer, weight = self._model()
        self._patch_probes(monkeypatch)
        monkeypatch.setattr(glm5_model, "mhc_pre_broadcast_tilelang", None)
        model.finalize_mhc_broadcast_weights()
        expected = weight.view(-1, 3, 2).sum(dim=1)
        assert torch.equal(layer.hc_attn_fn_broadcast, expected)

    def test_tilelang_preflight_runs_before_exposing_the_weight(self, monkeypatch):
        """A present TileLang kernel is preflighted once."""
        model, layer, weight = self._model()
        self._patch_probes(monkeypatch)
        monkeypatch.setattr(glm5_model, "mhc_pre_broadcast_tilelang", object())
        calls = []
        fake_module = types.ModuleType(
            "vllm_fl.dispatch.backends.tilelang.mhc_broadcast"
        )
        fake_module.preflight_mhc_broadcast = lambda: calls.append("preflight")
        monkeypatch.setitem(
            sys.modules,
            "vllm_fl.dispatch.backends.tilelang.mhc_broadcast",
            fake_module,
        )
        model.finalize_mhc_broadcast_weights()
        assert calls == ["preflight"]
        assert torch.equal(
            layer.hc_attn_fn_broadcast,
            weight.view(-1, 3, 2).sum(dim=1),
        )

    @pytest.mark.parametrize(
        "first_rank,nvidia,deep_gemm",
        [
            (False, True, True),
            (True, False, True),
            (True, True, False),
        ],
    )
    def test_capability_gates_leave_broadcast_unset(
        self, monkeypatch, first_rank, nvidia, deep_gemm
    ):
        """Pipeline rank, provider and DeepGEMM each gate the specialization."""
        model, layer, _ = self._model()
        self._patch_probes(
            monkeypatch,
            first_rank=first_rank,
            nvidia=nvidia,
            deep_gemm_supported=deep_gemm,
        )
        monkeypatch.setattr(glm5_model, "mhc_pre_broadcast_tilelang", None)
        model.finalize_mhc_broadcast_weights()
        assert layer.hc_attn_fn_broadcast is None

    def test_empty_pipeline_shard_is_a_no_op(self, monkeypatch):
        """A rank with no local layers returns before probing."""
        model, layer, _ = self._model()
        model.start_layer = 1
        self._patch_probes(monkeypatch)
        monkeypatch.setattr(glm5_model, "mhc_pre_broadcast_tilelang", None)
        model.finalize_mhc_broadcast_weights()
        assert layer.hc_attn_fn_broadcast is None


class TestGlm5NextForCausalLMConstruction:
    """Startup validation, preflight and last-rank head construction."""

    @staticmethod
    def _build(monkeypatch, last_rank):
        ns, config, _ = _model_namespace()
        pp = SimpleNamespace(is_first_rank=True, is_last_rank=last_rank)
        monkeypatch.setattr(glm5_model, "get_pp_group", lambda: pp)
        validated = []
        monkeypatch.setattr(
            glm5_runtime,
            "validate_glm5_config",
            lambda cfg: validated.append(cfg),
        )
        preflight = []
        monkeypatch.setattr(
            glm5_model,
            "INDEXER_BACKEND",
            SimpleNamespace(preflight=lambda: preflight.append("preflight")),
        )
        prefix_calls = []
        monkeypatch.setattr(
            glm5_model,
            "maybe_prefix",
            lambda prefix, name: (
                prefix_calls.append((prefix, name))
                or (f"{prefix}.{name}" if prefix else name)
            ),
        )
        monkeypatch.setattr(glm5_model, "Glm5NextModel", _RecordingModule)
        monkeypatch.setattr(glm5_model, "ParallelLMHead", _RecordingModule)
        monkeypatch.setattr(glm5_model, "LogitsProcessor", _RecordingModule)
        model = glm5_model.Glm5NextForCausalLM(vllm_config=ns, prefix="base")
        return model, ns, config, validated, preflight, prefix_calls

    def test_last_rank_builds_model_head_and_logits_processor(self, monkeypatch):
        """The wrapper validates, preflights and owns the LM head."""
        model, ns, config, validated, preflight, prefix_calls = self._build(
            monkeypatch, last_rank=True
        )
        assert validated == [ns]
        assert preflight == ["preflight"]
        assert model.config is config
        assert model.model_config is ns.model_config
        assert model.vllm_config is ns
        assert isinstance(model.model, _RecordingModule)
        assert model.model.kwargs == {
            "vllm_config": ns,
            "prefix": "base.model",
        }
        assert isinstance(model.lm_head, _RecordingModule)
        assert model.lm_head.args == (32, 8)
        assert model.lm_head.kwargs == {
            "quant_config": None,
            "prefix": "base.lm_head",
        }
        assert isinstance(model.logits_processor, _RecordingModule)
        assert model.logits_processor.args == (32,)
        assert prefix_calls == [
            ("base", "model"),
            ("base", "lm_head"),
        ]

    def test_non_last_rank_uses_pipeline_placeholder_head(self, monkeypatch):
        """A non-last rank has no LM head but still runs validation."""
        model, _, _, validated, preflight, _ = self._build(monkeypatch, last_rank=False)
        assert len(validated) == 1
        assert preflight == ["preflight"]
        assert isinstance(model.lm_head, PPMissingLayer)
