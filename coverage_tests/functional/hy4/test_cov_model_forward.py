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

"""Forward and pipeline-finalisation contracts of ``HYV4Model``.

The decoder stack and token embedding are replaced at their construction
boundaries with recording stand-ins, while the pre-branch hyper connections,
the final hyper head and the closing RMSNorm are the real modules.  The
expected stream evolution, head collapse and RMSNorm are written out
independently.  Pipeline entry/exit paths (first/last rank, intermediate
tensors) are driven through a fake one-rank PP group.
"""

from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F
from torch import nn

from . import _import_shim  # noqa: F401  (idempotent; no-op on CI)
from ._construction_helpers import runtime_plan, set_current_vllm_config
from vllm_fl.models.hy4 import model as hy_model
from vllm_fl.models.hy4.config import HYV4Config
from vllm_fl.models.hy4.model import HYV4ForCausalLM

RTOL = 1e-4
ATOL = 1e-4


def _tiny_config(**overrides):
    """A dimension-shrunk HYV4Config that still validates."""
    overrides.setdefault("vocab_size", 16)
    overrides.setdefault("hidden_size", 8)
    overrides.setdefault("num_hidden_layers", 2)
    overrides.setdefault("qk_nope_head_dim", 4)
    overrides.setdefault("qk_rope_head_dim", 4)
    overrides.setdefault("qk_head_dim", 8)
    overrides.setdefault("v_head_dim", 4)
    overrides.setdefault("q_lora_rank", 4)
    overrides.setdefault("kv_lora_rank", 6)
    overrides.setdefault("n_routed_experts", 2)
    overrides.setdefault("hc_mult", 2)
    overrides.setdefault("index_topk", 8)
    overrides.setdefault("indexer_types", ["full", "full"])
    return HYV4Config(**overrides)


def _vllm_config(config):
    from vllm.config import CompilationMode

    return SimpleNamespace(
        model_config=SimpleNamespace(hf_config=config, dtype=torch.bfloat16),
        parallel_config=SimpleNamespace(pipeline_parallel_size=1),
        scheduler_config=SimpleNamespace(max_num_batched_tokens=8),
        device_config=SimpleNamespace(device=torch.device("cpu")),
        quant_config=None,
        cache_config=SimpleNamespace(cache_dtype="auto", block_size=16),
        compilation_config=SimpleNamespace(
            mode=CompilationMode.NONE,
            max_cudagraph_capture_size=0,
            cudagraph_mode=None,
            custom_ops=["none"],
            enabled_custom_ops=set(),
            disabled_custom_ops=set(),
        ),
    )


def _rms_norm_reference(x, weight, eps):
    variance = x.float().square().mean(dim=-1, keepdim=True)
    return (x.float() * torch.rsqrt(variance + eps) * weight.float()).to(x.dtype)


def _head_reference(head, streams):
    flat = streams.flatten(1).float()
    norm = torch.rsqrt(flat.square().mean(dim=-1, keepdim=True) + head.normalize_eps)
    read = (
        torch.sigmoid(
            F.linear(flat, head.hc_head_fn) * norm * head.hc_head_scale
            + head.hc_head_base
        )
        + head.hc_eps
    )
    return (read.unsqueeze(-1) * streams.float()).sum(dim=1)


class _RankGroup:
    def __init__(self, first=True, last=True):
        self.is_first_rank = first
        self.is_last_rank = last


class _RecordingEmbedding(nn.Module):
    """Token embedding stand-in with a deterministic weight table."""

    instances = []
    last_init = {}

    def __init__(self, num_embeddings, embedding_dim, quant_config=None, prefix=""):
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.quant_config = quant_config
        self.prefix = prefix
        table = torch.arange(
            num_embeddings * embedding_dim, dtype=torch.float32
        ).reshape(num_embeddings, embedding_dim)
        self.weight = nn.Parameter(table * 0.01)
        self.forward_calls = 0
        _RecordingEmbedding.instances.append(self)
        _RecordingEmbedding.last_init = {
            "num_embeddings": num_embeddings,
            "embedding_dim": embedding_dim,
            "quant_config": quant_config,
            "prefix": prefix,
        }

    def forward(self, input_ids):
        self.forward_calls += 1
        return self.weight[input_ids]


class _DeltaLayer(nn.Module):
    """Adds a fixed per-stream delta and records every incoming stream."""

    def __init__(self, delta):
        super().__init__()
        self.delta = delta
        self.inputs = []

    def forward(self, positions, streams):
        self.inputs.append(streams.clone())
        return streams + self.delta


class _Installation:
    """Patched construction boundaries for one HYV4Model build."""

    def __init__(self, monkeypatch, first=True, last=True, num_layers=2):
        self.group = _RankGroup(first, last)
        self.layers = [
            _DeltaLayer(torch.full((1, 1, 8), float(index + 1)))
            for index in range(num_layers)
        ]
        self.make_layers_prefix = None
        _RecordingEmbedding.instances = []

        def _make_layers(num_hidden_layers, layer_fn, prefix):
            self.make_layers_prefix = prefix
            return 0, num_layers, nn.ModuleList(self.layers)

        monkeypatch.setattr(hy_model, "get_pp_group", lambda: self.group)
        monkeypatch.setattr(hy_model, "prepare_hy4_runtime", lambda cfg: runtime_plan())
        monkeypatch.setattr(hy_model, "VocabParallelEmbedding", _RecordingEmbedding)
        monkeypatch.setattr(hy_model, "make_layers", _make_layers)

    def build(self, config):
        vllm_config = _vllm_config(config)
        with set_current_vllm_config(vllm_config):
            return hy_model.HYV4Model(vllm_config=vllm_config, prefix="model")


def _expected_streams(embedding_weight, input_ids, config, num_layers=2):
    embeds = embedding_weight[input_ids].float()
    streams = embeds.unsqueeze(1).expand(-1, config.hc_mult, -1).clone()
    for index in range(num_layers):
        streams = streams + float(index + 1)
    return streams


def _seed_head(model):
    torch.manual_seed(13)
    with torch.no_grad():
        torch.nn.init.normal_(model.hc_head.hc_head_fn, std=0.5)
        torch.nn.init.normal_(model.hc_head.hc_head_base, std=0.5)
        model.hc_head.hc_head_scale.fill_(0.7)
        torch.nn.init.normal_(model.norm.weight, std=0.5)


class TestHyv4ModelForward:
    """First/last-rank stream broadcast, collapse and final RMSNorm."""

    def test_first_and_last_rank_forward_matches_reference(self, monkeypatch):
        """Embedding -> 4-stream expand -> layers -> head collapse -> RMSNorm."""
        config = _tiny_config()
        installation = _Installation(monkeypatch, first=True, last=True)
        model = installation.build(config)
        _seed_head(model)

        assert _RecordingEmbedding.last_init == {
            "num_embeddings": 16,
            "embedding_dim": 8,
            "quant_config": None,
            "prefix": "model.embed_tokens",
        }
        assert installation.make_layers_prefix == "model.layers"
        assert isinstance(model.hc_head, hy_model.HYV4HyperHead)

        input_ids = torch.tensor([3, 1, 4, 1, 5])
        positions = torch.arange(5)
        out = model(input_ids=input_ids, positions=positions, intermediate_tensors=None)

        streams = _expected_streams(
            model.embed_tokens.weight.detach(), input_ids, config
        )
        expected = _rms_norm_reference(
            _head_reference(model.hc_head, streams),
            model.norm.weight,
            config.rms_norm_eps,
        )
        assert model.embed_tokens.forward_calls == 1
        assert out.shape == (5, 8)
        torch.testing.assert_close(out, expected, rtol=RTOL, atol=ATOL)
        # Each layer saw the stream state produced by its predecessor.
        torch.testing.assert_close(
            installation.layers[0].inputs[0],
            streams - 3.0,
            rtol=RTOL,
            atol=ATOL,
        )
        torch.testing.assert_close(
            installation.layers[1].inputs[0],
            streams - 2.0,
            rtol=RTOL,
            atol=ATOL,
        )

    def test_embed_input_ids_delegates_to_embedding(self, monkeypatch):
        """embed_input_ids performs a table lookup through the embedding."""
        config = _tiny_config()
        model = _Installation(monkeypatch).build(config)
        ids = torch.tensor([0, 2, 4])
        out = model.embed_input_ids(ids)
        torch.testing.assert_close(
            out, model.embed_tokens.weight.detach()[ids], rtol=0, atol=0
        )
        # Explicit table-lookup identity: each row is exactly the embedding row
        # for its id, and the embedding was called once.
        assert out.tolist() == model.embed_tokens.weight.detach()[ids].tolist()
        assert model.embed_tokens.forward_calls == 1

    def test_inputs_embeds_skips_embedding_lookup(self, monkeypatch):
        """A provided inputs_embeds tensor bypasses embed_input_ids."""
        config = _tiny_config()
        model = _Installation(monkeypatch).build(config)
        _seed_head(model)
        embeds = torch.randn(3, 8)
        out = model(
            input_ids=None,
            positions=torch.arange(3),
            intermediate_tensors=None,
            inputs_embeds=embeds,
        )
        streams = embeds.unsqueeze(1).expand(-1, config.hc_mult, -1).clone()
        streams = streams + 1.0 + 2.0
        expected = _rms_norm_reference(
            _head_reference(model.hc_head, streams),
            model.norm.weight,
            config.rms_norm_eps,
        )
        assert model.embed_tokens.forward_calls == 0
        torch.testing.assert_close(out, expected, rtol=RTOL, atol=ATOL)

    def test_missing_both_inputs_raises(self, monkeypatch):
        """First rank without input_ids or inputs_embeds fails explicitly."""
        model = _Installation(monkeypatch).build(_tiny_config())
        with pytest.raises(ValueError, match="input_ids or inputs_embeds"):
            model(
                input_ids=None,
                positions=torch.arange(2),
                intermediate_tensors=None,
            )

    def test_non_first_rank_consumes_intermediate_streams(self, monkeypatch):
        """A later PP stage rebuilds streams from the flattened intermediate."""
        config = _tiny_config()
        model = _Installation(monkeypatch, first=False, last=True).build(config)
        _seed_head(model)
        streams = torch.randn(4, config.hc_mult, 8)
        intermediate = hy_model.IntermediateTensors({"streams": streams.flatten(1)})

        out = model(
            input_ids=None,
            positions=torch.arange(4),
            intermediate_tensors=intermediate,
        )

        expected = _rms_norm_reference(
            _head_reference(model.hc_head, streams + 1.0 + 2.0),
            model.norm.weight,
            config.rms_norm_eps,
        )
        assert isinstance(model.embed_tokens, hy_model.PPMissingLayer)
        torch.testing.assert_close(out, expected, rtol=RTOL, atol=ATOL)

    def test_non_first_rank_requires_intermediate_tensors(self, monkeypatch):
        """A later PP stage without intermediates fails explicitly."""
        model = _Installation(monkeypatch, first=False, last=True).build(_tiny_config())
        with pytest.raises(ValueError, match="pipeline rank requires"):
            model(
                input_ids=None,
                positions=torch.arange(2),
                intermediate_tensors=None,
            )

    def test_non_last_rank_returns_flattened_streams(self, monkeypatch):
        """A non-final PP stage forwards flattened streams, not hidden states."""
        config = _tiny_config()
        model = _Installation(monkeypatch, first=True, last=False).build(config)
        input_ids = torch.tensor([2, 7])
        out = model(
            input_ids=input_ids,
            positions=torch.arange(2),
            intermediate_tensors=None,
        )

        assert isinstance(out, hy_model.IntermediateTensors)
        expected = _expected_streams(
            model.embed_tokens.weight.detach(), input_ids, config
        ).flatten(1)
        torch.testing.assert_close(out["streams"], expected, rtol=RTOL, atol=ATOL)
        assert isinstance(model.hc_head, hy_model.PPMissingLayer)
        assert isinstance(model.norm, hy_model.PPMissingLayer)

    def test_empty_intermediate_tensors_factory_shape(self, monkeypatch):
        """The factory allocates one flat ``streams`` buffer per token."""
        config = _tiny_config()
        model = _Installation(monkeypatch).build(config)
        tensors = model.make_empty_intermediate_tensors(
            3, torch.float32, torch.device("cpu")
        )
        assert tensors["streams"].shape == (
            3,
            config.hc_mult * config.hidden_size,
        )
        assert torch.count_nonzero(tensors["streams"]) == 0


class _ModelCallable:
    def __init__(self):
        self.args = None

    def __call__(self, *args):
        self.args = args
        return "model-output"


class _LogitsCallable:
    def __init__(self):
        self.args = None

    def __call__(self, lm_head, hidden_states):
        self.args = (lm_head, hidden_states)
        return "logits-output"


class TestForCausalLMDelegation:
    """The causal-LM wrapper forwards to the backbone and logits processor."""

    def _wrapper(self):
        wrapper = object.__new__(HYV4ForCausalLM)
        nn.Module.__init__(wrapper)
        wrapper.model = _ModelCallable()
        wrapper.logits_processor = _LogitsCallable()
        wrapper.lm_head = "lm-head"
        return wrapper

    def test_forward_passes_all_arguments_positionally(self):
        """forward forwards ids/positions/intermediates/embeds unchanged."""
        wrapper = self._wrapper()
        ids = torch.tensor([1, 2])
        positions = torch.arange(2)
        intermediate = hy_model.IntermediateTensors({})
        embeds = torch.randn(2, 8)
        out = HYV4ForCausalLM.forward(
            wrapper, ids, positions, intermediate, inputs_embeds=embeds
        )
        assert out == "model-output"
        assert wrapper.model.args == (ids, positions, intermediate, embeds)

    def test_compute_logits_delegates_to_processor(self):
        """compute_logits hands the LM head and hidden states to the processor."""
        wrapper = self._wrapper()
        hidden = torch.randn(2, 8)
        out = HYV4ForCausalLM.compute_logits(wrapper, hidden)
        assert out == "logits-output"
        assert wrapper.logits_processor.args == ("lm-head", hidden)
