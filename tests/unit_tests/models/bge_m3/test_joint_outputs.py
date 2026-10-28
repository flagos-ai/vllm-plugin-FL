# Copyright (c) 2026 BAAI. All rights reserved.
from concurrent.futures import ThreadPoolExecutor
from threading import Barrier
from types import SimpleNamespace

import pytest
import torch
from pydantic import ValidationError
from torch import nn
from torch.nn import functional as F

from vllm.model_executor.layers.pooler.activations import PoolerNormalize
from vllm.model_executor.layers.pooler.seqwise.heads import EmbeddingPoolerHead
from vllm.model_executor.layers.pooler.tokwise.heads import (
    TokenClassifierPoolerHead,
    TokenEmbeddingPoolerHead,
)
from vllm.v1.pool.metadata import PoolingCursor, PoolingMetadata, PoolingStates

from vllm_fl.models.bge_m3.io_processor import BgeM3IOProcessor
from vllm_fl.models.bge_m3.pooling import BgeM3JointPooler


class Tokenizer:
    cls_token_id, pad_token_id, eos_token_id, unk_token_id = 0, 1, 2, 3

    def __call__(self, texts, **kwargs):
        self.kwargs = kwargs
        return {"input_ids": [[0, 10, 2] if text else [0, 2] for text in texts]}


@pytest.fixture
def processor():
    tokenizer = Tokenizer()
    config = SimpleNamespace(
        model_config=SimpleNamespace(
            hf_config=SimpleNamespace(hidden_size=4), max_model_len=8192
        )
    )
    return BgeM3IOProcessor(config, SimpleNamespace(get_tokenizer=lambda: tokenizer))


@pytest.fixture
def heads(monkeypatch):
    from vllm.model_executor.layers.pooler.tokwise import methods

    monkeypatch.setattr(
        methods,
        "get_current_vllm_config",
        lambda: SimpleNamespace(
            scheduler_config=SimpleNamespace(enable_chunked_prefill=False)
        ),
    )
    torch.manual_seed(17)
    sparse, colbert = nn.Linear(4, 1), nn.Linear(4, 4)
    dense_head = EmbeddingPoolerHead(activation=PoolerNormalize())
    sparse_head = TokenClassifierPoolerHead(classifier=sparse, activation=torch.relu)
    colbert_head = TokenEmbeddingPoolerHead(
        projector=colbert, activation=PoolerNormalize()
    )
    native = {
        "embed": SimpleNamespace(head=dense_head),
        "token_classify": SimpleNamespace(pooler=SimpleNamespace(head=sparse_head)),
        "token_embed": SimpleNamespace(pooler=SimpleNamespace(head=colbert_head)),
    }
    return BgeM3JointPooler(native), sparse, colbert


def metadata(ids, params):
    lengths = torch.tensor([len(x) for x in ids])
    starts = lengths.cumsum(0) - lengths
    padded = torch.zeros(len(ids), int(lengths.max()), dtype=torch.long)
    for row, tokens in zip(padded, ids):
        row[: len(tokens)] = torch.tensor(tokens)
    cursor = PoolingCursor(starts, starts + lengths - 1, lengths, lengths, lengths)
    return PoolingMetadata(
        lengths, padded, padded, params, [PoolingStates() for _ in ids], cursor
    )


def request(mask, sentences):
    return {
        "sentences": sentences,
        "return_dense": bool(mask & 1),
        "return_sparse": bool(mask & 2),
        "return_colbert_vecs": bool(mask & 4),
    }


def output(ids, tensor):
    return SimpleNamespace(prompt_token_ids=ids, outputs=SimpleNamespace(data=tensor))


@pytest.mark.parametrize("mask", range(8))
@pytest.mark.parametrize("single", [False, True])
def test_selected_heads_match_independent_math(processor, heads, mask, single):
    pooler, sparse, colbert = heads
    prompt = processor.parse_data(request(mask, "a" if single else ["a", ""]))
    processor.pre_process(prompt)
    ids = [[0, 10, 10, 3, 1, 2]] if single else [[0, 10, 10, 3, 1, 2], [0, 2]]
    params = [processor.merge_pooling_params() for _ in ids]
    hidden = torch.randn(sum(map(len, ids)), 4)
    calls = {"sparse": 0, "colbert": 0}
    hooks = []
    for name, head in [("sparse", sparse), ("colbert", colbert)]:

        def count(module, args, result, name=name):
            calls[name] += 1

        hooks.append(head.register_forward_hook(count))
    encoded = pooler(hidden, metadata(ids, params))
    for hook in hooks:
        hook.remove()
    assert calls == {
        "sparse": len(ids) if mask & 2 else 0,
        "colbert": len(ids) if mask & 4 else 0,
    }
    result = processor.post_process(
        [output(tokens, t) for tokens, t in zip(ids, encoded)]
    )
    chunks = hidden.split([len(x) for x in ids])
    for key, bit in [("dense_vecs", 1), ("lexical_weights", 2), ("colbert_vecs", 4)]:
        if not mask & bit:
            assert result[key] is None
            continue
        values = [result[key]] if single else result[key]
        assert len(values) == len(ids)
        for tokens, h, actual in zip(ids, chunks, values):
            if bit == 1:
                torch.testing.assert_close(
                    torch.tensor(actual), F.normalize(h[0], dim=0)
                )
            elif bit == 4:
                expected = F.normalize(
                    F.linear(h, colbert.weight, colbert.bias)[1:], dim=-1
                )
                torch.testing.assert_close(torch.tensor(actual), expected)
                assert len(actual) == len(tokens) - 1  # includes EOS and literal PAD
            else:
                weights = F.relu(F.linear(h, sparse.weight, sparse.bias)).flatten()
                expected = {}
                for token, weight in zip(tokens, weights.tolist()):
                    if token not in {0, 1, 2, 3} and weight > 0:
                        expected[str(token)] = max(expected.get(str(token), 0), weight)
                assert actual == expected


def test_defaults_alias_and_truncation_parameters(processor):
    prompt = processor.parse_data({"input": "", "max_length": 16})
    assert prompt.output_mask == 1
    assert processor.pre_process(prompt) == [{"prompt_token_ids": [0, 2]}]
    assert processor.tokenizer.kwargs == dict(
        padding=False, truncation=True, max_length=16, add_special_tokens=True
    )
    assert (
        processor.parse_data({"sentences": "a", "return_dense": None}).output_mask == 1
    )


@pytest.mark.parametrize(
    "data",
    [
        {"sentences": []},
        {"sentences": [1]},
        {"sentences": 1},
        {"sentences": "a", "return_sparse": "true"},
        {"sentences": "a", "max_length": 8193},
        {"sentences": "a", "max_length": 1},
        {"sentences": "a", "unknown_option": True},
    ],
)
def test_invalid_inputs_are_rejected(processor, data):
    with pytest.raises(ValidationError):
        processor.parse_data(data)


def test_concurrent_preprocessing_has_no_fifo_state(processor):
    barrier = Barrier(8)

    def process(mask):
        prompt = processor.parse_data(request(mask, "x" if mask % 2 else ["x"]))
        processor.pre_process(prompt, request_id=str(mask))
        barrier.wait(timeout=10)
        params = processor.merge_pooling_params()
        return params.extra_kwargs["bge_m3"]

    with ThreadPoolExecutor(max_workers=8) as executor:
        values = list(executor.map(process, range(8)))
    assert values == [{"outputs": m, "single_input": bool(m % 2)} for m in range(8)]
    assert not hasattr(processor, "online_requests")


def test_postprocessing_does_not_depend_on_latest_request(processor):
    processor.pre_process(processor.parse_data(request(0, "a")))
    p = processor.merge_pooling_params()
    processor.pre_process(processor.parse_data(request(7, ["b"])))
    assert p.extra_kwargs["bge_m3"]["outputs"] == 0
    assert processor.post_process([output([0, 2], torch.tensor([0.0, 1.0]))]) == {
        "dense_vecs": None,
        "lexical_weights": None,
        "colbert_vecs": None,
    }


def test_corrupt_transport_is_rejected(processor):
    with pytest.raises(ValueError):
        processor.post_process([output([0, 2], torch.tensor([1.0, 0.0, 2.0]))])


def test_mixed_selections_in_one_worker_batch(processor, heads):
    pooler, _, _ = heads
    ids = [[0, 10, 2], [0, 2], [0, 10, 10, 2]]
    params = []
    for mask in [1, 4, 7]:
        processor.pre_process(processor.parse_data(request(mask, "one")))
        params.append(processor.merge_pooling_params())
    encoded = pooler(torch.randn(9, 4), metadata(ids, params))
    for mask, tokens, tensor in zip([1, 4, 7], ids, encoded):
        result = processor.post_process([output(tokens, tensor)])
        assert (result["dense_vecs"] is not None) == bool(mask & 1)
        assert (result["lexical_weights"] is not None) == bool(mask & 2)
        assert (result["colbert_vecs"] is not None) == bool(mask & 4)


def test_native_configuration_does_not_expose_internal_transport(monkeypatch):
    from vllm_fl.models.bge_m3 import model as bge_m3

    native = object()
    monkeypatch.setattr(
        bge_m3.BgeM3EmbeddingModel, "_build_pooler", lambda self, cfg: native
    )
    monkeypatch.setattr(
        bge_m3,
        "get_current_vllm_config",
        lambda: SimpleNamespace(model_config=SimpleNamespace(io_processor_plugin=None)),
    )
    model = object.__new__(bge_m3.BgeM3EmbeddingModelFL)
    assert model._build_pooler(None) is native
