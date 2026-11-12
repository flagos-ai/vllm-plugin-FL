# Copyright (c) 2026 BAAI. All rights reserved.

import pytest
import torch
from transformers.models.xlm_roberta.modeling_xlm_roberta import (
    XLMRobertaEmbeddings,
)

from vllm_fl.models.bge_m3.model import packed_roberta_positions


@pytest.mark.parametrize(
    "sequences",
    [
        [[0, 15, 2], [0, 16, 17, 2]],
        [[0, 3, 0, 2, 1, 2], [0, 42, 2]],
        [[1, 1, 42, 2], [0, 1, 1, 2], [1, 1]],
        [[0, 2]],
        [[0] + [42] * 8190 + [2]],
    ],
)
@pytest.mark.parametrize("dtype", [torch.int32, torch.int64])
def test_packed_positions_match_huggingface(sequences, dtype):
    ids = torch.tensor([token for seq in sequences for token in seq])
    positions = torch.cat([torch.arange(len(seq), dtype=dtype) for seq in sequences])
    original = positions.clone()
    expected = torch.cat(
        [
            XLMRobertaEmbeddings.create_position_ids_from_input_ids(
                torch.tensor([seq]), 1
            ).squeeze(0)
            for seq in sequences
        ]
    )

    result = packed_roberta_positions(ids, positions, 1) + 2

    torch.testing.assert_close(result.long(), expected)
    torch.testing.assert_close(positions, original)


def test_bge_model_factory_uses_position_adapter(monkeypatch):
    from types import SimpleNamespace

    from vllm_fl.models.bge_m3 import model as bge_m3

    calls = []
    model = object()

    def build(**kwargs):
        calls.append(kwargs)
        return model

    monkeypatch.setattr(bge_m3, "BertModel", build)
    config = SimpleNamespace(
        model_config=SimpleNamespace(
            hf_config=SimpleNamespace(position_embedding_type="absolute")
        )
    )

    result = bge_m3.BgeM3EmbeddingModelFL._build_model(None, config, "model")

    assert result is model
    assert calls == [
        {
            "vllm_config": config,
            "prefix": "model",
            "embedding_class": bge_m3.BgeM3Embedding,
        }
    ]
