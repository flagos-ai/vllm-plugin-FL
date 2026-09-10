# Copyright (c) 2026 BAAI. All rights reserved.

import inspect
from types import SimpleNamespace

import pytest
import torch

from vllm.model_executor.parameter import ModelWeightParameter

from vllm_fl.dispatch.backends.vendor.ascend.impl import (
    vocab_parallel_embedding as ascend_vocab,
)


class _QuantMethod:
    def create_weights(
        self,
        layer,
        input_size_per_partition,
        output_partition_sizes,
        input_size,
        output_size,
        params_dtype,
        **extra_weight_attrs,
    ):
        del input_size, output_size
        weight = ModelWeightParameter(
            data=torch.empty(
                sum(output_partition_sizes),
                input_size_per_partition,
                dtype=params_dtype,
            ),
            input_dim=1,
            output_dim=0,
            weight_loader=extra_weight_attrs["weight_loader"],
        )
        layer.register_parameter("weight", weight)

    def embedding(self, layer, input_):
        return torch.nn.functional.embedding(input_, layer.weight)


class _QuantConfig:
    def get_quant_method(self, layer, prefix):
        del layer, prefix
        return _QuantMethod()


@pytest.fixture(autouse=True)
def _mock_parameter_construction_tp_rank(monkeypatch):
    """ModelWeightParameter reads global TP state before layer metadata is set."""
    monkeypatch.setattr(
        "vllm.model_executor.parameter.get_tensor_model_parallel_rank",
        lambda: 0,
    )
    monkeypatch.setattr(
        "vllm.model_executor.parameter.get_tensor_model_parallel_world_size",
        lambda: 1,
    )


@pytest.mark.parametrize(
    "layer_cls",
    [
        ascend_vocab.AscendVocabParallelEmbedding,
        ascend_vocab.AscendParallelLMHead,
    ],
)
def test_disable_tp_is_keyword_only(layer_cls):
    parameter = inspect.signature(layer_cls.__init__).parameters["disable_tp"]

    assert parameter.kind is inspect.Parameter.KEYWORD_ONLY
    assert parameter.default is False


@pytest.mark.parametrize(
    ("disable_tp", "expected_rank", "expected_size", "expected_partition_size"),
    [
        (False, 1, 4, 32),
        (True, 0, 1, 128),
    ],
)
def test_embedding_updates_parameter_tp_status(
    monkeypatch,
    disable_tp,
    expected_rank,
    expected_size,
    expected_partition_size,
):
    group_requests = []

    def get_tp_group():
        group_requests.append(True)
        return SimpleNamespace(world_size=4, rank_in_group=1)

    monkeypatch.setattr(
        ascend_vocab,
        "get_tp_group",
        get_tp_group,
    )

    layer = ascend_vocab.AscendVocabParallelEmbedding(
        65,
        16,
        quant_config=_QuantConfig(),
        disable_tp=disable_tp,
    )

    assert layer.disable_tp is disable_tp
    assert layer.tp_rank == expected_rank
    assert layer.tp_size == expected_size
    assert layer.num_embeddings_per_partition == expected_partition_size
    assert layer.weight.tp_rank == expected_rank
    assert layer.weight.tp_size == expected_size
    assert group_requests == ([] if disable_tp else [True])


def test_embedding_rejects_quant_method_without_embedding():
    quant_config = SimpleNamespace(
        get_quant_method=lambda layer, prefix: object(),
    )

    with pytest.raises(NotImplementedError, match="must implement.*embedding"):
        ascend_vocab.AscendVocabParallelEmbedding(
            65,
            16,
            quant_config=quant_config,
            disable_tp=True,
        )


def test_disable_tp_forward_skips_collective(monkeypatch):
    monkeypatch.setattr(
        ascend_vocab,
        "maybe_pad_and_reduce",
        lambda output: pytest.fail("disable_tp forward must not all-reduce"),
    )
    layer = ascend_vocab.AscendVocabParallelEmbedding(
        65,
        16,
        quant_config=_QuantConfig(),
        disable_tp=True,
    )
    input_ids = torch.tensor([[0, 64]])

    output = layer(input_ids)

    expected = torch.nn.functional.embedding(input_ids, layer.weight)
    torch.testing.assert_close(output, expected)


def test_lm_head_propagates_disable_tp(monkeypatch):
    monkeypatch.setattr(
        ascend_vocab,
        "get_tp_group",
        lambda: SimpleNamespace(world_size=4, rank_in_group=1),
    )

    layer = ascend_vocab.AscendParallelLMHead(
        65,
        16,
        bias=True,
        quant_config=_QuantConfig(),
        disable_tp=True,
    )

    assert layer.disable_tp
    assert layer.tp_rank == 0
    assert layer.tp_size == 1
    assert layer.bias.shape == (128,)
