# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 BAAI. All rights reserved.
"""CPU checks for vLLM 0.32 sharded embedding semantics on Sunrise."""

import importlib.util
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import Mock

import pytest
import torch


@pytest.fixture
def embedding_module(monkeypatch):
    def install(name, **attrs):
        module = ModuleType(name)
        vars(module).update(attrs)
        monkeypatch.setitem(sys.modules, name, module)
        return module

    reduce = Mock(side_effect=lambda tensor: tensor)
    install("vllm.distributed", tensor_model_parallel_all_reduce=reduce)
    install(
        "vllm.model_executor.layers.vocab_parallel_embedding",
        VocabParallelEmbedding=object,
        is_fp8=lambda dtype: dtype == torch.float8_e4m3fn,
    )
    path = Path(__file__).resolve().parents[3] / (
        "vllm_fl/dispatch/backends/vendor/sunrise/impl/vocab_parallel_embedding.py"
    )
    spec = importlib.util.spec_from_file_location("_sunrise_embedding_test", path)
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, spec.name, module)
    spec.loader.exec_module(module)
    return module, reduce


def make_embedding(module, *, tp_size=2, reduce_results=True, parallel_group=None):
    layer = module.SunriseVocabParallelEmbedding()
    layer.tp_size = tp_size
    layer.reduce_results = reduce_results
    layer.parallel_group = parallel_group
    layer.shard_indices = SimpleNamespace(
        org_vocab_start_index=2,
        org_vocab_end_index=4,
        num_org_vocab_padding=0,
        added_vocab_start_index=6,
        added_vocab_end_index=8,
    )
    layer.quant_method = SimpleNamespace(
        embedding=Mock(side_effect=lambda layer, ids: (ids.float() + 1).unsqueeze(-1))
    )
    return layer


def test_single_shard_needs_no_collective(embedding_module):
    module, reduce = embedding_module
    layer = make_embedding(module, tp_size=1)
    result = layer.forward(torch.tensor([2, 3]))
    torch.testing.assert_close(result, torch.tensor([[3.0], [4.0]]))
    reduce.assert_not_called()


@pytest.mark.parametrize("reduce_results", [True, False])
@pytest.mark.parametrize("custom_group", [True, False])
def test_sharded_embedding_preserves_reduction_selection(
    embedding_module, reduce_results, custom_group
):
    module, reduce = embedding_module
    group = SimpleNamespace(all_reduce=Mock(side_effect=lambda tensor: tensor))
    layer = make_embedding(
        module,
        reduce_results=reduce_results,
        parallel_group=group if custom_group else None,
    )
    result = layer.forward(torch.tensor([1, 2, 3, 6, 7, 8]))
    torch.testing.assert_close(
        result, torch.tensor([[0.0], [1.0], [2.0], [3.0], [4.0], [0.0]])
    )
    assert reduce.call_count == int(reduce_results and not custom_group)
    assert group.all_reduce.call_count == int(reduce_results and custom_group)


def test_fp8_storage_is_masked_and_reduced_as_int8(embedding_module):
    module, reduce = embedding_module
    layer = make_embedding(module)
    layer.quant_method.embedding = Mock(
        return_value=torch.ones(3, 1).to(torch.float8_e4m3fn)
    )
    result = layer.forward(torch.tensor([1, 2, 3]))
    assert result.dtype == torch.float8_e4m3fn
    assert reduce.call_args.args[0].dtype == torch.int8
    torch.testing.assert_close(result.float(), torch.tensor([[0.0], [1.0], [1.0]]))
