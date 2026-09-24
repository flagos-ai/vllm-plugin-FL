# Copyright (c) 2026 BAAI. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from contextlib import contextmanager
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch

import vllm_fl.worker.common_attention_metadata as metadata


@pytest.mark.parametrize("cuda", [False, True])
def test_producer_policy_defaults_to_stock_on_every_platform(monkeypatch, cuda):
    monkeypatch.setattr(
        metadata, "current_platform", SimpleNamespace(is_cuda=lambda: cuda)
    )
    monkeypatch.setattr(metadata, "supports_accelerator_graph", lambda: True)
    monkeypatch.delenv("VLLM_FL_COMMON_ATTENTION_METADATA", raising=False)
    assert metadata.resolve_metadata_policy().mode == "stock"
    monkeypatch.setenv("VLLM_FL_COMMON_ATTENTION_METADATA", "0")
    assert metadata.resolve_metadata_policy().mode == "stock"
    monkeypatch.setenv("VLLM_FL_COMMON_ATTENTION_METADATA", "1")
    assert metadata.resolve_metadata_policy().mode == "graph"


def test_graph_capture_replays_before_return_and_clear_drops_cache(monkeypatch):
    graph = Mock()
    capturing = False

    @contextmanager
    def capture(*args, **kwargs):
        nonlocal capturing
        capturing = True
        yield
        capturing = False

    platform = SimpleNamespace(
        torch_device_fn=SimpleNamespace(graph=capture),
        get_global_graph_pool=lambda: None,
    )
    monkeypatch.setattr(metadata, "current_platform", platform)
    monkeypatch.setattr(metadata, "Graph", SimpleNamespace(graph=lambda: graph))
    runner = metadata.CommonAttentionMetadataGraphRunner()
    output = torch.full((1,), -12345)

    def compute(*args):
        if capturing:
            # Capture records an operation; only replay executes it.
            graph.replay.side_effect = lambda: output.fill_(42)
            output.fill_(-12345)
        else:
            output.fill_(42)

    table = SimpleNamespace(block_tables=[])
    args = (table, 1, output, output, output, output)
    receipt = runner.run(*args, use_graph=True, capture=True, compute=compute)
    assert receipt.used_graph
    receipt.validate(table, 1, 1)
    assert output.item() == 42
    graph.replay.assert_called_once()
    output.fill_(-12345)
    assert runner.run(*args, use_graph=True, capture=False, compute=compute).used_graph
    assert output.item() == 42
    runner.clear()
    assert not runner.graphs
    assert not runner.run(
        *args, use_graph=True, capture=False, compute=compute
    ).used_graph
    assert output.item() == 42


def test_graph_api_must_be_callable(monkeypatch):
    monkeypatch.setattr(metadata, "Graph", SimpleNamespace(graph=None))
    monkeypatch.setattr(
        metadata,
        "current_platform",
        SimpleNamespace(
            torch_device_fn=SimpleNamespace(graph=Mock()),
        ),
    )
    assert not metadata.supports_accelerator_graph()
