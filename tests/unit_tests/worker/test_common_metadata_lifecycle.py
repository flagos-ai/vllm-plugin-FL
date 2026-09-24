from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch

from vllm_fl.worker import common_attention_metadata as metadata


@pytest.mark.parametrize(
    "value,expected",
    [
        (None, "stock"),
        ("stock", "stock"),
        ("0", "stock"),
        ("eager", "eager"),
        ("graph", "graph"),
        ("1", "graph"),
    ],
)
def test_modes_are_independent(monkeypatch, value, expected):
    monkeypatch.setattr(metadata, "supports_accelerator_graph", lambda: True)
    if value is None:
        monkeypatch.delenv("VLLM_FL_COMMON_ATTENTION_METADATA", raising=False)
    else:
        monkeypatch.setenv("VLLM_FL_COMMON_ATTENTION_METADATA", value)
    assert metadata.resolve_metadata_policy().mode == expected


@pytest.mark.parametrize("value", ["", "true", "2", "Graph", "graph "])
def test_mode_typo_is_rejected(monkeypatch, value):
    monkeypatch.setenv("VLLM_FL_COMMON_ATTENTION_METADATA", value)
    with pytest.raises(ValueError, match="must be"):
        metadata.resolve_metadata_policy()


@pytest.mark.parametrize(
    "kwargs", [{"use_ubatching": True}, {"async_spec_decode": True}]
)
def test_unvalidated_schedulers_use_stock(monkeypatch, kwargs):
    monkeypatch.setenv("VLLM_FL_COMMON_ATTENTION_METADATA", "graph")
    assert metadata.resolve_metadata_policy(**kwargs).mode == "stock"


def test_graph_api_fallback_is_distinct_from_producer_validation(monkeypatch):
    monkeypatch.setenv("VLLM_FL_COMMON_ATTENTION_METADATA", "graph")
    monkeypatch.setattr(metadata, "supports_accelerator_graph", lambda: False)
    policy = metadata.resolve_metadata_policy()
    assert policy.requested == "graph" and policy.mode == "eager"
    assert "API" in policy.reason


def test_receipt_cannot_skip_padding_after_another_step_or_rebind(monkeypatch):
    monkeypatch.setattr(metadata, "supports_accelerator_graph", lambda: False)
    runner = metadata.CommonAttentionMetadataGraphRunner()
    table = SimpleNamespace(block_tables=[])
    buffers = (torch.zeros(3), torch.zeros(16), torch.zeros(2), torch.zeros(2))
    receipt = runner.run(table, 2, *buffers, use_graph=False, compute=Mock())
    receipt.validate(table, 2, 16)
    for reqs, tokens in [(3, 16), (2, 17)]:
        with pytest.raises(RuntimeError, match="insufficient"):
            receipt.validate(table, reqs, tokens)
    runner.run(table, 2, *buffers, use_graph=False, compute=Mock())
    with pytest.raises(RuntimeError, match="Expired"):
        receipt.validate(table, 2, 16)
    receipt = runner.run(table, 2, *buffers, use_graph=False, compute=Mock())
    runner.clear()
    with pytest.raises(RuntimeError, match="Expired"):
        receipt.validate(table, 2, 16)


def test_input_batch_replacement_clears_graphs_before_releasing_owner(monkeypatch):
    from vllm_fl.worker import model_runner as module

    runner = object.__new__(module.ModelRunnerFL)
    runner.max_model_len = 64
    runner.max_encoder_len = 0
    runner.max_num_reqs = 4
    runner.max_num_tokens = 16
    runner.device = torch.device("cpu")
    runner.model_config = SimpleNamespace(get_vocab_size=lambda: 32)
    runner.num_spec_tokens = 0
    runner.is_pooling_model = False
    runner.vllm_config = SimpleNamespace(reasoning_config=None)
    runner._init_block_sizes = [4]
    runner._init_kernel_block_sizes = [4]
    old = SimpleNamespace(logitsprocs=None, logitsprocs_need_output_token_ids=False)
    runner.input_batch = old
    events = []
    monkeypatch.setattr(
        module, "_accelerator_synchronize", lambda: events.append("sync")
    )

    def clear():
        assert runner.input_batch is old
        events.append("clear")

    runner.common_attention_metadata_graph = SimpleNamespace(clear=clear)
    new = object()

    def replace(**kwargs):
        assert events == ["sync", "clear"]
        return new

    monkeypatch.setattr(module, "InputBatch", replace)
    config = SimpleNamespace(
        kv_cache_groups=[SimpleNamespace(kv_cache_spec=SimpleNamespace(block_size=8))]
    )
    runner.may_reinitialize_input_batch(config, [8])
    assert runner.input_batch is new


@pytest.mark.parametrize("method", ["_get_slot_mappings", "_build_attention_metadata"])
def test_consumers_reject_expired_receipt_before_skipping_cleanup(monkeypatch, method):
    from vllm_fl.worker.model_runner import ModelRunnerFL

    monkeypatch.setattr(metadata, "supports_accelerator_graph", lambda: False)
    producer = metadata.CommonAttentionMetadataGraphRunner()
    table = SimpleNamespace(block_tables=[])
    buffers = (torch.zeros(3), torch.zeros(16), torch.zeros(2), torch.zeros(2))
    receipt = producer.run(table, 2, *buffers, use_graph=False, compute=Mock())
    producer.clear()
    runner = object.__new__(ModelRunnerFL)
    runner.input_batch = SimpleNamespace(block_table=table)
    runner.kv_cache_config = SimpleNamespace(kv_cache_groups=[object()])
    kwargs = (
        dict(num_tokens_padded=4, num_reqs_padded=2, num_tokens_unpadded=4)
        if method == "_get_slot_mappings"
        else dict(num_tokens=4, num_reqs=2, max_query_len=2)
    )
    with pytest.raises(RuntimeError, match="Expired"):
        getattr(runner, method)(prepared_metadata=receipt, **kwargs)


def test_shutdown_releases_common_graphs_before_model_and_workspace(monkeypatch):
    from vllm.v1.worker import workspace

    from vllm_fl.worker import model_runner as module

    runner = object.__new__(module.ModelRunnerFL)
    events = []
    model = object()
    runner.model = model
    runner.compilation_config = SimpleNamespace(static_forward_context={})
    runner.cache_config = SimpleNamespace(num_gpu_blocks=1)
    runner.common_attention_metadata_graph = SimpleNamespace(
        clear=lambda: events.append("metadata-clear")
    )
    monkeypatch.setattr(
        module, "_accelerator_synchronize", lambda: events.append("sync")
    )
    monkeypatch.setattr(
        module, "current_platform", SimpleNamespace(is_rocm=lambda: True)
    )

    def clear_graphs():
        assert runner.model is model
        assert events == ["sync", "metadata-clear"]
        events.append("model-graphs-clear")

    monkeypatch.setattr(module.GraphWrapper, "clear_all_graphs", clear_graphs)
    monkeypatch.setattr(
        workspace, "reset_workspace_manager", lambda: events.append("workspace-clear")
    )
    monkeypatch.setattr(module.torch.accelerator, "empty_cache", lambda: None)
    runner.shutdown()
    assert runner.model is None
    assert events == [
        "sync",
        "metadata-clear",
        "model-graphs-clear",
        "workspace-clear",
        "sync",
    ]
