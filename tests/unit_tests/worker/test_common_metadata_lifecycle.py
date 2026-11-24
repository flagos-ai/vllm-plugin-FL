from types import SimpleNamespace

import pytest
import torch


def test_producer_success_enables_cleanup_skip_in_eager_mode():
    from unittest.mock import Mock

    from vllm.config import CUDAGraphMode

    from vllm_fl.worker.model_runner import ModelRunnerFL

    runner = object.__new__(ModelRunnerFL)
    runner.common_attention_metadata_graph = SimpleNamespace(
        run=Mock(return_value=False)
    )
    runner.input_batch = SimpleNamespace(block_table=object())
    runner.parallel_config = SimpleNamespace(use_ubatching=False)
    runner.query_start_loc = SimpleNamespace(gpu=torch.zeros(3, dtype=torch.int32))
    runner.positions = torch.zeros(16, dtype=torch.int64)
    runner.seq_lens = torch.zeros(2, dtype=torch.int32)
    runner.num_computed_tokens = torch.zeros(2, dtype=torch.int32)
    assert runner._run_common_attention_metadata(2, CUDAGraphMode.NONE)
    assert (
        runner.common_attention_metadata_graph.run.call_args.kwargs["use_graph"]
        is False
    )
    runner.common_attention_metadata_graph = None
    assert not runner._run_common_attention_metadata(2, CUDAGraphMode.NONE)


def test_producer_failure_propagates_before_consumers_skip_cleanup():
    from unittest.mock import Mock

    from vllm.config import CUDAGraphMode

    from vllm_fl.worker.model_runner import ModelRunnerFL

    runner = object.__new__(ModelRunnerFL)
    runner.common_attention_metadata_graph = SimpleNamespace(
        run=Mock(side_effect=RuntimeError("metadata launch failed"))
    )
    runner.input_batch = SimpleNamespace(block_table=object())
    runner.parallel_config = SimpleNamespace(use_ubatching=False)
    runner.query_start_loc = SimpleNamespace(gpu=torch.zeros(3, dtype=torch.int32))
    runner.positions = torch.zeros(16, dtype=torch.int64)
    runner.seq_lens = torch.zeros(2, dtype=torch.int32)
    runner.num_computed_tokens = torch.zeros(2, dtype=torch.int32)
    with pytest.raises(RuntimeError, match="metadata launch failed"):
        runner._run_common_attention_metadata(2, CUDAGraphMode.NONE)


@pytest.mark.parametrize(
    "ascend,mode_name", [(True, "FULL"), (True, "PIECEWISE"), (False, "FULL")]
)
def test_ascend_padding_runs_after_producer_and_keeps_consumer_cleanup(
    ascend, mode_name
):
    from unittest.mock import Mock

    from vllm.config import CUDAGraphMode

    from vllm_fl.worker.model_runner import ModelRunnerFL

    runner = object.__new__(ModelRunnerFL)
    runner.use_ascend_attention = ascend
    runner.query_start_loc = object()
    events = []
    runner._run_common_attention_metadata = Mock(
        side_effect=lambda *args, **kwargs: events.append("producer") or True
    )
    runner._pad_query_start_loc_for_fia = Mock(
        side_effect=lambda *args: events.append("pad") or 4
    )
    padded, prepared = runner._prepare_common_metadata_for_attention(
        3, getattr(CUDAGraphMode, mode_name), 4, 2, 3, capture=True
    )
    if ascend and mode_name == "FULL":
        assert (padded, prepared, events) == (4, False, ["producer", "pad"])
        runner._pad_query_start_loc_for_fia.assert_called_once_with(
            runner.query_start_loc, 4, 2, 3
        )
    else:
        assert (padded, prepared, events) == (3, True, ["producer"])
        runner._pad_query_start_loc_for_fia.assert_not_called()
    assert runner._run_common_attention_metadata.call_args.kwargs["capture"] is True


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


def test_shutdown_releases_common_graphs_before_model_and_workspace(monkeypatch):
    from vllm.v1.worker import workspace

    from vllm_fl.worker import model_runner as module

    runner = object.__new__(module.ModelRunnerFL)
    events = []
    model = object()
    runner.model = model
    runner.graph_runtime = None
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
        module.BreakableCUDAGraphWrapper,
        "clear_all_graphs",
        lambda: events.append("breakable-graphs-clear"),
    )
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
        "breakable-graphs-clear",
        "workspace-clear",
        "sync",
    ]
