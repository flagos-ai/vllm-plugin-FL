from types import SimpleNamespace

import pytest
import torch
from _qwen38_cpu import worker


def test_committed_context_excludes_rejected_draft_suffix():
    history = torch.tensor(
        [[10, 11, 12, 91, 92], [20, 81, 82, 83, 84], [99, 99, 99, 99, 99]]
    )
    result = worker.committed_context_cpu(history, [3, 1, 0], 2, 0)
    assert result.tolist() == [[11, 12], [0, 20], [0, 0]]
    with pytest.raises(ValueError, match="outside"):
        worker.committed_context_cpu(history, [6], 2, 0)


@pytest.mark.parametrize("k", [1, 3])
def test_zero_draft_decode_keeps_restore_path_and_prefill_does_not(k):
    counts = worker.spec_decode_counts(
        [1, k + 1, 1, 8, 0], [False, False, True, True, False], k
    )
    assert counts.tolist() == [0, k, -1, -1, -1]


def test_mrv1_hooks_are_idempotent_and_leave_other_models_unchanged():
    class Runner:
        model_config = SimpleNamespace(hf_text_config=SimpleNamespace(ple_layer_ids=[]))

        def _init_model_kwargs(self):
            return {"existing": 1}

        def _dummy_run(self, num_tokens):
            return num_tokens

        def _build_attention_metadata(self, **kwargs):
            return kwargs

    worker.install_qwen38_runner_hooks(Runner)
    first = Runner._init_model_kwargs
    worker.install_qwen38_runner_hooks(Runner)
    assert Runner._init_model_kwargs is first
    instance = Runner()
    assert instance._init_model_kwargs() == {"existing": 1}
    assert instance._dummy_run(4) == 4
    assert not hasattr(instance, "_qwen38_dummy_tokens")
    assert instance._build_attention_metadata(value=3) == {"value": 3}
    assert worker.get_ple_spec_metadata() is None


def test_mrv1_metadata_context_restores_after_builder_error():
    class Runner:
        def _init_model_kwargs(self):
            return {}

        def _dummy_run(self, n):
            return n

        def _build_attention_metadata(self, **kwargs):
            accepted, draft_counts = worker.get_ple_spec_metadata()
            assert accepted is self.num_accepted_tokens.gpu
            assert draft_counts is self.num_decode_draft_tokens.cpu
            assert kwargs["use_spec_decode"] is True
            raise RuntimeError("builder failure")

    worker.install_qwen38_runner_hooks(Runner)
    runner = Runner.__new__(Runner)
    runner.model_config = SimpleNamespace(
        hf_text_config=SimpleNamespace(
            supports_qwen38_ngram_context=True,
            ple_layer_ids=[2],
        )
    )
    runner.speculative_config = SimpleNamespace(num_speculative_tokens=1)
    runner.vllm_config = SimpleNamespace(
        parallel_config=SimpleNamespace(pipeline_parallel_size=1),
        cache_config=SimpleNamespace(
            mamba_cache_mode="none", enable_prefix_caching=False
        ),
        scheduler_config=SimpleNamespace(async_scheduling=False),
    )
    runner.num_accepted_tokens = SimpleNamespace(gpu=torch.tensor([2, 1]))
    runner.num_decode_draft_tokens = SimpleNamespace(cpu=torch.tensor([1, 0]))
    previous = ("outer context",)
    token = worker._ple_spec_context.set(previous)
    try:
        with pytest.raises(RuntimeError, match="builder failure"):
            runner._build_attention_metadata(for_cudagraph_capture=True)
        assert worker.get_ple_spec_metadata() is previous
    finally:
        worker._ple_spec_context.reset(token)


def test_prefill_tail_cannot_alias_k1_decode_graph(monkeypatch):
    from types import SimpleNamespace as NS

    import numpy as np
    from _qwen38_cpu import worker

    class Runner:
        def __init__(self):
            self.model_config = NS(
                hf_text_config=NS(supports_qwen38_ngram_context=True, ple_layer_ids=[2])
            )
            self.input_batch = NS(
                num_computed_tokens_cpu=np.array([20, 18]),
                num_prompt_tokens=np.array([20, 20]),
            )

        def _init_model_kwargs(self):
            return {}

        def _dummy_run(self, *args, **kwargs):
            return None

        def _build_attention_metadata(self, *args, **kwargs):
            return None

        def _determine_batch_execution_and_padding(self, *args, **kwargs):
            return kwargs

    worker.install_qwen38_runner_hooks(Runner)
    runner = Runner()
    assert runner._determine_batch_execution_and_padding(num_reqs=2, force_eager=False)[
        "force_eager"
    ]
    assert not runner._determine_batch_execution_and_padding(
        num_reqs=1, force_eager=False
    )["force_eager"]
    runner._qwen38_dummy_tokens = 4
    assert not runner._determine_batch_execution_and_padding(
        num_reqs=2, force_eager=False
    )["force_eager"]
    runner._qwen38_dummy_tokens = None
    runner.model_config.hf_text_config.supports_qwen38_ngram_context = False
    assert not runner._determine_batch_execution_and_padding(
        num_reqs=2, force_eager=False
    )["force_eager"]
