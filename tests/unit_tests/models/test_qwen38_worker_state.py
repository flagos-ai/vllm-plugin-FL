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


@pytest.mark.parametrize("k", [1, 2, 3])
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


@pytest.mark.parametrize("k,allowed", [(1, True), (2, True), (3, True), (4, False)])
def test_metax_config_accepts_only_validated_mtp_lengths(monkeypatch, k, allowed):
    from _qwen38_cpu import ROOT, declarations

    from vllm import envs

    monkeypatch.setattr(envs, "VLLM_USE_V2_MODEL_RUNNER", False)
    config_module = declarations(
        ROOT / "vllm_fl/patches/qwen3_8_flash_next.py",
        "_qwen38_metax_config_cpu",
        names={"_verify_metax_config"},
    )
    config = SimpleNamespace(
        speculative_config=SimpleNamespace(
            num_speculative_tokens=k,
            disable_padded_drafter_batch=True,
        ),
        model_config=SimpleNamespace(dtype=torch.bfloat16, multimodal_config=None),
        cache_config=SimpleNamespace(
            mamba_ssm_cache_dtype="float32",
            mamba_cache_mode="none",
            enable_prefix_caching=False,
        ),
        scheduler_config=SimpleNamespace(async_scheduling=False),
        parallel_config=SimpleNamespace(pipeline_parallel_size=1),
    )
    if allowed:
        config_module._verify_metax_config(config)
    else:
        with pytest.raises(NotImplementedError, match="MTP K1, K2 or K3"):
            config_module._verify_metax_config(config)


@pytest.mark.parametrize(
    "enabled,counts,spec,prompt_embeds,want_cpu",
    [
        (True, [0, 1], None, False, True),
        (False, [0, 1], None, False, False),
        (True, [4, 1], None, False, False),
        (True, [4, 4], None, False, False),
        (True, [0, 1], object(), False, False),
        (True, [0, 1], None, True, False),
    ],
)
def test_ple_uses_cpu_ids_only_for_synchronous_plain_prefill(
    enabled, counts, spec, prompt_embeds, want_cpu
):
    import numpy as np

    NS = SimpleNamespace
    captured = {}

    def prepare(ids, starts, context, **kwargs):
        captured.update(ids=ids, starts=starts, context=context, **kwargs)
        return captured

    runner = NS(
        vllm_config=NS(
            parallel_config=NS(pipeline_parallel_size=1),
            cache_config=NS(mamba_cache_mode="none", enable_prefix_caching=False),
            scheduler_config=NS(async_scheduling=False),
            additional_config={"qwen38_ple_cpu_metadata": enabled},
        ),
        model_config=NS(hf_text_config=NS(eos_token_id=0)),
        _qwen38_host_inputs=NS(context_len=2, prepare=prepare),
        input_batch=NS(
            num_reqs=2,
            token_ids_cpu_tensor=torch.tensor([[1, 2, 3, 4, 5], [6, 7, 8, 9, 10]]),
            num_computed_tokens_cpu=np.array(counts),
            num_prompt_tokens=np.array([4, 4]),
        ),
        input_ids=NS(cpu=object(), gpu=object()),
        query_start_loc=NS(cpu=torch.tensor([0, 3, 4])),
        speculative_config=spec,
        enable_prompt_embeds=prompt_embeds,
    )
    result = worker._mrv1_prepare(runner)
    assert result["ids"] is (runner.input_ids.cpu if want_cpu else runner.input_ids.gpu)
    assert result["use_cpu_metadata"] is enabled


def test_host_metadata_fast_path_keeps_forward_buffers_and_hash_identical():
    from _qwen38_cpu import ngram

    outputs = []
    config = SimpleNamespace(eos_token_id=0)
    ids = torch.tensor([3, 0, 5, 7, 8])
    starts = torch.tensor([0, 3, 5], dtype=torch.int32)
    context = torch.tensor([[0, 1], [2, 6]], dtype=torch.int64)

    def make():
        buffers = worker._HostInputBuffers.__new__(worker._HostInputBuffers)
        buffers.config = config
        buffers.query_start_loc = torch.zeros(5, dtype=torch.int32)
        buffers.context = torch.zeros((4, 2), dtype=torch.int64)

        def prepare(tokens, offsets, histories, dummy=False):
            actual = int(offsets[-1])
            outputs.append(
                ngram.compute_ngram_ids_cpu(
                    tokens[:actual],
                    offsets,
                    histories,
                    multipliers=torch.tensor([3, 5, 7]),
                    vocab_sizes=torch.tensor([31, 37]),
                    vocab_offsets=torch.tensor([0, 31]),
                    heads_per_ngram=1,
                    eos_token_id=0,
                )
            )

        buffers.layers = [SimpleNamespace(prepare_host_lookup=prepare)]
        return buffers

    old, new = make(), make()
    old_result = old.prepare(ids, starts, context)
    new_result = new.prepare(ids, starts, context, use_cpu_metadata=True)
    assert torch.equal(outputs[0], outputs[1])
    assert torch.equal(old_result["query_start_loc"], new_result["query_start_loc"])
    assert torch.equal(old_result["ngram_context"], new_result["ngram_context"])


def test_async_history_plan_handles_reorder_resume_and_reused_request_ids():
    states = {k: object() for k in ("a", "b", "c")}
    old = {"a": (states["a"], 4, 0), "b": (states["b"], 8, 1)}
    tokens = torch.full((3, 16), -1, dtype=torch.int32)
    tokens[2, :3] = torch.tensor([5, 6, 7])
    meta, following = worker.async_context_plan(
        ["b", "a", "c"],
        states,
        [8, 4, 3],
        torch.tensor([0, 1, 2, 5]),
        tokens,
        old,
        2,
        99,
    )
    assert meta.tolist() == [[1, 99, 99], [0, 99, 99], [-1, 6, 7]]
    assert [(v[1], v[2]) for v in following.values()] == [(9, 0), (5, 1), (6, 2)]
    # Recompute from zero and a reused logical ID cannot inherit stale suffixes.
    states["b"] = object()
    meta, _ = worker.async_context_plan(
        ["b", "a"],
        states,
        [0, 0],
        torch.tensor([0, 1, 2]),
        tokens[:2],
        old,
        2,
        99,
    )
    assert meta.tolist() == [[-1, 99, 99], [-1, 99, 99]]


def test_async_history_never_seeds_unresolved_native_placeholders():
    state = {"a": object()}
    with pytest.raises(ValueError, match="unresolved"):
        worker.async_context_plan(
            ["a"],
            state,
            [3],
            torch.tensor([0, 1]),
            torch.tensor([[1, -1, -1]]),
            {},
            2,
            0,
        )


def test_async_history_seed_padding_and_zero_length_requests():
    states = {k: object() for k in ("a", "b")}
    meta, following = worker.async_context_plan(
        ["a", "b"],
        states,
        [1, 0],
        torch.tensor([0, 0, 3]),
        torch.tensor([[7, 8, 9], [1, 2, 3]]),
        {},
        2,
        99,
    )
    assert meta.tolist() == [[-1, 99, 7], [-1, 99, 99]]
    assert following["a"][1] == 1 and following["b"][1] == 3
