"""Exercise real GPU history with deliberately unusable CPU decode tokens."""

from types import SimpleNamespace

import numpy as np
import pytest
import torch

from vllm_fl.worker.ple_token_history import PLETokenHistory


@pytest.fixture(params=["cpu", "cuda"])
def device(request):
    if request.param == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA required")
    return torch.device(request.param)


class Harness:
    def __init__(self, device, context_len=2):
        self.device = device
        self.history = PLETokenHistory(4, 32, 32, context_len, 0, device, False)
        self.context = torch.empty((4, context_len), dtype=torch.int32, device=device)
        self.cpu = torch.full((4, 32), -1, dtype=torch.int32)

    def step(self, req_ids, starts, chunks, *, seeds=None, masks=None):
        # Every non-seeded position is an async placeholder, even when old
        # output tokens have already been consumed by several GPU forwards.
        self.cpu.fill_(-1)
        for row, values in (seeds or {}).items():
            self.cpu[row, : len(values)] = torch.tensor(values, dtype=torch.int32)
        counts = np.array([len(chunk) for chunk in chunks], dtype=np.int32)
        indices = np.repeat(np.arange(len(req_ids)), counts)
        positions = (
            np.concatenate([np.arange(n) for n in counts])
            if len(counts)
            else np.array([], dtype=np.int64)
        )
        ids = torch.tensor(
            [x for c in chunks for x in c], dtype=torch.int32, device=self.device
        )
        self.history.prepare(
            req_ids=req_ids,
            num_computed_tokens=np.array(starts, dtype=np.int32),
            num_scheduled_tokens=counts,
            token_ids_cpu=self.cpu,
            is_token_ids=masks,
            req_indices=indices,
            query_positions=positions,
            input_ids=ids,
            context=self.context,
        )
        # The runner's synchronize_input_prep event protects pinned buffers
        # from CPU mutation; this harness waits when observing each output.
        return self.context.cpu().clone()


@pytest.mark.parametrize("context_len", [1, 2, 4])
def test_chunked_prefill_decode_reorder_and_padding(device, context_len):
    h = Harness(device, context_len)
    a, b = [5, 6, 0, 8, 9, 10, 11], [15, 16, 17, 18, 19]
    h.step(["a", "b"], [0, 0], [a[:3], b[:1]])
    expected = lambda seq: [0] * max(0, context_len - len(seq)) + seq[-context_len:]
    out = h.step(["b", "a"], [1, 3], [b[1:4], a[3:5]])
    assert out[:2].tolist() == [expected(b[:1]), expected(a[:3])]
    assert out[2:].count_nonzero() == 0
    out = h.step(["a", "b"], [5, 4], [a[5:6], b[4:5]])
    assert out[:2].tolist() == [expected(a[:5]), expected(b[:4])]
    out = h.step(["a"], [6], [a[6:7]])
    assert out[0].tolist() == expected(a[:6])
    assert out[1:].count_nonzero() == 0


def test_rollback_replaces_discarded_suffix_without_cpu_history(device):
    h = Harness(device)
    h.step(["a"], [0], [[1, 2, 3, 4, 5, 6]])
    out = h.step(["a"], [2], [[30, 40]])
    assert out[0].tolist() == [1, 2]
    out = h.step(["a"], [4], [[50]])
    assert out[0].tolist() == [30, 40]
    # Old tokens at positions5+ are no longer valid after the rollback.
    with pytest.raises(RuntimeError, match="async placeholders"):
        h.step(["a"], [6], [[70]])


def test_prefix_hits_resume_and_same_id_replacement(device):
    h = Harness(device)
    out = h.step(["a"], [5], [[6]], seeds={0: [1, 2, 3, 4, 5]})
    assert out[0].tolist() == [4, 5]
    # Evict a from the persistent batch, then resume from scheduler-owned ids.
    h.step(["b"], [0], [[91, 92]])
    out = h.step(["a", "b"], [6, 2], [[7], [93]], seeds={0: [1, 2, 3, 4, 5, 6]})
    assert out[:2].tolist() == [[5, 6], [91, 92]]
    # Forced preemption/streaming replacement may reuse a still-active id.
    h.history.forget(["a"])
    out = h.step(["a"], [2], [[13]], seeds={0: [11, 12]})
    assert out[0].tolist() == [11, 12]
    # Abort+resubmit: the new request must not see any old prefix.
    h.history.forget(["a"])
    out = h.step(["a"], [0], [[99]])
    assert out.count_nonzero() == 0


def test_forward_prefix_jump_keeps_real_gpu_tokens(device):
    h = Harness(device)
    h.step(["a"], [0], [[1, 2]])
    out = h.step(["a"], [4], [[5]], seeds={0: [-1, -1, 3, 4]})
    assert out[0].tolist() == [3, 4]
    out = h.step(["a"], [1], [[22]])
    assert out[0].tolist() == [0, 1]


def test_empty_batch_and_capacity_boundaries(device):
    h = Harness(device)
    assert h.step([], [], []).count_nonzero() == 0
    out = h.step(["a"], [31], [[32]], seeds={0: list(range(1, 32))})
    assert out[0].tolist() == [30, 31]
    with pytest.raises(ValueError, match="capacity"):
        h.step(["a"], [32], [[33]])


def test_non_token_prefix_positions_use_eos(device):
    h = Harness(device)
    masks = np.ones((4, 32), dtype=bool)
    masks[0, 1] = False
    out = h.step(["a"], [2], [[3]], seeds={0: [1, -1]}, masks=masks)
    assert out[0].tolist() == [1, 0]


def test_worker_state_updates_invalidate_finished_resumed_and_replaced_history(
    device, monkeypatch
):
    from unittest.mock import Mock

    from vllm_fl.worker import model_runner as module

    h = Harness(device)
    runner = object.__new__(module.ModelRunnerFL)
    runner.uses_ngram_embedding = True
    runner.ple_token_history = h.history
    runner.requests = {}
    runner.num_prompt_logprobs = {}
    runner.late_interaction_runner = Mock()
    runner.encoder_cache = {}
    runner.speculative_config = None
    runner.is_pooling_model = False
    runner.uses_mrope = False
    runner.uses_xdrope_dim = 0
    runner.use_async_spec_decode = False
    runner.use_async_scheduling = True
    runner.device = device
    runner._may_reorder_batch = lambda output: None
    indices = {}
    runner.input_batch = SimpleNamespace(
        req_id_to_index=indices,
        remove_request=lambda req_id: indices.pop(req_id, None),
        add_request=lambda request: indices.update({request.req_id: len(indices)}),
        update_req_spec_token_ids=lambda *args: None,
        condense=lambda: None,
        refresh_metadata=lambda: None,
    )
    monkeypatch.setattr(
        module, "get_pp_group", lambda: SimpleNamespace(is_last_rank=True)
    )

    def update(tokens=None, *, finished=False, resume=False):
        new = (
            []
            if tokens is None
            else [
                SimpleNamespace(
                    req_id="a",
                    prompt_token_ids=tokens,
                    prompt_embeds=None,
                    prompt_is_token_ids=None,
                    mm_features=[],
                    sampling_params=None,
                    pooling_params=None,
                    block_ids=([1],),
                    num_computed_tokens=2,
                    lora_request=None,
                )
            ]
        )
        cached = SimpleNamespace(
            req_ids=["a"] if resume else [],
            resumed_req_ids={"a"} if resume else set(),
            num_computed_tokens=[2],
            new_block_ids=[([2],)],
            num_output_tokens=[0],
            all_token_ids={},
        )
        runner._update_states(
            SimpleNamespace(
                finished_req_ids={"a"} if finished else set(),
                new_block_ids_to_zero=[],
                free_encoder_mm_hashes=[],
                num_scheduled_tokens={"a": 1} if tokens is not None or resume else {},
                scheduled_new_reqs=new,
                scheduled_cached_reqs=cached,
                scheduled_spec_decode_tokens={},
            )
        )

    update([1, 2, 3])
    h.step(["a"], [0], [[1, 2, 3]])
    # This invokes the real streaming replacement entry, not history.forget().
    update([11, 12, 13])
    assert h.step(["a"], [2], [[13]], seeds={0: [11, 12]})[0].tolist() == [11, 12]
    update(resume=True)
    assert h.step(["a"], [2], [[23]], seeds={0: [21, 22]})[0].tolist() == [21, 22]
    update(finished=True)
    assert "a" not in h.history._slots
    update([31, 32, 33], finished=True)
    assert h.step(["a"], [2], [[33]], seeds={0: [31, 32]})[0].tolist() == [31, 32]


def test_selected_runner_reads_resolved_device_inputs_and_dummy_is_stateless(device):
    from vllm_fl.worker.model_runner import ModelRunnerFL

    h = Harness(device)
    h.step(["a"], [0], [[1, 2]])
    gpu_ids = torch.tensor([3], dtype=torch.int32, device=device)
    runner = SimpleNamespace(
        uses_ngram_embedding=True,
        ngram_eos_token_id=0,
        enable_prompt_embeds=False,
        ple_token_history=h.history,
        input_batch=SimpleNamespace(
            req_ids=["a"],
            num_computed_tokens_cpu=np.array([2]),
            token_ids_cpu_tensor=h.cpu,
        ),
        query_start_loc=SimpleNamespace(np=np.array([0, 1])),
        num_scheduled_tokens=SimpleNamespace(np=np.array([1])),
        req_indices=SimpleNamespace(np=np.array([0])),
        query_pos=SimpleNamespace(np=np.array([0])),
        input_ids=SimpleNamespace(gpu=gpu_ids),
        ngram_context=SimpleNamespace(
            gpu=h.context,
            np=np.zeros((4, 2), dtype=np.int32),
            copy_to_gpu=lambda n: h.context[:n].zero_(),
        ),
    )
    context = ModelRunnerFL._prepare_ngram_context(runner, 1, 4)
    assert context[0].tolist() == [1, 2]
    kwargs = {}
    ModelRunnerFL._maybe_add_ngram_kwargs(
        runner,
        kwargs,
        num_reqs=1,
        num_reqs_padded=4,
        is_first_rank=True,
        is_encoder_decoder=False,
        use_dummy_context=True,
        query_start_loc=torch.tensor([0, 1, 1, 1, 1], device=device),
    )
    assert kwargs["ngram_context"].count_nonzero() == 0
    out = h.step(["a"], [3], [[4]])
    assert out[0].tolist() == [2, 3]


@pytest.mark.gpu
def test_fixed_address_consumer_smoke_reads_updated_context_and_padding():
    if not torch.cuda.is_available():
        pytest.skip("CUDA required")
    h = Harness(torch.device("cuda"))
    h.step(["a", "b"], [0, 0], [[1, 2], [11, 12]])
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(3):
            output = h.context * 7
    torch.cuda.current_stream().wait_stream(stream)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        output = h.context * 7
    for reqs, starts, chunks in [
        (["b", "a"], [2, 2], [[13], [3]]),
        (["a", "b"], [3, 3], [[4], [14]]),
        (["b"], [4], [[15]]),
    ]:
        expected = h.step(reqs, starts, chunks)
        graph.replay()
        torch.testing.assert_close(output.cpu(), expected * 7, rtol=0, atol=0)
