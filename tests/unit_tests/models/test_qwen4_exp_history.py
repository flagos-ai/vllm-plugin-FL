"""Core Qwen3.8 framework contracts; numerical tests live in FlagGems-vllm."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest
import torch

from vllm_fl.models.qwen4_exp.legacy_model_state import (
    Qwen4ExpLegacyModelState,
)
from vllm_fl.models.qwen4_exp.ple_token_history import PLETokenHistory


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


def test_legacy_model_state_prepares_inputs_and_keeps_dummy_history_clean(
    device, monkeypatch
):
    monkeypatch.setattr(
        "vllm_fl.models.qwen4_exp.legacy_model_state.get_pp_group",
        lambda: SimpleNamespace(ranks=[0], is_first_rank=True),
    )

    def buffer(*shape, dtype):
        cpu = torch.zeros(shape, dtype=dtype)
        gpu = torch.zeros(shape, dtype=dtype, device=device)
        return SimpleNamespace(
            gpu=gpu,
            np=cpu.numpy(),
            copy_to_gpu=lambda n=None: gpu.copy_(cpu),
        )

    ids = torch.tensor([7, 8], dtype=torch.int32, device=device)
    batch = SimpleNamespace(
        req_ids=["a"],
        num_computed_tokens_cpu=np.array([0]),
        token_ids_cpu_tensor=torch.full((4, 32), -1, dtype=torch.int32),
    )
    runner = SimpleNamespace(
        model_config=SimpleNamespace(
            hf_text_config=SimpleNamespace(
                ple_layer_ids=[1], ngram_size=3, eos_token_id=0
            ),
            is_encoder_decoder=False,
        ),
        max_num_reqs=4,
        max_model_len=32,
        max_num_tokens=32,
        device=device,
        pin_memory=False,
        _make_buffer=buffer,
        input_batch=batch,
        enable_prompt_embeds=False,
        query_start_loc=buffer(5, dtype=torch.int32),
        num_scheduled_tokens=SimpleNamespace(np=np.array([2])),
        req_indices=SimpleNamespace(np=np.array([0, 0])),
        query_pos=SimpleNamespace(np=np.array([0, 1])),
        input_ids=SimpleNamespace(gpu=ids),
    )
    state = Qwen4ExpLegacyModelState(runner)
    dummy = state.prepare_dummy_inputs(1, 4, np.array([2]))
    assert dummy["query_start_loc"].cpu().tolist() == [0, 2, 2, 2, 2]
    assert dummy["ngram_context"].count_nonzero().item() == 0
    assert state.ple_token_history._slots == {}
    state.prepare_inputs(1, 4, ids)
    batch.num_computed_tokens_cpu[:] = 2
    runner.num_scheduled_tokens.np[:] = 1
    runner.query_start_loc.np[:] = [0, 1, 1, 1, 1]
    runner.query_start_loc.copy_to_gpu()
    runner.input_ids.gpu = torch.tensor([9], dtype=torch.int32, device=device)
    state.prepare_dummy_inputs(1, 4, np.array([1]))
    real = state.prepare_inputs(1, 4, runner.input_ids.gpu)
    assert real["ngram_context"].cpu().tolist() == [[7, 8], [0, 0], [0, 0], [0, 0]]
    with pytest.raises(RuntimeError, match="token ids"):
        state.prepare_inputs(1, 4, None)
    state.remove_requests(["a"])
    batch.num_computed_tokens_cpu[:] = 0
    assert (
        state.prepare_inputs(1, 4, runner.input_ids.gpu)["ngram_context"]
        .count_nonzero()
        .item()
        == 0
    )


@pytest.mark.parametrize("context_len", [2])
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
