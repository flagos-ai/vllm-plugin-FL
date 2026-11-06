"""Core Qwen3.8 framework contracts; numerical tests live in FlagGems-vllm."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest
import torch

from vllm_fl.models.qwen3_8_flash_next.config import Qwen3_8FlashNextTextConfig
from vllm_fl.models.qwen3_8_flash_next.gpu import ple_layer
from vllm_fl.models.qwen3_8_flash_next.gpu.legacy_model_state import (
    Qwen3_8FlashNextLegacyModelState,
)
from vllm_fl.models.qwen3_8_flash_next.gpu.ple_token_history import PLETokenHistory


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
        "vllm_fl.models.qwen3_8_flash_next.gpu.legacy_model_state.get_pp_group",
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
    state = Qwen3_8FlashNextLegacyModelState(runner)
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


class _HashEchoEmbedding(torch.nn.Module):
    """Echoes hash ids so output differences track hash-id differences."""

    def __init__(self, num_embeddings: int, embedding_dim: int, **kwargs) -> None:
        super().__init__()
        self.embedding_dim = int(embedding_dim)

    def forward(self, ids: torch.Tensor) -> torch.Tensor:
        return ids.to(torch.float32).unsqueeze(-1).expand(-1, -1, self.embedding_dim)


@pytest.fixture()
def build(monkeypatch):
    monkeypatch.setattr(ple_layer, "VocabParallelEmbedding", _HashEchoEmbedding)

    def _build(max_num_reqs: int = 8, max_total_tokens: int = 64, ngram_size: int = 3):
        config = Qwen3_8FlashNextTextConfig(
            hc_count=4,
            hc_lowrank=8,
            ple_layer_ids=[1],
            ple_embed_dim=(ngram_size - 1) * 4,
            ple_conv_kernel_size=3,
            ngram_size=ngram_size,
            heads_per_ngram=2,
            ngram_vocab_size_base=97,
            make_ngram_vocab_size_divisible_by=8,
            output_gate_type="none",
            rope_parameters={"rope_theta": 10000.0},
            layer_types=["full_attention"],
            num_hidden_layers=1,
            vocab_size=256,
            eos_token_id=0,
            split_ngram_parts=8,
            seed=1234,
        )
        module = ple_layer.Qwen3_8FlashNextNGramEmbedding(
            config,
            embedding_dim=(ngram_size - 1) * 4,
            ple_dense_layer_id=0,
            max_total_tokens=max_total_tokens,
            max_num_reqs=max_num_reqs,
            prefix=".ple",
        )
        return module.eval()

    return _build


def _reference(module, ids, qsl, context):
    count = int(qsl[-1])
    hashes = ple_hash_ids_reference(
        ids[:count],
        qsl,
        context,
        ngram_size=module.ngram_size,
        heads_per_ngram=module.heads_per_ngram,
        multipliers=module.layer_multipliers,
        vocab_sizes=module.ngram_heads_vocab_sizes,
        offsets=module.ngram_heads_offsets,
        eos_token_id=module.eos_token_id,
    )
    return module.ngram_embedding(hashes).flatten(-2)


def test_graph_padding_has_no_effect_on_real_tokens(build):
    module = build()
    ids = torch.tensor([5, 6, 0, 11, 12, 13, 14])
    qsl = torch.tensor([0, 3, 3, 7, 7], dtype=torch.int32)
    context = torch.tensor([[1, 2], [9, 9], [4, 8], [0, 0]])
    expected = _reference(module, ids, qsl, context)
    padded = module(torch.cat([ids, torch.tensor([91, 92, 93])]), qsl, context)
    torch.testing.assert_close(padded[:7], expected, rtol=0, atol=0)
    assert torch.count_nonzero(padded[7:]) == 0


def test_conv_dispatch_forwards_decode_prefill_metadata(monkeypatch):
    from types import SimpleNamespace

    layer = object.__new__(ple_layer.Qwen3_8FlashNextPLELayer)
    torch.nn.Module.__init__(layer)
    layer.conv_state_len = 9
    layer.conv_kernel_size = 4
    layer.short_conv_dilation = 3
    metadata = SimpleNamespace(
        num_prefills=1,
        num_decodes=2,
        num_decode_tokens=2,
        num_prefill_tokens=5,
        num_actual_tokens=7,
        spec_sequence_masks=None,
        state_indices_tensor=torch.tensor([1, 3, 2], dtype=torch.int32),
        has_initial_states_d=torch.tensor([True, False]),
        has_initial_states_p=torch.tensor([True]),
        non_spec_query_start_loc=torch.tensor([0, 1, 2, 7], dtype=torch.int32),
        max_prefill_query_len=5,
    )
    state = torch.zeros(4, 3, 9)
    weight = torch.zeros(3, 4)
    calls = []

    def numerical(x, output, cache, weights, indices, **kwargs):
        assert cache is state and weights is weight
        calls.append((indices.tolist(), kwargs))
        output.fill_(len(calls))

    monkeypatch.setattr(ple_layer, "ple_short_conv_", numerical)
    output = layer._short_conv_dilated_dispatch(
        torch.zeros(7, 3), metadata, state, weight
    )
    assert output[:, 0].tolist() == [1, 1, 2, 2, 2, 2, 2]
    assert [indices for indices, _ in calls] == [[1, 3], [2]]
    assert calls[0][1]["has_initial_state"] is metadata.has_initial_states_d
    assert calls[1][1]["query_start_loc"].tolist() == [0, 5]
    assert all(kwargs["dilation"] == 3 for _, kwargs in calls)


def ple_hash_ids_reference(
    input_ids: torch.Tensor,
    query_start_loc: torch.Tensor,
    ngram_context: torch.Tensor,
    *,
    ngram_size: int,
    heads_per_ngram: int,
    multipliers: torch.Tensor,
    vocab_sizes: torch.Tensor,
    offsets: torch.Tensor,
    eos_token_id: int,
) -> torch.Tensor:
    """Return per-token hash-table IDs using the exact PLE boundary rules.

    ``ngram_context`` is the ``ngram_size - 1`` token prefix for every
    request.  ``query_start_loc`` is the flattened request boundary array.
    The implementation mirrors the model code: EOS fills the left context and
    prevents n-grams from crossing a segment boundary.
    """

    input_ids = input_ids.reshape(-1).long()
    query_start_loc = query_start_loc.reshape(-1).long()
    ngram_context = ngram_context.long()
    if query_start_loc.numel() == 0:
        raise ValueError("query_start_loc must contain at least the end marker")
    num_reqs = query_start_loc.numel() - 1
    if ngram_context.shape != (num_reqs, ngram_size - 1):
        raise ValueError("ngram_context must be [requests, ngram_size - 1]")
    if int(query_start_loc[-1]) != input_ids.numel():
        raise ValueError("query_start_loc does not cover input_ids")
    total_tokens = input_ids.numel()
    if total_tokens == 0:
        return input_ids.new_empty((0, (ngram_size - 1) * heads_per_ngram))

    max_query_len = int(torch.diff(query_start_loc).max().item()) if num_reqs else 0
    packed = input_ids.new_full((num_reqs, max_query_len), eos_token_id)
    positions = torch.arange(total_tokens, device=input_ids.device)
    request_indices = torch.searchsorted(query_start_loc, positions, right=True) - 1
    request_indices.clamp_(max=max(num_reqs - 1, 0))
    columns = positions - query_start_loc[request_indices]
    packed[request_indices, columns] = input_ids

    context = torch.cat([ngram_context, packed], dim=-1)
    positions_2d = torch.arange(context.shape[1], device=input_ids.device)
    eos_positions = torch.where(
        context == eos_token_id,
        positions_2d.unsqueeze(0),
        torch.full_like(context, -1),
    )
    previous_eos_inclusive = torch.cummax(eos_positions, dim=1).values
    previous_eos = torch.cat(
        [
            eos_positions.new_full((num_reqs, 1), -1),
            previous_eos_inclusive[:, :-1],
        ],
        dim=1,
    )
    position_in_segment = positions_2d.unsqueeze(0) - previous_eos - 1

    shifted = [context]
    for shift in range(1, ngram_size):
        source = positions_2d - shift
        gather_indices = source.clamp_min(0).unsqueeze(0).expand(num_reqs, -1)
        shifted_tokens = context.gather(1, gather_indices)
        valid = (source.unsqueeze(0) >= 0) & (position_in_segment >= shift)
        shifted.append(
            torch.where(valid, shifted_tokens, context.new_full((), eos_token_id))
        )

    if multipliers.numel() != ngram_size:
        raise ValueError("multipliers must have one entry per n-gram position")
    ngram_heads = (ngram_size - 1) * heads_per_ngram
    if vocab_sizes.numel() != ngram_heads or offsets.numel() != ngram_heads:
        raise ValueError("vocab_sizes and offsets must cover every hash head")

    adjusted_columns = columns + ngram_size - 1
    blocks = []
    for ngram in range(2, ngram_size + 1):
        start = (ngram - 2) * heads_per_ngram
        end = start + heads_per_ngram
        mixed = shifted[0] * multipliers[0]
        for index in range(1, ngram):
            mixed = torch.bitwise_xor(mixed, shifted[index] * multipliers[index])
        ids = (
            torch.remainder(mixed.unsqueeze(-1), vocab_sizes[start:end])
            + offsets[start:end]
        )
        blocks.append(ids[request_indices, adjusted_columns])
    return torch.cat(blocks, dim=-1)
