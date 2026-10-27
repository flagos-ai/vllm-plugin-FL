"""Opt-in accelerator contracts; no full checkpoint or model is instantiated.

Run only in an explicitly assigned GPU window with QWEN38_RUN_DEVICE_TESTS=1.
"""

import os
from types import SimpleNamespace

import pytest

if os.environ.get("QWEN38_RUN_DEVICE_TESTS") != "1":
    pytest.skip(
        "requires an explicitly assigned device window", allow_module_level=True
    )

import torch
import torch.nn.functional as F

from vllm_fl.models.qwen3_8_flash_next.gpu import host_ngram as ngram


def hash_oracle(ids, starts, contexts, m):
    multipliers = m.layer_multipliers.tolist()
    sizes = m.ngram_heads_vocab_sizes.tolist()
    offsets = m.ngram_heads_offsets.tolist()
    output = []
    for r, (begin, end) in enumerate(zip(starts, starts[1:])):
        history = list(contexts[r])
        for token in ids[begin:end]:
            sequence = history + [token]
            last_eos = max(
                (i for i, v in enumerate(sequence[:-1]) if v == m.eos_token_id),
                default=-1,
            )
            previous = [sequence[-1]]
            for shift in range(1, m.ngram_size):
                pos = len(sequence) - 1 - shift
                previous.append(
                    sequence[pos] if pos >= 0 and pos > last_eos else m.eos_token_id
                )
            row = []
            for order in range(2, m.ngram_size + 1):
                value = 0
                for i in range(order):
                    value ^= (previous[i] * multipliers[i]) & ((1 << 64) - 1)
                if value >= 1 << 63:
                    value -= 1 << 64
                for h in range(
                    (order - 2) * m.heads_per_ngram, (order - 1) * m.heads_per_ngram
                ):
                    row.append(value % sizes[h] + offsets[h])
            output.append(row)
            history.append(token)
    return torch.tensor(output, dtype=torch.int64)


def make_host_model(monkeypatch, tp_size=1, rank=0):
    monkeypatch.setattr(ngram, "get_tensor_model_parallel_world_size", lambda: tp_size)
    monkeypatch.setattr(ngram, "get_tensor_model_parallel_rank", lambda: rank)
    # This test checks vocabulary partitioning in one process, not MCCL.
    monkeypatch.setattr(ngram, "tensor_model_parallel_all_reduce", lambda x: x)
    config = SimpleNamespace(
        ngram_size=3,
        heads_per_ngram=8,
        vocab_size=248320,
        eos_token_id=248044,
        seed=1234,
        split_ngram_parts=128,
        ngram_vocab_size_base=257,
        make_ngram_vocab_size_divisible_by=128,
    )
    with torch.device("cuda"):
        model = ngram.HostNGramEmbedding(config, 2560, 0, 32, 4, "test")
    rows = model.ngram_embedding.org_vocab_size
    full = (
        (torch.arange(rows, device="cpu")[:, None] % 31) / 8
        + torch.arange(160, device="cpu")[None, :] / 256
    ).bfloat16()
    shard = model.ngram_embedding.shard_indices
    model.ngram_embedding.weight.data.copy_(
        full[shard.org_vocab_start_index : shard.org_vocab_end_index]
    )
    return model, full


def test_host_ngram_fixed_buffer_graph_refresh_and_partition(monkeypatch):
    model, full = make_host_model(monkeypatch)
    assert model.ngram_embedding.weight.device.type == "cpu"
    assert model._host_stage.is_pinned()
    ids = torch.tensor([17, 248044, 33, 7, 19, 22], dtype=torch.int32, device="cuda")
    starts = torch.tensor([0, 3, 6], dtype=torch.int32, device="cuda")
    context = torch.tensor([[248044, 248044], [5, 6]], dtype=torch.int64, device="cuda")
    model.prepare_host_lookup(ids, starts, context)
    pointer = model.prepared_embeddings.data_ptr()
    observed = torch.empty_like(model(ids, starts, context))
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        observed.copy_(model(ids, starts, context))
    for replacement in ([4, 5, 6, 248044, 31, 29], [101, 203, 55, 1, 2, 3]):
        ids.copy_(torch.tensor(replacement, dtype=torch.int32, device="cuda"))
        model.prepare_host_lookup(ids, starts, context)
        graph.replay()
        torch.cuda.synchronize()
        hashes = hash_oracle(
            replacement, starts.cpu().tolist(), context.cpu().tolist(), model
        )
        torch.testing.assert_close(
            observed.cpu(), full[hashes].flatten(-2), atol=0, rtol=0
        )
        assert model.prepared_embeddings.data_ptr() == pointer
    # Two independent CPU shards sum to the same full embedding on device.
    shards = [make_host_model(monkeypatch, 2, rank)[0] for rank in (0, 1)]
    partials = []
    for shard in shards:
        shard.prepare_host_lookup(ids, starts, context)
        partials.append(shard(ids, starts, context).clone())
    torch.testing.assert_close(partials[0] + partials[1], observed, atol=0, rtol=0)


def conv_oracle(x, state, weights, indices, starts, accepted, k):
    result = torch.zeros_like(x)
    expected = state.clone()
    for r, index in enumerate(indices.tolist()):
        begin, end = starts[r : r + 2].tolist()
        if begin == end or index == 0:
            continue
        offset = int(accepted[r]) - 1
        history = torch.cat(
            (state[index, :, offset : offset + 9], x[begin:end].T), dim=1
        )
        conv = F.conv1d(history[None], weights[:, None], groups=x.shape[1], dilation=3)
        result[begin:end].copy_(F.silu(conv[0]).T)
        count = 9 + end - begin - 1
        expected[index, :, :count].copy_(history[:, 1 : count + 1])
    return result, expected


@pytest.mark.parametrize("k", [1, 3])
def test_ple_varlen_accepted_rollback_and_changed_graph_replay(k):
    torch.manual_seed(2026 + k)
    channels = 10240
    lengths = [k + 1, 1, 0, k]
    starts = torch.tensor(
        [0, *torch.tensor(lengths).cumsum(0).tolist()], dtype=torch.int32, device="cuda"
    )
    x = (
        torch.randn(sum(lengths) + 2, channels, dtype=torch.bfloat16, device="cuda")
        * 0.1
    )
    weights = torch.randn(channels, 4, dtype=torch.bfloat16, device="cuda") * 0.1
    state = torch.randn(
        6, 9 + k, channels, dtype=torch.bfloat16, device="cuda"
    ).transpose(1, 2)
    indices = torch.tensor([1, 3, 4, 0], dtype=torch.int32, device="cuda")
    accepted = torch.tensor([k + 1, 1, 1, k + 1], dtype=torch.int32, device="cuda")
    output = torch.empty_like(x)

    from vllm_fl.dispatch import resolve_op

    ple_provider = resolve_op("ple_short_conv_")

    def execute():
        ple_provider(
            x,
            output,
            state,
            weights,
            indices,
            dilation=3,
            query_start_loc=starts,
            accepted_tokens=accepted,
            max_query_len=k + 1,
            null_block_id=0,
        )

    expected, expected_state = conv_oracle(
        x.cpu(),
        state.cpu(),
        weights.cpu(),
        indices.cpu(),
        starts.cpu(),
        accepted.cpu(),
        k,
    )
    execute()
    torch.cuda.synchronize()
    torch.testing.assert_close(output.cpu(), expected, atol=2e-2, rtol=2e-2)
    torch.testing.assert_close(state.cpu(), expected_state, atol=0, rtol=0)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        execute()
    x.normal_(std=0.1)
    state.normal_()
    indices.copy_(torch.tensor([2, 4, 0, 1], dtype=torch.int32, device="cuda"))
    accepted.fill_(1)
    expected, expected_state = conv_oracle(
        x.cpu(),
        state.cpu(),
        weights.cpu(),
        indices.cpu(),
        starts.cpu(),
        accepted.cpu(),
        k,
    )
    graph.replay()
    torch.cuda.synchronize()
    torch.testing.assert_close(output.cpu(), expected, atol=2e-2, rtol=2e-2)
    torch.testing.assert_close(state.cpu(), expected_state, atol=0, rtol=0)
