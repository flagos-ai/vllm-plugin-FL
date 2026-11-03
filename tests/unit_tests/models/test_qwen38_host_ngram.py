from types import SimpleNamespace

import pytest
import torch
from _qwen38_cpu import ngram


def oracle(ids, starts, contexts, multipliers, sizes, offsets, heads, eos):
    result = []
    for r, (begin, end) in enumerate(zip(starts, starts[1:])):
        history = list(contexts[r])
        for token in ids[begin:end]:
            sequence = history + [token]
            previous_eos = max(
                (i for i, v in enumerate(sequence[:-1]) if v == eos), default=-1
            )
            shifted = [sequence[-1]]
            for distance in range(1, len(multipliers)):
                position = len(sequence) - 1 - distance
                shifted.append(
                    sequence[position]
                    if position >= 0 and position > previous_eos
                    else eos
                )
            values = []
            for order in range(2, len(multipliers) + 1):
                value = 0
                for i in range(order):
                    value ^= (shifted[i] * multipliers[i]) & ((1 << 64) - 1)
                if value >= 1 << 63:
                    value -= 1 << 64
                for h in range((order - 2) * heads, (order - 1) * heads):
                    values.append(value % sizes[h] + offsets[h])
            result.append(values)
            history.append(token)
    return torch.tensor(result, dtype=torch.int64)


def test_hash_matches_independent_signed_integer_oracle_and_eos():
    ids, starts, context = [3, 9, 5, 7, 4], [0, 3, 3, 5], [[9, 2], [6, 7], [8, 9]]
    multipliers, sizes, offsets = [2**62 + 3, 19, 29], [23, 31, 37, 41], [0, 23, 54, 91]
    actual = ngram.compute_ngram_ids_cpu(
        torch.tensor(ids),
        torch.tensor(starts),
        torch.tensor(context),
        multipliers=torch.tensor(multipliers),
        vocab_sizes=torch.tensor(sizes),
        vocab_offsets=torch.tensor(offsets),
        heads_per_ngram=2,
        eos_token_id=9,
    )
    assert torch.equal(
        actual, oracle(ids, starts, context, multipliers, sizes, offsets, 2, 9)
    )


def test_hash_is_chunk_invariant():
    params = dict(
        multipliers=torch.tensor([5, 7, 11]),
        vocab_sizes=torch.tensor([23, 29]),
        vocab_offsets=torch.tensor([0, 23]),
        heads_per_ngram=1,
        eos_token_id=9,
    )
    whole = ngram.compute_ngram_ids_cpu(
        torch.tensor([1, 2, 9, 3, 4]),
        torch.tensor([0, 5]),
        torch.tensor([[9, 9]]),
        **params,
    )
    first = ngram.compute_ngram_ids_cpu(
        torch.tensor([1, 2, 9]), torch.tensor([0, 3]), torch.tensor([[9, 9]]), **params
    )
    second = ngram.compute_ngram_ids_cpu(
        torch.tensor([3, 4]), torch.tensor([0, 2]), torch.tensor([[2, 9]]), **params
    )
    assert torch.equal(torch.cat((first, second)), whole)


def make_model(monkeypatch, rank):
    monkeypatch.setattr(ngram, "get_tensor_model_parallel_world_size", lambda: 2)
    monkeypatch.setattr(ngram, "get_tensor_model_parallel_rank", lambda: rank)
    monkeypatch.setattr(ngram, "tensor_model_parallel_all_reduce", lambda value: value)
    cfg = SimpleNamespace(
        ngram_size=3,
        heads_per_ngram=1,
        eos_token_id=9,
        vocab_size=64,
        split_ngram_parts=4,
        seed=1234,
        ngram_vocab_size_base=17,
        make_ngram_vocab_size_divisible_by=8,
    )
    return ngram.HostNGramEmbedding(cfg, 8, 0, 16, 4, "ple")


def test_host_shards_load_and_sum_to_unsharded_embedding(monkeypatch):
    models = [make_model(monkeypatch, rank) for rank in (0, 1)]
    rows = models[0].ngram_embedding.org_vocab_size
    full = (torch.arange(rows)[:, None] % 23 + torch.arange(4)[None, :] / 8).bfloat16()
    shard_size = (rows + 3) // 4
    for model in models:
        model.load_weights(
            [
                (
                    f"ngram_embedding.shard_{i}.weight",
                    full[i * shard_size : (i + 1) * shard_size],
                )
                for i in range(4)
            ]
        )
        assert model.ngram_embedding.weight.device.type == "cpu"
        assert model.ngram_embedding.weight.shape[0] == rows // 2
    ids, starts, context = (
        torch.tensor([1, 2, 3, 4]),
        torch.tensor([0, 2, 4]),
        torch.tensor([[9, 9], [8, 9]]),
    )
    outputs = []
    for model in models:
        with pytest.raises(RuntimeError, match="not prepared"):
            model(ids, starts, context)
        pointer = model.prepared_embeddings.data_ptr()
        model.prepare_host_lookup(ids, starts, context)
        outputs.append(model(ids, starts, context).clone())
        # The model forward reads a fixed allocation; it cannot invoke hashing.
        assert model.prepared_embeddings.data_ptr() == pointer
    m = models[0]
    hashes = ngram.compute_ngram_ids_cpu(
        ids,
        starts,
        context,
        multipliers=m.layer_multipliers,
        vocab_sizes=m.ngram_heads_vocab_sizes,
        vocab_offsets=m.ngram_heads_offsets,
        heads_per_ngram=1,
        eos_token_id=9,
    )
    expected = full[hashes].flatten(-2)
    assert torch.equal(outputs[0] + outputs[1], expected)


def test_graph_padding_is_zero_and_host_weight_stays_host(monkeypatch):
    model = make_model(monkeypatch, 0)
    model.ngram_embedding.weight.data.fill_(1)
    ids, starts, context = (
        torch.tensor([1, 2, 7, 7]),
        torch.tensor([0, 2]),
        torch.tensor([[9, 9]]),
    )
    model.prepare_host_lookup(ids, starts, context)
    assert torch.count_nonzero(model(ids, starts, context)[2:]) == 0
    model.to("cpu")
    assert model.layer_multipliers.device.type == "cpu"
    assert model.ngram_embedding.weight.device.type == "cpu"
    with pytest.raises(ValueError, match="must not be omitted"):
        model(ids, None, context)


def test_shared_owner_loads_once_and_matches_hash_oracle_without_reduction(monkeypatch):
    from test_qwen38_shared_buffer import load_buffer

    model = make_model(monkeypatch, 0)
    rows = model.ngram_embedding.org_vocab_size
    model.ngram_embedding = ngram.HostVocabEmbedding(
        rows, 4, tp_rank=0, tp_size=1, dtype=torch.bfloat16
    )
    group = SimpleNamespace(rank_in_group=0, world_size=1, ranks=[0], cpu_group=None)
    model._shared_lookup = load_buffer()(group, 16, 8, torch.bfloat16)
    model.host_lookup_mode = "shared_buffer"

    def forbidden_reduce(_):
        raise AssertionError("shared CPU lookup must not reduce on GPUs")

    monkeypatch.setattr(ngram, "tensor_model_parallel_all_reduce", forbidden_reduce)
    full = (torch.arange(rows)[:, None] % 23 + torch.arange(4)[None, :] / 8).bfloat16()
    model.load_weights(
        [
            (f"ngram_embedding.shard_{i}.weight", part)
            for i, part in enumerate(full.chunk(4))
        ]
    )
    assert torch.equal(model.ngram_embedding.weight, full)
    for ids, starts, context in [
        ([1, 2, 9, 4, 7, 7], [0, 2, 4], [[9, 9], [3, 9]]),
        ([5, 6, 8, 8], [0, 1, 2], [[1, 2], [9, 4]]),
    ]:
        ids, starts, context = map(torch.tensor, (ids, starts, context))
        model.prepare_host_lookup(ids, starts, context)
        actual = int(starts[-1])
        hashes = oracle(
            ids[:actual].tolist(),
            starts.tolist(),
            context.tolist(),
            model.layer_multipliers.tolist(),
            model.ngram_heads_vocab_sizes.tolist(),
            model.ngram_heads_offsets.tolist(),
            1,
            9,
        )
        assert torch.equal(model.prepared_embeddings[:actual], full[hashes].flatten(-2))
        assert not torch.count_nonzero(model.prepared_embeddings[actual:])
    assert model._shared_lookup.productions == 2
    model._shared_lookup.close()


def test_shared_consumer_validates_checkpoint_without_allocating_table(monkeypatch):
    model = make_model(monkeypatch, 1)
    rows = model.ngram_embedding.org_vocab_size
    model.ngram_embedding = ngram.HostVocabEmbedding(
        rows, 4, tp_rank=0, tp_size=1, dtype=torch.bfloat16, allocate=False
    )
    model._shared_lookup = object()
    full = torch.zeros(rows, 4, dtype=torch.bfloat16)
    loaded = model.load_weights(
        [
            (f"ngram_embedding.shard_{i}.weight", part)
            for i, part in enumerate(full.chunk(4))
        ]
    )
    assert loaded == {"ngram_embedding.weight"}
    assert model.ngram_embedding.weight.numel() == 0
    with pytest.raises(ValueError, match="Shape mismatch"):
        model.load_weights([("ngram_embedding.shard_0.weight", torch.zeros(1, 4))])


@pytest.mark.parametrize("order,heads", [(2, 1), (3, 8), (4, 2), (6, 1)])
def test_packed_hash_with_empty_requests_strides_and_signed_overflow(order, heads):
    generator = torch.Generator().manual_seed(311 + order)
    lengths = [0, 1, 0, 17, 1, 33, 0]
    starts = [0]
    for length in lengths:
        starts.append(starts[-1] + length)
    ids = torch.randint(0, 12, (starts[-1], 2), generator=generator)[:, 0]
    context = torch.randint(0, 12, (len(lengths), order - 1, 2), generator=generator)[
        ..., 0
    ]
    offsets = torch.tensor([[s, s] for s in starts], dtype=torch.int32)[:, 0]
    ids[::7] = 9
    context[::2] = 9
    multipliers = torch.tensor([2**62 + 3 + 2 * i for i in range(order)])
    sizes = torch.arange((order - 1) * heads, dtype=torch.int64) + 23
    vocab_offsets = sizes.cumsum(0) - sizes
    actual = ngram.compute_ngram_ids_cpu(
        ids,
        offsets,
        context,
        multipliers=multipliers,
        vocab_sizes=sizes,
        vocab_offsets=vocab_offsets,
        heads_per_ngram=heads,
        eos_token_id=9,
    )
    expected = oracle(
        ids.tolist(),
        starts,
        context.tolist(),
        multipliers.tolist(),
        sizes.tolist(),
        vocab_offsets.tolist(),
        heads,
        9,
    )
    assert torch.equal(actual, expected)


@pytest.mark.parametrize("requests", [0, 3])
def test_empty_packed_hash_has_exact_shape(requests):
    actual = ngram.compute_ngram_ids_cpu(
        torch.empty(0, dtype=torch.int64),
        torch.zeros(requests + 1, dtype=torch.int64),
        torch.full((requests, 2), 9, dtype=torch.int64),
        multipliers=torch.tensor([5, 7, 11]),
        vocab_sizes=torch.tensor([23, 29]),
        vocab_offsets=torch.tensor([0, 23]),
        heads_per_ngram=1,
        eos_token_id=9,
    )
    assert actual.shape == (0, 2)
    assert actual.dtype == torch.int64
    assert actual.device.type == "cpu"


def test_uva_weight_fragments_wait_for_outer_completion(monkeypatch):
    model = make_model(monkeypatch, 0)
    calls = []
    model._uva_lookup = SimpleNamespace(finish_loading=lambda *args: calls.append(args))
    model._ple_embedding_load_state = {"shards": set(), "full": False, "buffers": set()}
    rows = model.ngram_embedding.org_vocab_size
    shard_size = (rows + 3) // 4
    full = torch.zeros((rows, 4), dtype=torch.bfloat16)
    for i in range(4):
        model.load_weights(
            [
                (
                    f"ngram_embedding.shard_{i}.weight",
                    full[i * shard_size : (i + 1) * shard_size],
                )
            ]
        )
        assert not calls
    with pytest.raises(ValueError, match="hash buffer"):
        model._finish_host_lookup_loading()
    model.load_weights(
        [
            (name, getattr(model, name).clone())
            for name in (
                "layer_multipliers",
                "ngram_heads_vocab_sizes",
                "ngram_heads_offsets",
            )
        ]
    )
    assert not calls
    model._finish_host_lookup_loading()
    assert len(calls) == 1


def test_uva_direct_weight_transaction_registers_once(monkeypatch):
    model = make_model(monkeypatch, 0)
    calls = []
    model._uva_lookup = SimpleNamespace(finish_loading=lambda *args: calls.append(args))
    rows = model.ngram_embedding.org_vocab_size
    shard_size = (rows + 3) // 4
    full = torch.zeros((rows, 4), dtype=torch.bfloat16)
    weights = [
        (name, getattr(model, name).clone())
        for name in (
            "layer_multipliers",
            "ngram_heads_vocab_sizes",
            "ngram_heads_offsets",
        )
    ]
    weights += [
        (
            f"ngram_embedding.shard_{i}.weight",
            full[i * shard_size : (i + 1) * shard_size],
        )
        for i in range(4)
    ]
    model.load_weights(weights)
    assert len(calls) == 1
    assert not hasattr(model, "_ple_embedding_load_state")
