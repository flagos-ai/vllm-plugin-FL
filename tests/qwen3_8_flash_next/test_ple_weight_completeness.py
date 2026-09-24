"""Exercise the real PLE loader with small, explicitly partitioned weights."""

from types import SimpleNamespace

import pytest
import torch

from vllm_fl.models.qwen3_8_flash_next.gpu.ple_layer import (
    Qwen3_8FlashNextNGramEmbedding,
    validate_ple_embedding_weights,
)


def make_loader(start=0, end=8, vocab_size=8, parts=2):
    module = Qwen3_8FlashNextNGramEmbedding.__new__(Qwen3_8FlashNextNGramEmbedding)
    torch.nn.Module.__init__(module)
    module.split_ngram_parts = parts
    for name in ("layer_multipliers", "ngram_heads_offsets", "ngram_heads_vocab_sizes"):
        module.register_buffer(name, torch.zeros(2, dtype=torch.long))
    embedding = torch.nn.Module()
    embedding.org_vocab_size = vocab_size
    embedding.embedding_dim = 2
    embedding.shard_indices = SimpleNamespace(
        org_vocab_start_index=start, org_vocab_end_index=end
    )
    embedding.weight = torch.nn.Parameter(
        torch.full((end - start, 2), -999.0), requires_grad=False
    )
    module.ngram_embedding = embedding
    return module


def shards(vocab_size=8, parts=2):
    weight = torch.arange(vocab_size * 2, dtype=torch.float32).reshape(-1, 2)
    size = (vocab_size + parts - 1) // parts
    return weight, [
        (f"ngram_embedding.shard_{i}.weight", weight[i * size : (i + 1) * size])
        for i in range(parts)
    ]


def test_missing_shard_cannot_certify_parameter():
    module = make_loader()
    _, weights = shards()
    with pytest.raises(ValueError, match="Missing PLE embedding shards.*1"):
        module.load_weights(weights[:1])


@pytest.mark.parametrize(
    "start,end,ids", [(0, 8, [1, 0]), (2, 6, [1, 0]), (4, 8, [1]), (4, 8, [0, 1])]
)
def test_coverage_accepts_reordering_tp_intersections_and_irrelevant_shards(
    start, end, ids
):
    module = make_loader(start, end)
    expected, weights = shards()
    assert module.load_weights(weights[i] for i in ids) == {"ngram_embedding.weight"}
    torch.testing.assert_close(module.ngram_embedding.weight, expected[start:end])


def test_only_nonintersecting_shard_is_not_local_coverage():
    _, weights = shards()
    with pytest.raises(ValueError, match="Missing PLE embedding shards.*1"):
        make_loader(4, 8).load_weights(weights[:1])


def test_tail_and_empty_shards():
    expected, weights = shards(vocab_size=5, parts=8)
    module = make_loader(3, 5, vocab_size=5, parts=8)
    assert module.load_weights(reversed(weights)) == {"ngram_embedding.weight"}
    torch.testing.assert_close(module.ngram_embedding.weight, expected[3:])


@pytest.mark.parametrize("order", ["duplicate", "full_first", "full_last"])
def test_rejects_duplicate_and_mixed_representations(order):
    full, weights = shards()
    bad = {
        "duplicate": [weights[0], weights[0]],
        "full_first": [("ngram_embedding.weight", full), *weights],
        "full_last": [*weights, ("ngram_embedding.weight", full)],
    }[order]
    with pytest.raises(ValueError, match="[Dd]uplicate|[Mm]ixed"):
        make_loader().load_weights(bad)


def test_full_weight_and_no_embedding_paths():
    full, _ = shards()
    module = make_loader()
    assert module.load_weights([("ngram_embedding.weight", full)]) == {
        "ngram_embedding.weight"
    }
    torch.testing.assert_close(module.ngram_embedding.weight, full)
    assert make_loader().load_weights([]) == set()


@pytest.mark.parametrize(
    "name,shape",
    [
        ("ngram_embedding.shard_-1.weight", (4, 2)),
        ("ngram_embedding.shard_2.weight", (4, 2)),
        ("ngram_embedding.shard_0.weight", (3, 2)),
    ],
)
def test_invalid_shard_name_index_or_shape(name, shape):
    with pytest.raises(ValueError):
        make_loader().load_weights([(name, torch.ones(shape))])


class FragmentedModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.first = make_loader()
        self.second = make_loader()

    @validate_ple_embedding_weights
    def load_weights(self, weights):
        from vllm.model_executor.models.utils import AutoWeightsLoader

        return AutoWeightsLoader(self).load_weights(weights)


def fragmented_weights():
    _, weights = shards()
    return [
        (f"{owner}.{name}", value)
        for name, value in weights
        for owner in ("first", "second")
    ]


def test_real_autoloader_accumulates_noncontiguous_module_fragments():
    model = FragmentedModel()
    assert model.load_weights(fragmented_weights()) == {
        "first.ngram_embedding.weight",
        "second.ngram_embedding.weight",
    }
    expected, _ = shards()
    torch.testing.assert_close(model.first.ngram_embedding.weight, expected)
    torch.testing.assert_close(model.second.ngram_embedding.weight, expected)
    # A subsequent complete model reload starts a new transaction.
    assert len(model.load_weights(fragmented_weights())) == 2


@pytest.mark.parametrize("failure", ["missing", "duplicate", "mixed"])
def test_fragmented_load_still_rejects_incomplete_or_ambiguous_checkpoints(failure):
    weights = fragmented_weights()
    if failure == "missing":
        weights.pop()
    elif failure == "duplicate":
        weights.append(weights[0])
    else:
        full, _ = shards()
        weights.append(("first.ngram_embedding.weight", full))
    with pytest.raises(ValueError, match="Missing|Duplicate|Mixed"):
        FragmentedModel().load_weights(weights)
