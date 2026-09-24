"""Reference, capacity and graph checks for flattened PLE hash windows."""

from __future__ import annotations

import pytest
import torch

from vllm_fl.models.qwen3_8_flash_next.config import Qwen3_8FlashNextTextConfig
from vllm_fl.models.qwen3_8_flash_next.gpu import ple_layer


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
    from tests.qwen3_8_flash_next.reference import ple_hash_ids_reference

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


@pytest.mark.parametrize("ngram_size", [2, 3, 4, 5])
@pytest.mark.parametrize("capacity", [64, 16384])
def test_flat_hash_matches_reference_with_eos_and_empty_requests(
    build, ngram_size, capacity
):
    module = build(max_total_tokens=capacity, ngram_size=ngram_size)
    generator = torch.Generator().manual_seed(73)
    ids = torch.randint(1, 100, (17,), generator=generator)
    ids[[0, 3, 8, 9, 16]] = 0
    qsl = torch.tensor([0, 0, 5, 5, 6, 17, 17], dtype=torch.int32)
    context = torch.randint(0, 10, (6, ngram_size - 1), generator=generator)
    context[3, -1] = 0
    actual = module(ids, qsl, context)
    assert actual.numel() > 0
    assert actual.shape == (17, module.embedding_dim)
    torch.testing.assert_close(
        actual, _reference(module, ids, qsl, context), rtol=0, atol=0
    )


def test_graph_padding_has_no_effect_on_real_tokens(build):
    module = build()
    ids = torch.tensor([5, 6, 0, 11, 12, 13, 14])
    qsl = torch.tensor([0, 3, 3, 7, 7], dtype=torch.int32)
    context = torch.tensor([[1, 2], [9, 9], [4, 8], [0, 0]])
    expected = _reference(module, ids, qsl, context)
    padded = module(torch.cat([ids, torch.tensor([91, 92, 93])]), qsl, context)
    torch.testing.assert_close(padded[:7], expected, rtol=0, atol=0)
    assert torch.count_nonzero(padded[7:]) == 0


def test_chunking_and_request_reordering_preserve_hashes(build):
    module = build()
    ids = torch.tensor([5, 6, 7, 0, 9, 11, 12, 13, 14])
    qsl = torch.tensor([0, 5, 9], dtype=torch.int32)
    context = torch.tensor([[1, 2], [3, 4]])
    whole = module(ids, qsl, context)
    first = module(torch.cat([ids[:2], ids[5:7]]), torch.tensor([0, 2, 4]), context)
    # Reverse request order for the second chunk.
    second = module(
        torch.cat([ids[7:], ids[2:5]]),
        torch.tensor([0, 2, 5]),
        torch.stack([ids[5:7], ids[:2]]),
    )
    torch.testing.assert_close(torch.cat([first[:2], second[2:]]), whole[:5])
    torch.testing.assert_close(torch.cat([first[2:], second[:2]]), whole[5:])


def test_capacity_does_not_expand_hash_intermediates(build):
    from torch.utils._python_dispatch import TorchDispatchMode

    class Shapes(TorchDispatchMode):
        def __init__(self):
            super().__init__()
            self.shapes = []

        def __torch_dispatch__(self, func, types, args=(), kwargs=None):
            result = func(*args, **(kwargs or {}))
            from torch.utils._pytree import tree_flatten

            self.shapes.extend(
                tuple(x.shape)
                for x in tree_flatten(result)[0]
                if isinstance(x, torch.Tensor)
            )
            return result

    traces = []
    for capacity in [64, 16384]:
        module = build(max_num_reqs=64, max_total_tokens=capacity)
        # The only capacity-sized storage is the one-dimensional position buffer.
        assert not hasattr(module, "padded_buffer")
        with Shapes() as trace:
            output = module(
                torch.arange(64), torch.arange(65), torch.zeros(64, 2, dtype=torch.long)
            )
        traces.append(trace.shapes)
        assert output.shape == (64, 8)
    assert traces[0] == traces[1]


def test_zero_requests_returns_empty_output(build):
    module = build(max_num_reqs=4)
    out = module(torch.empty(0, dtype=torch.long), torch.tensor([0]), torch.empty(0, 2))
    assert out.shape == (0, module.embedding_dim)


def test_request_count_beyond_workspace_fails(build):
    module = build(max_num_reqs=4)
    with pytest.raises(ValueError, match="at most 4"):
        module(torch.arange(5), torch.arange(6), torch.zeros(5, 2, dtype=torch.long))


def test_invalid_history_width_fails(build):
    with pytest.raises(ValueError, match="history width"):
        build()(torch.tensor([1]), torch.tensor([0, 1]), torch.zeros(1, 3))


@pytest.mark.gpu
def test_graph_replay_uses_changed_request_boundaries(build):
    module = build().cuda()
    ids = torch.tensor([5, 6, 0, 8, 9, 10, 11, 12], device="cuda")
    qsl = torch.tensor([0, 3, 3, 7], device="cuda", dtype=torch.int32)
    context = torch.tensor([[1, 2], [3, 4], [5, 6]], device="cuda")
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(3):
            module(ids, qsl, context)
    torch.cuda.current_stream().wait_stream(stream)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        output = module(ids, qsl, context)
    for bounds in ([0, 1, 5, 8], [0, 0, 2, 6], [0, 0, 0, 0]):
        qsl.copy_(torch.tensor(bounds, device="cuda", dtype=torch.int32))
        ids.add_(1)
        context.add_(1)
        graph.replay()
        torch.testing.assert_close(output, module(ids, qsl, context), rtol=0, atol=0)
