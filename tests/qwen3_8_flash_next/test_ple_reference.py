"""CPU checks for the exact PLE hash and dilated short-conv contracts."""

from __future__ import annotations

import pytest
import torch
import torch.nn.functional as F

from .reference import (
    dilated_short_conv_reference,
    nth_prime_after,
    ple_hash_ids_reference,
)


def _hash_geometry(ngram_size: int = 3, heads_per_ngram: int = 2):
    ngram_heads = (ngram_size - 1) * heads_per_ngram
    vocab_sizes = torch.tensor(
        [nth_prime_after(97, i + 1) for i in range(ngram_heads)], dtype=torch.long
    )
    offsets = torch.zeros(ngram_heads, dtype=torch.long)
    multipliers = torch.tensor(
        [1000003 + i * 10007 for i in range(ngram_size)], dtype=torch.long
    )
    return multipliers, vocab_sizes, offsets


def test_ple_hash_shape_and_range_for_variable_requests():
    multipliers, vocab_sizes, offsets = _hash_geometry()
    # Flattened requests exercise the same query_start_loc path used by vLLM.
    input_ids = torch.tensor([5, 6, 7, 11, 12, 13, 14], dtype=torch.long)
    query_start_loc = torch.tensor([0, 3, 3, 7], dtype=torch.int32)
    context = torch.tensor([[2, 3], [9, 9], [4, 8]], dtype=torch.long)
    ids = ple_hash_ids_reference(
        input_ids,
        query_start_loc,
        context,
        ngram_size=3,
        heads_per_ngram=2,
        multipliers=multipliers,
        vocab_sizes=vocab_sizes,
        offsets=offsets,
        eos_token_id=0,
    )
    assert ids.shape == (input_ids.numel(), 4)
    for head, size in enumerate(vocab_sizes.tolist()):
        assert bool((ids[:, head] >= offsets[head]).all())
        assert bool((ids[:, head] < offsets[head] + size).all())


def test_ple_hash_does_not_cross_eos_segment_boundary():
    multipliers, vocab_sizes, offsets = _hash_geometry()
    eos = 0
    # The final token is after EOS. Changing tokens before EOS must not change
    # its hash IDs; otherwise a request can leak an earlier sentence into PLE.
    query_start_loc = torch.tensor([0, 4], dtype=torch.int32)
    context = torch.tensor([[eos, eos]], dtype=torch.long)
    first = torch.tensor([3, 17, eos, 23], dtype=torch.long)
    changed = torch.tensor([91, 42, eos, 23], dtype=torch.long)
    kwargs = dict(
        query_start_loc=query_start_loc,
        ngram_context=context,
        ngram_size=3,
        heads_per_ngram=2,
        multipliers=multipliers,
        vocab_sizes=vocab_sizes,
        offsets=offsets,
        eos_token_id=eos,
    )
    first_ids = ple_hash_ids_reference(first, **kwargs)
    changed_ids = ple_hash_ids_reference(changed, **kwargs)
    torch.testing.assert_close(first_ids[-1], changed_ids[-1])
    assert not torch.equal(first_ids[1], changed_ids[1])


def test_ple_hash_is_deterministic_at_small_and_empty_boundaries():
    multipliers, vocab_sizes, offsets = _hash_geometry(ngram_size=2, heads_per_ngram=1)
    empty = ple_hash_ids_reference(
        torch.empty(0, dtype=torch.long),
        torch.tensor([0, 0], dtype=torch.int32),
        torch.tensor([[0]], dtype=torch.long),
        ngram_size=2,
        heads_per_ngram=1,
        multipliers=multipliers,
        vocab_sizes=vocab_sizes,
        offsets=offsets,
        eos_token_id=0,
    )
    assert empty.shape == (0, 1)
    token = torch.tensor([8], dtype=torch.long)
    kwargs = dict(
        query_start_loc=torch.tensor([0, 1], dtype=torch.int32),
        ngram_context=torch.tensor([[0]], dtype=torch.long),
        ngram_size=2,
        heads_per_ngram=1,
        multipliers=multipliers,
        vocab_sizes=vocab_sizes,
        offsets=offsets,
        eos_token_id=0,
    )
    first = ple_hash_ids_reference(token, **kwargs)
    second = ple_hash_ids_reference(token, **kwargs)
    assert torch.equal(first, second)


@pytest.mark.parametrize("hidden,kernel,dilation", [(1, 1, 1), (2, 3, 2), (4, 4, 3)])
def test_dilated_short_conv_stateful_matches_explicit_fconv(hidden, kernel, dilation):
    torch.manual_seed(hidden * 100 + kernel * 10 + dilation)
    x = torch.randn(7, hidden, dtype=torch.float32)
    weight = torch.randn(hidden, kernel, dtype=torch.float32)
    state_len = (kernel - 1) * dilation
    initial_state = torch.randn(hidden, state_len, dtype=torch.float32)

    expected, expected_state = dilated_short_conv_reference(
        x, weight, dilation=dilation, state=initial_state
    )
    # This is the exact F.conv1d formulation used by the plugin's prefill and
    # decode fallbacks: one depthwise channel, causal history, then SiLU.
    history = torch.cat([initial_state, x.transpose(0, 1)], dim=-1).unsqueeze(0)
    direct = F.conv1d(
        history,
        weight.unsqueeze(1),
        groups=hidden,
        dilation=dilation,
    ).squeeze(0).transpose(0, 1)
    torch.testing.assert_close(expected, F.silu(direct), rtol=1e-5, atol=1e-5)
    if state_len:
        torch.testing.assert_close(expected_state, history[..., -state_len:].squeeze(0))
    else:
        assert expected_state.shape == (hidden, 0)


def test_dilated_short_conv_chunking_preserves_state_and_output():
    torch.manual_seed(123)
    x = torch.randn(11, 3)
    weight = torch.randn(3, 4)
    initial = torch.randn(3, 9)
    whole, whole_state = dilated_short_conv_reference(
        x, weight, dilation=3, state=initial
    )
    first, state = dilated_short_conv_reference(
        x[:4], weight, dilation=3, state=initial
    )
    second, state = dilated_short_conv_reference(
        x[4:], weight, dilation=3, state=state
    )
    torch.testing.assert_close(torch.cat([first, second]), whole)
    torch.testing.assert_close(state, whole_state)
