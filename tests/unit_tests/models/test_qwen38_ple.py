from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F
from _qwen38_cpu import ple, ple_metadata


def native_conv(
    x,
    out,
    state,
    weight,
    indices,
    *,
    dilation,
    query_start_loc=None,
    has_initial_state=None,
    accepted_tokens=None,
    max_query_len=1,
    null_block_id=-1,
):
    """CPU state-transition oracle; does not execute the FGV GPU kernel."""
    out.zero_()
    length = (weight.shape[1] - 1) * dilation
    starts = query_start_loc.tolist()
    for req, index in enumerate(indices.tolist()):
        begin, end = starts[req : req + 2]
        if begin == end or index == null_block_id:
            continue
        offset = int(accepted_tokens[req]) - 1 if accepted_tokens is not None else 0
        initial = x.new_zeros((x.shape[1], length))
        if has_initial_state is None or bool(has_initial_state[req]):
            initial.copy_(state[index, :, offset : offset + length])
        history = torch.cat((initial, x[begin:end].T), dim=1)
        values = F.conv1d(
            history[None], weight[:, None], groups=x.shape[1], dilation=dilation
        )
        out[begin:end].copy_(F.silu(values[0]).T)
        if accepted_tokens is None:
            state[index, :, :length].copy_(history[:, -length:])
        else:
            count = length + end - begin - 1
            state[index, :, :count].copy_(history[:, 1 : count + 1])


def make_builder(k, monkeypatch):
    builder = ple_metadata.PleShortConvAttentionMetadataBuilder.__new__(
        ple_metadata.PleShortConvAttentionMetadataBuilder
    )
    builder.use_spec_decode = True
    builder.num_spec = k
    builder.use_full_cuda_graph = False
    builder.kv_cache_spec = object()
    builder.vllm_config = SimpleNamespace(
        cache_config=SimpleNamespace(mamba_cache_mode="none")
    )
    monkeypatch.setattr(
        ple_metadata, "mamba_get_block_table_tensor", lambda table, *args: table
    )
    return builder


@pytest.mark.parametrize("k", [1, 3])
def test_ragged_spec_prefill_and_zero_draft_mapping_matches_native(k, monkeypatch):
    # Requests are interleaved: spec, prefill, zero-draft decode, fresh token.
    lengths = [k + 1, 5, 1, 1]
    starts = torch.tensor(
        [0, *torch.tensor(lengths).cumsum(0).tolist()], dtype=torch.int32
    )
    total = int(starts[-1])
    common = SimpleNamespace(
        num_reqs=4,
        num_actual_tokens=total,
        query_start_loc=starts,
        query_start_loc_cpu=starts,
        block_table_tensor=torch.tensor([[1], [2], [3], [4]], dtype=torch.int32),
        seq_lens=torch.tensor(lengths) + torch.tensor([10, 0, 20, 0]),
        compute_num_computed_tokens=lambda: torch.tensor([10, 0, 20, 0]),
    )
    accepted = torch.tensor([k + 1, 1, k + 1, 1], dtype=torch.int32)
    counts = torch.tensor([k, -1, 0, -1], dtype=torch.int32)
    metadata = make_builder(k, monkeypatch).build(
        0, common, num_accepted_tokens=accepted, num_decode_draft_tokens_cpu=counts
    )
    assert metadata.spec_state_indices_tensor.tolist() == [1, 3]
    assert metadata.num_accepted_tokens.tolist() == [k + 1, k + 1]
    assert metadata.spec_query_start_loc.tolist() == [0, k + 1, k + 2]
    assert metadata.spec_token_indx.tolist() == list(range(k + 1)) + [k + 6]
    layer = ple.Qwen3_8FlashNextPLELayer.__new__(ple.Qwen3_8FlashNextPLELayer)
    torch.nn.Module.__init__(layer)
    layer.short_conv_dilation = 3
    torch.manual_seed(29)
    x = torch.randn(total, 7)
    weights = torch.randn(7, 4) * 0.1
    state = torch.randn(5, 7, 9 + k)
    expected_state = state.clone()
    expected = torch.empty_like(x)
    # Independent request-order calls, without the plugin's grouping/permutation.
    for r in range(4):
        a, b = int(starts[r]), int(starts[r + 1])
        kwargs = dict(
            dilation=3, query_start_loc=torch.tensor([0, b - a]), null_block_id=0
        )
        if r in (0, 2):
            kwargs.update(accepted_tokens=accepted[r : r + 1], max_query_len=k + 1)
        else:
            kwargs.update(has_initial_state=torch.tensor([False]))
        native_conv(
            x[a:b],
            expected[a:b],
            expected_state,
            weights,
            torch.tensor([r + 1]),
            **kwargs,
        )
    monkeypatch.setattr(layer, "_ple_short_conv", native_conv, raising=False)
    actual = layer._short_conv_dispatch_metax(x, metadata, state, weights)
    torch.testing.assert_close(actual, expected)
    torch.testing.assert_close(state, expected_state, atol=0, rtol=0)


def test_spec_padding_keeps_null_cache_and_zeroes_output(monkeypatch):
    layer = ple.Qwen3_8FlashNextPLELayer.__new__(ple.Qwen3_8FlashNextPLELayer)
    torch.nn.Module.__init__(layer)
    layer.short_conv_dilation = 3
    x, state, weights = torch.randn(6, 3), torch.randn(3, 3, 12), torch.randn(3, 4)
    null = state[0].clone()
    metadata = SimpleNamespace(
        num_actual_tokens=6,
        num_decodes=0,
        num_prefills=0,
        spec_sequence_masks=torch.tensor([True, False]),
        spec_state_indices_tensor=torch.tensor([1, 0]),
        spec_query_start_loc=torch.tensor([0, 2, 2]),
        num_accepted_tokens=torch.tensor([3, 1]),
        spec_query_len=4,
    )
    monkeypatch.setattr(layer, "_ple_short_conv", native_conv, raising=False)
    output = layer._short_conv_dispatch_metax(x, metadata, state, weights)
    assert torch.equal(state[0], null)
    assert torch.count_nonzero(output[2:]) == 0


def test_mtp_metadata_fails_without_worker_acceptance_context(monkeypatch):
    import sys

    from _qwen38_cpu import worker

    # Scope this lazy framework import stub to this test only.
    monkeypatch.setitem(sys.modules, "vllm_fl.worker.qwen3_8_flash_next", worker)
    builder = make_builder(1, monkeypatch)
    with pytest.raises(RuntimeError, match="worker hook"):
        builder.build(0, SimpleNamespace(num_reqs=1))


@pytest.mark.parametrize("speculative", [False, True])
def test_ple_state_addresses_use_wide_indices(monkeypatch, speculative):
    from types import SimpleNamespace as NS

    calls = []
    original = torch.tensor([11393], dtype=torch.int32)

    def kernel(x, out, state, weight, indices, **kwargs):
        calls.append(indices)
        assert indices.dtype == torch.int64
        assert indices.item() == 11393
        out.zero_()

    meta = NS(
        num_actual_tokens=2,
        spec_sequence_masks=torch.tensor([True]) if speculative else None,
        num_decodes=0,
        num_prefills=0 if speculative else 1,
        spec_state_indices_tensor=original,
        spec_query_start_loc=torch.tensor([0, 2]),
        num_accepted_tokens=torch.tensor([1]),
        spec_query_len=2,
        state_indices_tensor=original,
        non_spec_query_start_loc=torch.tensor([0, 2]),
        has_initial_states_p=torch.tensor([False]),
        has_initial_states_d=None,
    )
    ple.Qwen3_8FlashNextPLELayer._short_conv_dispatch_metax(
        NS(short_conv_dilation=3, _ple_short_conv=kernel),
        torch.ones(2, 7),
        meta,
        torch.empty(1, 7, 10),
        torch.ones(7, 4),
    )
    assert len(calls) == 1 and original.dtype == torch.int32
