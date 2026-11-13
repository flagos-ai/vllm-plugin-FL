# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Qwen3.8 integration and contract regressions."""

import json
import math
import os
from dataclasses import MISSING, fields
from pathlib import Path
from types import SimpleNamespace

import flag_gems as forward_flag_gems
import pytest
import torch
import torch.nn.functional as F

from vllm.config import set_current_vllm_config as forward_set_current_vllm_config
from vllm.distributed import parallel_state as forward_parallel_state
from vllm.engine.arg_utils import EngineArgs as forward_EngineArgs
from vllm.forward_context import set_forward_context as forward_set_forward_context
from vllm.model_executor.layers import linear as forward_linear

import vllm_fl as forward_vllm_fl
from vllm_fl.models.qwen3_8_flash_next.common.short_conv_attn import (
    PleShortConvAttentionMetadata as forward_PleShortConvAttentionMetadata,
)

pytestmark = pytest.mark.gpu
from vllm_fl.models.qwen3_8_flash_next.gpu import host_ngram as ngram, ple_layer as ple

# Forward contracts.

if os.environ.get("QWEN38_RUN_DEVICE_TESTS") != "1":
    pytest.skip(
        "requires an explicitly assigned device window", allow_module_level=True
    )

forward__engine_config_value = os.environ.get("QWEN38_ENGINE_CONFIG")
if not forward__engine_config_value:
    pytest.skip(
        "set QWEN38_ENGINE_CONFIG to a local EngineArgs fixture",
        allow_module_level=True,
    )
forward__ENGINE_CONFIG_PATH = Path(forward__engine_config_value).expanduser()
if not forward__ENGINE_CONFIG_PATH.is_file():
    pytest.skip(
        f"engine config fixture is missing: {forward__ENGINE_CONFIG_PATH}",
        allow_module_level=True,
    )
forward__fixture_config = json.loads(forward__ENGINE_CONFIG_PATH.read_text())
forward__model_value = forward__fixture_config.get("model")
if (
    not forward__model_value
    or not (Path(forward__model_value).expanduser() / "config.json").is_file()
):
    pytest.skip(
        "PLE test requires a local model config.json fixture", allow_module_level=True
    )
forward__spec_fixture = forward__fixture_config.get("speculative_config") or {}
if (
    not isinstance(forward__spec_fixture, dict)
    or forward__spec_fixture.get("num_speculative_tokens") != 1
):
    pytest.skip("PLE full-forward fixture must use MTP K1", allow_module_level=True)


forward__RESULTS = []


@pytest.fixture
def ple_results_dir(tmp_path):
    configured = os.environ.get("QWEN38_TEST_RESULTS_DIR")
    directory = Path(configured).expanduser() if configured else tmp_path
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def forward_grouped_norm(x, weight, eps, group):
    shape = x.shape
    y = x.float().reshape(-1, 4, group)
    y = y * torch.rsqrt(y.square().mean(-1, keepdim=True) + eps)
    return (y.reshape(-1, 4 * group) * (1 + weight.float())).to(x.dtype).reshape(shape)


def forward_cpu_conv(x, initial_state, weight, starts, indices, accepted, initial):
    expected_state = initial_state.clone()
    output = torch.zeros_like(x)
    for r, index in enumerate(indices.tolist()):
        a, b = starts[r : r + 2].tolist()
        if not b > a or index == 0:
            continue
        offset = int(accepted[r]) - 1 if accepted is not None else 0
        previous = x.new_zeros((x.shape[1], 9))
        if initial is None or bool(initial[r]):
            previous.copy_(initial_state[index, :, offset : offset + 9])
        history = torch.cat((previous, x[a:b].T), 1)
        value = F.conv1d(history[None], weight[:, None], groups=x.shape[1], dilation=3)
        output[a:b].copy_(F.silu(value[0]).T)
        if accepted is None:
            expected_state[index, :, :9].copy_(history[:, -9:])
        else:
            count = 9 + b - a - 1
            expected_state[index, :, :count].copy_(history[:, 1 : count + 1])
    return output, expected_state


def forward_native_reference(
    layer, hidden, ids, starts, context, state, indices, accepted, initial
):
    # Real BF16 projection shapes, evaluated without the FG dispatch context.
    embeddings = layer.ple_embedding.prepared_embeddings[: ids.numel()]
    key = F.linear(embeddings, layer.key_proj.weight)
    value = F.linear(embeddings, layer.value_proj.weight)
    key = forward_grouped_norm(
        key, layer.norm_key.weight, layer.norm_key.eps, 2560
    ).reshape(-1, 4, 2560)
    query = forward_grouped_norm(
        hidden, layer.norm_query.weight, layer.norm_query.eps, 2560
    ).reshape(-1, 4, 2560)
    score = (key * query).sum(-1, keepdim=True) / math.sqrt(2560)
    gate = torch.sigmoid(score.sign() * score.abs().clamp_min(1e-6).sqrt())
    gated = gate * value[:, None, :]
    normalized = forward_grouped_norm(
        gated.flatten(-2), layer.norm_conv.weight, layer.norm_conv.eps, 2560
    )
    conv, _ = forward_cpu_conv(
        normalized.cpu(),
        state.cpu(),
        layer.conv1d.weight[:, 0].cpu(),
        starts.cpu(),
        indices.cpu(),
        None if accepted is None else accepted.cpu(),
        None if initial is None else initial.cpu(),
    )
    return gated.flatten(-2).cpu() + conv


def forward_metadata(rows, starts, indices, accepted, initial):
    nreq = indices.numel()
    kwargs = {
        f.name: None
        for f in fields(forward_PleShortConvAttentionMetadata)
        if f.default is MISSING and f.default_factory is MISSING
    }
    kwargs.update(
        num_reqs=nreq,
        num_actual_tokens=rows,
        num_prefills=nreq if accepted is None else 0,
        num_prefill_tokens=rows if accepted is None else 0,
        num_decodes=0,
        num_decode_tokens=0,
        seq_lens=starts[1:] - starts[:-1] + 20,
        query_start_loc=starts,
        num_spec_decodes=nreq if accepted is not None else 0,
        num_spec_decode_tokens=rows if accepted is not None else 0,
        spec_query_len=2,
        max_prefill_query_len=rows,
        state_indices_tensor=indices if accepted is None else indices[:0],
        non_spec_query_start_loc=starts if accepted is None else None,
        has_initial_states_p=initial,
        spec_state_indices_tensor=indices if accepted is not None else None,
        spec_query_start_loc=starts if accepted is not None else None,
        spec_sequence_masks=torch.ones(nreq, dtype=torch.bool, device="cuda")
        if accepted is not None
        else None,
        num_accepted_tokens=accepted,
    )
    return forward_PleShortConvAttentionMetadata(**kwargs)


@pytest.mark.parametrize("rows", [1, 2, 8, 2048])
def test_complete_ple_forward_fg_eager_and_decode_graph(
    rows, monkeypatch, ple_results_dir
):
    forward_vllm_fl.register_model()
    kwargs = json.loads(forward__ENGINE_CONFIG_PATH.read_text())
    kwargs.pop("disable_log_stats", None)
    kwargs.pop("cudagraph_metrics", None)
    kwargs["tensor_parallel_size"] = 1
    cfg = forward_EngineArgs(**kwargs).create_engine_config()
    text = cfg.model_config.hf_text_config
    # Keep real H=2560/HC=4/head counts/hash multipliers and use a bounded test table.
    text.ngram_vocab_size_base = 257
    monkeypatch.setattr(ngram, "get_tensor_model_parallel_world_size", lambda: 1)
    monkeypatch.setattr(ngram, "get_tensor_model_parallel_rank", lambda: 0)
    # One-process replicated projections; do not create a distributed job.
    monkeypatch.setattr(forward_linear, "get_tensor_model_parallel_rank", lambda: 0)
    monkeypatch.setattr(
        forward_linear, "get_tensor_model_parallel_world_size", lambda: 1
    )
    monkeypatch.setattr(
        forward_parallel_state, "_TP", SimpleNamespace(rank_in_group=0, world_size=1)
    )
    prefix = "ple_complete_test"
    with forward_set_current_vllm_config(cfg):
        old_dtype = torch.get_default_dtype()
        try:
            torch.set_default_dtype(torch.bfloat16)
            with torch.device("cuda"):
                layer = ple.Qwen3_8FlashNextPLELayer(
                    text, cfg, layer_idx=1, ple_dense_layer_id=0, prefix=prefix
                )
        finally:
            torch.set_default_dtype(old_dtype)
        torch.manual_seed(104 + rows)
        with torch.no_grad():
            layer.key_proj.weight.normal_(std=1 / math.sqrt(2560))
            layer.value_proj.weight.normal_(std=1 / math.sqrt(2560))
            layer.norm_key.weight.normal_(std=0.01)
            layer.norm_query.weight.normal_(std=0.01)
            layer.norm_conv.weight.normal_(std=0.01)
            layer.conv1d.weight.normal_(std=0.05)
            layer.ple_embedding.ngram_embedding.weight.normal_(std=0.1)
        lengths = (
            [rows] if rows <= 2 else ([2] * 4 if rows == 8 else [17, 400, 800, 831])
        )
        starts = torch.tensor(
            [0, *torch.tensor(lengths).cumsum(0).tolist()],
            dtype=torch.int32,
            device="cuda",
        )
        indices = torch.tensor(
            [1, 3, 4, 5][: len(lengths)], dtype=torch.int32, device="cuda"
        )
        accepted = (
            torch.tensor(
                ([2, 1, 2, 1][: len(lengths)]), dtype=torch.int32, device="cuda"
            )
            if rows != 2048
            else None
        )
        initial = (
            torch.tensor([True, False, True, False], dtype=torch.bool, device="cuda")
            if rows == 2048
            else None
        )
        state = torch.randn(
            6, 10, 10240, dtype=torch.bfloat16, device="cuda"
        ).transpose(1, 2)
        # Respect vLLM's configured SD/DS cache orientation; the transaction
        # normalizes it back to [block,channel,history].
        layer.kv_cache = (
            state if ple.is_conv_state_dim_first() else state.transpose(-1, -2),
        )
        hidden = torch.randn(rows, 10240, dtype=torch.bfloat16, device="cuda") * 0.1
        ids = torch.arange(rows, device="cuda", dtype=torch.int32) % 1000 + 1
        context = torch.full(
            (len(lengths), 2), text.eos_token_id, dtype=torch.int64, device="cuda"
        )
        layer.prepare_host_lookup(ids, starts, context)
        meta = forward_metadata(rows, starts, indices, accepted, initial)
        saved = state.clone()
        reference = forward_native_reference(
            layer, hidden, ids, starts, context, saved, indices, accepted, initial
        )
        captured = torch.empty_like(hidden)
        original_short = layer._short_conv

        def trace_short(x, output=None):
            captured.copy_(x)
            return original_short(x)

        layer._short_conv = trace_short
        metrics = {"rows": rows, "hidden": 10240, "flag_gems": True}
        with forward_flag_gems.use_gems(
            record=True,
            once=True,
            path=str(ple_results_dir / f"ple-complete-fg-M{rows}.log"),
        ):
            with forward_set_forward_context({prefix: meta}, cfg, num_tokens=rows):
                actual = layer(hidden, ids, starts, context)
            torch.cuda.synchronize()
        torch.testing.assert_close(actual.cpu(), reference, atol=2e-2, rtol=2e-2)
        _, exact_state = forward_cpu_conv(
            captured.cpu(),
            saved.cpu(),
            layer.conv1d.weight[:, 0].cpu(),
            starts.cpu(),
            indices.cpu(),
            None if accepted is None else accepted.cpu(),
            None if initial is None else initial.cpu(),
        )
        torch.testing.assert_close(state.cpu(), exact_state, atol=0, rtol=0)
        metrics["eager_max_abs"] = (
            (actual.cpu().float() - reference.float()).abs().max().item()
        )
        metrics["state_exact_from_actual_normalized_input"] = True
        if rows != 2048:
            with forward_flag_gems.use_gems():
                graph = torch.cuda.CUDAGraph()
                with (
                    forward_set_forward_context({prefix: meta}, cfg, num_tokens=rows),
                    torch.cuda.graph(graph),
                ):
                    replayed = layer(hidden, ids, starts, context)
                hidden.normal_(std=0.1)
                ids.add_(31)
                state.normal_()
                accepted.fill_(1)
                # CPU hash/gather/H2D is outside the captured forward.
                layer.prepare_host_lookup(ids, starts, context)
                saved = state.clone()
            reference = forward_native_reference(
                layer, hidden, ids, starts, context, saved, indices, accepted, initial
            )
            graph.replay()
            torch.cuda.synchronize()
            torch.testing.assert_close(replayed.cpu(), reference, atol=2e-2, rtol=2e-2)
            _, exact_state = forward_cpu_conv(
                captured.cpu(),
                saved.cpu(),
                layer.conv1d.weight[:, 0].cpu(),
                starts.cpu(),
                indices.cpu(),
                accepted.cpu(),
                None,
            )
            torch.testing.assert_close(state.cpu(), exact_state, atol=0, rtol=0)
            metrics["graph_max_abs"] = (
                (replayed.cpu().float() - reference.float()).abs().max().item()
            )
            metrics["changed_input_graph"] = True
        else:
            metrics["graph"] = "prefill remains eager by contract"
        if rows == 1:
            before_dummy = state.clone()
            with (
                torch.inference_mode(),
                forward_set_forward_context(None, cfg, num_tokens=rows),
            ):
                dummy = layer._short_conv(captured)
            assert dummy.shape == captured.shape and bool(torch.isfinite(dummy).all())
            torch.testing.assert_close(state, before_dummy, atol=0, rtol=0)
        forward__RESULTS.append(metrics)
        (ple_results_dir / "ple-complete-forward-results.json").write_text(
            json.dumps(forward__RESULTS, indent=2)
        )


# History contracts.

if os.environ.get("QWEN38_RUN_DEVICE_TESTS") != "1":
    pytest.skip(
        "requires an explicitly assigned device window", allow_module_level=True
    )


def history_hash_oracle(ids, starts, contexts, m):
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


def history_make_host_model(monkeypatch, tp_size=1, rank=0):
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
    model, full = history_make_host_model(monkeypatch)
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
        hashes = history_hash_oracle(
            replacement, starts.cpu().tolist(), context.cpu().tolist(), model
        )
        torch.testing.assert_close(
            observed.cpu(), full[hashes].flatten(-2), atol=0, rtol=0
        )
        assert model.prepared_embeddings.data_ptr() == pointer
    # Two independent CPU shards sum to the same full embedding on device.
    shards = [history_make_host_model(monkeypatch, 2, rank)[0] for rank in (0, 1)]
    partials = []
    for shard in shards:
        shard.prepare_host_lookup(ids, starts, context)
        partials.append(shard(ids, starts, context).clone())
    torch.testing.assert_close(partials[0] + partials[1], observed, atol=0, rtol=0)


def history_conv_oracle(x, state, weights, indices, starts, accepted, k):
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

    expected, expected_state = history_conv_oracle(
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
    expected, expected_state = history_conv_oracle(
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
