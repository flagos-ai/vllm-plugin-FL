"""Actual PLE norm/projection/gate/conv under FlagGems, opt-in GPU test.

Set QWEN38_RUN_DEVICE_TESTS=1 and QWEN38_ENGINE_CONFIG to an EngineArgs JSON
fixture referring to a local model config and MTP K1. No model weights are
loaded: the test retains the real projection shapes and uses a tiny N-gram
table. QWEN38_TEST_RESULTS_DIR optionally selects the artifact directory;
otherwise pytest's temporary directory is used. Missing fixtures skip before
any accelerator/runtime imports.
"""

import json
import math
import os
from dataclasses import MISSING, fields
from pathlib import Path
from types import SimpleNamespace

import pytest

if os.environ.get("QWEN38_RUN_DEVICE_TESTS") != "1":
    pytest.skip(
        "requires an explicitly assigned device window", allow_module_level=True
    )

_engine_config_value = os.environ.get("QWEN38_ENGINE_CONFIG")
if not _engine_config_value:
    pytest.skip(
        "set QWEN38_ENGINE_CONFIG to a local EngineArgs fixture",
        allow_module_level=True,
    )
_ENGINE_CONFIG_PATH = Path(_engine_config_value).expanduser()
if not _ENGINE_CONFIG_PATH.is_file():
    pytest.skip(
        f"engine config fixture is missing: {_ENGINE_CONFIG_PATH}",
        allow_module_level=True,
    )
_fixture_config = json.loads(_ENGINE_CONFIG_PATH.read_text())
_model_value = _fixture_config.get("model")
if not _model_value or not (Path(_model_value).expanduser() / "config.json").is_file():
    pytest.skip(
        "PLE test requires a local model config.json fixture", allow_module_level=True
    )
_spec_fixture = _fixture_config.get("speculative_config") or {}
if (
    not isinstance(_spec_fixture, dict)
    or _spec_fixture.get("num_speculative_tokens") != 1
):
    pytest.skip("PLE full-forward fixture must use MTP K1", allow_module_level=True)

import flag_gems
import torch
import torch.nn.functional as F

from vllm.config import set_current_vllm_config
from vllm.distributed import parallel_state
from vllm.engine.arg_utils import EngineArgs
from vllm.forward_context import set_forward_context
from vllm.model_executor.layers import linear

import vllm_fl
from vllm_fl.models.qwen3_8_flash_next.common.short_conv_attn import (
    PleShortConvAttentionMetadata,
)
from vllm_fl.models.qwen3_8_flash_next.gpu import host_ngram as ngram, ple_layer as ple

_RESULTS = []


@pytest.fixture
def ple_results_dir(tmp_path):
    configured = os.environ.get("QWEN38_TEST_RESULTS_DIR")
    directory = Path(configured).expanduser() if configured else tmp_path
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def grouped_norm(x, weight, eps, group):
    shape = x.shape
    y = x.float().reshape(-1, 4, group)
    y = y * torch.rsqrt(y.square().mean(-1, keepdim=True) + eps)
    return (y.reshape(-1, 4 * group) * (1 + weight.float())).to(x.dtype).reshape(shape)


def cpu_conv(x, initial_state, weight, starts, indices, accepted, initial):
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


def native_reference(
    layer, hidden, ids, starts, context, state, indices, accepted, initial
):
    # Real BF16 projection shapes, evaluated without the FG dispatch context.
    embeddings = layer.ple_embedding.prepared_embeddings[: ids.numel()]
    key = F.linear(embeddings, layer.key_proj.weight)
    value = F.linear(embeddings, layer.value_proj.weight)
    key = grouped_norm(key, layer.norm_key.weight, layer.norm_key.eps, 2560).reshape(
        -1, 4, 2560
    )
    query = grouped_norm(
        hidden, layer.norm_query.weight, layer.norm_query.eps, 2560
    ).reshape(-1, 4, 2560)
    score = (key * query).sum(-1, keepdim=True) / math.sqrt(2560)
    gate = torch.sigmoid(score.sign() * score.abs().clamp_min(1e-6).sqrt())
    gated = gate * value[:, None, :]
    normalized = grouped_norm(
        gated.flatten(-2), layer.norm_conv.weight, layer.norm_conv.eps, 2560
    )
    conv, _ = cpu_conv(
        normalized.cpu(),
        state.cpu(),
        layer.conv1d.weight[:, 0].cpu(),
        starts.cpu(),
        indices.cpu(),
        None if accepted is None else accepted.cpu(),
        None if initial is None else initial.cpu(),
    )
    return gated.flatten(-2).cpu() + conv


def metadata(rows, starts, indices, accepted, initial):
    nreq = indices.numel()
    kwargs = {
        f.name: None
        for f in fields(PleShortConvAttentionMetadata)
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
    return PleShortConvAttentionMetadata(**kwargs)


@pytest.mark.parametrize("rows", [1, 2, 8, 2048])
def test_complete_ple_forward_fg_eager_and_decode_graph(
    rows, monkeypatch, ple_results_dir
):
    vllm_fl.register_model()
    kwargs = json.loads(_ENGINE_CONFIG_PATH.read_text())
    kwargs.pop("disable_log_stats", None)
    kwargs.pop("cudagraph_metrics", None)
    kwargs["tensor_parallel_size"] = 1
    cfg = EngineArgs(**kwargs).create_engine_config()
    text = cfg.model_config.hf_text_config
    # Keep real H=2560/HC=4/head counts/hash multipliers and use a bounded test table.
    text.ngram_vocab_size_base = 257
    monkeypatch.setattr(ngram, "get_tensor_model_parallel_world_size", lambda: 1)
    monkeypatch.setattr(ngram, "get_tensor_model_parallel_rank", lambda: 0)
    # One-process replicated projections; do not create a distributed job.
    monkeypatch.setattr(linear, "get_tensor_model_parallel_rank", lambda: 0)
    monkeypatch.setattr(linear, "get_tensor_model_parallel_world_size", lambda: 1)
    monkeypatch.setattr(
        parallel_state, "_TP", SimpleNamespace(rank_in_group=0, world_size=1)
    )
    prefix = "ple_complete_test"
    with set_current_vllm_config(cfg):
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
        meta = metadata(rows, starts, indices, accepted, initial)
        saved = state.clone()
        reference = native_reference(
            layer, hidden, ids, starts, context, saved, indices, accepted, initial
        )
        captured = torch.empty_like(hidden)
        original_short = layer._short_conv

        def trace_short(x, output=None):
            captured.copy_(x)
            return original_short(x)

        layer._short_conv = trace_short
        metrics = {"rows": rows, "hidden": 10240, "flag_gems": True}
        with flag_gems.use_gems(
            record=True,
            once=True,
            path=str(ple_results_dir / f"ple-complete-fg-M{rows}.log"),
        ):
            with set_forward_context({prefix: meta}, cfg, num_tokens=rows):
                actual = layer(hidden, ids, starts, context)
            torch.cuda.synchronize()
        torch.testing.assert_close(actual.cpu(), reference, atol=2e-2, rtol=2e-2)
        _, exact_state = cpu_conv(
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
            with flag_gems.use_gems():
                graph = torch.cuda.CUDAGraph()
                with (
                    set_forward_context({prefix: meta}, cfg, num_tokens=rows),
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
            reference = native_reference(
                layer, hidden, ids, starts, context, saved, indices, accepted, initial
            )
            graph.replay()
            torch.cuda.synchronize()
            torch.testing.assert_close(replayed.cpu(), reference, atol=2e-2, rtol=2e-2)
            _, exact_state = cpu_conv(
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
                set_forward_context(None, cfg, num_tokens=rows),
            ):
                dummy = layer._short_conv(captured)
            assert dummy.shape == captured.shape and bool(torch.isfinite(dummy).all())
            torch.testing.assert_close(state, before_dummy, atol=0, rtol=0)
        _RESULTS.append(metrics)
        (ple_results_dir / "ple-complete-forward-results.json").write_text(
            json.dumps(_RESULTS, indent=2)
        )
