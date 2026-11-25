# Copyright (c) 2026 BAAI. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from functools import partial
from types import SimpleNamespace

import pytest
import torch
from torch.nn.functional import scaled_dot_product_attention as torch_sdpa

from vllm_fl.dispatch import (
    OpManager,
    SelectionPolicy,
    manager as dispatch_manager,
    policy_context,
)
from vllm_fl.dispatch.backends.reference.register_ops import (
    register_builtins as register_reference,
)
from vllm_fl.dispatch.backends.vendor.ascend.impl.attention_mask import (
    AttentionMaskBuilder,
)
from vllm_fl.dispatch.backends.vendor.ascend.impl.ocr_kv_cache import (
    expand_head_major_slots,
)
from vllm_fl.dispatch.backends.vendor.ascend.register_ops import (
    register_builtins as register_ascend,
)
from vllm_fl.dispatch.ops import apply_top_k_top_p, scaled_dot_product_attention


@pytest.mark.parametrize(
    ("architecture", "options", "expected"),
    [
        ("UnlimitedOCRForCausalLM", {}, False),
        ("UnlimitedOCRForCausalLM", {"ocr_tp1_optimized": False}, False),
        ("UnlimitedOCRForCausalLM", {"ocr_tp1_optimized": True}, True),
        ("OtherModel", {"ocr_tp1_optimized": True}, False),
    ],
)
def test_ocr_policy_is_model_scoped(architecture, options, expected):
    from vllm_fl.models.ocr_runtime import ocr_tp1_optimized

    hf = SimpleNamespace(model_type="unlimited-ocr", architectures=[architecture])
    config = SimpleNamespace(
        model_config=SimpleNamespace(hf_config=hf, dtype=torch.bfloat16),
        parallel_config=SimpleNamespace(tensor_parallel_size=1),
        additional_config=options,
    )
    assert ocr_tp1_optimized(config) is expected
    if expected:
        config.additional_config = {"ocr_tp1_optimized": "false"}
        with pytest.raises(ValueError):
            ocr_tp1_optimized(config)


@pytest.mark.parametrize(
    ("async_scheduling", "tp", "optimized", "accepted"),
    [
        (False, 2, False, True),
        (True, 1, True, True),
        (True, 2, False, False),
        (False, 2, True, False),
    ],
)
def test_ocr_scheduler_support(monkeypatch, async_scheduling, tp, optimized, accepted):
    from vllm.config import CUDAGraphMode

    from vllm_fl import utils
    from vllm_fl.models.ocr_runtime import validate_ocr_config

    ns = SimpleNamespace
    hf = ns(
        model_type="unlimited-ocr",
        architectures=["UnlimitedOCRForCausalLM"],
        rswa_window=128,
    )
    config = ns(
        model_config=ns(
            hf_config=hf,
            hf_text_config=hf,
            dtype=torch.bfloat16,
            quantization=None,
            use_mla=False,
            use_fp64_gumbel=False,
            max_model_len=32768,
        ),
        scheduler_config=ns(
            async_scheduling=async_scheduling,
            max_num_seqs=4,
            max_num_batched_tokens=2048,
        ),
        parallel_config=ns(
            tensor_parallel_size=tp,
            pipeline_parallel_size=1,
            data_parallel_size=1,
            prefill_context_parallel_size=1,
            decode_context_parallel_size=1,
            enable_expert_parallel=False,
            use_ubatching=False,
        ),
        compilation_config=ns(cudagraph_mode=CUDAGraphMode.NONE),
        attention_config=ns(backend=None),
        cache_config=ns(
            cache_dtype="auto",
            calculate_kv_scales=False,
            enable_prefix_caching=False,
            sliding_window=None,
        ),
        additional_config={"ocr_tp1_optimized": optimized},
        use_v2_model_runner=False,
        quant_config=None,
        speculative_config=None,
        lora_config=None,
    )
    monkeypatch.setattr(utils, "is_oot_enabled", lambda: True)
    monkeypatch.setattr(utils, "get_oot_whitelist", lambda: None)
    monkeypatch.setattr(utils, "get_oot_blacklist", lambda: None)
    monkeypatch.setattr(utils, "use_flaggems", lambda: False)
    monkeypatch.setattr(utils, "get_flag_gems_whitelist_blacklist", lambda: ([], []))
    if accepted:
        assert validate_ocr_config(config, device_type="npu", vendor="ascend")
    else:
        with pytest.raises(ValueError, match="TP1|synchronous"):
            validate_ocr_config(config, device_type="npu", vendor="ascend")


def test_selected_top6_binding_propagates_execution_failure(monkeypatch):
    from vllm_fl.ops.fused_moe import router

    logits = torch.randn(4, 64)
    calls = []

    def fail(scores, k):
        assert scores is logits and k == 6
        calls.append(k)
        raise RuntimeError("routing launch failed")

    context = SimpleNamespace(additional_kwargs={"ocr_prefill_top6": fail})
    monkeypatch.setattr(router, "is_forward_context_available", lambda: True)
    monkeypatch.setattr(router, "get_forward_context", lambda: context)
    with pytest.raises(RuntimeError, match="routing launch failed"):
        router._fl_grouped_topk(logits, logits, 6, False, 1, 1)
    assert calls == [6]
    context.additional_kwargs.clear()
    weights, ids = router._fl_grouped_topk(logits, logits, 6, False, 1, 1)
    reference = logits.softmax(-1).topk(6).values
    torch.testing.assert_close(weights.sort(descending=True).values, reference)
    torch.testing.assert_close(logits.softmax(-1).gather(1, ids.long()), weights)


def _reference_rswa_mask(prefix, seq, query, kv, qmax, window):
    result = torch.ones((len(prefix), 1, qmax, kv), dtype=torch.int8)
    for request, query_len in enumerate(query):
        for offset in range(query_len):
            position = seq[request] - query_len + offset
            for key in range(seq[request]):
                if key <= position and (
                    key < prefix[request] or key > position - window
                ):
                    result[request, 0, offset, key] = 0
    return result


@pytest.mark.parametrize("limit", ["512", "2048"])
def test_rswa_mask_matches_scalar_positions(monkeypatch, limit):
    monkeypatch.setenv("VLLM_FL_RSWA_MASK_MAX_MIB", limit)
    prefix, seq, query = [2, 0, 5], [13, 8, 5], [1, 3, 0]
    builder = AttentionMaskBuilder(torch.device("cpu"), ocr_mode=True)
    actual = builder.get_rswa_attn_mask(
        torch.tensor(prefix), torch.tensor(seq), torch.tensor(query), 17, 5, 3
    )
    torch.testing.assert_close(
        actual, _reference_rswa_mask(prefix, seq, query, 17, 5, 3)
    )


@pytest.mark.parametrize("dtype", [torch.int32, torch.int64])
@pytest.mark.parametrize("block", [128, 256])
def test_ocr_head_major_slots_preserve_pages_and_skip_padding(dtype, block):
    pages, heads = 4, 10
    logical = torch.tensor([3 * block + 1, -1, 0, block - 1, -1, block], dtype=dtype)
    slots = expand_head_major_slots(logical, heads, block).reshape(-1, heads)
    assert slots.dtype == torch.int32
    addresses = torch.arange(pages * heads * block, dtype=torch.int32).reshape(
        pages, heads, block
    )
    for row, slot in enumerate(logical.tolist()):
        if slot >= 0:
            page, offset = divmod(slot, block)
            torch.testing.assert_close(slots[row], addresses[page, :, offset])
    padding = slots[logical < 0]
    assert (padding < 0).all() and padding.unique().numel() == padding.numel()


@pytest.fixture
def ocr_dispatch(monkeypatch):
    manager = OpManager(register_builtins=False)
    register_reference(manager.registry)
    register_ascend(manager.registry)
    monkeypatch.setattr(dispatch_manager, "_default_manager", manager)
    with policy_context(
        SelectionPolicy.from_dict(
            strict=True,
            per_op_order={
                "rotary_embedding": ["impl:vendor.ascend.torch_npu"],
                "scaled_dot_product_attention": ["vendor:ascend", "reference"],
                "apply_top_k_top_p": ["reference"],
            },
        )
    ):
        yield manager


def test_upstream_logit_filter_retains_top_k_ties(ocr_dispatch):
    pytest.importorskip("vllm.v1.sample.ops.topk_topp_sampler")
    logits = torch.tensor([[2.0, 2.0, 1.0]]).repeat(8, 1)
    result = apply_top_k_top_p(logits, torch.ones(8, dtype=torch.long), None)
    assert result is logits
    torch.testing.assert_close(
        result.softmax(dim=-1), torch.tensor([[0.5, 0.5, 0.0]]).repeat(8, 1)
    )
    assert ocr_dispatch.get_selected_impl_id("apply_top_k_top_p") == "reference.torch"


@pytest.fixture
def ocr_npu_device():
    torch_npu = pytest.importorskip("torch_npu")
    if not torch_npu.npu.is_available():
        pytest.skip("NPU unavailable")
    return torch.device("npu", torch_npu.npu.current_device())


def test_npu_sam_preserves_additive_bias(ocr_dispatch, ocr_npu_device):
    q, k, v = (torch.randn(1, 12, 196, 64, dtype=torch.bfloat16) for _ in range(3))
    bias = torch.randn(1, 12, 196, 196).to(torch.bfloat16)
    out = scaled_dot_product_attention(
        q.to(ocr_npu_device),
        k.to(ocr_npu_device),
        v.to(ocr_npu_device),
        bias.to(ocr_npu_device),
    ).cpu()
    expected = torch_sdpa(q.float(), k.float(), v.float(), attn_mask=bias.float())
    torch.testing.assert_close(out.float(), expected, atol=0.02, rtol=0.04)
    assert ocr_dispatch._called_ops["scaled_dot_product_attention"] == "vendor.ascend"


@pytest.mark.parametrize("interleaved", [False, True])
def test_npu_rotary_layout_and_optional_key(ocr_dispatch, ocr_npu_device, interleaved):
    q = torch.randn(3, 4, 128, dtype=torch.bfloat16)
    angles = torch.randn(8, 32)
    cos, sin = angles.cos().to(torch.bfloat16), angles.sin().to(torch.bfloat16)
    positions = torch.tensor([5, 1, 3])
    q_dev = q.to(ocr_npu_device)[..., ::2]
    q_in = q_dev.clone()
    actual, key = ocr_dispatch.call(
        "rotary_embedding",
        None,
        q_dev,
        None,
        cos.to(ocr_npu_device),
        sin.to(ocr_npu_device),
        positions.to(ocr_npu_device),
        rotary_interleaved=interleaved,
        inplace=False,
    )
    q32 = q[..., ::2].float()
    x = (
        torch.complex(q32[..., ::2], q32[..., 1::2])
        if interleaved
        else torch.complex(*q32.chunk(2, dim=-1))
    )
    phase = torch.complex(cos[positions].float(), sin[positions].float())[:, None, :]
    rotated = x * phase
    expected = (
        torch.view_as_real(rotated).flatten(-2)
        if interleaved
        else torch.cat((rotated.real, rotated.imag), dim=-1)
    )
    assert key is None
    torch.testing.assert_close(q_dev, q_in, atol=0, rtol=0)
    torch.testing.assert_close(actual.cpu().float(), expected, atol=0.032, rtol=0.032)


def test_batched_ngram_matches_core():
    from vllm.model_executor.models.deepseek_ocr import (
        NGramPerReqLogitsProcessor,
        NoRepeatNGramLogitsProcessor,
    )

    from vllm_fl.models.ocr_ngram import NGramPerReqLogitsProcessorFL

    expected = NGramPerReqLogitsProcessor(None, None, False)
    actual = NGramPerReqLogitsProcessorFL(None, None, False)
    state = partial(NoRepeatNGramLogitsProcessor(2, 35, {2}), [1, 2, 1, 3, 1])
    expected.req_info = actual.req_info = {1: state}
    logits = torch.zeros(2, 8)
    result = actual.apply(logits.clone())
    torch.testing.assert_close(result, expected.apply(logits))
    assert result[1, 3].isneginf() and result[1, 2] == 0


def test_ngram_tracks_updated_request_histories_after_row_reuse():
    from vllm.model_executor.models.deepseek_ocr import NoRepeatNGramLogitsProcessor

    from vllm_fl.models.ocr_ngram import NGramPerReqLogitsProcessorFL

    processor = NGramPerReqLogitsProcessorFL(None, None, False)
    first, second = [1, 3, 1], [2, 4, 2]
    rule = NoRepeatNGramLogitsProcessor(2, 35, set())
    processor.req_info = {0: partial(rule, first), 1: partial(rule, second)}

    def blocked():
        return processor.apply(torch.zeros(2, 8)).isneginf().nonzero().tolist()

    assert blocked() == [[0, 3], [1, 4]]
    second.extend([5, 2])
    processor.req_info = {0: partial(rule, second), 1: partial(rule, [6])}
    assert blocked() == [[0, 4], [0, 5]]


@pytest.mark.parametrize("missing_kernel", [False, True])
def test_sam_dispatch_falls_back_before_launch(
    ocr_dispatch, monkeypatch, missing_kernel
):
    from vllm_fl.dispatch.backends.vendor.ascend import register_ops
    from vllm_fl.dispatch.backends.vendor.ascend.ascend import AscendBackend

    monkeypatch.setattr(AscendBackend, "_available", True)
    monkeypatch.setattr(register_ops, "has_npu_operator", lambda *_: not missing_kernel)
    q, k, v = (torch.randn(1, 3, 7, 8) for _ in range(3))
    actual = scaled_dot_product_attention(q, k, v)
    torch.testing.assert_close(actual, torch_sdpa(q, k, v))
    assert ocr_dispatch._called_ops["scaled_dot_product_attention"] == "reference.torch"


@pytest.mark.parametrize(
    ("enabled", "whitelist", "accepted"),
    [
        (False, "", True),
        (True, "silu_and_mul", True),
        (True, "", False),
        (True, "silu_and_mul,rms_norm", False),
        (False, "silu_and_mul", False),
    ],
)
def test_ocr_flaggems_whitelist(monkeypatch, enabled, whitelist, accepted):
    from vllm_fl import utils
    from vllm_fl.models.ocr_runtime import _ocr_flaggems_silu

    monkeypatch.setattr(utils, "use_flaggems", lambda: enabled)
    monkeypatch.setenv("VLLM_FL_FLAGOS_WHITELIST", whitelist)
    monkeypatch.delenv("VLLM_FL_FLAGOS_BLACKLIST", raising=False)
    monkeypatch.delenv("VLLM_FL_USE_FLAGGEMS_VLLM", raising=False)
    if accepted:
        assert _ocr_flaggems_silu() is enabled
    else:
        with pytest.raises(ValueError, match="FlagGems|FLAGOS_WHITELIST"):
            _ocr_flaggems_silu()


@pytest.mark.parametrize("fragmented", [False, True])
@pytest.mark.parametrize("reverse", [False, True])
def test_rswa_skip_hidden_pages_preserves_read_tokens(reverse, fragmented):
    from types import SimpleNamespace

    from vllm_fl.dispatch.backends.vendor.ascend.impl.attention import (
        _stage_rswa_read_view,
    )

    rows = [(6010, 6999), (6010, 7011), (2938, 4000), (6010, 6011), (8192, 9400)]
    if fragmented:
        rows *= 5
    if reverse:
        rows.reverse()
    prefixes, lengths = map(list, zip(*rows))
    lengths.append(1)  # FULL bucket padding with no real request.
    block, width, window = 256, 10005, 128
    pages = 42  # Include an incomplete mask page and excess table columns.
    keys = torch.arange(width)
    mask = torch.ones(len(lengths), 1, 1, width, dtype=torch.int8)
    for i, (prefix, length) in enumerate(rows):
        mask[i] = ~((keys < prefix) | ((keys >= length - window) & (keys < length)))
    table = torch.randperm(len(lengths) * pages).reshape(len(lengths), pages)
    live = SimpleNamespace(
        seq_lens_list=lengths,
        rswa_read_prefixes=prefixes,
        kv_block_size=block,
        rswa_window=window,
        attn_mask=mask,
        block_tables=table,
    )
    stage_mask, stage_table = mask.clone(), table.clone()
    compact = _stage_rswa_read_view(stage_mask, stage_table, live)
    for row, length in enumerate(lengths):
        skip = 2 if length in (6999, 7011) else 0
        # ceil(6010/256)=24, floor((6999-128)/256)=26.
        assert compact[row] == length - skip * block
        actual = stage_table[row, keys // block] * block + keys % block
        expected = table[row, keys // block] * block + keys % block
        torch.testing.assert_close(
            actual[stage_mask[row].flatten() == 0], expected[mask[row].flatten() == 0]
        )
        assert stage_mask[row, ..., compact[row] :].all()
    live.rswa_read_prefixes = []
    assert _stage_rswa_read_view(mask.clone(), table.clone(), live) is lengths


def test_wide_rswa_mask_groups_bound_and_order():
    from unittest.mock import Mock

    from vllm_fl.dispatch.backends.vendor.ascend.impl.attention import (
        _bounded_rswa_groups,
    )

    builder = Mock()
    queries = torch.tensor([1] * 31 + [6081] + [1] * 32)
    sequences = torch.tensor([6100] * 31 + [6081] + [6300] * 32)
    prefixes = torch.full_like(queries, 6000)
    groups = _bounded_rswa_groups(builder, prefixes, sequences, queries, 32768, 128)
    assert [(g.token_start, g.token_end) for g in groups] == [
        (0, 31),
        (31, 6112),
        (6112, 6144),
    ]
    allocated = sum(
        c.kwargs["query_lens"].numel() * c.kwargs["max_query_len"] * 32768
        for c in builder.get_rswa_attn_mask.call_args_list
    )
    assert allocated == 6144 * 32768 < 8192 * (1 << 20)
    prefixes[31] = 6081
    groups = _bounded_rswa_groups(builder, prefixes, sequences, queries, 32768, 128)
    assert groups[1].prompt_causal and groups[1].actual_seq_lengths_q == [6081]
