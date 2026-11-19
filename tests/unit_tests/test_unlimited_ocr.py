# Copyright (c) 2026 BAAI. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from functools import partial

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
from vllm_fl.dispatch.backends.vendor.ascend.register_ops import (
    register_builtins as register_ascend,
)
from vllm_fl.dispatch.ops import apply_top_k_top_p, scaled_dot_product_attention


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
