# Copyright (c) 2026 BAAI. All rights reserved.
"""Real Sunrise/FlagGems routing checks for vLLM 0.32 empty builds."""

import importlib

import pytest
import torch

pytestmark = [pytest.mark.gpu, pytest.mark.flaggems]


@pytest.fixture(scope="module")
def sunrise_router():
    from vllm.platforms import current_platform

    if current_platform.device_type != "ptpu":
        pytest.skip("Sunrise PTPU routing regression")
    from vllm_fl.worker.runtime import _initialize_fl_runtime

    _initialize_fl_runtime(0)
    return importlib.import_module("vllm_fl.ops.fused_moe.router")


@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
@pytest.mark.parametrize("indices_type", [torch.int32, torch.int64])
@pytest.mark.parametrize("renormalize", [False, True])
@pytest.mark.parametrize("padded", [False, True])
def test_softmax_routing_matches_cpu(
    sunrise_router, monkeypatch, dtype, indices_type, renormalize, padded
):
    rows, experts, topk = 65, 256, 8
    # Every row contains a distinct permutation of values exactly representable
    # in all three input dtypes, so index comparisons do not depend on tie order.
    columns = torch.arange(experts)[None, :]
    row_offsets = torch.arange(rows)[:, None]
    logits = (((columns * 73 + row_offsets * 29) % experts - 128) * 0.125).to(dtype)
    mask = torch.arange(rows) % 3 == 1
    device_logits = logits.to("ptpu:0")
    device_mask = mask.to("ptpu:0") if padded else None
    monkeypatch.setattr(sunrise_router, "_get_padding_mask", lambda count: device_mask)
    weights, ids, source_rows = sunrise_router.fused_topk(
        torch.empty((rows, 16), device="ptpu:0", dtype=dtype),
        device_logits,
        topk,
        renormalize,
        indices_type,
    )
    scores = logits.float().softmax(-1)
    expected_ids = torch.argsort(scores, dim=-1, descending=True, stable=True)[:, :topk]
    expected_weights = scores.gather(1, expected_ids)
    if renormalize:
        expected_weights /= expected_weights.sum(-1, keepdim=True)
    if padded:
        expected_ids[mask] = -1
    expected_rows = torch.arange(topk)[None, :] * rows + torch.arange(rows)[:, None]
    torch.testing.assert_close(weights.cpu(), expected_weights, rtol=3e-5, atol=3e-6)
    torch.testing.assert_close(ids.cpu().long(), expected_ids)
    torch.testing.assert_close(source_rows.cpu().long(), expected_rows)
    assert weights.dtype == torch.float32
    assert ids.dtype == indices_type
    assert source_rows.dtype == torch.int32
