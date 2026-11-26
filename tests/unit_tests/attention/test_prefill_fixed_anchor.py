"""Fixed-anchor geometry and numerical coverage."""

import pytest
import torch

from vllm_fl.attention import kv_int8 as k8
from vllm_fl.attention.decode_config import DecodeConfig


@pytest.mark.parametrize(
    ("length", "expected_anchor"),
    [(4097, True), (4080, False), (65537, False)],
)
def test_plan_length_gate_preserves_wide(monkeypatch, length, expected_anchor):
    from vllm_fl.attention.gqa_hybrid import HybridPlan

    monkeypatch.setattr(
        k8,
        "prefill_scratch",
        lambda device, tokens: torch.empty(
            (tokens // k8.PAGE_TOKENS, 2, k8.PAGE_TOKENS, 2, 128),
            dtype=torch.bfloat16,
            device="meta",
        ),
    )
    tokens = ((length + k8.PAGE_TOKENS - 1) // k8.PAGE_TOKENS) * k8.PAGE_TOKENS
    plan = HybridPlan.build(
        [17, 1],
        [length, 33],
        device="cpu",
        pm=128,
        pn=32,
        prefill_reblock=True,
        prefill_wide=True,
        prefill_fixed_anchor=True,
        kv_int8_scratch_tokens=tokens,
        decode_external=True,
    )
    assert plan.prefill_wide
    assert plan.prefill_fixed_anchor is expected_anchor


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")
def test_fixed_anchor_selfcheck_covers_fast_and_fallback_tasks():
    from vllm_fl.attention import prefill_wide

    device = torch.device("cuda", 0)
    config = DecodeConfig(
        variant=3,
        tile=32,
        num_warps=2,
        num_stages=1,
        table=((1, 1),),
    )
    result = prefill_wide._fixed_anchor_selfcheck(
        device,
        k8.DecodeWorkspace(device),
        config,
        k8.prefill_scratch_cap_tokens(),
        lpt=True,
        split_prefix_mask=True,
    )
    assert result["fast_tasks"] > 0
    assert result["high_q_fallback_tasks"] > 0
    assert result["high_q_bitwise_equal_to_wide"]
