"""Numerical validation for the fixed-anchor INT8 prefill plan."""

from __future__ import annotations

import torch


def _fixed_anchor_selfcheck(
    device, decode_workspace, config, scratch_tokens, lpt, *, split_prefix_mask=False
):
    from vllm_fl.attention import hybrid_selfcheck, kv_int8, mixed_select
    from vllm_fl.attention.gqa_hybrid import HybridPlan

    query_lengths = [17, 1, 0, 33]
    lengths = [4097, 33, 0, 6145]
    case, _, _ = hybrid_selfcheck._make_case(query_lengths, lengths, 16, device)
    query_storage = torch.empty(
        (case["q"].shape[0] * 2, 16, 128),
        dtype=torch.bfloat16,
        device=device,
    )
    query_storage[::2].copy_(case["q"])
    case["q"] = query_storage[::2]
    case["out"] = torch.empty_like(query_storage)[::2]

    pages = case["k"].shape[0]
    raw = torch.empty(pages * kv_int8.PAGE_BYTES, dtype=torch.int8, device=device)
    cache = torch.as_strided(
        raw,
        (pages, 2, 16, 2, 128),
        (kv_int8.PAGE_BYTES, 4096, 256, 128, 1),
    )
    kv_int8.kv_int8_write(
        case["k"].flatten(0, 1).contiguous(),
        case["v"].flatten(0, 1).contiguous(),
        cache,
        torch.arange(pages * 16, device=device, dtype=torch.int64),
    )
    values = cache.detach().cpu().float()
    scales = kv_int8.scale_view(cache).detach().cpu().float()
    key = values[:, 0] * scales[:, 0].permute(0, 2, 1)[..., None]
    value = values[:, 1] * scales[:, 1].permute(0, 2, 1)[..., None]
    reference_case = dict(case)
    reference_case["k"], reference_case["v"] = key, value
    decode_reference = hybrid_selfcheck._reference(reference_case)
    reference_case["k"] = key.to(torch.bfloat16).float()
    reference_case["v"] = value.to(torch.bfloat16).float()
    prefill_reference = hybrid_selfcheck._reference(reference_case)
    reference = torch.empty_like(prefill_reference)
    starts = case["cu_seqlens_q"].detach().cpu().tolist()
    for row, query_len in enumerate(query_lengths):
        if query_len:
            source = prefill_reference if query_len > 1 else decode_reference
            reference[starts[row] : starts[row + 1]] = source[
                starts[row] : starts[row + 1]
            ]
    case.update(
        k=cache[:, 0],
        v=cache[:, 1],
        kv_scales=kv_int8.half_base(cache),
        kv_cache=cache,
    )

    def build(*, fixed_anchor=False):
        return HybridPlan.build(
            query_lengths,
            lengths,
            device=device,
            pm=128,
            pn=32,
            layers=1,
            split_prefix_mask=split_prefix_mask,
            prefill_order="lpt" if lpt else "natural",
            prefill_reblock=True,
            prefill_wide=True,
            prefill_fixed_anchor=fixed_anchor,
            kv_int8_scratch_tokens=scratch_tokens,
            decode_external=True,
        )

    wide = build()
    fixed = build(fixed_anchor=True)
    if not fixed.prefill_fixed_anchor or fixed.fixed_anchor_fallback_flags is None:
        raise RuntimeError("fixed-anchor fixture did not pass its plan gate")
    decode = mixed_select.decode_factory((decode_workspace, config))(1)

    def run(plan, stage):
        plan.reset()
        case["out"].fill_(float("nan"))
        mixed_select.run_mixed(plan, case, "separate", decode)
        torch.cuda.synchronize(device)
        result = case["out"].float().clone()
        if not torch.isfinite(result).all():
            raise RuntimeError(f"{stage} fixture produced non-finite output")
        return result

    wide_output = run(wide, "wide")
    fixed_output = run(fixed, "fixed-anchor")
    reference = reference.to(device)
    for name, actual in (("wide", wide_output), ("fixed-anchor", fixed_output)):
        if (
            (actual - reference).abs()
            > hybrid_selfcheck.ATOL + hybrid_selfcheck.RTOL * reference.abs()
        ).any():
            raise RuntimeError(f"{name} fixture differs from the CPU reference")
    normal_flags = fixed.fixed_anchor_fallback_flags[: len(fixed.prefill)].cpu()
    fast_tasks = int((normal_flags == 0).sum())
    if not fast_tasks:
        raise RuntimeError("fixed-anchor fixture never selected the fast path")

    start = 0
    for query_len in query_lengths:
        if query_len > 1:
            scaled = case["q"][start : start + query_len].detach().cpu() * 16
            case["q"][start : start + query_len].copy_(scaled.to(device))
        start += query_len
    fallback_wide = run(wide, "high-Q wide")
    fallback_fixed = run(fixed, "high-Q fixed-anchor fallback")
    fallback_flags = fixed.fixed_anchor_fallback_flags[: len(fixed.prefill)].cpu()
    if fallback_flags.numel() == 0 or not bool((fallback_flags == 1).all()):
        raise RuntimeError("high-Q fixture did not force every task to fallback")
    if not torch.equal(fallback_fixed, fallback_wide):
        raise RuntimeError("fixed-anchor fallback differs from the wide path")
    return {
        "fast_tasks": fast_tasks,
        "high_q_fallback_tasks": fallback_flags.numel(),
        "high_q_bitwise_equal_to_wide": True,
    }
