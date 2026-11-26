"""Numerically verified INT8 mixed-attention execution modes and prefill tiles."""

from __future__ import annotations

import dataclasses
import os
import time
from collections.abc import Callable
from functools import partial

import torch

from vllm.logger import init_logger

from vllm_fl import device_profiles, envs as fl_envs
from vllm_fl.attention import hybrid_selfcheck as sc, kv_int8 as k8
from vllm_fl.attention.gqa_hybrid import HybridPlan

logger = init_logger("vllm.vllm_fl.mixed_select")
ENV_REBLOCK = "VLLM_FL_GQA_HYBRID_PREFILL_REBLOCK"
ENV_WIDE = "VLLM_FL_GQA_HYBRID_PREFILL_WIDE"
ENV_FIXED_ANCHOR = "VLLM_FL_GQA_HYBRID_PREFILL_FIXED_ANCHOR"
TILES = {
    "served": (32, 64, False),
    "reblock": (128, 32, True),
    "p64x32": (64, 32, False),
}
MODES = ("fused", "separate")
_DECODE_LIMITS = (8, 24, 40)
_QUERY_LIMITS = (127, 255, 511, 1023)


def rule_tile(max_prefill_query: int, reblock_allowed: bool) -> str:
    return "reblock" if reblock_allowed and max_prefill_query >= 128 else "served"


def bucket_key(num_decode: int, max_prefill_query: int, max_kv: int) -> str:
    query_bucket = sum(max_prefill_query > limit for limit in _QUERY_LIMITS)
    context_bucket = int(max_kv >= 8192)
    if num_decode <= 0:
        return f"p:c{context_bucket}:q{query_bucket}"
    decode_bucket = sum(num_decode > limit for limit in _DECODE_LIMITS)
    return f"m:d{decode_bucket}:c{context_bucket}:q{query_bucket}"


@dataclasses.dataclass(frozen=True)
class Choice:
    mode: str = "fused"
    tile: str = "served"
    fused_decode_max: int = 0  # measured small-decode exception within this bucket


@dataclasses.dataclass
class MixedDecision:
    table: dict[str, Choice] = dataclasses.field(default_factory=dict)
    reblock_allowed: bool = True
    prefill_wide: bool = False
    prefill_fixed_anchor: bool = False
    reason: str = "rule-based"

    def lookup(self, num_decode: int, max_prefill_query: int, max_kv: int) -> Choice:
        found = self.table.get(bucket_key(num_decode, max_prefill_query, max_kv))
        tile = (
            found.tile if found else rule_tile(max_prefill_query, self.reblock_allowed)
        )
        if num_decode <= 0:
            return Choice("fused", tile)
        mode = found.mode if found else "fused"
        if found and num_decode <= found.fused_decode_max:
            mode = "fused"
        return Choice(mode, tile)


def decode_segments(config, num_decode: int) -> int:
    return config.lookup(num_decode)


def standalone_decode(kwargs: dict, workspace, config, segments: int) -> None:
    """The INT8 kernel skips prefill rows; use the full batch metadata."""
    k8.kv_int8_decode(
        kwargs["q"],
        kwargs["kv_cache"],
        kwargs["kv_scales"],
        kwargs["out"],
        kwargs["cu_seqlens_q"],
        kwargs["seqused_k"],
        kwargs["block_table"],
        kwargs["softmax_scale"],
        workspace,
        config=config,
        segments=segments,
    )


def decode_factory(decode):
    if decode is None:
        return None
    workspace, config = decode

    def make_decode(num_decode):
        return partial(
            standalone_decode,
            workspace=workspace,
            config=config,
            segments=decode_segments(config, num_decode),
        )

    return make_decode


def run_mixed(
    plan: HybridPlan,
    kwargs: dict,
    mode: str,
    decode: Callable | None = None,
    *,
    layer: int = 0,
):
    if mode == "fused":
        plan.run(kwargs, mode="fused", layer=layer)
    elif mode == "separate":
        if decode is None or not plan.decode_external:
            raise RuntimeError("separate mixed mode needs an external decode kernel")
        plan.run(kwargs, mode="separate", layer=layer)
        decode(kwargs)
    else:
        raise ValueError(f"unknown mixed mode: {mode}")


class Verifier:
    """Small CPU FP32 comparisons for each selected (tile, mode), once at startup."""

    BATCHES = (
        ([1, 1, 130, 40, 1], [200, 33, 500, 70, 1000]),
        ([130, 1, 1, 200, 1], [130, 300, 17, 500, 1]),
    )

    def __init__(self, device, page, lpt, make_decode, scratch_tokens=0):
        self.device, self.page, self.lpt = device, page, lpt
        self.make_decode, self.scratch_tokens = make_decode, scratch_tokens
        self.memo, self.cases = {}, []

    def _cases(self):
        if not self.cases:
            for qlens, lengths in self.BATCHES:
                case, _, _ = sc._make_case(qlens, lengths, self.page, self.device)
                pages = case["k"].shape[0]
                raw = torch.empty(
                    pages * k8.PAGE_BYTES, dtype=torch.int8, device=self.device
                )
                cache = torch.as_strided(
                    raw, (pages, 2, 16, 2, 128), (k8.PAGE_BYTES, 4096, 256, 128, 1)
                )
                k8.kv_int8_write(
                    case["k"].flatten(0, 1).contiguous(),
                    case["v"].flatten(0, 1).contiguous(),
                    cache,
                    torch.arange(pages * 16, device=self.device, dtype=torch.int64),
                )
                scales, values = k8.scale_view(cache).cpu().float(), cache.cpu().float()
                ref_case = dict(case)
                ref_case["k"] = values[:, 0] * scales[:, 0].permute(0, 2, 1)[..., None]
                ref_case["v"] = values[:, 1] * scales[:, 1].permute(0, 2, 1)[..., None]
                case.update(
                    k=cache[:, 0],
                    v=cache[:, 1],
                    kv_scales=k8.half_base(cache),
                    kv_cache=cache,
                )
                self.cases.append((qlens, lengths, case, sc._reference(ref_case)))
        return self.cases

    def check(self, tile: str, mode: str, *, prefill_wide=False) -> tuple[bool, str]:
        key = tile, mode, prefill_wide
        if key not in self.memo:
            try:
                self.memo[key] = self._check(tile, mode, prefill_wide=prefill_wide)
            except Exception as exc:
                self.memo[key] = (
                    False,
                    f"{type(exc).__name__}: {(str(exc) or type(exc).__name__).splitlines()[0]}",
                )
        return self.memo[key]

    def _check(self, tile, mode, *, prefill_wide=False):
        pm, pn, reblock = TILES[tile]
        if mode == "separate" and self.make_decode is None:
            return False, "no external decode kernel"
        for qlens, lengths, case, ref in self._cases():
            plan = HybridPlan.build(
                qlens,
                lengths,
                device=self.device,
                pm=pm,
                pn=pn,
                layers=1,
                prefill_order="lpt" if self.lpt else "natural",
                prefill_reblock=reblock,
                prefill_wide=prefill_wide,
                kv_int8_scratch_tokens=self.scratch_tokens,
                decode_external=mode == "separate",
            )
            plan.reset()
            case["out"].fill_(float("nan"))
            decode = self.make_decode(qlens.count(1)) if mode == "separate" else None
            run_mixed(plan, case, mode, decode)
            out = case["out"].float()
            ref = ref.to(self.device)
            if not torch.isfinite(out).all() or not torch.isfinite(ref).all():
                return False, "non-finite output or CPU reference"
            if ((out - ref).abs() > sc.ATOL + sc.RTOL * ref.abs()).any():
                return False, "mismatch vs CPU FP32 attention"
        return True, "ok"


_decisions = {}


def resolve(
    device, *, page=16, lpt=True, reblock_ok=True, decode=None, scratch_tokens=0
):
    """Verify a model preset once; any failure keeps the original fused rules."""
    device = torch.device(device)
    if device.index is None:
        device = torch.device("cuda", torch.cuda.current_device())
    key = (
        device,
        page,
        lpt,
        reblock_ok,
        repr(decode[1]) if decode else None,
        scratch_tokens,
        *(
            os.environ.get(n)
            for n in (
                ENV_REBLOCK,
                ENV_WIDE,
                ENV_FIXED_ANCHOR,
            )
        ),
    )
    if key in _decisions:
        return _decisions[key]
    started = time.monotonic()
    try:
        decision = _resolve(device, page, lpt, reblock_ok, decode, scratch_tokens)
    except Exception:
        logger.warning(
            "Mixed preset failed; keeping fused/rule-based execution", exc_info=True
        )
        decision = MixedDecision(reblock_allowed=reblock_ok)
    _decisions[key] = decision
    logger.info(
        "Mixed attention: %s; %d preset buckets; search skipped; %.2fs",
        decision.reason,
        len(decision.table),
        time.monotonic() - started,
    )
    return decision


def _resolve(device, page, lpt, reblock_ok, decode, scratch_tokens):
    decision = MixedDecision(reblock_allowed=reblock_ok)
    if page != 16:
        return decision
    profile = device_profiles.lookup(device.index)
    wide_default = bool((profile or {}).get("switches", {}).get(ENV_WIDE, False))
    fixed_anchor_default = bool(
        (profile or {}).get("switches", {}).get(ENV_FIXED_ANCHOR, False)
    )
    wide_requested = fl_envs.switch(ENV_WIDE, auto=wide_default)
    fixed_anchor_requested = fl_envs.switch(
        ENV_FIXED_ANCHOR,
        auto=fixed_anchor_default,
        reason="validated fixed-anchor startup gate",
    )
    data = (profile or {}).get("mixed_attention", {}).get("int8", {})
    if data.get("lpt") != lpt or data.get("scratch_tokens") != scratch_tokens:
        data = {}
    decision.prefill_wide = (
        wide_requested and reblock_ok and scratch_tokens > 0 and decode is not None
    )
    if wide_requested and not decision.prefill_wide:
        logger.info(
            "Prefill wide rejected: reblock, scratch, or decode prerequisites are unavailable"
        )
    explicit_reblock = fl_envs.tristate(
        ENV_REBLOCK
    ) is not None and not device_profiles.was_defaulted(ENV_REBLOCK)
    rule_only = not reblock_ok or explicit_reblock
    for key, value in data.get("table", {}).items():
        # q0 is entirely below the reblock threshold; every other bucket is >=128.
        tile = value["tile"]
        if rule_only:
            tile = "reblock" if reblock_ok and not key.endswith(":q0") else "served"
        mode = value["mode"]
        if mode == "separate" and decode is None:
            mode = "fused"
        decision.table[key] = Choice(
            mode if key.startswith("m:") else "fused",
            tile,
            value.get("fused_decode_max", 0),
        )
    pairs = {(c.tile, c.mode) for c in decision.table.values()}
    pairs.update(
        (c.tile, "fused") for c in decision.table.values() if c.fused_decode_max
    )
    verifier = None
    if pairs:
        verifier = Verifier(device, page, lpt, decode_factory(decode), scratch_tokens)
        for tile, mode in sorted(pairs):
            ok, why = verifier.check(tile, mode)
            if not ok:
                logger.warning(
                    "Mixed preset rejected (%s/%s): %s; keeping fused/rule-based execution",
                    tile,
                    mode,
                    why,
                )
                return MixedDecision(
                    reblock_allowed=reblock_ok, reason="numerical check rejected"
                )
    if decision.prefill_wide:
        if verifier is None:
            verifier = Verifier(
                device, page, lpt, decode_factory(decode), scratch_tokens
            )
        ok, why = verifier.check("reblock", "separate", prefill_wide=True)
        if not ok:
            decision.prefill_wide = False
            logger.warning(
                "Prefill wide rejected: %s; keeping the original reblock tile", why
            )
        else:
            logger.info("Prefill wide profile passed numerical validation")
        if decision.prefill_wide and fixed_anchor_requested:
            try:
                from vllm_fl.attention.prefill_wide import _fixed_anchor_selfcheck

                fixed_anchor_details = _fixed_anchor_selfcheck(
                    device,
                    decode[0],
                    decode[1],
                    scratch_tokens,
                    lpt,
                )
                fixed_anchor_passed = True
            except Exception as error:
                fixed_anchor_passed = False
                fixed_anchor_details = {"error": f"{type(error).__name__}: {error}"}
            decision.prefill_fixed_anchor = fixed_anchor_passed
            if fixed_anchor_passed:
                logger.info(
                    "Fixed-anchor profile passed numerical validation: %s",
                    fixed_anchor_details,
                )
            else:
                logger.info(
                    "Fixed-anchor rejected; retaining wide prefill: %s",
                    fixed_anchor_details,
                )
        elif decision.prefill_wide:
            logger.info("Fixed-anchor disabled by switch; retaining wide prefill")
    decision.reason = "verified model preset" if decision.table else "rule-based"
    return decision
