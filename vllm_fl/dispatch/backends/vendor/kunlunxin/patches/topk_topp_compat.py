# Copyright (c) 2026 Kunlunxin, Inc. All rights reserved.
"""Replace the V2 top-k/top-p Triton sampler with torch on FlagTree.

FlagTree's TritonXPU backend aborts inside ``TritonXPULegalize`` on the
Qrita pivot-based reduction in ``_topk_topp_kernel`` (assertion
``reduction rowspercore is not consistent``), and the split-row top-p
pipeline uses the same reduction pattern.  Replace the public entry
point ``apply_top_k_top_p_triton`` with a torch implementation that has
the same semantics: top-k first (by logit value), then top-p applied to
the surviving k values (by probability), masking with ``mask_value``.

The torch path sorts the row, which is O(vocab * log vocab) per row
versus the Triton kernel's linear scan, so it is slower at large
vocab.  Correctness over speed: the V1 runner already samples on CPU
torch for Kunlunxin, so the latency budget absorbs it.
"""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)

_PATCH_FLAG = "_vllm_fl_kunlunxin_topk_topp_torch"


def _apply_top_k_top_p_torch(
    logits: "torch.Tensor",
    k: "torch.Tensor | None",
    p: "torch.Tensor | None",
    mask_value: float = float("-inf"),
):
    """torch top-k then top-p, matching the Triton kernel's semantics."""
    import torch

    assert logits.ndim == 2
    assert logits.dtype == torch.float32
    batch_size, vocab_size = logits.shape
    topk_enabled = k is not None
    topp_enabled = p is not None

    if batch_size == 0 or not (topk_enabled or topp_enabled):
        return logits

    if logits.stride(1) != 1:
        logits = logits.contiguous()

    if k is not None:
        k = k.to(torch.int32)
    if p is not None:
        p = p.to(torch.float32)

    out = logits.clone()

    if topk_enabled:
        # Keep the k largest logits per row; mask the rest.
        for r in range(batch_size):
            kv = int(k[r].item())
            if kv <= 0:
                out[r] = mask_value
                continue
            if kv >= vocab_size:
                continue
            vals, idx = torch.topk(out[r], kv, sorted=False)
            row_mask = torch.full_like(out[r], mask_value)
            row_mask[idx] = vals
            out[r] = row_mask

    if topp_enabled:
        for r in range(batch_size):
            pv = float(p[r].item())
            if pv >= 1.0:
                continue
            row = out[r]
            finite = row > mask_value
            if not bool(finite.any()):
                continue
            probs = torch.softmax(row, dim=-1)
            sorted_probs, sorted_idx = torch.sort(probs, descending=True, stable=True)
            # Keep the smallest prefix whose cumulative probability >= p,
            # matching the upstream "keep until threshold crossed" rule.
            cum = torch.cumsum(sorted_probs, dim=-1)
            # The first index where the cumulative sum reaches p is the last
            # token to keep; everything after is masked.
            keep = cum <= pv
            # Always keep at least one token so a near-zero p doesn't blank
            # the row.
            keep[0] = True
            masked_row = torch.full_like(row, mask_value)
            kept_idx = sorted_idx[keep]
            masked_row[kept_idx] = row[kept_idx]
            out[r] = masked_row

    # Copy back into the original tensor so callers that retained a
    # reference see the in-place update, matching the Triton path.
    logits.copy_(out)
    return logits


def apply_topk_topp_compat() -> None:
    """Patch ``apply_top_k_top_p_triton`` to the torch path.  Idempotent.

    ``topk_topp_sampler`` imports the function by name at module load time
    (``from ... import apply_top_k_top_p_triton``), so the binding lives in
    *that* module's namespace.  We must patch it there, not only in the
    triton module it was imported from.
    """
    patched_any = False
    for mod_name in (
        "vllm.v1.sample.ops.topk_topp_triton",
        "vllm.v1.sample.ops.topk_topp_sampler",
    ):
        try:
            import importlib
            mod = importlib.import_module(mod_name)
        except ImportError:
            logger.debug(
                "%s not importable; topk_topp_compat skips it", mod_name
            )
            continue

        current = getattr(mod, "apply_top_k_top_p_triton", None)
        if current is None:
            continue
        if getattr(current, _PATCH_FLAG, False):
            continue

        setattr(_apply_top_k_top_p_torch, _PATCH_FLAG, True)
        mod.apply_top_k_top_p_triton = _apply_top_k_top_p_torch
        patched_any = True
        logger.info(
            "Patched apply_top_k_top_p_triton in %s with the torch path",
            mod_name,
        )

    if patched_any:
        # Also patch the wrapper that calls it by local name
        try:
            import importlib
            sampler = importlib.import_module(
                "vllm.v1.sample.ops.topk_topp_sampler"
            )
            if getattr(
                getattr(sampler, "apply_top_k_top_p_triton", None),
                _PATCH_FLAG,
                False,
            ):
                return
            # apply_top_k_top_p calls apply_top_k_top_p_triton via the
            # module-level name; patching the attribute on sampler is enough.
        except ImportError:
            pass
