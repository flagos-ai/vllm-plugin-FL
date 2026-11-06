# Copyright (c) 2026 BAAI. All rights reserved.

"""Back vLLM's native MoE helpers with FL dispatch on PTPU.

vLLM's functional ``fused_experts`` reaches for two C extensions that PTPU
wheels do not ship:

* ``vllm._custom_ops`` forwards ``moe_align_block_size``, ``moe_sum`` and
  ``topk_softmax`` straight to ``torch.ops._moe_C``, which only exists in CUDA
  and ROCm builds. Callers get a bare
  ``AttributeError: '_OpNamespace' '_moe_C' object has no attribute ...``.
* ``apply_moe_activation`` calls ``torch.ops._C.silu_and_mul``. The plugin
  registers a schema for that op so torch.compile can still pattern-match it,
  but never an implementation, so invoking it fails at runtime.

The plugin's own pipeline (``TritonExpertsFL``) never hits either gap: it
resolves every operator through ``CachedOp`` and uses
``vllm_fl.ops.fused_moe.activation``. Closing the gaps here is what lets vLLM's
functional experts -- and therefore upstream's ``TritonW8A8Experts`` -- run on
PTPU at all.

The ``_moe_C`` shims are skipped when that extension is present, so they never
shadow a real vendor kernel. The activation redirect is unconditional: this
module is sunrise-only, and the FL implementation is the one PTPU has been
validated against.
"""

from __future__ import annotations

import logging
import sys

import torch

logger = logging.getLogger(__name__)

_PATCHED = False
_MARKER = "_fl_sunrise_moe_c_shim"
_CACHED_OPS: dict[str, object] = {}


def _op(name: str):
    """Return the dispatch entry point for ``name`` (constructed once)."""
    op = _CACHED_OPS.get(name)
    if op is None:
        from vllm_fl.dispatch import CachedOp

        op = CachedOp(name)
        _CACHED_OPS[name] = op
    return op


def _moe_c_available() -> bool:
    """Report whether the native ``_moe_C`` MoE extension is importable."""
    try:
        return getattr(torch.ops._moe_C, "moe_align_block_size", None) is not None
    except (AttributeError, RuntimeError):
        return False


def _moe_align_block_size(
    topk_ids: torch.Tensor,
    num_experts: int,
    block_size: int,
    sorted_token_ids: torch.Tensor,
    experts_ids: torch.Tensor,
    num_tokens_post_pad: torch.Tensor,
    expert_map: torch.Tensor | None = None,
) -> None:
    """Out-parameter shim over the dispatch registry's return-value op.

    vLLM only forwards ``expert_map`` to the kernel when it wants invalid
    experts skipped in place; otherwise it remaps ``expert_ids`` itself after
    the call. Mirror that by tying ``ignore_invalid_experts`` to whether a map
    was passed down.
    """
    sorted_ids, expert_ids, post_pad = _op("moe_align_block_size")(
        topk_ids,
        block_size,
        num_experts,
        expert_map,
        ignore_invalid_experts=expert_map is not None,
    )

    for produced, destination, name in (
        (sorted_ids, sorted_token_ids, "sorted_token_ids"),
        (expert_ids, experts_ids, "expert_ids"),
        (post_pad, num_tokens_post_pad, "num_tokens_post_pad"),
    ):
        if produced.shape != destination.shape:
            raise RuntimeError(
                "moe_align_block_size shim: the dispatch backend returned "
                f"{name} with shape {tuple(produced.shape)}, but vLLM "
                f"allocated {tuple(destination.shape)}"
            )
        destination.copy_(produced)


def _moe_sum(input: torch.Tensor, output: torch.Tensor) -> None:
    _op("moe_sum")(input, output)


def _topk_softmax(
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    token_expert_indices: torch.Tensor,
    gating_output: torch.Tensor,
    renormalize: bool = False,
    e_score_correction_bias: torch.Tensor | None = None,
) -> None:
    if e_score_correction_bias is not None:
        raise NotImplementedError(
            "topk_softmax shim: e_score_correction_bias has no dispatch "
            "backend on PTPU; route this router through FusedTopKRouterFL"
        )

    weights, ids = _op("topk_softmax")(
        topk_weights,
        topk_ids,
        token_expert_indices,
        gating_output,
        renormalize,
    )

    # Backends are free to return fresh tensors instead of filling the caller's
    # buffers; _custom_ops.topk_softmax is out-parameter only.
    if weights is not topk_weights:
        topk_weights.copy_(weights)
    if ids is not topk_ids:
        topk_ids.copy_(ids.to(topk_ids.dtype))


_SHIMS = {
    "moe_align_block_size": _moe_align_block_size,
    "moe_sum": _moe_sum,
    "topk_softmax": _topk_softmax,
}


def _rebind_importers(name: str, original, replacement) -> None:
    """Update modules that imported ``name`` by value before we patched it."""
    for module in list(sys.modules.values()):
        if module is None:
            continue
        if getattr(module, name, None) is original:
            setattr(module, name, replacement)


def _patch_moe_c_ops() -> bool:
    """Point ``vllm._custom_ops`` MoE helpers at the FL dispatch registry."""
    if _moe_c_available():
        logger.debug("moe-c-shim: torch.ops._moe_C is available; not patching.")
        return True

    try:
        import vllm._custom_ops as _vllm_ops
    except Exception as exc:  # noqa: BLE001
        logger.debug("moe-c-shim: vllm._custom_ops unavailable (%s)", exc)
        return False

    for name, shim in _SHIMS.items():
        original = getattr(_vllm_ops, name, None)
        if original is None or getattr(original, _MARKER, False):
            continue

        setattr(shim, _MARKER, True)
        shim._fl_original = original  # type: ignore[attr-defined]
        setattr(_vllm_ops, name, shim)
        _rebind_importers(name, original, shim)

    logger.info(
        "moe-c-shim: torch.ops._moe_C is missing; vllm._custom_ops %s now "
        "resolve through the FL dispatch registry.",
        sorted(_SHIMS),
    )
    return True


def _patch_moe_activation() -> bool:
    """Give vLLM's fused-MoE activation a PTPU-capable implementation.

    ``vllm_fl.ops.fused_moe.activation.apply_moe_activation`` is the same
    function with the gated activations resolved through ``CachedOp`` instead of
    ``torch.ops._C``; ``TritonExpertsFL`` has always used it.
    """
    try:
        import vllm.model_executor.layers.fused_moe.activation as _vllm_act

        from vllm_fl.ops.fused_moe.activation import (
            apply_moe_activation as _fl_apply_moe_activation,
        )
    except Exception as exc:  # noqa: BLE001
        logger.debug("moe-c-shim: MoE activation module unavailable (%s)", exc)
        return False

    original = getattr(_vllm_act, "apply_moe_activation", None)
    if original is None or original is _fl_apply_moe_activation:
        return True

    _vllm_act.apply_moe_activation = _fl_apply_moe_activation
    # fused_moe.py imports the symbol by value at module scope.
    _rebind_importers("apply_moe_activation", original, _fl_apply_moe_activation)

    logger.info(
        "moe-c-shim: apply_moe_activation now resolves gated activations "
        "through the FL dispatch registry instead of torch.ops._C."
    )
    return True


def apply_patch() -> bool:
    """Close the native-op gaps in vLLM's fused-MoE path on PTPU."""
    global _PATCHED
    if _PATCHED:
        return True
    _PATCHED = _patch_moe_c_ops() and _patch_moe_activation()
    return _PATCHED
