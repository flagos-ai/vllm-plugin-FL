# Copyright (c) 2026 BAAI. All rights reserved.
"""Opt-in fp8 W8A8 path for the Qwen3.5 language-model head.

``VLLM_FL_LMHEAD_FP8`` (default off) serves the head projection from a cached
e4m3 copy on the fp8 block kernel instead of the 512-row chunked bf16 einsum.
The seam is the ``lm_head`` argument of ``LogitsProcessor``: only the head's
``quant_method`` is replaced, so vocab padding, scaling and soft-capping stay
upstream's. The patch lives with the evas backend rather than in
``vllm_fl/patches/`` because its whole point is an LPU device kernel.

"""

from __future__ import annotations

import json
import logging
import os
from typing import Any

import torch

logger = logging.getLogger(__name__)

# Weight build and matmul live with the numeric side of the backend; the knob
# and the hit/miss policy stay here so the backend module never reads the
# environment itself.
_IMPL_MODULE = "vllm_fl.dispatch.backends.vendor.evas.impl.lmhead_fp8"


def _env_flag(name: str, default: str = "0") -> bool:
    return os.environ.get(name, default).strip().lower() not in {
        "", "0", "false", "no", "off",
    }


_ENABLED = _env_flag("VLLM_FL_LMHEAD_FP8")
_DUMP_PATH = os.environ.get("DUMP_LOGITS") or None
# A prefill dump of a long prompt would be hundreds of MiB, so the dump keeps
# only the short calls (decode rows) unless the user raises the bound.
_DUMP_MAX_ROWS = int(os.environ.get("DUMP_LOGITS_MAX_ROWS", "64") or 0)
_dump_calls = 0
_impl: Any = None


class _Fp8QuantMethod:
    """Serve the head projection from a cached fp8 copy of the weight.

    ``build`` is lazy because weights load after construction; the cache is
    keyed on storage identity plus the tensor version so an in-place
    ``load_weights`` copy rebuilds instead of serving stale bytes. Once the
    master is dropped (``build_now`` + storage replacement under the enabled
    knob) the cache is frozen: it never consults the parameter again, because
    the parameter is an empty placeholder and any key check would trigger a
    rebuild from garbage.
    """

    def __init__(self, inner: Any, impl: Any) -> None:
        self._inner = inner
        self._impl = impl
        self._built: tuple[tuple[int, int, tuple[int, ...]], Any, Any] | None = None
        self.frozen = False

    def build_now(self, weight: torch.Tensor) -> None:
        """Build eagerly from the live master and freeze the cache."""
        key = (weight.data_ptr(), getattr(weight, "_version", 0), tuple(weight.shape))
        w_fp8, scales = self._impl.build(weight)
        self._built = (key, w_fp8, scales)
        self.frozen = True

    def _cached_weight(self, layer: Any) -> tuple[Any, Any]:
        if self.frozen:
            assert self._built is not None
            return self._built[1], self._built[2]
        weight = layer.weight
        key = (weight.data_ptr(), getattr(weight, "_version", 0), tuple(weight.shape))
        if self._built is None or self._built[0] != key:
            w_fp8, scales = self._impl.build(weight)
            self._built = (key, w_fp8, scales)
        return self._built[1], self._built[2]

    def apply(self, layer: Any, x: torch.Tensor, bias: torch.Tensor | None = None) -> torch.Tensor:
        # The fp8 kernel has no bias term; anything unusual stays on the
        # original method rather than being silently dropped. After the master
        # is dropped the original method would read an empty parameter, so a
        # bias call must fail loud instead of producing garbage.
        if bias is not None:
            if self.frozen:
                raise RuntimeError(
                    "lmhead fp8: the bf16 master is dropped and the fp8 "
                    "kernel has no bias path; a bias projection cannot be "
                    "served. This indicates an unexpected caller of the "
                    "wrapped head."
                )
            return self._inner.apply(layer, x, bias=bias)
        w_fp8, scales = self._cached_weight(layer)
        x2d = x if x.dim() == 2 else x.reshape(-1, x.shape[-1])
        out = self._impl.matmul(x2d, w_fp8, scales)
        if x.dim() == 2:
            return out
        return out.reshape(*x.shape[:-1], out.shape[-1])


def requested() -> bool:
    """True when the knob or a dump path asked for this patch.

    Read from the module constants so the registry gate and the install can
    never disagree. A default run has both off: the spec is then skipped,
    which is more honest than an "Applied" line for a rebind that never ran.
    """
    return _ENABLED or _DUMP_PATH is not None


def _load_impl() -> Any:
    global _impl
    if _impl is not None:
        return _impl
    import importlib

    try:
        _impl = importlib.import_module(_IMPL_MODULE)
    except ImportError as exc:  # pragma: no cover - depends on backend landing
        raise RuntimeError(
            "VLLM_FL_LMHEAD_FP8 is set but the fp8 head backend is missing at "
            f"{_IMPL_MODULE}; unset the knob or land the backend module. "
            "Refusing to fall back silently, which would make the A/B measure "
            "the bf16 path while reporting fp8."
        ) from exc
    return _impl


def _maybe_dump(logits: torch.Tensor | None) -> None:
    """Write raw logits for the numeric acceptance run.

    Sampler-reported logprobs are quantized to ``2**-4`` on this backend, so
    the acceptance criteria read these dumps instead. Files are numbered in call
    order; the caller keeps prompt and step ids alongside.
    """
    global _dump_calls
    if _DUMP_PATH is None or logits is None:
        return
    index = _dump_calls
    _dump_calls += 1
    if _DUMP_MAX_ROWS and logits.shape[0] > _DUMP_MAX_ROWS:
        logger.info("[lmhead-fp8] dump skipped: %s rows at call %d", logits.shape[0], index)
        return
    path = f"{_DUMP_PATH}.{index:05d}.pt"
    torch.save({"logits": logits.detach().to("cpu"), "call": index}, path)
    if index == 0:
        with open(f"{_DUMP_PATH}.env.json", "w", encoding="utf-8") as handle:
            json.dump({k: v for k, v in os.environ.items()
                       if k.startswith(("VLLM_FL_", "DUMP_LOGITS"))}, handle, indent=2)


def _compute_logits_experiment(self: Any, hidden_states: torch.Tensor) -> torch.Tensor | None:
    head = self.lm_head
    if _ENABLED and not isinstance(head.quant_method, _Fp8QuantMethod):
        head.quant_method = _Fp8QuantMethod(head.quant_method, _load_impl())
    logits = self.logits_processor(head, hidden_states)
    _maybe_dump(logits)
    return logits


def _build_and_drop_master(model: Any) -> None:
    """Build the fp8 copy from the live master, then free the master.

    Only a real ``ParallelLMHead`` is touched: a tied head shares its storage
    with the embedding (dropping it would corrupt input embeddings) and a
    pipeline-parallel non-last rank has no head at all. Runs inside
    ``load_weights`` so the recovery lands before KV-pool profiling.
    """
    from vllm.model_executor.layers.vocab_parallel_embedding import ParallelLMHead

    head = model.lm_head
    if not isinstance(head, ParallelLMHead):
        logger.info(
            "[lmhead-fp8] head is tied or absent; bf16 master kept"
        )
        return
    if not isinstance(head.quant_method, _Fp8QuantMethod):
        head.quant_method = _Fp8QuantMethod(head.quant_method, _load_impl())
    method = head.quant_method
    if method.frozen:
        return
    weight = head.weight
    method.build_now(weight)
    master_bytes = weight.numel() * weight.element_size()
    weight.data = torch.empty((0,), dtype=weight.dtype, device=weight.device)
    logger.info(
        "[lmhead-fp8] bf16 master dropped: %.2f GiB recovered, "
        "projection served from the fp8 cache",
        master_bytes / 2**30,
    )


def _install_load_weights_hook(base: Any) -> bool:
    """Wrap ``load_weights`` so the master is dropped after weights land.

    Enabled-knob only: the dump-only reference arm must keep serving the bf16
    path, which reads the master. A reload after the drop raises because the
    incoming tensors have no storage to land in; skipping them silently would
    report success while serving stale bytes.
    """
    if getattr(base, "_lmhead_fp8_load_installed", False):
        return True
    orig_load_weights = base.load_weights

    def _load_weights_then_fp8_head(self: Any, weights: Any) -> Any:
        head = getattr(self, "lm_head", None)
        method = getattr(head, "quant_method", None)
        if isinstance(method, _Fp8QuantMethod) and method.frozen:
            raise RuntimeError(
                "lmhead fp8: the bf16 master was already dropped, so this "
                "load_weights call has nowhere to write. Weight reload is "
                "unsupported with VLLM_FL_LMHEAD_FP8; run with the knob off "
                "to reload weights."
            )
        result = orig_load_weights(self, weights)
        _build_and_drop_master(self)
        return result

    base.load_weights = _load_weights_then_fp8_head
    base._lmhead_fp8_load_installed = True
    return True


def _install_out_of_scope_guard(cls: Any) -> bool:
    """Refuse the multimodal Qwen3.5 class loudly when the knob is on.

    ``Qwen3_5ForConditionalGeneration`` does not inherit the text base: its
    ``compute_logits`` is not rebound and its own ``load_weights`` is not
    wrapped by ``_install_load_weights_hook``, so with the knob enabled it
    would silently keep serving bf16 while the log said fp8.  Dropping the
    master there would also read an empty parameter, so the enabled knob
    fails the load instead.
    """
    if getattr(cls, "_lmhead_fp8_out_of_scope", False):
        return True

    def _load_weights_refused(self: Any, weights: Any) -> Any:
        raise RuntimeError(
            "lmhead fp8: the multimodal Qwen3.5 class is out of scope for "
            "VLLM_FL_LMHEAD_FP8, its head projection is not rebound; run "
            "with the knob off."
        )

    cls.load_weights = _load_weights_refused
    cls._lmhead_fp8_out_of_scope = True
    return True


def apply_lpu_lmhead_fp8_patch() -> bool | None:
    """Rebind ``compute_logits`` for the Qwen3.5 text models.

    Active when the fp8 knob is on or a dump path is set, since the acceptance
    run needs the bf16 arm dumped too. Repeated calls are safe.

    Returns ``None`` when the feature is opted out (knob off, no dump path):
    that is a requested state, not a missing target, so the registry must not
    report it as a declined install. ``True`` means the rebind is in place.
    """
    if not _ENABLED and _DUMP_PATH is None:
        return None
    from vllm.model_executor.models.qwen3_5 import Qwen3_5ForCausalLMBase

    if _ENABLED:
        _load_impl()  # fail loud at engine start, not mid-generation
        _install_load_weights_hook(Qwen3_5ForCausalLMBase)
        try:
            from vllm.model_executor.models.qwen3_5 import (
                Qwen3_5ForConditionalGeneration,
            )
        except ImportError:  # pragma: no cover - slim installs
            pass
        else:
            _install_out_of_scope_guard(Qwen3_5ForConditionalGeneration)
    if getattr(Qwen3_5ForCausalLMBase, "_lmhead_fp8_installed", False):
        return True
    Qwen3_5ForCausalLMBase.compute_logits = _compute_logits_experiment
    Qwen3_5ForCausalLMBase._lmhead_fp8_installed = True
    if _ENABLED:
        logger.info("[lmhead-fp8] Qwen3.5 head projection -> fp8 W8A8")
    if _DUMP_PATH is not None:
        logger.info("[lmhead-fp8] logits dump -> %s (measurement run, not timing)",
                    _DUMP_PATH)
    return True


__all__ = ["apply_lpu_lmhead_fp8_patch"]
