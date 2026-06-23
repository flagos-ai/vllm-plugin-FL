# Copyright (c) 2026 BAAI. All rights reserved.

"""
Patch vLLM's Sampler to use FlagGems kernels for softmax / div / to_copy
(and the exponential-sampling argmax inside forward_native).

Gating: applied only when ``use_flaggems_op("sampler")`` is True (i.e.
VLLM_FL_FLAGOS_WHITELIST contains "sampler"). No import-time side effect —
callers must invoke ``patch_sampler()`` explicitly. It is idempotent and
version-guarded against drift in vLLM's private sampler surface.

ORDERING REQUIREMENT (important):
    TopKTopPSampler.__init__ binds ``self.forward = self.forward_native`` as an
    *instance* attribute at construction time (early binding). A class-level
    replacement of ``forward_native`` is therefore only observed by instances
    created AFTER the patch. ``patch_sampler()`` MUST run before the
    TopKTopPSampler is instantiated (i.e. before model load). The current
    worker.py call site satisfies this.

Patches, on the vLLM v1 sampling call chain
    Sampler.forward -> Sampler.apply_temperature
                    -> TopKTopPSampler.forward_native:
  1. TopKTopPSampler.forward_native  (FlagGems softmax + exponential sampling)
  2. Sampler.apply_temperature       (FlagGems div)
  3. Sampler.forward                 (FlagGems to_copy for the fp32 upcast)

Note: the module-level ``random_sample`` is intentionally NOT patched. Its only
call site is inside forward_native (which we replace wholesale), so patching it
would be dead code. forward_cuda's generator fallback calls self.forward_native
(late-bound), so it also routes through patch #1.
"""

import functools
import inspect
import logging

import torch
import flag_gems.ops as gems_ops

logger = logging.getLogger(__name__)


# --------------------------------------------------------------------------- #
# Replacement implementations (same logic as upstream; ops swapped to FlagGems)
# --------------------------------------------------------------------------- #
def _gems_random_sample(probs, generators):
    """Exponential / Gumbel-max sampling: argmax_i (p_i / q_i), q_i ~ Exp(1)."""
    q = torch.empty_like(probs)
    gems_ops.exponential_(q, lambd=1.0)

    if generators:
        # Per-request seeded noise for reproducibility. torch's .exponential_
        # is used here because it accepts a per-request generator.
        for i, generator in generators.items():
            q[i].exponential_(generator=generator)

    gems_ops.true_divide_(probs, q)
    return gems_ops.argmax(probs, dim=-1).view(-1)


def _gems_forward_native(self, logits, generators, k, p):
    """TopKTopPSampler.forward_native with FlagGems softmax + sampling."""
    from vllm.v1.sample.ops.topk_topp_sampler import apply_top_k_top_p

    logits = apply_top_k_top_p(logits, k, p)

    logits_to_return = None
    if self.logprobs_mode == "processed_logits":
        logits_to_return = logits
    elif self.logprobs_mode == "processed_logprobs":
        logits_to_return = logits.log_softmax(dim=-1, dtype=torch.float32)

    probs = gems_ops.softmax(logits, dim=-1).to(torch.float32)
    return _gems_random_sample(probs, generators), logits_to_return


def _gems_apply_temperature(logits, temp, all_random):
    """Sampler.apply_temperature with FlagGems div: logits / temp."""
    _SAMPLING_EPS = 1e-5
    if not all_random:
        # Avoid divide-by-~0 on greedy rows.
        temp = torch.where(temp < _SAMPLING_EPS, 1.0, temp)
    return gems_ops.true_divide_(logits, temp.unsqueeze(dim=1))


# --------------------------------------------------------------------------- #
# Hardened application
# --------------------------------------------------------------------------- #
def _require_params(fn, names, what):
    """Fail fast if the upstream function's signature has drifted."""
    try:
        params = set(inspect.signature(fn).parameters)
    except (TypeError, ValueError) as e:
        raise RuntimeError(
            f"[gems_sampler] cannot introspect {what}: {e}. "
            "vLLM version is likely incompatible; refusing to patch."
        )
    missing = [n for n in names if n not in params]
    if missing:
        raise RuntimeError(
            f"[gems_sampler] {what} signature drift: missing {missing}. "
            "vLLM version is likely incompatible; refusing to patch."
        )


def patch_sampler():
    """Apply all Sampler patches. Idempotent and version-guarded.

    Must be called before TopKTopPSampler is instantiated (see module docstring).
    """
    from vllm.v1.sample.sampler import Sampler
    from vllm.v1.sample.ops.topk_topp_sampler import TopKTopPSampler

    if getattr(Sampler, "_gems_patched", False):
        logger.info("[gems_sampler] already patched, skipping")
        return

    # Version guard: verify the private surface we are about to override.
    _require_params(Sampler.forward, ["logits", "sampling_metadata"], "Sampler.forward")
    _require_params(Sampler.apply_temperature, ["temp"], "Sampler.apply_temperature")
    _require_params(
        TopKTopPSampler.forward_native, ["logits", "generators", "k", "p"],
        "TopKTopPSampler.forward_native",
    )

    # 1. TopKTopPSampler.forward_native  (FlagGems softmax + exponential sampling)
    TopKTopPSampler.forward_native = _gems_forward_native
    logger.info("[gems_sampler] patched TopKTopPSampler.forward_native -> FlagGems")

    # 2. Sampler.apply_temperature
    Sampler.apply_temperature = staticmethod(_gems_apply_temperature)
    logger.info("[gems_sampler] patched Sampler.apply_temperature -> FlagGems div")

    # 3. Wrap Sampler.forward to route the float32 upcast through FlagGems.
    #    Signature forwarded with *args/**kwargs so an upstream signature change
    #    does not break the wrapper.
    _orig_forward = Sampler.forward

    @functools.wraps(_orig_forward)
    def _patched_forward(self, logits, *args, **kwargs):
        if logits.dtype != torch.float32:
            logits = gems_ops.to_copy(logits, dtype=torch.float32)
        return _orig_forward(self, logits, *args, **kwargs)

    Sampler.forward = _patched_forward
    logger.info("[gems_sampler] patched Sampler.forward -> FlagGems to_copy")

    # Idempotency marker + handle kept for rollback in tests.
    Sampler._gems_patched = True
    Sampler._gems_orig_forward = _orig_forward
    logger.info("[gems_sampler] all patches applied")


def maybe_patch_sampler():
    """Apply the sampler patch iff ``use_flaggems_op("sampler")`` is enabled.

    This is the single entry point the worker should call. The gate lives here
    (co-located with the patch) so the call site stays trivial. Call once, early
    — before any TopKTopPSampler is instantiated (see ordering note above).
    """
    from vllm_fl.utils import use_flaggems_op

    if not use_flaggems_op("sampler"):
        return
    try:
        patch_sampler()
    except Exception as e:  # never let a patch failure take down the worker
        logger.warning(f"[gems_sampler] failed to apply sampler patch: {e}")


def unpatch_sampler():
    """Best-effort rollback (intended for tests)."""
    from vllm.v1.sample.sampler import Sampler

    if getattr(Sampler, "_gems_patched", False):
        Sampler.forward = Sampler._gems_orig_forward
        del Sampler._gems_orig_forward
        Sampler._gems_patched = False
        logger.info("[gems_sampler] Sampler.forward restored")
