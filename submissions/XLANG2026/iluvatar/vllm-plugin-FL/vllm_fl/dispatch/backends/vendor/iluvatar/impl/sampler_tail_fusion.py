# SPDX-License-Identifier: Apache-2.0
"""MetaX: fused sampler tail.

vLLM's eager sampler tail per decode step:
    logits.to(float32)          # cast pass over [batch, vocab]
    logits.div_(temperature)    # fp32 read+write pass
    argmax / gumbel-argmax      # fp32 read pass

This patch routes the common "plain sampling" case (no logprobs, no
penalties, no top-k/top-p, no logits processors, no per-request generators)
through flag_gems.fused.sampler_fused.fused_sample, which reads the bf16
lm_head output directly and folds temperature into the Gumbel-Max score.
Any request feature outside that set falls back to the stock Sampler.forward
unchanged.

Import-order note: this plugin module is first imported while
vllm.v1.sample.sampler may still be mid-import (circular import via platform
plugin discovery), so ``Sampler`` itself cannot be imported here.  The ops
submodule (topk_topp_sampler) is always fully initialized at this point —
Sampler.__init__ unconditionally constructs a TopKTopPSampler, so hooking its
__init__ applies the forward patch exactly once, at engine startup, before
any request is sampled.
"""

import os
import sys

import torch
import vllm.v1.sample.ops.topk_topp_sampler as _tts_mod
from vllm.logger import init_logger

logger = init_logger(__name__)

try:
    from flag_gems.fused.sampler_fused import fused_sample as _fused_sample
except Exception:  # FlagGems without the fused kernel: stay on stock path
    _fused_sample = None

try:
    from flag_gems.fused.top_p_sample import (
        fused_top_p_sample as _fused_top_p_sample,
    )
except Exception:
    _fused_top_p_sample = None

_patched = False
_orig_forward = None
_SamplerOutput = None
_engaged = {"n": 0, "rej": 0}

# vLLM always loads these builtin logits processors; each is a host-side
# no-op when no request activates it (empty state dict / zero count).
# Custom or unknown processors always force the stock path.
_BUILTIN_INACTIVE = {
    "MinTokensLogitsProcessor": lambda p: not p.min_toks,
    "LogitBiasLogitsProcessor": lambda p: not p.biases,
    "MinPLogitsProcessor": lambda p: not p.min_p_count,
}


def _procs_inactive(procs) -> bool:
    for p in procs:
        pred = _BUILTIN_INACTIVE.get(type(p).__name__)
        if pred is None or not pred(p):
            return False
    return True
# VLLM_FL_DISABLE_SAMPLER_FUSION=1: keep the stock Sampler.forward (A/B switch)
_DISABLED = os.getenv("VLLM_FL_DISABLE_SAMPLER_FUSION", "0") == "1"


def _fast_path_ok(
    self,
    logits: torch.Tensor,
    sm,
    predict_bonus_token: bool,
    logprobs_mode_override,
) -> bool:
    if _fused_sample is None or self.use_fp64_gumbel:
        return False
    if logits.dim() != 2 or logits.dtype not in (
        torch.bfloat16,
        torch.float16,
        torch.float32,
    ):
        return False
    if predict_bonus_token or logprobs_mode_override is not None:
        return False
    # Any form of logprobs request needs the processed logits tensor.
    if sm.max_num_logprobs is not None or sm.logprob_token_ids:
        return False
    if not sm.no_penalties:
        return False
    if sm.allowed_token_ids_mask is not None or sm.bad_words_token_ids:
        return False
    if not _procs_inactive(sm.logitsprocs.argmax_invariant):
        return False
    if not _procs_inactive(sm.logitsprocs.non_argmax_invariant):
        return False
    if sm.top_k is not None:
        return False
    # top_p tensors are supported via fused_top_p_sample; without it we must
    # fall back whenever top-p filtering is active.
    if sm.top_p is not None and _fused_top_p_sample is None:
        return False
    if sm.generators:
        return False
    holder = sm.thinking_budget_state_holder
    if holder is not None and holder.has_tracked_requests():
        return False
    if sm.spec_token_ids is not None and any(sm.spec_token_ids):
        return False
    if not sm.all_greedy and sm.temperature is None:
        return False
    return True


def _fast_path_reject_reason(self, logits, sm, predict_bonus_token, lpm):
    """Debug helper: why the fused tail was not taken (logged once)."""
    if _fused_sample is None:
        return "fused_sample kernel not importable"
    if self.use_fp64_gumbel:
        return "use_fp64_gumbel"
    if logits.dim() != 2 or logits.dtype not in (
        torch.bfloat16, torch.float16, torch.float32,
    ):
        return f"logits dim/dtype {logits.dim()} {logits.dtype}"
    if predict_bonus_token or lpm is not None:
        return "predict_bonus_token/logprobs_mode_override"
    if sm.max_num_logprobs is not None or sm.logprob_token_ids:
        return "logprobs requested"
    if not sm.no_penalties:
        return "penalties"
    if sm.allowed_token_ids_mask is not None or sm.bad_words_token_ids:
        return "allowed_mask/bad_words"
    if not _procs_inactive(sm.logitsprocs.argmax_invariant):
        return "argmax_invariant logitsprocs active"
    if not _procs_inactive(sm.logitsprocs.non_argmax_invariant):
        return "non_argmax_invariant logitsprocs active: " + ",".join(
            f"{type(p).__name__}" for p in sm.logitsprocs.non_argmax_invariant
        )
    if sm.top_k is not None:
        return "top_k active (unsupported in fused path)"
    if sm.top_p is not None and _fused_top_p_sample is None:
        return "top_p active but fused_top_p_sample unavailable"
    if sm.generators:
        return "generators"
    holder = sm.thinking_budget_state_holder
    if holder is not None and holder.has_tracked_requests():
        return "thinking_budget"
    if sm.spec_token_ids is not None and any(sm.spec_token_ids):
        return "spec_token_ids"
    if not sm.all_greedy and sm.temperature is None:
        return "temperature None"
    return "unknown (conditions passed?)"


def _forward_metax(
    self,
    logits: torch.Tensor,
    sampling_metadata,
    predict_bonus_token: bool = False,
    logprobs_mode_override=None,
):
    ok = _fast_path_ok(
        self, logits, sampling_metadata, predict_bonus_token, logprobs_mode_override
    )
    if not ok and _engaged["rej"] < 20:
        _engaged["rej"] += 1
        logger.info(
            "fl opt7: fast path rejected: %s",
            _fast_path_reject_reason(
                self, logits, sampling_metadata, predict_bonus_token,
                logprobs_mode_override,
            ),
        )
    if ok:
        if _engaged["n"] == 0:
            logger.info("fl opt7: fused sampler tail ENGAGED on first batch")
        _engaged["n"] += 1
        if sampling_metadata.all_greedy:
            sampled = _fused_sample(logits)
        elif sampling_metadata.top_p is not None:
            sampled = _fused_top_p_sample(
                logits, sampling_metadata.temperature, sampling_metadata.top_p
            )
        else:
            sampled = _fused_sample(logits, sampling_metadata.temperature)
        return _SamplerOutput(
            sampled_token_ids=sampled.unsqueeze(-1),
            logprobs_tensors=None,
        )
    return _orig_forward(
        self, logits, sampling_metadata, predict_bonus_token, logprobs_mode_override
    )


def _apply() -> bool:
    """Patch Sampler.forward once its module is fully imported.  Idempotent."""
    global _patched, _orig_forward, _SamplerOutput
    if _patched or _DISABLED:
        return _patched
    mod = sys.modules.get("vllm.v1.sample.sampler")
    sampler_cls = getattr(mod, "Sampler", None) if mod is not None else None
    if sampler_cls is None:
        return False
    from vllm.v1.outputs import SamplerOutput

    _SamplerOutput = SamplerOutput
    _orig_forward = sampler_cls.forward
    sampler_cls.forward = _forward_metax
    _patched = True
    logger.info("fl opt7: fused sampler tail armed (bf16 direct + temp fold)")
    return True


_orig_tts_init = _tts_mod.TopKTopPSampler.__init__


def _tts_init_metax(self, *args, **kwargs):
    _orig_tts_init(self, *args, **kwargs)
    # Sampler.__init__ constructs TopKTopPSampler: at this point
    # vllm.v1.sample.sampler is fully imported and no request has been
    # sampled yet.
    _apply()


_tts_mod.TopKTopPSampler.__init__ = _tts_init_metax

# Non-circular contexts (tests, direct imports): apply immediately.
_apply()
