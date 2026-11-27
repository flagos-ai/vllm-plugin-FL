# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 FlagOS Contributors

# MetaX: sample with FlagGems' fused top-p sampler.
#
# For a batch that only needs temperature / top-p random sampling (vLLM's
# default when generation_config sets temperature and top_p, as MiniCPM5's
# does), Sampler.forward converts the bf16 logits to fp32, divides by the
# temperature, masks the nucleus, takes a softmax, draws exponential noise and
# an argmax: ~15 passes over [batch, vocab] fp32 and ~17 kernel launches per
# step (2.8 ms of a 21.4 ms decode step at batch 64, vocab 130560).
# flag_gems.fused.top_p_sample computes the same sampling function in two
# passes over the bf16 logits (4 launches).  Every batch the fused kernel does
# not cover (logprobs, penalties, bad words, allowed-token masks, active logits
# processors, top-k, per-request generators, spec decode, processed logprobs)
# keeps the original forward.  VLLM_FL_METAX_FUSED_SAMPLER=0 disables it.

import os

import torch
import vllm.v1.sample.sampler as sampler_mod
from vllm.logger import init_logger
from vllm.v1.outputs import SamplerOutput

logger = init_logger(__name__)

_ENABLED = os.environ.get("VLLM_FL_METAX_FUSED_SAMPLER", "1").lower() not in ("0", "false", "off")
_original_forward = sampler_mod.Sampler.forward


def _logitsprocs_inactive(logitsprocs) -> bool:
    from vllm.v1.sample.logits_processor.builtin import (
        LogitBiasLogitsProcessor,
        MinPLogitsProcessor,
        MinTokensLogitsProcessor,
    )

    for proc in logitsprocs.all:
        if isinstance(proc, MinPLogitsProcessor):
            if proc.min_p_count:
                return False
        elif isinstance(proc, LogitBiasLogitsProcessor):
            if proc.biases:
                return False
        elif isinstance(proc, MinTokensLogitsProcessor):
            if proc.min_toks:
                return False
        else:  # custom processor: cannot tell whether it is a no-op
            return False
    return True


def _fused_ok(sampler, logits, md, predict_bonus_token, logprobs_mode_override) -> bool:
    if predict_bonus_token or logprobs_mode_override is not None:
        return False
    if logits.dim() != 2 or logits.stride(1) != 1 or logits.shape[0] == 0:
        return False
    if logits.dtype not in (torch.bfloat16, torch.float16, torch.float32):
        return False
    if md.all_greedy or md.temperature is None:
        return False
    if md.max_num_logprobs is not None or md.logprob_token_ids:
        return False
    if md.top_k is not None or md.generators:
        return False
    if sampler.use_fp64_gumbel:
        return False
    if not md.no_penalties or md.bad_words_token_ids or md.allowed_token_ids_mask is not None:
        return False
    holder = md.thinking_budget_state_holder
    if holder is not None and holder.has_tracked_requests():
        return False
    return _logitsprocs_inactive(md.logitsprocs)


_warmed_up = False
_kernel = None


def _top_p_sample():
    # Resolved on first use (FlagGems is imported by the worker, not by this
    # patch).  A FlagGems without the fused sampler disables the fast path
    # instead of failing every step.
    global _kernel, _ENABLED
    if _kernel is None and _ENABLED:
        try:
            from flag_gems.fused import top_p_sample

            _kernel = top_p_sample
        except Exception as e:
            _ENABLED = False
            logger.warning("MetaX fused sampler unavailable, keeping vLLM's sampler: %s", e)
    return _kernel


def _warm_up(logits) -> None:
    # Compile the fused kernels during the engine's dummy sampler run (startup),
    # not inside the first served step.  One variant per vocabulary size and
    # (top_p given or not); batch sizes do not add variants.
    global _warmed_up
    _warmed_up = True
    kernel = _top_p_sample()
    if kernel is None:
        return
    try:
        x = logits[:2].contiguous()
        temp = torch.ones(x.shape[0], device=x.device, dtype=torch.float32)
        kernel(x, temp, torch.full_like(temp, 0.9))
        kernel(x, temp, None)
    except Exception as e:  # never block startup; the first real step compiles instead
        logger.warning("MetaX fused sampler warm-up failed: %s", e)


def _forward(self, logits, sampling_metadata, predict_bonus_token=False,
             logprobs_mode_override=None):
    if not _warmed_up and _ENABLED and logits.dim() == 2 and logits.stride(1) == 1:
        _warm_up(logits)
    if _ENABLED and _fused_ok(self, logits, sampling_metadata, predict_bonus_token,
                              logprobs_mode_override):
        kernel = _top_p_sample()
        if kernel is not None:
            sampled = kernel(logits, sampling_metadata.temperature, sampling_metadata.top_p)
            return SamplerOutput(sampled_token_ids=sampled.unsqueeze(-1),
                                 logprobs_tensors=None)
    return _original_forward(self, logits, sampling_metadata, predict_bonus_token,
                             logprobs_mode_override)


if _ENABLED:
    sampler_mod.Sampler.forward = _forward
    logger.info("MetaX: Sampler uses FlagGems fused top-p sampling for plain random batches")
