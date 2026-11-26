# SPDX-License-Identifier: Apache-2.0
# 2026 - Modified by MetaX Integrated Circuits (Shanghai) Co., Ltd. All Rights Reserved.
"""MetaX sampler dispatch with zero-sync BF16 sampling and exact BF16 histogram top-p."""
import os
import torch
import vllm.v1.sample.ops.topk_topp_sampler as topk_topp_sampler

try:
    from flag_gems.fused.metax_exact_top_p import apply_exact_top_p_
except ImportError:
    apply_exact_top_p_ = None

try:
    from flag_gems.fused.top_p_bf16_histogram import VOCAB_WIDTH, apply_top_p_bf16_histogram_
except ImportError:
    VOCAB_WIDTH = 130560
    apply_top_p_bf16_histogram_ = None

try:
    from flag_gems.fused.top_p_pivot import apply_top_p_pivot_
except ImportError:
    apply_top_p_pivot_ = None

try:
    from flag_gems.fused.metax_sample_with_noise import metax_sample_with_noise
except ImportError:
    metax_sample_with_noise = None

try:
    from flag_gems.fused.metax_fused_sampler import metax_fused_bf16_top_p_sample
except ImportError:
    metax_fused_bf16_top_p_sample = None

_ENABLE_FUSED_BF16_SAMPLER = os.environ.get("FL_METAX_FUSED_SAMPLER", "1") != "0"


def _apply_top_k_top_p_metax(logits: torch.Tensor, k, p):
    if p is None and k is None:
        return logits

    if (
        k is None
        and p is not None
        and logits.ndim == 2
        and logits.dtype == torch.float32
        and logits.is_cuda
        and logits.stride(-1) == 1
        and logits.stride(0) >= logits.shape[1]
        and p.ndim == 1
        and p.shape[0] == logits.shape[0]
        and p.dtype == torch.float32
        and p.device == logits.device
        and p.stride(0) == 1
    ):
        batch, width = logits.shape
        if apply_top_p_bf16_histogram_ is not None and 1 <= batch <= 64 and width == VOCAB_WIDTH:
            if apply_exact_top_p_ is not None and batch >= 64:
                return apply_exact_top_p_(logits, p)
            return apply_top_p_bf16_histogram_(logits, p)
        # Short-vocabulary reference sorts can permute ties.
        if apply_top_p_pivot_ is not None and batch >= 8 and 4096 < width <= 131072:
            return apply_top_p_pivot_(logits, p)

    return topk_topp_sampler.apply_top_k_top_p_pytorch(logits, k, p)


topk_topp_sampler.apply_top_k_top_p = _apply_top_k_top_p_metax


_native_forward = topk_topp_sampler.TopKTopPSampler.forward_native


def _metax_forward(self, logits, generators, k, p):
    if (
        metax_sample_with_noise is None
        or k is not None
        or p is None
        or generators
        or self.use_fp64_gumbel
        or self.logprobs_mode in ("processed_logits", "processed_logprobs")
        or not logits.is_cuda
        or logits.dtype != torch.float32
        or logits.ndim != 2
        or not 1 <= logits.shape[0] <= 64
        or logits.shape[1] != VOCAB_WIDTH
    ):
        return _native_forward(self, logits, generators, k, p)

    filtered = topk_topp_sampler.apply_top_k_top_p(logits, k, p)
    probs = filtered.softmax(dim=-1, dtype=torch.float32)
    noise = topk_topp_sampler.empty_exponential_noise_like(probs, False)
    noise.exponential_()
    if not probs.is_contiguous():
        return topk_topp_sampler.sample_with_exponential_noise(probs, noise), None
    return metax_sample_with_noise(probs, noise), None


if metax_sample_with_noise is not None:
    topk_topp_sampler.TopKTopPSampler.forward_native = _metax_forward


_SAMPLER_PATCHED = False
_orig_sampler_forward = None
_SamplerOutput = None
_MinTokensLP = None
_LogitBiasLP = None
_MinPLP = None


def _logitsprocs_all_inactive(logitsprocs) -> bool:
    for proc in logitsprocs.non_argmax_invariant:
        if isinstance(proc, _MinTokensLP):
            if proc.min_toks:
                return False
        elif isinstance(proc, _LogitBiasLP):
            if proc.biases:
                return False
        else:
            return False
    for proc in logitsprocs.argmax_invariant:
        if isinstance(proc, _MinPLP):
            if proc.min_p_count:
                return False
        else:
            return False
    return True


def _sampler_forward_metax(
    self,
    logits: torch.Tensor,
    sampling_metadata,
    predict_bonus_token: bool = False,
    logprobs_mode_override=None,
):
    if _ENABLE_FUSED_BF16_SAMPLER and metax_fused_bf16_top_p_sample is not None:
        logprobs_mode = logprobs_mode_override or self.logprobs_mode
        holder = getattr(sampling_metadata, "thinking_budget_state_holder", None)
        if (
            sampling_metadata.max_num_logprobs is None
            and not sampling_metadata.logprob_token_ids
            and logprobs_mode not in ("processed_logits", "processed_logprobs")
            and not self.use_fp64_gumbel
            and sampling_metadata.allowed_token_ids_mask is None
            and not sampling_metadata.bad_words_token_ids
            and _logitsprocs_all_inactive(sampling_metadata.logitsprocs)
            and sampling_metadata.no_penalties
            and (holder is None or not holder.has_tracked_requests())
            and sampling_metadata.all_random
            and sampling_metadata.top_k is None
            and sampling_metadata.top_p is not None
            and sampling_metadata.temperature is not None
            and not sampling_metadata.generators
            and logits.ndim == 2
            and logits.is_cuda
            and logits.dtype == torch.bfloat16
            and logits.stride(1) == 1
            and logits.stride(0) >= logits.shape[1]
            and 1 <= logits.shape[0] <= 256
            and logits.shape[1] == VOCAB_WIDTH
            and sampling_metadata.top_p.ndim == 1
            and sampling_metadata.top_p.shape[0] == logits.shape[0]
            and sampling_metadata.top_p.dtype == torch.float32
            and sampling_metadata.top_p.stride(0) == 1
            and sampling_metadata.top_p.device == logits.device
            and sampling_metadata.temperature.ndim == 1
            and sampling_metadata.temperature.shape[0] == logits.shape[0]
            and sampling_metadata.temperature.dtype == torch.float32
            and sampling_metadata.temperature.stride(0) == 1
            and sampling_metadata.temperature.device == logits.device
        ):
            sampled = metax_fused_bf16_top_p_sample(
                logits,
                sampling_metadata.temperature,
                sampling_metadata.top_p,
            )
            return _SamplerOutput(sampled_token_ids=sampled, logprobs_tensors=None)

    return _orig_sampler_forward(
        self,
        logits,
        sampling_metadata,
        predict_bonus_token=predict_bonus_token,
        logprobs_mode_override=logprobs_mode_override,
    )


def _ensure_sampler_patched():
    global _SAMPLER_PATCHED, _orig_sampler_forward, _SamplerOutput, _MinTokensLP, _LogitBiasLP, _MinPLP
    if _SAMPLER_PATCHED or metax_fused_bf16_top_p_sample is None:
        return
    from vllm.v1.outputs import SamplerOutput
    from vllm.v1.sample.logits_processor.builtin import (
        LogitBiasLogitsProcessor,
        MinPLogitsProcessor,
        MinTokensLogitsProcessor,
    )
    from vllm.v1.sample.sampler import Sampler

    _SamplerOutput = SamplerOutput
    _MinTokensLP = MinTokensLogitsProcessor
    _LogitBiasLP = LogitBiasLogitsProcessor
    _MinPLP = MinPLogitsProcessor
    if Sampler.forward is not _sampler_forward_metax:
        _orig_sampler_forward = Sampler.forward
        Sampler.forward = _sampler_forward_metax
    _SAMPLER_PATCHED = True


_orig_tts_init = topk_topp_sampler.TopKTopPSampler.__init__


def _metax_tts_init(self, *args, **kwargs):
    _orig_tts_init(self, *args, **kwargs)
    _ensure_sampler_patched()


if metax_fused_bf16_top_p_sample is not None:
    topk_topp_sampler.TopKTopPSampler.__init__ = _metax_tts_init
