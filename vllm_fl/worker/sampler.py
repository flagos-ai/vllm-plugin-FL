# SPDX-License-Identifier: Apache-2.0
"""Vendor-neutral sort-free Top-P / Gumbel-Max sampler & chunked low-VRAM penalty hooks.

Provides a unified hook contract (`install_sort_free_sampler_hooks`) shared across
hardware backends to solve two major bottlenecks in vLLM V1 `Sampler`:
1. Full-vocabulary sort elimination: delegates standard Top-P / temperature /
   greedy sampling (`top_k is None`) directly to backend `fast_sample_fn`
   kernels (dual-pivot / threshold-search Gumbel-Max without `O(V log V)` radix sort).
2. Peak HBM workspace compression: replaces monolithic `[batch_size, vocab_size]`
   FP32 logits upcasting and penalty application (which spikes hundreds of MB of
   HBM at large batch sizes and reduces available KV-cache blocks) with fixed
   micro-batch slicing (`chunk_b = 4` with penalties, `8` without penalties),
   bounding peak temporary workspace to a few megabytes.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import torch

_ORIG_SAMPLER_FORWARD: Any = None
_SAMPLER_HOOKS_INSTALLED: bool = False
_WARMED_SAMPLER_SIGS: set[tuple[torch.device, torch.dtype, int]] = set()


def _are_logitsprocs_noop(logitsprocs: Any, only_non_argmax: bool = False) -> bool:
    """Return True if no custom logits processor has active per-request state in this step."""
    if logitsprocs is None:
        return True
    non_argmax_attrs = ("biases", "min_toks", "req_info")
    for p in getattr(logitsprocs, "non_argmax_invariant", ()):
        if (
            any(bool(getattr(p, a, True)) for a in non_argmax_attrs if hasattr(p, a))
            or not any(hasattr(p, a) for a in non_argmax_attrs)
        ):
            return False
    if only_non_argmax:
        return True
    argmax_attrs = ("min_p_count", "req_info")
    for p in getattr(logitsprocs, "argmax_invariant", ()):
        if (
            any(bool(getattr(p, a, True)) for a in argmax_attrs if hasattr(p, a))
            or not any(hasattr(p, a) for a in argmax_attrs)
        ):
            return False
    return True


def install_sort_free_sampler_hooks(
    fast_sample_fn: Callable[..., Any],
    *,
    apply_temp_fn: Callable[..., torch.Tensor] | None = None,
    prewarm_fn: Callable[[torch.device, torch.dtype, int], None] | None = None,
    apply_topk_topp_fn: Callable[..., torch.Tensor] | None = None,
    topk_topp_forward_fn: Callable[..., torch.Tensor] | None = None,
) -> None:
    """Install chunked low-VRAM penalties and sort-free Top-P sampling onto vLLM `Sampler`.

    Args:
        fast_sample_fn: Backend kernel implementing fused temperature + Top-P /
            greedy sampling directly from `[B, V]` logits without sorting.
        apply_temp_fn: Optional custom in-place temperature scaling kernel.
        prewarm_fn: Optional one-time kernel warmup callback keyed by
            `(device, dtype, vocab_size)` outside CUDA Graph capture.
        apply_topk_topp_fn: Optional custom joint `(top_k, top_p)` logits filter.
        topk_topp_forward_fn: Optional fused `(top_k, top_p)` random sampling kernel.
    """
    global _ORIG_SAMPLER_FORWARD, _SAMPLER_HOOKS_INSTALLED
    if _SAMPLER_HOOKS_INSTALLED:
        return
    import vllm.v1.sample.ops.topk_topp_sampler as tts_mod
    import vllm.v1.sample.sampler as sampler_mod

    temp_fn = apply_temp_fn or sampler_mod.Sampler.apply_temperature
    orig_apply_topk_topp = apply_topk_topp_fn or tts_mod.apply_top_k_top_p

    def _fast_topk_topp_forward(
        self: Any,
        logits: torch.Tensor,
        generators: dict[int, torch.Generator],
        k: torch.Tensor | None,
        p: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Dispatch Top-P/Top-K sampling via sort-free `fast_sample_fn`, fused kernel, or native fallback."""
        # Fast path 1: pure Top-P (`k is None`) with no processed-logits export -> sort-free kernel.
        if (
            k is None
            and logits.is_cuda
            and logits.dtype in (torch.bfloat16, torch.float16, torch.float32)
            and not self.use_fp64_gumbel
            and self.logprobs_mode not in ("processed_logits", "processed_logprobs")
        ):
            out = torch.empty((logits.shape[0],), dtype=torch.int64, device=logits.device)
            if logits.shape[0] > 0:
                fast_sample_fn(
                    logits,
                    p=p,
                    temp=None,
                    generators=generators,
                    out=out,
                    all_greedy=False,
                    use_splitv=True,
                )
            return out, None
        # Fast path 2: optional backend-fused joint (top_k, top_p) sampler.
        if (
            topk_topp_forward_fn is not None
            and p is not None
            and logits.shape[0] > 0
            and not self.use_fp64_gumbel
            and self.logprobs_mode not in ("processed_logits", "processed_logprobs")
        ):
            return topk_topp_forward_fn(logits, generators, k, p), None
        from vllm.v1.sample.ops.topk_topp_sampler import random_sample

        logits = orig_apply_topk_topp(logits, k, p)
        if self.logprobs_mode == "processed_logits":
            ret = logits
        elif self.logprobs_mode == "processed_logprobs":
            ret = logits.log_softmax(dim=-1, dtype=torch.float32)
        else:
            ret = None
        probs = logits.softmax(dim=-1, dtype=torch.float32)
        return random_sample(probs, generators, self.use_fp64_gumbel), ret

    if _ORIG_SAMPLER_FORWARD is None:
        _ORIG_SAMPLER_FORWARD = sampler_mod.Sampler.forward

    def _fast_sampler_forward(
        self: Any,
        logits: torch.Tensor,
        sampling_metadata: Any,
        predict_bonus_token: bool = False,
        logprobs_mode_override: Any = None,
    ) -> Any:
        """Intercept `Sampler.forward` with sort-free direct sampling and chunked low-VRAM penalties."""
        if logits.is_cuda and logits.ndim == 2 and logits.dtype in (torch.bfloat16, torch.float16, torch.float32):
            batch_size, vocab_size = logits.shape
            warm_key = (logits.device, logits.dtype, vocab_size)
            # One-time kernel prewarm per (device, dtype, vocab_size) outside CUDA Graph capture.
            if (
                prewarm_fn is not None
                and warm_key not in _WARMED_SAMPLER_SIGS
                and batch_size > 0
                and sampling_metadata.top_p is not None
                and sampling_metadata.temperature is not None
                and not torch.cuda.is_current_stream_capturing()
            ):
                _WARMED_SAMPLER_SIGS.add(warm_key)
                prewarm_fn(logits.device, logits.dtype, vocab_size)

            default_lp = getattr(self.topk_topp_sampler, "logprobs_mode", "raw_logprobs")
            lp_mode = logprobs_mode_override or getattr(self, "logprobs_mode", default_lp)
            tb_holder = getattr(sampling_metadata, "thinking_budget_state_holder", None)
            if (
                sampling_metadata.max_num_logprobs is None
                and not getattr(sampling_metadata, "logprob_token_ids", None)
                and not sampling_metadata.bad_words_token_ids
                and getattr(sampling_metadata, "allowed_token_ids_mask", None) is None
                and _are_logitsprocs_noop(
                    getattr(sampling_metadata, "logitsprocs", None),
                    only_non_argmax=bool(sampling_metadata.all_greedy),
                )
                and (tb_holder is None or not tb_holder.has_tracked_requests())
                and lp_mode not in ("processed_logits", "processed_logprobs")
                and not getattr(self.topk_topp_sampler, "use_fp64_gumbel", False)
            ):
                from vllm.v1.outputs import SamplerOutput

                if batch_size == 0:
                    empty_ids = torch.empty((0, 1), dtype=torch.int32, device=logits.device)
                    return SamplerOutput(sampled_token_ids=empty_ids, logprobs_tensors=None)
                # Zero-allocation fast path when neither penalties nor Top-K filtering are active.
                if sampling_metadata.no_penalties and sampling_metadata.top_k is None:
                    out_toks = torch.empty((batch_size, 1), dtype=torch.int32, device=logits.device)
                    fast_sample_fn(
                        logits,
                        p=sampling_metadata.top_p,
                        temp=sampling_metadata.temperature,
                        generators=sampling_metadata.generators,
                        out=out_toks,
                        all_greedy=bool(sampling_metadata.all_greedy),
                        use_splitv=True,
                    )
                    return SamplerOutput(sampled_token_ids=out_toks, logprobs_tensors=None)

                from vllm.v1.sample.ops.penalties import apply_all_penalties

                out_ids = sampling_metadata.output_token_ids
                if (
                    predict_bonus_token
                    and not sampling_metadata.no_penalties
                    and getattr(sampling_metadata, "spec_token_ids", None) is not None
                ):
                    out_ids = self._combine_outputs_with_spec_tokens(
                        out_ids, sampling_metadata.spec_token_ids
                    )

                # Process FP32 logits and repetition/frequency/presence penalties in
                # small batch slices (`chunk_b = 4` with penalties, `8` otherwise)
                # so temporary FP32 allocations stay at ~2.4-4.8 MB instead of O(B * V * 4B).
                has_topk = sampling_metadata.top_k is not None
                chunk_b = 4 if not sampling_metadata.no_penalties else 8
                sampled_out = torch.empty((batch_size,), dtype=torch.int32, device=logits.device)
                chunk_buf = torch.empty(
                    (min(batch_size, chunk_b), vocab_size),
                    dtype=torch.float32,
                    device=logits.device,
                )
                for s in range(0, batch_size, chunk_b):
                    e = min(s + chunk_b, batch_size)
                    chunk = chunk_buf[: e - s].copy_(logits[s:e])
                    if not sampling_metadata.no_penalties:
                        chunk = apply_all_penalties(
                            chunk,
                            sampling_metadata.prompt_token_ids[s:e],
                            sampling_metadata.presence_penalties[s:e],
                            sampling_metadata.frequency_penalties[s:e],
                            sampling_metadata.repetition_penalties[s:e],
                            out_ids[s:e],
                        )
                    t_c = sampling_metadata.temperature[s:e] if sampling_metadata.temperature is not None else None
                    p_c = sampling_metadata.top_p[s:e] if sampling_metadata.top_p is not None else None
                    # Remap per-request RNG generators to chunk-local batch indices.
                    sub_gens = (
                        {i - s: g for i, g in sampling_metadata.generators.items() if s <= i < e}
                        if sampling_metadata.generators
                        else {}
                    )
                    if not has_topk:
                        fast_sample_fn(
                            chunk,
                            p=p_c,
                            temp=t_c,
                            generators=sub_gens or None,
                            out=sampled_out[s:e],
                            all_greedy=bool(sampling_metadata.all_greedy),
                            use_splitv=True,
                        )
                    elif sampling_metadata.all_greedy:
                        sampled_out[s:e] = chunk.argmax(dim=-1).to(torch.int32)
                    else:
                        greedy_c = None if sampling_metadata.all_random else chunk.argmax(dim=-1).to(torch.int32)
                        chunk = temp_fn(chunk, t_c, sampling_metadata.all_random)
                        rand_c = self.topk_topp_sampler(
                            chunk, sub_gens, sampling_metadata.top_k[s:e], p_c
                        )[0].to(torch.int32)
                        if greedy_c is not None and t_c is not None:
                            torch.where(t_c < 1e-5, greedy_c, rand_c, out=sampled_out[s:e])
                        else:
                            sampled_out[s:e] = rand_c
                del chunk_buf
                return SamplerOutput(sampled_token_ids=sampled_out.unsqueeze(-1), logprobs_tensors=None)

        return _ORIG_SAMPLER_FORWARD(self, logits, sampling_metadata, predict_bonus_token, logprobs_mode_override)

    if apply_topk_topp_fn is not None:
        tts_mod.apply_top_k_top_p = apply_topk_topp_fn
    tts_mod.TopKTopPSampler.forward_native = _fast_topk_topp_forward
    tts_mod.TopKTopPSampler.forward = _fast_topk_topp_forward
    if apply_temp_fn is not None:
        sampler_mod.Sampler.apply_temperature = staticmethod(apply_temp_fn)
    sampler_mod.Sampler.forward = _fast_sampler_forward
    _SAMPLER_HOOKS_INSTALLED = True


__all__ = ["install_sort_free_sampler_hooks"]
