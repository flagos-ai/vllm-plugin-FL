"""Reuse W8 lm-head reductions for Batch=1 greedy and MTP-1 decode."""

from __future__ import annotations

import os

import torch


_ENABLED = True
_SPEC_ENABLED = True
_PATCHED = False
_HITS = 0
_SPEC_HITS = 0


def set_w8_cached_greedy_enabled(enabled: bool) -> bool:
    """Set the in-process A/B route and return its previous value."""
    global _ENABLED
    previous = _ENABLED
    _ENABLED = bool(enabled)
    os.environ["FLAGGEMS_W8_CACHED_GREEDY"] = "1" if _ENABLED else "0"
    return previous


def set_w8_cached_spec_greedy_enabled(enabled: bool) -> bool:
    """Set the in-process MTP rejection fast-path for controlled A/B tests."""
    global _SPEC_ENABLED
    previous = _SPEC_ENABLED
    _SPEC_ENABLED = bool(enabled)
    return previous


def w8_cached_greedy_hits() -> int:
    return _HITS


def w8_cached_spec_greedy_hits() -> int:
    return _SPEC_HITS


def _eligible(logits, metadata) -> bool:
    processors = metadata.logitsprocs
    temperature = metadata.temperature
    zero_temperature = (
        temperature is not None
        and temperature.numel() == 1
        and float(temperature.reshape(-1)[0]) < 1.0e-5
    )
    processor_state_is_safe = True
    for processor in processors.non_argmax_invariant:
        name = type(processor).__name__
        if name == "LogitBiasLogitsProcessor":
            processor_state_is_safe &= not processor.biases
        elif name == "MinTokensLogitsProcessor":
            # This processor only masks stop-token IDs.  The fast path checks
            # the cached winner against those IDs before accepting it.
            pass
        elif name == "ThinkingTokenBudgetLogitsProcessor":
            processor_state_is_safe &= not processor.is_enabled
        else:
            processor_state_is_safe = False
    return (
        _ENABLED
        and logits.device.type == "cpu"
        and logits.dtype == torch.bfloat16
        and logits.dim() == 2
        and logits.shape[0] == 1
        and logits.is_contiguous()
        # vLLM 0.20.2 can classify a single zero-temperature request as a
        # mixed batch (all_greedy=False).  The sampler nevertheless chooses
        # its greedy result for that only row, so recognize the semantics
        # directly instead of relying on the aggregate classification bit.
        and not metadata.all_random
        and (metadata.all_greedy or zero_temperature)
        and metadata.max_num_logprobs is None
        and not metadata.logprob_token_ids
        and metadata.allowed_token_ids_mask is None
        and not metadata.bad_words_token_ids
        and metadata.no_penalties
        and processor_state_is_safe
    )


def _masked_stop_tokens(metadata) -> set[int]:
    blocked: set[int] = set()
    for processor in metadata.logitsprocs.non_argmax_invariant:
        if type(processor).__name__ != "MinTokensLogitsProcessor":
            continue
        state = processor.min_toks.get(0)
        if state is not None:
            blocked.update(int(token) for token in state[2])
    return blocked


def _all_masked_stop_tokens(metadata) -> set[int]:
    """Conservatively union stop masks for every speculative logit row."""
    blocked: set[int] = set()
    for processor in metadata.logitsprocs.non_argmax_invariant:
        if type(processor).__name__ != "MinTokensLogitsProcessor":
            continue
        for state in processor.min_toks.values():
            blocked.update(int(token) for token in state[2])
    return blocked


def _spec_eligible(logits, metadata, sampling_metadata) -> bool:
    """Recognize the exact Batch=1/MTP-1 greedy fast-path semantics."""
    processors = sampling_metadata.logitsprocs
    temperature = sampling_metadata.temperature
    zero_temperature = (
        temperature is not None
        and temperature.numel() == 1
        and float(temperature.reshape(-1)[0]) < 1.0e-5
    )
    processor_state_is_safe = True
    for processor in processors.non_argmax_invariant:
        name = type(processor).__name__
        if name == "LogitBiasLogitsProcessor":
            processor_state_is_safe &= not processor.biases
        elif name == "MinTokensLogitsProcessor":
            # The fast path checks both cached winners against the masked
            # stop-token set before accepting the metadata.
            pass
        elif name == "ThinkingTokenBudgetLogitsProcessor":
            processor_state_is_safe &= not processor.is_enabled
        else:
            processor_state_is_safe = False
    target_indices = metadata.target_logits_indices
    bonus_indices = metadata.bonus_logits_indices
    return (
        _ENABLED
        and _SPEC_ENABLED
        and logits.device.type == "cpu"
        and logits.dtype == torch.bfloat16
        and logits.dim() == 2
        and logits.shape[0] == 2
        and logits.is_contiguous()
        and metadata.max_spec_len == 1
        and metadata.draft_token_ids.numel() == 1
        and metadata.num_draft_tokens == [1]
        and target_indices.numel() == 1
        and bonus_indices.numel() == 1
        and int(target_indices[0]) == 0
        and int(bonus_indices[0]) == 1
        and not sampling_metadata.all_random
        and (sampling_metadata.all_greedy or zero_temperature)
        and sampling_metadata.max_num_logprobs is None
        and not sampling_metadata.logprob_token_ids
        and sampling_metadata.allowed_token_ids_mask is None
        and not sampling_metadata.bad_words_token_ids
        and sampling_metadata.no_penalties
        and processor_state_is_safe
    )


def _multi_spec_eligible(logits, metadata, sampling_metadata) -> bool:
    """Recognize clean Batch=1 greedy speculative verification.

    vLLM's generic path gathers the bonus row, converts the target rows to
    FP32, clones them, and only then takes argmax.  BF16-to-FP32 is lossless,
    so a clean greedy request can reduce the original rows directly without
    changing either the target winners or rejection semantics.
    """
    if os.getenv(
        "VLLM_FL_FAST_GREEDY_SPEC_REJECTION", "0"
    ).lower() not in {"1", "true", "yes", "on"}:
        return False
    processors = sampling_metadata.logitsprocs
    temperature = sampling_metadata.temperature
    zero_temperature = (
        temperature is not None
        and temperature.numel() == 1
        and float(temperature.reshape(-1)[0]) < 1.0e-5
    )
    for processor in processors.non_argmax_invariant:
        name = type(processor).__name__
        if name == "LogitBiasLogitsProcessor":
            if processor.biases:
                return False
        elif name == "MinTokensLogitsProcessor":
            pass
        elif name == "ThinkingTokenBudgetLogitsProcessor":
            if processor.is_enabled:
                return False
        else:
            return False
    if (
        not _ENABLED
        or not _SPEC_ENABLED
        or logits.device.type != "cpu"
        or logits.dtype != torch.bfloat16
        or logits.dim() != 2
        or not logits.is_contiguous()
        or len(metadata.num_draft_tokens) != 1
        or metadata.max_spec_len <= 1
        or metadata.num_draft_tokens != [metadata.max_spec_len]
        or metadata.draft_token_ids.numel() != metadata.max_spec_len
        or logits.shape[0] != metadata.max_spec_len + 1
        or not sampling_metadata.all_greedy
        or sampling_metadata.all_random
        or not (sampling_metadata.all_greedy or zero_temperature)
        or sampling_metadata.max_num_logprobs is not None
        or sampling_metadata.logprob_token_ids
        or sampling_metadata.allowed_token_ids_mask is not None
        or sampling_metadata.bad_words_token_ids
        or not sampling_metadata.no_penalties
    ):
        return False
    target_indices = metadata.target_logits_indices
    bonus_indices = metadata.bonus_logits_indices
    expected = list(range(metadata.max_spec_len))
    return (
        target_indices.device.type == "cpu"
        and bonus_indices.device.type == "cpu"
        and target_indices.reshape(-1).tolist() == expected
        and bonus_indices.reshape(-1).tolist() == [metadata.max_spec_len]
    )


def _multi_spec_greedy_sample(logits, metadata, sampling_metadata):
    """Return exact greedy rejection output, or None for a masked winner."""
    winners = logits.argmax(dim=-1)
    blocked = _all_masked_stop_tokens(sampling_metadata)
    if blocked and any(int(winner) in blocked for winner in winners):
        return None
    draft = metadata.draft_token_ids.reshape(-1)
    width = metadata.max_spec_len
    sampled = torch.full(
        (1, width + 1), -1, dtype=torch.int32, device=logits.device
    )
    for position in range(width):
        target_token = int(winners[position])
        draft_token = int(draft[position])
        sampled[0, position] = (
            draft_token if draft_token == target_token else target_token
        )
        if draft_token != target_token:
            return sampled
    sampled[0, width] = int(winners[width])
    return sampled


def install_w8_cached_greedy() -> bool:
    """Patch vLLM's sampler once; unsupported requests retain stock behavior."""
    global _PATCHED
    if _PATCHED:
        return False
    if not hasattr(torch.ops.triton_jit_cpu, "w8_cached_argmax"):
        return False

    from vllm.v1.outputs import SamplerOutput
    from vllm.v1.sample.sampler import Sampler
    from vllm.v1.sample.rejection_sampler import RejectionSampler

    original_forward = Sampler.forward

    def forward(
        self,
        logits,
        sampling_metadata,
        predict_bonus_token=False,
        logprobs_mode_override=None,
    ):
        global _HITS
        if _eligible(logits, sampling_metadata):
            sampled = torch.ops.triton_jit_cpu.w8_cached_argmax(logits)
            if int(sampled[0]) not in _masked_stop_tokens(sampling_metadata):
                _HITS += 1
                return SamplerOutput(
                    sampled_token_ids=sampled.to(torch.int32).unsqueeze(-1),
                    logprobs_tensors=None,
                )
        return original_forward(
            self,
            logits,
            sampling_metadata,
            predict_bonus_token,
            logprobs_mode_override,
        )

    Sampler.forward = forward
    Sampler._vllm_fl_w8_cached_greedy_original_forward = original_forward

    original_spec_forward = RejectionSampler.forward

    def spec_forward(
        self,
        metadata,
        draft_probs,
        logits,
        sampling_metadata,
    ):
        global _SPEC_HITS
        if (
            not self.synthetic_mode
            and not self.is_processed_logprobs_mode
            and _spec_eligible(logits, metadata, sampling_metadata)
        ):
            winners = torch.ops.triton_jit_cpu.w8_cached_argmax(logits)
            if winners.numel() == 2:
                target = int(winners[0])
                bonus = int(winners[1])
                draft = int(metadata.draft_token_ids[0])
                blocked = _all_masked_stop_tokens(sampling_metadata)
                if target in blocked or bonus in blocked:
                    return original_spec_forward(
                        self,
                        metadata,
                        draft_probs,
                        logits,
                        sampling_metadata,
                    )
                sampled = torch.full(
                    (1, 2), -1, dtype=torch.int32, device=logits.device
                )
                sampled[0, 0] = draft if draft == target else target
                if draft == target:
                    sampled[0, 1] = bonus
                _SPEC_HITS += 1
                return SamplerOutput(
                    sampled_token_ids=sampled,
                    logprobs_tensors=None,
                )
        if (
            not self.synthetic_mode
            and not self.is_processed_logprobs_mode
            and _multi_spec_eligible(
                logits, metadata, sampling_metadata
            )
        ):
            sampled = _multi_spec_greedy_sample(
                logits, metadata, sampling_metadata
            )
            if sampled is not None:
                _SPEC_HITS += 1
                return SamplerOutput(
                    sampled_token_ids=sampled,
                    logprobs_tensors=None,
                )
        return original_spec_forward(
            self, metadata, draft_probs, logits, sampling_metadata
        )

    RejectionSampler.forward = spec_forward
    RejectionSampler._vllm_fl_w8_cached_greedy_original_forward = (
        original_spec_forward
    )
    _PATCHED = True
    return True


__all__ = [
    "install_w8_cached_greedy",
    "set_w8_cached_greedy_enabled",
    "set_w8_cached_spec_greedy_enabled",
    "w8_cached_greedy_hits",
    "w8_cached_spec_greedy_hits",
]
