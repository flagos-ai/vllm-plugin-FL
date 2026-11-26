# SPDX-License-Identifier: Apache-2.0
"""Keep unprocessed BF16 logits on the model device for native top-p sampling."""
from functools import wraps
from collections.abc import MutableMapping


class _FilterScratch(MutableMapping):
    """Scope the stock top-p kernel's mutable sorting buffer to its execution stream."""
    def __init__(self, existing):
        self._entries = {}
        self.update(existing)

    @staticmethod
    def _physical_key(key):
        import torch
        return key, torch.cuda.current_stream(key[0]).cuda_stream

    def __getitem__(self, key):
        return self._entries[self._physical_key(key)]

    def __setitem__(self, key, value):
        self._entries[self._physical_key(key)] = value

    def __delitem__(self, key):
        del self._entries[self._physical_key(key)]

    def __iter__(self):
        return (key for key, stream in tuple(self._entries) if self._physical_key(key)[1] == stream)

    def __len__(self):
        return sum(1 for _ in self)


def isolate_filter_scratch():
    from vllm.v1.sample.ops import topk_topp_triton
    existing = topk_topp_triton._TRITON_BUFFER_CACHE
    if not isinstance(existing, _FilterScratch):
        topk_topp_triton._TRITON_BUFFER_CACHE = _FilterScratch(existing)


def install_bf16_sampling():
    import torch
    from flag_gems.fused.iluvatar_hist_sampler import sample_bf16
    from vllm.v1.outputs import SamplerOutput
    from vllm.logger import init_logger
    from vllm.v1.sample.logits_processor.builtin import (
        LogitBiasLogitsProcessor, MinPLogitsProcessor, MinTokensLogitsProcessor,
    )
    from vllm.v1.sample.ops import topk_topp_sampler
    from vllm.v1.sample.sampler import Sampler

    if getattr(Sampler, "_iluvatar_bf16_sampling", False) or not topk_topp_sampler.HAS_TRITON:
        return
    isolate_filter_scratch()
    inactive_fields = {LogitBiasLogitsProcessor: "biases", MinPLogitsProcessor: "min_p_count",
                       MinTokensLogitsProcessor: "min_toks"}
    original = Sampler.forward
    log = init_logger(__name__)
    first_direct_batch = True

    @wraps(original)
    def sample(self, logits, sampling_metadata, predict_bonus_token=False, logprobs_mode_override=None):
        nonlocal first_direct_batch
        m = sampling_metadata
        if (logits.is_cuda and logits.dtype == torch.bfloat16 and logits.ndim == 2
                and 8 <= logits.shape[0] <= 64 and logits.shape[1] == 130560
                and m.all_random and not m.all_greedy and not predict_bonus_token
                and not self.use_fp64_gumbel and (logprobs_mode_override or self.logprobs_mode)
                not in ("processed_logits", "processed_logprobs")
                and m.top_k is None and m.top_p is not None and not m.generators
                and m.max_num_logprobs is None and not m.logprob_token_ids and m.no_penalties
                and m.allowed_token_ids_mask is None and not m.bad_words_token_ids
                and (m.thinking_budget_state_holder is None or not m.thinking_budget_state_holder.has_tracked_requests())
                and all(type(proc) in inactive_fields and not getattr(proc, inactive_fields[type(proc)])
                        for proc in m.logitsprocs.all)):
            result = sample_bf16(logits, m.top_p, m.temperature, topk_topp_sampler.apply_top_k_top_p)
            if result is not None:
                if first_direct_batch:
                    first_direct_batch = False
                    log.info("Iluvatar BF16 direct sampling active for %d rows and %d tokens per vocabulary",
                             logits.shape[0], logits.shape[1])
                return SamplerOutput(sampled_token_ids=result.unsqueeze(-1), logprobs_tensors=None)
        return original(self, logits, m, predict_bonus_token, logprobs_mode_override)

    Sampler.forward = sample
    Sampler._iluvatar_bf16_sampling = True
