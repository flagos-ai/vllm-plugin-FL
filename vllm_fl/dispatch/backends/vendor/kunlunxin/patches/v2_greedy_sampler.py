# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0

"""Use native argmax for Kunlunxin's V2 non-speculative greedy batches."""

import numpy as np
import torch


def apply_v2_greedy_sampler() -> None:
    from vllm.v1.worker.gpu import model_runner
    from vllm.v1.worker.gpu.sample.sampler import Sampler

    if model_runner.Sampler is not Sampler:
        return

    class KunlunxinSampler(Sampler):
        def __init__(self, vllm_config, *args, **kwargs):
            super().__init__(vllm_config, *args, **kwargs)
            self._non_speculative = vllm_config.speculative_config is None

        def _sample_random(
            self,
            processed_logits,
            expanded_idx_mapping,
            idx_mapping_np,
            pos,
            top_k,
            top_p,
            use_fused_sampler,
        ):
            if (
                self._non_speculative
                and top_k is None
                and top_p is None
                and not use_fused_sampler
                and processed_logits.shape[0] == len(idx_mapping_np)
                and np.all(self.sampling_states.temperature.np[idx_mapping_np] == 0.0)
            ):
                # Upstream's Triton Gumbel reduction disagrees with argmax for
                # some P800 batch rows even when temperature is zero.
                return torch.argmax(processed_logits.float(), dim=-1), processed_logits

            return super()._sample_random(
                processed_logits,
                expanded_idx_mapping,
                idx_mapping_np,
                pos,
                top_k,
                top_p,
                use_fused_sampler,
            )

    model_runner.Sampler = KunlunxinSampler
