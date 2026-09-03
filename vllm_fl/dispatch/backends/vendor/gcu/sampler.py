# Copyright (c) 2026 BAAI. All rights reserved.

"""GCU top-k/top-p sampler patch.

torch_gcu does not implement torch.Generator-seeded tensor in-place ops:
``Tensor.exponential_(generator=...)`` raises ``TypeError: exponential_() got
an unexpected keyword argument 'generator'`` on GCU devices.  This hits both
real per-request-seeded decoding and vLLM's dummy sampler warm-up
(``model_runner._dummy_sampler_run`` re-runs ``forward_native`` with a
non-empty generators dict during memory profiling).

Like the MetaX ``apply_top_k_top_p`` patch, this module monkey-patches
``vllm.v1.sample.ops.topk_topp_sampler``, but targets ``random_sample``:
on GCU the exponential noise is always drawn seed-free (``q.exponential_()``),
so per-request generators are ignored.  Sampling stays statistically correct;
per-request seed reproducibility is simply not supported by the torch_gcu
runtime.
"""

import torch

import vllm.v1.sample.ops.topk_topp_sampler as topk_topp_sampler


def _random_sample_gcu(
    probs: torch.Tensor,
    generators: dict[int, torch.Generator],
    use_fp64_gumbel: bool = False,
) -> torch.Tensor:
    del generators  # per-request seeds unsupported on torch_gcu
    q = topk_topp_sampler.empty_exponential_noise_like(probs, use_fp64_gumbel)
    q.exponential_()
    return topk_topp_sampler.sample_with_exponential_noise(probs, q)


# Replace random_sample so the per-request-generator branch is never taken.
topk_topp_sampler.random_sample = _random_sample_gcu
