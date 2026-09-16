# Copyright (c) 2026 BAAI. All rights reserved.

"""GCU top-k/top-p sampler patch.

torch_gcu's generator-backed ``Tensor.exponential_(generator=...)`` does not
draw from the generator: the values it returns grow with the element index,
scaled by the seed (seed 700 starts at ~8.4e-05, seed 123456 at ~0.015, seed 1
at ~1e-07), so element 0 is always the smallest.  vLLM samples with
``argmax(probs / q)``, so every seeded request collapses onto the same
distribution-independent token -- on uniform probabilities over 1000 tokens,
20 different seeds all return token 0.  The same op is reached for unseeded
rows through the default generator whenever the process was seeded, which
``set_random_seed(model_config.seed)`` does at worker start.

This module rebuilds the exponential noise from a uniform draw -- ``-log(u)``
with ``u ~ U(0, 1)`` is the inverse-CDF of Exp(1) -- because torch_gcu does
compute ``uniform_`` correctly from a generator, reproducibly for a given
seed.  Rows carrying their own generator are drawn from it, so per-request
seeds stay reproducible; the remaining rows share one draw from the default
generator, exactly as upstream does.

Like the MetaX ``apply_top_k_top_p`` patch, this module monkey-patches
``vllm.v1.sample.ops.topk_topp_sampler``, but targets ``random_sample``.
"""

import torch

import vllm.v1.sample.ops.topk_topp_sampler as topk_topp_sampler


def _exponential_noise_like(
    probs: torch.Tensor,
    use_fp64_gumbel: bool,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """Exp(1) noise shaped like ``probs``, drawn from ``generator`` if given."""
    dtype = torch.float64 if use_fp64_gumbel else probs.dtype
    u = torch.empty(probs.shape, dtype=dtype, device=probs.device)
    if generator is None:
        u.uniform_()
    else:
        u.uniform_(generator=generator)
    # u can be exactly 0, where -log(u) would be +inf; clamping keeps the noise
    # finite and still large enough that the element is never sampled.
    return -torch.log(u.clamp_min(torch.finfo(dtype).tiny))


def _random_sample_gcu(
    probs: torch.Tensor,
    generators: dict[int, torch.Generator],
    use_fp64_gumbel: bool = False,
) -> torch.Tensor:
    q = topk_topp_sampler.empty_exponential_noise_like(probs, use_fp64_gumbel)
    # NOTE(flagos): q.exponential_() is deliberately not used here or below --
    # see the module docstring for what torch_gcu's generator-backed
    # exponential_ returns instead of Exp(1) noise.
    if len(generators) != probs.shape[0]:
        q.copy_(_exponential_noise_like(probs, use_fp64_gumbel))
    for i, generator in generators.items():
        q[i] = _exponential_noise_like(probs[i], use_fp64_gumbel, generator)
    return topk_topp_sampler.sample_with_exponential_noise(probs, q)


def apply_random_sample_gcu_patch() -> None:
    """Replace random_sample with the Exp(1)-from-uniform implementation."""
    gcu = getattr(torch, "gcu", None)
    if gcu is None or not gcu.is_available():
        return

    topk_topp_sampler.random_sample = _random_sample_gcu
