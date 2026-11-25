# Copyright (c) 2026 BAAI. All rights reserved.
"""LPU sampler patch.

The top-k/top-p sampler's Triton kernel cannot run on LPU, so this patch
routes onto upstream's pure-PyTorch ``apply_top_k_top_p_pytorch`` (like
metax/kunlunxin/musa), bypassing only the Triton dispatcher. It also
replaces ``apply_temperature``: its ``temp < _SAMPLING_EPS`` launches a
native ``lt`` that under TP>1 runs concurrently during
``_dummy_sampler_run`` and segfaults the driver (``hpe_runtime.c:2392``);
the comparison is deterministic there (dummy 0.5 >= 1e-5); skipping it is identical.

"""

import os

import torch

_PATCHED = False

_SAMPLING_EPS = 1e-5

_DEVICE_ARGMAX_ENV = "VLLM_FL_SAMPLER_DEVICE_ARGMAX"


def _apply_temperature_lpu(
    self,
    logits: torch.Tensor,
    temp: torch.Tensor,
    all_random: bool,
) -> torch.Tensor:
    """``apply_temperature`` with the ``temp < eps`` clamp computed host-side.

    Upstream declares this a ``@staticmethod`` (``Sampler.apply_temperature``),
    so when it is called as ``self.apply_temperature(logits, temp, all_random)``
    the instance is passed as the first positional argument.  The replacement
    therefore must accept ``self`` and ignore it.

    Upstream body does ``temp = torch.where(temp < _SAMPLING_EPS, 1.0, temp)``
    for ``not all_random``, which launches a native LPU ``lt`` kernel.  Under
    TP>1 every worker launches the same kernel concurrently (KV-cache profile
    run), and the driver's unlocked ``_module_check_by_kname_nolock`` cache
    lookup (hpe_runtime.c:2392) segfaults on the concurrent access.  ``temp``
    is tiny (``(num_reqs,)``), so the clamp is computed host-side instead —
    identical result: ``where(False, 1.0, temp) == temp`` for the 0.5 dummy
    temperature, and the eps-clamp-to-1.0 for greedy ``temp == 0`` rows is
    preserved.
    """
    if not all_random:
        temp_cpu = temp.cpu()
        temp = torch.where(
            temp_cpu < _SAMPLING_EPS,
            torch.tensor(1.0, dtype=temp_cpu.dtype),
            temp_cpu,
        ).to(device=logits.device)
    return logits.div_(temp.unsqueeze(dim=1))


def _random_sample_lpu(
    probs: torch.Tensor,
    generators: dict[int, torch.Generator],
    use_fp64_gumbel: bool = False,
) -> torch.Tensor:
    """``random_sample`` with the final argmax computed host-side.

    Upstream ``random_sample`` ends with ``probs.div_(q).argmax(dim=-1)``.  On
    LPU ``Tensor.argmax`` dispatches to the flaggems evas Triton argmax override
    (``_evas/ops/argmax.py``), which compiles/launches a reduction kernel.  Under
    TP>1 every worker compiles the same kernel concurrently during the KV-cache
    profile run (``_dummy_sampler_run`` → ``topk_topp_sampler`` → ``random_sample``
    on a full-vocab ``(num_reqs, 129280)`` logits), and the concurrent Triton
    compilation + launch segfaults / fails ``RT stream synchronize`` in the evas
    driver (same family as the ``lt`` crash).  ``probs`` is
    ``(num_reqs, vocab)`` — huge — so only the tiny ``argmax`` index is brought
    host-side.  The Gumbel-max sampling distribution is unchanged (``q``
    exponential + ``div_`` still run on LPU); only the index reduction is done on
    the host.  The added CPU-GPU sync is irrelevant for the profile run and
    acceptable for TP4 bring-up.

    ``use_fp64_gumbel`` (vllm 0.24) is ignored: LPU aten supports only
    bf16/fp16/int8/uint8 (no float64), so the fp64 Gumbel option is not
    applicable.
    """
    q = torch.empty_like(probs)
    if len(generators) != probs.shape[0]:
        q.exponential_()
    if generators:
        for i, generator in generators.items():
            q[i].exponential_(generator=generator)
    scored = probs.div_(q)
    # Host-side argmax: avoids the concurrent Triton argmax reduction on LPU.
    return scored.cpu().argmax(dim=-1).view(-1).to(device=probs.device)


def _greedy_sample_lpu(
    self,
    logits: torch.Tensor,
) -> torch.Tensor:
    """``Sampler.greedy_sample`` with the argmax computed host-side by default.

    Upstream declares this a ``@staticmethod``, so the instance is passed as the
    first positional argument when called as ``self.greedy_sample(logits)`` —
    the replacement must accept and ignore ``self``.

    Default (host) path: upstream body is ``logits.argmax(dim=-1)``, which on
    LPU dispatches to the flaggems evas Triton argmax override.  Under TP>1
    every worker launches the same reduction concurrently during
    ``_dummy_sampler_run`` and the driver's concurrent Triton compile/launch
    fails with ``RT stream synchronize`` (see ``_random_sample_lpu``).
    ``logits`` is ``(num_reqs, vocab)`` — huge — so the default brings only the
    tiny index host-side.

    Opt-in device path (``VLLM_FL_SAMPLER_DEVICE_ARGMAX=1``, single request
    only): keeps the argmax on the device and skips the per-step full-vocab
    ``(num_reqs, vocab)`` float32 D2H + forced sync — the largest single
    per-step D2H payload in the greedy decode path
    (``VLLM_COMPUTE_NANS_IN_LOGITS`` is a separate per-step full-vocab device
    pass with only a tiny D2H).  This inverts the original
    crash condition, so it must be validated per topology before use:
    single-process (TP1) is safe; under TP>1 the concurrent first-compile race
    above may resurface unless the argmax kernel cache is warm before
    ``_dummy_sampler_run``.  Numerics are expected to match the host path (same
    torch-argmax tie semantics) — verified by a TP1 golden A/B run, NOT by the
    CPU unit tests (both branches are torch.argmax there); rows containing NaN
    or all-``-inf`` go through the flaggems override's reduction and are the
    residual divergence risk (platform logs have shown NaN logits — see
    ``VLLM_COMPUTE_NANS_IN_LOGITS``), so enable per-topology with a token
    A/B, not by default.
    """
    # The device arm is held to a single row: the vendored evas argmax kernel
    # returns wrong ids for multi-row inputs whose reduced extent exceeds
    # 32768, and that is silent wrong output.
    # Two or more requests keep the host path.
    if os.environ.get(_DEVICE_ARGMAX_ENV, "0") == "1" and logits.shape[0] == 1:
        return logits.argmax(dim=-1).view(-1)
    return logits.cpu().argmax(dim=-1).view(-1).to(device=logits.device)


def _sample_lpu(self, logits, sampling_metadata, logprobs_mode_override=None):
    """``Sampler.sample`` with the two ``temperature < eps`` lt kernels host-side.

    Mirror of upstream vLLM ``Sampler.sample``; the only difference is the
    final greedy/random ``torch.where`` selection reads ``temperature < eps``
    from a host copy instead of launching a native LPU ``lt`` kernel (same
    concurrent-TP>1 driver segfault as ``apply_temperature`` — see its
    docstring).  ``_dummy_sampler_run`` (all_greedy=False, all_random=False)
    reaches this where, so both lt sites must be patched.
    """
    logprobs_mode = logprobs_mode_override or self.logprobs_mode
    assert not (sampling_metadata.all_greedy and sampling_metadata.all_random)
    if sampling_metadata.all_random:
        greedy_sampled = None
    else:
        greedy_sampled = self.greedy_sample(logits)
        if sampling_metadata.all_greedy:
            processed_logprobs = None
            if (
                sampling_metadata.max_num_logprobs is not None
                or sampling_metadata.logprob_token_ids
            ):
                if logprobs_mode == "processed_logits":
                    processed_logprobs = logits
                elif logprobs_mode == "processed_logprobs":
                    processed_logprobs = self.compute_logprobs(logits)
            return greedy_sampled, processed_logprobs

    assert sampling_metadata.temperature is not None

    # Apply temperature.
    logits = self.apply_temperature(
        logits, sampling_metadata.temperature, sampling_metadata.all_random
    )

    # Apply logits processors that only apply to random sampling
    # (argmax invariant).
    for processor in sampling_metadata.logitsprocs.argmax_invariant:
        logits = processor.apply(logits)

    # Apply top_k and/or top_p.
    random_sampled, processed_logprobs = self.topk_topp_sampler(
        logits,
        sampling_metadata.generators,
        sampling_metadata.top_k,
        sampling_metadata.top_p,
    )

    if greedy_sampled is None:
        return random_sampled, processed_logprobs

    # LPU: host-side temperature<eps mask (no native lt kernel).  Mirrors the
    # upstream `torch.where(sampling_metadata.temperature < _SAMPLING_EPS, ...)`.
    sampled = torch.where(
        (sampling_metadata.temperature.cpu() < _SAMPLING_EPS).to(logits.device),
        greedy_sampled,
        random_sampled,
        out=greedy_sampled,  # Reuse tensor
    )
    return sampled, processed_logprobs


def apply_lpu_sampler_patch() -> None:
    """Patch vLLM's sampler to avoid Triton/lt kernels on LPU."""
    global _PATCHED
    if _PATCHED:
        return
    import vllm.v1.sample.ops.topk_topp_sampler as _topk_topp_sampler
    import vllm.v1.sample.sampler as _sampler

    _topk_topp_sampler.apply_top_k_top_p = (
        _topk_topp_sampler.apply_top_k_top_p_pytorch
    )
    _topk_topp_sampler.random_sample = _random_sample_lpu
    _sampler.Sampler.apply_temperature = _apply_temperature_lpu
    _sampler.Sampler.greedy_sample = _greedy_sample_lpu
    _sampler.Sampler.sample = _sample_lpu
    _PATCHED = True
