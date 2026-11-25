# Copyright (c) 2026 BAAI. All rights reserved.
"""LPU patch: vLLM speculative-decoding rejection sampler stand-ins.

The sampler's four Triton kernels plus its own ``apply_top_k_top_p`` alias are
rebound in the ``rejection_sampler`` namespace to host-driven stand-ins;
wrappers keep their upstream bodies, and the uniform-probability tensor is
drawn host-side in ``float64`` as upstream does (float32 can sample an exact
0.0; LPU float64 device storage is unproven).  Without the patch engine init
dies: memory profiling runs one dummy rejection step as soon as speculative
decoding is on.

"""

import os

import torch

from vllm.logger import init_logger

logger = init_logger(__name__)

_PATCHED = False

# Raise, instead of logging, when an upstream alias is found un-rebound.
_STRICT_REBIND_ENV = "VLLM_FL_STRICT_UPSTREAM_REBIND"

# The names read from the rejection sampler module's own namespace, split by
# whether the upstream value is a Triton kernel launch site.
_KERNEL_NAMES = (
    "expand_kernel",
    "rejection_greedy_sample_kernel",
    "rejection_random_sample_kernel",
    "sample_recovered_tokens_kernel",
)
_PLAIN_NAMES = ("generate_uniform_probs",)


class _TorchKernelLauncher:
    """Give a plain function the ``kernel[(grid,)](args...)`` call form.

    The upstream call sites launch the templates as Triton kernels
    (``expand_kernel[(batch_size,)](...)``); the replacement is subscripted the
    same way.  Mirrors ``vllm/utils/cpu_triton_utils._FuncWrapper``, which the
    upstream CPU path uses for the same purpose.
    """

    def __init__(self, fn):
        self._fn = fn

    def __getitem__(self, _grid):
        return self._fn


def _draft_ranges(cu_num_draft_tokens: torch.Tensor) -> list[tuple[int, int]]:
    """Host-side ``[start_idx, end_idx)`` per request, as the kernels compute it.

    Kernel body: ``start_idx = 0 if req_idx == 0 else cu[req_idx - 1]``.
    """
    cu = [int(v) for v in cu_num_draft_tokens.detach().cpu().tolist()]
    return [(cu[i - 1] if i else 0, cu[i]) for i in range(len(cu))]


def _check_synthetic(SYNTHETIC_MODE: bool) -> None:
    """``rejection_sample_method="synthetic"`` is not implemented on LPU.

    Fail loudly rather than silently computing a different acceptance rule.
    """
    if SYNTHETIC_MODE:
        raise NotImplementedError(
            "speculative decoding with rejection_sample_method='synthetic' is "
            "not supported on LPU (no stand-in for the synthetic acceptance "
            "rule in ops_fallback/rejection_sampler.py)"
        )


def _check_shape(
    output_token_ids: torch.Tensor,
    ranges: list[tuple[int, int]],
    max_spec_len: int,
    draft_token_ids: torch.Tensor,
) -> None:
    """Guard the kernels' implicit preconditions.

    The kernels write at ``req_idx * (max_spec_len + 1) + pos`` for
    ``pos <= num_draft_tokens``, so ``num_draft_tokens <= max_spec_len`` and a
    ``[batch_size, max_spec_len + 1]`` output are preconditions upstream never
    states; torch slicing would silently truncate or broadcast instead.
    """
    if output_token_ids.shape[0] != len(ranges):
        raise ValueError(
            f"rejection sampler: output has {output_token_ids.shape[0]} rows "
            f"but {len(ranges)} requests"
        )
    if output_token_ids.shape[1] != max_spec_len + 1:
        raise ValueError(
            f"rejection sampler: output width {output_token_ids.shape[1]} != "
            f"max_spec_len + 1 ({max_spec_len + 1})"
        )
    for req, (start, end) in enumerate(ranges):
        if end - start > max_spec_len:
            raise ValueError(
                f"rejection sampler: request {req} has {end - start} draft "
                f"tokens, more than max_spec_len={max_spec_len}"
            )
    if ranges and ranges[-1][1] != draft_token_ids.numel():
        raise ValueError(
            f"rejection sampler: cu_num_draft_tokens[-1]={ranges[-1][1]} but "
            f"{draft_token_ids.numel()} draft tokens"
        )


def _lpu_expand_kernel(
    output_ptr,
    input_ptr,
    cu_num_tokens_ptr,
    replace_from,
    replace_to,
    MAX_NUM_TOKENS=None,
):
    """LPU stand-in for ``expand_kernel``.

    Per request: ``output[start:end] = where(input[req] == replace_from,
    replace_to, input[req])``.  ``MAX_NUM_TOKENS`` only sizes the Triton tile.
    """
    num_tokens = output_ptr.numel()
    if num_tokens == 0:
        return
    cu = [int(v) for v in cu_num_tokens_ptr.detach().cpu().tolist()]
    src = input_ptr.detach().cpu().tolist()
    if not cu or cu[-1] != num_tokens:
        # upstream leaves the tail of its new_empty() output untouched when
        # cu[-1] < num_tokens; that is garbage, so refuse instead of feeding it
        # into the sampler.
        raise ValueError(
            f"expand_batch_to_tokens: cu_num_tokens[-1]="
            f"{cu[-1] if cu else None} does not cover {num_tokens} tokens"
        )
    values: list = []
    for req, end in enumerate(cu):
        start = cu[req - 1] if req else 0
        val = src[req]
        if val == replace_from:
            val = replace_to
        values.extend([val] * (end - start))
    host = torch.tensor(values, dtype=output_ptr.dtype)
    output_ptr.copy_(host.to(output_ptr.device))


def _lpu_rejection_greedy_sample_kernel(
    output_token_ids_ptr,
    cu_num_draft_tokens_ptr,
    draft_token_ids_ptr,
    target_argmax_ptr,
    bonus_token_ids_ptr,
    is_greedy_ptr,
    max_spec_len,
    uniform_probs_ptr,
    synthetic_conditional_rates_ptr,
    SYNTHETIC_MODE=False,
):
    """LPU stand-in for ``rejection_greedy_sample_kernel``.

    Literal transcription, keeping the kernel's quirks:

    - a request whose ``is_greedy`` is false is skipped whole (its row keeps the
      ``PLACEHOLDER_TOKEN_ID`` fill);
    - the store sits inside the ``if not rejected`` block that *sets*
      ``rejected``, so the first mismatching position **is** written, with the
      target argmax;
    - the bonus token is appended only when nothing was rejected.
    """
    _check_synthetic(SYNTHETIC_MODE)
    batch, width = output_token_ids_ptr.shape
    if batch == 0:
        return
    ranges = _draft_ranges(cu_num_draft_tokens_ptr)
    _check_shape(output_token_ids_ptr, ranges, max_spec_len, draft_token_ids_ptr)

    draft = [int(v) for v in draft_token_ids_ptr.detach().cpu().tolist()]
    target = [int(v) for v in target_argmax_ptr.detach().cpu().tolist()]
    bonus = [int(v) for v in bonus_token_ids_ptr.detach().cpu().reshape(-1).tolist()]
    greedy = (
        None
        if is_greedy_ptr is None
        else [bool(v) for v in is_greedy_ptr.detach().cpu().tolist()]
    )

    out = output_token_ids_ptr.detach().cpu()
    for req, (start, end) in enumerate(ranges):
        if greedy is not None and not greedy[req]:
            continue
        num_draft = end - start
        rejected = False
        for pos in range(num_draft):
            if not rejected:
                token_id = target[start + pos]
                rejected = draft[start + pos] != token_id
                out[req, pos] = token_id
        if not rejected:
            out[req, num_draft] = bonus[req]
    output_token_ids_ptr.copy_(out.to(output_token_ids_ptr.device))


def _lpu_rejection_random_sample_kernel(
    output_token_ids_ptr,
    cu_num_draft_tokens_ptr,
    draft_token_ids_ptr,
    draft_probs_ptr,
    target_probs_ptr,
    bonus_token_ids_ptr,
    recovered_token_ids_ptr,
    uniform_probs_ptr,
    is_greedy_ptr,
    max_spec_len,
    vocab_size,
    synthetic_conditional_rates_ptr,
    NO_DRAFT_PROBS=False,
    SYNTHETIC_MODE=False,
):
    """LPU stand-in for ``rejection_random_sample_kernel``.

    Acceptance is ``draft_prob > 0 and target_prob / draft_prob >= uniform``
    (``draft_prob = 1`` under ``NO_DRAFT_PROBS``); accepted positions emit the
    draft token, the first rejected position emits the recovered token and
    stops the loop, and the bonus token is appended only when nothing was
    rejected.  ``is_greedy`` here is always a tensor (see ``rejection_sample``).
    """
    _check_synthetic(SYNTHETIC_MODE)
    batch, _ = output_token_ids_ptr.shape
    if batch == 0:
        return
    ranges = _draft_ranges(cu_num_draft_tokens_ptr)
    _check_shape(output_token_ids_ptr, ranges, max_spec_len, draft_token_ids_ptr)

    draft = [int(v) for v in draft_token_ids_ptr.detach().cpu().tolist()]
    recovered = [int(v) for v in recovered_token_ids_ptr.detach().cpu().tolist()]
    bonus = [int(v) for v in bonus_token_ids_ptr.detach().cpu().reshape(-1).tolist()]
    greedy = [bool(v) for v in is_greedy_ptr.detach().cpu().tolist()]
    uniform = [float(v) for v in uniform_probs_ptr.detach().cpu().tolist()]
    # One small round trip; the probability rows stay float32 on the host, so
    # the ratio below carries the kernel's precision.
    target_probs = target_probs_ptr.detach().cpu()
    draft_probs = None if NO_DRAFT_PROBS else draft_probs_ptr.detach().cpu()

    out = output_token_ids_ptr.detach().cpu()
    for req, (start, end) in enumerate(ranges):
        if greedy[req]:
            continue
        num_draft = end - start
        rejected = False
        for pos in range(num_draft):
            if not rejected:
                token_idx = start + pos
                draft_id = draft[token_idx]
                if NO_DRAFT_PROBS:
                    draft_prob = 1
                else:
                    draft_prob = draft_probs[token_idx, draft_id]
                target_prob = target_probs[token_idx, draft_id]
                # The kernel divides the two float32 probabilities and only then
                # widens the comparison to the float64 uniform sample; dividing
                # in float64 instead would round differently right at the
                # acceptance boundary.  A zero draft probability would divide by
                # zero, which is what the upstream guard is there for.  (An
                # out-of-vocab draft id raises here, where the kernel would read
                # past the row.)
                accepted = bool(draft_prob > 0) and (
                    float(target_prob / draft_prob) >= uniform[token_idx]
                )
                if accepted:
                    token_id = draft_id
                else:
                    rejected = True
                    token_id = recovered[token_idx]
                out[req, pos] = token_id
        if not rejected:
            out[req, num_draft] = bonus[req]
    output_token_ids_ptr.copy_(out.to(output_token_ids_ptr.device))


def _lpu_sample_recovered_tokens_kernel(
    output_token_ids_ptr,
    cu_num_draft_tokens_ptr,
    draft_token_ids_ptr,
    draft_probs_ptr,
    target_probs_ptr,
    inv_q_ptr,
    vocab_size,
    BLOCK_SIZE=None,
    NO_DRAFT_PROBS=False,
    USE_FP64_GUMBEL=False,
):
    """LPU stand-in for ``sample_recovered_tokens_kernel``.

    Per (request, position) with ``pos < num_draft_tokens``: build the adjusted
    distribution, scale by the request's ``inv_q`` and take the argmax.

    - ``NO_DRAFT_PROBS``: ``prob = target_probs[token]`` with the draft id entry
      zeroed (the kernel masks it out and loads 0.0 there);
    - otherwise: ``prob = max(target_prob - draft_prob, 0)``;
    - ties, and rows left all-zero, resolve to the lowest index: the kernel
      keeps the first tile holding the *strict* maximum and ``tl.max`` returns
      the lowest index of that tile.  Its ``min(id, vocab_size - 1)`` clamps the
      out-of-vocabulary padding of the final tile, which the host version has no
      equivalent of;
    - positions at or past a request's draft count are left at 0 here, where the
      kernel leaves its ``empty_like`` output untouched; nothing reads them.

    ``BLOCK_SIZE`` only sizes the Triton tile.  ``USE_FP64_GUMBEL`` is refused:
    LPU has no float64 compute, so the gumbel-noise variant cannot be emulated
    faithfully.
    """
    if USE_FP64_GUMBEL:
        raise NotImplementedError(
            "rejection sampler: use_fp64_gumbel=True is not supported on LPU "
            "(no float64 compute)"
        )
    batch = int(cu_num_draft_tokens_ptr.shape[0])
    if batch == 0:
        return
    ranges = _draft_ranges(cu_num_draft_tokens_ptr)
    if ranges[-1][1] != draft_token_ids_ptr.numel():
        raise ValueError(
            f"sample_recovered_tokens: cu_num_draft_tokens[-1]={ranges[-1][1]} "
            f"but {draft_token_ids_ptr.numel()} draft tokens"
        )

    draft = [int(v) for v in draft_token_ids_ptr.detach().cpu().tolist()]
    target_probs = target_probs_ptr.detach().cpu()
    draft_probs = None if NO_DRAFT_PROBS else draft_probs_ptr.detach().cpu()
    inv_q = inv_q_ptr.detach().cpu()
    if target_probs.shape[-1] != vocab_size:
        raise ValueError(
            f"sample_recovered_tokens: target_probs width "
            f"{target_probs.shape[-1]} != vocab_size {vocab_size}"
        )

    out = [0] * len(draft)
    for req, (start, end) in enumerate(ranges):
        for pos in range(end - start):
            token_idx = start + pos
            if NO_DRAFT_PROBS:
                score = target_probs[token_idx].clone()
                draft_id = draft[token_idx]
                if 0 <= draft_id < vocab_size:
                    score[draft_id] = 0.0
            else:
                score = torch.clamp(
                    target_probs[token_idx] - draft_probs[token_idx], min=0.0
                )
            score = score * inv_q[req]
            out[token_idx] = int(torch.argmax(score))
    host = torch.tensor(out, dtype=output_token_ids_ptr.dtype)
    output_token_ids_ptr.copy_(host.to(output_token_ids_ptr.device))


def _host_generator(generator):
    """A host generator standing in for ``generator``, or None if unusable.

    ``Tensor.uniform_(generator=...)`` requires the generator and the tensor on
    the same device, and on LPU the per-request generators are built for the
    accelerator (``gpu_model_runner`` uses ``device=self.device``).  A request
    that carries a seed must stay reproducible, so its seed is reused on the
    host rather than dropping the generator.
    """
    device = getattr(generator, "device", None)
    if getattr(device, "type", None) == "cpu":
        return generator
    try:
        return torch.Generator().manual_seed(int(generator.initial_seed()))
    except Exception:  # noqa: BLE001 - LPU generator class is not introspectable
        return None


def _lpu_generate_uniform_probs(
    num_tokens: int,
    num_draft_tokens: list[int],
    generators: dict,
    device,
) -> torch.Tensor:
    """LPU stand-in for ``generate_uniform_probs``: the same draws, on the host.

    Upstream allocates a float64 tensor on ``device``; float64 device storage is
    not something LPU is known to provide, and a raise here would land in the
    middle of ``profile_run``.  The dtype and the per-request generator handling
    are kept as upstream has them (float64 on purpose: float32 can draw an exact
    0.0), only the storage is host-side -- safe because the value is consumed
    exclusively by the kernels this patch also rebinds, which read it on the
    host.  ``device`` is accepted and ignored.
    """
    uniform_probs = torch.rand((num_tokens,), dtype=torch.float64)
    start_idx = 0
    for req_idx, n in enumerate(num_draft_tokens):
        # Do not generate random numbers for requests with no draft tokens
        # (upstream skips these for reproducibility).  Prefill requests carry
        # num_draft_tokens == 0.
        if n == 0:
            continue
        end_idx = start_idx + n
        generator = generators.get(req_idx)
        if generator is not None:
            host_generator = _host_generator(generator)
            if host_generator is None:
                logger.warning_once(
                    "rejection sampler: a per-request generator cannot be "
                    "reproduced on the host, its uniform probabilities fall "
                    "back to the global generator"
                )
            uniform_probs[start_idx:end_idx].uniform_(generator=host_generator)
        start_idx = end_idx
    return uniform_probs


def _sweep_leaked_aliases(originals: dict) -> None:
    """Rebind any remaining alias of the four upstream kernels, then report.

    The kernels are reached through the module namespace of
    ``vllm.v1.sample.rejection_sampler``, so rebinding them *there* is
    sufficient today.  But ``from ... import X`` copies a name into the
    importing module's namespace, and one such copy anywhere would silently
    keep launching the upstream Triton kernel -- the failure mode this patch
    exists to remove would come back as "sometimes it still crashes".  The
    sweep runs at patch-install time and flags whatever it finds.

    Identity is compared against the objects captured before the rebind, so
    re-entry cannot mistake our own stand-ins for upstream ones.
    """
    import sys as _sys

    leaked: dict[str, list[str]] = {}
    for name, (original, replacement) in originals.items():
        holders = []
        for mod_name, mod in list(_sys.modules.items()):
            if mod is None or mod_name.startswith(("torch", "triton")):
                continue
            if getattr(mod, name, None) is not original:
                continue
            holders.append(mod_name)
            try:
                setattr(mod, name, replacement)
            except Exception:  # noqa: BLE001 - some modules are not writable
                pass
        if holders:
            leaked[name] = holders
    if leaked:
        logger.error(
            "rejection sampler stand-ins: upstream kernel aliases still bound "
            "in other module namespaces, rebound there too: %s",
            leaked,
        )
        print(f"[rejection-sampler] rebound leaked aliases: {leaked}")
        if os.environ.get(_STRICT_REBIND_ENV) == "1":
            raise RuntimeError(
                f"upstream rejection sampler kernel aliases leaked into {leaked}"
            )


_STANDINS = {
    "expand_kernel": _lpu_expand_kernel,
    "rejection_greedy_sample_kernel": _lpu_rejection_greedy_sample_kernel,
    "rejection_random_sample_kernel": _lpu_rejection_random_sample_kernel,
    "sample_recovered_tokens_kernel": _lpu_sample_recovered_tokens_kernel,
    "generate_uniform_probs": _lpu_generate_uniform_probs,
}


def apply_lpu_rejection_sampler_patch() -> None:
    """Rebind the spec-decode rejection sampler's device code to torch.

    Applied unconditionally on LPU: the rebindings are inert unless
    speculative decoding is enabled.

    Two names beyond the four kernels:

    - ``generate_uniform_probs`` allocates a float64 device tensor (see the
      module docstring);
    - ``apply_top_k_top_p`` is *not* defined in this module: ``rejection_sampler.py``
      imports it at module scope from ``vllm.v1.sample.ops.topk_topp_sampler``,
      so the rebind that ``ops_fallback/sampler.py`` makes in *that* module does
      not reach this copy, and its dispatcher would pick the Triton branch
      (``HAS_TRITON`` is true on LPU) for the >= 8-row logits of a prefill batch.
      Bound straight to the upstream pure-PyTorch implementation, so this patch
      does not depend on the other one having run first.
    """
    global _PATCHED

    from vllm_fl.utils import DeviceInfo

    if DeviceInfo().device_type != "lpu":
        return
    if _PATCHED:
        return

    import vllm.v1.sample.ops.topk_topp_sampler as _topk_topp
    import vllm.v1.sample.rejection_sampler as _rejection_sampler

    replacements = {
        # launch sites take the kernel[(grid,)] form; the plain helpers are
        # called directly
        name: _TorchKernelLauncher(_STANDINS[name]) for name in _KERNEL_NAMES
    }
    replacements.update({name: _STANDINS[name] for name in _PLAIN_NAMES})
    originals = {
        name: (getattr(_rejection_sampler, name), replacement)
        for name, replacement in replacements.items()
    }
    _rejection_sampler.apply_top_k_top_p = _topk_topp.apply_top_k_top_p_pytorch
    for name, replacement in replacements.items():
        setattr(_rejection_sampler, name, replacement)

    _PATCHED = True
    _sweep_leaked_aliases(originals)
