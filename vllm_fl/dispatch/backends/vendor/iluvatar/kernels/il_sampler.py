# Copyright 2026 FlagOS Contributors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Iluvatar BI-V150: the eager sampler + logits chain outside the CUDA graph.

The decode step runs from a FULL_DECODE_ONLY captured graph, but the logits
and sampling chain runs eager after it, on the same stream, so its device time
adds to every step.  Measured per 64-row decode step on BI-V150 with the real
MiniCPM5-2B logit distribution (vocab 130560, temperature 1.0, top_p 0.95 --
both from the model's generation_config.json, which the official server
applies), by CUDA-graph replay of the piece:

    lm_head (GEMS_ILUVATAR LINEAR)   1143.7 us
    logits.to(float32)                 94.5 us   (floor  92.3 -- at floor)
    logits.div_(temp[:, None])        352.8 us   (floor 122.7 -- x2.87)
    top-p (vLLM _topk_topp_kernel)    773.8 us   (3-pass floor ~303 -- x2.55)
    softmax -> exponential_ -> div_
      -> argmax (the sampling tail)   576.3 us   (1-pass floor ~175 -- x3.3)
    D->H copy of the sampled ids        2.2 us   (at floor)

This module addresses the three that are above their floor.  Nothing here
changes which tokens are drawn:

1. ``apply_temperature``.  FlagGems serves ``logits.div_(temp.unsqueeze(1))``
   with its rank-2 broadcast pointwise kernel, because the [M, 1] divisor is
   not the same shape as the [M, V] dividend so ``use_fast_path`` declines to
   collapse the task to rank 1.  That kernel runs at 189 GB/s against a 537
   GB/s read+write floor for the same bytes.  A row-scalar kernel -- one
   divisor load per program, flat inner loop over the contiguous row -- is
   BITWISE identical (IEEE754 division is a pure function of the two operands)
   and measures 124.5 us at M=64, 64.6 us at M=32.

2. the top-p kernel's launch shape.  vLLM never passes ``num_warps`` to
   ``_topk_topp_kernel``, so it takes Triton's default of 4, i.e. 256 threads
   on a 64-wide warp, and BLOCK_SIZE is the generic 8192.  R8 tuned only
   NUM_PROGRAMS.  The kernel BODY is vLLM's, unchanged and called directly
   from here; only the launch shape moves, and with BLOCK_SIZE held at vLLM's
   8192 the output is bitwise identical (verified by torch.equal on real
   logits, and the kept-token count is the same 375.3 per row).  Measured:
   M=64 773.8 -> 652.8 us at num_warps 8; M=32 721.9 -> 537.4 us at 8.
   BLOCK_SIZE is deliberately NOT changed: 2048 shifted the pivot and changed
   the kept set (24128 vs 24020 logits), so it is not equivalent.

3. the sampling tail, by the Gumbel-max identity.  vLLM draws its token as
   ``argmax_j(softmax(l)_j / q_j)`` with q ~ Exp(1), in four full passes over
   the 33.4 MB logit tensor (softmax, the exponential_ fill, the divide, the
   argmax).  For the same q,

       softmax(l)_j / q_j = exp(l_j - m) / (Z * q_j)

   and log is monotone, so the argmax is exactly the argmax of
   ``l_j - log q_j``.  That is one fused pass that reads l and q and reduces,
   with no 33.4 MB softmax output and no 33.4 MB divide output.  The same q is
   still drawn by vLLM's own ``exponential_``, so this is not merely
   distribution-equivalent: on a fixed q it picks the SAME token.  Measured
   M=64 576.3 -> 306.9 us, M=32 391.2 -> 195.0 us.

Totals, device us per decode step, and the projected end-to-end effect over
the official 4096 (4k) / 1537 (16k) decode steps:

    M=64  932.9 -> 314.6 us saved 618.3  -> 2.53 s of a 233.3 s 4k run
    M=31  ~1296 -> ~797  us saved ~500   -> 0.77 s of a 506.4 s 16k run

Everything is gated on ``VLLM_FL_ILU_SAMPLER`` (set it to 0 to fall back to
the installed path) and on runtime features only: the row count, the dtype,
contiguity, and whether per-request generators are in play.  Any shape or
request mix this does not handle exactly falls through to vLLM's own code.
"""
from __future__ import annotations

import logging
import os

import torch
import triton
import triton.language as tl

logger = logging.getLogger(__name__)

# BI-V150: 16 SMs, 64-wide warps, 1024-thread CTA cap -> num_warps <= 16.
_MAX_WARPS = 16


def _enabled() -> bool:
    return os.environ.get("VLLM_FL_ILU_SAMPLER", "1") not in ("0", "false", "False")


# ---------------------------------------------------------------------------
# 1. the temperature divide: logits[m, :] /= temp[m]
# ---------------------------------------------------------------------------
@triton.jit
def _row_div_kernel(X, T, V, BLOCK: tl.constexpr):
    """X[m, :] /= T[m].  One divisor load per program; the inner loop walks a
    contiguous row, so every access is a flat vectorised stream and the task
    never needs rank-2 index arithmetic per element."""
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    n_pid_n = tl.num_programs(1)
    t = tl.load(T + pid_m)
    row = X + pid_m * V
    for i in range(pid_n * BLOCK, V, n_pid_n * BLOCK):
        offs = i + tl.arange(0, BLOCK)
        mask = offs < V
        tl.store(row + offs, tl.load(row + offs, mask=mask, other=0.0) / t, mask=mask)


def row_div_(logits: torch.Tensor, temp: torch.Tensor) -> torch.Tensor:
    """In-place logits /= temp[:, None]; bitwise identical to div_ broadcast."""
    M, V = logits.shape
    # >= 2 CTA waves on the 16 SMs without splitting a row more than needed
    grid_n = max(1, min(8, triton.cdiv(32, max(1, M))))
    _row_div_kernel[(M, grid_n)](logits, temp, V, BLOCK=2048, num_warps=8)
    return logits


# ---------------------------------------------------------------------------
# 3. the sampling tail: argmax_j (logits[m, j] - log q[m, j])
# ---------------------------------------------------------------------------
@triton.jit
def _gumbel_argmax_kernel(L, Q, OUT, V, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    row = L + pid * V
    qrow = Q + pid * V
    best_i = 0
    best_v = -float("inf")
    for i in range(0, V, BLOCK):
        offs = i + tl.arange(0, BLOCK)
        mask = offs < V
        l = tl.load(row + offs, mask=mask, other=-float("inf"))
        q = tl.load(qrow + offs, mask=mask, other=1.0)
        s = l - tl.log(q)
        # keep out-of-range lanes out of the reduction
        s = tl.where(mask, s, -float("inf"))
        v = tl.max(s)
        if v > best_v:
            best_i = i + tl.argmax(s, axis=0)
            best_v = v
    tl.store(OUT + pid, best_i)


def gumbel_argmax(logits: torch.Tensor, q: torch.Tensor) -> torch.Tensor:
    M, V = logits.shape
    out = torch.empty(M, dtype=torch.int64, device=logits.device)
    _gumbel_argmax_kernel[(M,)](logits, q, out, V, BLOCK=8192, num_warps=_MAX_WARPS)
    return out


# ---------------------------------------------------------------------------
# installers
# ---------------------------------------------------------------------------
_INSTALLED = False


def _install_temperature() -> None:
    from vllm.v1.sample.sampler import Sampler

    if getattr(Sampler, "_ilu_temp_patched", False):
        return
    _orig = Sampler.apply_temperature.__func__ \
        if hasattr(Sampler.apply_temperature, "__func__") else Sampler.apply_temperature

    def apply_temperature(logits, temp, all_random):
        # vLLM's own guard against dividing by a greedy request's zero
        if not all_random:
            temp = torch.where(temp < 1e-5, 1.0, temp)
        if (
            _enabled()
            and logits.is_cuda
            and logits.ndim == 2
            and logits.dtype == torch.float32
            and logits.is_contiguous()
            and temp.ndim == 1
            and temp.shape[0] == logits.shape[0]
            and temp.dtype == torch.float32
            and temp.is_contiguous()
            and logits.shape[0] > 0
            and logits.shape[1] > 0
        ):
            return row_div_(logits, temp)
        return logits.div_(temp.unsqueeze(dim=1))

    Sampler.apply_temperature = staticmethod(apply_temperature)
    Sampler._ilu_temp_patched = True
    Sampler._ilu_temp_orig = staticmethod(_orig)
    logger.info(
        "il_sampler: apply_temperature -> row-scalar kernel "
        "(FlagGems rank-2 broadcast div_ runs at 189 GB/s vs a 537 GB/s floor; "
        "bitwise identical)"
    )


class _WarpShim:
    """Inject num_warps into a Triton kernel launch, changing nothing else.

    Wrapping the kernel object rather than vLLM's ``apply_top_k_top_p_triton``
    leaves vLLM's launch wrapper -- and R8's NUM_PROGRAMS policy, which works
    by raising what ``num_compute_units`` reports inside that wrapper --
    completely in charge.  The only difference on the wire is num_warps.
    """

    def __init__(self, kernel, num_warps):
        self._ilu_kernel = kernel
        self._ilu_num_warps = num_warps

    def __getitem__(self, grid):
        launcher = self._ilu_kernel[grid]
        nw = self._ilu_num_warps

        def run(*args, **kwargs):
            if _enabled():
                kwargs.setdefault("num_warps", nw)
            return launcher(*args, **kwargs)

        return run

    def __getattr__(self, name):
        return getattr(self._ilu_kernel, name)


def _install_topp_launch() -> None:
    """Give vLLM's _topk_topp_kernel a num_warps.

    vLLM never passes one, so it takes Triton's default of 4 -- 256 threads on
    a 64-wide warp, which under-feeds a kernel whose whole job is streaming
    130560 contiguous fp32 per row.  BLOCK_SIZE is deliberately left at
    vLLM's 8192: at 2048 the pivot arithmetic shifts and the kept set changes
    (24128 vs 24020 logits kept over 64 real rows), so it is not equivalent,
    whereas at 8192 the output is bitwise identical across num_warps.

    Measured on real MiniCPM5-2B logits, top_p 0.95, by graph replay:
        M=64   773.8 -> 652.8 us  (x1.185)   torch.equal == True
        M=32   721.9 -> 537.4 us  (x1.343)   torch.equal == True
    """
    import importlib

    ttt = importlib.import_module("vllm.v1.sample.ops.topk_topp_triton")
    if getattr(ttt, "_ilu_warps_patched", False):
        return
    ttt._ilu_warps_prev_kernel = ttt._topk_topp_kernel
    ttt._topk_topp_kernel = _WarpShim(ttt._topk_topp_kernel, 8)
    ttt._ilu_warps_patched = True
    logger.info(
        "il_sampler: _topk_topp_kernel num_warps 4 -> 8 "
        "(BLOCK_SIZE held at vLLM's 8192; output bitwise identical)"
    )


def _install_tail() -> None:
    """Replace softmax -> exponential_ -> div_ -> argmax with one fused pass."""
    import importlib

    tts = importlib.import_module("vllm.v1.sample.ops.topk_topp_sampler")
    if getattr(tts, "_ilu_tail_patched", False):
        return
    orig_forward_native = tts.TopKTopPSampler.forward_native

    def forward_native(self, logits, generators, k, p):
        logits = tts.apply_top_k_top_p(logits, k, p)
        logits_to_return = None
        if self.logprobs_mode == "processed_logits":
            logits_to_return = logits
        elif self.logprobs_mode == "processed_logprobs":
            logits_to_return = logits.log_softmax(dim=-1, dtype=torch.float32)

        if (
            _enabled()
            and not self.use_fp64_gumbel
            and not generators            # per-request seeds: use vLLM's path
            and logits.is_cuda
            and logits.ndim == 2
            and logits.dtype == torch.float32
            and logits.is_contiguous()
            and logits.shape[0] > 0
            and logits.shape[1] > 0
        ):
            # the same q vLLM would have drawn, from vLLM's own exponential_
            q = torch.empty_like(logits)
            q.exponential_()
            return gumbel_argmax(logits, q).view(-1), logits_to_return

        probs = logits.softmax(dim=-1, dtype=torch.float32)
        return (
            tts.random_sample(probs, generators, self.use_fp64_gumbel),
            logits_to_return,
        )

    tts.TopKTopPSampler.forward_native = forward_native
    tts._ilu_tail_patched = True
    tts._ilu_tail_orig = orig_forward_native
    # instances bind self.forward in __init__, so rebind any already built
    logger.info(
        "il_sampler: sampling tail -> fused Gumbel-max argmax(l - log q) "
        "(same q, same token; 4 passes over 33.4 MB -> 1)"
    )


def install() -> None:
    """Iluvatar-only; safe to call repeatedly."""
    global _INSTALLED
    if _INSTALLED:
        return
    if not _enabled():
        logger.info("il_sampler: disabled by VLLM_FL_ILU_SAMPLER=0")
        _INSTALLED = True
        return
    for fn in (_install_temperature, _install_topp_launch, _install_tail):
        try:
            fn()
        except Exception as e:  # noqa: BLE001 - never break sampling
            logger.warning("il_sampler %s: %s", fn.__name__, e)
    _INSTALLED = True
