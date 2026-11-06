# Copyright (c) 2026 BAAI. All rights reserved.

"""GCU workarounds in the shared ``ModelRunnerFL``.

``vllm_fl/worker/model_runner.py`` is the plugin's cross-vendor adaptation of
vLLM's ``gpu_model_runner.py`` -- the same ``ModelRunnerFL`` runs on GCU, MUSA,
Ascend and Kunlunxin -- so GCU-only workarounds do not belong in that file.
They are applied here instead, by rewriting individual statements in the shared
runner's source at import time (keeping the shared file untouched on disk).

Two statements are rewritten:

1. ``ModelRunnerFL._prepare_inputs`` -- the int64 positions add::

       self.positions[:total_num_scheduled_tokens] = (
           self.num_computed_tokens[req_indices_gpu].to(torch.int64)
           + self.query_pos.gpu[:total_num_scheduled_tokens]
       )

   torch_gcu native ATen has no int64 ``add`` kernel for this pattern: the call
   aborts inside ``topsatenAdd`` ("This binary op pattern not support!",
   ``TOPSATEN_STATUS_NOT_SUPPORT``) and takes the worker down through
   ``torch_gcu/csrc/gcu/gcu_utils.cpp:74`` -- a hard abort, not a catchable
   exception.  Both operands are token positions bounded by ``max_model_len``
   (far below 2**31), so doing the add in int32 and widening on store is
   lossless.

2. ``ModelRunnerFL.__init__`` -- the M-RoPE positions buffer dtype::

       self.mrope_positions = self._make_buffer(
           (3, self.max_num_tokens + 1), dtype=torch.int64
       )

   M-RoPE positions feed q/k-norm+RoPE fused kernels whose i64 indexing
   overflows GCU300 LLVM11 register allocation under graph mode (inductor
   cannot compile the kernel).  int32 is lossless here
   (``max_position_embeddings`` << 2**31).  The shared file keeps the upstream
   int64 default; on GCU we rewrite the buffer allocation to int32 so no env
   var is required.  Other vendors keep the shared file's int64 default.

Statements are rewritten as source text rather than by shipping copies of the
whole methods, which would silently rot the next time the shared runner
changes.  If a statement stops matching, we log a warning and leave the
original code in place.
"""

from __future__ import annotations

import inspect
import logging
import re
import textwrap

logger = logging.getLogger(__name__)

# Whitespace-tolerant so a pure reformat of the shared file does not silently
# disable the fix, but anchored on the whole statement so it cannot match some
# other int64 add.
_STATEMENT = re.compile(
    r"self\.positions\[:total_num_scheduled_tokens\]\s*=\s*\(\s*"
    r"self\.num_computed_tokens\[req_indices_gpu\]\.to\(torch\.int64\)\s*"
    r"\+\s*self\.query_pos\.gpu\[:total_num_scheduled_tokens\]\s*\)"
)

_REPLACEMENT = (
    "self.positions[:total_num_scheduled_tokens] = (\n"
    "            self.num_computed_tokens[req_indices_gpu]\n"
    "            + self.query_pos.gpu[:total_num_scheduled_tokens].to(torch.int32)\n"
    "        ).to(self.positions.dtype)"
)

# The M-RoPE positions buffer allocation in ``__init__``.  The shared file keeps
# the upstream int64 default; here we narrow it to int32 for GCU.  Anchored on
# the whole ``self.mrope_positions = self._make_buffer(... dtype=torch.int64)``
# call and tolerant of reformatting so a reflow of the shared file does not
# silently disable the fix.
_MROPE_STMT = re.compile(
    r"self\.mrope_positions\s*=\s*self\._make_buffer\(\s*"
    r"\(\s*3\s*,\s*self\.max_num_tokens\s*\+\s*1\s*\)\s*,\s*"
    r"dtype\s*=\s*torch\.int64\s*,?\s*\)"
)

_MROPE_REPL = (
    "self.mrope_positions = self._make_buffer(\n"
    "                (3, self.max_num_tokens + 1), dtype=torch.int32\n"
    "            )  # GCU: int32 M-RoPE positions (lossless, graph-compile safe)"
)


def _rewrite_method(cls, method_name, pattern, replacement, flag, what):
    """Source-rewrite one method of ``cls`` in place. Idempotent via ``flag``.

    Reads the live method source, applies ``pattern.subn(replacement, ...)``
    (which must match exactly once), recompiles it against the method's own
    globals, and rebinds it onto ``cls``.  On any miss/failure the original
    method is left untouched and a warning is logged -- GCU-only fixes must
    never crash the shared runner on other vendors or future refactors.
    """
    fn = getattr(cls, method_name, None)
    if fn is None:
        logger.warning("runner_compat: %s.%s missing; %s left in place",
                        cls.__name__, method_name, what)
        return
    if getattr(fn, flag, False):
        return

    try:
        source = textwrap.dedent(inspect.getsource(fn))
    except Exception as exc:
        logger.warning("runner_compat: cannot read %s source (%s)", method_name, exc)
        return

    rewritten, matches = pattern.subn(replacement, source)
    if matches != 1:
        logger.warning(
            "runner_compat: %s not found in %s.%s (%d matches); leaving it "
            "untouched.",
            what, cls.__name__, method_name, matches,
        )
        return

    # ``getsource`` returns the method starting at its own line 1; pad the
    # source back to the original offset so tracebacks (and py-spy) keep
    # pointing at the real lines of the shared file.
    padded = "\n" * (fn.__code__.co_firstlineno - 1) + rewritten

    try:
        namespace: dict = {}
        code = compile(
            padded, inspect.getsourcefile(fn) or "<model_runner>", "exec"
        )
        exec(code, fn.__globals__, namespace)
        patched = namespace[method_name]
    except Exception as exc:
        logger.warning("runner_compat: rewriting %s failed: %s", method_name, exc)
        return

    patched.__qualname__ = fn.__qualname__
    patched.__doc__ = fn.__doc__
    setattr(patched, flag, True)
    setattr(cls, method_name, patched)
    logger.info("runner_compat: rewrote %s in %s.%s for GCU",
                what, cls.__name__, method_name)


def apply_runner_compat_patches() -> None:
    """Apply the GCU source rewrites to ``ModelRunnerFL``. Idempotent."""
    try:
        from vllm_fl.worker.model_runner import ModelRunnerFL
    except Exception as exc:
        logger.warning(
            "runner_compat: ModelRunnerFL not importable (%s); GCU runner "
            "rewrites are left in place",
            exc,
        )
        return

    # int64 positions add -> int32 (GCU has no int64 add kernel; hard abort).
    _rewrite_method(
        ModelRunnerFL, "_prepare_inputs", _STATEMENT, _REPLACEMENT,
        "_gcu_i32_positions", "the int64 positions add",
    )
    # M-RoPE positions buffer -> int32 by default (graph-compile safe on GCU),
    # dropping the GCU_MROPE_INT32 env requirement.
    _rewrite_method(
        ModelRunnerFL, "__init__", _MROPE_STMT, _MROPE_REPL,
        "_gcu_i32_mrope", "the M-RoPE int32 dtype default",
    )
