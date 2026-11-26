"""First-call JIT audit: log every Triton compilation that takes noticeable time.

A kernel compiled on the serving path stalls the whole engine for the compile time (E021: 3-17 s per new input shape of
FlagGems' `index`). This hook makes such events visible in the serve log: one WARNING line per compilation longer than
`threshold` seconds, plus a running total. It changes no behaviour; hooks installed earlier are kept and still called.
"""
import logging
import threading
import time

logger = logging.getLogger(__name__)

_state = {"t0": {}, "count": 0, "total": 0.0}
_installed = False


def _kernel_name(kw, args):
    fn = kw.get("fn")
    if fn is None and len(args) > 2:
        fn = args[2]
    for attr in ("name", "__name__"):
        name = getattr(fn, attr, None)
        if name:
            return str(name)
    inner = getattr(fn, "fn", None)
    return getattr(inner, "__name__", "?")


def install(threshold: float = 0.5) -> None:
    global _installed
    if _installed:
        return
    try:
        from triton import knobs
    except Exception:
        return
    prev_pre = knobs.runtime.jit_cache_hook
    prev_post = knobs.runtime.jit_post_compile_hook

    def pre(*args, **kw):
        _state["t0"][threading.get_ident()] = time.perf_counter()
        return prev_pre(*args, **kw) if prev_pre else None

    def post(*args, **kw):
        t0 = _state["t0"].pop(threading.get_ident(), None)
        if t0 is not None:
            dt = time.perf_counter() - t0
            _state["count"] += 1
            _state["total"] += dt
            if dt >= threshold:
                logger.warning("BI-V150 JIT: compiled %s in %.2f s (compilations so far: %d, %.1f s)",
                               _kernel_name(kw, args), dt, _state["count"], _state["total"])
        return prev_post(*args, **kw) if prev_post else None

    knobs.runtime.jit_cache_hook = pre
    knobs.runtime.jit_post_compile_hook = post
    _installed = True
    logger.warning("BI-V150 JIT audit installed (log compilations >= %.1f s)", threshold)
