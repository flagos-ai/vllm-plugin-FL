import sys, types
try:
    import vllm.platforms as _p
    _cp = getattr(_p, "current_platform", None)
    if _cp is not None and not getattr(_cp, "device_type", ""):
        _cls = _cp.__class__
        try:
            _cls.device_type = property(lambda self: "npu")
        except Exception:
            pass
except Exception:
    pass
if "numba" not in sys.modules:
    _m = types.ModuleType("numba")
    _m.get_num_threads = lambda: 1
    _m.set_num_threads = lambda n: None
    _m.jit = lambda *a, **k: (lambda f: f)
    _m.njit = lambda *a, **k: (lambda f: f)
    _m.prange = range
    _m.__version__ = "0.0.0-stub"
    sys.modules["numba"] = _m
