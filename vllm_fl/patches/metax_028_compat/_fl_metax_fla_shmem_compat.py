"""FL/MetaX C550 shim: cap the FLA chunk_delta_h prefill autotuner to the
64 KiB Triton shared-memory budget of the C550 (Maca) backend.

Why
---
vLLM 0.28 moved the FLA kernels from vllm.model_executor.layers.fla to
vllm.third_party.flash_linear_attention.  The vendor copy under vllm_metax
(vllm_metax/patch/triton_support/chunk_delta_h.py, num_stages=[1]) still
patches the OLD path and dies on import, taking the whole vllm_metax.patch
package with it.  FL's own equivalent
(vllm_fl/dispatch/backends/vendor/metax/patches/chunk_delta_h.py) targets the
correct 0.28 module, but its only trigger is
PlatformFL.import_kernels() -- and under VLLM_PLUGINS=metax FL deliberately
yields the platform slot to vllm_metax, so that hook never runs and the patch
was never applied.

Unpatched, the native autotuner keeps num_stages in {2,3,4} and the first
Qwen-GDN prefill aborts with

    triton.runtime.errors.OutOfResources: Required: 69696, Hardware limit: 65536

This shim arms a sys.meta_path finder for the real 0.28 module and, the first
time it is imported (lazily, long after vllm itself is up, so import order is
untouched), loads FL's patch file directly FROM DISK -- bypassing the
vllm_fl...patches package __init__ and its side effects -- and installs the
num_stages=[1] kernel.

Nothing on disk is modified.  Kill switch: FL_METAX_FLA_SHMEM_COMPAT=0
"""

import importlib.util
import os
import sys

_KERNEL = "chunk_gated_delta_rule_fwd_kernel_h_blockdim64"
_TARGET = "vllm.third_party.flash_linear_attention.ops.chunk_delta_h"
_HERE = os.path.dirname(os.path.abspath(__file__))
_PATCH_FILE = os.path.join(
    os.path.dirname(os.path.dirname(_HERE)),
    "dispatch", "backends", "vendor", "metax", "patches", "chunk_delta_h.py",
)
_STATE = {"skipped": 0, "matched": 0, "patched": 0, "detail": "init"}


def _enabled():
    v = os.environ.get("FL_METAX_FLA_SHMEM_COMPAT", "1")
    return v.strip().lower() not in ("", "0", "false", "no", "off")


def _applies():
    from ._fl_metax_gate import metax_active

    return metax_active()


def _publish_attr_chain(module):
    """Publish every ancestor link of the import chain so the patch file's
    trailing `import a.b.c.d` + attribute access resolves even while the
    parent packages are still mid-exec.  Each level must carry its OWN
    module object (sys.modules[child]) -- never the leaf, or the whole
    chain collapses onto the leaf and re-imports raise
    'partially initialized module ... has no attribute ...'."""
    name = getattr(module, "__name__", "")
    parent = name.rpartition(".")[0]
    while "." in parent:
        target = sys.modules.get(name)
        pm = sys.modules.get(parent)
        if target is not None and pm is not None:
            child = name.rpartition(".")[2]
            if getattr(pm, child, None) is not target:
                try:
                    setattr(pm, child, target)
                except Exception:
                    pass
        name, parent = parent, parent.rpartition(".")[0]


def _install_kernel(module):
    spec = importlib.util.spec_from_file_location(
        "_fl_metax_chunk_delta_h_shmem_patch", _PATCH_FILE
    )
    patched = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = patched
    try:
        spec.loader.exec_module(patched)
    except Exception as exc:
        _STATE["detail"] = "patch file exec failed: %r" % (exc,)
        return
    kernel = getattr(patched, _KERNEL, None)
    if kernel is None:
        _STATE["detail"] = "kernel missing from patch file"
        return
    try:
        setattr(module, _KERNEL, kernel)
    except Exception as exc:
        _STATE["detail"] = "setattr failed: %r" % (exc,)
        return
    stages = None
    configs = getattr(kernel, "configs", None)
    if configs is not None:
        try:
            stages = sorted({c.num_stages for c in configs})
        except Exception:
            stages = None
    _STATE["patched"] += 1
    _STATE["detail"] = "%s num_stages=%s" % (_KERNEL, stages)


def _apply(module):
    try:
        _publish_attr_chain(module)
        _install_kernel(module)
    except Exception as exc:
        _STATE["detail"] = "apply failed: %r" % (exc,)


class _Loader:
    def __init__(self, inner):
        self._inner = inner

    def create_module(self, spec):
        if hasattr(self._inner, "create_module"):
            return self._inner.create_module(spec)
        return None

    def exec_module(self, module):
        self._inner.exec_module(module)
        _apply(module)

    def __getattr__(self, name):
        return getattr(self._inner, name)


class _Finder:
    def find_spec(self, fullname, path=None, target=None):
        if fullname != _TARGET:
            return None
        if not _enabled() or not _applies():
            _STATE["skipped"] += 1
            return None
        _STATE["matched"] += 1
        for finder in list(sys.meta_path):
            if finder is self:
                continue
            find = getattr(finder, "find_spec", None)
            if find is None:
                continue
            try:
                spec = find(fullname, path, target)
            except Exception:
                continue
            if spec is None or spec.loader is None:
                continue
            if not hasattr(spec.loader, "exec_module"):
                continue
            spec.loader = _Loader(spec.loader)
            return spec
        return None


def install():
    if not _enabled() or not _applies():
        _STATE["detail"] = "skipped: metax backend not active"
        return False
    if not any(isinstance(f, _Finder) for f in sys.meta_path):
        sys.meta_path.insert(0, _Finder())
    _STATE["detail"] = "armed for %s" % _TARGET
    return True


def status():
    return dict(_STATE)


install()
