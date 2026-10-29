"""MetaX (MACA) compatibility shims for the vLLM 0.28 empty wheel.

These modules patch ``torch`` / ``triton`` / ``vllm`` in memory before the
affected kernels are first compiled. They used to be installed as ``.pth``
files in ``site-packages``, which left hooks behind after uninstalling the
wheel and required an out-of-band installer.

They are now part of the plugin's normal lifecycle: :func:`install` is called
from the FL plugin entry point (``vllm_fl.register``) before the platform slot
is claimed or yielded, so a plain ``pip install`` is enough. Each module
applies its patch as an import side effect and self-gates on
:func:`_fl_metax_gate.metax_active`.

Kill switches (``FL_METAX_*_PATCH=0``) are documented in the individual
modules. ``FL_BOOT_SHIMS_PATCH=0`` disables the whole set.
"""

import os

from ._fl_metax_gate import metax_active

_MASTER_SWITCH = "FL_BOOT_SHIMS_PATCH"


def _master_enabled() -> bool:
    value = os.environ.get(_MASTER_SWITCH, "1").strip().lower()
    return value not in ("", "0", "false", "no", "off")


def install() -> bool:
    """Apply every MetaX compatibility shim in this process.

    Returns ``True`` when the shims were installed (or already were), and
    ``False`` when this is not a MetaX runtime or the master switch is off.
    """
    if not metax_active() or not _master_enabled():
        return False

    # Import order matters only for the triton stub: it must be registered
    # before vllm's triton_utils imports triton.experimental.gluon.
    from . import _fl_triton_experimental_stub  # noqa: F401

    from . import (  # noqa: F401
        _fl_metax_annassign_compat,
        _fl_metax_boolchain_compat,
        _fl_metax_functorch_compat,
        _fl_metax_kths_compat,
        _fl_metax_load_ptr_compat,
        _fl_metax_moe_is_monolithic_compat,
        _fl_metax_sampler_compat,
        _fl_metax_torch_accel,
    )
    return True


__all__ = ["install", "metax_active"]
