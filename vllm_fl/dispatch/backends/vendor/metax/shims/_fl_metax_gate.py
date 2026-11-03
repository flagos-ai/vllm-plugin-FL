"""Shared gate for the MetaX (MACA) vLLM 0.28 compatibility shims.

Every shim in this package must apply whenever the process runs on the MetaX
runtime, no matter which platform plugin owns the platform slot.  Keying the
gate on ``VLLM_PLUGINS`` is wrong: the supported configuration is
``VLLM_PLUGINS=fl`` with ``GEMS_VENDOR=metax``, where the FL plugin keeps the
platform slot and the vendor plugin name never appears in the list.

The gate therefore asks the runtime:
  * an already resolved vLLM platform reporting ``is_metax()``;
  * ``GEMS_VENDOR=metax``;
  * the MACA userspace library (``mcoplib``) being importable.

It deliberately avoids resolving ``current_platform`` itself, because this
runs from the plugin entry point while vLLM is still resolving it.
"""

import importlib.util
import os


def metax_active() -> bool:
    """True when this process runs on the MetaX (MACA) backend."""
    try:
        from vllm_fl.ops._C_ops_registry import _is_metax_backend

        return bool(_is_metax_backend())
    except Exception:
        pass
    if os.environ.get("GEMS_VENDOR", "").strip().lower() == "metax":
        return True
    try:
        return importlib.util.find_spec("mcoplib") is not None
    except Exception:
        return False
