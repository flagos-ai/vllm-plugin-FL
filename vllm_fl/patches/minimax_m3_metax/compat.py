# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 FlagOS Contributors

"""Central registry for the retained SDK/framework compatibility hooks.

This module manages real runtime patches. It does not claim the deployment is
patch-free. Static M3 operators, model routing and shape policies do not use it.
"""

import hashlib
import importlib
import importlib.metadata
import importlib.util
import inspect
import json
import os
import sys
from pathlib import Path

from vllm_fl.minimax_m3_config import ROOT

_APPLIED = {}
_INITIALIZING = False


def require_stack():
    expected = {
        "torch": "2.10.0+metax3.8.1.0",
        "vllm": "0.24.0",
        "flagtree": "0.6.1+metax3.6",
    }
    for package, version in expected.items():
        actual = importlib.metadata.version(package)
        if package == "vllm":
            actual = actual.split("+", 1)[0]
        if actual != version:
            raise RuntimeError("Unvalidated hook stack: " + package)


def signature(obj):
    while hasattr(obj, "fn") and obj.fn is not obj:
        obj = obj.fn
    try:
        return [
            [p.name, str(p.kind)] for p in inspect.signature(obj).parameters.values()
        ]
    except (ValueError, TypeError):
        return None


def identity(obj):
    try:
        source = inspect.getsourcefile(obj)
    except TypeError:
        source = None
    return {
        "module": getattr(obj, "__module__", None),
        "qualname": getattr(obj, "__qualname__", None),
        "signature": signature(obj),
        "source": source,
    }


def target(spec):
    module = sys.modules.get(spec["module"])
    if module is None:
        return None
    value = module
    for name in spec["attribute"].split("."):
        if not hasattr(value, name):
            return None
        value = getattr(value, name)
    return value


def flush():
    root = Path(os.environ.get("VLLM_CACHE_ROOT", "/tmp")).parent / "audit"
    root.mkdir(parents=True, exist_ok=True)
    (root / ("compatibility-pid" + str(os.getpid()) + ".json")).write_text(
        json.dumps({"pid": os.getpid(), "hooks": _APPLIED}, indent=2) + "\n"
    )


def ensure_dtype_alias():
    require_stack()
    import torch

    if not hasattr(torch, "float4_e2m1fn_x2"):
        torch.float4_e2m1fn_x2 = torch.uint8
        _APPLIED["torch.float4_metadata_alias"] = {
            "reason": "vLLM imports this dtype in tolerance metadata; model stays BF16",
            "replacement": "torch.uint8",
            "active": True,
        }


def install_platform_hooks(accelerator, custom_ops, transformers):
    require_stack()
    for name, fn in (
        ("accelerator_api", accelerator),
        ("custom_op_schemas", custom_ops),
        ("transformers_layer_types", transformers),
    ):
        if name not in _APPLIED:
            fn()
            _APPLIED[name] = {
                "reason": "Retained platform API/ABI compatibility",
                "installer": identity(fn),
            }
    flush()


def install_metax_hooks():
    global _INITIALIZING
    if _INITIALIZING:
        return
    require_stack()
    specs = json.loads(Path(__file__).with_name("compatibility.json").read_text())
    _INITIALIZING = True
    try:
        for spec in specs:
            name = spec["name"]
            if name in _APPLIED:
                for t in spec["targets"]:
                    obj = target(t)
                    previous = _APPLIED[name]["after"].get(
                        t["module"] + ":" + t["attribute"]
                    )
                    if (
                        obj is not None
                        and previous is not None
                        and identity(obj) != previous
                    ):
                        raise RuntimeError("Hook binding changed: " + name)
                continue
            path = ROOT / spec["relative_source"]
            if hashlib.sha256(path.read_bytes()).hexdigest() != spec["sha256"]:
                raise RuntimeError("Unexpected hook source: " + name)
            before = {}
            for t in spec["targets"]:
                package_name = t["source_file"].split("/", 1)[0]
                package_spec = importlib.util.find_spec(package_name)
                if package_spec is None or not package_spec.submodule_search_locations:
                    raise RuntimeError("Missing compatibility target: " + package_name)
                original_file = (
                    Path(next(iter(package_spec.submodule_search_locations))).parent
                    / t["source_file"]
                )
                if (
                    hashlib.sha256(original_file.read_bytes()).hexdigest()
                    != t["source_sha256"]
                ):
                    raise RuntimeError(
                        "Unexpected target source: " + str(original_file)
                    )
                obj = target(t)
                key = t["module"] + ":" + t["attribute"]
                before[key] = identity(obj) if obj is not None else None
                if (
                    obj is not None
                    and t.get("signature") is not None
                    and signature(obj) != t["signature"]
                ):
                    raise RuntimeError("Unexpected target signature: " + key)
            importlib.import_module(spec["import_module"])
            after = {
                t["module"]
                + ":"
                + t["attribute"]: identity(target(t)) if target(t) is not None else None
                for t in spec["targets"]
            }
            _APPLIED[name] = {
                "reason": spec["reason"],
                "source_sha256": spec["sha256"],
                "before": before,
                "after": after,
                "guard": "Exact package version and original source SHA; live signature checked when target is already imported",
            }
        flush()
    finally:
        _INITIALIZING = False
