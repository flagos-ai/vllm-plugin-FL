# Copyright (c) 2026 BAAI. All rights reserved.
"""Restore early MetaX patches at the last boundary before model creation."""

import ast
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest

SOURCE = Path(__file__).resolve().parents[3] / "vllm_fl/worker/runtime.py"


@pytest.mark.parametrize("vendor", ["metax", "nvidia"])
def test_worker_restores_overwritten_bindings_before_registering_ops(
    monkeypatch, vendor
):
    events = []
    bindings = {"patched": False}
    custom_ops = ModuleType("vllm_fl.ops.custom_ops")

    def register_oot_ops():
        # Simulate an early successful patch subsequently replaced by the
        # remainder of _custom_ops's module body. A stale module flag cannot
        # tell us that the live callable has been overwritten.
        assert bindings["patched"] == (vendor == "metax")
        events.append("register")

    custom_ops.register_oot_ops = register_oot_ops
    platforms = ModuleType("vllm.platforms")
    platforms.current_platform = SimpleNamespace(vendor_name=vendor)
    monkeypatch.setitem(sys.modules, custom_ops.__name__, custom_ops)
    monkeypatch.setitem(sys.modules, platforms.__name__, platforms)
    patches_name = "vllm_fl.dispatch.backends.vendor.metax.patches"
    if vendor == "metax":
        patches = ModuleType(patches_name)

        def ensure_patches():
            bindings["patched"] = True
            events.append("restore")
            return {}

        patches.ensure_patches = ensure_patches
        monkeypatch.setitem(sys.modules, patches_name, patches)
    else:
        # Non-MetaX workers must not import or execute MetaX patch code.
        monkeypatch.setitem(sys.modules, patches_name, None)

    tree = ast.parse(SOURCE.read_text(encoding="utf-8"))
    fn = next(
        node
        for node in tree.body
        if getattr(node, "name", None) == "_initialize_fl_runtime"
    )
    namespace = {
        "fl_envs": SimpleNamespace(USE_FLAGGEMS=False),
        "patch_mm_encoder_attention": lambda: events.append("encoder"),
    }
    module = ast.fix_missing_locations(ast.Module(body=[fn], type_ignores=[]))
    exec(compile(module, str(SOURCE), "exec"), namespace)
    namespace[fn.name](0)
    expected = ["encoder", "register"]
    if vendor == "metax":
        expected.insert(0, "restore")
    assert events == expected
