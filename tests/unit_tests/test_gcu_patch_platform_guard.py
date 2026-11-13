# Copyright (c) 2026 BAAI. All rights reserved.

"""Configuration-only imports must not initialize an absent vendor runtime."""

import subprocess
import sys
import textwrap

import pytest


@pytest.mark.parametrize("state", ["missing", "unavailable"])
def test_gcu_package_import_without_runtime(state):
    # A fresh interpreter matters: importing the package a second time can hide
    # initialization failures behind sys.modules, as in parallel CI workers.
    code = textwrap.dedent(
        f"""
        import importlib.abc
        import sys
        from types import SimpleNamespace
        import torch
        import vllm.platforms

        class NoTorchGCU(importlib.abc.MetaPathFinder):
            def find_spec(self, fullname, path=None, target=None):
                if fullname == 'torch_gcu' or fullname.startswith('torch_gcu.'):
                    raise AssertionError('attempted vendor runtime import: ' + fullname)

        if {state!r} == 'missing':
            if hasattr(torch, 'gcu'):
                del torch.gcu
        else:
            torch.gcu = SimpleNamespace(is_available=lambda: False)
        for name in list(sys.modules):
            if name == 'torch_gcu' or name.startswith('torch_gcu.'):
                del sys.modules[name]
        sys.meta_path.insert(0, NoTorchGCU())
        vllm.platforms.current_platform.vendor_name = 'enflame'
        from vllm_fl.dispatch.backends.vendor.gcu import patch
        patch.apply_gcu_patches()
        assert patch._patches_applied is False
        print('GCU configuration import without runtime: PASS')
        """
    )
    result = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, timeout=120
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "GCU configuration import without runtime: PASS" in result.stdout
