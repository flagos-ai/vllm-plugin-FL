# Copyright (c) 2026 BAAI. All rights reserved.

import importlib.util
import sys
from dataclasses import dataclass, fields
from pathlib import Path
from types import ModuleType

import pytest


@pytest.mark.parametrize("existing", [None, False, True])
def test_hook_metadata_preserves_compiler_options_and_cache_key(monkeypatch, existing):
    @dataclass(frozen=True)
    class TXDAOptions:
        num_warps: int = 4
        num_stages: int = 2

        def hash(self):
            return tuple(sorted(self.__dict__.items()))

    if existing is not None:
        TXDAOptions.launch_cooperative_grid = existing
    options = TXDAOptions()
    before_options = dict(options.__dict__)
    before_fields = fields(options)
    before_hash = options.hash()
    for name in ("triton", "triton.backends", "triton.backends.tsingmicro"):
        package = ModuleType(name)
        package.__path__ = []
        monkeypatch.setitem(sys.modules, name, package)
    compiler = ModuleType("triton.backends.tsingmicro.compiler")
    compiler.TXDAOptions = TXDAOptions
    monkeypatch.setitem(sys.modules, compiler.__name__, compiler)
    path = (
        Path(__file__).parents[3]
        / "vllm_fl/dispatch/backends/vendor/txda/impl/patchs/triton_compat.py"
    )
    spec = importlib.util.spec_from_file_location("txda_triton_compat_test", path)
    patch = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(patch)
    patch.patch_triton_compile_hooks()  # Reapplying is safe.

    assert options.launch_cooperative_grid is (False if existing is None else existing)
    assert options.__dict__ == before_options
    assert fields(options) == before_fields
    assert options.hash() == before_hash
