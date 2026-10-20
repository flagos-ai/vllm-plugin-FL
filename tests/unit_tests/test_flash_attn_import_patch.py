# Copyright (c) 2026 BAAI. All rights reserved.

import builtins
import sys
import unittest
from contextlib import contextmanager
from types import ModuleType
from unittest.mock import patch

import vllm_fl

_PARENT = "vllm.vllm_flash_attn"
_INTERFACE = f"{_PARENT}.flash_attn_interface"
_OTHER_CHILD = f"{_PARENT}.other_child"
_MISSING = object()


@contextmanager
def _preserve_modules(*names):
    original = {name: sys.modules.get(name, _MISSING) for name in names}
    try:
        yield
    finally:
        for name, module in original.items():
            if module is _MISSING:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = module


class TestFlashAttnImportPatch(unittest.TestCase):
    def test_failed_probe_removes_orphan_and_installs_fallback(self):
        orphan = ModuleType(_INTERFACE)
        unrelated_child = ModuleType(_OTHER_CHILD)
        real_import = builtins.__import__
        probe_count = 0

        def import_with_failed_flash_attn_probe(
            name, globals=None, locals=None, fromlist=(), level=0
        ):
            nonlocal probe_count
            if name == _PARENT:
                probe_count += 1
                # Reproduce vLLM's failed package import: the parent is removed
                # while its successfully imported interface remains cached.
                sys.modules[_INTERFACE] = orphan
                raise ImportError("_vllm_fa2_C is unavailable")
            return real_import(name, globals, locals, fromlist, level)

        with (
            _preserve_modules(_PARENT, _INTERFACE, _OTHER_CHILD),
            patch(
                "builtins.__import__",
                side_effect=import_with_failed_flash_attn_probe,
            ),
        ):
            sys.modules.pop(_PARENT, None)
            sys.modules.pop(_INTERFACE, None)
            sys.modules[_OTHER_CHILD] = unrelated_child

            vllm_fl._patch_flash_attn_import()

            fallback = sys.modules[_PARENT]
            self.assertIsInstance(fallback, ModuleType)
            self.assertFalse(fallback.FA2_AVAILABLE)
            self.assertFalse(fallback.FA3_AVAILABLE)
            self.assertFalse(fallback.is_fa_version_supported(2))
            self.assertIsNone(fallback.flash_attn_varlen_func)
            self.assertIsNone(fallback.get_scheduler_metadata)
            self.assertIn(
                "not available",
                fallback.fa_version_unsupported_reason(2),
            )
            self.assertNotIn(_INTERFACE, sys.modules)
            self.assertIs(sys.modules[_OTHER_CHILD], unrelated_child)

            # The installed fallback makes subsequent calls idempotent and
            # prevents another probe of the unavailable CUDA extension.
            vllm_fl._patch_flash_attn_import()
            self.assertIs(sys.modules[_PARENT], fallback)
            self.assertEqual(probe_count, 1)

    def test_successful_probe_preserves_loaded_modules(self):
        parent = ModuleType(_PARENT)
        interface = ModuleType(_INTERFACE)
        real_import = builtins.__import__

        def import_with_successful_flash_attn_probe(
            name, globals=None, locals=None, fromlist=(), level=0
        ):
            if name == _PARENT:
                sys.modules[_PARENT] = parent
                sys.modules[_INTERFACE] = interface
            return real_import(name, globals, locals, fromlist, level)

        with (
            _preserve_modules(_PARENT, _INTERFACE),
            patch(
                "builtins.__import__",
                side_effect=import_with_successful_flash_attn_probe,
            ),
        ):
            sys.modules.pop(_PARENT, None)
            sys.modules.pop(_INTERFACE, None)

            vllm_fl._patch_flash_attn_import()

            self.assertIs(sys.modules[_PARENT], parent)
            self.assertIs(sys.modules[_INTERFACE], interface)


if __name__ == "__main__":
    unittest.main()
