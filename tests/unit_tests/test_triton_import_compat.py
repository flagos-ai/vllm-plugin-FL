# Copyright (c) 2026 BAAI. All rights reserved.

import builtins
import os
import unittest
from contextlib import contextmanager
from unittest.mock import patch

import vllm_fl

_VENDOR_ENV_VARS = ("VLLM_FL_PLATFORM", "VLLM_VENDOR", "GEMS_VENDOR")


@contextmanager
def _vendor_env(**values):
    with patch.dict(os.environ, {}, clear=False):
        for env_name in _VENDOR_ENV_VARS:
            os.environ.pop(env_name, None)
        os.environ.update(values)
        yield


class TestFlagGemsTritonImportCompat(unittest.TestCase):
    def test_ascend_skips_triton_import_regardless_of_flaggems_setting(self):
        real_import = builtins.__import__

        def reject_triton_import(
            name, globals=None, locals=None, fromlist=(), level=0
        ):
            if name == "triton" or name.startswith("triton."):
                raise AssertionError(f"unexpected Triton import: {name}")
            return real_import(name, globals, locals, fromlist, level)

        for use_flaggems in ("0", "1"):
            with (
                self.subTest(USE_FLAGGEMS=use_flaggems),
                _vendor_env(
                    VLLM_FL_PLATFORM="ascend",
                    USE_FLAGGEMS=use_flaggems,
                ),
                patch(
                    "builtins.__import__",
                    side_effect=reject_triton_import,
                ),
            ):
                vllm_fl._patch_flag_gems_triton_import_compat()

    def test_explicit_vendor_hints_skip_non_kunlunxin_platforms(self):
        for env_name in ("VLLM_VENDOR", "GEMS_VENDOR"):
            with (
                self.subTest(env_name=env_name),
                _vendor_env(**{env_name: "ascend"}),
            ):
                self.assertFalse(
                    vllm_fl._should_patch_flag_gems_triton_import_compat()
                )

    def test_explicit_platform_has_precedence_over_vendor_hints(self):
        with _vendor_env(
            VLLM_FL_PLATFORM="ascend",
            VLLM_VENDOR="kunlunxin",
            GEMS_VENDOR="kunlunxin",
        ):
            self.assertFalse(
                vllm_fl._should_patch_flag_gems_triton_import_compat()
            )

    def test_generic_cuda_platform_uses_vendor_hint(self):
        with _vendor_env(VLLM_FL_PLATFORM="cuda", GEMS_VENDOR="ascend"):
            self.assertFalse(
                vllm_fl._should_patch_flag_gems_triton_import_compat()
            )

        with _vendor_env(VLLM_FL_PLATFORM="cuda", GEMS_VENDOR="kunlunxin"):
            self.assertTrue(
                vllm_fl._should_patch_flag_gems_triton_import_compat()
            )

    def test_auto_detection_and_explicit_kunlunxin_keep_probe(self):
        real_import = builtins.__import__

        def record_missing_triton(
            name, globals=None, locals=None, fromlist=(), level=0
        ):
            if name == "triton":
                triton_imports.append(name)
                raise ImportError("simulated missing Triton")
            return real_import(name, globals, locals, fromlist, level)

        for values in ({}, {"GEMS_VENDOR": "kunlunxin"}):
            triton_imports = []
            with (
                self.subTest(values=values),
                _vendor_env(**values),
                patch(
                    "builtins.__import__",
                    side_effect=record_missing_triton,
                ),
            ):
                vllm_fl._patch_flag_gems_triton_import_compat()
            self.assertEqual(triton_imports, ["triton"])


if __name__ == "__main__":
    unittest.main()
