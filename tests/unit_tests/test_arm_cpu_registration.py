# Copyright (c) 2026 BAAI. All rights reserved.

import os
import subprocess
import sys
import textwrap
import unittest
from types import ModuleType, SimpleNamespace
from unittest.mock import Mock, patch

import vllm_fl


class TestArmCpuRegistration(unittest.TestCase):
    def test_import_does_not_eagerly_import_flaggems(self):
        script = textwrap.dedent(
            """
            import builtins

            original_import = builtins.__import__

            def guarded_import(name, *args, **kwargs):
                if name == "flag_gems" or name.startswith("flag_gems."):
                    raise RuntimeError("vllm_fl imported FlagGems eagerly")
                return original_import(name, *args, **kwargs)

            builtins.__import__ = guarded_import
            import vllm_fl
            """
        )
        completed = subprocess.run(
            [sys.executable, "-c", script],
            capture_output=True,
            text=True,
        )
        self.assertEqual(completed.returncode, 0, completed.stderr)

    def test_arm_cpu_build_is_selected(self):
        with (
            patch.object(vllm_fl.platform, "machine", return_value="aarch64"),
            patch.object(vllm_fl.metadata, "version", return_value="0.24.0+cpu"),
        ):
            self.assertTrue(vllm_fl._is_arm_cpu_build())

    def test_plain_version_honors_explicit_cpu_target(self):
        with (
            patch.object(vllm_fl.platform, "machine", return_value="aarch64"),
            patch.object(vllm_fl.metadata, "version", return_value="0.24.0"),
            patch.dict(os.environ, {"VLLM_TARGET_DEVICE": "cpu"}),
        ):
            self.assertTrue(vllm_fl._is_arm_cpu_build())

    def test_arm_accelerator_build_is_not_selected_as_cpu(self):
        with (
            patch.object(vllm_fl.platform, "machine", return_value="aarch64"),
            patch.object(vllm_fl.metadata, "version", return_value="0.24.0"),
            patch.dict(os.environ, {}, clear=True),
        ):
            self.assertFalse(vllm_fl._is_arm_cpu_build())

    def test_non_arm_cpu_build_is_not_selected(self):
        with (
            patch.object(vllm_fl.platform, "machine", return_value="x86_64"),
            patch.object(vllm_fl.metadata, "version", return_value="0.24.0+cpu"),
        ):
            self.assertFalse(vllm_fl._is_arm_cpu_build())

    def test_arm_cpu_register_uses_native_backed_platform(self):
        with (
            patch.object(vllm_fl, "_is_arm_cpu_build", return_value=True),
            patch.object(vllm_fl, "_patch_custom_ops") as custom_ops,
            patch.object(vllm_fl, "_patch_flash_attn_import") as flash_attn,
            patch.object(vllm_fl, "_patch_transformers_compat") as transformers,
        ):
            self.assertEqual(
                vllm_fl.register(),
                "vllm_fl.platform_cpu.CpuPlatformFL",
            )
            custom_ops.assert_not_called()
            flash_attn.assert_not_called()
            transformers.assert_not_called()

    def test_arm_registration_installs_cpu_compat_without_quant(self):
        compat_module = ModuleType("vllm_fl.patches.arm_cpu_vllm_0240")
        compat_module.install_arm_cpu_vllm_0240_compat = Mock()

        with (
            patch(
                "vllm_fl.patches.qwen3_5_text.apply_qwen3_5_text_patches"
            ) as qwen_compat,
            patch("vllm_fl.patches.moe_sum.patch_vllm_moe_sum") as moe_sum,
            patch.object(vllm_fl, "_is_arm_cpu_build", return_value=True),
            patch(
                "vllm.platforms.current_platform",
                SimpleNamespace(device_type="cpu"),
            ),
            patch.dict(
                sys.modules,
                {compat_module.__name__: compat_module},
            ),
            patch.dict(os.environ, {}, clear=True),
        ):
            vllm_fl.register_model()

        qwen_compat.assert_called_once_with()
        compat_module.install_arm_cpu_vllm_0240_compat.assert_called_once_with()
        moe_sum.assert_not_called()


if __name__ == "__main__":
    unittest.main()
