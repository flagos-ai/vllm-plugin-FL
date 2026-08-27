# Copyright (c) 2026 BAAI. All rights reserved.

import sys
import unittest
from types import ModuleType, SimpleNamespace
from unittest.mock import Mock, patch

import vllm_fl


def _fake_vllm_platforms(cpu_platform="vllm.platforms.cpu.CpuPlatform"):
    module = ModuleType("vllm.platforms")
    module.cpu_platform_plugin = Mock(return_value=cpu_platform)
    module.current_platform = SimpleNamespace(device_type="cpu")
    return module


class TestArmCpuRegistration(unittest.TestCase):
    def test_arm_cpu_target_uses_standard_flaggems_vendor_selection(self):
        platforms = _fake_vllm_platforms()
        with (
            patch("platform.machine", return_value="aarch64"),
            patch("flag_gems.vendor_name", "arm"),
            patch.dict(sys.modules, {platforms.__name__: platforms}),
        ):
            self.assertTrue(vllm_fl._is_arm_cpu_target())

    def test_arm_cpu_register_uses_native_backed_cpu_platform(self):
        with (
            patch.object(vllm_fl, "_is_arm_cpu_target", return_value=True),
            patch.object(vllm_fl, "_patch_custom_ops") as custom_ops,
            patch.object(vllm_fl, "_patch_flash_attn_import") as flash_attn,
            patch.object(vllm_fl, "_patch_transformers_compat") as transformers,
        ):
            self.assertEqual(
                vllm_fl.register(),
                "vllm.platforms.cpu.CpuPlatform",
            )
            custom_ops.assert_not_called()
            flash_attn.assert_not_called()
            transformers.assert_not_called()

    def test_arm_accelerator_vendor_does_not_select_cpu(self):
        platforms = _fake_vllm_platforms()
        with (
            patch("platform.machine", return_value="aarch64"),
            patch("flag_gems.vendor_name", "ascend"),
            patch.dict(sys.modules, {platforms.__name__: platforms}),
        ):
            self.assertFalse(vllm_fl._is_arm_cpu_target())
            platforms.cpu_platform_plugin.assert_not_called()

    def test_arm_target_requires_vllm_cpu_backend(self):
        platforms = _fake_vllm_platforms(cpu_platform=None)
        with (
            patch("platform.machine", return_value="aarch64"),
            patch("flag_gems.vendor_name", "arm"),
            patch.dict(sys.modules, {platforms.__name__: platforms}),
        ):
            self.assertFalse(vllm_fl._is_arm_cpu_target())

    def test_non_arm_cpu_build_is_not_selected(self):
        platforms = _fake_vllm_platforms()
        with (
            patch("platform.machine", return_value="x86_64"),
            patch.dict(sys.modules, {platforms.__name__: platforms}),
        ):
            self.assertFalse(vllm_fl._is_arm_cpu_target())
            platforms.cpu_platform_plugin.assert_not_called()

    def test_arm_registration_enables_quant_runtime_without_mode_flags(self):
        integration_module = ModuleType("flag_gems.integrations.vllm")
        integration_module.install_arm_cpu_runtime = Mock()
        platforms = _fake_vllm_platforms()

        with (
            patch(
                "vllm_fl.patches.qwen3_5_text.apply_qwen3_5_text_patches"
            ) as qwen_compat,
            patch.object(vllm_fl, "_is_arm_cpu_target", return_value=True),
            patch("vllm_fl.patches.moe_sum.patch_vllm_moe_sum") as moe_sum,
            patch.dict(
                sys.modules,
                {
                    integration_module.__name__: integration_module,
                    platforms.__name__: platforms,
                },
            ),
        ):
            vllm_fl.register_model()

        qwen_compat.assert_called_once_with()
        integration_module.install_arm_cpu_runtime.assert_called_once_with()
        moe_sum.assert_not_called()


if __name__ == "__main__":
    unittest.main()
