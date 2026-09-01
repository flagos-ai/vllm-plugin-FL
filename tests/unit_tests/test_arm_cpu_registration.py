# Copyright (c) 2026 BAAI. All rights reserved.

import os
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
    def test_arm_host_uses_vllm_cpu_platform(self):
        platforms = _fake_vllm_platforms()
        with (
            patch("platform.machine", return_value="aarch64"),
            patch.dict(sys.modules, {platforms.__name__: platforms}),
        ):
            self.assertEqual(
                vllm_fl._arm_cpu_platform(),
                "vllm.platforms.cpu.CpuPlatform",
            )

    def test_arm_cpu_register_uses_vllm_cpu_platform(self):
        with (
            patch.object(
                vllm_fl,
                "_arm_cpu_platform",
                return_value="vllm.platforms.cpu.CpuPlatform",
            ),
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

    def test_arm_target_requires_vllm_cpu_backend(self):
        platforms = _fake_vllm_platforms(cpu_platform=None)
        with (
            patch("platform.machine", return_value="aarch64"),
            patch.dict(sys.modules, {platforms.__name__: platforms}),
        ):
            self.assertIsNone(vllm_fl._arm_cpu_platform())

    def test_arm64_gpu_build_preserves_existing_platform_registration(self):
        platforms = _fake_vllm_platforms(cpu_platform=None)
        with (
            patch("platform.machine", return_value="aarch64"),
            patch.dict(sys.modules, {platforms.__name__: platforms}),
            patch.object(vllm_fl, "_patch_custom_ops") as custom_ops,
            patch.object(vllm_fl, "_patch_flash_attn_import") as flash_attn,
            patch.object(vllm_fl, "_patch_transformers_compat") as transformers,
            patch.object(vllm_fl, "_get_op_config") as get_op_config,
            patch(
                "vllm_fl.patches.glm_moe_dsa.apply_platform_patches"
            ) as platform_patches,
            patch.dict(os.environ, {"VLLM_WORKER_MULTIPROC_METHOD": "spawn"}),
        ):
            self.assertEqual(vllm_fl.register(), "vllm_fl.platform.PlatformFL")

        custom_ops.assert_called_once_with()
        flash_attn.assert_called_once_with()
        transformers.assert_called_once_with()
        platform_patches.assert_called_once_with()
        get_op_config.assert_called_once_with()

    def test_arm64_gpu_model_registration_does_not_import_arm_cpu_hooks(self):
        platforms = _fake_vllm_platforms(cpu_platform=None)
        platforms.current_platform.device_type = "cuda"
        adapter_module = ModuleType("vllm_fl.quantization.arm_cpu_w4a8")
        adapter_module.install_arm_cpu_packed_w4a8 = Mock()
        gdn_module = ModuleType("vllm_fl.patches.arm_cpu_gdn")
        gdn_module.apply_arm_cpu_gdn_state_indices_patch = Mock()

        with (
            patch("platform.machine", return_value="aarch64"),
            patch(
                "vllm_fl.patches.qwen3_5_text.apply_qwen3_5_text_patches"
            ) as qwen_compat,
            patch("vllm_fl.patches.moe_sum.patch_vllm_moe_sum") as moe_sum,
            patch.object(vllm_fl, "_arm_cpu_platform") as arm_cpu_platform,
            patch.object(vllm_fl, "_register_flagcx_connector") as flagcx,
            patch.object(vllm_fl, "register_quant_linear") as quant_linear,
            patch.object(vllm_fl, "register_router") as router,
            patch.dict(
                sys.modules,
                {
                    adapter_module.__name__: adapter_module,
                    gdn_module.__name__: gdn_module,
                    platforms.__name__: platforms,
                },
            ),
        ):
            vllm_fl.register_model()

        qwen_compat.assert_called_once_with()
        arm_cpu_platform.assert_not_called()
        gdn_module.apply_arm_cpu_gdn_state_indices_patch.assert_not_called()
        adapter_module.install_arm_cpu_packed_w4a8.assert_not_called()
        moe_sum.assert_called_once_with()
        flagcx.assert_called_once_with()
        quant_linear.assert_called_once_with()
        router.assert_called_once_with()

    def test_non_arm_cpu_build_is_not_selected(self):
        platforms = _fake_vllm_platforms()
        with (
            patch("platform.machine", return_value="x86_64"),
            patch.dict(sys.modules, {platforms.__name__: platforms}),
        ):
            self.assertIsNone(vllm_fl._arm_cpu_platform())
            platforms.cpu_platform_plugin.assert_not_called()

    def test_arm_registration_enables_quant_runtime_without_mode_flags(self):
        adapter_module = ModuleType("vllm_fl.quantization.arm_cpu_w4a8")
        adapter_module.install_arm_cpu_packed_w4a8 = Mock()
        platforms = _fake_vllm_platforms()

        with (
            patch(
                "vllm_fl.patches.arm_cpu_gdn.apply_arm_cpu_gdn_state_indices_patch"
            ) as gdn_compat,
            patch(
                "vllm_fl.patches.qwen3_5_text.apply_qwen3_5_text_patches"
            ) as qwen_compat,
            patch.object(vllm_fl, "_arm_cpu_platform", return_value=True),
            patch("flag_gems.vendor_name", "arm"),
            patch("vllm_fl.patches.moe_sum.patch_vllm_moe_sum") as moe_sum,
            patch.dict(
                sys.modules,
                {
                    adapter_module.__name__: adapter_module,
                    platforms.__name__: platforms,
                },
            ),
        ):
            vllm_fl.register_model()

        qwen_compat.assert_called_once_with()
        gdn_compat.assert_called_once_with()
        adapter_module.install_arm_cpu_packed_w4a8.assert_called_once_with()
        moe_sum.assert_not_called()

    def test_missing_flaggems_disables_packed_w4a8_integration(self):
        platforms = _fake_vllm_platforms()

        with (
            self.assertLogs("vllm_fl", level="WARNING") as logs,
            patch(
                "vllm_fl.patches.arm_cpu_gdn.apply_arm_cpu_gdn_state_indices_patch"
            ) as gdn_compat,
            patch(
                "vllm_fl.patches.qwen3_5_text.apply_qwen3_5_text_patches"
            ) as qwen_compat,
            patch.object(vllm_fl, "_arm_cpu_platform", return_value=True),
            patch("vllm_fl.patches.moe_sum.patch_vllm_moe_sum") as moe_sum,
            patch.dict(
                sys.modules,
                {
                    "flag_gems": None,
                    platforms.__name__: platforms,
                },
            ),
        ):
            vllm_fl.register_model()

        qwen_compat.assert_called_once_with()
        gdn_compat.assert_called_once_with()
        moe_sum.assert_not_called()
        self.assertIn("FlagGems is not installed", logs.output[0])

    def test_non_arm_flaggems_vendor_is_reported(self):
        platforms = _fake_vllm_platforms()

        with (
            self.assertLogs("vllm_fl", level="WARNING") as logs,
            patch(
                "vllm_fl.patches.arm_cpu_gdn.apply_arm_cpu_gdn_state_indices_patch"
            ) as gdn_compat,
            patch("vllm_fl.patches.qwen3_5_text.apply_qwen3_5_text_patches"),
            patch.object(vllm_fl, "_arm_cpu_platform", return_value=True),
            patch("flag_gems.vendor_name", "cuda"),
            patch("vllm_fl.patches.moe_sum.patch_vllm_moe_sum") as moe_sum,
            patch.dict(sys.modules, {platforms.__name__: platforms}),
        ):
            vllm_fl.register_model()

        gdn_compat.assert_called_once_with()
        moe_sum.assert_not_called()
        self.assertIn("selected vendor 'cuda'", logs.output[0])


if __name__ == "__main__":
    unittest.main()
