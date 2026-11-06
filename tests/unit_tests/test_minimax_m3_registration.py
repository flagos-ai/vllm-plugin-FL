# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 FlagOS Contributors
"""CPU checks for shared M3 registration, configuration scope and rollback."""

import importlib.util
import sys
import types
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

ROOT = Path(__file__).resolve().parents[2]
ARCHS = ("MiniMaxM3SparseForCausalLM", "MiniMaxM3SparseForConditionalGeneration")
NAMES = (*ARCHS, "MiniMaxM3MTP")


class RegistrationTests(unittest.TestCase):
    def setUp(self):
        self.enabled = False
        self.vendor = "metax"
        self.install = Mock()
        self.models = {name: object() for name in (*NAMES, "OtherModel")}
        self.configs = {"OtherModel": object()}
        self.supported = set(NAMES)
        self.failure_at = None

        def register(name, target):
            if name == self.failure_at:
                raise RuntimeError("injected registration failure")
            module, cls = target.split(":")
            self.models[name] = types.SimpleNamespace(
                module_name=module, class_name=cls
            )

        registry = types.SimpleNamespace(
            models=self.models,
            get_supported_archs=lambda: self.supported,
            register_model=register,
        )
        self.platform = types.SimpleNamespace(
            vendor_name="metax", is_cuda=lambda: self.vendor == "nvidia"
        )
        modules = {}
        for name in (
            "vllm",
            "vllm.platforms",
            "vllm.model_executor",
            "vllm.model_executor.models",
            "vllm.model_executor.models.config",
            "vllm_fl",
            "vllm_fl.models",
            "vllm_fl.models.minimax_m3_metax",
            "vllm_fl.models.minimax_m3_metax.bindings",
            "vllm_fl.minimax_m3_config",
            "torch",
        ):
            module = types.ModuleType(name)
            module.__path__ = []
            modules[name] = module
        modules["vllm.platforms"].current_platform = self.platform
        modules["vllm.model_executor.models"].ModelRegistry = registry
        cfg = modules["vllm.model_executor.models.config"]
        cfg.VerifyAndUpdateConfig = type("VerifyAndUpdateConfig", (), {})
        cfg.MODELS_CONFIG_MAP = self.configs
        modules["vllm_fl.models.minimax_m3_metax.bindings"].install = self.install
        policy = modules["vllm_fl.minimax_m3_config"]
        policy.is_enabled = lambda: self.enabled
        policy.load = lambda: types.SimpleNamespace(
            framework=types.SimpleNamespace(tensor_parallel_size=16)
        )
        modules["torch"].bfloat16 = "bf16"
        self.module_patch = patch.dict(sys.modules, modules)
        self.module_patch.start()
        spec = importlib.util.spec_from_file_location(
            "_m3_registration_test", ROOT / "vllm_fl/models/minimax_m3_registration.py"
        )
        self.module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(self.module)

    def tearDown(self):
        self.module_patch.stop()

    def test_non_cuda_without_profile_is_unchanged(self):
        before = self.models.copy()
        for vendor in ("metax", "cpu", "ascend"):
            self.vendor = self.platform.vendor_name = vendor
            self.module.register_minimax_m3_support()
        self.assertEqual(before, self.models)
        self.install.assert_not_called()

    def test_native_cuda_reuses_existing_entry(self):
        self.vendor = self.platform.vendor_name = "nvidia"
        mtp = self.models["MiniMaxM3MTP"]
        self.module.register_minimax_m3_support()
        for name in ARCHS:
            self.assertEqual(self.models[name].module_name, "vllm_fl.models.minimax_m3")
            self.assertEqual(self.models[name].class_name, name)
        self.assertIs(self.models["MiniMaxM3MTP"], mtp)
        self.install.assert_not_called()

    def test_metax_registration_is_idempotent_and_scoped(self):
        self.enabled = True
        other = self.models["OtherModel"]
        self.module.register_minimax_m3_support()
        first = self.models.copy()
        self.module.register_minimax_m3_support()
        self.assertTrue(all(self.models[name] is first[name] for name in NAMES))
        self.assertIs(self.models["OtherModel"], other)
        self.assertEqual(set(self.configs), {*NAMES, "OtherModel"})
        self.assertEqual(self.install.call_count, 2)

    def test_wrong_vendor_is_rejected_before_mutation(self):
        self.enabled = True
        self.vendor = self.platform.vendor_name = "nvidia"
        before = self.models.copy()
        with self.assertRaisesRegex(ValueError, "requires MetaX"):
            self.module.register_minimax_m3_support()
        self.assertEqual(self.models, before)
        self.install.assert_not_called()

    def test_missing_architecture_is_rejected(self):
        self.enabled = True
        self.supported.remove("MiniMaxM3MTP")
        before = self.models.copy()
        with self.assertRaisesRegex(RuntimeError, "architectures"):
            self.module.register_minimax_m3_support()
        self.assertEqual(self.models, before)
        self.install.assert_not_called()

    def test_registration_and_binding_failures_restore_registries(self):
        self.enabled = True
        for fail_registry in (True, False):
            with self.subTest(fail_registry=fail_registry):
                self.failure_at = ARCHS[1] if fail_registry else None
                self.install.side_effect = (
                    None if fail_registry else RuntimeError("bind")
                )
                before_models, before_configs = self.models.copy(), self.configs.copy()
                with self.assertRaises(RuntimeError):
                    self.module.register_minimax_m3_support()
                self.assertEqual(self.models, before_models)
                self.assertEqual(self.configs, before_configs)

    def test_profile_verifier_preserves_graph_and_selects_existing_worker(self):
        self.enabled = True
        graph = object()
        config = types.SimpleNamespace(
            model_config=types.SimpleNamespace(
                hf_config=types.SimpleNamespace(architectures=[ARCHS[1]]),
                dtype="bf16",
                quantization=None,
            ),
            parallel_config=types.SimpleNamespace(
                tensor_parallel_size=16, worker_cls="auto"
            ),
            compilation_config=graph,
        )
        self.module.MiniMaxM3MetaXConfig.verify_and_update_config(config)
        self.assertEqual(
            config.parallel_config.worker_cls,
            "vllm_fl.worker.minimax_m3.MiniMaxWorkerFL",
        )
        self.assertIs(config.compilation_config, graph)
        for kind in ("architecture", "dtype", "quantization", "tp"):
            with self.subTest(kind=kind):
                config.model_config.hf_config.architectures = (
                    ["OtherModel"] if kind == "architecture" else [ARCHS[0]]
                )
                config.model_config.dtype = "fp16" if kind == "dtype" else "bf16"
                config.model_config.quantization = (
                    "int8" if kind == "quantization" else None
                )
                config.parallel_config.tensor_parallel_size = 8 if kind == "tp" else 16
                config.parallel_config.worker_cls = "unchanged"
                with self.assertRaises(ValueError):
                    self.module.MiniMaxM3MetaXConfig.verify_and_update_config(config)
                self.assertEqual(config.parallel_config.worker_cls, "unchanged")

    def test_fused_mapping_preserves_existing_metadata(self):
        old = {"qkv_proj": ["custom_qkv"], "another": ["keep"]}
        cls = type("Model", (), {"packed_modules_mapping": old})
        self.module.add_fused_quantization_mapping(cls)
        self.assertIsNot(cls.packed_modules_mapping, old)
        self.assertEqual(old, {"qkv_proj": ["custom_qkv"], "another": ["keep"]})
        self.assertEqual(cls.packed_modules_mapping["qkv_proj"], ["custom_qkv"])
        self.assertEqual(
            cls.packed_modules_mapping["gate_up_proj"], ["gate_proj", "up_proj"]
        )
        previous = cls.packed_modules_mapping
        self.module.add_fused_quantization_mapping(cls)
        self.assertIs(cls.packed_modules_mapping, previous)

    def test_binding_preflight_and_partial_install_rollback(self):
        for failure in ("missing", "apply", None):
            with self.subTest(failure=failure):

                class TargetModule(types.ModuleType):
                    fail_once = False

                    def __setattr__(self, name, value):
                        if name == "minimax_m3_sparse_attn_decode" and self.fail_once:
                            self.fail_once = False
                            raise RuntimeError("injected binding failure")
                        super().__setattr__(name, value)

                native = types.ModuleType("native")
                mtp = types.ModuleType("mtp")
                indexer = types.ModuleType("indexer")
                sparse = TargetModule("sparse_attention")
                tower = types.ModuleType("vision_tower")
                model, vision = (
                    types.ModuleType("adapted_model"),
                    types.ModuleType("adapted_vision"),
                )
                attention, layers, msa = (
                    types.ModuleType(n) for n in ("attention", "layers", "msa")
                )
                model_names = (
                    "MiniMAXGemmaRMSNorm",
                    "MiniMaxM3MoE",
                    "MiniMaxM3Attention",
                    "MiniMaxM3SparseAttention",
                )
                for name in model_names:
                    setattr(native, name, object())
                    setattr(model, name, object())
                native.SiluAndMulWithClamp = object()
                native.MiniMaxVLVisionModel = object()
                layers.SiluAndMulWithClampFL = object()
                for name in ("MiniMaxVLPatchEmbed", "MiniMaxVLVisionModel"):
                    setattr(tower, name, object())
                    setattr(vision, name, object())
                tower.MMEncoderAttention = object()
                for name in (
                    "minimax_m3_index_decode",
                    "minimax_m3_index_score",
                    "minimax_m3_index_topk",
                ):
                    setattr(indexer, name, object())
                    setattr(msa, name, object())
                sparse.minimax_m3_sparse_attn = object()
                sparse.minimax_m3_sparse_attn_decode = object()
                attention.sparse_attn, attention.sparse_decode = object(), object()
                targets = (native, mtp, indexer, sparse, tower)
                if failure == "missing":
                    del sparse.minimax_m3_sparse_attn_decode
                before = [
                    {k: v for k, v in vars(m).items() if not k.startswith("_")}
                    for m in targets
                ]
                if failure == "apply":
                    sparse.fail_once = True
                modules = {
                    "vllm.models.minimax_m3.common": types.SimpleNamespace(
                        indexer=indexer, sparse_attention=sparse, vision_tower=tower
                    ),
                    "vllm.models.minimax_m3.nvidia": types.SimpleNamespace(
                        model=native, mtp=mtp
                    ),
                    "vllm_fl.models.minimax_m3_metax": types.SimpleNamespace(
                        model=model, vision=vision
                    ),
                    "vllm_fl.attention.minimax_m3": types.SimpleNamespace(
                        bridge=attention
                    ),
                    "vllm_fl.ops.minimax_m3": types.SimpleNamespace(
                        layers=layers, msa=msa
                    ),
                    "vllm_fl.ops.minimax_m3.vision": types.SimpleNamespace(
                        M3VisionAttentionFL=object()
                    ),
                    "vllm_fl.minimax_m3_config": types.SimpleNamespace(
                        load=lambda: types.SimpleNamespace(
                            framework=types.SimpleNamespace(
                                flagos_msa=True, flagos_swiglu=True
                            )
                        )
                    ),
                }
                spec = importlib.util.spec_from_file_location(
                    "vllm_fl.models.minimax_m3_metax._binding_test",
                    ROOT / "vllm_fl/models/minimax_m3_metax/bindings.py",
                )
                binding = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(binding)
                with (
                    patch.dict(sys.modules, modules),
                    patch.object(binding, "validate_upstream"),
                ):
                    if failure:
                        with self.assertRaises(RuntimeError):
                            binding.install()
                        self.assertFalse(binding._INSTALLED)
                        self.assertEqual(binding._BINDINGS, [])
                        for module, expected in zip(targets, before):
                            actual = {
                                k: v
                                for k, v in vars(module).items()
                                if not k.startswith("_") and k != "fail_once"
                            }
                            self.assertEqual(actual, expected)
                    else:
                        binding.install()
                        installed = list(binding._BINDINGS)
                        binding.install()
                        self.assertEqual(binding._BINDINGS, installed)
                        native.MiniMaxM3Attention = object()
                        with self.assertRaisesRegex(RuntimeError, "binding changed"):
                            binding.install()


if __name__ == "__main__":
    unittest.main()
