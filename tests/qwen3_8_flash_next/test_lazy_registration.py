"""Verify lazy registration boundaries without loading accelerator dependencies."""

import builtins
import importlib.util
import sys
import unittest
from pathlib import Path
from types import ModuleType, SimpleNamespace as NS
from unittest.mock import patch

import pytest
import torch

ROOT = Path(__file__).resolve().parents[2]


def load_file(name, path, package=False):
    options = {"submodule_search_locations": [str(path.parent)]} if package else {}
    spec = importlib.util.spec_from_file_location(name, path, **options)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def forbid_runtime_imports():
    original = builtins.__import__

    def importing(name, *args, **kwargs):
        if name.startswith("vllm.v1"):
            raise AssertionError(f"unrequested runtime import: {name}")
        return original(name, *args, **kwargs)

    return patch("builtins.__import__", importing)


def registration_modules():
    names = [
        "vllm",
        "vllm.envs",
        "vllm.config",
        "vllm.config.speculative",
        "vllm.config.compilation",
        "vllm.model_executor",
        "vllm.model_executor.models",
        "vllm.model_executor.models.config",
        "vllm.model_executor.models.registry",
        "vllm.transformers_utils",
        "vllm.transformers_utils.config",
        "vllm_fl",
        "vllm_fl.models",
        "vllm_fl.models.qwen3_8_flash_next",
        "vllm_fl.models.qwen3_8_flash_next.config",
        "vllm_fl.models.qwen3_8_flash_next.gpu.runtime",
    ]
    modules = {name: ModuleType(name) for name in names}
    modules["vllm_fl.patches"] = ModuleType("vllm_fl.patches")
    modules["vllm_fl.patches._qwen38_installation"] = load_file(
        "qwen38_setup_test_helper", ROOT / "vllm_fl/patches/_qwen38_installation.py"
    )
    modules["vllm_fl.models.qwen3_8_flash_next.gpu.runtime"].is_metax = lambda: True
    modules["vllm"].__version__ = "0.24.0"
    modules["vllm"].envs = modules["vllm.envs"]
    modules["vllm.envs"].VLLM_USE_V2_MODEL_RUNNER = False

    class Spec:
        hf_config_override = staticmethod(lambda config: ("original", config))

    modules["vllm.config.speculative"].SpeculativeConfig = Spec

    class Registry:
        registered = {}
        models = registered

        @classmethod
        def get_supported_archs(cls):
            return set(cls.registered)

        @classmethod
        def register_model(cls, name, path):
            cls.registered[name] = path

    modules["vllm.model_executor.models"].ModelRegistry = Registry
    modules["vllm.model_executor.models.registry"].ModelRegistry = Registry
    registry_module = modules["vllm.model_executor.models.registry"]
    registry_module._TEXT_GENERATION_MODELS = {}
    registry_module._MULTIMODAL_MODELS = {}
    registry_module._VLLM_MODELS = {}
    config_module = modules["vllm.model_executor.models.config"]
    config_module.MODELS_CONFIG_MAP = {}
    config_module.Qwen3_5ForConditionalGenerationConfig = type(
        "Verifier", (), {"verify_and_update_config": staticmethod(lambda cfg: None)}
    )
    modules["vllm.model_executor.models"].config = config_module
    modules["vllm.model_executor.models"].registry = modules[
        "vllm.model_executor.models.registry"
    ]
    modules["vllm.transformers_utils"].config = modules[
        "vllm.transformers_utils.config"
    ]
    modules["vllm.transformers_utils.config"]._CONFIG_REGISTRY = {}
    modules["vllm.config.compilation"].CompilationConfig = type(
        "CompilationConfig", (), {"_attention_ops": []}
    )
    models = modules["vllm_fl.models.qwen3_8_flash_next.config"]
    models.Qwen3_8FlashNextConfig = type("CompositeConfig", (), {})
    models.Qwen3_8FlashNextTextConfig = type("TextConfig", (), {})
    return modules, Spec, Registry


class LazyRegistrationTest(unittest.TestCase):
    def test_public_registration_does_not_import_proposer_or_mrv2(self):
        modules, Spec, Registry = registration_modules()
        with patch.dict(sys.modules, modules), forbid_runtime_imports():
            compat = load_file(
                "qwen38_lazy_compat", ROOT / "vllm_fl/patches/qwen3_8_flash_next_mtp.py"
            )
            sys.modules["qwen38_lazy_registration.qwen3_8_flash_next_mtp"] = compat
            registration = load_file(
                "qwen38_lazy_registration",
                ROOT / "vllm_fl/patches/qwen3_8_flash_next.py",
                package=True,
            )
            registration.apply_qwen3_8_flash_next_patches()
            first = Spec.hf_config_override
            registration.apply_qwen3_8_flash_next_patches()
            self.assertIs(Spec.hf_config_override, first)
            other = NS(model_type="llama", architectures=["LlamaForCausalLM"])
            self.assertEqual(Spec.hf_config_override(other), ("original", other))
            target = NS(
                model_type="qwen4_exp",
                text_config=NS(
                    model_type="qwen4_exp_text",
                    hc_count=4,
                    mtp_num_hidden_layers=1,
                    ple_layer_ids=[2],
                    hidden_size=2560,
                ),
            )
            draft = Spec.hf_config_override(target)
            self.assertEqual(draft.architectures, ["Qwen3_8FlashNextMTP"])
            self.assertEqual(draft.hc_mult, 4)
            self.assertEqual(target.text_config.ple_layer_ids, [2])
            self.assertEqual(len(Registry.registered), 6)

    def test_worker_constructor_installs_only_qwen_mtp_before_original_init(self):
        events = []
        fake = ModuleType("vllm_fl.patches.qwen3_8_flash_next_mtp")
        fake.install_qwen38_mtp_runtime_compat = lambda: events.append("install")
        with forbid_runtime_imports():
            worker = load_file(
                "qwen38_lazy_worker",
                ROOT / "vllm_fl/worker/qwen3_8_flash_next.py",
                package=True,
            )

        class Runner:
            def __init__(self, vllm_config, device=None):
                events.append("original")
                self.model_config = vllm_config.model_config

            def _init_model_kwargs(self):
                return {"original": 7}

            def _dummy_run(self, n):
                return n

            def _build_attention_metadata(self, **kwargs):
                return kwargs

        worker.is_metax = lambda: True
        worker.install_qwen38_runner_hooks(Runner)
        first = Runner.__init__
        worker.install_qwen38_runner_hooks(Runner)
        self.assertIs(first, Runner.__init__)

        def cfg(qwen, method):
            return NS(
                model_config=NS(
                    hf_text_config=NS(
                        supports_qwen38_ngram_context=qwen, ple_layer_ids=[]
                    )
                ),
                speculative_config=None if method is None else NS(method=method),
            )

        fake_modules = {
            name: ModuleType(name)
            for name in (
                "vllm_fl",
                "vllm_fl.patches",
                "vllm_fl.patches.qwen3_8_flash_next",
            )
        }
        fake_modules[fake.__name__] = fake
        with patch.dict(sys.modules, fake_modules), forbid_runtime_imports():
            other = Runner(cfg(False, "mtp"))
            self.assertEqual(events, ["original"])
            self.assertEqual(other._init_model_kwargs(), {"original": 7})
            self.assertEqual(other._dummy_run(3), 3)
            self.assertEqual(other._build_attention_metadata(x=2), {"x": 2})
            events.clear()
            Runner(cfg(True, None))
            self.assertEqual(events, ["original"])
            events.clear()
            Runner(vllm_config=cfg(True, "mtp"))
            self.assertEqual(events, ["install", "original"])

    def test_verifier_accepts_causal_text_and_rejects_mrv2_or_visual(self):
        modules, _, _ = registration_modules()
        with patch.dict(sys.modules, modules), forbid_runtime_imports():
            registration = load_file(
                "qwen38_lazy_verify",
                ROOT / "vllm_fl/patches/qwen3_8_flash_next.py",
                package=True,
            )
            text = NS(
                hc_count=4,
                ple_layer_ids=[2],
                indexer_n_heads=4,
                mamba_ssm_dtype="float32",
                rope_parameters={},
            )
            config = NS(
                speculative_config=None,
                model_config=NS(
                    hf_config=text,
                    hf_text_config=text,
                    dtype=torch.bfloat16,
                    multimodal_config=None,
                ),
                cache_config=NS(
                    mamba_ssm_cache_dtype="float32",
                    mamba_cache_mode="none",
                    enable_prefix_caching=False,
                ),
                parallel_config=NS(
                    enable_dbo=False, ubatch_size=1, pipeline_parallel_size=1
                ),
                scheduler_config=NS(
                    async_scheduling=False, max_num_batched_tokens=2048
                ),
                additional_config={},
            )
            verify = (
                registration.Qwen3_8FlashNextForCausalLMConfig.verify_and_update_config
            )
            verify(config)
            modules["vllm.envs"].VLLM_USE_V2_MODEL_RUNNER = True
            with self.assertRaisesRegex(NotImplementedError, "MRV1"):
                verify(config)
            modules["vllm.envs"].VLLM_USE_V2_MODEL_RUNNER = False
            config.model_config.multimodal_config = NS(language_model_only=False)
            with self.assertRaisesRegex(NotImplementedError, "text-only"):
                verify(config)


if __name__ == "__main__":
    unittest.main()


@pytest.mark.parametrize(
    "k,async_mode,lookup,mbt,valid",
    [
        (0, True, "uva_auto", 8192, True),
        (1, False, "shared_buffer", 2048, True),
        (2, False, "shared_buffer", 2048, True),
        (3, False, "shared_buffer", 2048, True),
        (1, True, "shared_buffer", 2048, False),
        (1, False, "uva_auto", 2048, False),
        (1, False, "shared_buffer", 8192, False),
        (0, True, "uva_auto", 16384, False),
    ],
)
def test_native_serving_configuration_guards(k, async_mode, lookup, mbt, valid):
    modules, _, _ = registration_modules()
    with patch.dict(sys.modules, modules):
        model = load_file(
            "qwen_shell_native_verifier", ROOT / "vllm_fl/patches/qwen3_8_flash_next.py"
        )
        config = NS(
            model_config=NS(
                dtype=torch.bfloat16, multimodal_config=NS(language_model_only=True)
            ),
            cache_config=NS(
                mamba_ssm_cache_dtype="float32",
                mamba_cache_mode="none",
                enable_prefix_caching=False,
            ),
            speculative_config=NS(
                num_speculative_tokens=k, disable_padded_drafter_batch=True
            )
            if k
            else None,
            scheduler_config=NS(
                async_scheduling=async_mode, max_num_batched_tokens=mbt
            ),
            parallel_config=NS(pipeline_parallel_size=1),
            additional_config={
                "qwen38_ngram_mode": lookup,
                "qwen38_tuning_profile": True,
            },
        )
        if valid:
            model._verify_metax_config(config)
        else:
            with pytest.raises((ValueError, NotImplementedError)):
                model._verify_metax_config(config)
