# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Qwen3.8 integration and contract regressions."""

import ast
import builtins as registration_builtins
import copy as installation_copy
import importlib.util
import json
import os
import subprocess
import sys
import unittest
from contextvars import ContextVar
from functools import wraps
from pathlib import Path
from types import ModuleType, SimpleNamespace, SimpleNamespace as NS
from unittest.mock import patch

import pytest
import torch
import yaml
from _qwen38_cpu import ROOT

# Registration contracts.


def registration_load_file(name, path, package=False):
    options = {"submodule_search_locations": [str(path.parent)]} if package else {}
    spec = importlib.util.spec_from_file_location(name, path, **options)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def registration_forbid_runtime_imports():
    original = registration_builtins.__import__

    def importing(name, *args, **kwargs):
        if name.startswith("vllm.v1"):
            raise AssertionError(f"unrequested runtime import: {name}")
        return original(name, *args, **kwargs)

    return patch("builtins.__import__", importing)


def registration_registration_modules():
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
    modules["vllm_fl.patches._qwen38_installation"] = registration_load_file(
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


class registration_LazyRegistrationTest(unittest.TestCase):
    def test_public_registration_does_not_import_proposer_or_mrv2(self):
        modules, Spec, Registry = registration_registration_modules()
        with patch.dict(sys.modules, modules), registration_forbid_runtime_imports():
            compat = registration_load_file(
                "qwen38_lazy_compat", ROOT / "vllm_fl/patches/qwen3_8_flash_next_mtp.py"
            )
            sys.modules["qwen38_lazy_registration.qwen3_8_flash_next_mtp"] = compat
            registration = registration_load_file(
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
        with registration_forbid_runtime_imports():
            worker = registration_load_file(
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
        with (
            patch.dict(sys.modules, fake_modules),
            registration_forbid_runtime_imports(),
        ):
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
        modules, _, _ = registration_registration_modules()
        with patch.dict(sys.modules, modules), registration_forbid_runtime_imports():
            registration = registration_load_file(
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
    modules, _, _ = registration_registration_modules()
    with patch.dict(sys.modules, modules):
        model = registration_load_file(
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


# Installation contracts.

# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0


installation_HELPER = ROOT / "vllm_fl/patches/_qwen38_installation.py"
installation_spec = importlib.util.spec_from_file_location(
    "qwen_install_transaction", installation_HELPER
)
installation_install = importlib.util.module_from_spec(installation_spec)
installation_spec.loader.exec_module(installation_install)


def installation_declaration(name):
    path = (
        ROOT
        / "vllm_fl/dispatch/backends/vendor/metax/impl/qwen3_8_flash_next/launch_profile.py"
    )
    tree = ast.parse(path.read_text())
    node = next(n for n in tree.body if getattr(n, "name", None) == name)
    ns = {"copy": installation_copy.copy}
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(path), "exec"), ns)
    return ns[name]


def test_partial_attribute_install_is_rolled_back_and_retry_succeeds():
    first = SimpleNamespace(value="old")

    class FailingSetter:
        value = "old"

        def __setattr__(self, name, value):
            object.__setattr__(self, name, value)
            if value == "fail":
                raise RuntimeError("injected setter failure")

    last = FailingSetter()
    with pytest.raises(RuntimeError, match="injected"):
        installation_install.apply_attributes(
            [(first, "value", "new"), (first, "added", 1), (last, "value", "fail")]
        )
    assert first.value == last.value == "old"
    assert not hasattr(first, "added")
    installation_install.apply_attributes(
        [(first, "value", "new"), (last, "value", "new")]
    )
    assert first.value == last.value == "new"


def test_registry_sequence_and_staticmethod_descriptor_are_restored():
    mapping, sequence = {"old": 1}, ["old"]

    class Config:
        @staticmethod
        def override(value):
            return value

    original = Config.__dict__["override"]
    with (
        pytest.raises(RuntimeError),
        installation_install.InstallationTransaction() as transaction,
    ):
        transaction.mapping(mapping)
        transaction.sequence(sequence)
        transaction.attribute(Config, "override")
        mapping["new"] = 2
        sequence.append("new")
        Config.override = staticmethod(lambda value: None)
        raise RuntimeError("injected after registry update")
    assert mapping == {"old": 1} and sequence == ["old"]
    assert Config.__dict__["override"] is original
    assert Config.override(4) == 4


def test_preset_overlay_never_writes_pinned_rows_to_native_cache():
    Cache = installation_declaration("_PresetConfigCache")
    native = {"existing": 1}
    cache = Cache(native, {"pinned": 2})
    assert native == {"existing": 1}
    assert cache["pinned"] == 2 and cache["existing"] == 1
    cache["pinned"] = 999
    assert cache["pinned"] == 2 and "pinned" not in native
    cache["new"] = 3
    assert native["new"] == 3
    with pytest.raises(KeyError):
        _ = cache["missing"]


def test_fresh_entry_does_not_mutate_original_dispatch_caches():
    clone = installation_declaration("_copy_entry")
    fn = object()
    original = SimpleNamespace(
        fn=fn, kernel_cache=[{"compiled": object()}], _cpu_cache={"cpu": object()}
    )
    replacement = clone(original, object())
    assert original.fn is fn and original.kernel_cache[0] and original._cpu_cache
    assert replacement.kernel_cache == [{}] and replacement._cpu_cache == {}
    assert replacement.kernel_cache is not original.kernel_cache


def test_scoped_rollback_preserves_other_model_registration():
    registry = {"other": object()}
    unrelated = object()
    ops = ["other_op"]
    with (
        pytest.raises(RuntimeError),
        installation_install.InstallationTransaction() as transaction,
    ):
        transaction.mapping(registry, ("qwen",))
        transaction.sequence_additions(ops, ("qwen_op",))
        registry["qwen"] = object()
        registry["new_other"] = unrelated
        ops.extend(["qwen_op", "new_other_op"])
        raise RuntimeError("failure after another model registered")
    assert "qwen" not in registry and registry["new_other"] is unrelated
    assert ops == ["other_op", "new_other_op"]


def installation_sampler_module(native, monkeypatch):
    path = ROOT / "vllm_fl/dispatch/backends/vendor/metax/patches/topk_topp_sampler.py"
    tree = ast.parse(path.read_text())
    body = [
        n
        for n in tree.body
        if isinstance(n, (ast.FunctionDef, ast.Assign, ast.AnnAssign))
    ]
    namespace = {"ContextVar": ContextVar, "wraps": wraps, "topk_topp_sampler": native}
    monkeypatch.setitem(
        sys.modules, "vllm_fl.patches._qwen38_installation", installation_install
    )
    exec(compile(ast.Module(body=body, type_ignores=[]), str(path), "exec"), namespace)
    return namespace


def test_sampling_group_install_is_idempotent(monkeypatch):
    native = SimpleNamespace(
        apply_top_k_top_p=lambda *a: None,
        random_sample=lambda *a: None,
        sample_with_exponential_noise=lambda *a: None,
    )
    original = native.random_sample
    module = installation_sampler_module(native, monkeypatch)
    assert native.random_sample is original
    module["install_metax_sampling_patches"]()
    first = native.random_sample
    module["install_metax_sampling_patches"]()
    assert native.random_sample is first and first.__wrapped__ is original


def test_sampling_group_rolls_back_partial_publication(monkeypatch):
    class Native(SimpleNamespace):
        fail = False

        def __setattr__(self, name, value):
            super().__setattr__(name, value)
            if (
                self.fail
                and name == "sample_with_exponential_noise"
                and getattr(value, "_fl_metax_sampling_patch", False)
            ):
                self.fail = False
                raise RuntimeError("injected sampler setup failure")

    native = Native(
        apply_top_k_top_p=lambda *a: None,
        random_sample=lambda *a: None,
        sample_with_exponential_noise=lambda *a: None,
    )
    originals = dict(native.__dict__)
    module = installation_sampler_module(native, monkeypatch)
    native.fail = True
    with pytest.raises(RuntimeError, match="injected sampler"):
        module["install_metax_sampling_patches"]()
    assert all(getattr(native, key) is value for key, value in originals.items())
    module["install_metax_sampling_patches"]()
    assert native.random_sample is module["_random_sample"]


# Serving contracts.

# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0


serving_ENTRY = ROOT / "examples/qwen3_8_flash_next/serve_metax.sh"


def serving_launch(tmp_path, mode="off", extra=(), missing=None):
    spy = tmp_path / "native vllm"
    spy.write_text(r"""#!/bin/sh
printf '%s\0' "$@" > "$ARG_LOG"
printf '%s\n' "$USE_FLAGGEMS|$VLLM_FL_USE_FLAGGEMS_VLLM|$TRITON_CACHE_DIR|$PYTHONPATH|$OMP_NUM_THREADS" > "$ENV_LOG"
""")
    spy.chmod(0o755)
    env = dict(
        os.environ,
        MODEL_PATH="/checkpoint with spaces",
        QWEN38_CACHE_ROOT=str(tmp_path / "cache root"),
        MTP=mode,
        VLLM_BIN=str(spy),
        ARG_LOG=str(tmp_path / "argv"),
        ENV_LOG=str(tmp_path / "env"),
        PYTHONPATH="/existing packages",
        OMP_NUM_THREADS="7",
        FLAGCX_PATH="/unused",
    )
    if missing:
        env.pop(missing, None)
    result = subprocess.run(
        ["sh", str(serving_ENTRY), *extra], env=env, capture_output=True, text=True
    )
    args = (
        (tmp_path / "argv").read_bytes().decode().split("\0")[:-1]
        if (tmp_path / "argv").exists()
        else []
    )
    return result, args, tmp_path / "env"


def serving_effective(args):
    config = yaml.safe_load(Path(args[args.index("--config") + 1]).read_text())
    rest = args[args.index("--config") + 2 :]
    i = 0
    while i < len(rest):
        flag = rest[i]
        if flag.startswith("--no-"):
            config[flag.removeprefix("--no-")] = False
            i += 1
        else:
            value = rest[i + 1]
            config[flag.removeprefix("--")] = (
                json.loads(value)
                if flag in ("--compilation-config", "--speculative-config")
                else value
            )
            i += 2
    return config


@pytest.mark.parametrize("mode,k", [("off", 0), ("k1", 1), ("k2", 2), ("k3", 3)])
def test_direct_native_entry_preserves_mode_contract(tmp_path, mode, k):
    result, args, env_file = serving_launch(tmp_path, mode)
    assert result.returncode == 0, result.stderr
    assert args[:2] == ["serve", "/checkpoint with spaces"]
    config = serving_effective(args)
    assert config["dtype"] == "bfloat16" and config["tensor-parallel-size"] == 8
    assert config["pipeline-parallel-size"] == 1
    assert config["mamba-ssm-cache-dtype"] == "float32"
    assert config["generation-config"] == "auto"
    assert "seed" not in config and "override-generation-config" not in config
    assert config["max-num-seqs"] == 64 and config["max-model-len"] == 69632
    assert not config["enable-prefix-caching"] and config["enable-chunked-prefill"]
    assert config["max-num-batched-tokens"] == (2048 if k else 8192)
    assert config["async-scheduling"] is (not bool(k))
    extra = config["additional-config"]
    assert extra["qwen38_ngram_mode"] == ("shared_buffer" if k else "uva_auto")
    assert extra["qwen38_mtp_graph"] is bool(k)
    for key in (
        "qwen38_qsa_native_attention",
        "qwen38_qsa_packing",
        "qwen38_fused_qsa_metadata",
        "qwen38_hc_small_fusions",
        "qwen38_reuse_gdn_metadata",
        "qwen38_ple_cpu_metadata",
    ):
        assert extra[key] is (not bool(k))
    compilation = config["compilation-config"]
    assert (
        compilation["mode"] == 0 and compilation["cudagraph_mode"] == "FULL_DECODE_ONLY"
    )
    assert compilation["cudagraph_capture_sizes"] == [
        (k + 1) * n for n in (1, 2, 4, 8, 16, 32, 64)
    ]
    assert compilation["max_cudagraph_capture_size"] == (k + 1) * 64
    if k:
        spec = config["speculative-config"]
        assert spec["method"] == "mtp" and spec["num_speculative_tokens"] == k
        assert spec["disable_padded_drafter_batch"] and not spec["parallel_drafting"]
    else:
        assert "speculative-config" not in config
    fields = env_file.read_text().strip().split("|")
    assert fields[:2] == ["1", "1"]
    assert fields[2].endswith(mode + "/triton")
    assert fields[3] == str(ROOT) + ":/existing packages"
    assert fields[4] == "7"


@pytest.mark.parametrize("mode", ["off", "k1", "k2", "k3"])
def test_native_arguments_are_passed_without_reparsing(tmp_path, mode):
    extra = ("--port", "18480", "--served-model-name", "name with spaces")
    result, args, _ = serving_launch(tmp_path, mode, extra)
    assert result.returncode == 0
    assert args[-len(extra) :] == list(extra)


@pytest.mark.parametrize("mode", ["k0", "k4", "invalid"])
def test_invalid_mode_does_not_execute(tmp_path, mode):
    result, args, _ = serving_launch(tmp_path, mode)
    assert result.returncode != 0 and not args
    assert not (tmp_path / "cache root").exists()


@pytest.mark.parametrize("name", ["MODEL_PATH", "QWEN38_CACHE_ROOT"])
def test_required_paths_fail_before_native_execution(tmp_path, name):
    result, args, _ = serving_launch(tmp_path, missing=name)
    assert result.returncode != 0 and not args


def test_shell_is_posix_and_has_no_custom_interpreter_layer():
    result = subprocess.run(
        ["sh", "-n", str(serving_ENTRY)], capture_output=True, text=True
    )
    assert result.returncode == 0
    text = serving_ENTRY.read_text()
    assert text.startswith("#!/bin/sh")
    assert not (serving_ENTRY.parent / "serve_metax.py").exists()
    assert "python " not in text.lower() and "serve_metax.py" not in text
    assert "exec " in text and " serve " in text
