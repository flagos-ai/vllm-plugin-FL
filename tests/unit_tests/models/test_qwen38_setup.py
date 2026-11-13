# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# Copyright 2026 FlagOS Contributors
"Native serving, scoped startup, rollback and optional logging contracts."

import ast
import copy as installation_copy
import importlib.util
import json
import logging
import os
import subprocess
import sys
from contextvars import ContextVar
from functools import wraps
from pathlib import Path
from types import ModuleType, SimpleNamespace, SimpleNamespace as NS

import pytest
import torch
import yaml
from _qwen38_cpu import ROOT


def serving_config(k=0, async_mode=True, lookup="uva_auto", mbt=8192):
    return NS(
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
        scheduler_config=NS(async_scheduling=async_mode, max_num_batched_tokens=mbt),
        parallel_config=NS(pipeline_parallel_size=1),
        additional_config={"qwen38_ngram_mode": lookup, "qwen38_tuning_profile": True},
    )


@pytest.fixture
def native_verifier(monkeypatch):
    from _qwen38_cpu import declarations

    from vllm import envs

    monkeypatch.setattr(envs, "VLLM_USE_V2_MODEL_RUNNER", False)
    return declarations(
        ROOT / "vllm_fl/patches/qwen3_8_flash_next.py",
        "qwen_native_config_contract",
        names={"_verify_metax_config"},
    )._verify_metax_config


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
def test_native_serving_configuration_guards(
    native_verifier, k, async_mode, lookup, mbt, valid
):
    cfg = serving_config(k, async_mode, lookup, mbt)
    if valid:
        native_verifier(cfg)
    else:
        with pytest.raises((ValueError, NotImplementedError)):
            native_verifier(cfg)


@pytest.mark.parametrize("case", ["mrv2", "visual", "dtype", "drafts"])
def test_native_serving_rejects_unsupported_modes(native_verifier, monkeypatch, case):
    from vllm import envs

    cfg = serving_config()
    if case == "mrv2":
        monkeypatch.setattr(envs, "VLLM_USE_V2_MODEL_RUNNER", True)
    elif case == "visual":
        cfg.model_config.multimodal_config.language_model_only = False
    elif case == "dtype":
        cfg.model_config.dtype = torch.float32
    else:
        cfg.speculative_config = NS(
            num_speculative_tokens=4, disable_padded_drafter_batch=True
        )
    with pytest.raises((ValueError, NotImplementedError)):
        native_verifier(cfg)


@pytest.mark.parametrize(
    "qwen,method,expected",
    [
        (False, "mtp", ["original"]),
        (True, None, ["original"]),
        (True, "mtp", ["install", "original"]),
    ],
)
def test_worker_constructor_is_scoped_and_installs_before_native_init(
    monkeypatch, qwen, method, expected
):
    from _qwen38_cpu import worker

    events = []
    modules = {
        name: ModuleType(name)
        for name in (
            "vllm_fl",
            "vllm_fl.patches",
            "vllm_fl.patches.qwen3_8_flash_next_mtp",
        )
    }
    modules[
        "vllm_fl.patches.qwen3_8_flash_next_mtp"
    ].install_qwen38_mtp_runtime_compat = lambda: events.append("install")
    for name, module in modules.items():
        monkeypatch.setitem(sys.modules, name, module)
    monkeypatch.setattr(worker, "is_metax", lambda: True)

    class Runner:
        def __init__(self, vllm_config):
            events.append("original")
            self.model_config = vllm_config.model_config

        def _init_model_kwargs(self):
            return {"original": 7}

        def _dummy_run(self, n):
            return n

        def _build_attention_metadata(self, **kwargs):
            return kwargs

    worker.install_qwen38_runner_hooks(Runner)
    installed = Runner.__init__
    worker.install_qwen38_runner_hooks(Runner)
    assert Runner.__init__ is installed
    cfg = NS(
        model_config=NS(
            hf_text_config=NS(supports_qwen38_ngram_context=qwen, ple_layer_ids=[])
        ),
        speculative_config=NS(method=method) if method else None,
    )
    runner = Runner(vllm_config=cfg)
    assert events == expected
    assert runner._init_model_kwargs() == {"original": 7}
    assert runner._dummy_run(3) == 3
    assert runner._build_attention_metadata(x=2) == {"x": 2}


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


serving_ENTRY = ROOT / "examples/qwen3_8_flash_next/serve_metax.sh"


def serving_launch(tmp_path, mode="off", extra=(), missing=None):
    spy = tmp_path / "native vllm"
    spy.write_text(r"""#!/bin/sh
printf '%s\0' "$@" > "$ARG_LOG"
printf '%s\n' "$USE_FLAGGEMS|$VLLM_FL_USE_FLAGGEMS_VLLM|$TRITON_CACHE_DIR|$PYTHONPATH|$OMP_NUM_THREADS|$FLAGGEMS_DB_URL" > "$ENV_LOG"
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
        FLAGGEMS_DB_URL="sqlite:///:memory:",
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
    assert fields[3] == "/existing packages"
    assert fields[4] == "7"
    assert fields[5] == "sqlite:///:memory:"


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


spec = importlib.util.spec_from_file_location(
    "fg_logging_policy", ROOT / "vllm_fl/worker/flaggems_logging.py"
)


policy = importlib.util.module_from_spec(spec)


spec.loader.exec_module(policy)


@pytest.fixture
def fg_logger():
    log = logging.getLogger("flag_gems")
    previous = log.level
    yield log
    log.setLevel(previous)


@pytest.mark.parametrize("rank", [0, 1, 7])
def test_legacy_and_explicit_recording(rank, fg_logger):
    fg_logger.setLevel(logging.DEBUG)
    assert policy.configure_oplist_recording({}, rank) == (rank == 0)
    assert policy.configure_oplist_recording({"flaggems_record_ops": True}, rank) == (
        rank == 0
    )
    assert fg_logger.level == logging.DEBUG


@pytest.mark.parametrize(
    "level", [logging.NOTSET, logging.DEBUG, logging.INFO, logging.WARNING]
)
def test_disabled_skips_construction_and_preserves_stricter_levels(
    level, fg_logger, monkeypatch
):
    root = logging.getLogger()
    monkeypatch.setattr(root, "level", logging.DEBUG)
    fg_logger.setLevel(level)
    calls = []
    original = fg_logger.makeRecord

    def counted(*args, **kwargs):
        calls.append(1)
        return original(*args, **kwargs)

    monkeypatch.setattr(fg_logger, "makeRecord", counted)
    for rank in [0, 1, 7]:
        assert (
            policy.configure_oplist_recording({"flaggems_record_ops": False}, rank)
            is False
        )
        for _ in range(100):
            fg_logger.debug("probe")
    assert calls == []
    assert fg_logger.level == max(logging.INFO, level)
    assert fg_logger.isEnabledFor(logging.WARNING)


@pytest.mark.parametrize("value", ["false", 0, 1, None])
def test_invalid_value_rejected(value, fg_logger):
    with pytest.raises(ValueError):
        policy.configure_oplist_recording({"flaggems_record_ops": value}, 0)
