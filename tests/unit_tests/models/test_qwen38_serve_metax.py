"""Portable CPU contracts for the native Qwen3.8 MetaX serving example.

The execute tests intercept execvpe before any server starts. The optional
native integration runs in an isolated CPU subprocess with a tiny local
generation-config fixture; no checkpoint, HTTP service or accelerator is used.
"""

import builtins
import importlib.util
import json
import os
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[3]
LAUNCHER = ROOT / "examples/qwen3_8_flash_next/serve_metax.py"


@pytest.fixture
def launcher():
    spec = importlib.util.spec_from_file_location("qwen38_serve_contract", LAUNCHER)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def launch_paths(tmp_path, monkeypatch):
    fg = tmp_path / "fg" / "flag_gems"
    fgv = tmp_path / "fgv" / "flaggems_vllm"
    for package in (fg, fgv):
        package.mkdir(parents=True)
        (package / "__init__.py").write_text("")
    cache = tmp_path / "cache"
    monkeypatch.setenv("QWEN38_FLAGGEMS_ROOT", str(fg))
    monkeypatch.setenv("QWEN38_FGV_ROOT", str(fgv))
    monkeypatch.setenv("QWEN38_CACHE_ROOT", str(cache))
    monkeypatch.setenv("MODEL_PATH", "/checkpoint")
    return fg, fgv, cache


@pytest.mark.parametrize("mode,k", (("off", 0), ("k1", 1), ("k2", 2), ("k3", 3)))
def test_mtp_configuration_contract(launcher, mode, k):
    config = launcher.build_config("/checkpoint", mode)
    assert config["generation_config"] == "auto"
    assert "override_generation_config" not in config
    assert "worker_extension_cls" not in config
    assert config["tensor_parallel_size"] == 8
    assert config["pipeline_parallel_size"] == 1
    assert config["dtype"] == "bfloat16"
    assert config["max_num_seqs"] == 64
    assert config["max_num_batched_tokens"] == 2048
    assert config["max_model_len"] == 69632
    assert config["gpu_memory_utilization"] == 0.92
    assert config["mamba_ssm_cache_dtype"] == "float32"
    assert config["enable_prefix_caching"] is False
    assert config["async_scheduling"] is False
    assert config["additional_config"] == {
        "qwen38_ngram_mode": "shared_buffer",
        "qwen38_mtp_graph": bool(k),
        "qwen38_qsa_length_buckets": True,
        "qwen38_tuning_profile": True,
    }
    buckets = [(k + 1) * value for value in (1, 2, 4, 8, 16, 32, 64)]
    assert config["compilation_config"] == {
        "mode": 0,
        "cudagraph_mode": "FULL_DECODE_ONLY",
        "cudagraph_capture_sizes": buckets,
        "max_cudagraph_capture_size": buckets[-1],
    }
    if k:
        assert config["speculative_config"] == {
            "method": "mtp",
            "num_speculative_tokens": k,
            "parallel_drafting": False,
            "use_local_argmax_reduction": False,
            "disable_padded_drafter_batch": True,
        }
    else:
        assert "speculative_config" not in config


def test_default_dry_run_has_no_backend_import_or_cache_write(
    launcher, launch_paths, monkeypatch, capsys
):
    _, _, cache = launch_paths
    monkeypatch.setattr(sys, "argv", [str(LAUNCHER)])
    original_import = builtins.__import__

    def guarded_import(name, *args, **kwargs):
        assert name.split(".")[0] not in {
            "torch",
            "vllm",
            "vllm_fl",
            "flag_gems",
            "flaggems_vllm",
        }
        return original_import(name, *args, **kwargs)

    def unexpected_exec(*args, **kwargs):
        raise AssertionError("Dry run attempted to start a process")

    monkeypatch.setattr(builtins, "__import__", guarded_import)
    monkeypatch.setattr(launcher.os, "execvpe", unexpected_exec)
    launcher.main()
    record, _ = json.JSONDecoder().raw_decode(capsys.readouterr().out)
    assert record["mtp"] == "off"
    assert record["execute"] is False
    assert "speculative_config" not in record["config"]
    assert not cache.exists()


class _ExecIntercepted(Exception):
    pass


@pytest.mark.parametrize("mode,k", ((None, 0), ("k1", 1), ("k2", 2), ("k3", 3)))
def test_execute_passes_effective_environment_to_native_cli(
    launcher, launch_paths, tmp_path, monkeypatch, mode, k
):
    fg, fgv, cache = launch_paths
    old_log = tmp_path / "historical-enabled-ops.txt"
    old_log.write_text("historical evidence\n")
    for key in launcher.UNSET_ENV:
        monkeypatch.setenv(key, "stale")
    monkeypatch.setenv("VLLM_FL_PREFER", "vendor")
    monkeypatch.setenv("VLLM_FL_OOT_ENABLED", "0")
    monkeypatch.setenv("FLAGGEMS_ENABLE_OPLIST_PATH", str(old_log))
    monkeypatch.setenv("PYTHONPATH", "/retained/path")
    original_visibility = {
        key: os.environ.get(key)
        for key in ("CUDA_VISIBLE_DEVICES", "MACA_VISIBLE_DEVICES")
    }
    argv = [str(LAUNCHER), "--execute", "--vllm-bin", "/native/vllm"]
    if mode:
        argv.extend(("--mtp", mode))
    monkeypatch.setattr(sys, "argv", argv)
    captured = {}

    def intercept(program, command, environment):
        captured.update(program=program, command=command, environment=environment)
        raise _ExecIntercepted

    monkeypatch.setattr(launcher.os, "execvpe", intercept)
    with pytest.raises(_ExecIntercepted):
        launcher.main()
    command = captured["command"]
    env = captured["environment"]
    effective_mode = mode or "off"
    assert captured["program"] == "/native/vllm"
    assert command[:3] == ["/native/vllm", "serve", "/checkpoint"]
    assert command[command.index("--generation-config") + 1] == "auto"
    assert "--override-generation-config" not in command
    assert "--worker-extension-cls" not in command
    if k:
        spec = json.loads(command[command.index("--speculative-config") + 1])
        assert spec["num_speculative_tokens"] == k
    else:
        assert "--speculative-config" not in command
    assert all(key not in env for key in launcher.UNSET_ENV)
    assert env["VLLM_FL_PREFER"] == "flagos"
    assert env["VLLM_FL_OOT_ENABLED"] == "1"
    assert env["USE_FLAGGEMS"] == env["VLLM_FL_USE_FLAGGEMS_VLLM"] == "1"
    assert env["PYTHONPATH"].split(os.pathsep)[:3] == [
        str(ROOT),
        str(fg.parent),
        str(fgv.parent),
    ]
    assert "/retained/path" in env["PYTHONPATH"].split(os.pathsep)
    assert env["VLLM_FL_CONFIG"] == str(
        ROOT / "vllm_fl/dispatch/config/qwen3_8_flash_next_metax.yaml"
    )
    assert env["FLAGGEMS_ENABLE_OPLIST_PATH"] == str(
        cache / effective_mode / "enabled-ops.txt"
    )
    assert env["TRITON_CACHE_DIR"] == str(cache / effective_mode / "triton")
    assert (cache / effective_mode / "triton").is_dir()
    assert old_log.read_text() == "historical evidence\n"
    assert {key: env.get(key) for key in original_visibility} == original_visibility


def test_plan_output_never_overwrites_existing_evidence(
    launcher, launch_paths, tmp_path, monkeypatch
):
    output = tmp_path / "plan"
    monkeypatch.setattr(sys, "argv", [str(LAUNCHER), "--output-dir", str(output)])
    launcher.main()
    before = {path.name: path.read_bytes() for path in output.iterdir()}
    assert set(before) == {
        "server-config.json",
        "command.json",
        "environment.json",
        "launch-plan.json",
    }
    with pytest.raises(FileExistsError):
        launcher.main()
    assert {path.name: path.read_bytes() for path in output.iterdir()} == before


def test_missing_operator_package_fails_before_execution(
    launcher, launch_paths, monkeypatch, tmp_path
):
    monkeypatch.setenv("QWEN38_FLAGGEMS_ROOT", str(tmp_path / "missing" / "flag_gems"))
    monkeypatch.setattr(sys, "argv", [str(LAUNCHER), "--execute"])
    calls = []
    monkeypatch.setattr(launcher.os, "execvpe", lambda *args: calls.append(args))
    with pytest.raises(FileNotFoundError):
        launcher.main()
    assert calls == []


def test_native_parser_and_checkpoint_sampling_in_cpu_subprocess(tmp_path):
    if importlib.util.find_spec("vllm") is None:
        pytest.skip("Native vLLM is needed for parser/sampling integration")
    model = tmp_path / "checkpoint"
    model.mkdir()
    (model / "generation_config.json").write_text(
        json.dumps({"do_sample": True, "temperature": 1.0, "top_p": 0.95, "top_k": 20})
    )
    code = textwrap.dedent(
        """
        import importlib.util
        import json
        import sys
        import torch
        assert not torch.cuda.is_initialized()
        def no_device_init(*args, **kwargs):
            raise AssertionError("GPU initialization is forbidden")
        torch.cuda._lazy_init = no_device_init
        torch.cuda.init = no_device_init

        import vllm.platforms as platforms
        from vllm.platforms.cpu import CpuPlatform
        platforms._current_platform = CpuPlatform()
        from vllm.config.model import ModelConfig
        from vllm.entrypoints.openai.cli_args import make_arg_parser
        from vllm.entrypoints.openai.chat_completion.protocol import ChatCompletionRequest
        from vllm.entrypoints.openai.completion.protocol import CompletionRequest
        from vllm.sampling_params import SamplingType
        from vllm.utils.argparse_utils import FlexibleArgumentParser

        spec = importlib.util.spec_from_file_location("launcher", sys.argv[1])
        launcher = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(launcher)
        parser = make_arg_parser(FlexibleArgumentParser())
        for mode, count in launcher.MTP_COUNTS.items():
            config = launcher.build_config(sys.argv[2], mode)
            command = launcher.native_command(
                config, host="127.0.0.1", port=18170,
                served_model_name="qwen38", vllm_bin="vllm")
            parsed = parser.parse_args(command[2:])
            assert parsed.generation_config == "auto"
            assert parsed.override_generation_config == {}
            assert parsed.additional_config["qwen38_mtp_graph"] == bool(count)
            if count:
                assert parsed.speculative_config["num_speculative_tokens"] == count
            else:
                assert parsed.speculative_config is None

        config = object.__new__(ModelConfig)
        for key, value in {
            "model": sys.argv[2], "hf_config_path": None,
            "generation_config": "auto", "override_generation_config": {},
            "trust_remote_code": False, "revision": None,
            "config_format": "auto", "hf_token": None,
        }.items():
            setattr(config, key, value)
        defaults = config.get_diff_sampling_param()
        cases = [
            (ChatCompletionRequest, {"model": "qwen38", "messages": [{"role": "user", "content": "hi"}]}),
            (CompletionRequest, {"model": "qwen38", "prompt": "hi"}),
        ]
        rows = []
        for Request, payload in cases:
            request = Request(**payload)
            params = request.to_sampling_params(max_tokens=32, default_sampling_params=defaults)
            assert (params.temperature, params.top_p, params.top_k) == (1.0, 0.95, 20)
            assert params.seed is None and not params.ignore_eos
            assert params.sampling_type is SamplingType.RANDOM
            if isinstance(request, ChatCompletionRequest):
                assert request.chat_template_kwargs is None
            override = Request(**payload, temperature=0.7, top_p=0.8, top_k=7, seed=123)
            explicit = override.to_sampling_params(max_tokens=32, default_sampling_params=defaults)
            assert (explicit.temperature, explicit.top_p, explicit.top_k, explicit.seed) == (0.7, 0.8, 7, 123)
            assert explicit.sampling_type is SamplingType.RANDOM_SEED
            rows.append(params.sampling_type.name)
        assert not torch.cuda.is_initialized()
        print(json.dumps({"modes": list(launcher.MTP_COUNTS), "sampling": rows, "cuda_initialized": False}))
        """
    )
    environment = dict(os.environ)
    environment.update(
        CUDA_VISIBLE_DEVICES="",
        MACA_VISIBLE_DEVICES="",
        VLLM_PLUGINS="",
        HF_HUB_OFFLINE="1",
        TRANSFORMERS_OFFLINE="1",
        PYTHONDONTWRITEBYTECODE="1",
    )
    completed = subprocess.run(
        [sys.executable, "-c", code, str(LAUNCHER), str(model)],
        env=environment,
        text=True,
        capture_output=True,
        check=False,
        timeout=120,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr
    assert json.loads(completed.stdout.splitlines()[-1]) == {
        "modes": ["off", "k1", "k2", "k3"],
        "sampling": ["RANDOM", "RANDOM"],
        "cuda_initialized": False,
    }
