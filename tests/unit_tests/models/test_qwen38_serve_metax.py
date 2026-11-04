# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0
"""Test the POSIX entry and native configuration without starting a service."""

import json
import os
import subprocess
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[3]
ENTRY = ROOT / "examples/qwen3_8_flash_next/serve_metax.sh"


def launch(tmp_path, mode="off", extra=(), missing=None):
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
        ["sh", str(ENTRY), *extra], env=env, capture_output=True, text=True
    )
    args = (
        (tmp_path / "argv").read_bytes().decode().split("\0")[:-1]
        if (tmp_path / "argv").exists()
        else []
    )
    return result, args, tmp_path / "env"


def effective(args):
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
    result, args, env_file = launch(tmp_path, mode)
    assert result.returncode == 0, result.stderr
    assert args[:2] == ["serve", "/checkpoint with spaces"]
    config = effective(args)
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
    result, args, _ = launch(tmp_path, mode, extra)
    assert result.returncode == 0
    assert args[-len(extra) :] == list(extra)


@pytest.mark.parametrize("mode", ["k0", "k4", "invalid"])
def test_invalid_mode_does_not_execute(tmp_path, mode):
    result, args, _ = launch(tmp_path, mode)
    assert result.returncode != 0 and not args
    assert not (tmp_path / "cache root").exists()


@pytest.mark.parametrize("name", ["MODEL_PATH", "QWEN38_CACHE_ROOT"])
def test_required_paths_fail_before_native_execution(tmp_path, name):
    result, args, _ = launch(tmp_path, missing=name)
    assert result.returncode != 0 and not args


def test_shell_is_posix_and_has_no_custom_interpreter_layer():
    result = subprocess.run(["sh", "-n", str(ENTRY)], capture_output=True, text=True)
    assert result.returncode == 0
    text = ENTRY.read_text()
    assert text.startswith("#!/bin/sh")
    assert not (ENTRY.parent / "serve_metax.py").exists()
    assert "python " not in text.lower() and "serve_metax.py" not in text
    assert "exec " in text and " serve " in text
