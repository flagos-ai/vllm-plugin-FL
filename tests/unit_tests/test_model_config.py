# Copyright (c) 2025 BAAI. All rights reserved.

from __future__ import annotations

import json
import textwrap

import pytest

from tests.utils import platform_config
from tests.utils.model_config import ModelConfig


def _write(path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(textwrap.dedent(content), encoding="utf-8")


def test_load_applies_device_case_overrides(tmp_path, monkeypatch):
    models_dir = tmp_path / "models"
    platforms_dir = tmp_path / "platforms"
    monkeypatch.setattr(platform_config, "_PLATFORMS_DIR", platforms_dir)

    _write(
        models_dir / "qwen3_6" / "27b_tp2_eager.yaml",
        """
        llm:
          model: "/data/models/Qwen/Qwen3.6-27B"
          tensor_parallel_size: 2
          max_model_len: 8192
          gpu_memory_utilization: 0.95
          trust_remote_code: false
          disable_custom_all_reduce: true
        generate:
          prompts:
            - "Where is the capital of France"
          sampling:
            max_tokens: 100
            temperature: 0.0
        serve:
          endpoints: ["chat"]
          max_tokens: 256
        """,
    )
    _write(
        platforms_dir / "demo.yaml",
        """
        platform: demo
        vendor: test
        device_types:
          demo_card:
            memory_gb: 80
        tolerance: {}
        device_overrides:
          demo_card:
            cases:
              - qwen3_6/27b_tp2_eager
            llm:
              max_model_len: 4096
              gpu_memory_utilization: 0.8
              trust_remote_code: true
              disable_custom_all_reduce: null
            serve:
              startup_retries: 180
              max_tokens: 1024
        demo_card:
          name: "demo_card"
          tests: {}
        """,
    )

    cfg = ModelConfig.load(
        "qwen3_6",
        "27b_tp2_eager",
        models_dir=models_dir,
        platform="demo",
        device="demo_card",
    )

    assert cfg.engine["tensor_parallel_size"] == 2
    assert cfg.engine["max_model_len"] == 4096
    assert cfg.engine["gpu_memory_utilization"] == 0.8
    assert cfg.engine["trust_remote_code"] is True
    assert "disable_custom_all_reduce" not in cfg.engine
    assert cfg.generate.prompts == ["Where is the capital of France"]
    assert cfg.serve.startup_retries == 180
    assert cfg.serve.max_tokens == 1024


def test_load_without_matching_device_case_override_uses_base_config(
    tmp_path, monkeypatch
):
    models_dir = tmp_path / "models"
    platforms_dir = tmp_path / "platforms"
    monkeypatch.setattr(platform_config, "_PLATFORMS_DIR", platforms_dir)

    _write(
        models_dir / "qwen3_6" / "27b_tp2_eager.yaml",
        """
        llm:
          model: "/data/models/Qwen/Qwen3.6-27B"
          tensor_parallel_size: 2
          gpu_memory_utilization: 0.95
        generate:
          prompts:
            - "Where is the capital of France"
          sampling:
            max_tokens: 100
        """,
    )
    _write(
        platforms_dir / "demo.yaml",
        """
        platform: demo
        vendor: test
        device_types:
          demo_card:
            memory_gb: 80
        tolerance: {}
        device_overrides:
          demo_card:
            cases:
              - qwen3_6/other_case
            llm:
              gpu_memory_utilization: 0.8
        demo_card:
          name: "demo_card"
          tests: {}
        """,
    )

    cfg = ModelConfig.load(
        "qwen3_6",
        "27b_tp2_eager",
        models_dir=models_dir,
        platform="demo",
        device="demo_card",
    )

    assert cfg.engine["tensor_parallel_size"] == 2
    assert cfg.engine["gpu_memory_utilization"] == 0.95


@pytest.mark.parametrize(
    "case",
    ["27b_tp2_eager", "27b_tp2_graph", "35b_a3b_tp2_eager", "35b_a3b_tp2_graph"],
)
def test_enflame_s60_compilation_config_reaches_engine_and_cli(case, monkeypatch):
    monkeypatch.delenv("FL_TEST_PLATFORM", raising=False)
    monkeypatch.delenv("FL_TEST_DEVICE", raising=False)
    cfg = ModelConfig.load("qwen3_6", case, platform="enflame", device="s60")
    expected = {
        "custom_ops": ["none", "+SiluAndMul"],
        "cudagraph_capture_sizes": [1, 2, 4, 8, 16, 32],
    }
    assert cfg.engine_kwargs()["compilation_config"] == expected
    assert cfg.engine_kwargs()["max_num_batched_tokens"] == 4096
    assert cfg.engine_kwargs()["max_model_len"] == 8192
    args = cfg.serve_args()
    assert json.loads(args[args.index("--compilation-config") + 1]) == expected
    assert args[args.index("--max-num-batched-tokens") + 1] == "4096"
    baseline = ModelConfig.load("qwen3_6", case)
    assert "compilation_config" not in baseline.engine_kwargs()
    assert "max_num_batched_tokens" not in baseline.engine_kwargs()


@pytest.mark.parametrize(
    "case",
    ["27b_tp2_eager", "27b_tp2_graph", "35b_a3b_tp2_eager", "35b_a3b_tp2_graph"],
)
def test_enflame_s60_compilation_config_inherits_runner_environment(case, monkeypatch):
    monkeypatch.setenv("FL_TEST_PLATFORM", "enflame")
    monkeypatch.setenv("FL_TEST_DEVICE", "s60")
    inherited = ModelConfig.load("qwen3_6", case)
    explicit = ModelConfig.load("qwen3_6", case, platform="enflame", device="s60")
    assert inherited.engine_kwargs() == explicit.engine_kwargs()
    assert inherited.engine_kwargs()["compilation_config"] == {
        "custom_ops": ["none", "+SiluAndMul"],
        "cudagraph_capture_sizes": [1, 2, 4, 8, 16, 32],
    }
    assert inherited.serve_args() == explicit.serve_args()


@pytest.mark.parametrize(
    "case",
    ["27b_tp4_eager", "27b_tp4_graph", "35b_a3b_tp4_eager", "35b_a3b_tp4_graph"],
)
@pytest.mark.parametrize("inherit_environment", [False, True])
def test_thead_graph_capture_sizes_reach_engine_and_cli(
    case, inherit_environment, monkeypatch
):
    monkeypatch.delenv("FL_TEST_PLATFORM", raising=False)
    monkeypatch.delenv("FL_TEST_DEVICE", raising=False)
    if inherit_environment:
        monkeypatch.setenv("FL_TEST_PLATFORM", "thead")
        monkeypatch.setenv("FL_TEST_DEVICE", "zw810e")
        cfg = ModelConfig.load("qwen3_6", case)
    else:
        cfg = ModelConfig.load("qwen3_6", case, platform="thead", device="zw810e")
    expected = {
        "cudagraph_capture_sizes": [1, 2, 4, 8, 16, 32, 64, 128, 256],
        "max_cudagraph_capture_size": 256,
    }
    assert cfg.engine_kwargs()["compilation_config"] == expected
    assert cfg.engine_kwargs()["enforce_eager"] == case.endswith("eager")
    args = cfg.serve_args()
    assert json.loads(args[args.index("--compilation-config") + 1]) == expected
    assert ("--enforce-eager" in args) == case.endswith("eager")


def test_serve_args_serializes_nested_json(tmp_path):
    _write(
        tmp_path / "nested.yaml",
        """
        llm:
          model: test-model
          compilation_config:
            custom_ops: ["none", "+SiluAndMul"]
            nested:
              enabled: true
              optional: null
              label: 'a "quoted" value'
        """,
    )
    cfg = ModelConfig.load("nested", models_dir=tmp_path)
    args = cfg.serve_args()
    assert (
        json.loads(args[args.index("--compilation-config") + 1])
        == (cfg.engine_kwargs()["compilation_config"])
    )


def test_ascend_910c_caps_offline_profile_batch():
    cfg = ModelConfig.load(
        "qwen3_6",
        "35b_a3b_tp2_eager",
        platform="ascend",
        device="910c",
    )

    assert cfg.engine_kwargs()["max_num_batched_tokens"] == 4096
    assert (
        cfg.serve_args()[cfg.serve_args().index("--max-num-batched-tokens") + 1]
        == "4096"
    )


def test_enflame_ci_does_not_mask_known_failures_or_expand_defaults():
    from tests.utils.platform_config import PlatformConfig

    config = PlatformConfig.load("enflame", device="s60")
    assert "TORCHINDUCTOR_EMULATE_PRECISION_CASTS" not in config.env_defaults
    overrides = config.get_device_case_overrides("qwen3_6", "35b_a3b_tp2_graph")
    assert overrides["llm"]["gpu_memory_utilization"] == 0.90
    cfg = ModelConfig.load(
        "qwen3_6", "35b_a3b_tp2_graph", platform="enflame", device="s60"
    )
    assert cfg.engine_kwargs()["max_num_seqs"] == 64
    args = cfg.serve_args()
    assert args[args.index("--max-num-seqs") + 1] == "64"
    assert overrides["serve"] == {"startup_retries": 180}
    assert cfg.serve.startup_retries == 180
    for phase in ("pr", "nightly"):
        tasks = config.device_tests["s60"]["tests"]["e2e"][phase]
        for families in tasks.values():
            assert "35b_a3b_tp2_eager" not in families.get("qwen3_6", [])
