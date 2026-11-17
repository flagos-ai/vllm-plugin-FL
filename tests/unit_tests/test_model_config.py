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
    ["27b_tp4_eager", "27b_tp4_graph", "35b_a3b_tp4_eager", "35b_a3b_tp4_graph"],
)
def test_kunlunxin_ci_uses_validated_batch_capacity(case):
    """Keep serving and offline CI aligned with the 1..256 regression profile."""
    cfg = ModelConfig.load("qwen3_6", case, platform="kunlunxin", device="p800")

    assert cfg.engine_kwargs()["max_num_seqs"] == 256
    args = cfg.serve_args(**cfg.serve.extra_engine)
    assert args.count("--max-num-seqs") == 1
    assert args[args.index("--max-num-seqs") + 1] == "256"
    assert cfg.engine["enforce_eager"] is case.endswith("_eager")
    assert ("--enforce-eager" in args) is case.endswith("_eager")
    assert "--compilation-config" not in args


def test_case_overrides_extend_shared_device_overrides(tmp_path, monkeypatch):
    models_dir = tmp_path / "models"
    platforms_dir = tmp_path / "platforms"
    monkeypatch.setattr(platform_config, "_PLATFORMS_DIR", platforms_dir)

    _write(
        models_dir / "qwen3_6" / "27b_tp2_graph.yaml",
        """
        llm:
          model: "/data/models/Qwen/Qwen3.6-27B"
          tensor_parallel_size: 2
          enforce_eager: false
        generate:
          prompts:
            - "Where is the capital of France"
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
              - qwen3_6/27b_tp2_graph
            llm:
              max_model_len: 4096
            case_overrides:
              qwen3_6/27b_tp2_graph:
                llm:
                  compilation_config:
                    cudagraph_mode: FULL_DECODE_ONLY
        demo_card:
          name: "demo_card"
          tests: {}
        """,
    )

    cfg = ModelConfig.load(
        "qwen3_6",
        "27b_tp2_graph",
        models_dir=models_dir,
        platform="demo",
        device="demo_card",
    )

    assert cfg.engine["max_model_len"] == 4096
    assert cfg.engine["compilation_config"] == {"cudagraph_mode": "FULL_DECODE_ONLY"}
    args = cfg.serve_args()
    index = args.index("--compilation-config")
    assert json.loads(args[index + 1]) == {"cudagraph_mode": "FULL_DECODE_ONLY"}


@pytest.mark.parametrize(
    "case",
    ["27b_tp2_eager", "27b_tp2_graph", "35b_a3b_tp2_eager", "35b_a3b_tp2_graph"],
)
def test_ascend_ci_requests_supported_cudagraph_mode(case):
    """Ascend accepts only NONE or FULL_DECODE_ONLY for non-eager runs."""
    cfg = ModelConfig.load("qwen3_6", case, platform="ascend", device="910c")
    graph_case = case.endswith("_graph")

    assert cfg.engine["enforce_eager"] is not graph_case
    assert ("compilation_config" in cfg.engine) is graph_case
    if graph_case:
        assert cfg.engine["compilation_config"] == {
            "cudagraph_mode": "FULL_DECODE_ONLY"
        }
    args = cfg.serve_args()
    assert ("--compilation-config" in args) is graph_case
    if graph_case:
        index = args.index("--compilation-config")
        assert json.loads(args[index + 1]) == {"cudagraph_mode": "FULL_DECODE_ONLY"}
