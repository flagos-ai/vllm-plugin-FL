# Copyright (c) 2025 BAAI. All rights reserved.

from __future__ import annotations

import textwrap
from dataclasses import asdict

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
    ("case", "variable", "path"),
    [
        ("27b_tp4_eager", "MUSA_GATE_MODEL_27B", "/data/models/Qwen3.6-27B"),
        ("27b_tp4_graph", "MUSA_GATE_MODEL_27B", "/data/models/Qwen3.6-27B"),
        (
            "35b_a3b_tp4_eager",
            "MUSA_GATE_MODEL_35B_A3B",
            "/data/models/Qwen3.6-35B-A3B",
        ),
        (
            "35b_a3b_tp4_graph",
            "MUSA_GATE_MODEL_35B_A3B",
            "/data/models/Qwen3.6-35B-A3B",
        ),
    ],
)
def test_musa_allocation_paths_only_change_selected_model(
    monkeypatch, case, variable, path
):
    monkeypatch.delenv("MUSA_GATE_MODEL_27B", raising=False)
    monkeypatch.delenv("MUSA_GATE_MODEL_35B_A3B", raising=False)
    catalog = ModelConfig.load("qwen3_6", case, platform="musa", device="s5000")
    monkeypatch.setenv("MUSA_GATE_MODEL_27B", "/data/unused/27B")
    monkeypatch.setenv("MUSA_GATE_MODEL_35B_A3B", "/data/unused/35B-A3B")
    monkeypatch.setenv(variable, path)
    resolved = ModelConfig.load("qwen3_6", case, platform="musa", device="s5000")
    expected = asdict(catalog)
    expected["model"] = path
    assert asdict(resolved) == expected
    assert resolved.engine_kwargs()["model"] == path
    assert resolved.serve_args() == catalog.serve_args()
    assert resolved.engine["tensor_parallel_size"] == 4
    assert resolved.engine["gpu_memory_utilization"] == 0.80
    assert resolved.engine["trust_remote_code"] is True


@pytest.mark.parametrize(
    "case",
    ["27b_tp4_eager", "27b_tp4_graph", "35b_a3b_tp4_eager", "35b_a3b_tp4_graph"],
)
def test_cuda_ignores_musa_allocation_paths(monkeypatch, case):
    monkeypatch.delenv("MUSA_GATE_MODEL_27B", raising=False)
    monkeypatch.delenv("MUSA_GATE_MODEL_35B_A3B", raising=False)
    catalog = ModelConfig.load("qwen3_6", case, platform="cuda", device="a100")
    monkeypatch.setenv("FL_TEST_PLATFORM", "musa")
    monkeypatch.setenv("MUSA_GATE_MODEL_27B", "/data/replacement/27B")
    monkeypatch.setenv("MUSA_GATE_MODEL_35B_A3B", "/data/replacement/35B-A3B")
    resolved = ModelConfig.load("qwen3_6", case, platform="cuda", device="a100")
    assert asdict(resolved) == asdict(catalog)


@pytest.mark.parametrize(
    "path",
    [
        "relative/model",
        "/tmp/model",
        "/data/../model",
        "/data/model name",
        "/data/model;command",
    ],
)
def test_musa_allocation_path_validation_fails_closed(monkeypatch, path):
    monkeypatch.setenv("MUSA_GATE_MODEL_27B", path)
    with pytest.raises(ValueError, match="approved absolute /data path"):
        ModelConfig.load("qwen3_6", "27b_tp4_eager", platform="musa", device="s5000")


def test_unmapped_musa_case_keeps_catalog_path(monkeypatch):
    monkeypatch.delenv("MUSA_GATE_MODEL_27B", raising=False)
    catalog = ModelConfig.load(
        "qwen3_6", "27b_tp2_eager", platform="musa", device="s5000"
    )
    monkeypatch.setenv("MUSA_GATE_MODEL_27B", "/data/replacement/27B")
    resolved = ModelConfig.load(
        "qwen3_6", "27b_tp2_eager", platform="musa", device="s5000"
    )
    assert asdict(resolved) == asdict(catalog)


def test_musa_platform_environment_uses_same_allocation_path(monkeypatch):
    monkeypatch.setenv("FL_TEST_PLATFORM", "musa")
    monkeypatch.setenv("FL_TEST_DEVICE", "s5000")
    monkeypatch.setenv("MUSA_GATE_MODEL_27B", "/data/models/Qwen3.6-27B")
    resolved = ModelConfig.load("qwen3_6", "27b_tp4_eager")
    assert resolved.model == "/data/models/Qwen3.6-27B"


def test_other_model_family_keeps_catalog_path(tmp_path, monkeypatch):
    _write(
        tmp_path / "other_model" / "27b_tp4_eager.yaml",
        """
        llm:
          model: "/data/catalog/other-model"
          tensor_parallel_size: 4
        """,
    )
    monkeypatch.setenv("MUSA_GATE_MODEL_27B", "/data/replacement/27B")
    resolved = ModelConfig.load(
        "other_model",
        "27b_tp4_eager",
        models_dir=tmp_path,
        platform="musa",
        device="s5000",
    )
    assert resolved.model == "/data/catalog/other-model"


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
