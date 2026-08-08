# Copyright (c) 2025 BAAI. All rights reserved.

from __future__ import annotations

import textwrap

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
