# Copyright (c) 2025 BAAI. All rights reserved.

from __future__ import annotations

import json
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


def test_serve_args_emits_structured_options_as_json(tmp_path, monkeypatch):
    """Structured engine options must reach ``vllm serve`` as JSON.

    ``str()`` on a dict produces a Python repr with single quotes, which vLLM
    rejects with a pydantic json_invalid error before the server starts.
    Booleans and scalars keep their existing shape.
    """
    models_dir = tmp_path / "models"
    platforms_dir = tmp_path / "platforms"
    monkeypatch.setattr(platform_config, "_PLATFORMS_DIR", platforms_dir)

    _write(
        models_dir / "qwen3_6" / "27b_tp2_graph.yaml",
        """
        llm:
          model: "/data/models/Qwen/Qwen3.6-27B"
          tensor_parallel_size: 2
          max_model_len: 8192
          enforce_eager: false
          trust_remote_code: false
          compilation_config:
            cudagraph_mode: FULL_DECODE_ONLY
            cudagraph_capture_sizes: [1, 2, 4, 8]
        generate:
          prompts:
            - "Where is the capital of France"
          sampling:
            max_tokens: 100
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
        device_overrides: {}
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
    args = cfg.serve_args()

    value = args[args.index("--compilation-config") + 1]
    assert json.loads(value) == {
        "cudagraph_mode": "FULL_DECODE_ONLY",
        "cudagraph_capture_sizes": [1, 2, 4, 8],
    }

    # Scalars and booleans are untouched by the JSON branch.
    assert args[args.index("--max-model-len") + 1] == "8192"
    assert "--enforce-eager" not in args
    assert "--trust-remote-code" not in args
