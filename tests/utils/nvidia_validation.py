# Copyright 2026 FlagOS Contributors
"""Opt-in NVIDIA EMPTY inference evidence; no CUDA imports at discovery time."""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import os
from pathlib import Path

import yaml

from tests.utils.model_config import ModelConfig

ROOT = Path(__file__).resolve().parents[2]
MANUAL_CASES = {
    ("qwen3", "4b_v1_graph"),
    ("qwen3", "4b_v2_eager"),
    ("minicpm4_7", "1b_multimodal"),
    ("minicpm4_7", "1b_paired_images"),
    ("qwen3_8", "flash_next_tp8_eager"),
    ("deepseek_v4_1", "flash_tp8_eager"),
}
RUNNERS = {
    "v1": "vllm.v1.worker.gpu_model_runner.GPUModelRunner",
    "v2": "vllm.v1.worker.gpu.model_runner.GPUModelRunner",
}


def inference_environment(config, model, case, output_dir):
    """Only the CUDA device's explicitly configured cases opt in."""
    if config.platform != "cuda":
        return {}
    runner = (
        config.device_tests.get(config.device, {})
        .get("inference_runners", {})
        .get(f"{model}/{case}")
    )
    if runner is None:
        return {}
    if runner not in RUNNERS:
        raise ValueError(f"Invalid NVIDIA runner: {runner}")
    return {
        "VLLM_USE_V2_MODEL_RUNNER": str(int(runner == "v2")),
        "FL_NVIDIA_RUNNER": runner,
        "FL_NVIDIA_RESULT_PATH": str(
            Path(output_dir).resolve() / f"nvidia-{model}-{case}.json"
        ),
    }


def load_manual_case(model, case, model_path):
    if (model, case) not in MANUAL_CASES:
        raise ValueError(f"Unknown manual NVIDIA case: {model}/{case}")
    cfg = ModelConfig.load(model, case, platform="cuda", device="h100")
    cfg.model = str(Path(model_path).resolve())
    raw = yaml.safe_load((ROOT / "tests/models" / model / f"{case}.yaml").read_text())
    return cfg, raw["nvidia_validation"]


def checkpoint_facts(model_path, expected_architectures=None):
    path = Path(model_path)
    config = path / "config.json"
    if not config.is_file():
        raise FileNotFoundError(f"Selected model is missing config.json: {path}")
    architectures = json.loads(config.read_text(encoding="utf-8")).get(
        "architectures", []
    )
    if expected_architectures is not None and architectures != expected_architectures:
        raise ValueError(
            f"Wrong checkpoint architecture: expected {expected_architectures}, got {architectures}"
        )
    index = path / "model.safetensors.index.json"
    if index.is_file():
        names = set(json.loads(index.read_text())["weight_map"].values())
        weights = [path / name for name in names]
    else:
        weights = list(path.glob("*.safetensors")) + list(
            path.glob("pytorch_model*.bin")
        )
    if not weights or any(not p.is_file() or p.stat().st_size == 0 for p in weights):
        raise FileNotFoundError(f"Selected model has missing/empty weights: {path}")
    return {
        "path": str(path),
        "weight_files": len(weights),
        "architectures": architectures,
        "metadata_sha256": {
            p.name: hashlib.sha256(p.read_bytes()).hexdigest()
            for p in (config, index)
            if p.is_file()
        },
    }


def validate_resources(devices, cfg, minimum_memory_gib):
    required = cfg.engine.get("tensor_parallel_size", 1) * cfg.engine.get(
        "pipeline_parallel_size", 1
    )
    if len(devices) < required:
        raise RuntimeError(
            f"Selected case requires {required} visible GPUs; found {len(devices)}"
        )
    for device in devices[:required]:
        if device["total_bytes"] < minimum_memory_gib * 1024**3:
            raise RuntimeError(f"GPU capacity below {minimum_memory_gib} GiB: {device}")
        if device["free_bytes"] < device["total_bytes"] * cfg.engine.get(
            "gpu_memory_utilization", 0.9
        ):
            raise RuntimeError(
                f"Insufficient free GPU memory; reserve an idle device: {device}"
            )


class NvidiaWorkerEvidence:
    """Named, read-only RPC; leaves upstream callable serialization disabled."""

    def nvidia_regression_facts(self):
        from vllm_fl.nvidia_compat import has_native_vllm_kernels

        runner = self.model_runner
        model = runner.get_model()
        return {
            "rank": self.rank,
            "use_v2_model_runner": self.use_v2_model_runner,
            "runner_class": f"{type(runner).__module__}.{type(runner).__name__}",
            "model_class": f"{type(model).__module__}.{type(model).__name__}",
            "native_vllm_kernels": has_native_vllm_kernels(),
        }


class InferenceEvidence:
    def __init__(self, cfg, runner, output):
        if runner not in RUNNERS:
            raise ValueError(f"Invalid NVIDIA runner: {runner}")
        self.cfg, self.runner, self.path = cfg, runner, Path(output)
        self.data = {
            "status": "running",
            "runner": runner,
            "engine_kwargs": cfg.engine_kwargs(),
            "results": [],
        }
        self.save()

    def save(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(
            json.dumps(self.data, indent=2, ensure_ascii=False, default=str) + "\n",
            encoding="utf-8",
        )

    def workers(self, llm, expected_model_class=None):
        facts = llm.collective_rpc("nvidia_regression_facts", timeout=180)
        self.data["actual_workers"] = facts
        self.data["versions"] = {
            n: importlib.metadata.version(n)
            for n in ("vllm", "torch", "flagtree", "flag-gems")
        }
        self.save()
        required = self.cfg.engine.get("tensor_parallel_size", 1) * self.cfg.engine.get(
            "pipeline_parallel_size", 1
        )
        assert len(facts) == required and {f["rank"] for f in facts} == set(
            range(required)
        ), "Missing worker ranks"
        assert all(f["use_v2_model_runner"] is (self.runner == "v2") for f in facts), (
            "Wrong worker runner choice"
        )
        assert all(f["runner_class"] == RUNNERS[self.runner] for f in facts), (
            "Wrong native runner class"
        )
        assert all(not f["native_vllm_kernels"] for f in facts), (
            "Expected EMPTY vLLM workers"
        )

        if expected_model_class is not None:
            assert all(f["model_class"] == expected_model_class for f in facts), (
                "Wrong actual model class"
            )

    def outputs(self, name, first, repeat, expected, match="ordered_terms"):
        rows = [
            {"text": x.outputs[0].text, "token_ids": list(x.outputs[0].token_ids)}
            for x in first
        ]
        repeats = [list(x.outputs[0].token_ids) for x in repeat]
        self.data["results"].append(
            {
                "name": name,
                "outputs": rows,
                "repeat_token_ids": repeats,
                "match": match,
                "expected": expected,
            }
        )
        self.save()
        assert len(rows) == len(expected) == len(repeats) and rows, (
            "Missing generation results"
        )
        assert all(r["text"].strip() and r["token_ids"] for r in rows), (
            "Empty generation"
        )
        assert [r["token_ids"] for r in rows] == repeats, "Greedy repeat differs"
        for row, terms in zip(rows, expected, strict=True):
            if match == "exact":
                assert row["text"].strip() in terms, (
                    f"Expected exact answer {terms!r}, got {row['text']!r}"
                )
                continue
            assert match == "ordered_terms", f"Unsupported answer matching: {match}"
            offset = 0
            for term in terms:
                position = row["text"].casefold().find(term.casefold(), offset)
                assert position >= 0, (
                    f"Expected ordered term {term!r}, got {row['text']!r}"
                )
                offset = position + len(term)

    def finish(self, error=None):
        if error is None and (
            not self.data.get("actual_workers") or not self.data["results"]
        ):
            error = RuntimeError(
                "No worker or generation evidence; regression did not execute"
            )
            self.finish(error)
            raise error
        self.data["status"] = "passed" if error is None else "failed"
        if error is not None:
            self.data["error"] = f"{type(error).__name__}: {error}"
        self.save()


def evidence_from_environment(cfg):
    runner = os.environ.get("FL_NVIDIA_RUNNER")
    if not runner or os.environ.get("FL_TEST_PLATFORM") != "cuda":
        return None
    return InferenceEvidence(cfg, runner, os.environ["FL_NVIDIA_RESULT_PATH"])


def verify_execution(report_path, evidence_path=None):
    """Reject collect-only/skip-only runs and missing actual inference evidence."""
    data = json.loads(Path(report_path).read_text(encoding="utf-8"))
    summary = data["summary"]
    if (
        data.get("exitcode", 0) != 0
        or summary.get("passed") != 1
        or summary.get("collected") != 1
    ):
        raise RuntimeError(f"Expected one executed regression, got: {summary}")
    if any(summary.get(key, 0) for key in ("failed", "error", "skipped", "deselected")):
        raise RuntimeError(f"Incomplete regression: {summary}")
    if evidence_path is not None:
        evidence = json.loads(Path(evidence_path).read_text(encoding="utf-8"))
        if (
            evidence.get("status") != "passed"
            or not evidence.get("actual_workers")
            or not evidence.get("results")
        ):
            raise RuntimeError("Missing successful worker/generation evidence")
