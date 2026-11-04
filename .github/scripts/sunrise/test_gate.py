# Copyright 2026 FlagOS Contributors
"""CPU contracts for acceptance evidence, not model or GPU execution."""

import importlib.util
import io
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

HERE = Path(__file__).resolve().parent


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def gate(monkeypatch):
    module = load_module("run_gate", HERE / "run_gate.py")
    monkeypatch.setitem(sys.modules, "run_gate", module)
    return module


def write_shared_reports(
    gate, root, model="/models/Qwen3.6-27B", served="owned", port=18360
):
    root.mkdir(parents=True)
    for name, ids in gate.REQUEST_IDS.items():
        count = len(ids)
        document = {
            "case": {
                "scenario": name,
                "model_path": model,
                "served_model_name": served,
                "port": str(port),
                "concurrency": count,
            },
            "input": [{"request_id": item} for item in ids],
            "output": [
                {
                    "request_id": item,
                    "passed": True,
                    "text": "answer",
                    "checks": dict.fromkeys(gate.CHECKS, True),
                }
                for item in ids
            ],
            "summary": {
                "request_count": count,
                "passed_count": count,
                "failed_count": 0,
                "passed": True,
            },
        }
        (root / (name + ".json")).write_text(json.dumps(document))


def test_shared_gate_requires_all_26_requests(gate, tmp_path):
    write_shared_reports(gate, tmp_path / "requests")
    assert gate.validate_results(
        tmp_path / "requests", "/models/Qwen3.6-27B", "owned", 18360
    ) == {"scenarios": 5, "requests": 26, "passed": True}


@pytest.mark.parametrize(
    "mutation",
    [
        "duplicate_id",
        "wrong_server",
        "wrong_concurrency",
        "missing_check",
        "false_check",
        "failed_summary",
        "empty",
        "error",
    ],
)
def test_shared_gate_rejects_incomplete_or_foreign_evidence(gate, tmp_path, mutation):
    root = tmp_path / "requests"
    write_shared_reports(gate, root)
    path = root / "text_concurrent_8.json"
    doc = json.loads(path.read_text())
    if mutation == "duplicate_id":
        doc["output"][1]["request_id"] = doc["output"][0]["request_id"]
    elif mutation == "wrong_server":
        doc["case"]["served_model_name"] = "unrelated"
    elif mutation == "wrong_concurrency":
        doc["case"]["concurrency"] = 1
    elif mutation == "missing_check":
        doc["output"][0]["checks"].pop("expected_order")
    elif mutation == "false_check":
        doc["output"][0]["checks"]["expected_order"] = False
    elif mutation == "failed_summary":
        doc["summary"]["failed_count"] = 1
    elif mutation == "empty":
        doc["output"][0]["text"] = " "
    else:
        doc["output"][0]["error"] = "failure"
    path.write_text(json.dumps(doc))
    with pytest.raises(ValueError):
        gate.validate_results(root, "/models/Qwen3.6-27B", "owned", 18360)


@pytest.mark.parametrize("mode", ["FULL", "CUDAGraphMode.FULL"])
def test_native_request_time_full_graph_metric(gate, mode):
    evidence = gate.replay_evidence(f"| 1 | 2 | 1 | {mode} | 4 |")
    assert evidence["native_full_executions"] == 4


@pytest.mark.parametrize(
    "text",
    [
        "Graph capturing finished",
        "| 1 | 1 | 0 | NONE | 8 |",
        "| 1 | 1 | 0 | FULL | 0 |",
        "| 0 | 1 | 1 | FULL | 8 |",
    ],
)
def test_capture_or_no_request_full_metric_is_not_replay_evidence(gate, text):
    with pytest.raises(ValueError, match="request-time FULL"):
        gate.replay_evidence(text)


def test_server_graph_configuration_is_explicit(gate, tmp_path):
    args = ("/models/m", "owned", 18360, "graph", tmp_path, 4294967296, 2)
    command = gate.server_command(*args, repo=tmp_path)
    configuration = json.loads(command[command.index("--compilation-config") + 1])
    assert configuration == {
        "mode": 0,
        "backend": "eager",
        "cudagraph_mode": "FULL_DECODE_ONLY",
        "cudagraph_capture_sizes": [1, 2, 4, 8],
    }
    assert "--cudagraph-metrics" in command
    assert "--enforce-eager" not in command
    eager = gate.server_command(*args[:3], "eager", *args[4:], repo=tmp_path)
    assert "--enforce-eager" in eager


def test_kv_capacity_uses_actual_token_capacity(gate):
    assert gate.kv_cache_capacity("PTPU KV cache size: 12,345 tokens", 8192)[
        "token_capacities"
    ] == [12345]
    with pytest.raises(ValueError):
        gate.kv_cache_capacity("GPU KV cache size: 4,096 tokens", 8192)


def test_reuse_server_is_local_explicit_and_not_owned(gate, tmp_path):
    log = tmp_path / "external.log"
    log.write_text("keep")
    assert gate.reuse_endpoint("http://127.0.0.1:18360", "owned", log) == (
        18360,
        "http://127.0.0.1:18360/v1",
    )
    assert gate.stop_owned_session(None) is True
    assert log.read_text() == "keep"


@pytest.mark.parametrize(
    "url",
    [
        "http://remote:18360",
        "http://127.0.0.1",
        "http://user@localhost:18360",
        "http://127.0.0.1:18360/api",
        "http://127.0.0.1:18360?x=1",
    ],
)
def test_reuse_rejects_ambiguous_endpoints(gate, tmp_path, url):
    log = tmp_path / "external.log"
    log.touch()
    with pytest.raises(ValueError):
        gate.reuse_endpoint(url, "owned", log)


def test_reuse_requires_matching_model_root(gate, monkeypatch):
    document = {"data": [{"id": "owned", "root": "/models/other"}]}
    opener = SimpleNamespace(
        open=lambda *args, **kwargs: io.BytesIO(json.dumps(document).encode())
    )
    monkeypatch.setattr(gate, "build_opener", lambda *args: opener)
    with pytest.raises(RuntimeError, match="root does not match"):
        gate.wait_owned_service(None, 18360, "owned", 5, model="/models/expected")


@pytest.mark.parametrize(
    "measured,answer,passes",
    [
        (4096, "violet maple 42", True),
        (4095, "violet maple 42", False),
        (4096, "wrong phrase", False),
    ],
)
def test_long_prefill_uses_measured_server_usage_and_semantics(
    gate, monkeypatch, tmp_path, measured, answer, passes
):
    tokenizer = SimpleNamespace(encode=lambda text, **kwargs: list(range(len(text))))
    monkeypatch.setitem(
        sys.modules,
        "transformers",
        SimpleNamespace(
            AutoTokenizer=SimpleNamespace(
                from_pretrained=lambda *args, **kwargs: tokenizer
            )
        ),
    )
    monkeypatch.setitem(sys.modules, "openai", SimpleNamespace(AsyncOpenAI=object))
    monkeypatch.setenv("SUNRISE_GATE_REPO", str(HERE.parents[2]))
    doc = {
        "choices": [{"message": {"content": answer}}],
        "usage": {"prompt_tokens": measured},
    }
    monkeypatch.setattr(
        gate,
        "build_opener",
        lambda *args: SimpleNamespace(
            open=lambda *args, **kwargs: io.BytesIO(json.dumps(doc).encode())
        ),
    )
    if passes:
        assert (
            gate.measured_long_prefill(
                "/models/m", "owned", 18360, 4096, 8192, tmp_path
            )["measured_prompt_tokens"]
            == 4096
        )
    else:
        with pytest.raises(ValueError, match="Measured long-prefill"):
            gate.measured_long_prefill(
                "/models/m", "owned", 18360, 4096, 8192, tmp_path
            )
    assert json.loads((tmp_path / "long-prefill.json").read_text())["passed"] is passes


def create_matrix(gate, root):
    directories = []
    for model in ("Qwen3.6-27B", "Qwen3.6-35B-A3B"):
        for mode in ("eager", "graph"):
            directory = root / (model + "-" + mode)
            model_path = "/models/" + model
            write_shared_reports(gate, directory / "requests", model=model_path)
            log = directory / "server.log"
            log.write_text("startup\n| 1 | 1 | 0 | FULL | 4 |\n")
            receipt = {
                "model": model_path,
                "mode": mode,
                "served_model_name": "owned",
                "port": 18360,
                "passed": True,
                "runner_v2": True,
                "only_owned_cleanup_complete": True,
                "requested_configuration": {
                    "tensor_parallel_size": 2,
                    "max_model_len": 8192,
                    "max_num_seqs": 8,
                    "kv_cache_memory_bytes": 4294967296,
                },
                "gate": {"scenarios": 5, "requests": 26, "passed": True},
                "gate_returncode": 0,
                "gate_source_sha256": {"case_utils.py": "a" * 64},
                "long_prefill": {"measured_prompt_tokens": 4100},
                "server_log": str(log),
                "gate_start_log_offset": 8,
                "graph_capture_evidence": True,
                "graph_replay_evidence": {"native_full_executions": 4},
            }
            long = {
                "model": model_path,
                "passed": True,
                "target_prompt_tokens": 4096,
                "measured_prompt_tokens": 4100,
                "checks": dict.fromkeys(gate.CHECKS, True),
                "response": {"usage": {"prompt_tokens": 4100}},
            }
            (directory / "summary.json").write_text(json.dumps(receipt))
            (directory / "long-prefill.json").write_text(json.dumps(long))
            directories.append(directory)
    return directories


def test_four_scenario_matrix_counts_original_and_supplemental_requests(gate, tmp_path):
    matrix = load_module("verify_gate_matrix", HERE / "verify_gate_matrix.py")
    result = matrix.validate_matrix(create_matrix(gate, tmp_path))
    assert result["shared_requests"] == 104
    assert result["supplemental_long_prefill_requests"] == 4
    assert result["total_requests"] == 108


def test_matrix_rejects_duplicate_scenario_and_absent_post_ready_replay(gate, tmp_path):
    matrix = load_module("verify_gate_matrix", HERE / "verify_gate_matrix.py")
    directories = create_matrix(gate, tmp_path)
    with pytest.raises(ValueError, match="Duplicate"):
        matrix.validate_matrix([directories[0], directories[0], *directories[2:]])
    (directories[1] / "server.log").write_text("startup\nGraph capturing finished")
    with pytest.raises(ValueError, match="request-time FULL"):
        matrix.validate_matrix(directories)
