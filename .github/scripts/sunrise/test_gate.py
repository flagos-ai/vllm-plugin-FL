# Copyright 2026 FlagOS Contributors
"""CPU contracts for acceptance evidence, not model or GPU execution."""

import importlib.util
import io
import json
import sys
from pathlib import Path
from types import SimpleNamespace
from urllib.error import HTTPError

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


def test_gate_removes_inherited_whitelist_from_server_launch(
    gate, monkeypatch, tmp_path
):
    model = tmp_path / "model"
    model.mkdir()
    (model / "config.json").write_text("{}")
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_gate.py",
            "--model",
            str(model),
            "--mode",
            "eager",
            "--output-dir",
            str(tmp_path / "output"),
        ],
    )
    monkeypatch.setenv(gate.FLAGOS_WHITELIST_ENV, "rsqrt")
    monkeypatch.setenv("SUNRISE_GATE_REPO", str(HERE.parents[2]))
    monkeypatch.delenv(
        "VLLM_FL_SUNRISE_BENCH_BASELINE_GDN_CORE_ATTN_ZEROS", raising=False
    )
    launches = []
    popen = gate.subprocess.Popen
    check = f"import os; assert {gate.FLAGOS_WHITELIST_ENV!r} not in os.environ"
    server_probe = (
        f"{check}; import subprocess, sys; "
        f"subprocess.run([sys.executable, '-c', {check!r}], check=True)"
    )

    def record_launch(command, *, env, **kwargs):
        launches.append((command, env))
        with popen([sys.executable, "-c", server_probe], env=env) as probe:
            launches.append(probe.wait(timeout=10))
        raise RuntimeError("Stop before launching the server")

    monkeypatch.setattr(gate.subprocess, "Popen", record_launch)
    assert gate.main() == 1
    (command, env), probe_returncode = launches
    assert "serve" in command
    assert gate.FLAGOS_WHITELIST_ENV not in env
    assert probe_returncode == 0
    assert gate.os.environ[gate.FLAGOS_WHITELIST_ENV] == "rsqrt"


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
            log.write_text(
                "startup\n" + native_worker_log() + "| 1 | 1 | 0 | FULL | 4 |\n"
            )
            effective = expected_effective(mode)
            proof = {
                "source": "server_info",
                "model": model_path,
                "served_model_name": "owned",
                "port": 18360,
                "configuration": effective,
            }
            (directory / "effective-configuration.json").write_text(json.dumps(proof))
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
                "effective_configuration": effective,
                "effective_config_evidence": "effective-configuration.json",
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
    (directories[1] / "server.log").write_text(
        "startup\n" + native_worker_log() + "Graph capturing finished"
    )
    with pytest.raises(ValueError, match="request-time FULL"):
        matrix.validate_matrix(directories)


def native_worker_log():
    return "(Worker pid=10) INFO [vllm_fl/worker/sunrise_worker.py:139] Using Sunrise Model Runner V2\n"


def expected_effective(mode="eager"):
    return {
        "tensor_parallel_size": 2,
        "max_model_len": 8192,
        "max_num_seqs": 8,
        "kv_cache_memory_bytes": 4294967296,
        "enforce_eager": mode == "eager",
        "use_v2_model_runner": True,
        "compilation_mode": 0,
        "compilation_backend": "eager",
        "cudagraph_mode": "NONE" if mode == "eager" else "FULL_DECODE_ONLY",
        "cudagraph_capture_sizes": [] if mode == "eager" else [1, 2, 4, 8],
    }


def actual_server_info(mode="eager"):
    graph = 0 if mode == "eager" else [2, 0]
    return {
        "vllm_config": {
            "model_config": {
                "model": "/models/m",
                "served_model_name": ["owned"],
                "max_model_len": 8192,
                "enforce_eager": mode == "eager",
            },
            "parallel_config": {"tensor_parallel_size": 2},
            "scheduler_config": {"max_num_seqs": 8},
            "cache_config": {"kv_cache_memory_bytes": 4294967296},
            "compilation_config": {
                "mode": 0,
                "backend": "eager",
                "cudagraph_mode": graph,
                "cudagraph_capture_sizes": [] if mode == "eager" else [1, 2, 4, 8],
            },
        },
        "vllm_env": {"POTENTIAL_SECRET": "MUST_NOT_BE_RECORDED"},
        "system_env": {"environment": "MUST_NOT_BE_RECORDED"},
    }


def actual_startup_log(mode="eager"):
    args = {
        "model": "/models/m",
        "served_model_name": ["owned"],
        "port": 18360,
        "max_model_len": 8192,
        "tensor_parallel_size": 2,
        "max_num_seqs": 8,
        "kv_cache_memory_bytes": 4294967296,
    }
    eager = mode == "eager"
    enum = (
        "<CUDAGraphMode.NONE: 0>"
        if eager
        else "<CUDAGraphMode.FULL_DECODE_ONLY: (2, 0)>"
    )
    compilation = (
        "{'mode': <CompilationMode.NONE: 0>, 'backend': 'eager', 'cudagraph_mode': "
        + enum
        + ", 'cudagraph_capture_sizes': "
        + repr([] if eager else [1, 2, 4, 8])
        + ", 'extra': {'nested': True}}"
    )
    return (
        "(APIServer pid=1) INFO [api_utils.py:286] non-default args: "
        + repr(args)
        + "\n"
        "(EngineCore pid=2) INFO [core.py:133] Initializing a V1 LLM engine (v0.32.0) with config: model='/models/m', max_seq_len=8192, tensor_parallel_size=2, enforce_eager="
        + repr(eager)
        + ", compilation_config="
        + compilation
        + ", kernel_config=KernelConfig()\n"
        + native_worker_log()
    )


@pytest.mark.parametrize("mode", ["eager", "graph"])
def test_server_info_uses_real_numeric_vllm_config(gate, mode):
    result = gate.effective_from_server_info(
        actual_server_info(mode), "/models/m", "owned", native_worker_log()
    )
    assert result == expected_effective(mode)
    assert (
        gate.validate_effective_configuration(
            result,
            {
                key: result[key]
                for key in (
                    "tensor_parallel_size",
                    "max_model_len",
                    "max_num_seqs",
                    "kv_cache_memory_bytes",
                )
            },
            mode,
        )
        == result
    )


@pytest.mark.parametrize(
    "section,field",
    [
        ("parallel_config", "tensor_parallel_size"),
        ("scheduler_config", "max_num_seqs"),
        ("cache_config", "kv_cache_memory_bytes"),
        ("model_config", "max_model_len"),
        ("model_config", "enforce_eager"),
        ("compilation_config", "mode"),
        ("compilation_config", "backend"),
        ("compilation_config", "cudagraph_mode"),
        ("compilation_config", "cudagraph_capture_sizes"),
    ],
)
def test_missing_actual_config_fields_never_use_requested_defaults(
    gate, section, field
):
    document = actual_server_info()
    del document["vllm_config"][section][field]
    with pytest.raises(ValueError):
        gate.effective_from_server_info(
            document, "/models/m", "owned", native_worker_log()
        )


@pytest.mark.parametrize(
    "field,value",
    [
        ("tensor_parallel_size", 1),
        ("max_model_len", 4096),
        ("max_num_seqs", 4),
        ("kv_cache_memory_bytes", 2147483648),
        ("enforce_eager", False),
        ("use_v2_model_runner", False),
        ("cudagraph_mode", "FULL_DECODE_ONLY"),
        ("cudagraph_capture_sizes", [1, 2]),
    ],
)
def test_actual_wrong_configuration_is_rejected(gate, field, value):
    actual = expected_effective()
    actual[field] = value
    requested = {
        key: expected_effective()[key]
        for key in (
            "tensor_parallel_size",
            "max_model_len",
            "max_num_seqs",
            "kv_cache_memory_bytes",
        )
    }
    with pytest.raises(ValueError, match="Actual server"):
        gate.validate_effective_configuration(actual, requested, "eager")


@pytest.mark.parametrize(
    "field,value",
    [
        ("tensor_parallel_size", "2"),
        ("tensor_parallel_size", True),
        ("max_num_seqs", 8.0),
        ("compilation_mode", False),
        ("compilation_mode", 0.0),
        ("enforce_eager", "True"),
        ("cudagraph_capture_sizes", None),
    ],
)
def test_actual_config_types_are_strict(gate, field, value):
    actual = expected_effective()
    actual[field] = value
    with pytest.raises(ValueError):
        gate._normalize_effective(actual)


@pytest.mark.parametrize("mode", ["eager", "graph"])
def test_startup_log_extracts_typed_engine_and_actual_server_values(gate, mode):
    assert gate.effective_from_startup_log(
        actual_startup_log(mode), "/models/m", "owned", 18360
    ) == expected_effective(mode)


def test_startup_log_requires_missing_actual_fields(gate):
    text = actual_startup_log().replace("'max_num_seqs': 8, ", "")
    with pytest.raises(ValueError, match="max_num_seqs"):
        gate.effective_from_startup_log(text, "/models/m", "owned", 18360)
    with pytest.raises(ValueError, match="another model/server"):
        gate.effective_from_startup_log(
            actual_startup_log(), "/models/m", "owned", 18361
        )


def test_observation_keeps_only_selected_config_and_not_environment(gate, monkeypatch):
    document = actual_server_info()
    monkeypatch.setattr(
        gate,
        "build_opener",
        lambda *args: SimpleNamespace(
            open=lambda *args, **kwargs: io.BytesIO(json.dumps(document).encode())
        ),
    )
    proof = gate.observe_effective_configuration(
        "http://127.0.0.1:18360/v1", "/models/m", "owned", 18360, native_worker_log()
    )
    assert proof["source"] == "server_info"
    assert proof["configuration"] == expected_effective()
    assert "MUST_NOT_BE_RECORDED" not in json.dumps(proof)
    assert "vllm_env" not in proof and "system_env" not in proof


@pytest.mark.parametrize("status", [404, 405])
def test_only_unsupported_info_endpoint_falls_back_to_real_log(
    gate, monkeypatch, status
):
    def missing(*args, **kwargs):
        raise HTTPError(
            "http://127.0.0.1:18360/server_info", status, "unsupported", {}, None
        )

    monkeypatch.setattr(
        gate, "build_opener", lambda *args: SimpleNamespace(open=missing)
    )
    proof = gate.observe_effective_configuration(
        "http://127.0.0.1:18360/v1", "/models/m", "owned", 18360, actual_startup_log()
    )
    assert proof["source"] == "startup_log"
    assert proof["configuration"] == expected_effective()


def test_server_info_internal_error_does_not_get_hidden_by_fallback(gate, monkeypatch):
    def failure(*args, **kwargs):
        raise HTTPError("http://127.0.0.1:18360/server_info", 500, "failure", {}, None)

    monkeypatch.setattr(
        gate, "build_opener", lambda *args: SimpleNamespace(open=failure)
    )
    with pytest.raises(HTTPError):
        gate.observe_effective_configuration(
            "http://127.0.0.1:18360/v1",
            "/models/m",
            "owned",
            18360,
            actual_startup_log(),
        )


def test_matrix_checks_effective_not_claimed_requested_configuration(gate, tmp_path):
    matrix = load_module("verify_gate_matrix", HERE / "verify_gate_matrix.py")
    directories = create_matrix(gate, tmp_path)
    path = directories[0] / "effective-configuration.json"
    proof = json.loads(path.read_text())
    proof["configuration"]["tensor_parallel_size"] = 1
    path.write_text(json.dumps(proof))
    with pytest.raises(ValueError, match="tensor_parallel_size"):
        matrix.validate_matrix(directories)


def test_matrix_requires_effective_proof(gate, tmp_path):
    matrix = load_module("verify_gate_matrix", HERE / "verify_gate_matrix.py")
    directories = create_matrix(gate, tmp_path)
    path = directories[0] / "summary.json"
    receipt = json.loads(path.read_text())
    del receipt["effective_config_evidence"]
    path.write_text(json.dumps(receipt))
    with pytest.raises(ValueError, match="actual server configuration evidence"):
        matrix.validate_matrix(directories)


@pytest.mark.parametrize("selector", [False, "True"])
def test_api_v2_selector_cannot_be_hidden_by_old_worker_log(gate, selector):
    document = actual_server_info()
    document["vllm_env"]["VLLM_USE_V2_MODEL_RUNNER"] = selector
    with pytest.raises(ValueError, match="Runner V2"):
        gate.effective_from_server_info(
            document, "/models/m", "owned", native_worker_log()
        )
