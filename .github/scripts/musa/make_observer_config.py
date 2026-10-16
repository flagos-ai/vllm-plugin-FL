"""Create a readonly actual-source/original-request config; never send a request."""

import argparse
import importlib.util
import os
import sys
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from aggregate_gate import canonical_sha
from gate_common import (
    CHECKS,
    GATE_FILES,
    ROUTES,
    SCENARIOS,
    fingerprint,
    read_json,
    require,
    write_json,
)
from preflight import selected_owner, selected_record

ROLE_SOURCES = {
    "api": [
        "vllm.plugins",
        "vllm.entrypoints.openai.api_server",
        "vllm.entrypoints.openai.chat_completion.serving",
        "vllm.v1.engine.async_llm",
        "vllm.v1.engine.input_processor",
    ],
    "worker": [
        "vllm.plugins",
        "vllm.v1.worker.worker_base",
        "vllm_fl.worker.worker",
        "vllm_fl.worker.model_runner",
        "vllm_fl.compilation.graph",
        "vllm.forward_context",
        "torch_musa.musa_graph.graphs",
    ],
    "client": [
        "case_utils",
        "test_text",
        "test_image",
        "test_mix_text_image",
        "httpx._client",
        "openai._base_client",
    ],
}
ROLE_HOOKS = {
    "api": [
        "OpenAIServingChat.create_chat_completion",
        "AsyncLLM.generate",
        "AsyncLLM._add_request",
    ],
    "worker": [
        "runner.execute_model",
        "GraphWrapper.__call__",
        "torch.musa.graph.__enter__",
        "torch.musa.graph.__exit__",
        "MUSAGraph.replay",
        "worker.shutdown",
    ],
    "client": [
        "case_utils.run_case",
        "case_utils.execute_request",
        "httpx.AsyncClient.send",
    ],
}


def make(binding_path, run_id, trace_directory):
    require(
        not sys.flags.no_site,
        "Use normal-site Python for the unchanged case_utils/OpenAI imports",
    )
    binding = read_json(binding_path, 64 * 1024 * 1024)
    kit = Path(binding["gate_dir"]).resolve(strict=True)
    package_sources = binding["observer_sources"]
    require(
        package_sources is not None,
        "The independently built observer wheel is not installed/bound",
    )
    own = {
        name: {key: row[key] for key in ("bytes", "sha256")}
        for name, row in package_sources.items()
    }
    require(
        all(fingerprint(row["path"]) == row for row in package_sources.values()),
        "Installed observer source changed",
    )
    gate = {
        name: {key: row[key] for key in ("path", "bytes", "sha256")}
        for name, row in binding["original_gate"].items()
    }
    require(
        set(gate) == set(GATE_FILES)
        and all(fingerprint(row["path"]) == row for row in gate.values()),
        "Original committed Gate files changed",
    )
    sources = {}
    record_paths = {}
    for name in sorted({name for names in ROLE_SOURCES.values() for name in names}):
        if name in ("case_utils", "test_text", "test_image", "test_mix_text_image"):
            sources[name] = gate[name + ".py"]
            continue
        top, *parts = name.split(".")
        package_init = Path(binding["modules"][top]["path"])
        require(
            package_init.name == "__init__.py",
            "Bound top module is not a source package",
        )
        base = package_init.parent.joinpath(*parts)
        candidates = [
            path
            for path in (base.with_suffix(".py"), base / "__init__.py")
            if path.is_file()
        ]
        require(len(candidates) == 1, f"No unique actual source module: {name}")
        path = candidates[0].resolve(strict=True)
        sources[name] = fingerprint(path)
        owner = selected_owner(top, path)
        record_paths.setdefault(owner, set()).add(path)
    source_records = {
        name: selected_record(name, sorted(paths))
        for name, paths in record_paths.items()
    }
    for name, record in source_records.items():
        require(
            record["record"] == binding["distributions"][name]["record"],
            "Additional observer source RECORD differs from the preflight origin",
        )
    # Import only original case_utils. No unused test, model or HTTP call is run.
    spec = importlib.util.spec_from_file_location("case_utils", kit / "case_utils.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    routes = []
    for rid, model, mode, port in ROUTES:
        requests = []
        for scenario in SCENARIOS:
            originals = module.scenario_requests(scenario)
            require(
                len(originals) == SCENARIOS[scenario], "Original scenario count changed"
            )
            for original in originals:
                requests.append(
                    {
                        "logical_id": f"{rid}:{scenario}:{original['request_id']}",
                        "original_request_id": original["request_id"],
                        "scenario": scenario,
                        "ordinal": len(requests),
                        "request_input_sha256": canonical_sha(original),
                        "required_checks": sorted(CHECKS),
                    }
                )
        routes.append(
            {
                "route_id": rid,
                "model": model,
                "mode": mode,
                "expected_ranks": [0, 1, 2, 3],
                "requests": requests,
            }
        )
    return {
        "schema_version": 1,
        "run_id": run_id,
        "source_binding_id": fingerprint(binding_path, 64 * 1024 * 1024)["sha256"],
        "observer_sources": own,
        "observer_source_sha256": canonical_sha(own),
        "trace_directory": str(Path(trace_directory).absolute()),
        "sources": sources,
        "source_records": source_records,
        "gate_kit_files": gate,
        "required_role_sources": ROLE_SOURCES,
        "required_role_hooks": ROLE_HOOKS,
        "vendor_graph": {"module": "torch_musa.musa_graph.graphs"},
        "routes": routes,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-binding", required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--trace-directory", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--report", required=True)
    args = parser.parse_args()
    report = {
        "schema_version": 1,
        "actual_argv": sys.argv,
        "passed": False,
        "request_or_model_execution": False,
        "errors": [],
    }
    try:
        config = make(args.source_binding, args.run_id, args.trace_directory)
        ref = write_json(args.output, config)
        os.chmod(args.output, 0o400)
        report.update(
            passed=True,
            config=ref,
            observer_source_sha256=config["observer_source_sha256"],
        )
    except Exception:
        report["errors"].append(traceback.format_exc())
    write_json(args.report, report)
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
