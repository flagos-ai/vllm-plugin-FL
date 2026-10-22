#!/usr/bin/env python3
# Copyright (c) 2026 BAAI. All rights reserved.
"""Compare installed output helpers in an owned native MUSA graph child."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import inspect
import json
import os
import sys
import traceback
from pathlib import Path

IMAGE = "harbor.baai.ac.cn/plugin/vllm-plugin-fl@sha256:8f865d2e2ee21e62b1c0c56f06079c072f9c2023d82fe7e240af4919fc75ce06"


def seal(path, value):
    data = (json.dumps(value, indent=2, sort_keys=True) + "\n").encode()
    with path.open("xb") as stream:
        stream.write(data)
        stream.flush()
        os.fsync(stream.fileno())


def worker(root):
    report = {
        "status": "STARTED",
        "cases": [],
        "pid": os.getpid(),
        "scope": "Native captured output descriptor/value/Tensor-owner controls only",
        "allocation_lifetime_certified": False,
        "model_equivalence_certified": False,
        "github_sha": os.environ.get("GITHUB_SHA"),
        "github_run_id": os.environ.get("GITHUB_RUN_ID"),
        "github_run_attempt": os.environ.get("GITHUB_RUN_ATTEMPT"),
        "declared_container_image": os.environ.get("MUSA_CI_SUITE_IMAGE"),
    }

    def fatal(exc):
        report["status"] = "FAIL"
        report["failure"] = {
            "type": type(exc).__name__,
            "message": str(exc),
            "traceback": traceback.format_exc(),
        }
        try:
            seal(root / "worker.json", report)
        finally:
            os._exit(1)

    try:
        guard_spec = importlib.util.spec_from_file_location(
            "_ownership_context_guard",
            Path(__file__).with_name("musa_graph_compare_worker.py"),
        )
        guard_module = importlib.util.module_from_spec(guard_spec)
        guard_spec.loader.exec_module(guard_module)
        guard = guard_module.FatalNativeContext
        import gc
        import weakref

        import torch
        import torch_musa

        from vllm.utils.torch_utils import weak_ref_tensors as upstream

        from vllm_fl.dispatch.backends.vendor.musa.compilation.graph import (
            _weak_ref_musa_tensors as backend,
        )

        if os.environ.get("MUSA_CI_SUITE_IMAGE") != IMAGE:
            raise RuntimeError("Fixed comparison image mismatch")
        if torch.musa.device_count() != 4:
            raise RuntimeError("Requires the four allocated visible MUSA devices")
        torch.musa.set_device(0)
        stream = torch.musa.Stream()
        report["runtime"] = {
            "torch": str(torch.__version__),
            "torch_musa": str(torch_musa.__version__),
            "sys_prefix": sys.prefix,
            "visible_device_count": 4,
            "device_ordinal": 0,
        }
        kept = []
        for label, helper in (("backend", backend), ("upstream_default", upstream)):
            source = Path(inspect.getsourcefile(helper))
            row = {
                "helper": label,
                "module": helper.__module__,
                "source_path": str(source),
                "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
            }
            report["cases"].append(row)
            with guard(torch.musa.stream(stream), fatal):
                value = torch.arange(32, device="musa", dtype=torch.float32).reshape(
                    4, 8
                )
                expected = value.sin()
                torch.musa.synchronize()
                graph = torch.musa.MUSAGraph()
                with guard(torch.musa.graph(graph, stream=stream), fatal):
                    owner = value.sin()
                    view = owner[1:, 1::2]
                    owner_ref, view_ref = weakref.ref(owner), weakref.ref(view)
                    original = {"owner": owner, "view": view, "repeat": view}
                    descriptors = helper(original)
                    metadata = {
                        "size": tuple(view.size()),
                        "stride": tuple(view.stride()),
                        "offset": view.storage_offset(),
                        "address": view.data_ptr(),
                        "storage_impl": view.untyped_storage()._cdata,
                        "dtype": str(view.dtype),
                        "device": str(view.device),
                        "layout": str(view.layout),
                    }
                row["view_metadata"] = metadata
                row["returns_original_structure"] = descriptors is original
                row["view_is_original_tensor"] = descriptors["view"] is view
                row["repeated_identity_preserved"] = (
                    descriptors["view"] is descriptors["repeat"]
                )
                row["view_same_data_pointer"] = (
                    descriptors["view"].data_ptr() == metadata["address"]
                )
                row["view_same_storage_impl"] = (
                    descriptors["view"].untyped_storage()._cdata
                    == metadata["storage_impl"]
                )
                assert row["view_same_data_pointer"]
                assert tuple(descriptors["view"].size()) == metadata["size"]
                assert tuple(descriptors["view"].stride()) == metadata["stride"]
                assert descriptors["view"].storage_offset() == metadata["offset"]
                del original, owner, view
                gc.collect()
                row["original_owner_tensor_alive"] = owner_ref() is not None
                row["original_view_tensor_alive"] = view_ref() is not None
                graph.replay()
                torch.musa.synchronize()
                assert torch.equal(descriptors["owner"].cpu(), expected.cpu())
                assert torch.equal(descriptors["view"].cpu(), expected[1:, 1::2].cpu())
                value.add_(1)
                expected = value.sin()
                graph.replay()
                torch.musa.synchronize()
                assert torch.equal(descriptors["owner"].cpu(), expected.cpu())
                assert torch.equal(descriptors["view"].cpu(), expected[1:, 1::2].cpu())
                row["capture_replay_value_checks"] = 4
                if label == "backend":
                    assert not row["original_owner_tensor_alive"]
                    assert not row["original_view_tensor_alive"]
                    assert row["repeated_identity_preserved"]
                    assert not row["view_same_storage_impl"]
                row["status"] = "PASS"
                seal(root / (label + ".json"), row)
                # Retain real graph/input/pool owners through every dereference.
                kept.append((graph, value, descriptors))
        report["status"] = "PASS"
        report["graphs_alive_through_all_dereferences"] = len(kept) == 2
        seal(root / "worker.json", report)
        os._exit(0)
    except BaseException as exc:
        fatal(exc)


def execution_passed(execution):
    return (
        execution.get("actual_exit") == 0
        and all(
            execution.get(key) is True
            for key in (
                "process_started",
                "direct_process_reaped",
                "cleanup_confirmed",
                "safe_to_continue",
            )
        )
        and execution.get("remaining_owned_children_count") == 0
        and execution.get("deadline_exceeded") is False
        and execution.get("deadline_or_signal_exit") is False
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--worker", action="store_true")
    args = parser.parse_args()
    root = Path(args.output_dir).resolve()
    if args.worker:
        worker(root)
    root.mkdir(parents=True, exist_ok=False)
    if os.environ.get("MUSA_CI_SUITE_IMAGE") != IMAGE:
        seal(
            root / "execution.json",
            {
                "status": "FAIL",
                "reason": "fixed_image_mismatch",
                "gpu_children_started": 0,
            },
        )
        return 1
    spec = importlib.util.spec_from_file_location(
        "_ownership_owned_run", Path(__file__).with_name("run_ci_stage.py")
    )
    helper = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(helper)
    helper.prepare_owned_reaping()
    execution = helper.bounded_run(
        [
            sys.executable,
            "-I",
            "-B",
            str(Path(__file__).resolve()),
            "--worker",
            "--output-dir",
            str(root),
        ],
        300,
    )
    detail = None
    try:
        path = root / "worker.json"
        if path.stat().st_size > 1024 * 1024:
            raise ValueError("Ownership report too large")
        detail = json.loads(path.read_text())
    except (OSError, ValueError):
        pass
    passed = (
        execution_passed(execution)
        and isinstance(detail, dict)
        and detail.get("status") == "PASS"
        and detail.get("github_sha") == os.environ.get("GITHUB_SHA")
        and detail.get("github_run_id") == os.environ.get("GITHUB_RUN_ID")
        and detail.get("github_run_attempt") == os.environ.get("GITHUB_RUN_ATTEMPT")
        and detail.get("declared_container_image") == IMAGE
        and detail.get("graphs_alive_through_all_dereferences") is True
        and isinstance(detail.get("cases"), list)
        and [row.get("helper") for row in detail["cases"]]
        == ["backend", "upstream_default"]
        and all(
            row.get("status") == "PASS" and row.get("capture_replay_value_checks") == 4
            for row in detail["cases"]
        )
    )
    seal(
        root / "execution.json",
        {
            "status": "PASS" if passed else "FAIL",
            "execution": execution,
            "worker": detail,
            "post_failure_gpu_queries": False,
        },
    )
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
