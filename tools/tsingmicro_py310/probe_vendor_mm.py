#!/usr/bin/env python3
# Copyright (c) 2026 BAAI. All rights reserved.
"""Audit direct Tsingmicro GPU MM without dispatcher registration.

Use the same vendor login environment as the model worker, then prepend the
tools/tsingmicro_py310 bootstrap to PYTHONPATH when running Python 3.10.
--bf16-only audits BF16 without the FP32 router, which may fail its own stricter
gate. --stage selects subsets without changing the original case seeds.
"""

import argparse
import hashlib
import importlib
import importlib.metadata
import inspect
import json
import math
import os
import statistics
import time
import traceback
from pathlib import Path


def _case_plans(bf16_only=False, selected_stages=None):
    plans = [
        ("small_bfloat16", [(1, 128, 128, "bfloat16"), (8, 128, 128, "bfloat16")]),
        (
            "actual_bfloat16",
            [
                (m, k, n, "bfloat16")
                for k, n in ((5120, 17408), (8704, 5120))
                for m in (1, 8, 65)
            ],
        ),
        ("router_float32", [(8, 2048, 256, "float32")]),
        (
            "vision_bfloat16",
            [(m, 1152, 4304, "bfloat16") for m in (8, 65)],
        ),
        (
            "lm_head_bfloat16",
            [(m, 5120, 124160, "bfloat16") for m in (1, 8)],
        ),
        (
            "max_prefill_bfloat16",
            [(512, 5120, 17408, "bfloat16"), (512, 8704, 5120, "bfloat16")],
        ),
        (
            "gdn_narrow_bfloat16",
            [
                (1, 5120, 48, "bfloat16"),
                (65, 5120, 48, "bfloat16"),
                (1, 2048, 32, "bfloat16"),
                (65, 2048, 32, "bfloat16"),
            ],
        ),
    ]
    selected = set(selected_stages or ())
    result = []
    case_index = 0
    for stage, cases in plans:
        indexed = []
        for m, k, n, dtype in cases:
            if (not bf16_only or dtype == "bfloat16") and (
                not selected or stage in selected
            ):
                indexed.append((m, k, n, dtype, case_index))
            case_index += 1
        if indexed:
            result.append((stage, indexed))
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--bf16-only",
        action="store_true",
        help="Skip the FP32 router audit; preserve every selected BF16 case seed.",
    )
    parser.add_argument(
        "--stage",
        action="append",
        choices=[name for name, _ in _case_plans()],
        help="Run only selected stages (repeat to select multiple stages).",
    )
    args = parser.parse_args()
    plans = _case_plans(args.bf16_only, args.stage)
    if not plans:
        parser.error("No cases match --stage and --bf16-only")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    report = {
        "ok": False,
        "status": "initializing",
        "mask": os.environ.get("TXDA_VISIBLE_DEVICES"),
        "scale": 0.02,
        "bf16_only": args.bf16_only,
        "selected_stages": args.stage,
        "seed_base": 20261104,
        "stages": [],
        "cases": [],
        "thresholds": {
            "bfloat16": {"relative_l2": 0.01, "max_abs": 0.002},
            "float32": {"relative_l2": 0.0001, "max_abs": 0.00005},
        },
    }

    def persist():
        args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")

    def version(name):
        try:
            return importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            return None

    try:
        if report["mask"] != "0":
            raise RuntimeError("This audit requires TXDA_VISIBLE_DEVICES=0")
        torch = importlib.import_module("torch")
        torch_txda = importlib.import_module("torch_txda")
        torch.set_num_threads(32)
        torch.txda.set_device(0)
        triton = importlib.import_module("triton")
        flag_gems = importlib.import_module("flag_gems")
        device = torch.device("txda:0")
        module = importlib.import_module("flag_gems.runtime.backend._tsingmicro.ops.mm")
        mm = module.mm
        if mm.__module__ != "flag_gems.runtime.backend._tsingmicro.ops.mm":
            raise RuntimeError(f"Unexpected callable: {mm.__module__}")
        source_file = inspect.getfile(mm)
        report.update(
            torch_version=torch.__version__,
            torch_txda_version=getattr(
                torch_txda, "__version__", version("torch-txda")
            ),
            triton_version=triton.__version__,
            flagtree_version=version("flagtree"),
            flag_gems_version=getattr(flag_gems, "__version__", version("flag-gems")),
            callable_module=mm.__module__,
            callable_name=mm.__qualname__,
            callable_file=source_file,
            callable_source_sha256=hashlib.sha256(
                Path(source_file).read_bytes()
            ).hexdigest(),
            script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            cpu_threads=torch.get_num_threads(),
            device=str(device),
            python_path=os.environ.get("PYTHONPATH"),
            gems_vendor=os.environ.get("GEMS_VENDOR"),
            py310_compat=os.environ.get("FL_TXDA_PY310_COMPAT"),
            dispatcher_registration_performed=False,
            status="running",
        )
        persist()

        def summary(tensor):
            value = tensor.float().reshape(-1)
            finite = torch.isfinite(value)
            finite_values = value[finite]
            samples = [
                number if math.isfinite(number) else str(number)
                for number in value[:12].tolist()
            ]
            return {
                "shape": list(tensor.shape),
                "dtype": str(tensor.dtype),
                "device": str(tensor.device),
                "finite_count": int(finite.sum().item()),
                "nonfinite_count": int((~finite).sum().item()),
                "min": finite_values.min().item() if finite_values.numel() else None,
                "max": finite_values.max().item() if finite_values.numel() else None,
                "first12": samples,
            }

        def compare(actual, reference, limits):
            actual_f32, reference_f32 = actual.float(), reference.float()
            finite = bool(torch.isfinite(actual_f32).all().item())
            if finite:
                difference = actual_f32 - reference_f32
                max_abs = difference.abs().max().item()
                relative_l2 = torch.linalg.vector_norm(difference).item() / max(
                    torch.linalg.vector_norm(reference_f32).item(), 1e-12
                )
            else:
                max_abs = relative_l2 = None
            return {
                "ok": finite
                and relative_l2 <= limits["relative_l2"]
                and max_abs <= limits["max_abs"],
                "all_finite": finite,
                "relative_l2": relative_l2,
                "max_abs": max_abs,
            }

        def run_case(stage_name, m, k, n, dtype_name, index):
            case = {
                "stage": stage_name,
                "m": m,
                "k": k,
                "n": n,
                "dtype": dtype_name,
                "seed": report["seed_base"] + index,
                "status": "preparing",
                "ok": False,
            }
            report["cases"].append(case)
            persist()
            print(json.dumps({"event": "case_start", **case}), flush=True)
            dtype = getattr(torch, dtype_name)
            generator = torch.Generator(device="cpu").manual_seed(case["seed"])
            x = (
                torch.randn(
                    (m, k), dtype=torch.float32, device="cpu", generator=generator
                )
                * 0.02
            ).to(dtype)
            weight = (
                torch.randn(
                    (n, k), dtype=torch.float32, device="cpu", generator=generator
                )
                * 0.02
            ).to(dtype)
            reference = torch.nn.functional.linear(x.float(), weight.float()).to(dtype)
            x_gpu, weight_gpu = x.to(device=device), weight.to(device=device)
            rhs_gpu = weight_gpu.T
            assert not rhs_gpu.is_contiguous()
            assert x_gpu.device == device and weight_gpu.device == device
            case.update(
                x=summary(x),
                weight=summary(weight),
                reference=summary(reference),
                gpu_input_stride=list(x_gpu.stride()),
                gpu_weight_stride=list(weight_gpu.stride()),
                gpu_rhs_shape=list(rhs_gpu.shape),
                gpu_rhs_stride=list(rhs_gpu.stride()),
                gpu_rhs_contiguous=rhs_gpu.is_contiguous(),
                thresholds=report["thresholds"][dtype_name],
                status="cold_launch",
            )
            persist()
            torch.txda.current_stream(device).synchronize()
            started = time.perf_counter()
            output = mm(x_gpu, rhs_gpu)
            torch.txda.current_stream(device).synchronize()
            case["cold_wall_seconds"] = time.perf_counter() - started
            if (
                output.device != device
                or output.dtype != dtype
                or list(output.shape) != [m, n]
            ):
                raise AssertionError(
                    f"Unexpected output metadata: {output.device}, {output.dtype}, {output.shape}"
                )
            actual = output.cpu()
            case.update(
                cold_comparison=compare(actual, reference, case["thresholds"]),
                actual=summary(actual),
                status="validating_cold_output",
            )
            persist()
            if not case["cold_comparison"]["ok"]:
                case.update(
                    status="failed", failure="cold output exceeded original threshold"
                )
                persist()
                print(json.dumps({"event": "case_failed", **case}), flush=True)
                return False
            # The cold launch is the warmup. Each timed repeat is synchronized,
            # then separately bulk-copied and checked outside its timing window.
            samples, comparisons = [], []
            for repeat in range(3):
                started = time.perf_counter()
                output = mm(x_gpu, rhs_gpu)
                torch.txda.current_stream(device).synchronize()
                elapsed = (time.perf_counter() - started) * 1000
                actual = output.cpu()
                comparison = compare(actual, reference, case["thresholds"])
                samples.append(elapsed)
                comparisons.append(comparison)
                case.update(
                    milliseconds=samples,
                    repeat_comparisons=comparisons,
                    actual=summary(actual),
                )
                persist()
                if not comparison["ok"]:
                    case.update(
                        status="failed",
                        failure=f"repeat {repeat + 1} exceeded original threshold",
                    )
                    persist()
                    print(json.dumps({"event": "case_failed", **case}), flush=True)
                    return False
            case.update(
                status="passed",
                ok=True,
                warmup=1,
                repeats=3,
                median_ms=statistics.median(samples),
                min_ms=min(samples),
                max_ms=max(samples),
            )
            persist()
            print(json.dumps({"event": "case_passed", **case}), flush=True)
            return True

        actual_bf16_selected = any(name == "actual_bfloat16" for name, _ in plans)
        report["planned_cases"] = [
            {
                "stage": stage,
                "m": m,
                "k": k,
                "n": n,
                "dtype": dtype,
                "seed": report["seed_base"] + case_index,
            }
            for stage, cases in plans
            for m, k, n, dtype, case_index in cases
        ]
        with torch.inference_mode():
            for stage_name, cases in plans:
                if (
                    stage_name
                    in (
                        "vision_bfloat16",
                        "lm_head_bfloat16",
                        "max_prefill_bfloat16",
                    )
                    and actual_bf16_selected
                    and not any(
                        stage["name"] == "actual_bfloat16"
                        and stage["status"] == "passed"
                        for stage in report["stages"]
                    )
                ):
                    report["stages"].append(
                        {
                            "name": stage_name,
                            "status": "skipped",
                            "reason": "actual BF16 shapes did not pass",
                        }
                    )
                    persist()
                    continue
                stage = {
                    "name": stage_name,
                    "status": "running",
                    "planned_count": len(cases),
                }
                report["stages"].append(stage)
                persist()
                passed = True
                for m, k, n, dtype, case_index in cases:
                    if not run_case(stage_name, m, k, n, dtype, case_index):
                        passed = False
                        break
                stage["status"] = "passed" if passed else "failed"
                persist()
                if not passed and stage_name == "small_bfloat16":
                    report["status"] = "stopped_after_small_failure"
                    break
        report["ok"] = len(report["stages"]) == len(plans) and all(
            stage["status"] == "passed" for stage in report["stages"]
        )
        report["status"] = "passed" if report["ok"] else report["status"]
        if not report["ok"] and report["status"] == "running":
            report["status"] = "failed"
    except BaseException as error:
        report.update(
            status="error", error=repr(error), traceback=traceback.format_exc()
        )
    persist()
    print(json.dumps({"event": "audit_summary", **report}), flush=True)
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
