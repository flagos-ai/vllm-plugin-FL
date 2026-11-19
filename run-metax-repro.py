"""Clean-environment reproducible runner for the MetaX PR #743 coverage candidate.

Baseline repository:
  vllm-plugin-FL @ d6d83803980fb78d84dcc1b3bc5b915b1fddb237
Upstream test source:
  PR #743 (commit 608a2ff) + MetaX adaptations + MetaX native tests
Batch 1 RMSNorm / Activation source (bundled in candidate metax_tests/):
  - metax_tests/test_cov_rms_norm.py
  - metax_tests/test_cov_activation.py
MetaX Attention / MLA source (bundled in candidate metax_attention_tests/):
  - metax_attention_tests/test_cov_metax_attention_mla.py

Set METAX_C550_HOST to the expected worker hostname to restrict host execution.

Exit-code policy:
  Original UT requires all 3 sub-reports (report-metax.json, report-metax-gpu.json,
  report-metax-multigpu.json) to exist without collection errors, setup/teardown
  errors, or unexpected failures. Only normal test failures matching the 2 known
  single-card UT failures are tolerated to continue aggregation; any missing
  sub-report, collection error, unexpected failure, or unaccepted group makes the
  final exit code non-zero.

Historical reference (shared_plus_metax):
  18,423 / 22,888 executable lines (80.49%)
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import socket
import subprocess
import sys
import tempfile
import xml.etree.ElementTree as ET
from pathlib import Path

PILOT = Path("/data/metax-rmsnorm-pilot")
REPO = PILOT / "vllm-plugin-FL"
HOST = os.environ.get("METAX_C550_HOST")
REVISION = "d6d83803980fb78d84dcc1b3bc5b915b1fddb237"
FLAGGEMS_PIN = "80a8e9716abed1bac3fc0ba5fd4e9f650ae3f989"
IMAGE = (
    "harbor.baai.ac.cn/flagos-dev/vllm-plugin-fl@sha256:"
    "89464b079e9995115634e2deba6f09256f1afd49b97c88063dcd85d642ef269c"
)
BASE_DIR = Path(__file__).resolve().parent
TEST_ROOT = BASE_DIR / "coverage_tests"
METAX_ATTN_TEST = BASE_DIR / "metax_attention_tests" / "test_cov_metax_attention_mla.py"
BATCH1_ACT_RMS_DIR = BASE_DIR / "metax_tests"
VENDOR = "vllm_fl/dispatch/backends/vendor/"
HISTORICAL_HITS = 18423
HISTORICAL_EXEC = 22888

KNOWN_SINGLE_CARD_UT_FAILURES = {
    "tests/unit_tests/distributed/test_collective_compile.py::test_all_reduce_custom_op_keeps_ctypes_out_of_dynamo",
    "tests/unit_tests/ops/test_hip_activation_dispatch.py::test_hip_activations_keep_oot_dispatch_and_native_controls",
}

BATCH2_GROUP_DIRS = [
    "unit/kv_transfer",
    "unit/model_runner",
    "unit/worker",
]

# Static and runtime exclusions with explicit reasons
EXCLUDED_RECORDS = {
    "functional/backends/test_cov_cuda_activation_norm.py": "CUDA vendor only",
    "functional/backends/test_cov_cuda_backend.py": "CUDA vendor only",
    "functional/backends/test_cov_cuda_fused_moe.py": "CUDA vendor only",
    "functional/backends/test_cov_cuda_rotary.py": "CUDA vendor only",
    "functional/qwen3_8/test_cov_nvidia_fast_paths.py": "NVIDIA-specific fast paths",
    "functional/qwen3_8/test_cov_mtp.py": "PR #743 reports missing vLLM 0.24 API",
    "functional/backends/test_cov_tilelang_mhc_broadcast.py (7 cases)": "ModuleNotFoundError: No module named 'tilelang'",
    "functional/hy4 (7 cases)": "Requires NVIDIA CUDA runtime / FLASH_ATTN MLA prefill / _C::per_token_group_fp8_quant",
    "unit/platform/test_cov_platform_nvml_config.py::test_get_device_capability_vendor_paths": "NVIDIA-only (NvmlCudaPlatform / vllm._C_stable_libtorch)",
}

STATIC_EXCLUDED_FILES = {
    "functional/backends/test_cov_cuda_activation_norm.py",
    "functional/backends/test_cov_cuda_backend.py",
    "functional/backends/test_cov_cuda_fused_moe.py",
    "functional/backends/test_cov_cuda_rotary.py",
    "functional/qwen3_8/test_cov_nvidia_fast_paths.py",
    "functional/qwen3_8/test_cov_mtp.py",
}


def output(*args):
    print(*args, flush=True)


def check_output(command):
    return subprocess.check_output(command, text=True).strip()


def run_logged(command, log, *, env=None):
    with (
        log.open("wb") as dest,
        subprocess.Popen(
            command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, env=env
        ) as proc,
    ):
        for line in proc.stdout:
            dest.write(line)
            dest.flush()
            sys.stdout.buffer.write(line)
            sys.stdout.buffer.flush()
        return proc.wait()


def coverage_lines(path):
    report = json.loads(path.read_text(encoding="utf-8"))
    if report["meta"]["branch_coverage"]:
        raise ValueError(f"Expected line coverage in {path}")
    lines = {}
    for name, row in report["files"].items():
        hit = set(row["executed_lines"])
        executable = hit | set(row["missing_lines"])
        lines[name] = (hit, executable, set(row["excluded_lines"]))
    return lines


def is_metax_scope(path):
    return not path.startswith(VENDOR) or path.startswith(VENDOR + "metax/")


def scope_totals(lines_map, hits_override=None, *, metax_only=False):
    hits = 0
    executable = 0
    for fname, (orig_hit, exe, _) in lines_map.items():
        if metax_only and not is_metax_scope(fname):
            continue
        cur_hit = hits_override[fname] if hits_override is not None else orig_hit
        hits += len(cur_hit)
        executable += len(exe)
    pct = (100.0 * hits / executable) if executable else 0.0
    return {"hits": hits, "executable": executable, "coverage_pct": pct}


def count_results(path):
    if not path.is_file():
        return {}
    root = ET.parse(path).getroot()
    counts = dict(passed=0, failed=0, errors=0, skipped=0)
    for case in root.iter("testcase"):
        for tag, kind in (
            ("error", "errors"),
            ("failure", "failed"),
            ("skipped", "skipped"),
        ):
            if case.find(tag) is not None:
                counts[kind] += 1
                break
        else:
            counts["passed"] += 1
    return counts


def _group_files(rel_dir):
    folder = TEST_ROOT / rel_dir
    return [
        str(p)
        for p in sorted(folder.glob("test_*.py"))
        if p.relative_to(TEST_ROOT).as_posix() not in STATIC_EXCLUDED_FILES
    ]


def build_batch2_jobs():
    return [
        (grp, [], _group_files(grp), "batch2_upstream_608a2ff")
        for grp in BATCH2_GROUP_DIRS
    ]


def build_jobs():
    func_flags = [
        "-p",
        "tests.conftest",
        "-p",
        "tests.functional_tests.conftest",
        "--platform",
        "metax",
        "--device",
        "c550",
    ]
    b_nodes = [
        str(BASE_DIR / line)
        for line in (BASE_DIR / "passed_backends_nodeids.txt")
        .read_text(encoding="utf-8")
        .splitlines()
        if line.strip()
    ]
    h_nodes = [
        str(BASE_DIR / line)
        for line in (BASE_DIR / "passed_hy4_nodeids.txt")
        .read_text(encoding="utf-8")
        .splitlines()
        if line.strip()
    ]
    return [
        # Upstream PR #743 groups passing as-is
        ("unit/dispatch", [], _group_files("unit/dispatch"), "upstream_608a2ff"),
        ("unit/distributed", [], _group_files("unit/distributed"), "upstream_608a2ff"),
        (
            "unit/model_runner_tail",
            [],
            _group_files("unit/model_runner_tail"),
            "upstream_608a2ff",
        ),
        ("unit/ops", [], _group_files("unit/ops"), "upstream_608a2ff"),
        (
            "unit/quantization",
            [],
            _group_files("unit/quantization"),
            "upstream_608a2ff",
        ),
        (
            "functional/compilation",
            [],
            _group_files("functional/compilation"),
            "upstream_608a2ff",
        ),
        (
            "functional/worker",
            [],
            _group_files("functional/worker"),
            "upstream_608a2ff",
        ),
        (
            "functional/glm5",
            func_flags,
            _group_files("functional/glm5"),
            "upstream_608a2ff",
        ),
        (
            "functional/ops",
            func_flags,
            _group_files("functional/ops"),
            "upstream_608a2ff",
        ),
        (
            "functional/qwen3_8",
            func_flags,
            _group_files("functional/qwen3_8"),
            "upstream_608a2ff",
        ),
        # Exact passing nodeids recovered from partial upstream groups
        (
            "functional/backends_passed",
            func_flags,
            b_nodes,
            "upstream_608a2ff_subset",
        ),
        ("functional/hy4_passed", func_flags, h_nodes, "upstream_608a2ff_subset"),
        # MetaX-adapted upstream groups
        (
            "unit/compilation_adapted",
            [],
            _group_files("unit/compilation"),
            "metax_adapted",
        ),
        (
            "unit/platform_adapted",
            ["-k", "not test_get_device_capability_vendor_paths"],
            _group_files("unit/platform"),
            "metax_adapted",
        ),
        # Newly added MetaX native Attention / MLA group (outside generic #743/tests paths)
        ("metax_attention", func_flags, [str(METAX_ATTN_TEST)], "metax_new"),
    ]


def verify_against_baseline(cand, baseline, label):
    if cand.keys() != baseline.keys():
        raise ValueError(f"Coverage file sets differ from baseline in {label}")
    for fname, (hit, exe, exc) in cand.items():
        orig_hit, orig_exe, orig_exc = baseline[fname]
        if exe != orig_exe:
            raise ValueError(f"Executable lines mismatch in {label}::{fname}")
        if exc != orig_exc:
            raise ValueError(f"Excluded lines mismatch in {label}::{fname}")
        if orig_hit - hit:
            raise ValueError(f"Baseline hit lines lost in {label}::{fname}")


def run_job(group, extra_args, targets, source_kind, out, base_data, baseline):
    name = group.replace("/", "-")
    data = out / f".coverage-{name}"
    report = out / f"coverage-{name}.json"
    junit = out / f"{name}.xml"
    shutil.copyfile(base_data, data)
    output(f"--- {group} [{source_kind}] ({len(targets)} targets) ---")
    rc = run_logged(
        [
            "timeout",
            "--kill-after=30s",
            "15m",
            sys.executable,
            "-m",
            "pytest",
            "-q",
            "--tb=short",
            "-p",
            "no:cacheprovider",
            *extra_args,
            "--continue-on-collection-errors",
            "--cov=vllm_fl",
            "--cov-append",
            "--cov-config=pyproject.toml",
            f"--cov-report=json:{report}",
            f"--junitxml={junit}",
            *targets,
        ],
        out / f"{name}.log",
        env={**os.environ, "COVERAGE_FILE": str(data)},
    )
    summary = {
        "group": group,
        "source": source_kind,
        "exit_code": rc,
        "junit_xml": str(junit),
        "coverage_json": str(report),
        "log": str(out / f"{name}.log"),
    }
    try:
        summary.update(count_results(junit))
    except (OSError, ET.ParseError) as exc:
        summary["junit_error"] = str(exc)
    if rc == 0 and summary.get("passed", 0):
        try:
            cand = coverage_lines(report)
            verify_against_baseline(cand, baseline, group)
            summary["accepted"] = True
        except (OSError, ValueError, KeyError) as exc:
            summary["coverage_error"] = str(exc)
    summary.setdefault("accepted", False)
    output(json.dumps(summary))
    return summary


UT_REPORT_NAMES = (
    "report-metax.json",
    "report-metax-gpu.json",
    "report-metax-multigpu.json",
)


def evaluate_ut_reports(ut_rc, report_dir):
    """Validate all 3 UT sub-reports and only tolerate the 2 known call failures."""
    report_dir = Path(report_dir)
    missing_reports = []
    collection_errors = []
    invalid_exitcodes = []
    known = []
    unexpected = []

    for rname in UT_REPORT_NAMES:
        path = report_dir / rname
        if not path.is_file():
            missing_reports.append(rname)
            continue
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            collection_errors.append({"report": rname, "error": str(exc)})
            continue

        for collector in data.get("collectors", []):
            if collector.get("outcome") != "passed":
                collection_errors.append(
                    {
                        "report": rname,
                        "nodeid": collector.get("nodeid", ""),
                        "outcome": collector.get("outcome"),
                        "longrepr": collector.get("longrepr", ""),
                    }
                )

        summary = data.get("summary", {})
        if summary.get("error", 0):
            collection_errors.append(
                {
                    "report": rname,
                    "error_count": summary.get("error", 0),
                }
            )

        rep_known_count = 0
        rep_unexpected_count = 0
        for test in data.get("tests", []):
            outcome = test.get("outcome")
            if outcome in ("passed", "skipped", "xfailed", "xpassed"):
                continue
            nodeid = test.get("nodeid", "")
            tb = ""
            for phase in ("setup", "call", "teardown"):
                longrepr = test.get(phase, {}).get("longrepr")
                if longrepr:
                    tb = longrepr
                    break
            is_normal_call_fail = (
                outcome == "failed"
                and test.get("call", {}).get("outcome") == "failed"
                and test.get("setup", {}).get("outcome", "passed") == "passed"
                and test.get("teardown", {}).get("outcome", "passed") == "passed"
            )
            rec = {
                "report": rname,
                "nodeid": nodeid,
                "outcome": outcome,
                "traceback": tb,
            }
            if is_normal_call_fail and nodeid in KNOWN_SINGLE_CARD_UT_FAILURES:
                known.append(rec)
                rep_known_count += 1
            else:
                unexpected.append(rec)
                rep_unexpected_count += 1

        rep_rc = data.get("exitcode")
        if rep_rc == 0:
            if rep_known_count or rep_unexpected_count:
                invalid_exitcodes.append({"report": rname, "exitcode": rep_rc})
        elif rep_rc == 1:
            if rep_known_count == 0 or rep_unexpected_count > 0:
                invalid_exitcodes.append({"report": rname, "exitcode": rep_rc})
        else:
            invalid_exitcodes.append({"report": rname, "exitcode": rep_rc})

    ut_ok = (
        not missing_reports
        and not collection_errors
        and not invalid_exitcodes
        and not unexpected
        and ((ut_rc == 0 and not known) or (ut_rc == 1 and bool(known)))
    )
    return {
        "known_single_card_failures": known,
        "unexpected_failures": unexpected,
        "missing_reports": missing_reports,
        "collection_errors": collection_errors,
        "invalid_report_exitcodes": invalid_exitcodes,
        "only_known_single_card_failures": ut_ok,
    }


def run_original_ut_ft_and_batch1(out):
    for stale in (
        REPO / ".coverage",
        REPO / "coverage-metax.json",
        REPO / "report-metax.json",
        REPO / "report-metax-gpu.json",
        REPO / "report-metax-multigpu.json",
    ):
        if stale.is_file():
            stale.unlink()
    if (REPO / "pytest-diagnostics").exists():
        shutil.rmtree(REPO / "pytest-diagnostics")

    cov_ut_ft = out / ".coverage-ut-ft"
    env_cov = {**os.environ, "COVERAGE_FILE": str(cov_ut_ft)}

    # 1. Original UT via tests/run.py
    output("=== Stage 1: Original UT via tests/run.py ===")
    ut_rc = run_logged(
        [
            sys.executable,
            "tests/run.py",
            "--platform",
            "metax",
            "--device",
            "c550",
            "--scope",
            "unit",
        ],
        out / "original-ut.log",
        env=env_cov,
    )
    ut_cov_json = out / "coverage-original-ut.json"
    if (REPO / "coverage-metax.json").is_file():
        shutil.move(str(REPO / "coverage-metax.json"), str(ut_cov_json))

    for rname in UT_REPORT_NAMES:
        src = REPO / rname
        if src.is_file():
            shutil.move(str(src), str(out / rname))

    ut_eval = evaluate_ut_reports(ut_rc, out)
    output(
        f"Original UT exit_code={ut_rc}, "
        f"known={len(ut_eval['known_single_card_failures'])}, "
        f"unexpected={len(ut_eval['unexpected_failures'])}, "
        f"missing_reports={len(ut_eval['missing_reports'])}, "
        f"collection_errors={len(ut_eval['collection_errors'])}, "
        f"ok={ut_eval['only_known_single_card_failures']}"
    )

    # 2. Original FT via tests/run.py (appending to .coverage-ut-ft)
    output("=== Stage 2: Original FT via tests/run.py ===")
    ft_rc = run_logged(
        [
            sys.executable,
            "tests/run.py",
            "--platform",
            "metax",
            "--device",
            "c550",
            "--scope",
            "functional",
        ],
        out / "original-ft.log",
        env=env_cov,
    )
    ut_ft_cov_json = out / "coverage-ut-ft.json"
    if (REPO / "coverage-metax.json").is_file():
        shutil.move(str(REPO / "coverage-metax.json"), str(ut_ft_cov_json))

    # Clean diagnostics artifacts left by tests/run.py in REPO
    diag_dir = REPO / "pytest-diagnostics"
    if diag_dir.exists():
        shutil.move(str(diag_dir), str(out / "pytest-diagnostics"))

    if ft_rc != 0:
        raise RuntimeError(f"Original FT failed with exit_code={ft_rc}")

    # 3. Batch 1: RMSNorm + Activation 4 cases
    output("=== Stage 3: Batch 1 RMSNorm / Activation (4 cases) ===")
    act_rms_files = [
        str(BATCH1_ACT_RMS_DIR / "test_cov_rms_norm.py"),
        str(BATCH1_ACT_RMS_DIR / "test_cov_activation.py"),
    ]
    for f in act_rms_files:
        if not Path(f).is_file():
            raise RuntimeError(f"Missing Batch 1 test file: {f}")

    cov_act_rms = out / ".coverage-ut-ft-plus-act-rms"
    shutil.copyfile(cov_ut_ft, cov_act_rms)
    act_rms_json = out / "coverage-ut-ft-plus-act-rms.json"
    act_rms_xml = out / "batch1-act-rms.xml"
    act_rms_rc = run_logged(
        [
            sys.executable,
            "-m",
            "pytest",
            "-v",
            "--tb=short",
            "-p",
            "no:cacheprovider",
            "-p",
            "tests.conftest",
            "-p",
            "tests.functional_tests.conftest",
            "--platform",
            "metax",
            "--device",
            "c550",
            "--cov=vllm_fl",
            "--cov-append",
            "--cov-config=pyproject.toml",
            f"--cov-report=json:{act_rms_json}",
            f"--junitxml={act_rms_xml}",
            *act_rms_files,
        ],
        out / "batch1-act-rms.log",
        env={**os.environ, "COVERAGE_FILE": str(cov_act_rms)},
    )
    if act_rms_rc != 0:
        raise RuntimeError(
            f"Batch 1 RMSNorm/Activation failed with exit_code={act_rms_rc}"
        )

    ut_ft_lines = coverage_lines(ut_ft_cov_json)
    act_rms_lines = coverage_lines(act_rms_json)
    verify_against_baseline(act_rms_lines, ut_ft_lines, "batch1_act_rms")

    return (
        {
            "ut": {
                "exit_code": ut_rc,
                "coverage_json": str(ut_cov_json),
                "log": str(out / "original-ut.log"),
                **ut_eval,
            },
            "ft": {
                "exit_code": ft_rc,
                "coverage_json": str(ut_ft_cov_json),
                "log": str(out / "original-ft.log"),
            },
            "batch1_act_rms": {
                "exit_code": act_rms_rc,
                "coverage_json": str(act_rms_json),
                "junit_xml": str(act_rms_xml),
                "log": str(out / "batch1-act-rms.log"),
                **count_results(act_rms_xml),
            },
        },
        ut_ft_lines,
        cov_act_rms,
        act_rms_lines,
    )


def run_inside(out):
    os.chdir(REPO)
    os.environ["PATH"] = "/opt/conda/bin:" + os.environ["PATH"]
    paths = [str(BASE_DIR), str(REPO), os.environ.get("PYTHONPATH", "")]
    os.environ["PYTHONPATH"] = os.pathsep.join(value for value in paths if value)
    subprocess.run(
        ["git", "config", "--global", "--add", "safe.directory", str(REPO)],
        check=True,
    )
    if check_output(["git", "rev-parse", "HEAD"]) != REVISION:
        raise RuntimeError(f"Unexpected vllm-plugin-FL commit (expected {REVISION})")
    subprocess.run(
        [
            "git",
            "diff",
            "--quiet",
            "HEAD",
            "--",
            "vllm_fl",
            "tests",
            "pyproject.toml",
            ".github/scripts",
        ],
        check=True,
    )
    mirror = PILOT / "FlagGems-vllm"
    if check_output(["git", "-C", str(mirror), "rev-parse", "HEAD"]) != FLAGGEMS_PIN:
        raise RuntimeError(
            f"Unexpected FlagGems-vLLM mirror commit (expected {FLAGGEMS_PIN})"
        )
    subprocess.run(
        [
            "git",
            "config",
            "--global",
            f"url.{mirror.as_uri()}.insteadOf",
            "https://github.com/flagos-ai/FlagGems-vllm.git",
        ],
        check=True,
    )
    rc = run_logged(["bash", ".github/scripts/metax/setup.sh"], out / "setup.log")
    if rc:
        raise RuntimeError(f"MetaX setup failed, exit={rc}")
    subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "from tests.conftest import get_device, get_device_count; "
                "d=get_device(); n=get_device_count(); "
                "print('DEVICE_FIXTURE_CHECK', repr(d), 'count=', n); "
                "assert str(d) == 'cuda:0' and n == 1"
            ),
        ],
        check=True,
    )

    stage_info, ut_ft_lines, base_data, baseline = run_original_ut_ft_and_batch1(out)

    output("=== Stage 4: Batch 2 (3 groups) ===")
    batch2_records = []
    for group, extra_args, targets, source_kind in build_batch2_jobs():
        status = run_job(
            group, extra_args, targets, source_kind, out, base_data, baseline
        )
        batch2_records.append(status)

    output("=== Stage 5: Candidate 15 groups ===")
    statuses = []
    for group, extra_args, targets, source_kind in build_jobs():
        status = run_job(
            group, extra_args, targets, source_kind, out, base_data, baseline
        )
        statuses.append(status)
        all_statuses = batch2_records + statuses
        (out / "group-results.json").write_text(
            json.dumps(all_statuses, indent=2) + "\n", encoding="utf-8"
        )
        (out / "exit-codes.txt").write_text(
            "\n".join(f"{g['group']}={g['exit_code']}" for g in all_statuses) + "\n",
            encoding="utf-8",
        )

    # Merge fresh UT+FT+ActRMS baseline + fresh Batch 2 + fresh 15 candidate groups
    union_hits = {k: set(v[0]) for k, v in baseline.items()}
    for g in batch2_records + statuses:
        if g["accepted"]:
            cand = coverage_lines(Path(g["coverage_json"]))
            verify_against_baseline(cand, baseline, g["group"])
            for k, (h, _, _) in cand.items():
                union_hits[k].update(h)

    def build_scope_summary(*, metax_only: bool):
        ut_ft_s = scope_totals(ut_ft_lines, metax_only=metax_only)
        b1_s = scope_totals(baseline, metax_only=metax_only)
        final_s = scope_totals(
            baseline, hits_override=union_hits, metax_only=metax_only
        )
        return {
            "ut_ft_baseline": ut_ft_s,
            "ut_ft_plus_act_rms_baseline": b1_s,
            "final_union": final_s,
            "net_gain_vs_ut_ft": {
                "lines": final_s["hits"] - ut_ft_s["hits"],
                "percentage_points": final_s["coverage_pct"] - ut_ft_s["coverage_pct"],
            },
            "net_gain_vs_act_rms": {
                "lines": final_s["hits"] - b1_s["hits"],
                "percentage_points": final_s["coverage_pct"] - b1_s["coverage_pct"],
            },
        }

    shared_plus_metax = build_scope_summary(metax_only=True)
    shared_plus_metax["historical_reference"] = {
        "hits": HISTORICAL_HITS,
        "executable": HISTORICAL_EXEC,
        "coverage_pct": 100.0 * HISTORICAL_HITS / HISTORICAL_EXEC,
    }
    all_vllm_fl = build_scope_summary(metax_only=False)

    summary = {
        "baseline_repo": f"vllm-plugin-FL@{REVISION}",
        "upstream_test_source": "PR #743 (608a2ff)",
        "image": IMAGE,
        "flaggems_pin": FLAGGEMS_PIN,
        "original_stages": stage_info,
        "batch2_groups": batch2_records,
        "candidate_groups": statuses,
        "excluded_records": EXCLUDED_RECORDS,
        "all_vllm_fl": all_vllm_fl,
        "shared_plus_metax": shared_plus_metax,
    }
    (out / "summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    output("=== ALL_VLLM_FL ===")
    output(json.dumps(all_vllm_fl, indent=2))
    output("=== SHARED_PLUS_METAX ===")
    output(json.dumps(shared_plus_metax, indent=2))

    ut_ok = stage_info["ut"]["only_known_single_card_failures"]
    all_ok = ut_ok and all(g["accepted"] for g in batch2_records + statuses)
    return 0 if all_ok else 1


def run_host(card):
    if HOST and socket.gethostname().split(".", 1)[0] != HOST:
        raise RuntimeError(f"Expected host {HOST}")
    gpu = check_output(["mx-smi", "-i", str(card)])
    if "no process found" not in gpu.lower():
        raise RuntimeError(f"Card {card} could not be confirmed idle")
    out = Path(tempfile.mkdtemp(prefix="metax-repro-results.", dir=PILOT))
    output(f"RESULT_DIR={out}")
    return run_logged(
        [
            "docker",
            "run",
            "--rm",
            "--pull=never",
            "--ipc=host",
            "--device=/dev/dri",
            "--device=/dev/mxcd",
            "-v",
            f"{PILOT}:{PILOT}",
            "-v",
            f"{BASE_DIR}:/metax-repro:ro",
            "-w",
            str(REPO),
            "-e",
            f"MACA_VISIBLE_DEVICES={card}",
            "-e",
            "GEMS_VENDOR=metax",
            "-e",
            "VLLM_FL_EXPECTED_VENDOR=metax",
            "-e",
            "VLLM_PLUGINS=fl",
            "-e",
            "PYTHONDONTWRITEBYTECODE=1",
            IMAGE,
            "/opt/conda/bin/python",
            "/metax-repro/run-metax-repro.py",
            "--inside",
            str(out),
        ],
        out / "container.log",
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--card", type=int, choices=range(8))
    mode.add_argument("--inside", type=Path, help=argparse.SUPPRESS)
    mode.add_argument("--plan", action="store_true")
    args = parser.parse_args()
    if args.plan:
        for group, _, targets, kind in build_batch2_jobs() + build_jobs():
            print(f"{group:30s} [{kind:24s}] targets={len(targets)}")
        return 0
    if args.inside is not None:
        return run_inside(args.inside)
    return run_host(args.card)


if __name__ == "__main__":
    raise SystemExit(main())
