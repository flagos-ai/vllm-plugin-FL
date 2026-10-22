"""No-argument CI wiring under the shell's whole-action timeout."""

import os
import re
import signal
import subprocess
import sys
import time
import traceback
import uuid
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from gate_common import read_bytes, read_json, require, write_json
from run_gate import cleanup_complete, interrupted, run_command

_DIAGNOSTIC_FILE_CAP = 64 * 1024
_DIAGNOSTIC_TEXT_CAP = 4096
_DIAGNOSTIC_LINES = 48
_SECRET_LINE = re.compile(
    r"\b(?:authorization|bearer)\b"
    r"|\b(?:[A-Za-z0-9]+[_-])*(?:token|secret|password|passwd|credential|api[_-]?key)"
    r"(?:[_-][A-Za-z0-9]+)*\b"
    r"|https?://[^/\s]*@"
    r"|^\s*(?:env:|[A-Z][A-Z0-9_]{2,}\s*=)",
    re.IGNORECASE,
)


def _diagnostic_text(label, text):
    """Prefix every bounded line so child output cannot issue workflow commands."""
    if not isinstance(text, str):
        return
    # Filter complete original lines before taking any tail. Otherwise a long
    # credential value could lose its key and be printed as an innocuous suffix.
    sanitized = []
    for line in text.splitlines():
        if _SECRET_LINE.search(line):
            safe = "[credential/environment line omitted]"
        elif len(line) > _DIAGNOSTIC_TEXT_CAP:
            safe = "[oversized diagnostic line omitted]"
        else:
            safe = "".join(c if c.isprintable() or c == "\t" else "?" for c in line)
        sanitized.append(safe)
    text = "\n".join(sanitized)
    if len(text) > _DIAGNOSTIC_TEXT_CAP:
        text = (
            "[tail; earlier diagnostic text omitted]\n" + text[-_DIAGNOSTIC_TEXT_CAP:]
        )
    lines = text.splitlines()
    if len(lines) > _DIAGNOSTIC_LINES:
        lines = ["[earlier diagnostic lines omitted]", *lines[-_DIAGNOSTIC_LINES:]]
    for line in lines:
        print(f"MUSA Gate diagnostic [{label}] {line}", file=sys.stderr)


def _failure_diagnostics(inputs, report, phase):
    """Read only fixed small failure files; never dump argv, environment or binding."""
    for error in report["errors"][-2:]:
        _diagnostic_text("entry", error)
    files = {
        "preflight": ("preflight-command", "preflight.json"),
        "observer-config": ("config-command", "observer-config-report.json"),
    }
    if phase not in files:
        return
    command, name = files[phase]
    try:
        detail = read_json(inputs / name, _DIAGNOSTIC_FILE_CAP, inputs)
        for error in detail.get("errors", [])[:3]:
            _diagnostic_text(phase, error)
        _diagnostic_text(phase, detail.get("traceback"))
    except (OSError, ValueError, TypeError, AttributeError) as error:
        _diagnostic_text(
            phase, "Small failure report unavailable: " + type(error).__name__
        )
    try:
        stderr = read_bytes(
            inputs / command / "stderr.raw", _DIAGNOSTIC_FILE_CAP, inputs
        ).decode("utf-8", "replace")
        _diagnostic_text(phase + " stderr", stderr)
    except (OSError, ValueError) as error:
        _diagnostic_text(
            phase, "Small child stderr unavailable: " + type(error).__name__
        )


def main():
    signal.signal(signal.SIGTERM, interrupted)
    signal.signal(signal.SIGINT, interrupted)
    tools = Path(__file__).resolve().parent
    python = sys.executable
    if len(sys.argv) > 1:
        os.execv(
            python,
            [
                python,
                "-I",
                "-B",
                str(tools / "run_gate.py"),
                "--python",
                python,
                *sys.argv[1:],
            ],
        )
    repo = Path(os.environ.get("GITHUB_WORKSPACE", os.getcwd())).resolve(strict=True)
    kit = repo / "tools" / "adaptation-gate-cases"
    run_id = os.environ.get("MUSA_CI_RUN_ID") or uuid.uuid4().hex
    parent = Path(
        os.environ.get(
            "MUSA_CI_RESULT_ROOT",
            str(Path(os.environ.get("RUNNER_TEMP", "/tmp")) / "musa-gate"),
        )
    )
    parent.mkdir(parents=True, exist_ok=True)
    parent = parent.resolve(strict=True)
    inputs = parent / (run_id + "-inputs")
    inputs.mkdir()  # exclusive; no existing receipt or result can be reused
    output = parent / run_id
    cache = (
        Path(os.environ.get("MUSA_CI_CACHE_ROOT", "/root/.cache/musa-gate")) / run_id
    )
    cache.mkdir(parents=True)
    for name in ("triton", "flaggems", "torchinductor"):
        (cache / name).mkdir()
    cache = cache.resolve(strict=True)
    env = os.environ.copy()
    env.update(
        {
            "TRITON_CACHE_DIR": str(cache / "triton"),
            "FLAGGEMS_CACHE_DIR": str(cache / "flaggems"),
            "TORCHINDUCTOR_CACHE_DIR": str(cache / "torchinductor"),
            "VLLM_PLUGINS": "fl",
        }
    )
    report = {
        "schema_version": 1,
        "actual_argv": sys.argv,
        "run_id": run_id,
        "passed": False,
        "steps": [],
        "errors": [],
    }
    diagnostic_phase = "preflight"
    try:
        commit = os.environ.get("MUSA_CI_EXPECTED_COMMIT") or os.environ.get(
            "GITHUB_SHA"
        )
        if not commit:
            git = subprocess.run(
                ["git", "-C", str(repo), "rev-parse", "HEAD"],
                capture_output=True,
                timeout=30,
                check=False,
            )
            require(git.returncode == 0, "Cannot resolve actual CI checkout commit")
            commit = git.stdout.decode("ascii").strip()
        binding = inputs / "source-binding.json"
        preflight_output = inputs / "preflight.json"
        argv = [
            python,
            "-I",
            "-B",
            str(tools / "preflight.py"),
            "--repo",
            str(repo),
            "--expected-commit",
            commit,
            "--gate-dir",
            str(kit),
            "--cache-root",
            str(cache),
            "--create-source-binding",
            str(binding),
            "--output",
            str(preflight_output),
        ]
        step = run_command(
            argv,
            env,
            str(repo),
            inputs / "preflight-command",
            300,
            time.monotonic() + 330,
        )
        report["steps"].append(step)
        require(
            step["actual_exit"] == 0
            and not step["errors"]
            and cleanup_complete(step["cleanup"])
            and read_json(preflight_output)["passed"] is True,
            "Actual source/cache/device origin preflight failed",
        )
        diagnostic_phase = "observer-config"
        config = inputs / "observer-config.json"
        config_report = inputs / "observer-config-report.json"
        argv = [
            python,
            "-I",
            "-B",
            str(tools / "make_observer_config.py"),
            "--source-binding",
            str(binding),
            "--run-id",
            run_id,
            "--trace-directory",
            str(output / "observer-traces"),
            "--output",
            str(config),
            "--report",
            str(config_report),
        ]
        step = run_command(
            argv, env, str(repo), inputs / "config-command", 120, time.monotonic() + 150
        )
        report["steps"].append(step)
        config_result = read_json(config_report)
        require(
            step["actual_exit"] == 0
            and not step["errors"]
            and cleanup_complete(step["cleanup"])
            and config_result["passed"] is True,
            "Actual readonly observer source/request configuration failed",
        )
        report["passed"] = True
        write_json(inputs / "entry-result.json", report)
        argv = [
            python,
            "-I",
            "-B",
            str(tools / "run_gate.py"),
            "--python",
            python,
            "--repo",
            str(repo),
            "--expected-commit",
            commit,
            "--gate-dir",
            str(kit),
            "--source-binding",
            str(binding),
            "--cache-root",
            str(cache),
            "--run-id",
            run_id,
            "--output-root",
            str(output),
            "--observer-config",
            str(config),
            "--observer-source-sha256",
            config_result["observer_source_sha256"],
            "--model-27b",
            os.environ.get("MUSA_GATE_MODEL_27B", "/data/models/Qwen3.6-27B"),
            "--model-35b",
            os.environ.get("MUSA_GATE_MODEL_35B_A3B", "/data/models/Qwen3.6-35B-A3B"),
            "--ports",
            os.environ.get("MUSA_CI_PORTS", "18062,18063,18064,18065"),
        ]
        os.environ.update(
            {
                key: env[key]
                for key in (
                    "TRITON_CACHE_DIR",
                    "FLAGGEMS_CACHE_DIR",
                    "TORCHINDUCTOR_CACHE_DIR",
                )
            }
        )
        diagnostic_phase = "launch"
        os.execv(python, argv)
    except Exception:
        report["errors"].append(traceback.format_exc())
        if not (inputs / "entry-result.json").exists():
            write_json(inputs / "entry-result.json", report)
        else:
            write_json(inputs / "entry-launch-failure.json", report)
        print(f"MUSA Gate entry failed; evidence: {inputs}", file=sys.stderr)
        try:
            _failure_diagnostics(inputs, report, diagnostic_phase)
        except Exception as error:
            # Diagnostics cannot bypass the original entry failure or launch a route.
            print(
                "MUSA Gate failure diagnostics unavailable: " + type(error).__name__,
                file=sys.stderr,
            )
        return 1
    return None


if __name__ == "__main__":
    raise SystemExit(main())
