from __future__ import annotations

import importlib.util
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]


def load_script(name: str):
    path = ROOT / ".github" / "scripts" / name
    spec = importlib.util.spec_from_file_location(name.removesuffix(".py"), path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


detect = load_script("detect_platforms.py")
retry = load_script("run_with_retry.py")
command = load_script("ci_command.py")


def test_docs_only_selects_no_hardware():
    enabled = detect.enabled_platforms()
    assert detect.select_by_changes(["docs/ci.md", "README.md"], enabled) == []


def test_vendor_change_selects_owner():
    enabled = detect.enabled_platforms()
    selected = detect.select_by_changes(
        ["vllm_fl/dispatch/backends/vendor/kunlunxin/patch.py"], enabled
    )
    assert selected == ["kunlunxin"]


def test_each_enabled_platform_owns_its_config():
    enabled = detect.enabled_platforms()
    for platform in enabled:
        assert detect.select_by_changes(
            [f".github/configs/{platform}.yml"], enabled
        ) == [platform]


def test_unmapped_shared_change_runs_all():
    enabled = detect.enabled_platforms()
    assert detect.select_by_changes(["vllm_fl/worker/worker.py"], enabled) == enabled


def test_platform_labels_override_paths():
    enabled = detect.enabled_platforms()
    selected = detect.parse_label_override(
        json.dumps(["tests", "platform:kunlunxin", "platform:thead"]), enabled
    )
    assert selected == [name for name in enabled if name in {"kunlunxin", "thead"}]


def test_unknown_platform_label_falls_back_to_all():
    enabled = detect.enabled_platforms()
    assert detect.parse_label_override('["platform:unknown"]', enabled) == enabled


def test_retry_classifier_is_narrow():
    assert retry.is_transient("HTTP Error 503: service unavailable")
    assert retry.is_transient("Could not resolve host: github.com")
    assert not retry.is_transient("AssertionError: tensors differ")
    assert not retry.is_transient("CUDA out of memory")


def test_transient_failure_is_retried_once(monkeypatch):
    outcomes = iter([(1, "HTTP Error 503"), (0, "ok")])
    calls = []
    monkeypatch.setattr(
        retry, "run", lambda value: calls.append(value) or next(outcomes)
    )
    monkeypatch.setattr(retry.time, "sleep", lambda _: None)

    assert retry.main(["--delay", "0", "--", "command"]) == 0
    assert calls == [["command"], ["command"]]


def test_test_failure_is_not_retried(monkeypatch):
    calls = []
    monkeypatch.setattr(
        retry,
        "run",
        lambda value: calls.append(value) or (1, "AssertionError: tensors differ"),
    )

    assert retry.main(["--", "command"]) == 1
    assert calls == [["command"]]


def test_ci_command_parsing():
    assert command.parse_command("/ci run kunlunxin,thead") == "kunlunxin,thead"
    assert command.parse_command("/test") == "all"


def test_ci_command_rejects_unknown_platform():
    try:
        command.parse_command("/ci run mystery")
    except ValueError as exc:
        assert "unavailable platform" in str(exc)
    else:
        raise AssertionError("unknown platform should be rejected")


def test_ci_command_rejects_disabled_platform():
    try:
        command.parse_command("/ci run ascend")
    except ValueError as exc:
        assert "unavailable platform" in str(exc)
    else:
        raise AssertionError("disabled platform should be rejected")


def test_ci_command_does_not_accept_shell_text():
    try:
        command.parse_command("/ci run cuda; whoami")
    except ValueError as exc:
        assert "unavailable platform" in str(exc)
    else:
        raise AssertionError("shell text should be rejected")
