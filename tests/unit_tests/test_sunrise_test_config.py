# Copyright 2026 FlagOS Contributors
"""Sunrise case discovery and pytest selection in the unified runner."""

from pathlib import Path

import pytest

from tests.run import TestRunner as Runner


@pytest.mark.parametrize("stage", ["pr", "nightly"])
def test_sunrise_discovers_all_serving_cases(stage):
    runner = Runner("sunrise", "persistence_m", scope="e2e", stage=stage)
    cases = runner.discover_tests()
    assert [(case.task, case.model, case.case) for case in cases] == [
        ("serving", "qwen3_6", "27b_tp2_eager"),
        ("serving", "qwen3_6", "27b_tp2_graph"),
        ("serving", "qwen3_6", "35b_a3b_tp2_eager"),
        ("serving", "qwen3_6", "35b_a3b_tp2_graph"),
    ]


@pytest.mark.parametrize(
    ("scope", "representative"),
    [
        ("unit", "test_sunrise_topk"),
        ("functional", "test_sunrise_kv_cache"),
    ],
)
def test_sunrise_selectors_are_existing_pytest_module_names(
    monkeypatch, scope, representative
):
    monkeypatch.setattr("tests.utils.device_utils.get_device_count", lambda: 0)
    runner = Runner("sunrise", "persistence_m", scope=scope)
    for case in runner.discover_tests():
        args = case.extra_args
        keywords = args[args.index("-k") + 1].split(" or ")
        modules = {path.stem for path in Path(case.pytest_path).rglob("test_*.py")}
        assert representative in keywords
        assert set(keywords) <= modules
