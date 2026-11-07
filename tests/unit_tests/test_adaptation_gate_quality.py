import importlib.util
from pathlib import Path

import pytest

SOURCE = (
    Path(__file__).resolve().parents[2] / "tools/adaptation-gate-cases/case_utils.py"
)
spec = importlib.util.spec_from_file_location("adaptation_gate_quality", SOURCE)
gate = importlib.util.module_from_spec(spec)
spec.loader.exec_module(gate)


@pytest.mark.parametrize(
    "text",
    ["4ally", "4a", "4\n                    \n42", "4en19 + 23 = 42", "142", "420"],
)
def test_number_only_request_rejects_corrupt_or_embedded_answers(text):
    request = gate.text_request("text-01", 0)
    checks = gate.quality_checks(
        text, request["required_terms"], expected_exact=request["expected_exact"]
    )
    assert not checks["exact_answer"]
    assert not all(checks.values())


@pytest.mark.parametrize("index", [0, 3, 6])
def test_number_only_answers_accept_only_the_expected_number(index):
    request = gate.text_request("number", index)
    expected = request["expected_exact"]
    checks = gate.quality_checks(
        f" \n{expected}\n", request["required_terms"], expected_exact=expected
    )
    assert all(checks.values())


def test_mixed_scenario_keeps_numeric_answer_constraints():
    request = gate.scenario_requests("mixed_concurrent_8")[0]
    assert request["expected_exact"] == "42"


def test_semantic_text_and_image_answers_keep_existing_checks():
    checks = gate.quality_checks("Pacific Ocean", ["pacific"])
    assert all(checks.values())
    assert "exact_answer" not in checks
