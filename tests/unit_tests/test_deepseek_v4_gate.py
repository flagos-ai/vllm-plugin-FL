# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 FlagOS Contributors
"""Check the gate's answer normalization without sending model requests."""

import importlib.util
from pathlib import Path

import pytest

PATH = (
    Path(__file__).resolve().parents[2]
    / "tools/adaptation-gate-cases/test_deepseek_v4.py"
)
spec = importlib.util.spec_from_file_location("dsv4_gate_checks", PATH)
gate = importlib.util.module_from_spec(spec)
spec.loader.exec_module(gate)


@pytest.mark.parametrize(
    "text,term", [("H₂O", "h2o"), ("Seven.", "7"), ("PARIS", "paris")]
)
def test_valid_answer_forms(text, term):
    assert all(gate.quality(text, [term], 1).values())


@pytest.mark.parametrize("text", ["", "\ufffd", "!!!", "yes yes yes"])
def test_corruption_is_rejected(text):
    assert not all(gate.quality(text, ["yes"], 1).values())
