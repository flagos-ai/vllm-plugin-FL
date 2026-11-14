"""Reuse the committed adaptation requests without duplicating their assertions."""

import importlib.util
from functools import lru_cache
from pathlib import Path

SCENARIOS = (
    "text_single",
    "text_concurrent_8",
    "image_single",
    "image_concurrent_8",
    "mixed_concurrent_8",
)
GATE_ROOT = Path(__file__).resolve().parents[2] / "tools/adaptation-gate-cases"


@lru_cache(maxsize=1)
def load_gate():
    spec = importlib.util.spec_from_file_location(
        "fl_adaptation_gate_cases", GATE_ROOT / "case_utils.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module
