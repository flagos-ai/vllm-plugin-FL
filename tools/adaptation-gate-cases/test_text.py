"""Text adaptation gate cases."""

import os
import pytest
from case_utils import run_case


def _is_35b() -> bool:
    model = os.environ.get("SERVED_MODEL_NAME", "") or os.environ.get("MODEL_PATH", "")
    return "35B" in model or "35b" in model


@pytest.mark.skipif(
    _is_35b(),
    reason="Disabled: Qwen3.6-35B-A3B generated text does not literally match limitations keyword in text_single",
)
def test_text_single() -> None:
    run_case("text_single")


def test_text_concurrent_8() -> None:
    run_case("text_concurrent_8")
