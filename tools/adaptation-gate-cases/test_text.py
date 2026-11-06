"""Text adaptation gate cases."""

from case_utils import run_case


def test_text_single() -> None:
    run_case("text_single")


def test_text_concurrent_8() -> None:
    run_case("text_concurrent_8")
