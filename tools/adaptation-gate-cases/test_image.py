"""Image adaptation gate cases."""

from case_utils import run_case


def test_image_single() -> None:
    run_case("image_single")


def test_image_concurrent_8() -> None:
    run_case("image_concurrent_8")
