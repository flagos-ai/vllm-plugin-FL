"""Mixed text and image adaptation gate case."""

from case_utils import run_case


def test_mix_text_image_concurrent_8() -> None:
    run_case("mixed_concurrent_8")
