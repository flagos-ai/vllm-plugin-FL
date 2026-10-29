#!/usr/bin/env python3
# Copyright (c) 2026 BAAI. All rights reserved.
"""Run the shared PPU cases with explicit labels for the ordered text contract."""

import sys
from pathlib import Path

CASES = Path(__file__).resolve().parents[3] / "tools" / "adaptation-gate-cases"
ORDERED_SECTIONS = (
    " Use these exact section headings in this order: Large language models; "
    "Training; Inference; Limitations. Cover typical capabilities and responsible "
    "use within these sections."
)


def main():
    # This child is used only by the THead CI runner. The shared case files,
    # expected terms, quality checks and generation limits remain unchanged.
    sys.path.insert(0, str(CASES))
    import case_utils
    import pytest

    prompt, required = case_utils.LLM_DETAILS_TASK
    case_utils.LLM_DETAILS_TASK = (prompt + ORDERED_SECTIONS, required)
    return pytest.main(sys.argv[1:])


if __name__ == "__main__":
    raise SystemExit(main())
