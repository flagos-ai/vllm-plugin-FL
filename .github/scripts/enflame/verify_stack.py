# Copyright (c) 2026 BAAI. All rights reserved.
"""Metadata gate; device/compiler execution is checked separately."""

import argparse
from importlib.metadata import version

from packaging.version import Version

RUNTIME = {
    "torch": "2.11.0+cpu",
    "torch-gcu": "2.11.0+3.8.20260713",
}
PACKAGES = {
    **RUNTIME,
    "flag-gems": "5.5.0.dev0",
    "flagtree": "0.7.0+enflame3.6",
    "triton-gcu": "3.6.0+1.0.20260722",
    "openai": "3.28.0",
    "pandas": "3.0.6",
}


def verify_stack(runtime_only=False):
    if not runtime_only:
        actual = version("vllm")
        parsed = Version(actual)
        if parsed.base_version != "0.32.0" or "empty" not in (parsed.local or "").split(
            "."
        ):
            raise RuntimeError(f"Expected vLLM 0.32.0 empty build, found {actual}")
    for package, wanted in (RUNTIME if runtime_only else PACKAGES).items():
        actual = version(package)
        matches = (
            Version(actual).public == wanted
            if package == "flag-gems"
            else actual == wanted
        )
        if not matches:
            raise RuntimeError(f"Expected {package} {wanted}, found {actual}")
    print(
        "Enflame runtime versions match"
        if runtime_only
        else "Enflame CI package versions match the vLLM 0.32 recipe"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runtime-only", action="store_true")
    verify_stack(parser.parse_args().runtime_only)
