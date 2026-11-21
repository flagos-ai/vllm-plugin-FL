# SPDX-License-Identifier: Apache-2.0
"""Prepare the pinned Git sources used by the DeepSeek V4.1 C550 example."""

import argparse
import json
import subprocess
from pathlib import Path


def prepare_vllm(checkout, pin):
    git = ["git", "-C", str(checkout)]
    revision = pin["revision"]
    if (
        subprocess.check_output([*git, "rev-parse", "HEAD"], text=True).strip()
        == revision
    ):
        return
    subprocess.run(
        [*git, "fetch", "--no-tags", pin["repository"], revision], check=True
    )
    # Preserve the release image's editable installation and native binaries.
    subprocess.run([*git, "merge", "--ff-only", revision], check=True)
    head = subprocess.check_output([*git, "rev-parse", "HEAD"], text=True).strip()
    if head != revision:
        raise SystemExit(
            f"{checkout} is ahead of the pinned vLLM revision. "
            "Use a separate checkout of the 0.32.0 image source."
        )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dependency", choices=["vllm"])
    parser.add_argument("checkout", type=Path, help="dedicated, writable Git checkout")
    args = parser.parse_args()
    manifest = json.loads(
        Path(__file__).with_name("operator_dependencies.json").read_text()
    )
    prepare_vllm(args.checkout, manifest["vllm_runtime"])


if __name__ == "__main__":
    main()
