# Copyright (c) 2026 BAAI. All rights reserved.

"""Helpers for benchmark smoke tests."""

from __future__ import annotations

import json
import os
import subprocess
from typing import Any


def load_benchmark_case() -> dict[str, Any]:
    """Load benchmark case config injected by tests/run.py."""
    raw = os.environ.get("FL_BENCHMARK_CASE")
    if not raw:
        raise RuntimeError("FL_BENCHMARK_CASE is not set")
    return json.loads(raw)


def to_cli_args(params: dict[str, Any], skip: set[str] | None = None) -> list[str]:
    """Convert a parameter mapping to CLI args.

    Underscores are converted to dashes to match vLLM CLI conventions.
    Boolean true and empty string values are treated as flags.
    """
    skip = skip or set()
    args: list[str] = []

    for key, value in params.items():
        if key in skip or value is None or value is False:
            continue

        flag = "--" + key.replace("_", "-")
        if value is True or value == "":
            args.append(flag)
        else:
            args.extend([flag, str(value)])

    return args


def run_command(command: list[str]) -> subprocess.CompletedProcess[str]:
    """Run command and print captured output for CI logs."""
    print("[benchmark] Command:", " ".join(command))

    env = os.environ.copy()
    local_no_proxy = "127.0.0.1,localhost,::1"
    for key in ("NO_PROXY", "no_proxy"):
        current = env.get(key, "")
        env[key] = ",".join(filter(None, [current, local_no_proxy]))

    result = subprocess.run(command, capture_output=True, text=True, env=env)
    print(result.stdout)
    print(result.stderr)
    return result
