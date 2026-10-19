#!/usr/bin/env python3
"""Run a command and retry once only for explicit transient network failures."""

from __future__ import annotations

import argparse
import os
import re
import subprocess
import time

TRANSIENT_PATTERNS = tuple(
    re.compile(pattern, re.IGNORECASE)
    for pattern in (
        r"temporary failure in name resolution",
        r"could not resolve host",
        r"connection reset by peer",
        r"remote end closed connection",
        r"read timed? ?out",
        r"connect timed? ?out",
        r"http(?: error)?\s+(?:502|503|504)\b",
        r"status(?: code)?[=: ]+(?:502|503|504)\b",
    )
)


def is_transient(output: str) -> bool:
    return any(pattern.search(output) for pattern in TRANSIENT_PATTERNS)


def run(command: list[str]) -> tuple[int, str]:
    process = subprocess.Popen(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        errors="replace",
        bufsize=1,
    )
    lines: list[str] = []
    assert process.stdout is not None
    for line in process.stdout:
        print(line, end="", flush=True)
        lines.append(line)
    return process.wait(), "".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--retries", type=int, default=int(os.getenv("CI_TRANSIENT_RETRIES", "1"))
    )
    parser.add_argument("--delay", type=float, default=5.0)
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args(argv)
    command = args.command[1:] if args.command[:1] == ["--"] else args.command
    if not command:
        parser.error("a command is required after --")

    for attempt in range(args.retries + 1):
        if attempt:
            print(f"::notice::Retrying transient failure ({attempt}/{args.retries})")
        returncode, output = run(command)
        if returncode == 0:
            if attempt:
                print("::warning::Command recovered after a transient CI failure")
            return 0
        if returncode < 0 or not is_transient(output) or attempt >= args.retries:
            return returncode or 1
        time.sleep(args.delay)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
