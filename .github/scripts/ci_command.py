#!/usr/bin/env python3
"""Parse a trusted maintainer CI command without shell evaluation."""

from __future__ import annotations

import argparse
import re

SUPPORTED = {
    "cuda",
    "hygon",
    "metax",
    "musa",
    "enflame",
    "kunlunxin",
    "thead",
    "iluvatar",
}


def parse_command(body: str) -> str:
    first_line = body.splitlines()[0].strip().lower() if body.splitlines() else ""
    match = re.fullmatch(r"/(?:ci\s+run|test)(?:\s+(.+))?", first_line)
    if not match:
        raise ValueError(
            "expected `/ci run [platforms|all]` or `/test [platforms|all]`"
        )
    raw = (match.group(1) or "all").replace(",", " ")
    requested = [item for item in raw.split() if item]
    if requested == ["all"]:
        return "all"
    invalid = sorted(set(requested) - SUPPORTED)
    if invalid:
        raise ValueError(f"unavailable platform(s): {', '.join(invalid)}")
    return ",".join(name for name in sorted(SUPPORTED) if name in requested)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("body")
    args = parser.parse_args()
    try:
        print(parse_command(args.body))
    except ValueError as exc:
        print(exc)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
