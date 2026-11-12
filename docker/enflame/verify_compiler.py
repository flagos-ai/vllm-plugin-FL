#!/usr/bin/env python3
# Copyright 2026 FlagOS Contributors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Verify the active Triton files against FlagTree's installed RECORD.

This guard uses only the standard library and does not import Triton or use a
GPU. Generated, unrecorded ``__pycache__/*.pyc`` files are reported separately.
"""

import argparse
import base64
import csv
import hashlib
import importlib.metadata
import importlib.util
import io
import json
import stat
import sys
from pathlib import Path, PurePosixPath

DEFAULT_VERSION = "0.7.0+enflame3.6"
MAX_SUMMARY_BYTES = 8192


def _file_digest(path: Path, expected_size: int) -> tuple[int, bytes]:
    before = path.stat()
    if not stat.S_ISREG(before.st_mode):
        raise ValueError("not a regular file")
    digest = hashlib.sha256()
    size = 0
    with path.open("rb") as stream:
        while chunk := stream.read(min(1024 * 1024, expected_size - size + 1)):
            size += len(chunk)
            digest.update(chunk)
            if size > expected_size:
                raise ValueError("file exceeds RECORD size")
    after = path.stat()
    fields = ("st_dev", "st_ino", "st_size", "st_mtime_ns")
    if any(getattr(before, key) != getattr(after, key) for key in fields):
        raise ValueError("file changed during verification")
    if size != expected_size:
        raise ValueError("file size differs from RECORD")
    return size, digest.digest()


def verify_compiler(expected_version: str = DEFAULT_VERSION) -> dict:
    summary = {
        "schema_version": 1,
        "passed": False,
        "scope": "FlagTree RECORD release files; no Triton import or GPU calls",
        "provider": {"name": "flagtree", "expected_version": expected_version},
        "recorded_triton_count": 0,
        "verified_release_files": 0,
        "verified_release_bytes": 0,
        "unverified_generated_count": 0,
        "error_count": 0,
        "errors": [],
    }

    def error(message: str) -> None:
        summary["error_count"] += 1
        if len(summary["errors"]) < 8:
            summary["errors"].append(message[:400])

    try:
        distribution = importlib.metadata.distribution("flagtree")
        summary["provider"]["version"] = distribution.version
        if distribution.version != expected_version:
            error("FlagTree version differs from expected version")
        root = Path(distribution.locate_file("")).resolve(strict=True)
        summary["provider"]["distribution_root"] = str(root)
        records = [
            entry
            for entry in (distribution.files or ())
            if str(entry).endswith(".dist-info/RECORD")
        ]
        if len(records) != 1:
            raise ValueError("expected exactly one FlagTree distribution RECORD")
        record_path = Path(distribution.locate_file(records[0])).resolve(strict=True)
        if not record_path.is_relative_to(root):
            raise ValueError("FlagTree RECORD is outside the distribution root")
        record_raw = record_path.read_bytes()
        summary["provider"]["record_path"] = str(record_path)
        summary["provider"]["record_bytes"] = len(record_raw)
        summary["provider"]["record_sha256"] = hashlib.sha256(record_raw).hexdigest()
        rows = list(csv.reader(io.StringIO(record_raw.decode("utf-8"))))
        selected = [row for row in rows if row and row[0].startswith("triton/")]
        summary["recorded_triton_count"] = len(selected)
        if not selected:
            raise ValueError("FlagTree RECORD contains no triton/ files")
        if sum(row[0] == "triton/__init__.py" for row in selected) != 1:
            raise ValueError("RECORD must contain exactly one triton/__init__.py")
        expected_origin = Path(distribution.locate_file("triton/__init__.py")).resolve(
            strict=True
        )
        spec = importlib.util.find_spec("triton")
        if spec is None or spec.origin in (None, "built-in", "frozen"):
            raise ValueError("active Triton has no regular module origin")
        active_origin = Path(spec.origin).resolve(strict=True)
        summary["active_triton_origin"] = str(active_origin)
        summary["expected_triton_origin"] = str(expected_origin)
        if not active_origin.samefile(expected_origin):
            error("active Triton origin is shadowing the FlagTree distribution")
        seen = set()
        for row in selected:
            relative = row[0]
            try:
                if len(row) != 3:
                    raise ValueError("RECORD row must have three columns")
                parts = relative.split("/")
                if "\\" in relative or any(part in ("", ".", "..") for part in parts):
                    raise ValueError("unsafe RECORD path")
                if relative in seen:
                    raise ValueError("duplicate RECORD path")
                seen.add(relative)
                path = Path(distribution.locate_file(PurePosixPath(relative))).resolve(
                    strict=True
                )
                if not path.is_relative_to(root):
                    raise ValueError("file resolves outside the distribution root")
                if not stat.S_ISREG(path.stat().st_mode):
                    raise ValueError("not a regular file")
                recorded_hash, recorded_size = row[1:]
                generated = (
                    len(parts) >= 3
                    and parts[-2] == "__pycache__"
                    and parts[-1].endswith(".pyc")
                )
                if not recorded_hash or not recorded_size:
                    if generated:
                        summary["unverified_generated_count"] += 1
                        continue
                    raise ValueError("release file is missing RECORD hash or size")
                algorithm, separator, encoded = recorded_hash.partition("=")
                if algorithm != "sha256" or not separator:
                    raise ValueError("RECORD hash must use sha256")
                expected_hash = base64.b64decode(
                    encoded + "=" * (-len(encoded) % 4), altchars=b"-_", validate=True
                )
                if len(expected_hash) != 32:
                    raise ValueError("invalid sha256 digest length")
                if not recorded_size.isascii() or not recorded_size.isdecimal():
                    raise ValueError("invalid RECORD size")
                size, digest = _file_digest(path, int(recorded_size))
                if digest != expected_hash:
                    raise ValueError("sha256 differs from FlagTree RECORD")
                summary["verified_release_files"] += 1
                summary["verified_release_bytes"] += size
            except (OSError, ValueError) as exc:
                error(f"{relative}: {exc}")
    except Exception as exc:
        error(f"{type(exc).__name__}: {exc}")
    summary["passed"] = summary["error_count"] == 0
    return summary


def _summary_bytes(summary: dict) -> bytes:
    # Bound unusually long environment paths without losing the failure status.
    result = dict(summary)
    result["provider"] = dict(summary["provider"])
    truncated = False
    for mapping in (result, result["provider"]):
        for key, value in list(mapping.items()):
            if isinstance(value, str) and len(value.encode("utf-8")) > 512:
                mapping[key] = value.encode("utf-8")[:512].decode("utf-8", "ignore")
                truncated = True
    result["display_truncated"] = truncated
    raw = json.dumps(result, ensure_ascii=False, sort_keys=True).encode("utf-8")
    if len(raw) + 1 > MAX_SUMMARY_BYTES:
        result["errors"] = ["Error details exceed summary limit"]
        result["display_truncated"] = True
        raw = json.dumps(result, ensure_ascii=False, sort_keys=True).encode("utf-8")
    if len(raw) + 1 > MAX_SUMMARY_BYTES:
        raise ValueError("summary exceeds byte limit")
    return raw + b"\n"


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--expected-version", default=DEFAULT_VERSION)
    args = parser.parse_args(argv)
    summary = verify_compiler(args.expected_version)
    sys.stdout.buffer.write(_summary_bytes(summary))
    return 0 if summary["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
