#!/usr/bin/env python3
"""Extract one pinned public archive into a NEW private source tree; no imports/install."""

import argparse
import hashlib
import io
import json
import os
import re
import stat
import tarfile
from pathlib import Path, PurePosixPath

MANIFEST_SHA256 = "f38c79d091c426cc311dda21ef0e941f78cf4e2991db6ee508dcc37ec21c492d"
DEFAULT_PROFILE = "historical-cf174"
PROFILES = {
    DEFAULT_PROFILE: ("source-patch-manifest.json", MANIFEST_SHA256),
    "latest-bfc6": (
        "source-patch-manifest-latest-bfc6.json",
        "6bb6708e75b973136a4bdf2e0fd0ebef45093e06d5c03948507366209253b691",
    ),
    "latest-bfc6-ndsafe512": (
        "source-patch-manifest-latest-bfc6-ndsafe512.json",
        "8bc1e37ac07e2473e58ad0f370008034d1aaf33415b89b873bc3a3001f8793db",
    ),
}
PATCH_FILES = {
    "production.patch": {
        "bytes": 4900,
        "sha256": "d27c93abc17cfcaf0d50556e15461f1fbb14bb58a5403e2fd4e3bc6fae8e3f12",
    },
    "musa-nd-safe512.patch": {
        "bytes": 1449,
        "sha256": "b2a06482710863ac5707609f6500058ffeceaa413edc814e64f88b3272fff258",
    },
}
BUNDLE = Path(__file__).resolve(strict=True).parent


def require(value, message):
    if not value:
        raise ValueError(message)


def unique_object(pairs):
    result = {}
    for key, value in pairs:
        require(key not in result, "Duplicate JSON key")
        result[key] = value
    return result


def regular_bytes(path, limit, expected=None):
    path = Path(path).absolute()
    require(".." not in path.parts, "Noncanonical path")
    for parent in path.parents:
        require(stat.S_ISDIR(parent.lstat().st_mode), "Symlink/non-directory ancestor")
    before = path.lstat()
    require(
        stat.S_ISREG(before.st_mode) and before.st_size <= limit, "Regular file/cap"
    )
    fd = os.open(path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
    with os.fdopen(fd, "rb") as stream:
        opened = os.fstat(stream.fileno())
        require(
            (opened.st_dev, opened.st_ino, opened.st_size)
            == (before.st_dev, before.st_ino, before.st_size),
            "Opened identity changed",
        )
        body = stream.read(limit + 1)
        after = os.fstat(stream.fileno())
    require(
        len(body) == before.st_size
        and (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns)
        == (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns),
        "Read changed",
    )
    proof = {"bytes": len(body), "sha256": hashlib.sha256(body).hexdigest()}
    if expected is not None:
        require(
            proof == {key: expected[key] for key in ["bytes", "sha256"]},
            "Whole-file pin mismatch",
        )
    return body, proof


def exclusive_write(path, body):
    fd = os.open(
        path, os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0), 0o600
    )
    with os.fdopen(fd, "wb") as stream:
        stream.write(body)
        stream.flush()
        os.fsync(stream.fileno())


def inventory(source):
    rows = {}
    for path in sorted(source.rglob("*")):
        info = path.lstat()
        require(
            stat.S_ISDIR(info.st_mode) or stat.S_ISREG(info.st_mode),
            "Special source member",
        )
        if stat.S_ISREG(info.st_mode):
            _, row = regular_bytes(path, 16 * 1024 * 1024)
            rows[path.relative_to(source).as_posix()] = row
    return rows


def load_profile(profile):
    require(profile in PROFILES, "Unknown fixed recipe profile")
    filename, expected_sha256 = PROFILES[profile]
    manifest_bytes, manifest_fp = regular_bytes(BUNDLE / filename, 64 * 1024)
    require(manifest_fp["sha256"] == expected_sha256, "Fixed recipe manifest mismatch")
    manifest = json.loads(
        manifest_bytes,
        object_pairs_hook=unique_object,
        parse_constant=lambda value: (_ for _ in ()).throw(ValueError(value)),
    )
    return manifest, manifest_fp


def parse_patch(body):
    """Accept only the fixed unified-diff format, preserving every source byte."""
    lines = body.splitlines(keepends=True)
    files = {}
    index = 0
    while index < len(lines):
        require(lines[index].startswith(b"--- a/"), "Exact patch old path")
        name = lines[index][6:-1].decode("utf-8")
        require(
            lines[index] == f"--- a/{name}\n".encode()
            and index + 1 < len(lines)
            and lines[index + 1] == f"+++ b/{name}\n".encode()
            and name not in files,
            "Exact unique patch paths",
        )
        index += 2
        hunks = []
        while index < len(lines) and lines[index].startswith(b"@@ "):
            match = re.fullmatch(rb"@@ -(\d+),(\d+) \+(\d+),(\d+) @@\n", lines[index])
            require(match is not None, "Exact patch hunk header")
            old_start, old_count, new_start, new_count = map(int, match.groups())
            require(old_start > 0 and new_start > 0, "Positive patch hunk positions")
            index += 1
            old, new = [], []
            while index < len(lines) and lines[index][:1] in {b" ", b"+", b"-"}:
                if lines[index].startswith(b"--- a/"):
                    break
                line = lines[index]
                require(line.endswith(b"\n"), "Complete patch line")
                if line[:1] != b"+":
                    old.append(line[1:])
                if line[:1] != b"-":
                    new.append(line[1:])
                index += 1
            require(
                len(old) == old_count and len(new) == new_count,
                "Exact patch hunk line counts",
            )
            hunks.append((old_start, new_start, old, new))
        require(hunks, "Patch file must have hunks")
        files[name] = hunks
    require(files, "Patch must contain files")
    return files


def apply_exact_patch(body, hunks, expected):
    """Use the specified positions only; no offset search, context fuzz or Git."""
    original = body.splitlines(keepends=True)
    result = []
    cursor = 0
    for old_start, new_start, old, new in hunks:
        start = old_start - 1
        require(cursor <= start <= len(original), "Ordered exact patch position")
        result.extend(original[cursor:start])
        require(len(result) == new_start - 1, "Exact patch new position")
        require(original[start : start + len(old)] == old, "Exact patch context")
        result.extend(new)
        cursor = start + len(old)
    result.extend(original[cursor:])
    payload = b"".join(result)
    require(
        {"bytes": len(payload), "sha256": hashlib.sha256(payload).hexdigest()}
        == expected,
        "Whole-file patch postimage mismatch",
    )
    return payload


def load_patches(profile):
    names = ["production.patch"]
    if profile == "latest-bfc6-ndsafe512":
        names.append("musa-nd-safe512.patch")
    files, proofs = {}, {}
    for name in names:
        body, proofs[name] = regular_bytes(BUNDLE / name, 64 * 1024, PATCH_FILES[name])
        parsed = parse_patch(body)
        require(not files.keys() & parsed.keys(), "No duplicate patch file")
        files.update(parsed)
    return files, proofs


def prepare(archive_path, output_path, variant, profile=DEFAULT_PROFILE):
    manifest, manifest_fp = load_profile(profile)
    require(variant in {"master", "corrected"}, "Unknown variant")
    expected = manifest["archive"]
    raw, archive_fp = regular_bytes(archive_path, 32 * 1024 * 1024, expected)
    replacement_key, replacement_count = (
        ("six_production_replacements", 6)
        if profile == "latest-bfc6-ndsafe512"
        else ("four_production_replacements", 4)
    )
    replacements = manifest[replacement_key]
    patch_files, patch_proofs = load_patches(profile)
    require(
        len(replacements) == replacement_count
        and len({row["source_path"] for row in replacements}) == replacement_count
        and set(patch_files) == {row["source_path"] for row in replacements},
        "Exact profile production-file count and patch scope",
    )
    output = Path(output_path).absolute()
    require(
        ".." not in output.parts and not output.exists(),
        "Output must be new and normalized",
    )
    for parent in output.parents:
        require(
            stat.S_ISDIR(parent.lstat().st_mode),
            "Output parent must already exist, without symlinks",
        )
    output.mkdir(mode=0o700)
    source = output / "FlagGems"
    source.mkdir(mode=0o700)
    with tarfile.open(fileobj=io.BytesIO(raw), mode="r:gz") as archive:
        members = archive.getmembers()
        require(len(members) == expected["members"], "Exact public member count")
        seen = set()
        total = 0
        for member in members:
            parts = PurePosixPath(member.name).parts
            canonical = "/".join(parts)
            require(
                parts
                and parts[0] == expected["root"]
                and member.name.rstrip("/") == canonical
                and ".." not in parts
                and ".git" not in parts
                and "\\" not in member.name
                and not member.name.startswith("/"),
                "Canonical archive member",
            )
            require(
                canonical not in seen and (member.isdir() or member.isfile()),
                "Duplicate/special member",
            )
            seen.add(canonical)
            if len(parts) == 1:
                require(member.isdir(), "Archive root must be directory")
                continue
            target = source.joinpath(*parts[1:])
            if member.isdir():
                target.mkdir(mode=0o700, parents=True, exist_ok=True)
                continue
            total += member.size
            require(
                member.size <= 16 * 1024 * 1024 and total <= expected["expanded_bytes"],
                "Expanded cap",
            )
            target.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
            stream = archive.extractfile(member)
            require(stream is not None, "Regular member stream")
            body = stream.read(member.size + 1)
            require(len(body) == member.size, "Member EOF mismatch")
            exclusive_write(target, body)
            target.chmod(0o644)
        require(total == expected["expanded_bytes"], "Exact expanded byte total")
    before = inventory(source)
    require(len(before) == expected["regular_files"], "Exact fresh regular file count")
    patch_inputs = []
    for row in replacements:
        require(
            before[row["source_path"]] == row["preimage"], "Exact production preimage"
        )
        body, _ = regular_bytes(
            source / row["source_path"], 128 * 1024, row["preimage"]
        )
        payload = apply_exact_patch(body, patch_files[row["source_path"]], row["after"])
        patch_inputs.append((row, payload))
    if variant == "corrected":
        for row, payload in patch_inputs:
            path = source / row["source_path"]
            regular_bytes(path, 128 * 1024, row["preimage"])
            fd = os.open(path, os.O_WRONLY | os.O_TRUNC | getattr(os, "O_NOFOLLOW", 0))
            with os.fdopen(fd, "wb") as stream:
                stream.write(payload)
                stream.flush()
                os.fsync(stream.fileno())
            regular_bytes(path, 128 * 1024, row["after"])
    after = inventory(source)
    changed = sorted(name for name in before if before[name] != after.get(name))
    allowed = (
        sorted(row["source_path"] for row, _ in patch_inputs)
        if variant == "corrected"
        else []
    )
    require(
        set(before) == set(after) and changed == allowed,
        "Exact original-file change scope",
    )
    report = {
        "schema_version": 1,
        "scope": "Private source preparation only; no build/install/GPU",
        "profile": profile,
        "source_commit": manifest["source_commit"],
        "archive": archive_fp,
        "manifest": manifest_fp,
        "patches": patch_proofs,
        "variant": variant,
        "explicit_archive_version": manifest["versions"][variant],
        "SCM_version_is_true_Git_version": False,
        "PR_whole_head_installed": False,
        "source_before": before,
        "source_after": after,
        "changed_original_files": changed,
        "unchanged_original_file_count": len(before) - len(changed),
        "Q_padding_patch_included": False,
        "whole_model_gate_acceptance": False,
    }
    exclusive_write(
        output / "source-binding.json", (json.dumps(report, indent=2) + "\n").encode()
    )
    print(
        json.dumps(
            {
                "source": str(source),
                "binding": str(output / "source-binding.json"),
                "profile": profile,
                "variant": variant,
                "changed_original_files": changed,
                "explicit_archive_version": manifest["versions"][variant],
            }
        )
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", required=True)
    parser.add_argument(
        "--output", required=True, help="New leaf below an existing private parent"
    )
    parser.add_argument(
        "--variant", choices=["master", "corrected"], default="corrected"
    )
    parser.add_argument(
        "--profile",
        choices=sorted(PROFILES),
        default=DEFAULT_PROFILE,
        help="Fixed source pin; historical profile remains the default",
    )
    arguments = parser.parse_args()
    prepare(arguments.archive, arguments.output, arguments.variant, arguments.profile)


if __name__ == "__main__":
    main()
