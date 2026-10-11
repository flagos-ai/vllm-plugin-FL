"""Shared, stdlib-only evidence readers for the MUSA Gate draft."""

import hashlib
import json
import math
import os
from pathlib import Path
import stat

SCENARIOS = {
    "text_single": 1,
    "text_concurrent_8": 8,
    "image_single": 1,
    "image_concurrent_8": 8,
    "mixed_concurrent_8": 8,
}
CHECKS = frozenset({
    "non_empty", "minimum_length", "expected_semantics", "expected_order",
    "no_bang_triplet", "no_mojibake", "no_control_characters",
    "no_long_character_run", "no_repeated_word_run", "no_repeated_phrase",
})
TEST_FILES = ("test_text.py", "test_image.py", "test_mix_text_image.py")
GATE_FILES = (
    ".gitignore", "README.md", "case_utils.py", "generate_images.py",
    "run_serve.sh", "run_serve_eager.sh", "run_serve_graph.sh", "run_test.sh",
    *TEST_FILES, *(f"images/image_{n:02d}.png" for n in range(1, 9)),
)
ROUTES = (
    ("27b-eager", "Qwen3.6-27B", "eager", 18062),
    ("27b-graph", "Qwen3.6-27B", "graph", 18063),
    ("35b-a3b-eager", "Qwen3.6-35B-A3B", "eager", 18064),
    ("35b-a3b-graph", "Qwen3.6-35B-A3B", "graph", 18065),
)


def require(condition, message):
    if not condition:
        raise ValueError(message)


def _identity(info):
    return (info.st_dev, info.st_ino, info.st_mode, info.st_size,
            info.st_mtime_ns, info.st_ctime_ns)


def read_bytes(path, cap=8 * 1024 * 1024, root=None):
    """Read a complete, stable, canonical regular file; never follow a link."""
    path = Path(path).absolute()
    require(path.resolve(strict=True) == path, f"Noncanonical input: {path}")
    if root is not None:
        path.relative_to(Path(root).resolve(strict=True))
    before = path.lstat()
    require(stat.S_ISREG(before.st_mode), f"Not a regular file: {path}")
    require(not getattr(before, "st_file_attributes", 0) & 0x400,
            f"Reparse input: {path}")
    require(0 <= before.st_size <= cap, f"Input exceeds cap: {path}")
    fd = os.open(path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
                 | getattr(os, "O_BINARY", 0) | getattr(os, "O_CLOEXEC", 0))
    try:
        first = os.fstat(fd)
        cross_count = 5 if os.name == "nt" else 6
        require(_identity(before)[:cross_count] == _identity(first)[:cross_count],
                f"Changed input at open: {path}")
        chunks = []
        remaining = cap + 1
        while remaining:
            part = os.read(fd, min(65536, remaining))
            if not part:
                break
            chunks.append(part)
            remaining -= len(part)
        data = b"".join(chunks)
        require(len(data) == first.st_size and len(data) <= cap,
                f"Incomplete or oversized input: {path}")
        require(_identity(first) == _identity(os.fstat(fd)),
                f"Input changed while reading: {path}")
    finally:
        os.close(fd)
    require(_identity(before) == _identity(path.lstat()),
            f"Input path changed while reading: {path}")
    return data


def fingerprint(path, cap=8 * 1024 * 1024, root=None):
    data = read_bytes(path, cap, root)
    return {"path": str(Path(path).absolute()), "bytes": len(data),
            "sha256": hashlib.sha256(data).hexdigest()}



def fingerprint_streaming(path, cap=8 * 1024 * 1024, root=None):
    """Hash a complete, stable regular file without retaining its body."""
    path = Path(path).absolute()
    require(path.resolve(strict=True) == path, f"Noncanonical input: {path}")
    if root is not None:
        path.relative_to(Path(root).resolve(strict=True))
    before = path.lstat()
    require(stat.S_ISREG(before.st_mode), f"Not a regular file: {path}")
    require(not getattr(before, "st_file_attributes", 0) & 0x400,
            f"Reparse input: {path}")
    require(0 <= before.st_size <= cap, f"Input exceeds cap: {path}")
    fd = os.open(path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
                 | getattr(os, "O_BINARY", 0) | getattr(os, "O_CLOEXEC", 0))
    try:
        first = os.fstat(fd)
        cross_count = 5 if os.name == "nt" else 6
        require(_identity(before)[:cross_count] == _identity(first)[:cross_count],
                f"Changed input at open: {path}")
        digest, count = hashlib.sha256(), 0
        while count <= cap:
            part = os.read(fd, min(65536, cap + 1 - count))
            if not part:
                break
            digest.update(part)
            count += len(part)
        require(count == first.st_size and count <= cap,
                f"Incomplete or oversized input: {path}")
        require(_identity(first) == _identity(os.fstat(fd)),
                f"Input changed while reading: {path}")
    finally:
        os.close(fd)
    require(_identity(before) == _identity(path.lstat()),
            f"Input path changed while reading: {path}")
    return {"path": str(path), "bytes": count, "sha256": digest.hexdigest()}


def strict_json(data):
    def pairs(items):
        result = {}
        for key, value in items:
            require(key not in result, f"Duplicate JSON key: {key}")
            result[key] = value
        return result

    def reject(value):
        raise ValueError(f"Nonfinite JSON value: {value}")

    value = json.loads(data.decode("utf-8"), object_pairs_hook=pairs,
                       parse_constant=reject)
    def finite(item):
        if isinstance(item, float):
            require(math.isfinite(item), "Nonfinite JSON exponent")
        elif isinstance(item, dict):
            for child in item.values():
                finite(child)
        elif isinstance(item, list):
            for child in item:
                finite(child)
    finite(value)
    return value


def read_json(path, cap=8 * 1024 * 1024, root=None):
    return strict_json(read_bytes(path, cap, root))


def bound_json(ref, root=None, cap=8 * 1024 * 1024):
    require(set(ref) == {"path", "bytes", "sha256"}, "Bad fingerprint schema")
    require(fingerprint(ref["path"], cap, root) == ref, "Fingerprint mismatch")
    return read_json(ref["path"], cap, root)


def write_json(path, value):
    data = (json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)
            + "\n").encode("utf-8")
    with Path(path).open("xb") as stream:
        stream.write(data)
        stream.flush()
        os.fsync(stream.fileno())
    return fingerprint(path, max(len(data), 8 * 1024 * 1024))
