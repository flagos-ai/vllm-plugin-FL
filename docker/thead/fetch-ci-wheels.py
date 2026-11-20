# Copyright 2026 FlagOS Contributors
"""Fetch the CI lock's exact wheels in parallel and produce an offline pip lock."""

import argparse
import hashlib
import os
import time
from concurrent.futures import ThreadPoolExecutor
from itertools import repeat
from pathlib import Path
from shutil import copyfile
from tempfile import NamedTemporaryFile
from urllib.parse import unquote, urlsplit
from urllib.request import urlopen


def sha256(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def fetch(line, cache, seed):
    name, _, url, hash_option = line.split()
    digest = hash_option.removeprefix("--hash=sha256:")
    parsed = urlsplit(url)
    if parsed.scheme != "https" or len(digest) != 64:
        raise ValueError(f"Expected an HTTPS wheel and SHA256 in {line!r}")
    filename = Path(unquote(parsed.path)).name
    if not filename.endswith(".whl"):
        raise ValueError(f"Expected a wheel in {line!r}")
    destination = cache / digest / filename
    destination.parent.mkdir(exist_ok=True, parents=True)
    seeded = seed / filename
    if not destination.is_file() and seeded.is_file():
        if sha256(seeded) != digest:
            raise ValueError(f"SHA256 mismatch for seeded wheel {filename}")
        copyfile(seeded, destination)
    if not destination.exists() or sha256(destination) != digest:
        for attempt in range(3):
            temporary = None
            try:
                with (
                    urlopen(url, timeout=90) as response,
                    NamedTemporaryFile(dir=destination.parent, delete=False) as stream,
                ):
                    temporary = Path(stream.name)
                    actual = hashlib.sha256()
                    while chunk := response.read(1024 * 1024):
                        actual.update(chunk)
                        stream.write(chunk)
                if actual.hexdigest() != digest:
                    raise ValueError(f"SHA256 mismatch for {filename}")
                os.replace(temporary, destination)
                break
            except (OSError, ValueError):
                if attempt == 2:
                    raise
                time.sleep(2**attempt)
            finally:
                if temporary is not None:
                    temporary.unlink(missing_ok=True)
    return f"{name} @ {destination.as_uri()} {hash_option}"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("lock", type=Path)
    parser.add_argument("cache", type=Path)
    parser.add_argument("--seed", type=Path, required=True)
    args = parser.parse_args()
    entries = [
        line
        for line in args.lock.read_text().splitlines()
        if line and not line.startswith("#")
    ]
    if not entries:
        parser.error("wheel lock is empty")
    with ThreadPoolExecutor(max_workers=12) as executor:
        for requirement in executor.map(
            fetch, entries, repeat(args.cache.resolve()), repeat(args.seed)
        ):
            print(requirement)


if __name__ == "__main__":
    main()
