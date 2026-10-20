# Copyright 2026 FlagOS Contributors
"""Record the actual FL wheels produced from the Dockerfile's pinned sources."""

import email
import hashlib
import importlib.util
import json
import sys
import zipfile
from pathlib import Path


def main():
    checker, wheels, output = map(Path, sys.argv[1:])
    spec = importlib.util.spec_from_file_location("cuda_ci_stack", checker)
    stack = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(stack)
    provenance = {}
    for name, source in stack.FL_SOURCES.items():
        candidates = list(wheels.glob(f"{name.replace('-', '_')}-*.whl"))
        if len(candidates) != 1:
            raise RuntimeError(f"Expected one {name} wheel, found {candidates}")
        wheel = candidates[0]
        with zipfile.ZipFile(wheel) as archive:
            (metadata_name,) = [
                entry
                for entry in archive.namelist()
                if entry.endswith(".dist-info/METADATA")
            ]
            metadata = email.message_from_bytes(archive.read(metadata_name))
        if metadata["Version"] != source["version"]:
            raise RuntimeError(f"Unexpected {name} version: {metadata['Version']}")
        with wheel.open("rb") as stream:
            wheel_sha256 = hashlib.file_digest(stream, "sha256").hexdigest()
        provenance[name] = {
            **source,
            "wheel": wheel.name,
            "wheel_sha256": wheel_sha256,
        }
    output.write_text(json.dumps(provenance, indent=2) + "\n")


if __name__ == "__main__":
    main()
