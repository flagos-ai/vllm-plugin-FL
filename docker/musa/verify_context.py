#!/usr/bin/env python3
"""Check the minimal private context before invoking Docker; print fixed build args."""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from prepare_context import BASE
from verify_stack import fingerprint, load_manifest, normalized, require, wheel_review


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--context", required=True)
    parser.add_argument("--format", choices=("json", "build-args"), default="json")
    args = parser.parse_args()
    context = Path(args.context).resolve(strict=True)
    require(context.is_dir(), "MUSA private context directory required")
    recipe = Path(__file__).resolve().parent
    manifest = load_manifest(context / "stack-manifest.json")
    require(
        manifest.get("base_image") == BASE
        and manifest.get("base_image_config_digest")
        == "sha256:863675633eb0ffd7055d60351ca9f208f271a60dc7cae811a7c68548627fdde9",
        "Fixed vendor digest required",
    )
    expected_files = {
        "Dockerfile",
        "verify_stack.py",
        "stack-manifest.json",
        "dependency-manifest.json",
    }
    expected_files |= {"wheelhouse/" + row["filename"] for row in manifest["wheels"]}
    actual_files = set()
    for path in context.rglob("*"):
        require(not path.is_symlink(), "No context symlinks")
        if path.is_file():
            actual_files.add(path.relative_to(context).as_posix())
        elif path.is_dir():
            require(
                path.relative_to(context).as_posix() == "wheelhouse",
                "No unapproved context directories",
            )
        else:
            raise ValueError("No special context files")
    require(
        actual_files == expected_files,
        "Private context must contain exactly ten approved files",
    )
    for filename in ("Dockerfile", "verify_stack.py", "dependency-manifest.json"):
        require(
            fingerprint(recipe / filename)["sha256"]
            == fingerprint(context / filename)["sha256"],
            "Context recipe differs from reviewed public source: " + filename,
        )
    dependency = json.loads((context / "dependency-manifest.json").read_bytes())
    require(
        manifest["vendor_versions"] == dependency["vendor_versions"]
        and manifest["vendor_files"] == dependency["vendor_files"]
        and manifest["source_lineage"] == dependency["source_lineage"],
        "Dependency/vendor lineage differs",
    )
    exact_three = {normalized(row["name"]): row for row in dependency["wheels"]}
    for row in manifest["wheels"]:
        wheel_review(context / "wheelhouse" / row["filename"], row)
        if normalized(row["name"]) in exact_three:
            expected = exact_three[normalized(row["name"])]
            require(
                (
                    row["version"],
                    row["bytes"],
                    row["sha256"],
                    "/opt/musa/wheelhouse/" + row["filename"],
                )
                == (
                    expected["version"],
                    expected["bytes"],
                    expected["sha256"],
                    expected["path"],
                ),
                "Three dependency pins differ",
            )
    empty = next(
        row["filename"]
        for row in manifest["wheels"]
        if normalized(row["name"]) == "vllm"
    )
    fields = {
        "MUSA_BASE_IMAGE": BASE,
        "PLUGIN_COMMIT": manifest["plugin_commit"],
        "STACK_MANIFEST_SHA256": fingerprint(context / "stack-manifest.json")["sha256"],
        "DEPENDENCY_MANIFEST_SHA256": fingerprint(context / "dependency-manifest.json")[
            "sha256"
        ],
        "EMPTY_WHEEL_FILENAME": empty,
    }
    if args.format == "build-args":
        for key, value in fields.items():
            print(key + "=" + value)
    else:
        print(
            json.dumps(
                {
                    "context": str(context),
                    "files": sorted(expected_files),
                    "build_args": fields,
                    "source_only_context_verification": True,
                    "image_built": False,
                },
                indent=2,
            )
        )


if __name__ == "__main__":
    main()
