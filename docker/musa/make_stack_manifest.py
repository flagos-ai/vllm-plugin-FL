#!/usr/bin/env python3
"""Bind six supplied wheels and two actual source-build receipts to an image manifest."""

import argparse
import email.parser
import json
import re
import sys
import zipfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from prepare_context import BASE
from verify_stack import FIXED, fingerprint, normalized, require, wheel_review


def wheel_row(path):
    path = Path(path).resolve(strict=True)
    fp = fingerprint(path)
    with zipfile.ZipFile(path) as wheel:
        members = [
            name
            for name in wheel.namelist()
            if len(Path(name).parts) == 2 and name.endswith(".dist-info/METADATA")
        ]
        require(len(members) == 1, "One ordinary wheel METADATA")
        metadata = email.parser.BytesParser().parsebytes(wheel.read(members[0]))
    row = {
        "name": metadata["Name"],
        "version": metadata["Version"],
        "filename": path.name,
        "bytes": fp["bytes"],
        "sha256": fp["sha256"],
    }
    wheel_review(path, row)
    return row


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--plugin-commit", required=True)
    parser.add_argument("--plugin-wheel", required=True)
    parser.add_argument("--observer-wheel", required=True)
    parser.add_argument("--plugin-build-receipt", required=True)
    parser.add_argument("--observer-build-receipt", required=True)
    parser.add_argument("--dependency-wheelhouse", required=True)
    parser.add_argument("--empty-wheel", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    require(
        re.fullmatch(r"[0-9a-f]{40}", args.plugin_commit), "Exact committed source SHA"
    )
    dependency = json.loads(
        Path(__file__).with_name("dependency-manifest.json").read_bytes()
    )
    root = Path(args.dependency_wheelhouse).resolve(strict=True)
    rows = [
        wheel_row(args.empty_wheel),
        wheel_row(args.plugin_wheel),
        wheel_row(args.observer_wheel),
    ]
    require(
        [normalized(x["name"]) for x in rows]
        == ["vllm", "vllm-plugin-fl", "musa-gate-observer"],
        "Wrong core wheel inputs",
    )
    for row in rows[1:]:
        row["source_commit"] = args.plugin_commit
    for row in dependency["wheels"]:
        actual = wheel_row(root / Path(row["path"]).name)
        require(
            (actual["version"], actual["bytes"], actual["sha256"])
            == FIXED[normalized(actual["name"])],
            "Dependency artifact differs from validated CPU07",
        )
        rows.append(actual)
    manifest = {
        "schema_version": 1,
        "ordinary_wheels_only": True,
        "base_image": BASE,
        "base_image_config_digest": "sha256:863675633eb0ffd7055d60351ca9f208f271a60dc7cae811a7c68548627fdde9",
        "plugin_commit": args.plugin_commit,
        "observer_commit": args.plugin_commit,
        "wheels": rows,
        "vendor_versions": dependency["vendor_versions"],
        "vendor_files": dependency["vendor_files"],
        "source_lineage": dependency["source_lineage"],
        "build_receipt_binding_scope": "Preserved actual ordinary source-build receipts; commit/source proofs must be reviewed before invocation",
        "communication": "MCCL",
        "FlagCX_installed": False,
        "GPU_executed_for_this_image": False,
        "CI_passed_for_this_image": False,
    }
    for key, value in (
        ("plugin_build_receipt", args.plugin_build_receipt),
        ("observer_build_receipt", args.observer_build_receipt),
    ):
        fp = fingerprint(value, 16 * 1024 * 1024)
        require(fp["bytes"] > 0, "Actual build receipt cannot be empty")
        manifest[key] = {
            "filename": Path(fp["path"]).name,
            "bytes": fp["bytes"],
            "sha256": fp["sha256"],
        }
    output = Path(args.output).absolute()
    require(not output.exists(), "Exclusive actual manifest output")
    output.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(fingerprint(output), indent=2))


if __name__ == "__main__":
    main()
