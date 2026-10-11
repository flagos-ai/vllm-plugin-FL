#!/usr/bin/env python3
"""Create a minimal private Docker context from whole-SHA-bound ordinary wheels."""

import argparse
import json
import shutil
import sys
from pathlib import Path

# Import only this public verifier; no target package or remote source is imported.
sys.path.insert(0, str(Path(__file__).resolve().parent))
from verify_stack import fingerprint, load_manifest, normalized, require, wheel_review

BASE = "harbor.baai.ac.cn/plugin/musa-ph1.4.3.5-tree0.6.1a2-triton3.6.0-cxnone-plugin0.3.0-vllm0.24.0-cp310-pt290-x64@sha256:6f7c2209c76166c4033bf7693bf5164ed6aee96c1020307b1bd358993c69ac87"


def write(path, data):
    path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return fingerprint(path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--wheelhouse", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    source = Path(args.wheelhouse).resolve(strict=True)
    manifest = load_manifest(args.manifest)
    require(manifest.get("base_image") == BASE, "Fixed vendor RepoDigest required")
    require(
        manifest.get("base_image_config_digest")
        == "sha256:863675633eb0ffd7055d60351ca9f208f271a60dc7cae811a7c68548627fdde9",
        "Fixed vendor image config required",
    )
    reviewed = [
        wheel_review(source / row["filename"], row) for row in manifest["wheels"]
    ]
    output = Path(args.output).absolute()
    require(not output.exists(), "Exclusive new build context required")
    output.mkdir(parents=True)
    target = output / "wheelhouse"
    target.mkdir()
    for row in reviewed:
        shutil.copyfile(source / row["filename"], target / row["filename"])
        fp = fingerprint(target / row["filename"])
        require(
            (fp["bytes"], fp["sha256"]) == (row["bytes"], row["sha256"]),
            "Copied wheel changed",
        )
    for filename in ("Dockerfile", "verify_stack.py"):
        original = Path(__file__).resolve().parent / filename
        shutil.copyfile(original, output / filename)
        require(
            fingerprint(original)["sha256"] == fingerprint(output / filename)["sha256"],
            "Public recipe copy differs",
        )
    stack = write(output / "stack-manifest.json", manifest)
    dependency_path = Path(__file__).with_name("dependency-manifest.json")
    dependency = json.loads(dependency_path.read_bytes())
    require(
        manifest["vendor_versions"] == dependency["vendor_versions"]
        and manifest["vendor_files"] == dependency["vendor_files"]
        and manifest["source_lineage"] == dependency["source_lineage"],
        "Pinned dependency/vendor lineage differs",
    )
    expected_dependencies = {
        normalized(row["name"]): row for row in dependency["wheels"]
    }
    for row in manifest["wheels"]:
        name = normalized(row["name"])
        if name in expected_dependencies:
            expected = expected_dependencies[name]
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
                "Dependency path/pin differs",
            )
    shutil.copyfile(dependency_path, output / "dependency-manifest.json")
    dependency_fp = fingerprint(output / "dependency-manifest.json")
    require(
        dependency_fp["sha256"] == fingerprint(dependency_path)["sha256"],
        "Copied dependency manifest changed",
    )
    empty = next(
        row["filename"]
        for row in manifest["wheels"]
        if normalized(row["name"]) == "vllm"
    )
    argv = [
        "docker",
        "build",
        "--pull=false",
        "--network=none",
        "--target",
        "ci",
        "--build-arg",
        "MUSA_BASE_IMAGE=" + BASE,
        "--build-arg",
        "PLUGIN_COMMIT=" + manifest["plugin_commit"],
        "--build-arg",
        "STACK_MANIFEST_SHA256=" + stack["sha256"],
        "--build-arg",
        "DEPENDENCY_MANIFEST_SHA256=" + dependency_fp["sha256"],
        "--build-arg",
        "EMPTY_WHEEL_FILENAME=" + empty,
    ]
    # Root supplies a unique private registry tag after the source commit is fixed.
    result = {
        "context": str(output),
        "docker_build_argv_prefix": argv,
        "stack_manifest": stack,
        "dependency_manifest": dependency_fp,
        "empty_wheel_filename": empty,
        "required_context_files": sorted(
            str(x.relative_to(output)) for x in output.rglob("*") if x.is_file()
        ),
        "image_built": False,
        "image_published": False,
        "GPU_executed": False,
        "CI_passed": False,
    }
    write(output.parent / (output.name + ".context-evidence.json"), result)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
