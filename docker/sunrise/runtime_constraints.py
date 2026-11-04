# Copyright (c) 2026 BAAI. All rights reserved.

"""Preserve the Sunrise image's native packages while replacing vLLM."""

import argparse
import importlib.metadata
import json
import re
from pathlib import Path

_NATIVE_PREFIXES = (
    "torch",
    "triton",
    "flagtree",
    "flag-tree",
    "flag-gems",
    "flaggems",
    "flagcx",
    "flag-cx",
    "flash-attn",
    "flash-mla",
    "tang",
    "ptpu",
    "pccl",
    "numpy",
    "nvidia-",
)


def native_versions() -> dict[str, str]:
    versions = {}
    for distribution in importlib.metadata.distributions():
        name = re.sub(r"[-_.]+", "-", distribution.metadata["Name"]).lower()
        vendor_build = any(
            marker in distribution.version.lower()
            for marker in ("sunrise", "ptpu", "tangrt", "flagos")
        )
        if name not in ("vllm", "vllm-plugin-fl") and (
            name.startswith(_NATIVE_PREFIXES) or vendor_build
        ):
            if name in versions and versions[name] != distribution.version:
                raise RuntimeError(f"Multiple active versions of native package {name}")
            versions[name] = distribution.version
    return versions


def capture(directory: Path) -> None:
    versions = native_versions()
    if "torch" not in versions or "torch-ptpu" not in versions:
        raise RuntimeError("The Sunrise base must supply torch and torch_ptpu")
    if not any(name in versions for name in ("triton", "flagtree", "flag-tree")):
        raise RuntimeError("The Sunrise base must supply its native Triton/FlagTree")
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "native-stack.json").write_text(
        json.dumps(versions, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (directory / "native-overrides.txt").write_text(
        "".join(f"{name}=={version}\n" for name, version in sorted(versions.items())),
        encoding="utf-8",
    )
    # FlagTree can own the triton import without a triton distribution. Stop
    # transitive PyPI Triton from replacing that compiler namespace.
    (directory / "uv.toml").write_text(
        'exclude-dependencies = ["triton"]\n' if "triton" not in versions else "",
        encoding="utf-8",
    )
    print(f"Preserving native Sunrise packages: {versions}")


def verify(directory: Path, vllm_version: str | None = None) -> None:
    before = json.loads((directory / "native-stack.json").read_text(encoding="utf-8"))
    after = native_versions()
    changed = {
        name: (version, after.get(name))
        for name, version in before.items()
        if after.get(name) != version
    }
    changed.update(
        {name: (None, version) for name, version in after.items() if name not in before}
    )
    if changed:
        raise RuntimeError(
            f"Dependency installation changed Sunrise packages: {changed}"
        )
    if vllm_version is not None:
        distribution = importlib.metadata.distribution("vllm")
        if distribution.version != vllm_version + "+empty":
            raise RuntimeError(
                f"Expected vLLM {vllm_version}+empty, got {distribution.version}"
            )
        direct_url = json.loads(distribution.read_text("direct_url.json") or "{}")
        if direct_url.get("dir_info", {}).get("editable"):
            raise RuntimeError("vLLM must be installed from an ordinary empty wheel")
        extensions = [
            str(path)
            for path in distribution.files or ()
            if path.name.startswith(("_C.", "_C_", "_moe_C"))
            and path.name.endswith((".so", ".pyd"))
        ]
        if extensions:
            raise RuntimeError(
                f"The empty wheel contains native vLLM extensions: {extensions}"
            )
    print("Native Sunrise package versions are unchanged")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("capture", "verify"))
    parser.add_argument("directory", type=Path)
    parser.add_argument("--vllm-version")
    args = parser.parse_args()
    if args.action == "capture":
        capture(args.directory)
    else:
        verify(args.directory, args.vllm_version)


if __name__ == "__main__":
    main()
