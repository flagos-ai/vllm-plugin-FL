# Copyright (c) 2026 BAAI. All rights reserved.

"""Keep the base image's MACA binaries while updating vLLM's Python dependencies."""

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
    "mcoplib",
    "maca",
    "flash-attn",
    "flash-mla",
    "flag-gems",
    "flaggems",
    "flagcx",
    "flag-cx",
    "numpy",
    "nvidia-",
)


def native_versions() -> dict[str, str]:
    versions = {}
    for distribution in importlib.metadata.distributions():
        name = re.sub(r"[-_.]+", "-", distribution.metadata["Name"]).lower()
        # Vendor forks such as xgrammar+flagos also carry accelerator kernels.
        # vLLM and the FL plugin are the two packages this build replaces.
        vendor_build = any(
            marker in distribution.version.lower()
            for marker in ("metax", "maca", "flagos")
        )
        if name not in ("vllm", "vllm-plugin-fl") and (
            name.startswith(_NATIVE_PREFIXES) or vendor_build
        ):
            versions[name] = distribution.version
    return versions


def capture(directory: Path) -> None:
    versions = native_versions()
    if "torch" not in versions or not any(
        name in versions for name in ("triton", "flagtree", "flag-tree")
    ):
        raise RuntimeError("The MetaX base must supply torch and Triton/FlagTree")
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "native-stack.json").write_text(
        json.dumps(versions, indent=2, sort_keys=True) + "\n"
    )
    (directory / "native-overrides.txt").write_text(
        "".join(f"{name}=={version}\n" for name, version in sorted(versions.items()))
    )
    # FlagTree owns the triton import namespace without providing a triton
    # distribution. Remove transitive PyPI Triton requirements explicitly;
    # pinning FlagTree alone cannot prevent namespace replacement.
    (directory / "uv.toml").write_text(
        'exclude-dependencies = ["triton"]\n' if "triton" not in versions else ""
    )
    print(f"Preserving native MACA packages: {versions}")


def verify(directory: Path, vllm_version: str | None) -> None:
    before = json.loads((directory / "native-stack.json").read_text())
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
        raise RuntimeError(f"Dependency installation changed MACA packages: {changed}")
    if vllm_version is not None:
        distribution = importlib.metadata.distribution("vllm")
        if distribution.version != vllm_version + "+empty":
            raise RuntimeError(
                f"Expected the vLLM empty wheel, got {distribution.version}"
            )
        direct_url = json.loads(distribution.read_text("direct_url.json") or "{}")
        if direct_url.get("dir_info", {}).get("editable"):
            raise RuntimeError("The vLLM installation must use an ordinary wheel")
        native_extensions = [
            str(path)
            for path in distribution.files or ()
            if path.name.startswith(("_C.", "_C_", "_moe_C"))
            and path.name.endswith(".so")
        ]
        if native_extensions:
            raise RuntimeError(
                f"The empty wheel contains CUDA extensions: {native_extensions}"
            )
    print("Native MACA package versions are unchanged")


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
