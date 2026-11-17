# Copyright (c) 2026 BAAI. All rights reserved.

"""Preserve the selected Ascend compiler environment while upgrading vLLM."""

import argparse
import base64
import hashlib
import importlib.metadata
import importlib.util
import json
import re
import sys
from pathlib import Path

# These distributions contain vendor binaries, own Triton's import namespace,
# or have an ABI dependency on the runtime supplied by the base image.
_NATIVE_PREFIXES = (
    "torch",
    "triton",
    "flagtree",
    "flag-tree",
    "flag-gems",
    "flaggems",
    "flagcx",
    "flag-cx",
    "cann",
    "ascend",
    "npu",
    "numpy",
    "nvidia-",
)
_REPLACED_PACKAGES = {
    "vllm",
    "vllm-plugin-fl",
    "vllm-ascend",
    # CI deliberately installs its separately pinned integration release.
    "flaggems-vllm",
}


def normalize(name: str) -> str:
    return re.sub(r"[-_.]+", "-", name).lower()


def native_versions() -> dict[str, str]:
    """Read only this interpreter's environment, not other compiler venvs."""
    versions = {}
    for distribution in importlib.metadata.distributions():
        name = normalize(distribution.metadata["Name"])
        vendor_build = any(
            marker in distribution.version.lower() for marker in ("ascend", "flagos")
        )
        if name in _REPLACED_PACKAGES or not (
            name.startswith(_NATIVE_PREFIXES) or vendor_build
        ):
            continue
        if name in versions and versions[name] != distribution.version:
            raise RuntimeError(
                f"Conflicting {name} distributions in the selected interpreter: "
                f"{versions[name]} and {distribution.version}"
            )
        versions[name] = distribution.version
    return versions


def compiler_snapshot() -> dict:
    """Identify this interpreter's Triton entrypoint without importing Triton."""
    spec = importlib.util.find_spec("triton")
    if spec is None or spec.origin is None:
        raise RuntimeError("The selected interpreter cannot find the triton module")
    entrypoint = Path(spec.origin).resolve(strict=True)
    source = entrypoint.read_bytes()
    owners = {}
    for distribution in importlib.metadata.distributions():
        for path in distribution.files or ():
            # Only resolve candidate entrypoints, never traverse or hash the
            # compiler's large shared libraries and generated kernel caches.
            if not str(path).replace("\\", "/").endswith("triton/__init__.py"):
                continue
            if Path(distribution.locate_file(path)).resolve() != entrypoint:
                continue
            recorded = getattr(path, "hash", None)
            record_hash = None
            if recorded is not None:
                digest = hashlib.new(recorded.mode, source).digest()
                actual = base64.urlsafe_b64encode(digest).decode().rstrip("=")
                if actual != recorded.value:
                    # Multiple vendor compiler distributions can claim the
                    # same module. Its active owner must match installed bytes.
                    continue
                record_hash = f"{recorded.mode}={recorded.value}"
            name = normalize(distribution.metadata["Name"])
            owners[name] = {
                "name": name,
                "version": distribution.version,
                "record_hash": record_hash,
            }
    if not owners:
        raise RuntimeError(
            "The Triton entrypoint has no matching distribution RECORD owner: "
            f"{entrypoint}"
        )
    return {
        "executable": str(Path(sys.executable).resolve()),
        "prefix": str(Path(sys.prefix).resolve()),
        "entrypoint": str(entrypoint),
        "sha256": hashlib.sha256(source).hexdigest(),
        "owners": [owners[name] for name in sorted(owners)],
    }


def capture(directory: Path) -> None:
    versions = native_versions()
    required = {"torch", "torch-npu", "flag-gems"}
    missing = required - versions.keys()
    if missing or not any(name in versions for name in ("flagtree", "flag-tree")):
        raise RuntimeError(
            "Select the base image's FlagTree compiler environment before capture; "
            f"it must supply torch, torch-npu, FlagGems and FlagTree (missing {missing})"
        )
    compiler = compiler_snapshot()
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "compiler-entrypoint.json").write_text(
        json.dumps(compiler, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (directory / "native-stack.json").write_text(
        json.dumps(versions, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (directory / "native-overrides.txt").write_text(
        "".join(f"{name}=={version}\n" for name, version in sorted(versions.items())),
        encoding="utf-8",
    )
    # A FlagTree distribution may supply the triton module without providing
    # a triton distribution. A transitive PyPI requirement must not replace it.
    excluded = [name for name in ("triton", "triton-ascend") if name not in versions]
    (directory / "uv.toml").write_text(
        "exclude-dependencies = " + json.dumps(excluded) + "\n", encoding="utf-8"
    )
    print(f"Preserving native Ascend packages: {versions}")


def verify_empty_wheel(vllm_version: str) -> None:
    distribution = importlib.metadata.distribution("vllm")
    expected = vllm_version + "+empty"
    if distribution.version != expected:
        raise RuntimeError(
            f"Expected the vLLM empty wheel {expected}, got {distribution.version}"
        )
    direct_url = json.loads(distribution.read_text("direct_url.json") or "{}")
    if direct_url.get("dir_info", {}).get("editable"):
        raise RuntimeError("The vLLM installation must use an ordinary wheel")
    native_extensions = [
        str(path)
        for path in distribution.files or ()
        if Path(str(path)).name.startswith(("_C.", "_C_", "_moe_C", "_qutlass_C"))
        and str(path).endswith((".so", ".pyd"))
    ]
    if native_extensions:
        raise RuntimeError(
            f"The empty wheel contains CUDA extensions: {native_extensions}"
        )
    try:
        ascend_version = importlib.metadata.version("vllm-ascend")
    except importlib.metadata.PackageNotFoundError:
        ascend_version = None
    if ascend_version is not None:
        raise RuntimeError(f"Remove vllm-ascend=={ascend_version}; use the FL platform")
    if importlib.util.find_spec("vllm_ascend") is not None:
        raise RuntimeError(
            "Remove the residual vllm_ascend module; use the FL platform"
        )


def verify_imports(vllm_version: str | None) -> None:
    # Initialize torch_npu before FlagTree's backend discovery. This check is
    # deliberately opt-in: docker build normally has no attached host driver.
    # isort: off
    import torch
    import torch_npu  # noqa: F401
    import triton
    import flag_gems
    # isort: on

    if not torch.npu.is_available():
        raise RuntimeError("The selected Ascend runtime cannot access an NPU")
    if flag_gems.vendor_name != "ascend":
        raise RuntimeError(f"Expected Ascend FlagGems, got {flag_gems.vendor_name}")
    if vllm_version is not None:
        import vllm

        if vllm.__version__.split("+")[0] != vllm_version:
            raise RuntimeError(
                f"Imported vLLM {vllm.__version__}; expected {vllm_version}"
            )
    print(
        f"NPU devices={torch.npu.device_count()}, "
        f"Triton={triton.__version__} ({triton.__file__}), "
        f"FlagGems={flag_gems.__file__}"
    )


def verify(
    directory: Path,
    vllm_version: str | None = None,
    check_imports: bool = False,
) -> None:
    before = json.loads((directory / "native-stack.json").read_text(encoding="utf-8"))
    after = native_versions()
    changed = {
        name: (before.get(name), after.get(name))
        for name in before.keys() | after.keys()
        if before.get(name) != after.get(name)
    }
    if changed:
        raise RuntimeError(
            f"Dependency installation changed Ascend packages: {changed}"
        )
    before_compiler = json.loads(
        (directory / "compiler-entrypoint.json").read_text(encoding="utf-8")
    )
    after_compiler = compiler_snapshot()
    if after_compiler != before_compiler:
        raise RuntimeError(
            "Dependency installation changed the Triton compiler entrypoint: "
            f"{before_compiler} -> {after_compiler}"
        )
    if vllm_version is not None:
        verify_empty_wheel(vllm_version)
    if check_imports:
        verify_imports(vllm_version)
    print("Native Ascend package versions are unchanged")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("capture", "verify"))
    parser.add_argument("directory", type=Path)
    parser.add_argument("--vllm-version")
    parser.add_argument("--check-imports", action="store_true")
    args = parser.parse_args()
    if args.action == "capture":
        capture(args.directory)
    else:
        verify(args.directory, args.vllm_version, args.check_imports)


if __name__ == "__main__":
    main()
