# Copyright 2026 FlagOS Contributors
"""Fail fast when Ascend CI is not using the verified FlagOS Runtime 2.3.0 stack."""

import argparse
import base64
import hashlib
import importlib
import importlib.metadata as metadata
import json
import sys
from contextlib import redirect_stdout
from pathlib import Path

RUNTIME_VERSIONS = {
    "torch": "2.13.0+cpu",
    "torch-npu": "2.13.0",
    "torchvision": "0.28.0+cpu",
    "torchaudio": "2.11.0+cpu",
    "flag-gems": "5.4.0",
    "flagtree": "0.7.1a1+ascend3.5",
    "flagcx": "0.14.1+cann9.2.1",
    "cann-shmem": "1.6.0",
    "numpy": "1.26.4",
}

FLAGTREE_TRITON_VERSIONS = {
    "0.7.1a1+ascend3.5": "3.5.1",
}
FLAGTREE_TRITON_VERSION = FLAGTREE_TRITON_VERSIONS[RUNTIME_VERSIONS["flagtree"]]

VENDOR_REPACK_VERSIONS = {
    "xgrammar": "0.2.8+flagos",
    "opencv-python-headless": "5.0.0.93+flagos",
    "compressed-tensors": "0.17.0+flagos",
}

FLAGGEMS_VLLM_SOURCE = {
    "release": "v0.2.0",
    "commit": "b5c2c2249319277ccda4d41aa0f65164bd9b2a19",
    "repository": "https://github.com/flagos-ai/FlagGems-vllm.git",
}


def validate_flaggems_vllm_provenance(version, source):
    """Check the Git source; this release still reports metadata version 0.1.0."""
    expected = FLAGGEMS_VLLM_SOURCE
    vcs = source.get("vcs_info", {})
    if (
        source.get("url", "").removesuffix(".git")
        != expected["repository"].removesuffix(".git")
        or vcs.get("vcs") != "git"
        or vcs.get("commit_id") != expected["commit"]
        or source.get("dir_info", {}).get("editable", False)
    ):
        raise RuntimeError(
            "FlagGems-vllm must be installed from "
            f"{expected['release']} commit {expected['commit']} without editable mode"
        )
    return {**expected, "version": version, "direct_url": source}


def validate_runtime_versions(installed):
    """Reject dependency resolution that replaces the supplied Runtime stack."""
    for name, expected in RUNTIME_VERSIONS.items():
        if installed.get(name) != expected:
            raise RuntimeError(
                f"Expected {name}=={expected}, found {installed.get(name)}"
            )
    return {name: installed[name] for name in RUNTIME_VERSIONS}


def validate_vendor_repack_versions(installed):
    """Ensure Ascend vendor-repacked vLLM dependencies come from FlagOS PyPI."""
    for name, expected in VENDOR_REPACK_VERSIONS.items():
        if installed.get(name) != expected:
            raise RuntimeError(
                f"Expected {name}=={expected}, found {installed.get(name)}"
            )
    return {name: installed[name] for name in VENDOR_REPACK_VERSIONS}


def standalone_triton_packages():
    """Return any standalone triton distributions visible in active metadata."""
    found = {}
    for name in ("triton", "triton-ascend"):
        try:
            found[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            continue
    return found


def validate_compiler(version, origin, expected_origin, standalone_packages=None):
    """The Runtime's default compiler is FlagTree (/opt/flagtree), not standby Triton."""
    if standalone_packages:
        raise RuntimeError(
            f"Standalone Triton distributions conflict with FlagTree: {standalone_packages}"
        )
    if (
        version != FLAGTREE_TRITON_VERSION
        or origin is None
        or expected_origin is None
        or Path(origin).resolve() != Path(expected_origin).resolve()
    ):
        raise RuntimeError(
            f"Active Triton {FLAGTREE_TRITON_VERSION} must be supplied by Runtime FlagTree (/opt/flagtree)"
        )


def validate_vllm_provider(
    module_version,
    distribution_version,
    module_origin,
    distribution_origin,
    *,
    editable=False,
    native_libraries=(),
):
    facts = {
        "vllm_module_version": module_version,
        "vllm_distribution_version": distribution_version,
        "vllm_origin": str(module_origin),
        "vllm_distribution_origin": str(distribution_origin),
    }
    same_origin = (
        module_origin is not None
        and distribution_origin is not None
        and Path(module_origin).resolve() == Path(distribution_origin).resolve()
    )
    if (
        module_version != "0.32.0"
        or distribution_version != "0.32.0+flagos"
        or not same_origin
        or editable
        or bool(native_libraries)
    ):
        raise RuntimeError(
            "Ascend CI requires the pure-Python vLLM 0.32.0+flagos wheel "
            "without foreign native extensions or source-tree shadowing; "
            f"actual: {facts}, editable={editable}, "
            f"native_libraries={list(native_libraries)}"
        )
    return facts


def native_vllm_libraries(package_dir):
    return sorted(
        str(path.relative_to(package_dir))
        for path in Path(package_dir).rglob("*")
        if path.is_file()
        and (path.suffix in {".so", ".pyd", ".dll", ".dylib"} or ".so." in path.name)
        and not path.name.startswith("_rust_")
    )


def validate_vllm_record(distribution, package_dir):
    """Verify the installed vLLM package against its wheel RECORD."""
    recorded = {str(path): path for path in distribution.files or ()}
    init_path = Path(package_dir) / "__init__.py"
    entry = recorded.get("vllm/__init__.py")
    if entry is None or entry.hash is None or entry.hash.mode != "sha256":
        raise RuntimeError("vLLM __init__.py has no wheel RECORD sha256 hash")
    with init_path.open("rb") as stream:
        digest = hashlib.file_digest(stream, "sha256").digest()
    actual_hash = base64.urlsafe_b64encode(digest).rstrip(b"=").decode()
    if actual_hash != entry.hash.value or init_path.stat().st_size != entry.size:
        raise RuntimeError("Installed vLLM __init__.py differs from wheel RECORD")
    return {
        "vllm_init": str(init_path.resolve()),
        "vllm_init_sha256": hashlib.sha256(init_path.read_bytes()).hexdigest(),
    }


def check_build_payload():
    """Inspect installed artifacts during image build without NPU driver mounts."""
    from packaging.specifiers import SpecifierSet

    dist = metadata.distribution("vllm")
    if dist.version != "0.32.0+flagos":
        raise RuntimeError(f"Unexpected vLLM distribution: {dist.version}")
    if dist.read_text("WHEEL") is None:
        raise RuntimeError("vLLM must be installed as a wheel")
    direct = json.loads(dist.read_text("direct_url.json") or "{}")
    if direct.get("dir_info", {}).get("editable", False):
        raise RuntimeError("vLLM must not be installed in editable mode")
    package_dir = Path(dist.locate_file("vllm"))
    if native_vllm_libraries(package_dir):
        raise RuntimeError(
            f"Unexpected native libraries in Ascend vLLM wheel: {native_vllm_libraries(package_dir)}"
        )
    fused_dist = metadata.distribution("flaggems-vllm")
    standalone = standalone_triton_packages()
    if standalone:
        raise RuntimeError(
            f"Standalone Triton distributions conflict with FlagTree: {standalone}"
        )
    triton_origin = Path(
        metadata.distribution("flagtree").locate_file("triton/__init__.py")
    )
    if not triton_origin.is_file():
        raise RuntimeError(f"Missing FlagTree Triton module: {triton_origin}")
    transformers = metadata.version("transformers")
    if transformers not in SpecifierSet(">=5.16.1,<5.20.0"):
        raise RuntimeError(f"Transformers does not satisfy vLLM 0.32: {transformers}")
    return {
        "runtime": validate_runtime_versions(
            {name: metadata.version(name) for name in RUNTIME_VERSIONS}
        ),
        "vendor_repack": validate_vendor_repack_versions(
            {name: metadata.version(name) for name in VENDOR_REPACK_VERSIONS}
        ),
        "flaggems-vllm": validate_flaggems_vllm_provenance(
            fused_dist.version,
            json.loads(fused_dist.read_text("direct_url.json") or "{}"),
        ),
        "vllm_record": validate_vllm_record(dist, package_dir),
        "flagtree_triton_origin": str(triton_origin.resolve()),
        "transformers": transformers,
    }


def check_stack(*, runtime_only=False):
    import torch
    import torch_npu  # noqa: F401
    import triton
    from packaging.specifiers import SpecifierSet

    import vllm

    dist = metadata.distribution("vllm")
    direct = json.loads(dist.read_text("direct_url.json") or "{}")
    if dist.read_text("WHEEL") is None:
        raise RuntimeError("vLLM must be installed as a wheel")
    package_dir = Path(vllm.__file__).parent
    facts = validate_vllm_provider(
        vllm.__version__,
        dist.version,
        vllm.__file__,
        dist.locate_file("vllm/__init__.py"),
        editable=direct.get("dir_info", {}).get("editable", False),
        native_libraries=native_vllm_libraries(package_dir),
    )
    facts["vllm_record"] = validate_vllm_record(dist, package_dir)
    fused_dist = metadata.distribution("flaggems-vllm")
    facts["flaggems-vllm"] = validate_flaggems_vllm_provenance(
        fused_dist.version,
        json.loads(fused_dist.read_text("direct_url.json") or "{}"),
    )
    facts["runtime"] = validate_runtime_versions(
        {name: metadata.version(name) for name in RUNTIME_VERSIONS}
    )
    facts["vendor_repack"] = validate_vendor_repack_versions(
        {name: metadata.version(name) for name in VENDOR_REPACK_VERSIONS}
    )
    triton_origin = metadata.distribution("flagtree").locate_file("triton/__init__.py")
    validate_compiler(
        triton.__version__,
        triton.__file__,
        triton_origin,
        standalone_triton_packages(),
    )
    transformers = metadata.version("transformers")
    if transformers not in SpecifierSet(">=5.16.1,<5.20.0"):
        raise RuntimeError(f"Transformers does not satisfy vLLM 0.32: {transformers}")
    facts.update(
        triton=triton.__version__,
        triton_origin=str(Path(triton.__file__).resolve()),
        transformers=transformers,
    )
    if not runtime_only:
        import flag_gems
        import flaggems_vllm

        from vllm import LLM
        from vllm.platforms import current_platform
        from vllm_fl.dispatch.backends.vendor.ascend.ascend import AscendBackend

        fused_origin = fused_dist.locate_file("flaggems_vllm/__init__.py")
        if Path(flaggems_vllm.__file__).resolve() != Path(fused_origin).resolve():
            raise RuntimeError("FlagGems-vllm must come from the pinned installation")
        worker = importlib.import_module("vllm_fl.worker.worker").WorkerFL
        if (
            getattr(current_platform, "vendor_name", None) != "ascend"
            or getattr(current_platform, "device_type", None) != "npu"
        ):
            raise RuntimeError(
                "Normal FL registration did not activate Ascend NPU platform: "
                f"vendor_name={getattr(current_platform, 'vendor_name', None)!r}, "
                f"device_type={getattr(current_platform, 'device_type', None)!r}"
            )
        if not torch.npu.is_available() or torch.npu.device_count() <= 0:
            raise RuntimeError("Ascend CI requires at least one visible NPU device")
        if not AscendBackend().is_available():
            raise RuntimeError("AscendBackend reported unavailable despite visible NPU")
        facts.update(
            platform=type(current_platform).__qualname__,
            vendor_name=current_platform.vendor_name,
            device_type=current_platform.device_type,
            llm_class=f"{LLM.__module__}.{LLM.__qualname__}",
            flaggems_origin=flag_gems.__file__,
            flaggems_vllm_origin=flaggems_vllm.__file__,
            worker_class=f"{worker.__module__}.{worker.__qualname__}",
            device_count=torch.npu.device_count(),
            device_name=torch.npu.get_device_name(0),
        )
    return facts


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runtime-only", action="store_true")
    parser.add_argument(
        "--build-only", action="store_true", help="No NPU driver imports"
    )
    args = parser.parse_args()
    with redirect_stdout(sys.stderr):
        facts = (
            check_build_payload()
            if args.build_only
            else check_stack(runtime_only=args.runtime_only)
        )
    print(json.dumps(facts, indent=2))


if __name__ == "__main__":
    main()
