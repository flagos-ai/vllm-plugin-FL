# Copyright 2026 FlagOS Contributors
"""Fail fast when NVIDIA CI is not using the verified native vLLM CUDA stack."""

import argparse
import base64
import hashlib
import importlib
import importlib.machinery
import importlib.metadata as metadata
import json
import os
import sys
from contextlib import redirect_stdout
from pathlib import Path

FL_SOURCES = {
    "flagtree": {
        "version": "0.7.0+git317a0249",
        "commit": "317a0249ec25c6756e0aee3a87e8e3c92f154335",
        "archive_sha256": "075eb5b382dc438e9a54b87d9de6895b5e22dd1c674c04379802cff6973501fb",
    },
    "flag-gems": {
        "version": "5.4.0+g6fe7709e7717",
        "commit": "6fe7709e7717e49cf12bb00346719a0885f505fa",
        "archive_sha256": "1925bcf08657f011cc0aa1193ecf96742a7c18fa66121fb1ab36c8142d8107b7",
    },
}


FLAGGEMS_VLLM_SOURCE = {
    "release": "v0.2.0",
    "commit": "b5c2c2249319277ccda4d41aa0f65164bd9b2a19",
    "repository": "https://github.com/flagos-ai/FlagGems-vllm.git",
}

# Required payloads in the pinned official v0.32.0-cu129 image. Upstream now
# supplies the stable libtorch modules rather than the old _C/_moe_C modules.
NATIVE_VLLM_MODULES = (
    "_C_stable_libtorch",
    "_moe_C_stable_libtorch",
    "_flashmla_C",
    "_flashmla_extension_C",
    "vllm_flash_attn._vllm_fa2_C",
    "vllm_flash_attn._vllm_fa3_C",
    "cumem_allocator",
)
NATIVE_VLLM_LIBRARIES = {
    module: module.replace(".", "/") + ".abi3.so" for module in NATIVE_VLLM_MODULES
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


def validate_fl_provenance(provenance, installed):
    """Bind installed ordinary wheels to the checksummed source build record."""
    for name, expected in FL_SOURCES.items():
        recorded = provenance.get(name, {})
        actual = installed[name]
        if any(recorded.get(key) != value for key, value in expected.items()):
            raise RuntimeError(f"{name} source provenance does not match the CI pin")
        wheel_hash = recorded.get("wheel_sha256", "")
        if (
            actual["version"] != expected["version"]
            or len(wheel_hash) != 64
            or actual.get("wheel_sha256") != wheel_hash
            or actual.get("editable", False)
        ):
            raise RuntimeError(f"{name} is not the wheel recorded by the source build")
    return provenance


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
        or distribution_version != "0.32.0+cu129"
        or not same_origin
        or editable
        or not set(NATIVE_VLLM_LIBRARIES.values()).issubset(native_libraries)
    ):
        raise RuntimeError(
            "NVIDIA CI requires the ordinary official vLLM 0.32.0+cu129 wheel "
            "with its native CUDA extensions and without source-tree shadowing; "
            f"actual: {facts}, editable={editable}, "
            f"native_libraries={list(native_libraries)}"
        )
    return facts


def native_vllm_libraries(package_dir):
    # Inspect the actual payload, including nested extensions.
    return sorted(
        str(path.relative_to(package_dir))
        for path in Path(package_dir).rglob("*")
        if path.is_file()
        and (path.suffix in {".so", ".pyd", ".dll", ".dylib"} or ".so." in path.name)
        and not path.name.startswith("_rust_")
    )


def validate_native_payload(distribution, package_dir):
    """Verify real ELF payloads against the installed official wheel's RECORD."""
    recorded = {str(path): path for path in distribution.files or ()}
    origins = {}
    for module, relative in NATIVE_VLLM_LIBRARIES.items():
        path = Path(package_dir) / relative
        entry = recorded.get(f"vllm/{relative}")
        if entry is None or entry.hash is None or entry.hash.mode != "sha256":
            raise RuntimeError(f"Native vLLM extension has no wheel hash: {relative}")
        with path.open("rb") as stream:
            if stream.read(4) != b"\x7fELF":
                raise RuntimeError(f"Native vLLM extension is not ELF: {relative}")
            stream.seek(0)
            digest = hashlib.file_digest(stream, "sha256").digest()
        actual_hash = base64.urlsafe_b64encode(digest).rstrip(b"=").decode()
        if actual_hash != entry.hash.value or path.stat().st_size != entry.size:
            raise RuntimeError(
                f"Native vLLM extension differs from its wheel: {relative}"
            )
        origins[f"vllm.{module}"] = str(path.resolve())
    return origins


def validate_native_import(module_name, module, expected_origin):
    """Reject Python shims and sys.modules placeholders for upstream extensions."""
    spec = getattr(module, "__spec__", None)
    origin = getattr(module, "__file__", None)
    if (
        spec is None
        or not isinstance(spec.loader, importlib.machinery.ExtensionFileLoader)
        or not origin
        or Path(origin).resolve() != Path(expected_origin).resolve()
        or not spec.origin
        or Path(spec.origin).resolve() != Path(expected_origin).resolve()
    ):
        raise RuntimeError(f"{module_name} must load its genuine native extension")


def check_stack(*, runtime_only=False):
    import torch
    import torchcodec
    import triton
    from packaging.specifiers import SpecifierSet

    import vllm

    dist = metadata.distribution("vllm")
    direct = json.loads(dist.read_text("direct_url.json") or "{}")
    if dist.read_text("WHEEL") is None:
        raise RuntimeError("vLLM must be installed as a wheel")
    facts = validate_vllm_provider(
        vllm.__version__,
        dist.version,
        vllm.__file__,
        dist.locate_file("vllm/__init__.py"),
        editable=direct.get("dir_info", {}).get("editable", False),
        native_libraries=native_vllm_libraries(Path(vllm.__file__).parent),
    )
    native_origins = validate_native_payload(dist, Path(vllm.__file__).parent)
    facts["vllm_native_extensions"] = native_origins
    fused_dist = metadata.distribution("flaggems-vllm")
    facts["flaggems-vllm"] = validate_flaggems_vllm_provenance(
        fused_dist.version,
        json.loads(fused_dist.read_text("direct_url.json") or "{}"),
    )
    expected_versions = {
        "torch": "2.13.0+cu129",
        **{name: source["version"] for name, source in FL_SOURCES.items()},
        "xgboost": "3.3.0",
        "humming-kernels": "0.1.17",
        "torchcodec": "0.16.0+cu129",
    }
    for name, expected in expected_versions.items():
        actual = metadata.version(name)
        if actual != expected:
            raise RuntimeError(f"Expected {name}=={expected}, found {actual}")
        facts[name] = actual
    installed = {}
    for name in FL_SOURCES:
        source_dist = metadata.distribution(name)
        source_url = json.loads(source_dist.read_text("direct_url.json") or "{}")
        installed[name] = {
            "version": source_dist.version,
            "wheel_sha256": source_url.get("archive_info", {})
            .get("hashes", {})
            .get("sha256"),
            "editable": source_url.get("dir_info", {}).get("editable", False),
        }
    provenance_path = Path(
        os.environ.get("FL_STACK_PROVENANCE", "/opt/fl-ci/provenance.json")
    )
    facts["fl_source_provenance"] = validate_fl_provenance(
        json.loads(provenance_path.read_text()), installed
    )
    if torch.version.cuda != "12.9" or triton.__version__ != "3.6.0":
        raise RuntimeError(
            f"Expected CUDA 12.9 and Triton 3.6.0, found "
            f"{torch.version.cuda} and {triton.__version__}"
        )
    try:
        standalone = metadata.version("triton")
    except metadata.PackageNotFoundError:
        standalone = None
    if standalone is not None:
        raise RuntimeError(f"Standalone Triton conflicts with FlagTree: {standalone}")
    triton_origin = metadata.distribution("flagtree").locate_file("triton/__init__.py")
    if Path(triton.__file__).resolve() != Path(triton_origin).resolve():
        raise RuntimeError("Active Triton must be supplied by the pinned FlagTree")
    transformers = metadata.version("transformers")
    if transformers not in SpecifierSet(">=5.16.1,<5.20.0"):
        raise RuntimeError(f"Transformers does not satisfy vLLM 0.32: {transformers}")
    facts.update(
        cuda=torch.version.cuda,
        triton=triton.__version__,
        transformers=transformers,
        torchcodec_origin=torchcodec.__file__,
    )
    if not runtime_only:
        import flag_gems
        import flaggems_vllm

        from vllm import LLM
        from vllm.platforms import current_platform

        fused_origin = fused_dist.locate_file("flaggems_vllm/__init__.py")
        if Path(flaggems_vllm.__file__).resolve() != Path(fused_origin).resolve():
            raise RuntimeError("FlagGems-vllm must come from the pinned installation")
        worker = importlib.import_module("vllm_fl.worker.nvidia_worker").NvidiaWorkerFL
        if getattr(current_platform, "vendor_name", None) != "nvidia":
            raise RuntimeError("Normal FL registration did not activate NVIDIA")
        if not torch.cuda.is_available():
            raise RuntimeError("NVIDIA CI requires a visible CUDA device")
        for name, origin in native_origins.items():
            validate_native_import(name, importlib.import_module(name), origin)
        facts.update(
            platform=type(current_platform).__qualname__,
            llm_class=f"{LLM.__module__}.{LLM.__qualname__}",
            flaggems_origin=flag_gems.__file__,
            flaggems_vllm_origin=flaggems_vllm.__file__,
            worker_class=f"{worker.__module__}.{worker.__qualname__}",
            device_count=torch.cuda.device_count(),
        )
    return facts


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runtime-only", action="store_true")
    args = parser.parse_args()
    with redirect_stdout(sys.stderr):
        facts = check_stack(runtime_only=args.runtime_only)
    print(json.dumps(facts, indent=2))


if __name__ == "__main__":
    main()
