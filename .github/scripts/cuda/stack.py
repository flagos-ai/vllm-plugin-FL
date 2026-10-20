# Copyright 2026 FlagOS Contributors
"""Fail fast when NVIDIA CI is not using the verified vLLM EMPTY stack."""

import argparse
import importlib
import importlib.metadata as metadata
import json
import sys
from contextlib import redirect_stdout
from pathlib import Path


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
        or distribution_version != "0.32.0+empty"
        or not same_origin
        or editable
        or native_libraries
    ):
        raise RuntimeError(
            "NVIDIA CI requires the ordinary vLLM 0.32.0+empty wheel, "
            "without upstream CUDA extensions or source-tree shadowing; "
            f"actual: {facts}, editable={editable}, "
            f"native_libraries={list(native_libraries)}"
        )
    return facts


def native_vllm_libraries(package_dir):
    # Optional Rust helpers are independent of the EMPTY device target. All
    # upstream CUDA/shared-library payloads must be absent, including leftovers
    # that are no longer recorded by the currently installed distribution.
    return sorted(
        str(path.relative_to(package_dir))
        for path in Path(package_dir).rglob("*")
        if path.is_file()
        and (path.suffix in {".so", ".pyd", ".dll", ".dylib"} or ".so." in path.name)
        and not path.name.startswith("_rust_")
    )


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
    expected_versions = {
        "torch": "2.13.0+cu129",
        "flagtree": "0.7.0",
        "flag-gems": "5.4.0",
        "xgboost": "3.3.0",
        "humming-kernels": "0.1.17",
        "torchcodec": "0.16.0+cu129",
    }
    for name, expected in expected_versions.items():
        actual = metadata.version(name)
        if actual != expected:
            raise RuntimeError(f"Expected {name}=={expected}, found {actual}")
        facts[name] = actual
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

        from vllm import LLM
        from vllm.platforms import current_platform

        extension = importlib.import_module("vllm_fl._C")
        worker = importlib.import_module("vllm_fl.worker.nvidia_worker").NvidiaWorkerFL
        if getattr(current_platform, "vendor_name", None) != "nvidia":
            raise RuntimeError("Normal FL registration did not activate NVIDIA")
        if not torch.cuda.is_available():
            raise RuntimeError("NVIDIA CI requires a visible CUDA device")
        facts.update(
            platform=type(current_platform).__qualname__,
            llm_class=f"{LLM.__module__}.{LLM.__qualname__}",
            flaggems_origin=flag_gems.__file__,
            plugin_extension=extension.__file__,
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
