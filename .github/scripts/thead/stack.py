# Copyright 2026 FlagOS Contributors
"""Pinned PPU stack provenance and normal import checks."""

import argparse
import hashlib
import importlib
import importlib.metadata as metadata
import json
import subprocess
import sys
import sysconfig
from contextlib import redirect_stdout
from pathlib import Path

HERE = Path(__file__).resolve().parent
STACK = json.loads((HERE / "stack.json").read_text())


def file_fact(path):
    raw = Path(path).read_bytes()
    return {"bytes": len(raw), "sha256": hashlib.sha256(raw).hexdigest()}


def verify_files(root, phase):
    for row in STACK["files"]:
        actual = file_fact(Path(root) / row["path"])
        if actual != row[phase]:
            raise RuntimeError(f"FlagGems {phase} binding failed: {row['path']}")
    return {"phase": phase, "files": STACK["files"]}


def apply_patch(source):
    source = Path(source)
    head = subprocess.check_output(
        ["git", "-C", str(source), "rev-parse", "HEAD"], text=True
    ).strip()
    if head != STACK["flaggems_commit"]:
        raise RuntimeError("FlagGems source is not the reviewed base commit")
    tracked = subprocess.check_output(
        ["git", "-C", str(source), "status", "--porcelain", "-uno"], text=True
    )
    if tracked.strip():
        raise RuntimeError("FlagGems source must have no tracked changes before patch")
    verify_files(source / "src", "before")
    patch = HERE / "flaggems-6894.patch"
    if file_fact(patch)["sha256"] != STACK["patch_sha256"]:
        raise RuntimeError("Reviewed three-file patch changed")
    subprocess.run(
        ["git", "-C", str(source), "apply", "--check", str(patch)], check=True
    )
    subprocess.run(["git", "-C", str(source), "apply", str(patch)], check=True)
    verify_files(source / "src", "after")


def vendor_snapshot():
    # This finite protection is not a whole SDK or vendor payload audit.
    import torch

    if torch.__version__.split("+")[0] != "2.10.0":
        raise RuntimeError("This PPU stack requires the vendor Torch 2.10.0")
    return {
        "torch_version": torch.__version__,
        "torch_init": {"path": torch.__file__, **file_fact(torch.__file__)},
        "torch_C": {"path": torch._C.__file__, **file_fact(torch._C.__file__)},
    }


def validate_vllm_provider(
    module_version, distribution_version, module_origin, distribution_module_origin
):
    # Upstream writes the SCM version before adding +empty to wheel metadata.
    actual = {
        "vllm_module_version": module_version,
        "vllm_distribution_version": distribution_version,
        "vllm_origin": str(module_origin),
        "vllm_distribution_origin": str(distribution_module_origin),
    }
    same_origin = (
        module_origin is not None
        and distribution_module_origin is not None
        and Path(module_origin).resolve() == Path(distribution_module_origin).resolve()
    )
    if (
        module_version != "0.28.0"
        or distribution_version != "0.28.0+empty"
        or not same_origin
    ):
        raise RuntimeError(
            "CI requires vLLM module 0.28.0 from distribution 0.28.0+empty "
            "at the same origin; actual: " + repr(actual)
        )
    return actual


def check_stack():
    # Preserve the ordinary import order used for the verified vendor stack.
    torch = importlib.import_module("torch")
    vllm = importlib.import_module("vllm")
    triton_utils = importlib.import_module("vllm.triton_utils")
    vllm_fl = importlib.import_module("vllm_fl")
    WorkerFL = importlib.import_module("vllm_fl.worker.worker").WorkerFL
    current_platform = importlib.import_module("vllm.platforms").current_platform
    flag_gems = importlib.import_module("flag_gems")
    triton = importlib.import_module("triton")

    if metadata.version("flagtree") != STACK["flagtree"]["version"]:
        raise RuntimeError("CI image must provide the reviewed PPU FlagTree wheel")
    expected_triton = metadata.distribution("flagtree").locate_file(
        "triton/__init__.py"
    )
    if Path(triton.__file__).resolve() != Path(expected_triton).resolve():
        raise RuntimeError("Active Triton must be supplied by the reviewed FlagTree")
    if not metadata.version("flag-gems").startswith("5.4.0rc2.post1+gb4b37a751"):
        raise RuntimeError("FlagGems base version does not match the reviewed commit")
    if triton.__version__ != "3.6.0":
        raise RuntimeError("Active Triton module must come from PPU FlagTree 3.6")
    vllm_distribution = metadata.distribution("vllm")
    vllm_provider = validate_vllm_provider(
        vllm.__version__,
        vllm_distribution.version,
        vllm.__file__,
        vllm_distribution.locate_file("vllm/__init__.py"),
    )
    if getattr(current_platform, "vendor_name", None) != "thead":
        raise RuntimeError("Normal FL registration did not activate the THead platform")
    if not torch.cuda.is_available() or torch.cuda.device_count() != 4:
        raise RuntimeError("Exactly four reserved PPU devices must be visible")
    if (
        Path(vllm_fl.__file__).resolve().parent.parent
        != Path(sysconfig.get_path("purelib")).resolve()
    ):
        raise RuntimeError("Plugin must be an ordinary wheel in this CI venv")
    gems_root = Path(flag_gems.__file__).resolve().parent.parent
    verify_files(gems_root, "after")
    # Versions alone cannot distinguish pristine b4b37 from PR6894's fix.
    return {
        "vendor": vendor_snapshot(),
        "flagtree_version": metadata.version("flagtree"),
        "triton_version": triton.__version__,
        "triton_origin": triton.__file__,
        "flaggems_version": metadata.version("flag-gems"),
        "flaggems_origin": flag_gems.__file__,
        "flaggems_base": STACK["flaggems_commit"],
        "fix_commit": STACK["fix_commit"],
        "patched_files": STACK["files"],
        "vllm_triton_utils_origin": triton_utils.__file__,
        "vllm_version": vllm.__version__,
        **vllm_provider,
        "plugin_origin": vllm_fl.__file__,
        "worker_class": WorkerFL.__qualname__,
        "platform": type(current_platform).__qualname__,
        "device_count": torch.cuda.device_count(),
        "normal_import_only": True,
        "python_prefix": sys.prefix,
        "plugin_install": "ordinary wheel in isolated CI venv",
        "upstream_declared_torch_2_13_requirement_satisfied": False,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=["apply-patch", "snapshot", "check"])
    parser.add_argument("--source", type=Path)
    parser.add_argument("--compare", type=Path)
    args = parser.parse_args()
    if args.action == "apply-patch":
        if args.source is None:
            parser.error("apply-patch requires --source")
        apply_patch(args.source)
        return
    # Keep ordinary import logs in stderr so stdout remains one JSON document.
    with redirect_stdout(sys.stderr):
        facts = vendor_snapshot() if args.action == "snapshot" else check_stack()
    if args.compare is not None:
        expected = json.loads(args.compare.read_text())
        actual = facts if args.action == "snapshot" else facts["vendor"]
        if actual != expected:
            raise RuntimeError(
                "Vendor Torch version, origins or protected bytes changed"
            )
    print(json.dumps(facts, indent=2))


if __name__ == "__main__":
    main()
