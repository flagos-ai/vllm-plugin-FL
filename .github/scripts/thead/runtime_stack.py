# Copyright 2026 FlagOS Contributors
"""Validate the vendored wheel provider and the immutable PPU runtime stack."""

import argparse
import hashlib
import importlib
import importlib.metadata as metadata
import json
import sys
from contextlib import redirect_stdout
from pathlib import Path
from types import SimpleNamespace

STACK = json.loads(Path(__file__).with_name("runtime-stack.json").read_text())
ATTENTION = (
    "vllm_fl.dispatch.backends.vendor.thead.impl.attention.TheadFlashAttentionBackend"
)


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def wheel_digest(data):
    return data.get("archive_info", {}).get("hashes", {}).get("sha256")


def audit_plan(report):
    """Reject any pip action on a Runtime-owned Python distribution."""
    for entry in report["install"]:
        project = entry["metadata"]["name"].lower().replace("_", "-")
        require(project not in STACK["protected"], f"pip would modify {project}")
        require(
            wheel_digest(entry["download_info"]),
            f"pip did not verify a wheel hash for {project}",
        )


def protected_snapshot():
    facts = {}
    for project, expected in STACK["protected"].items():
        if project == "triton":
            # Vendor fallback is deliberately absent from the default sys.path.
            (dist,) = metadata.distributions(name=project, path=["/opt/triton"])
        else:
            dist = metadata.distribution(project)
        require(
            dist.version == expected,
            f"{project} must be {expected}, found {dist.version}",
        )
        files = tuple(file for file in dist.files or () if file.hash)
        require(files, f"{project} has no hashed entries in RECORD")
        payload = hashlib.sha256()
        for file in sorted(files):
            if file.name == "RECORD":
                continue
            path = dist.locate_file(file)
            with path.open("rb") as stream:
                digest = hashlib.file_digest(stream, "sha256")
            payload.update(f"{file}:{digest.hexdigest()}\n".encode())
        facts[project] = {
            "version": dist.version,
            "path": str(dist.locate_file("")),
            "payload_sha256": payload.hexdigest(),
        }
    return facts


def module_fact(project, module, relative=None):
    dist = metadata.distribution(project)
    path = Path(module.__file__).resolve()
    if relative is None:
        relative = module.__name__.replace(".", "/") + "/__init__.py"
    expected = Path(dist.locate_file(relative)).resolve()
    require(
        path == expected, f"{module.__name__} loaded from {path}, expected {expected}"
    )
    return {
        "version": dist.version,
        "path": str(path),
        "sha256": wheel_digest(json.loads(dist.read_text("direct_url.json") or "{}")),
    }


def validate_imports(facts):
    require(facts["python"] == (3, 12), "T-Head CI requires Python 3.12")
    for name in ("vllm", "flash-attn-3", "flaggems-vllm"):
        for key in ("version", "sha256"):
            require(
                facts[name].get(key) == STACK[name][key], f"Wrong {name} wheel {key}"
            )
    for name in ("torch", "torchvision", "torchaudio", "numpy", "flag-gems"):
        require(
            facts[name]["version"] == STACK["protected"][name],
            f"Wrong {name} version",
        )
    require(facts["vllm_module_version"] == "0.32.0", "Wrong vLLM module version")
    require(
        facts["flagtree_version"] == STACK["protected"]["flagtree"]
        and facts["triton"]
        == {"version": "3.6.0", "path": "/opt/flagtree/triton/__init__.py"},
        "Active Triton must be supplied by the Runtime FlagTree",
    )
    require(facts["flash_op_registered"], "FlashAttention 3 forward op is missing")
    require(facts["plugin_registered"], "FL platform plugin entry point is missing")


def validate_devices(facts):
    require(
        facts["vendor"] == "thead", f"Expected T-Head; detected {facts['vendor']!r}"
    )
    require(
        facts["attention"] == ATTENTION,
        f"Wrong attention backend: {facts['attention']}",
    )
    require(
        facts["devices"] == ["PPU-ZW810E"] * 4,
        f"Expected four ZW810E devices, found {facts['devices']}",
    )


def check_stack(with_devices):
    import torch

    flash = importlib.import_module("flash_attn_3._C")
    vllm = importlib.import_module("vllm")
    triton = importlib.import_module("triton")
    ep = metadata.entry_points(group="vllm.platform_plugins", name="fl")
    facts = {
        "python": sys.version_info[:2],
        "executable": sys.executable,
        "vllm": module_fact("vllm", vllm),
        "vllm_module_version": vllm.__version__,
        "flagtree_version": metadata.version("flagtree"),
        "triton": {
            "path": str(Path(triton.__file__).resolve()),
            "version": triton.__version__,
        },
        "flash-attn-3": module_fact(
            "flash-attn-3", flash, "flash_attn_3/" + Path(flash.__file__).name
        ),
        "flash_op_registered": hasattr(torch.ops.flash_attn_3, "fwd"),
        "plugin_registered": [entry.value for entry in ep] == ["vllm_fl:register"],
    }
    for name in (
        "torch",
        "torchvision",
        "torchaudio",
        "numpy",
        "flag-gems",
        "flaggems-vllm",
    ):
        module = importlib.import_module(name.replace("-", "_"))
        facts[name] = module_fact(name, module)
    plugin_module = importlib.import_module("vllm_fl")
    facts["plugin"] = module_fact("vllm-plugin-fl", plugin_module)
    facts["regex"] = importlib.import_module("regex").__file__
    validate_imports(facts)

    if with_devices:
        from vllm.platforms import current_platform

        facts["vendor"] = getattr(current_platform, "vendor_name", None)
        require(
            facts["vendor"] == "thead", f"Expected T-Head; detected {facts['vendor']!r}"
        )
        facts["platform"] = type(current_platform).__qualname__
        facts["worker"] = importlib.import_module(
            "vllm_fl.worker.worker"
        ).WorkerFL.__qualname__
        facts["attention"] = current_platform.get_attn_backend_cls(
            None, SimpleNamespace(use_mla=False, use_sparse=False)
        )
        module, name = facts["attention"].rsplit(".", 1)
        getattr(importlib.import_module(module), name)
        facts["devices"] = [
            torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())
        ]
        validate_devices(facts)
        for i in range(len(facts["devices"])):
            tensor = torch.ones((32,), device=f"cuda:{i}")
            torch.cuda.synchronize(i)
            require(
                tensor.cpu().eq(1).all().item(), f"Tensor allocation failed on PPU {i}"
            )
        facts["device_properties"] = [
            str(torch.cuda.get_device_properties(i))
            for i in range(len(facts["devices"]))
        ]
    return facts


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "action", choices=("snapshot", "audit-plan", "imports", "check")
    )
    parser.add_argument("report", nargs="?", type=Path)
    args = parser.parse_args()
    if args.action == "audit-plan":
        if args.report is None:
            parser.error("audit-plan requires a pip report")
        audit_plan(json.loads(args.report.read_text()))
        return

    with redirect_stdout(sys.stderr):
        if args.action == "snapshot":
            facts = protected_snapshot()
        else:
            facts = check_stack(with_devices=args.action == "check")
    print(json.dumps(facts, indent=2))


if __name__ == "__main__":
    main()
