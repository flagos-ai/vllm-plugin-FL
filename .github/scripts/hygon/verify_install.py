#!/usr/bin/env python3
"""Verify the ordinary-wheel Hygon stack without importing the checkout."""

import argparse
import ast
import base64
import hashlib
import importlib
import json
import os
import re
import stat
import sys
import sysconfig
from importlib import metadata, util
from pathlib import Path


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def regular_bytes(path, cap=2 * 1024 * 1024):
    path = Path(path).resolve(strict=True)
    before = path.stat()
    require(
        stat.S_ISREG(before.st_mode) and before.st_size <= cap,
        f"Not a bounded regular file: {path}",
    )
    identity = lambda s: (s.st_dev, s.st_ino, s.st_mode, s.st_size, s.st_mtime_ns)
    with path.open("rb") as source:
        require(identity(os.fstat(source.fileno())) == identity(before), str(path))
        raw = source.read(cap + 1)
        require(
            len(raw) <= cap and identity(os.fstat(source.fileno())) == identity(before),
            str(path),
        )
    require(identity(path.stat()) == identity(before), f"File changed: {path}")
    return raw


def ordinary_distribution(name):
    dist = metadata.distribution(name)
    direct_url = dist.read_text("direct_url.json")
    if direct_url:
        require(
            not json.loads(direct_url).get("dir_info", {}).get("editable", False),
            f"Editable distribution: {name}",
        )
    files = dist.files
    require(files is not None, f"Missing installed RECORD: {name}")
    require(
        not any(
            "__editable__" in str(f) or str(f).endswith(".egg-link") for f in files
        ),
        f"Editable startup files: {name}",
    )
    return dist


def owned_origin(dist, origin):
    require(origin is not None, f"No origin for {dist.metadata['Name']}")
    path = Path(origin).resolve(strict=True)
    entries = [f for f in dist.files if Path(dist.locate_file(f)).resolve() == path]
    require(len(entries) == 1, f"Origin is outside installed RECORD: {path}")
    entry = entries[0]
    raw = regular_bytes(path)
    require(
        entry.size == len(raw)
        and entry.hash is not None
        and entry.hash.mode == "sha256",
        f"Origin has no SHA256/size binding: {path}",
    )
    digest = (
        base64.urlsafe_b64encode(hashlib.sha256(raw).digest())
        .rstrip(b"=")
        .decode("ascii")
    )
    require(
        entry.hash.value == digest,
        f"Installed origin differs from wheel RECORD: {path}",
    )
    return {
        "path": str(path),
        "bytes": len(raw),
        "sha256": hashlib.sha256(raw).hexdigest(),
    }


def spec_origin(dist, module):
    spec = util.find_spec(module)
    require(spec is not None, f"Missing module: {module}")
    return owned_origin(dist, spec.origin)


def file_facts(path, cap=2 * 1024 * 1024):
    path = Path(path).resolve(strict=True)
    raw = regular_bytes(path, cap)
    return {
        "path": str(path),
        "bytes": len(raw),
        "sha256": hashlib.sha256(raw).hexdigest(),
    }


def distribution_snapshot(dist):
    result = {
        "version": dist.version,
        "root": str(Path(dist.locate_file("")).resolve()),
    }
    for name in ("METADATA", "RECORD"):
        entries = [
            f
            for f in dist.files
            if f.name == name and f.parent.name.endswith(".dist-info")
        ]
        require(len(entries) == 1, f"Missing unique {name}: {dist.metadata['Name']}")
        result[name] = file_facts(dist.locate_file(entries[0]), 16 * 1024 * 1024)
    return result


def triton_origin(distributions, origin=None):
    if origin is None:
        spec = util.find_spec("triton")
        require(spec is not None, "Missing vendor Triton module")
        origin = spec.origin
    owners = {}
    for name in ("flagtree", "triton"):
        try:
            owners[name] = owned_origin(distributions[name], origin)
        except RuntimeError:
            # The vendor distributions can both list Triton files. The actual
            # bytes must match at least one installed owner, then stay unchanged.
            continue
    require(owners, "Triton origin has no matching vendor RECORD owner")
    facts = next(iter(owners.values()))
    return {**facts, "record_owners": sorted(owners)}


def vendor_snapshot(distributions):
    torch_dist = distributions["torch"]
    init = spec_origin(torch_dist, "torch")
    version_path = Path(init["path"]).parent / "version.py"
    owned_origin(torch_dist, version_path)
    raw = regular_bytes(version_path)
    literals = [
        node.value.value
        for node in ast.parse(raw).body
        if isinstance(node, (ast.Assign, ast.AnnAssign))
        and isinstance(node.value, ast.Constant)
        and isinstance(node.value.value, str)
        and any(
            isinstance(t, ast.Name) and t.id == "__version__"
            for t in (node.targets if isinstance(node, ast.Assign) else [node.target])
        )
    ]
    require(len(literals) == 1, "Torch version.py must contain one literal __version__")
    sdk = Path("/opt/rocm/.info/version-dev").resolve(strict=True)
    sdk_raw = regular_bytes(sdk, 8192)
    return {
        "distributions": {
            name: distribution_snapshot(dist) for name, dist in distributions.items()
        },
        "flag_gems_init": spec_origin(distributions["flag-gems"], "flag_gems"),
        "triton_init": triton_origin(distributions),
        "torch_distribution_version": torch_dist.version,
        "torch_distribution_root": str(Path(torch_dist.locate_file("")).resolve()),
        "torch_init": init,
        "torch_version_file": {
            "path": str(version_path),
            "bytes": len(raw),
            "sha256": hashlib.sha256(raw).hexdigest(),
        },
        "torch_runtime_literal": literals[0],
        "sdk_version_file": {
            "path": str(sdk),
            "bytes": len(sdk_raw),
            "sha256": hashlib.sha256(sdk_raw).hexdigest(),
        },
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--static-only", action="store_true")
    parser.add_argument("--write-vendor-snapshot", type=Path)
    parser.add_argument("--compare-vendor-snapshot", type=Path)
    parser.add_argument("--min-devices", type=int, default=2)
    args = parser.parse_args()
    require(sys.flags.isolated, "Run this verifier with Python -I outside the checkout")
    vendor_distributions = {
        name: ordinary_distribution(name)
        for name in ("torch", "flag-gems", "flagtree", "triton")
    }
    torch_dist = vendor_distributions["torch"]
    snapshot = vendor_snapshot(vendor_distributions)
    require(
        torch_dist.version.split("+", 1)[0] == "2.10.0",
        f"Unexpected vendor Torch: {torch_dist.version}",
    )
    if args.write_vendor_snapshot:
        with args.write_vendor_snapshot.open(
            "x", encoding="utf-8", newline="\n"
        ) as target:
            json.dump(snapshot, target, sort_keys=True, indent=2)
            target.write("\n")
        print(
            json.dumps(
                {"vendor_snapshot_saved": str(args.write_vendor_snapshot), **snapshot},
                sort_keys=True,
            )
        )
        return
    if args.compare_vendor_snapshot:
        before = json.loads(args.compare_vendor_snapshot.read_bytes())
        require(
            snapshot == before,
            "Plugin build/install changed vendor Torch/Gems/FlagTree/Triton or SDK files",
        )
    distributions = {
        name: ordinary_distribution(name) for name in ("vllm", "vllm-plugin-fl")
    }
    distributions.update(
        {name: dist for name, dist in vendor_distributions.items() if name != "torch"}
    )
    require(
        sys.prefix != sys.base_prefix,
        "Use the image's isolated Hygon virtual environment",
    )
    site_roots = {
        Path(sysconfig.get_path(name)).resolve() for name in ("purelib", "platlib")
    }
    for name in ("vllm", "vllm-plugin-fl"):
        dist_root = Path(distributions[name].locate_file("")).resolve()
        require(
            dist_root in site_roots,
            f"{name} metadata is outside the active virtual environment: {dist_root}",
        )
    # This full-stack check is intentionally after the vendor-only snapshot.
    # Capturing the old base does not require the new vLLM/plugin versions.
    for root in {
        Path(d.locate_file("")).resolve(strict=True)
        for d in (*distributions.values(), torch_dist)
    }:
        require(
            not any(root.glob("*.egg-link"))
            and not any(root.glob("__editable__*.pth")),
            f"Editable installation artifacts remain in {root}",
        )
    require(
        distributions["vllm"].version == "0.28.0+empty",
        f"Expected noneditable vLLM 0.28.0+empty: {distributions['vllm'].version}",
    )
    for dist_name, variable in (
        ("flag-gems", "HYGON_FLAGGEMS_VERSION"),
        ("flagtree", "HYGON_FLAGTREE_VERSION"),
    ):
        expected = os.environ.get(variable)
        if expected:
            require(
                distributions[dist_name].version == expected,
                f"{dist_name}: expected {expected}, got {distributions[dist_name].version}",
            )
    modules = {
        "vllm": "vllm",
        "vllm-plugin-fl": "vllm_fl",
        "flag-gems": "flag_gems",
        "flagtree": "triton",
        "triton": "triton",
    }
    result = {
        "mode": "static" if args.static_only else "ordinary_gpu_import",
        "vendor": snapshot,
        "distributions": {},
    }
    for name, module in modules.items():
        dist = distributions[name]
        origin = (
            triton_origin(distributions)
            if module == "triton"
            else spec_origin(dist, module)
        )
        if name in ("vllm", "vllm-plugin-fl"):
            require(
                Path(origin["path"]).parent.parent in site_roots,
                f"{name} module is outside the active virtual environment",
            )
        result["distributions"][name] = {
            "version": dist.version,
            "spec_origin": origin,
            "editable": False,
        }
    expected_commit = os.environ.get("PLUGIN_SOURCE_SHA")
    if expected_commit:
        require(
            len(expected_commit) == 40
            and all(c in "0123456789abcdef" for c in expected_commit),
            "PLUGIN_SOURCE_SHA must be a full lowercase Git commit",
        )
        dist = distributions["vllm-plugin-fl"]
        version_files = [f for f in dist.files if str(f) == "vllm_fl/_version.py"]
        require(len(version_files) == 1, "Missing wheel-owned SCM version file")
        version_file = Path(dist.locate_file(version_files[0]))
        owned_origin(dist, version_file)
        versions = {
            n.value.value
            for n in ast.parse(regular_bytes(version_file)).body
            if isinstance(n, (ast.Assign, ast.AnnAssign))
            and isinstance(n.value, ast.Constant)
            and isinstance(n.value.value, str)
            and any(
                isinstance(t, ast.Name) and t.id == "__version__"
                for t in (n.targets if isinstance(n, ast.Assign) else [n.target])
            )
        }
        require(
            versions == {dist.version},
            "Wheel SCM version literal differs from metadata",
        )
        match = re.search(
            r"(?:^|[.+-])g(?P<sha>[0-9a-f]{7,40})(?:[.+-]|$)", dist.version
        )
        require(match is not None, "Wheel SCM version has no source commit prefix")
        recorded_commit = match.group("sha")
        require(
            expected_commit.startswith(recorded_commit),
            f"Wheel SCM commit does not match requested source: {recorded_commit}",
        )
        result["plugin_scm"] = {
            "requested_full_commit": expected_commit,
            "wheel_recorded_commit_prefix": recorded_commit,
        }
    if not args.static_only:
        require(os.environ.get("GEMS_VENDOR") == "hygon", "GEMS_VENDOR must be hygon")
        require(os.environ.get("VLLM_PLUGINS") == "fl", "VLLM_PLUGINS must be fl")
        # These are genuine ordinary imports; any native/driver error is retained.
        torch = importlib.import_module("torch")
        require(
            torch.__version__ == snapshot["torch_runtime_literal"],
            "Torch runtime/version.py mismatch",
        )
        result["torch_module_origin"] = owned_origin(torch_dist, torch.__file__)
        result["torch_runtime_version"] = torch.__version__
        result["torch_hip_version"] = getattr(torch.version, "hip", None)
        for name, module in modules.items():
            loaded = importlib.import_module(module)
            facts = result["distributions"][name]
            facts["module_origin"] = (
                triton_origin(distributions, loaded.__file__)
                if module == "triton"
                else owned_origin(distributions[name], loaded.__file__)
            )
            require(
                facts["module_origin"] == facts["spec_origin"],
                f"{module} ordinary import changed its origin",
            )
            facts["runtime_version"] = getattr(loaded, "__version__", None)
        require(
            result["distributions"]["vllm"]["runtime_version"] == "0.28.0",
            "Imported vLLM must be 0.28.0; metadata alone is insufficient",
        )
        require(
            result["distributions"]["flagtree"]["runtime_version"].startswith("3.6.0"),
            "Expected the FlagTree Triton3.6 runtime",
        )
        available, count = torch.cuda.is_available(), torch.cuda.device_count()
        require(
            available and count >= args.min_devices,
            f"Hygon accelerator unavailable/count={count}",
        )
        result["accelerator"] = {"available": available, "count": count}
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
