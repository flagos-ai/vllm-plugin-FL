"""Normal-site MUSA/source/cache checks; also create an actual origin binding.

This imports the installed vendor normally. Any import or API error is retained.
Creating a binding records real source origins; it does not certify vendor ABI.
"""

import argparse
import base64
import csv
import hashlib
import importlib
import importlib.metadata
import io
import os
import re
import subprocess
import sys
import tempfile
import traceback
from pathlib import Path

# Only these new stdlib tool siblings are added; vendor imports remain normal-site.
sys.path.insert(0, str(Path(__file__).resolve().parent))
from gate_common import (
    GATE_FILES,
    fingerprint,
    read_bytes,
    read_json,
    require,
    write_json,
)

MODULES = (
    "torch",
    "torch_musa",
    "vllm",
    "vllm_fl",
    "flag_gems",
    "triton",
    "pytest",
    "openai",
    "httpx",
    "vllm_fl.platform",
    "vllm_fl.worker.model_runner",
    "vllm_fl.compilation.graph",
    "triton.knobs",
    "flag_gems.utils.code_cache",
    "torch._inductor.runtime.cache_dir_utils",
)


def git(repo, args, commands):
    argv = ["git", "-C", str(repo), *args]
    result = subprocess.run(argv, capture_output=True, timeout=30, check=False)
    commands.append(
        {
            "argv": argv,
            "actual_exit": result.returncode,
            "stdout_bytes": len(result.stdout),
            "stdout_sha256": hashlib.sha256(result.stdout).hexdigest(),
            "stderr_utf8": result.stderr.decode("utf-8", "replace"),
        }
    )
    require(result.returncode == 0, "Source Git command failed")
    return result.stdout


def selected_record(dist_name, paths):
    dist = importlib.metadata.distribution(dist_name)
    files = list(dist.files or ())
    records = [f for f in files if str(f).endswith(".dist-info/RECORD")]
    require(len(records) == 1, f"No unique ordinary RECORD for {dist_name}")
    record = Path(dist.locate_file(records[0])).resolve(strict=True)
    raw = read_bytes(record, 64 * 1024 * 1024)
    rows = list(csv.reader(io.StringIO(raw.decode("utf-8"))))
    lookup = {}
    for name, digest, size in rows:
        require(name not in lookup, "Duplicate RECORD member")
        lookup[name] = (digest, size)
    verified = []
    for path in paths:
        matches = [
            name for name in lookup if Path(dist.locate_file(name)).resolve() == path
        ]
        require(len(matches) == 1, f"Imported source is not uniquely in RECORD: {path}")
        name = matches[0]
        digest, size = lookup[name]
        body = read_bytes(path)
        encoded = (
            base64.urlsafe_b64encode(hashlib.sha256(body).digest()).decode().rstrip("=")
        )
        require(
            digest == "sha256=" + encoded and size == str(len(body)),
            f"Imported source differs from installed RECORD: {path}",
        )
        verified.append({"member": name, **fingerprint(path)})
    direct_urls = [f for f in files if str(f).endswith(".dist-info/direct_url.json")]
    metadata = [f for f in files if str(f).endswith(".dist-info/METADATA")]
    require(len(metadata) == 1, "No unique distribution METADATA")
    return {
        "name": dist.metadata["Name"],
        "version": dist.version,
        "record": fingerprint(record, 64 * 1024 * 1024),
        "metadata": fingerprint(
            Path(dist.locate_file(metadata[0])).resolve(strict=True)
        ),
        "direct_url": fingerprint(
            Path(dist.locate_file(direct_urls[0])).resolve(strict=True)
        )
        if len(direct_urls) == 1
        else None,
        "selected_source_record_verified": verified,
        "full_vendor_binary_verification": "NOT_PERFORMED_BY_THIS_TOOL",
    }


def selected_owner(top_name, path):
    """Resolve the real imported path, de-duplicating same-project metadata.

    A system-site venv can advertise the same canonical project more than once.
    Only the distribution whose selected RECORD really contains these bytes wins.
    """
    advertised = importlib.metadata.packages_distributions().get(top_name, [])
    candidates = {}
    for name in advertised:
        canonical = re.sub(r"[-_.]+", "-", name).lower()
        candidates.setdefault(canonical, name)
    matches = []
    failures = []
    for name in candidates.values():
        try:
            observed = selected_record(name, [path])
            matches.append(observed["name"])
        except (ValueError, OSError, importlib.metadata.PackageNotFoundError) as error:
            failures.append({"name": name, "error": f"{type(error).__name__}: {error}"})
    require(
        len(matches) == 1,
        f"No unique actual RECORD owner for {top_name}: {matches}; {failures}",
    )
    return matches[0]


def origins(repo, commit, kit):
    commands = []
    require(
        re.fullmatch(r"[0-9a-f]{40}", commit), "Expected Git commit must be a full OID"
    )
    resolved = git(repo, ["rev-parse", commit + "^{commit}"], commands).decode().strip()
    require(resolved == commit, "Expected commit changed")
    tree = git(repo, ["ls-tree", "-r", "-z", "--full-tree", commit], commands)
    blobs = {}
    for row in tree.split(b"\0"):
        if row:
            descriptor, path = row.split(b"\t", 1)
            mode, kind, oid = descriptor.decode().split()
            blobs[path.decode()] = (mode, kind, oid)
    modules = {name: importlib.import_module(name) for name in MODULES}
    pins = {}
    for name, module in modules.items():
        path = Path(module.__file__).resolve(strict=True)
        pins[name] = {
            "origin": str(Path(module.__file__).absolute()),
            **fingerprint(path),
        }
    dist_paths = {}
    owners = {}
    for name, pin in pins.items():
        top = name.split(".")[0]
        owner = selected_owner(top, Path(pin["path"]))
        owners[top] = owner
        dist_paths.setdefault(owner, set()).add(Path(pin["path"]))
    distributions = {
        name: selected_record(name, sorted(paths))
        for name, paths in sorted(dist_paths.items())
    }
    upstream_version = importlib.metadata.version("vllm")
    require(
        upstream_version.split("+")[0] == "0.28.0",
        f"This Gate draft targets actual vLLM 0.28.0, got {upstream_version}",
    )
    plugin_root = Path(modules["vllm_fl"].__file__).resolve(strict=True).parent
    plugin_members = []
    for relative, (mode, kind, oid) in sorted(blobs.items()):
        if not relative.startswith("vllm_fl/") or not relative.endswith(
            (".py", ".yaml", ".json")
        ):
            continue
        require(kind == "blob", "Plugin member is not a Git blob")
        raw = git(repo, ["cat-file", "blob", oid], commands)
        target = plugin_root / relative.removeprefix("vllm_fl/")
        if mode == "120000":
            link = raw.decode("utf-8")
            require(
                "/" not in link and "\\" not in link and link not in (".", ".."),
                "Plugin symlink is outside its directory",
            )
            target_git = str(Path(relative).parent / link).replace("\\", "/")
            require(
                target_git in blobs and blobs[target_git][0] != "120000",
                "Bad link target",
            )
            raw = git(repo, ["cat-file", "blob", blobs[target_git][2]], commands)
        # Optional package JSON data is not always shipped. Every .py/.yaml is required.
        if not target.exists() and relative.endswith(".json"):
            plugin_members.append(
                {
                    "git_path": relative,
                    "git_blob": oid,
                    "package_status": "NOT_PACKAGED_OPTIONAL_JSON",
                }
            )
            continue
        require(
            read_bytes(target.resolve(strict=True)) == raw,
            f"Installed plugin differs from committed Git blob: {relative}",
        )
        plugin_members.append(
            {
                "git_path": relative,
                "git_blob": oid,
                "package_status": "MATCHED",
                **fingerprint(target.resolve(strict=True)),
            }
        )
    require(
        any(
            item["git_path"] == "vllm_fl/worker/model_runner.py"
            and item["package_status"] == "MATCHED"
            for item in plugin_members
        ),
        "No model-runner Git binding",
    )
    generated = plugin_root / "_version.py"
    generated_version = fingerprint(generated) if generated.exists() else None
    plugin_owner = owners["vllm_fl"]
    plugin_paths = {
        Path(item["path"])
        for item in plugin_members
        if item["package_status"] == "MATCHED"
    }
    plugin_paths.update(dist_paths[plugin_owner])
    if generated_version is not None:
        plugin_paths.add(generated)
    distributions[plugin_owner] = selected_record(plugin_owner, sorted(plugin_paths))
    observer_sources = None
    try:
        observer_dist = importlib.metadata.distribution("musa-gate-observer")
    except importlib.metadata.PackageNotFoundError:
        pass
    else:
        names = (
            "__init__.py",
            "trace.py",
            "server.py",
            "pytest_plugin.py",
            "summarize.py",
        )
        observer_sources = {
            name: fingerprint(
                Path(observer_dist.locate_file("musa_gate_observer/" + name)).resolve(
                    strict=True
                )
            )
            for name in names
        }
        distributions[observer_dist.metadata["Name"]] = selected_record(
            observer_dist.metadata["Name"],
            sorted(Path(row["path"]) for row in observer_sources.values()),
        )
    kit_relative = kit.relative_to(repo).as_posix()
    gate = {}
    for relative in GATE_FILES:
        git_path = f"{kit_relative}/{relative}"
        require(git_path in blobs, f"Original Gate member absent from Git: {relative}")
        mode, kind, oid = blobs[git_path]
        require(
            mode == "100644" or mode == "100755", "Gate members must be regular files"
        )
        raw = git(repo, ["cat-file", "blob", oid], commands)
        actual = read_bytes(kit / relative)
        require(
            actual == raw, f"Original Gate source differs from Git LF blob: {relative}"
        )
        gate[relative] = {"git_blob": oid, **fingerprint(kit / relative)}
    return {
        "schema_version": 1,
        "source_commit": commit,
        "upstream_version": upstream_version,
        "repo": str(repo),
        "gate_dir": str(kit),
        "original_gate": gate,
        "modules": pins,
        "distributions": distributions,
        "plugin_git_members": plugin_members,
        "generated_version": generated_version,
        "observer_sources": observer_sources,
        "actual_git_commands": commands,
        "source_origin_scope": "Committed plugin package members, original 19 Gate files, imported module source and selected RECORD bodies",
        "abi_certified": False,
    }


def verify_binding(binding, repo, commit, kit):
    actual = origins(repo, commit, kit)
    # argv observations are evidence, not a time-dependent validation field.
    for key in (
        "source_commit",
        "upstream_version",
        "repo",
        "gate_dir",
        "original_gate",
        "modules",
        "distributions",
        "plugin_git_members",
        "generated_version",
        "observer_sources",
    ):
        require(actual[key] == binding[key], f"Source origin binding changed: {key}")
    return actual


def real_device_and_caches(cache_root):
    import torch
    import torch_musa  # noqa: F401 -- actual registration is required
    from flag_gems.utils import code_cache as gems_cache
    from torch._inductor.runtime import cache_dir_utils
    from triton import knobs

    from vllm.platforms import current_platform

    require(torch.musa.is_available() is True, "Real MUSA device is unavailable")
    count = torch.musa.device_count()
    require(
        count == 4, f"CI must expose exactly four logical MUSA devices, got {count}"
    )
    require(
        current_platform.device_type == "musa", "vLLM did not select its MUSA platform"
    )
    expected = {
        "triton": cache_root / "triton",
        "flaggems": cache_root / "flaggems",
        "torchinductor": cache_root / "torchinductor",
    }
    actual = {
        "triton": knobs.cache.dir,
        "flaggems": gems_cache.cache_dir_path(),
        "torchinductor": cache_dir_utils.cache_dir(),
    }
    rows = []
    for name, expected_path in expected.items():
        path = Path(actual[name]).resolve(strict=True)
        require(
            path == expected_path.resolve(strict=True),
            f"Cache API/env mismatch: {name}",
        )
        require(
            hasattr(os, "ST_NOEXEC") and not os.statvfs(path).f_flag & os.ST_NOEXEC,
            f"Native JIT cache is mounted noexec: {path}",
        )
        fd, filename = tempfile.mkstemp(prefix="musa-gate-write-probe-", dir=path)
        try:
            os.write(fd, b"musa-gate-cache-probe\n")
            os.fsync(fd)
            os.close(fd)
            fd = None
            require(
                read_bytes(Path(filename)) == b"musa-gate-cache-probe\n",
                "Cache readback failed",
            )
        finally:
            if fd is not None:
                os.close(fd)
            Path(filename).unlink()
        rows.append(
            {
                "name": name,
                "effective_path": str(path),
                "write_read_delete_passed": True,
                "noexec": False,
            }
        )
    # The second Gems API can differ by an actual code-cache subdirectory; retain it.
    return {
        "logical_device_count": count,
        "platform": current_platform.device_type,
        "properties": [
            {"ordinal": n, "repr": str(torch.musa.get_device_properties(n))}
            for n in range(count)
        ],
        "caches": rows,
        "flaggems_code_cache_dir": str(gems_cache.code_cache_dir()),
        "native_allocation_or_collectives_tested": False,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", required=True)
    parser.add_argument("--expected-commit", required=True)
    parser.add_argument("--gate-dir", required=True)
    parser.add_argument("--cache-root", required=True)
    choice = parser.add_mutually_exclusive_group(required=True)
    choice.add_argument("--source-binding")
    choice.add_argument("--create-source-binding")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    report = {
        "schema_version": 1,
        "passed": False,
        "source_binding_id": None,
        "logical_device_count": None,
        "normal_site": not sys.flags.no_site,
        "actual_argv": sys.argv,
        "errors": [],
    }
    try:
        require(
            sys.platform == "linux" and not sys.flags.no_site,
            "Preflight requires normal-site Linux Python",
        )
        repo = Path(args.repo).resolve(strict=True)
        kit = Path(args.gate_dir).resolve(strict=True)
        cache_root = Path(args.cache_root).resolve(strict=True)
        if args.create_source_binding:
            binding = origins(repo, args.expected_commit, kit)
            binding_ref = write_json(args.create_source_binding, binding)
        else:
            binding_ref = fingerprint(args.source_binding, 64 * 1024 * 1024)
            binding = read_json(args.source_binding, 64 * 1024 * 1024)
            verify_binding(binding, repo, args.expected_commit, kit)
        report["source_binding_id"] = binding_ref["sha256"]
        report["source_binding"] = binding_ref
        report.update(real_device_and_caches(cache_root))
        report["passed"] = True
    except Exception as error:
        report["errors"].append(f"{type(error).__name__}: {error}")
        report["traceback"] = traceback.format_exc()
    write_json(args.output, report)
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
