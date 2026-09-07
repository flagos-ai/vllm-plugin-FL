#!/usr/bin/env python3
"""Record the exact mounted source/runtime identity without constructing a model.

This is intentionally a preflight only: it imports packages and queries CUDA
identity, but never starts a worker, allocates a model, or calls a benchmark.
The JSON contains hashes and versions, not source files or logs.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import sys
from typing import Any


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def tree_identity(root: Path) -> dict[str, Any]:
    """Hash the relative path/size manifest without copying source contents."""
    entries: list[str] = []
    total_bytes = 0
    file_count = 0
    if not root.is_dir():
        return {"exists": False, "file_count": 0, "total_bytes": 0, "relative_size_manifest_sha256": None}
    for path in sorted(root.rglob("*")):
        if not path.is_file() or "__pycache__" in path.parts or ".git" in path.parts:
            continue
        try:
            size = path.stat().st_size
            entries.append(f"{path.relative_to(root).as_posix()}\t{size}\n")
            total_bytes += size
            file_count += 1
        except OSError:
            continue
    return {
        "exists": True,
        "file_count": file_count,
        "total_bytes": total_bytes,
        "relative_size_manifest_sha256": hashlib.sha256("".join(entries).encode()).hexdigest(),
    }


def selected_files(root: Path, relatives: tuple[str, ...]) -> dict[str, str | None]:
    result: dict[str, str | None] = {}
    for relative in relatives:
        path = root / relative
        try:
            result[relative] = sha256_file(path)
        except OSError:
            result[relative] = None
    return result


def package_identity(name: str) -> dict[str, Any]:
    try:
        version = importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        version = None
    return {"version": version}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, type=Path)
    parser.add_argument("--plugin-root", required=True, type=Path)
    parser.add_argument("--flaggems-root", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--expected-init-sha256", required=True)
    parser.add_argument("--expected-plan-cache-sha256", required=True)
    parser.add_argument("--expected-gpu-count", required=True, type=int)
    parser.add_argument("--expected-gpu-substring", required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    errors: list[str] = []
    plugin_root = args.plugin_root.resolve()
    flaggems_root = args.flaggems_root.resolve()
    init_file = flaggems_root / "src/flag_gems/__init__.py"
    cache_file = flaggems_root / "src/flag_gems/utils/aten_plan_cache.py"
    bridge_file = plugin_root / "vllm_fl/patches/flaggems_aten_plan_cache.py"

    file_hashes: dict[str, str | None] = {}
    for label, path in (
        ("flaggems_init", init_file),
        ("flaggems_plan_cache", cache_file),
        ("plugin_plan_cache_bridge", bridge_file),
    ):
        try:
            file_hashes[label] = sha256_file(path)
        except OSError:
            file_hashes[label] = None
            errors.append(f"missing/unreadable {label}: {path}")

    if file_hashes["flaggems_init"] != args.expected_init_sha256:
        errors.append("FlagGems __init__.py hash mismatch")
    if file_hashes["flaggems_plan_cache"] != args.expected_plan_cache_sha256:
        errors.append("FlagGems aten_plan_cache.py hash mismatch")

    bridge_text = ""
    try:
        bridge_text = bridge_file.read_text(encoding="utf-8")
    except OSError:
        pass
    for marker in (
        "VLLM_FL_FLAGGEMS_ATEN_PLAN_CACHE_REQUIRE",
        "post-warmup",
        "stats.get(\"hits\"",
    ):
        if marker not in bridge_text:
            errors.append(f"plan-cache bridge marker absent: {marker}")

    plugin_identity_files = (
        "vllm_fl/patches/flaggems_aten_plan_cache.py",
        "vllm_fl/patches/qwen3_8_flash_next.py",
        "vllm_fl/worker/worker.py",
        "vllm_fl/worker/model_runner.py",
        "vllm_fl/models/qwen3_8_flash_next/gpu/ops/qsa.py",
        "vllm_fl/models/qwen3_8_flash_next/gpu/qsa.py",
        "vllm_fl/models/qwen3_8_flash_next/common/hyperconnection.py",
    )
    flaggems_identity_files = (
        "src/flag_gems/__init__.py",
        "src/flag_gems/utils/aten_plan_cache.py",
    )
    plugin_selected_hashes = selected_files(plugin_root, plugin_identity_files)
    for relative, digest in plugin_selected_hashes.items():
        if digest is None:
            errors.append(f"required all-on plugin source file missing: {relative}")
    flaggems_selected_hashes = selected_files(flaggems_root, flaggems_identity_files)

    # Put the mounted trees before image site-packages. This is the check that
    # catches the historical ssh-docker preflight importing image FlagGems.
    sys.path.insert(0, str(plugin_root))
    sys.path.insert(0, str(flaggems_root / "src"))
    imported: dict[str, Any] = {}
    try:
        import torch  # type: ignore

        imported["torch"] = {
            "version": getattr(torch, "__version__", None),
            "path": str(Path(torch.__file__).resolve()),
        }
        cuda_available = bool(torch.cuda.is_available())
        gpu_count = int(torch.cuda.device_count()) if cuda_available else 0
        gpu_names = [str(torch.cuda.get_device_name(i)) for i in range(gpu_count)]
    except Exception as exc:  # pragma: no cover - exercised in the container
        cuda_available = False
        gpu_count = 0
        gpu_names = []
        errors.append(f"torch/CUDA identity failed: {type(exc).__name__}")

    try:
        import vllm  # type: ignore

        imported["vllm"] = {
            "version": getattr(vllm, "__version__", None),
            "path": str(Path(vllm.__file__).resolve()),
        }
    except Exception as exc:  # pragma: no cover
        errors.append(f"vllm import failed: {type(exc).__name__}")

    try:
        import vllm_fl  # type: ignore

        vllm_fl_path = Path(vllm_fl.__file__).resolve()
        imported["vllm_fl"] = {
            "version": getattr(vllm_fl, "__version__", None),
            "path": str(vllm_fl_path),
        }
        if plugin_root not in vllm_fl_path.parents:
            errors.append(f"vllm_fl resolved outside mounted plugin: {vllm_fl_path}")
    except Exception as exc:  # pragma: no cover
        errors.append(f"vllm_fl import failed: {type(exc).__name__}")

    try:
        import flag_gems  # type: ignore

        flaggems_path = Path(flag_gems.__file__).resolve()
        imported["flag_gems"] = {
            "version": getattr(flag_gems, "__version__", None),
            "path": str(flaggems_path),
            "enable_aten_plan_cache": callable(
                getattr(flag_gems, "enable_aten_plan_cache", None)
            ),
            "aten_plan_cache_stats": callable(
                getattr(flag_gems, "aten_plan_cache_stats", None)
            ),
        }
        if flaggems_root not in flaggems_path.parents:
            errors.append(f"flag_gems resolved outside mounted FlagGems: {flaggems_path}")
        if not imported["flag_gems"]["enable_aten_plan_cache"]:
            errors.append("modern FlagGems enable_aten_plan_cache API missing")
        if not imported["flag_gems"]["aten_plan_cache_stats"]:
            errors.append("modern FlagGems aten_plan_cache_stats API missing")
        stats_fn = getattr(flag_gems, "aten_plan_cache_stats", None)
        try:
            imported["flag_gems"]["stats_before_bridge"] = stats_fn() if stats_fn else None
        except Exception as exc:  # stats can be unavailable before bridge apply
            imported["flag_gems"]["stats_before_bridge"] = {
                "unavailable": type(exc).__name__
            }
    except Exception as exc:  # pragma: no cover
        errors.append(f"flag_gems import failed: {type(exc).__name__}")

    env_keys = (
        "PYTHONPATH",
        "VLLM_PLUGINS",
        "USE_FLAGGEMS",
        "VLLM_FL_PREFER",
        "VLLM_FL_OOT_ENABLED",
        "VLLM_FL_FLAGGEMS_ATEN_PLAN_CACHE",
        "VLLM_FL_FLAGGEMS_ATEN_PLAN_CACHE_REPORT",
        "VLLM_FL_FLAGGEMS_ATEN_PLAN_CACHE_REQUIRE",
        "FLAGGEMS_ATEN_PLAN_CACHE",
        "FLAGGEMS_ATEN_PLAN_CACHE_SIZE",
        "QWEN4_QSA_FUSED_COMPRESS",
        "QWEN4_QSA_MQA_DOT",
        "QWEN4_QSA_SPLIT_TOPK",
        "QWEN4_QSA_SPLIT_REQUIRE",
        "VLLM_FL_PACKED_BLOCK_TABLE_ARENA",
        "VLLM_FL_PACKED_BLOCK_TABLE_REQUIRE",
        "VLLM_FL_GDN_STRICT_PATCH",
        "QWEN4_QSA_SPLIT_REQUIRE",
        "VLLM_FL_PACKED_BLOCK_TABLE_ARENA",
        "QWEN4_HC_BACKEND",
        "VLLM_FL_FLAGOS_BLACKLIST",
    )
    selected_env = {key: os.environ.get(key) for key in env_keys}
    expected_env = {
        "USE_FLAGGEMS": "1",
        "VLLM_FL_FLAGGEMS_ATEN_PLAN_CACHE": "1",
        "VLLM_FL_FLAGGEMS_ATEN_PLAN_CACHE_REPORT": "1",
        "VLLM_FL_FLAGGEMS_ATEN_PLAN_CACHE_REQUIRE": "1",
        "FLAGGEMS_ATEN_PLAN_CACHE": "0",
        "FLAGGEMS_ATEN_PLAN_CACHE_SIZE": "128",
        "VLLM_PLUGINS": "fl",
        "VLLM_FL_PREFER": "flagos",
        "VLLM_FL_OOT_ENABLED": "1",
        "QWEN4_QSA_FUSED_COMPRESS": "1",
        "QWEN4_QSA_MQA_DOT": "1",
        "QWEN4_QSA_SPLIT_REQUIRE": "1",
        "VLLM_FL_PACKED_BLOCK_TABLE_ARENA": "1",
        "QWEN4_HC_BACKEND": "fallback",
        "VLLM_FL_FLAGOS_BLACKLIST": "index_put_,index_put,_index_put_impl_,nonzero,copy_,to_copy,index,index_select,conv1d,_conv_depthwise2d,conv2d,pad,constant_pad_nd,mul",
    }
    for key, value in expected_env.items():
        if os.environ.get(key) != value:
            errors.append(f"unexpected {key}={os.environ.get(key)!r}; expected {value!r}")
    path_value = os.environ.get("PYTHONPATH", "")
    if "/opt/vllm-plugin-FL" not in path_value or "/opt/FlagGems/src" not in path_value:
        errors.append("mounted plugin/FlagGems paths absent from PYTHONPATH")

    model_files: list[dict[str, Any]] = []
    config_path = args.model / "config.json"
    if not config_path.is_file():
        errors.append(f"model config missing: {config_path}")
    else:
        try:
            model_files.append(
                {
                    "path": str(config_path),
                    "size": config_path.stat().st_size,
                    "sha256": sha256_file(config_path),
                }
            )
        except OSError:
            errors.append("model config unreadable")

    payload: dict[str, Any] = {
        "status": "pass" if not errors else "fail",
        "errors": errors,
        "source": {
            "plugin_root": str(plugin_root),
            "flaggems_root": str(flaggems_root),
            "file_sha256": file_hashes,
            "plugin_tree": tree_identity(plugin_root),
            "flaggems_tree": tree_identity(flaggems_root),
            "plugin_selected_files_sha256": plugin_selected_hashes,
            "flaggems_selected_files_sha256": flaggems_selected_hashes,
        },
        "packages": imported,
        "cuda": {
            "available": cuda_available,
            "device_count": gpu_count,
            "device_names": gpu_names,
        },
        "model": {
            "root": str(args.model),
            "files": model_files,
            "tree": tree_identity(args.model),
        },
        "environment": selected_env,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")

    if gpu_count != args.expected_gpu_count:
        errors.append(f"GPU count {gpu_count} != expected {args.expected_gpu_count}")
    if not cuda_available:
        errors.append("CUDA is not available")
    needle = args.expected_gpu_substring.lower()
    if gpu_names and any(needle not in name.lower() for name in gpu_names):
        errors.append(f"not all GPU names contain {args.expected_gpu_substring}")
    if errors:
        # Update status after CUDA checks so the output is truthful.
        payload["status"] = "fail"
        payload["errors"] = errors
        args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
        print(f"identity preflight: FAIL ({len(errors)} checks)")
        return 1
    print(
        "identity preflight: PASS "
        f"gpus={gpu_count} flag_gems={imported.get('flag_gems', {}).get('path')}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
