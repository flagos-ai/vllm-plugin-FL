#!/usr/bin/env python3
"""Launch Qwen3.8 MetaX with native vLLM options. Dry-run unless --execute.

MTP is off by default. Sampling defaults are inherited from the checkpoint via
generation_config=auto; clients can explicitly override native request fields.
This entry point changes configuration only, never model/operator source.
"""

from __future__ import annotations

import argparse
import json
import os
import shlex
from pathlib import Path

PLUGIN_ROOT = Path(__file__).resolve().parents[2]
REQUEST_BUCKETS = (1, 2, 4, 8, 16, 32, 64)
MTP_COUNTS = {"off": 0, "k1": 1, "k2": 2, "k3": 3}
UNSET_ENV = (
    "FLAGCX_PATH",
    "FLAGGEMS_CACHE_DIR",
    "FLAGGEMS_DB_URL",
    "GEMS_CACHE_PATH",
    "FLAGGEMS_FORCE_TLE",
    "VLLM_PLUGINS",
    "VLLM_SERVER_DEV_MODE",
    "VLLM_FL_FLAGOS_WHITELIST",
    "VLLM_FL_FLAGOS_BLACKLIST",
    "VLLM_FL_FLAGOS_BLACKLIST_APPEND",
    "VLLM_FL_OOT_WHITELIST",
    "VLLM_FL_OOT_BLACKLIST",
    "VLLM_FL_STRICT",
    "VLLM_FL_DENY_VENDORS",
    "VLLM_FL_ALLOW_VENDORS",
    "VLLM_FL_PER_OP",
    "VLLM_FL_OP_CONFIG",
    "VLLM_FL_PLUGIN_MODULES",
    "VLLM_FL_IO_DUMP",
    "VLLM_FL_IO_DUMP_OPS",
    "VLLM_FL_IO_DUMP_MODULES",
    "VLLM_FL_IO_DUMP_LAYERS",
    "VLLM_FL_IO_DUMP_MAX_CALLS",
    "VLLM_FL_IO_DUMP_STEP_RANGE",
    "VLLM_FL_IO_DUMP_WITH_TORCH_FUNCS",
    "VLLM_FL_IO_DUMP_RANK",
    "VLLM_FL_IO_DUMP_WITH_VALUES",
    "VLLM_FL_IO_DUMP_WITH_METAS",
    "VLLM_FL_IO_DUMP_WITH_PRINT",
    "VLLM_FL_IO_DUMP_BG_WORKERS",
)


def build_config(model: str, mtp: str = "off") -> dict:
    """Build one complete native configuration; no prior variant is mutated."""
    if mtp not in MTP_COUNTS:
        raise ValueError("MTP must be off, k1, k2 or k3")
    k = MTP_COUNTS[mtp]
    buckets = [(k + 1) * count for count in REQUEST_BUCKETS]
    config = {
        "model": model,
        "tensor_parallel_size": 8,
        "pipeline_parallel_size": 1,
        "dtype": "bfloat16",
        "max_model_len": 69632,
        "max_num_seqs": 64,
        "max_num_batched_tokens": 2048,
        "gpu_memory_utilization": 0.92,
        "language_model_only": True,
        "enable_prefix_caching": False,
        "enable_chunked_prefill": True,
        "mamba_cache_mode": "none",
        "mamba_ssm_cache_dtype": "float32",
        "async_scheduling": False,
        "generation_config": "auto",
        "compilation_config": {
            "mode": 0,
            "cudagraph_mode": "FULL_DECODE_ONLY",
            "cudagraph_capture_sizes": buckets,
            "max_cudagraph_capture_size": buckets[-1],
        },
        "disable_custom_all_reduce": True,
        "cudagraph_metrics": True,
        "additional_config": {
            "qwen38_ngram_mode": "shared_buffer",
            "qwen38_mtp_graph": bool(k),
            "qwen38_qsa_length_buckets": True,
            "qwen38_tuning_profile": True,
        },
    }
    if k:
        config["speculative_config"] = {
            "method": "mtp",
            "num_speculative_tokens": k,
            "parallel_drafting": False,
            "use_local_argmax_reduction": False,
            "disable_padded_drafter_batch": True,
        }
    return config


def native_command(
    config: dict, *, host: str, port: int, served_model_name: str, vllm_bin: str
) -> list[str]:
    command = [
        vllm_bin,
        "serve",
        config["model"],
        "--host",
        host,
        "--port",
        str(port),
        "--served-model-name",
        served_model_name,
    ]
    for key, value in config.items():
        if key == "model":
            continue
        flag = "--" + key.replace("_", "-")
        if isinstance(value, bool):
            command.append(flag if value else "--no-" + key.replace("_", "-"))
        else:
            encoded = (
                json.dumps(value, separators=(",", ":"))
                if isinstance(value, (dict, list))
                else str(value)
            )
            command.extend((flag, encoded))
    return command


def environment_changes(args, current: dict) -> dict:
    """Use this candidate and explicitly selected external operator packages."""
    fg = Path(args.flag_gems_root).resolve()
    fgv = Path(args.fgv_root).resolve()
    if fg.name != "flag_gems" or fgv.name != "flaggems_vllm":
        raise ValueError(
            "Pass the actual flag_gems and flaggems_vllm package directories"
        )
    if not (fg / "__init__.py").is_file() or not (fgv / "__init__.py").is_file():
        raise FileNotFoundError("Pinned operator packages are missing")
    paths = [str(PLUGIN_ROOT), str(fg.parent), str(fgv.parent)]
    paths.extend(
        p
        for p in current.get("PYTHONPATH", "").split(os.pathsep)
        if p and p not in paths
    )
    return {
        "PYTHONPATH": os.pathsep.join(paths),
        "VLLM_USE_V2_MODEL_RUNNER": "0",
        "USE_FLAGGEMS": "1",
        "VLLM_FL_USE_FLAGGEMS_VLLM": "1",
        "VLLM_FL_PREFER_ENABLED": "1",
        "VLLM_FL_PREFER": "flagos",
        "VLLM_FL_OOT_ENABLED": "1",
        "FLAGGEMS_ENABLE_OPLIST_PATH": str(
            Path(args.cache_root).resolve() / args.mtp / "enabled-ops.txt"
        ),
        "VLLM_FL_CONFIG": str(
            PLUGIN_ROOT / "vllm_fl/dispatch/config/qwen3_8_flash_next_metax.yaml"
        ),
        "VLLM_TUNED_CONFIG_FOLDER": str(
            PLUGIN_ROOT / "examples/qwen3_8_flash_next/metax_tuning/moe-configs"
        ),
        "FLAGGEMS_METAX_TLE": "0",
        "TRITON_CACHE_DIR": str(Path(args.cache_root).resolve() / args.mtp / "triton"),
    }


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--model", default=os.environ.get("MODEL_PATH", "/models/Qwen3.8-Flash-Next")
    )
    p.add_argument("--mtp", choices=tuple(MTP_COUNTS), default="off")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=18170)
    p.add_argument("--served-model-name", default="qwen38")
    p.add_argument("--vllm-bin", default="vllm")
    p.add_argument("--flag-gems-root", default=os.environ.get("QWEN38_FLAGGEMS_ROOT"))
    p.add_argument("--fgv-root", default=os.environ.get("QWEN38_FGV_ROOT"))
    p.add_argument("--cache-root", default=os.environ.get("QWEN38_CACHE_ROOT"))
    p.add_argument(
        "--output-dir",
        type=Path,
        help="Write new reviewable config/command/environment files; never overwrite",
    )
    p.add_argument(
        "--execute", action="store_true", help="Replace this process with native vLLM"
    )
    return p


def main() -> None:
    p = parser()
    args = p.parse_args()
    if not 0 < args.port < 65536:
        p.error("--port must be between 1 and 65535")
    if not args.flag_gems_root or not args.fgv_root or not args.cache_root:
        p.error(
            "Set QWEN38_FLAGGEMS_ROOT, QWEN38_FGV_ROOT, QWEN38_CACHE_ROOT or pass their CLI options"
        )
    config = build_config(args.model, args.mtp)
    command = native_command(
        config,
        host=args.host,
        port=args.port,
        served_model_name=args.served_model_name,
        vllm_bin=args.vllm_bin,
    )
    changes = environment_changes(args, dict(os.environ))
    record = {
        "mtp": args.mtp,
        "config": config,
        "argv": command,
        "shell": shlex.join(command),
        "environment_updates": changes,
        "environment_unset": list(UNSET_ENV),
        "sampling": "Native checkpoint generation_config=auto; no request sampling or thinking override",
        "execute": args.execute,
    }
    if args.output_dir is not None:
        args.output_dir.mkdir(parents=True, exist_ok=False)
        for name, value in (
            ("server-config.json", config),
            ("command.json", {"argv": command, "shell": shlex.join(command)}),
            ("environment.json", {"updates": changes, "unset": list(UNSET_ENV)}),
            ("launch-plan.json", record),
        ):
            (args.output_dir / name).write_text(json.dumps(value, indent=2) + "\n")
    print(json.dumps(record, indent=2), flush=True)
    if not args.execute:
        print(
            "DRY RUN: no vLLM import, GPU initialization, HTTP request or service start.",
            flush=True,
        )
        return
    Path(changes["FLAGGEMS_ENABLE_OPLIST_PATH"]).parent.mkdir(
        parents=True, exist_ok=True
    )
    Path(changes["TRITON_CACHE_DIR"]).mkdir(parents=True, exist_ok=True)
    child_env = dict(os.environ)
    for key in UNSET_ENV:
        child_env.pop(key, None)
    child_env.update(changes)
    os.execvpe(command[0], command, child_env)


if __name__ == "__main__":
    main()
