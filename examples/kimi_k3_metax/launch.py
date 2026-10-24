"""Launch one rank of the accepted four-node profile using vLLM's CLI config."""

import argparse
import json
import os
import sys
from importlib.metadata import distribution
from importlib.util import find_spec
from pathlib import Path


def build_launch(args):
    if args.node_rank not in range(4):
        raise ValueError("node-rank must be 0, 1, 2 or 3")
    cache = str(Path(args.cache).resolve())
    environment = {
        "VLLM_PLUGINS": "fl",
        "USE_FLAGGEMS": "1",
        "USE_FLAGTUNE": "0",
        "VLLM_FL_PREFER_ENABLED": "1",
        "VLLM_FL_PREFER": "flagos",
        "VLLM_FL_STRICT": "1",
        "VLLM_FL_CONFIG": str(Path(__file__).with_name("dispatch.yaml").resolve()),
        "OMP_NUM_THREADS": "2",
        "K3_CACHE_ROOT": cache,
        "VLLM_CACHE_ROOT": cache + "/bootstrap/vllm",
        "HF_HOME": cache + "/model_metadata",
        "TMPDIR": cache + "/bootstrap/tmp",
        "VLLM_DISABLE_SHARED_EXPERTS_STREAM": "1",
        "VLLM_FLOAT32_MATMUL_PRECISION": "highest",
        "VLLM_PP_LAYER_PARTITION": "24,24,24,21",
        "VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS": "7200",
        "VLLM_ENGINE_READY_TIMEOUT_S": "10800",
        "GLOO_SOCKET_IFNAME": args.net_interface,
    }
    # Existing MACA/MCCL transport settings, independent of operator selection.
    for prefix in ("NCCL", "MCCL"):
        for suffix, value in {
            "SOCKET_IFNAME": args.net_interface,
            "IB_DISABLE": "0",
            "NET": "IB",
            "IB_HCA": "=" + args.ib_hca,
            "IB_GID_INDEX": str(args.gid_index),
            "MIN_NCHANNELS": "16",
            "MAX_NCHANNELS": "16",
            "MAX_P2P_NCHANNELS": "16",
            "BUFFSIZE": "1048576",
        }.items():
            environment[prefix + "_" + suffix] = value
    command = [
        sys.executable,
        "-I",
        "-m",
        "vllm.entrypoints.cli.main",
        "serve",
        args.model,
        "--config",
        str(Path(__file__).with_name("serve.yaml")),
        "--node-rank",
        str(args.node_rank),
        "--api-server-count",
        "0" if args.node_rank else "1",
        "--master-addr",
        args.master_addr,
        "--master-port",
        str(args.master_port),
        "--host",
        args.host,
        "--port",
        str(args.port),
    ]
    if args.node_rank:
        command.append("--headless")
    return command, environment


def verify_dependencies():
    """Reject same-version packages installed from another source revision."""
    lock = json.loads(Path(__file__).with_name("dependency-lock.json").read_text())
    for name, item in lock["dependencies"].items():
        installed = distribution(item["distribution"])
        provenance = json.loads(installed.read_text("direct_url.json") or "{}")
        vcs = provenance.get("vcs_info", {})
        if vcs.get("vcs") != "git" or vcs.get("commit_id") != item["commit"]:
            raise RuntimeError(
                f"{name}: install the pinned Git source in requirements.txt"
            )
        package = item["import_package"]
        spec = find_spec(package)
        expected = Path(installed.locate_file(package + "/__init__.py")).resolve()
        if (
            spec is None
            or spec.origin is None
            or Path(spec.origin).resolve() != expected
        ):
            raise RuntimeError(
                f"{name}: import path shadows the pinned installed package"
            )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--node-rank", required=True, type=int)
    parser.add_argument("--master-addr", required=True)
    parser.add_argument("--master-port", type=int, default=29864)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--cache", required=True)
    parser.add_argument("--net-interface", required=True)
    parser.add_argument("--ib-hca", required=True)
    parser.add_argument("--gid-index", type=int, default=3)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    command, environment = build_launch(args)
    if os.environ.get("FLAGCX_PATH"):
        parser.error("This profile uses MCCL; unset FLAGCX_PATH.")
    if args.dry_run:
        print(json.dumps({"command": command, "environment": environment}, indent=2))
        return
    verify_dependencies()
    for name in ("K3_CACHE_ROOT", "VLLM_CACHE_ROOT", "HF_HOME", "TMPDIR"):
        Path(environment[name]).mkdir(parents=True, exist_ok=True)
    os.execve(command[0], command, {**os.environ, **environment})


if __name__ == "__main__":
    main()
