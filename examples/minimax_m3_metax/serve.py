# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 FlagOS Contributors

"""Print an isolated M3 launch command; --execute starts this process only."""

import argparse
import fcntl
import hashlib
import importlib.metadata
import importlib.util
import json
import os
import shlex
import subprocess
import sys
import urllib.parse
from pathlib import Path

HERE = Path(__file__).resolve().parent
LEGACY_ENV = {
    "VLLM_FL_FLAGOS_BLACKLIST",
    "VLLM_FL_FLAGOS_WHITELIST",
    "VLLM_FL_FLAGOS_BLACKLIST_APPEND",
    "VLLM_FL_OOT_BLACKLIST",
    "VLLM_FL_OOT_WHITELIST",
    "VLLM_FL_OOT_ENABLED",
    "VLLM_FL_PREFER",
    "VLLM_FL_STRICT",
    "VLLM_FL_PER_OP",
    "VLLM_FL_OP_CONFIG",
    "VLLM_FL_PREFER_ENABLED",
    "VLLM_FL_PLUGIN_MODULES",
    "VLLM_FL_USE_FLAGGEMS_VLLM",
}


def source_identity():
    repo = HERE.parents[1]
    try:
        commit = (
            subprocess.check_output(
                ["git", "-C", str(repo), "rev-parse", "HEAD"], stderr=subprocess.DEVNULL
            )
            .decode()
            .strip()
        )
        diff = subprocess.check_output(
            [
                "git",
                "-C",
                str(repo),
                "diff",
                "HEAD",
                "--",
                "vllm_fl",
                "examples/minimax_m3_metax",
            ],
            stderr=subprocess.DEVNULL,
        )
        return commit + ":" + hashlib.sha256(diff).hexdigest()
    except (OSError, subprocess.CalledProcessError):
        try:
            return importlib.metadata.version("vllm-plugin-fl")
        except importlib.metadata.PackageNotFoundError:
            return "uninstalled"


def build_environment(profile, config_path, state, compiler_root, tuning_path):
    env = {
        k: v
        for k, v in os.environ.items()
        if not k.startswith("M3_") and k not in LEGACY_ENV
    }
    env.update(profile["minimax_m3"]["infrastructure_env"])
    env.update(
        {
            "VLLM_FL_CONFIG": str(config_path),
            "PYTHONPATH": str(compiler_root),
            "FLAGTREE_ROOT": str(compiler_root),
            "TRITON_CACHE_DIR": str(state / "triton"),
            "TORCHINDUCTOR_CACHE_DIR": str(state / "inductor"),
            "VLLM_CACHE_ROOT": str(state / "vllm"),
            "FLAGGEMS_CACHE_DIR": str(state / "flaggems"),
            "FLAGGEMS_VLLM_CACHE_DIR": str(state / "flaggems_vllm"),
            "FLAGGEMS_ENABLE_OPLIST_PATH": str(state / "flaggems-enabled.txt"),
            "PYTHONDONTWRITEBYTECODE": "1",
            "MCCL_TUNING_FILE": str(tuning_path),
        }
    )
    return env


def _installed_source_commit(origin):
    commit = origin.get("vcs_info", {}).get("commit_id")
    if commit:
        return commit
    # pip records a directory URL for ordinary installs from a local checkout.
    # Keep that checkout available so an offline deployment remains auditable.
    url = urllib.parse.urlsplit(origin.get("url", ""))
    if "dir_info" not in origin or url.scheme != "file" or url.netloc:
        return None
    checkout = Path(urllib.parse.unquote(url.path))
    try:
        commit = subprocess.check_output(
            ["git", "-C", str(checkout), "rev-parse", "HEAD"],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
        dirty = subprocess.check_output(
            [
                "git",
                "-C",
                str(checkout),
                "status",
                "--porcelain",
                "--untracked-files=normal",
            ],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return None
    if dirty:
        raise RuntimeError(f"Dependency checkout is not clean: {checkout}")
    return commit


def _verify_checkout_install(module, dist, origin):
    if "dir_info" not in origin:
        return
    checkout = Path(urllib.parse.unquote(urllib.parse.urlsplit(origin["url"]).path))
    names = subprocess.check_output(
        ["git", "-C", str(checkout), "ls-files", "-z", "--", "src/" + module, module],
        text=True,
        stderr=subprocess.DEVNULL,
    ).split("\0")
    checked = 0
    for name in names:
        if not name or Path(name).suffix not in {".py", ".json", ".yaml", ".yml"}:
            continue
        source = checkout / name
        relative = name.removeprefix("src/")
        installed = Path(dist.locate_file(relative))
        if not installed.is_file() or source.read_bytes() != installed.read_bytes():
            raise RuntimeError(
                f"{module} installation differs from its checkout: {relative}"
            )
        checked += 1
    if not checked:
        raise RuntimeError(f"No tracked package sources found for {module}")


def verified_dependencies():
    """Verify installed commits and reject a shadowing source tree."""
    lock = json.loads((HERE / "dependency-lock.json").read_text())
    installed = {}
    for module, spec in lock.items():
        dist = importlib.metadata.distribution(spec["distribution"])
        origin = json.loads(dist.read_text("direct_url.json") or "{}")
        commit = _installed_source_commit(origin)
        if commit != spec["commit"]:
            raise RuntimeError(
                f"{module} must be installed from public commit {spec['commit']}; "
                "see requirements.txt"
            )
        _verify_checkout_install(module, dist, origin)
        module_spec = importlib.util.find_spec(module)
        expected_root = Path(dist.locate_file(module)).resolve()
        if (
            module_spec is None
            or module_spec.origin is None
            or Path(module_spec.origin).resolve().parent != expected_root
        ):
            raise RuntimeError(f"{module} is shadowed by another source tree")
        installed[module] = {"commit": commit, "version": dist.version}
    return installed


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--model", required=True, help="Local BF16 MiniMax-M3 checkpoint directory"
    )
    p.add_argument("--config", type=Path, default=HERE / "config.json")
    p.add_argument("--cache-dir", type=Path, required=True)
    p.add_argument(
        "--flagtree-root",
        type=Path,
        required=True,
        help="Vendor FlagTree installation containing triton/",
    )
    p.add_argument("--node-rank", type=int, choices=(0, 1), required=True)
    p.add_argument("--master-addr", required=True)
    p.add_argument("--port", type=int, default=18050)
    p.add_argument("--master-port", type=int, default=29750)
    p.add_argument(
        "--communication-tuning", type=Path, default=HERE / "tuning-ar512k.cfg"
    )
    p.add_argument(
        "--probe", action="store_true", help="Cold import and binding checks only"
    )
    p.add_argument("--execute", action="store_true")
    a = p.parse_args(argv)
    profile = json.loads(a.config.read_text())
    if profile.get("model_profile") != "minimax_m3_metax":
        raise ValueError("Expected the MiniMax M3 MetaX configuration")
    compiler = a.flagtree_root.resolve()
    tuning = a.communication_tuning.resolve()
    if not (compiler / "triton/__init__.py").is_file():
        raise FileNotFoundError("The FlagTree root must contain triton/__init__.py")
    if not tuning.is_file():
        raise FileNotFoundError(tuning)
    model = Path(a.model).resolve()
    if not a.probe:
        metadata = json.loads((model / "config.json").read_text())
        architectures = set(metadata.get("architectures", []))
        if not architectures or not architectures <= {
            "MiniMaxM3SparseForCausalLM",
            "MiniMaxM3SparseForConditionalGeneration",
        }:
            raise ValueError("This profile is restricted to MiniMax M3 BF16")
        if metadata.get("quantization_config"):
            raise ValueError("This profile does not select a quantized checkpoint")
    state = a.cache_dir.resolve() / ("node" + str(a.node_rank))
    payload = (
        json.dumps(profile, sort_keys=True, separators=(",", ":")) + "\n"
    ).encode()
    identity = {
        "profile_sha256": hashlib.sha256(payload).hexdigest(),
        "fl_source": source_identity(),
        "node_rank": a.node_rank,
        "flagtree_root": str(compiler),
        "dependencies": json.loads((HERE / "dependency-lock.json").read_text()),
        "communication_sha256": hashlib.sha256(tuning.read_bytes()).hexdigest(),
    }
    config_path = state / "profile.json"
    env = build_environment(profile, config_path, state, compiler, tuning)
    command = [
        sys.executable,
        "-P",
        "-m",
        "vllm_fl.patches.minimax_m3_metax.launch",
        "--model",
        str(model),
        "--node-rank",
        str(a.node_rank),
        "--master-addr",
        a.master_addr,
        "--port",
        str(a.port),
        "--master-port",
        str(a.master_port),
    ]
    if a.probe:
        command.append("--probe")
    print(
        json.dumps(
            {
                "command": command,
                "cache": str(state),
                "profile_sha256": identity["profile_sha256"],
            },
            indent=2,
        ),
        flush=True,
    )
    print(shlex.join(command), flush=True)
    if not a.execute:
        return
    identity["installed_dependencies"] = verified_dependencies()
    # One launch owns this cache for its entire lifetime. Never stop a service
    # or remove a directory to make an incompatible cache usable.
    state.mkdir(parents=True, exist_ok=True)
    with (state / "LOCK").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        marker = state / "CACHE_ID.json"
        if marker.exists():
            if json.loads(marker.read_text()) != identity:
                raise RuntimeError("Cache identity changed; select a new --cache-dir")
        else:
            if any(x.name != "LOCK" for x in state.iterdir()):
                raise RuntimeError("Refusing an unrecognized nonempty cache")
            marker.write_text(json.dumps(identity, indent=2) + "\n")
        for name in [
            "triton",
            "inductor",
            "vllm",
            "flaggems",
            "flaggems_vllm",
            "audit",
        ]:
            (state / name).mkdir(exist_ok=True)
        config_path.write_bytes(payload)
        subprocess.run(command, env=env, check=True)


if __name__ == "__main__":
    main()
