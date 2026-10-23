# SPDX-License-Identifier: Apache-2.0
"""Launch the installed FL package with the DS-only, versioned configuration."""

import argparse
import hashlib
import importlib.metadata
import json
import os
import shlex
import shutil
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]


def make_config(model, cache, port=18001, thinking=False):
    config = json.loads((HERE / "config.json").read_text())
    policy = config["deepseek_v4_metax"]
    policy["service"]["model"] = str(Path(model).resolve())
    policy["service"]["port"] = port
    policy["moe"]["tuned_config_folder"] = str(HERE / "tuned_configs")
    policy["audit_dir"] = str(Path(cache).resolve() / "startup_audit")
    policy["hash_tables"] = json.loads((HERE / "hash_tables.json").read_text())
    if thinking:
        service = policy["service"]
        service["max_model_len"] = 131072
        service["max_num_seqs"] = 32
        service["compilation_config"]["cudagraph_capture_sizes"] = [1, 2, 4, 8, 16, 32]
        service["compilation_config"]["max_cudagraph_capture_size"] = 32
    return config


def command_for(config):
    service = config["deepseek_v4_metax"]["service"]
    command = ["vllm", "serve", service["model"]]
    for name, value in service.items():
        if name == "model":
            continue
        option = "--" + name.replace("_", "-")
        if isinstance(value, bool):
            if value:
                command.append(option)
        else:
            command += [
                option,
                json.dumps(value) if isinstance(value, (dict, list)) else str(value),
            ]
    return command


def environment_for(config, cache, inherited=None):
    env = dict(os.environ if inherited is None else inherited)
    for key in list(env):
        if key.startswith(
            ("VLLM_FL_METAX_", "DSV4_", "MACA_VLLM_ENABLE_MCTLASS_")
        ) or key in {
            "VLLM_FL_FLAGOS_BLACKLIST",
            "VLLM_FL_FLAGOS_WHITELIST",
            "FLAGGEMS_VLLM_ENABLE_METAX_GDN_COMPILER_PATCH",
        }:
            env.pop(key)
    cache = Path(cache).resolve()
    env.update(config["deepseek_v4_metax"]["infrastructure_env"])
    env.update(
        {
            "USE_FLAGGEMS": "1",
            "VLLM_FL_CONFIG": str(cache / "config.json"),
            "PYTHONPATH": str(HERE / "bootstrap"),
            "TRITON_CACHE_DIR": str(cache / "triton"),
            "TORCHINDUCTOR_CACHE_DIR": str(cache / "torchinductor"),
            "VLLM_CACHE_ROOT": str(cache / "vllm"),
            "FLAGGEMS_CACHE_DIR": str(cache / "flaggems"),
            "FLAGGEMS_VLLM_CACHE_DIR": str(cache / "flaggems_vllm"),
        }
    )
    return env


def cache_identity(config):
    digest = hashlib.sha256(json.dumps(config, sort_keys=True).encode())
    # Hash the small integration checkout, not installed libraries or model weights.
    # Cache invalidation is separate from the CPU source-equivalence audit.
    for root in (REPO / "vllm_fl", HERE):
        for path in sorted(root.rglob("*")):
            if (
                path.is_file()
                and path.suffix in {".py", ".json"}
                and "__pycache__" not in path.parts
            ):
                digest.update(
                    str(path.relative_to(REPO)).encode() + b"\0" + path.read_bytes()
                )
    return digest.hexdigest()


def prepare_cache(config, cache):
    cache = Path(cache).resolve()
    marker = cache / "runtime.json"
    identity = {"source_config_sha256": cache_identity(config)}
    if marker.exists():
        if json.loads(marker.read_text()) != identity:
            raise ValueError(
                "Cache belongs to another source/configuration; choose a new directory"
            )
        if json.loads((cache / "config.json").read_text()) != config:
            raise ValueError("Effective configuration was modified")
        return
    if cache.exists() and any(cache.iterdir()):
        raise ValueError("Refusing a nonempty, unrecognized cache")
    cache.mkdir(parents=True, exist_ok=True)
    for name in (
        "triton",
        "torchinductor",
        "vllm",
        "flaggems",
        "flaggems_vllm",
        "startup_audit",
    ):
        (cache / name).mkdir()
    (cache / "config.json").write_text(json.dumps(config, indent=2) + "\n")
    marker.write_text(json.dumps(identity, indent=2) + "\n")


def check_model(model):
    metadata = json.loads((Path(model) / "config.json").read_text())
    if "DeepseekV4ForCausalLM" not in metadata.get("architectures", []):
        raise ValueError("This launcher is only for DeepSeek V4")
    expected = {
        "expert_dtype": "int8",
        "hidden_size": 4096,
        "num_hidden_layers": 43,
        "num_attention_heads": 64,
    }
    for name, value in expected.items():
        if metadata.get(name) != value:
            raise ValueError(f"Unsupported checkpoint {name}: {metadata.get(name)}")


def check_dependency_pins():
    lock = json.loads((HERE / "dependency-lock.json").read_text())
    revisions = {}
    for entry in lock["libraries"]:
        package = entry["distribution"]
        distribution = importlib.metadata.distribution(package)
        provenance = distribution.read_text("direct_url.json")
        info = json.loads(provenance) if provenance else {}
        revision = info.get("vcs_info", {}).get("commit_id")
        if revision != entry["commit"]:
            raise RuntimeError(
                f"{package}: install the pinned Git requirement; "
                f"expected {entry['commit']}, got {revision}"
            )
        revisions[package] = revision
    return revisions


def check_environment():
    expected = {
        "vllm": "0.20.2",
        "torch": "2.8.0+metax3.7.0.7",
        "flag_gems": "5.0.2",
    }
    for package, version in expected.items():
        found = importlib.metadata.version(package)
        # MetaX wheels may carry a local "+empty" build suffix for vLLM.
        actual = found.split("+", 1)[0] if package == "vllm" else found
        if actual != version:
            raise RuntimeError(f"{package}: expected {version}, got {found}")
    for package in ("flaggems-vllm", "vllm-plugin-fl", "mctlassEx", "flagtree"):
        importlib.metadata.distribution(package)
    if not shutil.which("vllm"):
        raise RuntimeError("vllm executable is not in PATH")
    print("DSV4_DEPENDENCIES " + json.dumps(check_dependency_pins()), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--port", type=int, default=18001)
    parser.add_argument("--thinking", action="store_true")
    parser.add_argument(
        "--execute",
        action="store_true",
        help="Start vLLM; otherwise only print the command",
    )
    args = parser.parse_args()
    config = make_config(args.model, args.cache_dir, args.port, args.thinking)
    command = command_for(config)
    env = environment_for(config, args.cache_dir)
    print(shlex.join(command), flush=True)
    if not args.execute:
        return
    if not args.model.is_dir():
        raise FileNotFoundError(args.model)
    check_model(args.model)
    check_environment()
    prepare_cache(config, args.cache_dir)
    os.execvpe(command[0], command, env)


if __name__ == "__main__":
    main()
