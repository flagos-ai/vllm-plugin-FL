# Copyright 2026 FlagOS Contributors
"""Resolve real Sunrise CI resources; never infer labels from a runner name."""

import argparse
import json
import os
import re
from pathlib import Path


def resolve_config(config, env):
    result = dict(config)

    def text(variable, key, default=""):
        value = env.get(variable, "").strip() or config.get(key, default)
        if not isinstance(value, str) or any(c in value for c in "\r\n\0"):
            raise ValueError(variable + " must be single-line text")
        return value.strip()

    def sequence(variable, key):
        raw = env.get(variable, "").strip()
        value = json.loads(raw) if raw else config.get(key, [])
        if not isinstance(value, list) or any(
            not isinstance(item, str) or not item.strip() for item in value
        ):
            raise ValueError(variable + " must be a JSON array of nonempty strings")
        return value

    labels = sequence("SUNRISE_CI_RUNNER_LABELS", "runner_labels")
    if not labels:
        raise ValueError(
            "Set SUNRISE_CI_RUNNER_LABELS to the real dedicated Sunrise runner "
            "labels as JSON. Sunrise8 is a verified runner name, not a verified label."
        )
    image = text("SUNRISE_CI_IMAGE", "ci_image")
    if not image or not re.fullmatch(
        r"[^\s]+(?::[^/\s:]+|@sha256:[0-9a-f]{64})", image
    ):
        raise ValueError("SUNRISE_CI_IMAGE requires an explicit image tag or digest")
    wheel = text("SUNRISE_CI_VLLM_WHEEL", "vllm_wheel")
    source = text("SUNRISE_CI_VLLM_SOURCE", "vllm_source")
    if wheel and source:
        raise ValueError(
            "Select one of SUNRISE_CI_VLLM_WHEEL and SUNRISE_CI_VLLM_SOURCE"
        )
    if "vllm0.24.0" in image and not (wheel or source):
        raise ValueError(
            "The 0.24 vendor base needs SUNRISE_CI_VLLM_WHEEL or "
            "SUNRISE_CI_VLLM_SOURCE for a real vLLM 0.28.0 empty build; "
            "otherwise provide a prepared 0.28 stack via SUNRISE_CI_IMAGE"
        )
    devices = text("SUNRISE_CI_VISIBLE_DEVICES", "visible_devices")
    if not re.fullmatch(r"[0-9]+,[0-9]+", devices) or len(set(devices.split(","))) != 2:
        raise ValueError("Reserve two distinct indices in SUNRISE_CI_VISIBLE_DEVICES")
    values = {
        "model_27b": text("SUNRISE_CI_MODEL_27B", "model_27b"),
        "model_35b": text("SUNRISE_CI_MODEL_35B", "model_35b"),
        "flagcx_path": text("SUNRISE_CI_FLAGCX_PATH", "flagcx_path"),
    }
    for value in [*values.values(), wheel, source]:
        if value and not value.startswith("/"):
            raise ValueError(
                "Sunrise model, FlagCX and wheel/source paths must be absolute container paths"
            )
    if not all(values.values()):
        raise ValueError("Sunrise model paths and FlagCX path are required")
    numbers = {}
    for variable, key, default in (
        ("SUNRISE_CI_MAX_MODEL_LEN", "max_model_len", 8192),
        ("SUNRISE_CI_MAX_NUM_SEQS", "max_num_seqs", 8),
        ("SUNRISE_CI_KV_CACHE_MEMORY_BYTES", "kv_cache_memory_bytes", 4294967296),
    ):
        value = int(env.get(variable, "").strip() or config.get(key, default))
        if value <= 0:
            raise ValueError(variable + " must be positive")
        numbers[key] = value
    result.update(
        **numbers,
        **values,
        ci_image=image,
        runner_labels=labels,
        visible_devices=devices,
        container_volumes=sequence("SUNRISE_CI_CONTAINER_VOLUMES", "container_volumes"),
        container_options=text("SUNRISE_CI_CONTAINER_OPTIONS", "container_options"),
        base_python=text("SUNRISE_CI_BASE_PYTHON", "base_python", "python3"),
        vllm_wheel=wheel,
        vllm_source=source,
    )
    return result


def main():
    import yaml

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=".github/configs/sunrise.yml")
    args = parser.parse_args()
    config = resolve_config(yaml.safe_load(Path(args.config).read_text()), os.environ)
    value = json.dumps(config, separators=(",", ":"))
    output = os.environ.get("GITHUB_OUTPUT")
    if output:
        with open(output, "a") as stream:
            stream.write("config=" + value + "\n")
    print(value)


if __name__ == "__main__":
    main()
