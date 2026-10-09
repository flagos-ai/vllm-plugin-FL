# Copyright 2026 FlagOS Contributors
"""Resolve administrator-provided THead runner resources."""

import json
import re


def resolve_config(config, env):
    result = dict(config)
    required = (
        "THEAD_CI_IMAGE",
        "THEAD_CI_VISIBLE_DEVICES",
        "THEAD_CI_MODEL_27B",
        "THEAD_CI_MODEL_35B",
    )
    missing = [name for name in required if not env.get(name, "").strip()]
    if missing:
        raise ValueError("Configure repository variables: " + ", ".join(missing))
    image = env["THEAD_CI_IMAGE"].strip()
    if not re.fullmatch(r"[^\s@]+@sha256:[0-9a-f]{64}", image):
        raise ValueError(
            "THEAD_CI_IMAGE must be an existing OCI image pinned by digest"
        )
    labels = json.loads(env.get("THEAD_CI_RUNNER_LABELS") or '["flagcicd-810e"]')
    volumes = json.loads(env.get("THEAD_CI_CONTAINER_VOLUMES") or "[]")
    for name, value in (("runner labels", labels), ("container volumes", volumes)):
        if not isinstance(value, list) or any(
            not isinstance(item, str) or not item.strip() for item in value
        ):
            raise ValueError(name + " must be a JSON array of nonempty strings")
    if not labels:
        raise ValueError("THead requires a registered, dedicated PPU runner")
    options = (
        env.get("THEAD_CI_CONTAINER_OPTIONS") or config["container_options"]
    ).strip()
    if "--privileged" in options:
        raise ValueError(
            "Grant the required PPU devices explicitly; do not use privileged"
        )
    devices = env["THEAD_CI_VISIBLE_DEVICES"].strip()
    if not re.fullmatch(r"[0-9]+(,[0-9]+){3}", devices):
        raise ValueError("Reserve exactly four PPU indices in THEAD_CI_VISIBLE_DEVICES")
    if len(set(devices.split(","))) != 4:
        raise ValueError("Reserved PPU indices must be distinct")
    models = [env["THEAD_CI_MODEL_27B"], env["THEAD_CI_MODEL_35B"]]
    if any(not value.startswith("/") or "\n" in value for value in models):
        raise ValueError("Model variables must be absolute paths inside the container")
    result.update(
        ci_image=image,
        runner_labels=labels,
        container_volumes=volumes,
        container_options=options,
        visible_devices=devices,
        model_27b=models[0],
        model_35b=models[1],
    )
    return result
