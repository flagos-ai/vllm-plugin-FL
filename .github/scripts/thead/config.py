# Copyright 2026 FlagOS Contributors
"""Resolve versioned THead resources with optional administrator overrides."""

import json
import re


def resolve_config(config, env):
    result = dict(config)

    def text_value(variable, key, default=None):
        override = env.get(variable, "")
        if not isinstance(override, str):
            raise ValueError(variable + " must be text")
        value = override.strip() or config.get(key, default)
        if not isinstance(value, str) or not value.strip():
            raise ValueError("Configure THead resource: " + variable)
        return value.strip()

    image = text_value("THEAD_CI_IMAGE", "ci_image")
    if not re.fullmatch(r"[^\s@]+@sha256:[0-9a-f]{64}", image):
        raise ValueError(
            "THEAD_CI_IMAGE must be an existing OCI image pinned by digest"
        )

    def list_value(variable, key, default):
        override = env.get(variable, "")
        if not isinstance(override, str):
            raise ValueError(variable + " must be JSON text")
        return json.loads(override) if override.strip() else config.get(key, default)

    labels = list_value("THEAD_CI_RUNNER_LABELS", "runner_labels", ["flagcicd-810e"])
    volumes = list_value("THEAD_CI_CONTAINER_VOLUMES", "container_volumes", [])
    for name, value in (("runner labels", labels), ("container volumes", volumes)):
        if not isinstance(value, list) or any(
            not isinstance(item, str) or not item.strip() for item in value
        ):
            raise ValueError(name + " must be a JSON array of nonempty strings")
    if not labels:
        raise ValueError("THead requires a registered, dedicated PPU runner")
    options = text_value("THEAD_CI_CONTAINER_OPTIONS", "container_options")
    if "--privileged" in options:
        raise ValueError(
            "Grant the required PPU devices explicitly; do not use privileged"
        )
    devices = text_value("THEAD_CI_VISIBLE_DEVICES", "visible_devices")
    if not re.fullmatch(r"[0-9]+(,[0-9]+){3}", devices):
        raise ValueError("Reserve exactly four PPU indices in THEAD_CI_VISIBLE_DEVICES")
    if len(set(devices.split(","))) != 4:
        raise ValueError("Reserved PPU indices must be distinct")
    models = [
        text_value("THEAD_CI_MODEL_27B", "model_27b"),
        text_value("THEAD_CI_MODEL_35B", "model_35b"),
    ]
    base_python = text_value(
        "THEAD_CI_BASE_PYTHON", "base_python", "/opt/thead/venv/bin/python"
    )
    if any(
        not value.startswith("/") or any(c in value for c in "\r\n\0")
        for value in [*models, base_python]
    ):
        raise ValueError("Models and Python must be absolute container paths")
    result.update(
        ci_image=image,
        runner_labels=labels,
        container_volumes=volumes,
        container_options=options,
        visible_devices=devices,
        model_27b=models[0],
        model_35b=models[1],
        base_python=base_python,
    )
    return result
