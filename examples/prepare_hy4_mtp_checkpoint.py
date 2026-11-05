# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0
"""Restore Hy4's MTP projection from its matching original BF16 checkpoint.

Usage: python examples/prepare_hy4_mtp_checkpoint.py W8A8 BF16 OUTPUT

Both inputs must be indexed safetensors checkpoints. The original checkpoints
are read only. CPU memory holds one changed W8A8 shard and the BF16 projection;
no model or GPU is initialized. The output contains a rewritten shard, config
and index, with symlinks to the remaining W8A8 files. Keep that W8A8 source in
place for the lifetime of the derived checkpoint. The BF16 donor must be the
matching original model, not a cast of quantized weights.

Publication uses Linux's atomic no-replace rename. An existing output is never
replaced; an unsuccessful attempt removes only its own temporary directory.
"""

import argparse
import ctypes
import errno
import json
import os
import shutil
import tempfile
from pathlib import Path

from safetensors import safe_open
from safetensors.torch import load_file, save_file

WEIGHT = "model.mtp_layers.0.eh_proj.weight"
SCALE = "model.mtp_layers.0.eh_proj.weight_scale"
INDEX = "model.safetensors.index.json"


def _read_json(path):
    with path.open(encoding="utf-8") as source:
        return json.load(source)


def _indexed_shard(root, index, name):
    filename = index["weight_map"][name]
    relative = Path(filename)
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError("checkpoint shard paths must stay within their directory")
    path = root / relative
    if not path.is_file():
        raise FileNotFoundError(path)
    return path


def _publish(staging, destination):
    """Atomically rename without replacing a concurrently created output."""
    rename = getattr(ctypes.CDLL(None, use_errno=True), "renameat2", None)
    if rename is None:
        raise RuntimeError("atomic checkpoint publication requires Linux renameat2")
    rename.argtypes = (
        ctypes.c_int,
        ctypes.c_char_p,
        ctypes.c_int,
        ctypes.c_char_p,
        ctypes.c_uint,
    )
    rename.restype = ctypes.c_int
    # AT_FDCWD and RENAME_NOREPLACE; both paths are absolute.
    if rename(-100, os.fsencode(staging), -100, os.fsencode(destination), 1):
        error = ctypes.get_errno()
        if error == errno.EEXIST:
            raise FileExistsError(error, os.strerror(error), str(destination))
        raise OSError(error, os.strerror(error), str(destination))


def prepare_checkpoint(w8a8, bf16, output):
    """Create a separate checkpoint, preserving all other tensor values."""
    requested_output = Path(output).absolute()
    if os.path.lexists(requested_output):
        raise FileExistsError(requested_output)
    base, reference = Path(w8a8).resolve(strict=True), Path(bf16).resolve(strict=True)
    destination = requested_output.resolve()
    if destination.is_relative_to(base) or destination.is_relative_to(reference):
        raise ValueError("output must be outside the read-only source checkpoints")
    config = _read_json(base / "config.json")
    index = _read_json(base / INDEX)
    reference_index = _read_json(reference / INDEX)
    text_config = config.get("text_config", config)
    hidden_size = text_config["hidden_size"]
    layer_count = text_config["num_hidden_layers"]
    if type(hidden_size) is not int or hidden_size <= 0:
        raise ValueError("hidden_size must be a positive integer")
    if type(layer_count) is not int or layer_count < 1:
        raise ValueError("num_hidden_layers must be a positive integer")
    quantization = config.get("quantization_config")
    if quantization is None:
        quantization = text_config.get("quantization_config")
    if not isinstance(quantization, dict) or quantization.get("quant_method") not in (
        "compressed-tensors",
        "compressed_tensors",
    ):
        raise ValueError("W8A8 source requires compressed-tensors quantization")
    ignore = quantization.setdefault("ignore", [])
    if not isinstance(ignore, list) or not all(isinstance(x, str) for x in ignore):
        raise ValueError("compressed-tensors ignore must be a list of module names")
    source_shard = _indexed_shard(base, index, WEIGHT)
    if _indexed_shard(base, index, SCALE) != source_shard:
        raise ValueError("MTP weight and weight_scale must share one W8A8 shard")
    reference_shard = _indexed_shard(reference, reference_index, WEIGHT)
    expected_shape = [hidden_size, 2 * hidden_size]
    with safe_open(source_shard, framework="pt", device="cpu") as source:
        source_weight = source.get_slice(WEIGHT)
        if source_weight.get_dtype() != "I8":
            raise ValueError("W8A8 MTP projection must contain INT8 weight data")
        if source_weight.get_shape() != expected_shape:
            raise ValueError("W8A8 MTP projection shape does not match hidden_size")
        shard_metadata = source.metadata()
    with safe_open(reference_shard, framework="pt", device="cpu") as donor:
        projection = donor.get_slice(WEIGHT)
        if projection.get_dtype() != "BF16":
            raise ValueError("reference MTP projection must contain original BF16 data")
        if projection.get_shape() != expected_shape:
            raise ValueError("reference MTP projection shape does not match W8A8")
        native_weight = donor.get_tensor(WEIGHT)
    tensors = load_file(source_shard, device="cpu")
    old_bytes = tensors[WEIGHT].numel() * tensors[WEIGHT].element_size()
    removed_scale = tensors.pop(SCALE)
    old_bytes += removed_scale.numel() * removed_scale.element_size()
    del removed_scale
    tensors[WEIGHT] = native_weight
    new_bytes = native_weight.numel() * native_weight.element_size()
    metadata = index.setdefault("metadata", {})
    total_size = metadata.get("total_size")
    if type(total_size) is not int or total_size < old_bytes:
        raise ValueError("checkpoint index requires a valid metadata.total_size")
    metadata["total_size"] = total_size + new_bytes - old_bytes
    del index["weight_map"][SCALE]
    for name in (
        "model.mtp_layers.0.eh_proj",
        f"model.layers.{layer_count}.eh_proj",
    ):
        if name not in ignore:
            ignore.append(name)
    # Check all unchanged indexed shards before creating a derived directory.
    checked_shards = set()
    for name, filename in index["weight_map"].items():
        if filename not in checked_shards:
            _indexed_shard(base, index, name)
            checked_shards.add(filename)
    destination.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(
        tempfile.mkdtemp(prefix=f".{destination.name}.", dir=destination.parent)
    )
    try:
        changed_path = source_shard.relative_to(base)
        (staging / changed_path).parent.mkdir(parents=True, exist_ok=True)
        save_file(tensors, staging / changed_path, metadata=shard_metadata)
        for path in base.rglob("*"):
            if not path.is_file():
                continue
            relative = path.relative_to(base)
            if relative in (Path("config.json"), Path(INDEX), changed_path):
                continue
            target = staging / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            target.symlink_to(path)
        for filename, data in (("config.json", config), (INDEX, index)):
            (staging / filename).write_text(
                json.dumps(data, indent=2) + "\n", encoding="utf-8"
            )
        _publish(staging, destination)
    finally:
        if staging.exists():
            shutil.rmtree(staging)
    return destination


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("w8a8", type=Path)
    parser.add_argument("bf16", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    try:
        prepare_checkpoint(args.w8a8, args.bf16, args.output)
    except (KeyError, OSError, RuntimeError, ValueError) as error:
        parser.exit(1, f"checkpoint preparation failed: {error}\n")


if __name__ == "__main__":
    main()
