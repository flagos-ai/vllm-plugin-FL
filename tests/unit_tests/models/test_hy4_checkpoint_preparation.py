# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0
"""Small real safetensors checkpoints; CPU only, with no model initialization."""

import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import pytest
import torch
from safetensors import safe_open
from safetensors.torch import load_file, save_file

ROOT = Path(__file__).resolve().parents[3]


def _load_script():
    path = ROOT / "examples/prepare_hy4_mtp_checkpoint.py"
    spec = importlib.util.spec_from_file_location("hy4_checkpoint_preparation", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _write_json(path, data):
    path.write_text(json.dumps(data), encoding="utf-8")


def _checkpoints(tmp_path, *, reference_dtype=torch.bfloat16, reference_shape=(4, 8)):
    base, reference = tmp_path / "w8a8", tmp_path / "bf16"
    base.mkdir()
    reference.mkdir()
    weight = "model.mtp_layers.0.eh_proj.weight"
    scale = "model.mtp_layers.0.eh_proj.weight_scale"
    local = {
        weight: torch.arange(32, dtype=torch.int8).reshape(4, 8),
        scale: torch.full((4, 1), 0.25),
        "model.mtp_layers.0.enorm.weight": torch.tensor(
            [0.0, -0.0, 1.25, -2.5], dtype=torch.bfloat16
        ),
    }
    other = {"model.layers.0.weight": torch.tensor([1.0, -3.5])}
    save_file(local, base / "shard-a.safetensors", metadata={"format": "pt"})
    save_file(other, base / "shard-b.safetensors")
    projection = (
        torch.arange(torch.tensor(reference_shape).prod().item())
        .reshape(reference_shape)
        .to(reference_dtype)
    )
    save_file({weight: projection}, reference / "original.safetensors")
    _write_json(
        reference / "model.safetensors.index.json",
        {"weight_map": {weight: "original.safetensors"}},
    )
    total = sum(
        x.numel() * x.element_size() for x in (*local.values(), *other.values())
    )
    _write_json(
        base / "model.safetensors.index.json",
        {
            "metadata": {"total_size": total, "other": "keep"},
            "weight_map": {
                **{name: "shard-a.safetensors" for name in local},
                **{name: "shard-b.safetensors" for name in other},
            },
        },
    )
    _write_json(
        base / "config.json",
        {
            "hidden_size": 4,
            "num_hidden_layers": 2,
            "quantization_config": {
                "quant_method": "compressed-tensors",
                "ignore": ["lm_head"],
            },
        },
    )
    _write_json(base / "tokenizer_config.json", {"tokenizer_class": "test"})
    return base, reference, projection, local, other


def test_restores_original_weight_preserves_other_tensors_and_sources(tmp_path):
    script = _load_script()
    base, reference, projection, local, other = _checkpoints(tmp_path)
    original = {path: path.read_bytes() for path in base.iterdir()}
    donor_original = {path: path.read_bytes() for path in reference.iterdir()}
    output = tmp_path / "derived"
    assert script.prepare_checkpoint(base, reference, output) == output
    changed = load_file(output / "shard-a.safetensors", device="cpu")
    assert changed[script.WEIGHT].dtype == torch.bfloat16
    assert torch.equal(changed[script.WEIGHT], projection)
    assert script.SCALE not in changed
    for name, tensor in local.items():
        if name not in (script.WEIGHT, script.SCALE):
            assert torch.equal(
                changed[name].view(torch.uint8), tensor.view(torch.uint8)
            )
    assert not (output / "shard-a.safetensors").is_symlink()
    assert (output / "shard-b.safetensors").is_symlink()
    assert (output / "shard-b.safetensors").resolve() == base / "shard-b.safetensors"
    assert (output / "tokenizer_config.json").is_symlink()
    with safe_open(output / "shard-a.safetensors", framework="pt") as handle:
        assert handle.metadata() == {"format": "pt"}
    index = json.loads((output / script.INDEX).read_text())
    assert script.SCALE not in index["weight_map"]
    assert index["metadata"]["other"] == "keep"
    assert index["metadata"]["total_size"] == sum(
        tensor.numel() * tensor.element_size()
        for tensor in (*changed.values(), *other.values())
    )
    config = json.loads((output / "config.json").read_text())
    assert config["quantization_config"]["ignore"] == [
        "lm_head",
        "model.mtp_layers.0.eh_proj",
        "model.layers.2.eh_proj",
    ]
    assert all(path.read_bytes() == data for path, data in original.items())
    assert all(path.read_bytes() == data for path, data in donor_original.items())


@pytest.mark.parametrize("dtype", [torch.int8, torch.float16, torch.float32])
def test_rejects_non_bf16_reference(tmp_path, dtype):
    script = _load_script()
    base, reference, *_ = _checkpoints(tmp_path, reference_dtype=dtype)
    output = tmp_path / "derived"
    with pytest.raises(ValueError, match="original BF16"):
        script.prepare_checkpoint(base, reference, output)
    assert not output.exists()
    assert list(tmp_path.glob(".derived.*")) == []


def test_rejects_reference_shape_mismatch(tmp_path):
    script = _load_script()
    base, reference, *_ = _checkpoints(tmp_path, reference_shape=(4, 7))
    with pytest.raises(ValueError, match="shape"):
        script.prepare_checkpoint(base, reference, tmp_path / "derived")
    assert list(tmp_path.glob(".derived.*")) == []


def test_rejects_existing_output_without_touching_it(tmp_path):
    script = _load_script()
    base, reference, *_ = _checkpoints(tmp_path)
    output = tmp_path / "derived"
    output.mkdir()
    (output / "keep.txt").write_text("untouched")
    with pytest.raises(FileExistsError):
        script.prepare_checkpoint(base, reference, output)
    assert (output / "keep.txt").read_text() == "untouched"


def test_cli_accepts_checkpoint_directories(tmp_path):
    base, reference, projection, *_ = _checkpoints(tmp_path)
    output = tmp_path / "derived"
    result = subprocess.run(
        [
            sys.executable,
            str(ROOT / "examples/prepare_hy4_mtp_checkpoint.py"),
            str(base),
            str(reference),
            str(output),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert torch.equal(
        load_file(output / "shard-a.safetensors")["model.mtp_layers.0.eh_proj.weight"],
        projection,
    )


def test_failed_write_cleans_only_its_temporary_directory(tmp_path, monkeypatch):
    script = _load_script()
    base, reference, *_ = _checkpoints(tmp_path)
    keep = tmp_path / ".derived.unrelated"
    keep.mkdir()
    (keep / "keep.txt").write_text("untouched")

    def fail(*args, **kwargs):
        raise RuntimeError("write failed")

    monkeypatch.setattr(script, "save_file", fail)
    with pytest.raises(RuntimeError, match="write failed"):
        script.prepare_checkpoint(base, reference, tmp_path / "derived")
    assert list(tmp_path.glob(".derived.*")) == [keep]
    assert (keep / "keep.txt").read_text() == "untouched"


def test_concurrent_output_is_never_replaced(tmp_path, monkeypatch):
    script = _load_script()
    base, reference, *_ = _checkpoints(tmp_path)
    publish = script._publish
    output = tmp_path / "derived"

    def race(staging, destination):
        destination.mkdir()
        (destination / "keep.txt").write_text("concurrent output")
        publish(staging, destination)

    monkeypatch.setattr(script, "_publish", race)
    with pytest.raises(OSError):
        script.prepare_checkpoint(base, reference, output)
    assert (output / "keep.txt").read_text() == "concurrent output"
    assert list(tmp_path.glob(".derived.*")) == []
