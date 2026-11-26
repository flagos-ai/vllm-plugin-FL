"""Startup selection boundaries and fail-closed behavior for large-M mm tiles."""

import importlib
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import yaml

from vllm_fl.ops import mm_large_tiles


def _results(candidate_ms):
    return [
        {"baseline_median_ms": 100.0, "candidate_median_ms": value}
        for value in candidate_ms
    ]


def test_measurement_gate_requires_a_clear_win_and_bounds_every_shape():
    assert mm_large_tiles._passes_measurement_gate(_results((97.0, 102.0, 102.0)))
    assert not mm_large_tiles._passes_measurement_gate(_results((100.0, 101.0, 100.0)))
    assert not mm_large_tiles._passes_measurement_gate(_results((97.0, 102.01, 100.0)))
    assert not mm_large_tiles._passes_measurement_gate(_results((97.0, 102.0)))
    assert not mm_large_tiles._passes_measurement_gate(
        _results((97.0, float("nan"), 100.0))
    )


def test_explicit_enable_still_fails_closed_when_raw_check_fails(monkeypatch):
    calls = []
    backend = SimpleNamespace(set_mm_large_tiles=lambda enabled: calls.append(enabled))
    monkeypatch.setattr(mm_large_tiles, "_get_mm_module", lambda: backend)
    monkeypatch.setattr(mm_large_tiles.device_profiles, "lookup", lambda _: None)
    monkeypatch.setattr(
        mm_large_tiles,
        "_validate_raw_candidates",
        lambda *_: (_ for _ in ()).throw(RuntimeError("synthetic raw-check failure")),
    )
    monkeypatch.setenv(mm_large_tiles._SWITCH, "1")

    assert not mm_large_tiles.install(SimpleNamespace(type="cuda", index=0))
    assert calls == [False, False]


def test_backend_pruner_excludes_large_tiles_outside_bf16_large_nt(monkeypatch):
    try:
        backend = importlib.import_module("flag_gems.runtime.backend._metax.ops.mm")
    except Exception as exc:
        pytest.skip(
            f"native BF16 backend is unavailable in this test environment: {exc}"
        )

    monkeypatch.setattr(backend, "_MM_LARGE_TILES_ENABLED", True)
    configs = list(backend._MM_LARGE_TILE_CONFIGS)
    tune_configs = yaml.safe_load(
        (Path(backend.__file__).resolve().parents[1] / "tune_configs.yaml").read_text()
    )
    base_config_key_sets = {
        frozenset(config["META"]) for config in tune_configs["mm_nt"]
    }
    assert all(frozenset(config.kwargs) in base_config_key_sets for config in configs)

    def args(m, n, k, dtype):
        tensor = SimpleNamespace(dtype=dtype)
        return {"M": m, "N": n, "K": k, "A": tensor, "B": tensor, "C": tensor}

    def has_candidate(pruned):
        return any(
            backend._mm_large_tile_signature(config)
            in backend._MM_LARGE_TILE_SIGNATURES
            for config in pruned
        )

    assert has_candidate(
        backend._prune_mm_dense_configs_nt(
            configs, args(1024, 128, 256, torch.bfloat16)
        )
    )
    for shape in (
        args(1023, 128, 256, torch.bfloat16),
        args(1024, 128, 256, torch.float16),
        args(1024, 128, 256, torch.float32),
        args(1024, 127, 256, torch.bfloat16),
    ):
        assert not has_candidate(backend._prune_mm_dense_configs_nt(configs, shape))
    assert not has_candidate(
        backend._prune_mm_dense_configs(configs, args(1024, 128, 256, torch.bfloat16))
    )
