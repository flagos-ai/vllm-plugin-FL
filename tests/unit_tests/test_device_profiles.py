"""Worker defaults preserve explicit overrides and unknown-device behavior."""

import os
from types import SimpleNamespace

import pytest

from vllm_fl import device_profiles as profiles


def _config(*, heads=16, kv_heads=2, head_size=128, dtype=None):
    import torch

    model = SimpleNamespace(
        dtype=dtype or torch.bfloat16,
        get_num_attention_heads=lambda _parallel: heads,
        get_num_kv_heads=lambda _parallel: kv_heads,
        get_head_size=lambda: head_size,
    )
    parallel = SimpleNamespace(
        tensor_parallel_size=1,
        pipeline_parallel_size=1,
        decode_context_parallel_size=1,
        use_ubatching=False,
    )
    cache = SimpleNamespace(cache_dtype="auto", block_size=16)
    return SimpleNamespace(
        model_config=model,
        parallel_config=parallel,
        cache_config=cache,
    )


@pytest.fixture
def profile_env(monkeypatch):
    monkeypatch.setattr(profiles, "_DEFAULTED", {})
    monkeypatch.setattr(profiles, "_ACTIVE", {})
    profile = profiles.PROFILES[profiles.HYBRID_INT8_PROFILE]
    for name in profile["switches"]:
        monkeypatch.setenv(name, "auto")
    monkeypatch.setattr(profiles, "lookup", lambda _: profile)
    return profile


def test_unknown_model_does_not_change_environment(profile_env, monkeypatch):
    monkeypatch.setattr(profiles, "lookup", lambda _: None)
    before = dict(os.environ)
    profiles.apply_defaults(0)
    assert dict(os.environ) == before


def test_profile_selection_requires_validated_model_geometry(monkeypatch):
    monkeypatch.setattr(
        profiles, "_profile_name", lambda _: profiles.NATIVE_SPARSE_PROFILE
    )
    assert (
        profiles.select(0, _config())
        is profiles.PROFILES[profiles.NATIVE_SPARSE_PROFILE]
    )
    assert profiles.select(0, _config(heads=32)) is None
    assert profiles.select(0, _config(kv_heads=4)) is None
    assert profiles.select(0, _config(head_size=64)) is None


def test_unmatched_model_does_not_activate_profile(monkeypatch):
    monkeypatch.setattr(
        profiles, "_profile_name", lambda _: profiles.HYBRID_INT8_PROFILE
    )
    monkeypatch.setattr(profiles, "_ACTIVE", {0: {"stale": True}})
    before = dict(os.environ)
    profiles.apply_defaults(0, _config(heads=32))
    assert profiles.lookup(0) is None
    assert dict(os.environ) == before
