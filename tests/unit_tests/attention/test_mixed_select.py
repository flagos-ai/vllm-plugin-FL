"""Preset selection, failure fallback, and INT8 mixed execution boundaries."""

from types import SimpleNamespace

import pytest
import torch

from vllm_fl import device_profiles
from vllm_fl.attention import kv_int8 as k8, mixed_select as ms
from vllm_fl.attention.decode_config import DecodeConfig

DEV = torch.device("cuda", 0)


@pytest.fixture(autouse=True)
def clean(monkeypatch):
    profile = device_profiles.PROFILES[device_profiles.HYBRID_INT8_PROFILE]
    for name in profile["switches"]:
        monkeypatch.setenv(name, "auto")
    monkeypatch.setenv(ms.ENV_WIDE, "0")
    monkeypatch.setenv(ms.ENV_FIXED_ANCHOR, "0")
    monkeypatch.setattr(device_profiles, "_DEFAULTED", {})
    monkeypatch.setattr(ms, "_decisions", {})


@pytest.fixture
def profile(monkeypatch):
    profile = device_profiles.PROFILES[device_profiles.HYBRID_INT8_PROFILE]
    monkeypatch.setattr(device_profiles, "lookup", lambda _: profile)
    device_profiles.apply_defaults(0)
    return profile


def choose(**kwargs):
    config = DecodeConfig(8, 32, 1, 1, ((1, 1),))
    return ms.resolve(DEV, decode=(None, config), scratch_tokens=65536, **kwargs)


def test_failed_verification_keeps_fused_rules(profile, monkeypatch):
    monkeypatch.setattr(
        ms, "Verifier", lambda *a: SimpleNamespace(check=lambda *a: (False, "mismatch"))
    )
    d = choose()
    assert not d.table
    assert d.lookup(8, 128, 4096) == ms.Choice("fused", "reblock")
    monkeypatch.setattr(ms, "_decisions", {})

    def fail(*args):
        raise RuntimeError("compile")

    monkeypatch.setattr(ms, "Verifier", fail)
    assert not choose().table


@pytest.fixture
def decode_config():
    if not torch.cuda.is_available():
        pytest.skip("needs a GPU")
    profile = device_profiles.PROFILES[device_profiles.HYBRID_INT8_PROFILE]
    return k8.DecodeWorkspace(DEV), DecodeConfig(**profile["int8_decode"])


def test_int8_mixed_modes_match_cpu_on_strided_tensors(decode_config):
    shape = ([130, 1, 0, 1, 129], [130, 300, 0, 1, 1025])
    for mode in ms.MODES:
        verifier = ms.Verifier(DEV, 16, True, ms.decode_factory(decode_config), 4096)
        verifier.BATCHES = (shape,)
        for _, _, case, _ in verifier._cases():
            for name in ("q", "out"):
                old = case[name]
                storage = torch.empty(
                    (old.shape[0] * 2, *old.shape[1:]),
                    dtype=old.dtype,
                    device=DEV,
                )
                case[name] = storage[::2].copy_(old)
        assert verifier.check("p64x32", mode) == (True, "ok")
