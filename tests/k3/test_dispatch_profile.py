"""Early profile selection and operator policy contracts, without GPU arithmetic."""

import json
from pathlib import Path

import pytest
import yaml

from vllm_fl.patches.kimi_k3.profile import is_kimi_k3_profile

ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture(autouse=True)
def clear_profile_cache():
    is_kimi_k3_profile.cache_clear()
    yield
    is_kimi_k3_profile.cache_clear()


def test_no_profile_preserves_default_backend(monkeypatch):
    monkeypatch.delenv("VLLM_FL_CONFIG", raising=False)
    assert not is_kimi_k3_profile()


def test_existing_config_selects_only_k3_profile(tmp_path, monkeypatch):
    config = tmp_path / "dispatch.yaml"
    config.write_text("model_profile: kimi_k3_metax\nprefer: flagos\n")
    monkeypatch.setenv("VLLM_FL_CONFIG", str(config))
    assert is_kimi_k3_profile()


def test_other_model_config_does_not_enable_k3(tmp_path, monkeypatch):
    config = tmp_path / "dispatch.yaml"
    config.write_text("model_profile: another_model\n")
    monkeypatch.setenv("VLLM_FL_CONFIG", str(config))
    assert not is_kimi_k3_profile()


def test_invalid_config_fails_instead_of_silent_fallback(tmp_path, monkeypatch):
    config = tmp_path / "dispatch.yaml"
    config.write_text("- not-a-policy\n")
    monkeypatch.setenv("VLLM_FL_CONFIG", str(config))
    with pytest.raises(ValueError, match="mapping"):
        is_kimi_k3_profile()


def test_generic_ops_remain_library_only_and_local_ops_use_vendor():
    cfg = yaml.safe_load((ROOT / "examples/kimi_k3_metax/dispatch.yaml").read_text())
    assert cfg["strict"] is True and cfg["prefer"] == "flagos"
    for name in ("grouped_topk", "rms_norm", "silu_and_mul", "rotary_embedding"):
        assert cfg["op_backends"][name] == ["flagos"]
    local = {k: v for k, v in cfg["op_backends"].items() if k.startswith("kimi_k3_")}
    assert len(local) >= 16
    assert all(v == ["vendor:metax"] for v in local.values())
    assert cfg["op_backends"]["attention_backend"] == ["vendor:metax"]


def test_example_has_no_source_patch_or_install_hook():
    example = ROOT / "examples/kimi_k3_metax"
    assert not list(example.rglob("*.patch"))
    assert not (example / "prepare.py").exists()
    lock = json.loads((example / "dependency-lock.json").read_text())
    assert (
        lock["dependencies"]["vllm"]["commit"]
        == "c88c603e1f3d3bde7ad18bd7299cb3d502f7b7a7"
    )
