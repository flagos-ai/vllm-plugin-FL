from __future__ import annotations

import sys

import pytest

from vllm_fl import ascend_custom_ops


@pytest.mark.parametrize(
    ("soc_version", "family", "prebuilt_dir"),
    [
        ("ascend910b1", "ascend910b", "ascend910b1"),
        ("ascend910b2c", "ascend910b", "ascend910b1"),
        ("ascend910b4-1", "ascend910b", "ascend910b1"),
        ("910b3", "ascend910b", "ascend910b1"),
        ("ascend910_9391", "ascend910_93", "ascend910_93"),
        ("ascend910_9392", "ascend910_93", "ascend910_93"),
        ("910c", "ascend910_93", "ascend910_93"),
    ],
)
def test_soc_family_candidates(monkeypatch, soc_version, family, prebuilt_dir):
    monkeypatch.setenv("SOC_VERSION", soc_version)

    assert ascend_custom_ops._soc_family(soc_version) == family
    candidates = ascend_custom_ops._soc_candidates()
    assert candidates[0] == soc_version
    assert prebuilt_dir in candidates


def test_a3_never_falls_back_to_a2(monkeypatch):
    monkeypatch.setenv("SOC_VERSION", "ascend910_9391")

    assert "ascend910b1" not in ascend_custom_ops._soc_candidates()


def test_find_compatible_extension_uses_current_python_abi(tmp_path):
    lib = tmp_path / "lib"
    lib.mkdir()
    current = lib / f"_C_ascend.{sys.implementation.cache_tag}-aarch64-linux-gnu.so"
    other = lib / "_C_ascend.cpython-312-aarch64-linux-gnu.so"
    current.touch()
    other.touch()

    assert ascend_custom_ops._find_compatible_extension(tmp_path) == current


def test_opp_family_validation(tmp_path):
    tbe = tmp_path / "op_impl" / "ai_core" / "tbe"
    (tbe / "kernel" / "ascend910b").mkdir(parents=True)
    (tbe / "config" / "ascend910b").mkdir(parents=True)

    assert ascend_custom_ops._opp_supports_soc_family(tmp_path, "ascend910b")
    assert not ascend_custom_ops._opp_supports_soc_family(tmp_path, "ascend910_93")


def test_custom_op_loader_has_no_external_runtime_fallback(monkeypatch):
    monkeypatch.setattr(ascend_custom_ops, "_ENABLED", None)
    monkeypatch.setattr(ascend_custom_ops, "_load_local", lambda: False)

    assert ascend_custom_ops.enable_custom_op() is False
    assert "_load_vllm_ascend_fallback" not in vars(ascend_custom_ops)
