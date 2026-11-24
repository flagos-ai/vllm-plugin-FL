# Coverage tests: vllm_fl/quantization/quant_linear.py.

from types import SimpleNamespace

import pytest
from vllm.model_executor.kernels import linear as linear_kernels
from vllm.platforms import PlatformEnum

from vllm_fl.quantization import quant_linear
from vllm_fl.quantization.w8a8 import linear as w8a8_linear
from vllm_fl.quantization.w8a8 import moe as w8a8_moe
from vllm_fl.quantization.w8a8 import packed as w8a8_packed


@pytest.mark.parametrize(
    ("cuda_alike", "rocm", "cpu", "expected"),
    [
        (True, False, False, PlatformEnum.CUDA),
        (True, True, False, PlatformEnum.CUDA),
        (False, True, False, PlatformEnum.ROCM),
        (False, False, True, PlatformEnum.CPU),
        (False, False, False, PlatformEnum.CUDA),
    ],
)
def test_resolve_source_platform(monkeypatch, cuda_alike, rocm, cpu, expected):
    platform = SimpleNamespace(
        is_cuda_alike=lambda: cuda_alike,
        is_rocm=lambda: rocm,
        is_cpu=lambda: cpu,
    )
    monkeypatch.setattr(quant_linear, "current_platform", platform)

    assert quant_linear._resolve_source_platform() == expected


_DICTS = (
    "_POSSIBLE_KERNELS",
    "_POSSIBLE_INT8_KERNELS",
    "_POSSIBLE_FP8_KERNELS",
    "_POSSIBLE_FP8_BLOCK_KERNELS",
    "_POSSIBLE_MXFP8_KERNELS",
)


def _patch_installers(monkeypatch):
    calls = {}
    monkeypatch.setattr(
        w8a8_linear,
        "register_fl_w8a8_linear_kernel",
        lambda registry: calls.__setitem__("linear", registry) or True,
    )
    monkeypatch.setattr(
        w8a8_packed,
        "install_packed_w8a8_scheme",
        lambda: calls.__setitem__("packed", True) or True,
    )
    monkeypatch.setattr(
        w8a8_moe,
        "install_fl_w8a8_moe_selector",
        lambda: calls.__setitem__("moe", True) or True,
    )
    return calls


def test_add_oot_quant_kernel_inherits_from_source(monkeypatch):
    monkeypatch.setattr(quant_linear, "_resolve_source_platform", lambda: PlatformEnum.CUDA)
    candidate = object()
    sentinel = object()
    for name in _DICTS:
        table = getattr(linear_kernels, name)
        monkeypatch.setitem(table, PlatformEnum.CUDA, [candidate])
        monkeypatch.delitem(table, PlatformEnum.OOT, raising=False)
        monkeypatch.setitem(table, PlatformEnum.ROCM, [sentinel])
    calls = _patch_installers(monkeypatch)

    quant_linear.add_oot_quant_kernel()

    for name in _DICTS:
        table = getattr(linear_kernels, name)
        assert table[PlatformEnum.OOT] == [candidate]
    assert calls["linear"] is linear_kernels._POSSIBLE_INT8_KERNELS
    assert calls["packed"] is True
    assert calls["moe"] is True


def test_add_oot_quant_kernel_keeps_existing_entries(monkeypatch):
    monkeypatch.setattr(quant_linear, "_resolve_source_platform", lambda: PlatformEnum.CUDA)
    existing = ["already"]
    for name in _DICTS:
        table = getattr(linear_kernels, name)
        monkeypatch.setitem(table, PlatformEnum.CUDA, ["candidate"])
        monkeypatch.setitem(table, PlatformEnum.OOT, list(existing))
    _patch_installers(monkeypatch)

    quant_linear.add_oot_quant_kernel()

    for name in _DICTS:
        table = getattr(linear_kernels, name)
        assert table[PlatformEnum.OOT] == existing


def test_add_oot_quant_kernel_uses_empty_source_when_absent(monkeypatch):
    monkeypatch.setattr(
        quant_linear, "_resolve_source_platform", lambda: PlatformEnum.ROCM
    )
    for name in _DICTS:
        table = getattr(linear_kernels, name)
        monkeypatch.delitem(table, PlatformEnum.ROCM, raising=False)
        monkeypatch.delitem(table, PlatformEnum.OOT, raising=False)
    _patch_installers(monkeypatch)

    quant_linear.add_oot_quant_kernel()

    for name in _DICTS:
        assert getattr(linear_kernels, name)[PlatformEnum.OOT] == []
