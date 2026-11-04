# SPDX-License-Identifier: Apache-2.0
"""CPU launch contracts and real Ascend temperature arithmetic regression."""

import importlib.util
import sys
import types
from pathlib import Path
from unittest.mock import patch

import pytest
import torch

_ROOT = Path(__file__).resolve().parents[2]


def _adapter():
    spec = importlib.util.spec_from_file_location(
        "ascend_v2_temperature_under_test",
        _ROOT / "vllm_fl/patches/ascend_v2_sampling.py",
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class _KernelSpy:
    def __init__(self):
        self.launches = []

    def __getitem__(self, grid):
        def launch(*args, **kwargs):
            self.launches.append((grid, args, kwargs))

        return launch


def _reference(logits, mapping, temperatures):
    result = logits.clone()
    for row, slot in enumerate(mapping.tolist()):
        temperature = float(temperatures[slot])
        if temperature not in (0.0, 1.0):
            result[row] /= temperatures[slot]
    return result


@pytest.mark.parametrize("vocab_size", [1, 1024, 1025, 151936])
def test_temperature_uses_immutable_source_and_independent_physical_strides(vocab_size):
    adapter = _adapter()
    spy = _KernelSpy()
    kernels = types.ModuleType(adapter._ASCEND_TEMPERATURE_KERNEL_MODULE)
    kernels._temperature_kernel = spy
    storage = torch.arange(4 * (vocab_size + 13), dtype=torch.float32).view(4, -1)
    logits = storage[:, :vocab_size]
    mapping = torch.tensor([3, 0, 2, 1], dtype=torch.int64)
    temperatures = torch.tensor([0.7, 1.0, 0.0, 1.3])
    with patch.dict(sys.modules, {kernels.__name__: kernels}):
        adapter.apply_ascend_v2_temperature(logits, mapping, temperatures)
    grid, args, kwargs = spy.launches[0]
    assert grid == (4, (vocab_size + 1023) // 1024)
    assert kwargs == {"BLOCK_SIZE": 1024}
    source, source_stride, output, output_stride, indices, scales, size = args
    assert source.data_ptr() != logits.data_ptr()
    torch.testing.assert_close(source, logits, rtol=0, atol=0)
    assert source_stride == vocab_size
    assert output is logits and output_stride == vocab_size + 13
    assert indices is mapping and scales is temperatures and size == vocab_size


@pytest.mark.parametrize("shape", [(0, 1031), (4, 0)])
def test_empty_temperature_logits_need_no_kernel_module(shape):
    adapter = _adapter()
    with patch.object(adapter.importlib, "import_module", side_effect=AssertionError):
        adapter.apply_ascend_v2_temperature(
            torch.empty(shape), torch.empty(0, dtype=torch.int64), torch.empty(0)
        )


@pytest.mark.gpu
def test_temperature_matches_cpu_with_slot_reuse_and_strided_output():
    if importlib.util.find_spec("torch_npu") is None:
        pytest.skip("torch_npu is unavailable")
    import torch_npu  # noqa: F401

    if not torch.npu.is_available():
        pytest.skip("An Ascend NPU is unavailable")
    adapter = _adapter()
    device = torch.device("npu", torch.npu.current_device())
    generator = torch.Generator().manual_seed(631)
    temperatures = torch.tensor([0.7, 1.0, 0.0, 1.3], dtype=torch.float32)
    temperatures_device = temperatures.to(device)
    for vocab_size in (1031, 8193, 151936):
        original = torch.randn(4, vocab_size, generator=generator) * 2
        original[0, 0] = -2.0
        original[0, -1] = 3.0
        for order in ([0, 1, 2, 3], [3, 0, 2, 1]):
            mapping = torch.tensor(order, dtype=torch.int64)
            mapping_device = mapping.to(device)
            storage = torch.full((4, vocab_size + 13), 12345.0).to(device)
            values = storage[:, :vocab_size]
            values.copy_(original)
            expected = original.clone()
            for _ in range(3):
                # Reusing a strided output must still divide only once per call.
                expected = _reference(expected, mapping, temperatures)
                adapter.apply_ascend_v2_temperature(
                    values, mapping_device, temperatures_device
                )
                torch.npu.synchronize()
                torch.testing.assert_close(values.cpu(), expected, rtol=1e-6, atol=1e-6)
                torch.testing.assert_close(
                    storage[:, vocab_size:].cpu(),
                    torch.full((4, 13), 12345.0),
                    rtol=0,
                    atol=0,
                )
                for row, slot in enumerate(order):
                    if float(temperatures[slot]) in (0.0, 1.0):
                        torch.testing.assert_close(
                            values[row].cpu(), original[row], rtol=0, atol=0
                        )
