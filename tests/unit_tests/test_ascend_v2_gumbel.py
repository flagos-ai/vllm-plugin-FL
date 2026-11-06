# SPDX-License-Identifier: Apache-2.0
"""Murmur3 logical-shift regression and independent real-NPU sampling oracle."""

import ast
import importlib.util
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch


class _Unsigned32WithArithmeticShift:
    """Model the observed NPU signed-shift lowering and uint32 arithmetic."""

    def __init__(self, value):
        self.bits = int(value) & 0xFFFFFFFF

    def __int__(self):
        return self.bits

    def __and__(self, other):
        return type(self)(self.bits & int(other))

    def __or__(self, other):
        return type(self)(self.bits | int(other))

    def __xor__(self, other):
        return type(self)(self.bits ^ int(other))

    def __mul__(self, other):
        return type(self)(self.bits * int(other))

    def __add__(self, other):
        return type(self)(self.bits + int(other))

    def __lshift__(self, shift):
        return type(self)(self.bits << shift)

    def __rshift__(self, shift):
        signed = self.bits if self.bits < 0x80000000 else self.bits - 0x100000000
        return type(self)(signed >> shift)


def _production_integer_helpers():
    path = Path(__file__).resolve().parents[2] / "vllm_fl/patches/ascend_v2_gumbel.py"
    tree = ast.parse(path.read_text(encoding="utf-8"))
    names = {"_murmur3_rotl32", "_murmur3_mix", "_murmur3_fmix32"}
    nodes = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in names]
    for node in nodes:
        node.decorator_list = []
    namespace = {"tl": SimpleNamespace(constexpr=int)}
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(path), "exec"), namespace)
    return namespace


def _scalar_mix(h, key):
    key = (key * 0xCC9E2D51) & 0xFFFFFFFF
    key = ((key << 15) | (key >> 17)) & 0xFFFFFFFF
    key = (key * 0x1B873593) & 0xFFFFFFFF
    h ^= key
    h = ((h << 13) | (h >> 19)) & 0xFFFFFFFF
    return (h * 5 + 0xE6546B64) & 0xFFFFFFFF


def test_murmur_helpers_preserve_unsigned_bits_under_arithmetic_right_shift():
    functions = _production_integer_helpers()
    values = [0, 1, 0x7FFFFFFF, 0x80000000, 0xFFFFFFFF, 0xCC9E2D51]
    values += np.random.default_rng(13).integers(0, 1 << 32, 100).tolist()
    for value in values:
        wrapped = _Unsigned32WithArithmeticShift(value)
        for shift in (13, 15):
            expected = ((value << shift) | (value >> (32 - shift))) & 0xFFFFFFFF
            assert int(functions["_murmur3_rotl32"](wrapped, shift)) == expected
        assert int(functions["_murmur3_mix"](wrapped, wrapped)) == _scalar_mix(
            value, value
        )
        expected = value ^ (value >> 16)
        expected = (expected * 0x85EBCA6B) & 0xFFFFFFFF
        expected ^= expected >> 13
        expected = (expected * 0xC2B2AE35) & 0xFFFFFFFF
        expected ^= expected >> 16
        assert int(functions["_murmur3_fmix32"](wrapped)) == expected


def _cpu_noise(seed, position, vocab_size):
    def mix(h, key):
        key = np.asarray(key, dtype=np.uint32) * np.uint32(0xCC9E2D51)
        key = (key << np.uint32(15)) | (key >> np.uint32(17))
        key = key * np.uint32(0x1B873593)
        h = np.asarray(h, dtype=np.uint32) ^ key
        h = (h << np.uint32(13)) | (h >> np.uint32(19))
        return h * np.uint32(5) + np.uint32(0xE6546B64)

    seed = int(seed) & ((1 << 64) - 1)
    with np.errstate(over="ignore"):
        h = mix(np.uint32(0), np.uint32(seed & 0xFFFFFFFF))
        h = mix(h, np.uint32(seed >> 32))
        h = mix(h, np.uint32(position & 0xFFFFFFFF))
        h = mix(h, np.arange(vocab_size, dtype=np.uint32)) ^ np.uint32(16)
        h ^= h >> np.uint32(16)
        h *= np.uint32(0x85EBCA6B)
        h ^= h >> np.uint32(13)
        h *= np.uint32(0xC2B2AE35)
        h ^= h >> np.uint32(16)
    hi = (h >> np.uint32(16)).astype(np.float32)
    lo = (h & np.uint32(0xFFFF)).astype(np.float32)
    u = hi * np.float32(2**-16) + (lo + np.float32(0.5)) * np.float32(2**-32)
    # Use exact log1p as an independent numerical reference. Fixture winning
    # margins are over 0.16, far beyond the upstream polynomial's 6e-7 error.
    return -np.log(-np.log1p(-u))


@pytest.mark.gpu
def test_seeded_gumbel_matches_independent_cpu_with_slot_permutations():
    if importlib.util.find_spec("torch_npu") is None:
        pytest.skip("torch_npu is unavailable")
    import torch_npu  # noqa: F401

    if not torch.npu.is_available():
        pytest.skip("An Ascend NPU is unavailable")
    from vllm.v1.worker.gpu.sample import gumbel

    from vllm_fl.patches.ascend_v2_sampling import install_ascend_v2_sampling_shims

    assert install_ascend_v2_sampling_shims()
    device = torch.device("npu", torch.npu.current_device())
    vocab_size = 151936
    original = (
        torch.randn(4, vocab_size, generator=torch.Generator().manual_seed(631)) * 2
    )
    for column, value in (
        (7, -2),
        (31, 3),
        (1023, 2),
        (1024, -4),
        (8191, 4),
        (8192, -3),
    ):
        original[:, column] = value
    temperatures = torch.tensor([0.7, 1.0, 0.0, 1.3])
    positions = torch.tensor([5, 13, 9, 8], dtype=torch.int64)
    values = original.to(device)
    for seeds in ([7, 7, 7, 7], [7, 13, (1 << 40) + 7, -31]):
        seed_device = torch.tensor(seeds, dtype=torch.int64).to(device)
        for order in ([0, 1, 2, 3], [3, 0, 2, 1]):
            expected = []
            for row, slot in enumerate(order):
                scores = original[row].numpy().copy()
                if float(temperatures[slot]) != 0:
                    scores += _cpu_noise(seeds[slot], int(positions[row]), vocab_size)
                expected.append(int(np.argmax(scores)))
            mapping = torch.tensor(order, dtype=torch.int64).to(device)
            for _ in range(3):
                actual = gumbel.gumbel_sample(
                    values,
                    mapping,
                    temperatures.to(device),
                    seed_device,
                    positions.to(device),
                    apply_temperature=False,
                    is_drafting=False,
                )
                torch.npu.synchronize()
                assert actual.cpu().tolist() == expected
                torch.testing.assert_close(values.cpu(), original, rtol=0, atol=0)
