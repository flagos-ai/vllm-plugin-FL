# SPDX-License-Identifier: Apache-2.0
"""CPU launch-contract tests for the Ascend V2 penalty tile adapter."""

import importlib.util
import sys
import types
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch


class _KernelSpy:
    def __init__(self):
        self.launches = []

    def __getitem__(self, grid):
        def launch(*args, **kwargs):
            self.launches.append((grid, args, kwargs))

        return launch


class AscendV2SamplingTests(unittest.TestCase):
    def setUp(self):
        path = (
            Path(__file__).resolve().parents[2]
            / "vllm_fl/patches/ascend_v2_sampling.py"
        )
        spec = importlib.util.spec_from_file_location(
            "ascend_v2_sampling_under_test", path
        )
        self.adapter = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(self.adapter)
        self.kernel = _KernelSpy()
        self.upstream_launcher = object()
        self.penalties = types.ModuleType("vllm.v1.worker.gpu.sample.penalties")
        self.penalties._penalties_kernel = self.kernel
        self.penalties.apply_penalties = self.upstream_launcher
        self.ascend_kernel = types.ModuleType(
            "vllm_fl.patches.ascend_v2_penalties_kernel"
        )
        self.ascend_kernel._penalties_kernel = self.kernel
        self.platform = types.ModuleType("vllm.platforms")
        self.platform.current_platform = SimpleNamespace(device_type="npu")
        modules = {
            "vllm.platforms": self.platform,
            "vllm.v1.worker.gpu.sample.penalties": self.penalties,
            "vllm_fl.patches.ascend_v2_penalties_kernel": self.ascend_kernel,
        }
        module_patch = patch.dict(sys.modules, modules)
        module_patch.start()
        self.addCleanup(module_patch.stop)

    def inputs(self, vocab_size=1031, num_tokens=4):
        logits = torch.zeros(num_tokens, vocab_size + 13)[:, :vocab_size]
        prompt_words = (vocab_size + 31) // 32
        prompt = torch.zeros(3, prompt_words + 5, dtype=torch.int32)[:, :prompt_words]
        counts = torch.zeros(3, vocab_size + 7, dtype=torch.int32)[:, :vocab_size]
        return (
            logits,
            torch.tensor([2, 0, 2, 1][:num_tokens], dtype=torch.int32),
            torch.tensor([31, 32, 31, 1024][:num_tokens], dtype=torch.int32),
            torch.tensor([0, 0, 1, 0][:num_tokens], dtype=torch.int32),
            torch.tensor([1.2, 0.7, 1.7]),
            torch.tensor([0.4, -0.25, 0.5]),
            torch.tensor([-0.3, 0.6, 0.2]),
            prompt,
            counts,
        )

    def test_independent_source_and_output_preserve_state_and_row_strides(self):
        inputs = self.inputs()
        self.adapter.apply_ascend_v2_penalties(*inputs)
        grid, args, kwargs = self.kernel.launches[0]
        self.assertEqual(grid, (4, 2))
        self.assertEqual(kwargs, {"BLOCK_SIZE": 1024})
        source = args[0]
        self.assertNotEqual(source.data_ptr(), inputs[0].data_ptr())
        torch.testing.assert_close(source, inputs[0], rtol=0, atol=0)
        self.assertEqual(source.stride(0), inputs[0].shape[1])
        expected = (
            source,
            source.stride(0),
            inputs[0],
            inputs[0].stride(0),
            *inputs[1:7],
            inputs[7],
            inputs[7].stride(0),
            inputs[8],
            inputs[8].stride(0),
            1031,
        )
        self.assertEqual(len(args), len(expected))
        for actual, original in zip(args, expected):
            if isinstance(original, torch.Tensor):
                self.assertIs(actual, original)
            else:
                self.assertEqual(actual, original)

    def test_non_tile_aligned_vocabulary_tail_is_included(self):
        for vocab_size in (1, 31, 32, 1024, 1025, 151939):
            with self.subTest(vocab_size=vocab_size):
                self.adapter.apply_ascend_v2_penalties(*self.inputs(vocab_size))
                grid, _, kwargs = self.kernel.launches[-1]
                self.assertEqual(grid[1], (vocab_size + 1023) // 1024)
                self.assertEqual(kwargs["BLOCK_SIZE"] % 32, 0)

    def test_empty_logits_do_not_launch_an_invalid_zero_grid(self):
        self.adapter.apply_ascend_v2_penalties(*self.inputs(num_tokens=0))
        self.adapter.apply_ascend_v2_penalties(*self.inputs(vocab_size=0))
        self.assertEqual(self.kernel.launches, [])

    def test_npu_install_is_idempotent_and_only_replaces_the_launcher(self):
        self.assertTrue(self.adapter.install_ascend_v2_sampling_shims())
        self.assertTrue(self.adapter.install_ascend_v2_sampling_shims())
        self.assertIs(
            self.penalties.apply_penalties, self.adapter.apply_ascend_v2_penalties
        )
        self.assertIs(self.penalties._penalties_kernel, self.kernel)

    def test_other_platform_install_keeps_upstream_launcher(self):
        self.platform.current_platform.device_type = "cuda"
        self.assertFalse(self.adapter.install_ascend_v2_sampling_shims())
        self.assertIs(self.penalties.apply_penalties, self.upstream_launcher)


def _penalty_fixture(vocab_size):
    rows = 9
    generator = torch.Generator().manual_seed(873)
    logits_storage = torch.full((rows, vocab_size + 7), 1234.0)
    logits = logits_storage[:, :vocab_size]
    logits.copy_(torch.randn(rows, vocab_size, generator=generator) * 2)
    logits[:, 31] = 3
    logits[:, 32] = 2
    logits[:, 1023] = 2
    logits[:, 1024] = 2
    logits[0, 7] = -2
    logits[2, 7] = -2
    logits[0, -1] = -4
    logits[-1, 0] = float("-inf")

    mapping = torch.tensor([2, 2, 2, 2, 0, 0, 0, 1, 3], dtype=torch.int32)
    local_pos = torch.tensor([0, 1, 2, 3, 0, 1, 2, 0, 0], dtype=torch.int32)
    token_ids = torch.tensor(
        [31, 31, 7, 31, 32, 31, vocab_size - 1, 33, 1024], dtype=torch.int32
    )
    repetition = torch.tensor([1.2, 0.7, 1.7, 1.0], dtype=torch.float32)
    frequency = torch.tensor([0.4, -0.25, 0.5, 0.0], dtype=torch.float32)
    presence = torch.tensor([-0.3, 0.6, 0.2, 0.0], dtype=torch.float32)
    prompts = (
        {31, 32, 1024, vocab_size - 1},
        {0, 31, 33, vocab_size - 2},
        {7, 31, 1023, 1024, vocab_size - 1},
        {31, 1024},
    )
    words = (vocab_size + 31) // 32
    packed_storage = torch.full((4, words + 3), 123, dtype=torch.int32)
    packed = packed_storage[:, :words]
    packed.zero_()
    for req, tokens in enumerate(prompts):
        bits = {}
        for token in tokens:
            word, bit = divmod(token, 32)
            bits[word] = bits.get(word, 0) | (1 << bit)
        for word, value in bits.items():
            packed[req, word] = value if value < (1 << 31) else value - (1 << 32)

    counts_storage = torch.full((4, vocab_size + 11), 71, dtype=torch.int32)
    counts = counts_storage[:, :vocab_size]
    counts.zero_()
    for req, tokens in enumerate(
        (
            {31: 2, 5: 1, vocab_size - 1: 3},
            {33: 4, vocab_size - 2: 1},
            {7: 2, 31: 3, vocab_size - 1: 1},
            {31: 5},
        )
    ):
        for token, count in tokens.items():
            counts[req, token] = count
    return {
        "logits_storage": logits_storage,
        "logits": logits,
        "mapping": mapping,
        "local_pos": local_pos,
        "tokens": token_ids,
        "repetition": repetition,
        "frequency": frequency,
        "presence": presence,
        "packed_storage": packed_storage,
        "packed": packed,
        "prompts": prompts,
        "counts_storage": counts_storage,
        "counts": counts,
    }


def _penalty_cpu_reference(case):
    result = case["logits"].clone()
    for row, request in enumerate(case["mapping"].tolist()):
        counts = case["counts"][request].clone()
        position = int(case["local_pos"][row])
        start = row - position
        for previous in range(1, position + 1):
            counts[int(case["tokens"][start + previous])] += 1
        output_seen = counts > 0
        seen = output_seen.clone()
        for token in case["prompts"][request]:
            seen[token] = True
        scale = torch.where(seen, case["repetition"][request], 1.0)
        result[row] *= torch.where(result[row] > 0, scale.reciprocal(), scale)
        result[row] -= case["frequency"][request] * counts
        result[row] -= case["presence"][request] * output_seen
    return result


def _verify_penalty_oracle(case, expected):
    # These values exercise independent prompt-only and repeated-draft cases.
    rep = float(case["repetition"][2])
    freq = float(case["frequency"][2])
    pres = float(case["presence"][2])
    checks = (
        (0, 31, 3 / rep - freq * 3 - pres),
        (1, 31, 3 / rep - freq * 4 - pres),
        (3, 31, 3 / rep - freq * 5 - pres),
        (0, 7, -2 * rep - freq * 2 - pres),
        (2, 7, -2 * rep - freq * 3 - pres),
        (0, 1023, 2 / rep),
        (0, 1024, 2 / rep),
        (0, 32, 2),
    )
    for row, token, value in checks:
        assert abs(float(expected[row, token]) - value) < 1e-6
    torch.testing.assert_close(expected[-1], case["logits"][-1], rtol=0, atol=0)
    assert int(case["packed"][2, 0]) < 0, "Packed bit 31 was not exercised"


@pytest.mark.gpu
class AscendV2PenaltiesNumericalTests(unittest.TestCase):
    def test_upstream_kernel_matches_cpu_penalty_and_draft_reference(self):
        if importlib.util.find_spec("torch_npu") is None:
            self.skipTest("torch_npu is unavailable")
        import torch_npu  # noqa: F401

        if not torch.npu.is_available():
            self.skipTest("An Ascend NPU is unavailable")
        device = torch.device("npu", torch.npu.current_device())
        source = (
            Path(__file__).resolve().parents[2]
            / "vllm_fl/patches/ascend_v2_sampling.py"
        )
        spec = importlib.util.spec_from_file_location(
            "ascend_v2_sampling_numerical", source
        )
        adapter = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(adapter)
        for vocab_size in (1031, 2057, 151939):
            with self.subTest(vocab_size=vocab_size):
                case = _penalty_fixture(vocab_size)
                expected = _penalty_cpu_reference(case)
                _verify_penalty_oracle(case, expected)
                logits_storage = case["logits_storage"].to(device)
                logits = logits_storage[:, :vocab_size]
                packed_storage = case["packed_storage"].to(device)
                packed = packed_storage[:, : case["packed"].shape[1]]
                counts_storage = case["counts_storage"].to(device)
                counts = counts_storage[:, :vocab_size]
                adapter.apply_ascend_v2_penalties(
                    logits,
                    case["mapping"].to(device),
                    case["tokens"].to(device),
                    case["local_pos"].to(device),
                    case["repetition"].to(device),
                    case["frequency"].to(device),
                    case["presence"].to(device),
                    packed,
                    counts,
                )
                torch.npu.synchronize()
                actual = logits.cpu()
                torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-6)
                torch.testing.assert_close(
                    logits_storage[:, vocab_size:].cpu(),
                    case["logits_storage"][:, vocab_size:],
                    rtol=0,
                    atol=0,
                )
                torch.testing.assert_close(
                    packed_storage.cpu(),
                    case["packed_storage"],
                    rtol=0,
                    atol=0,
                )
                torch.testing.assert_close(
                    counts_storage.cpu(),
                    case["counts_storage"],
                    rtol=0,
                    atol=0,
                )
                torch.testing.assert_close(
                    actual[-1],
                    case["logits"][-1],
                    rtol=0,
                    atol=0,
                )


if __name__ == "__main__":
    unittest.main()
