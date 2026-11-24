# Copyright 2025 BAAI. All rights reserved.

"""Real-PPU kernel-launch coverage for the T-Head graph-safe cache writer.

``reshape_and_cache_flash_thead`` validates tensor metadata on the CPU but
every interesting behaviour after that lives on the device: the Triton kernel
itself, the ``num_tokens == 0`` early return (which the CPU-only contract
tests can never reach because the same-device check precedes it), and the
NHD/HND stride handling through the real Triton compiler on the PPU.

This module runs the real writer on the real PPU against the slot-scatter
oracle lifted from ``tests/unit_tests/dispatch/test_thead_cache.py``
(``_expected``), so the coverage lands on lines the CPU suite cannot execute
(both the ``num_tokens == 0`` early return and the kernel launch tail
including ``triton.cdiv`` tile computation).

``vllm.triton_utils`` resolves ``vllm.platforms.current_platform`` lazily at
import; the container's ``empty`` wheel lacks ``vllm._C_stable_libtorch``
so that resolution raises. The module fixture pre-publishes an inert thead
platform (the same approach the CPU contract suite uses) before the first
import and leaves it in place for the session.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest
import torch

pytestmark = [pytest.mark.supplementary, pytest.mark.platforms("thead")]


@pytest.fixture(scope="module")
def cache_writer():
    """Load the cache module once per session; skip without a PPU."""
    if not torch.cuda.is_available():
        pytest.skip("THead PPU accelerator required")
    # vllm.triton_utils (imported by cache.py) resolves the platform lazily;
    # the empty wheel's resolution raises, so publish an inert thead
    # platform first. On wheels where resolution works this is equivalent.
    import vllm.platforms as platforms_mod

    if getattr(platforms_mod, "_current_platform", None) is None:
        try:
            from vllm.platforms.interface import UnspecifiedPlatform
        except Exception:  # pragma: no cover - very old vLLM layout
            UnspecifiedPlatform = object

        class _InertPlatform(UnspecifiedPlatform):
            device_type = "cuda"
            vendor_name = "thead"
            dist_backend = "nccl"

        inert = _InertPlatform()
        platforms_mod.__dict__["current_platform"] = inert
        platforms_mod._current_platform = inert

    path = (
        Path(__file__).resolve().parents[4]
        / "vllm_fl/dispatch/backends/vendor/thead/impl/cache.py"
    )
    spec = importlib.util.spec_from_file_location("_thead_cache_gpu_under_test", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module.reshape_and_cache_flash_thead


gpu = pytest.mark.skipif(not torch.cuda.is_available(), reason="PPU required")

_SHAPE = (3, 4, 3, 33)  # Non-power-of-two head size exercises element masking.


def _cache(layout, dtype, fill):
    blocks, block_size, heads, dim = _SHAPE
    if layout == "HND":
        return torch.full(
            (blocks, heads, block_size, dim), fill, device="cuda", dtype=dtype
        ).permute(0, 2, 1, 3)
    return torch.full(_SHAPE, fill, device="cuda", dtype=dtype)


def _inputs(dtype, tokens=6):
    shape = (tokens, _SHAPE[2], _SHAPE[3])
    key = (
        torch.arange(tokens * shape[1] * shape[2], device="cuda") % 29
    ).reshape(shape).to(dtype)
    return key, key + 41


def _write(cache_writer, key, value, key_cache, value_cache, slots, cache_dtype="auto"):
    scale = torch.ones((), device=key.device)
    cache_writer(key, value, key_cache, value_cache, slots, cache_dtype, scale, scale)


def _expected(cache, source, slots):
    result = cache.detach().cpu().clone()
    source = source.detach().cpu()
    block_size = cache.shape[1]
    for token, slot in enumerate(slots.detach().cpu().tolist()):
        if 0 <= slot < cache.shape[0] * block_size:
            result[slot // block_size, slot % block_size] = source[token]
    return result


@gpu
@pytest.mark.gpu
@pytest.mark.parametrize(
    "dtype,cache_dtype", [(torch.float16, "fp16"), (torch.bfloat16, "bf16")]
)
@pytest.mark.parametrize("layout", ["NHD", "HND"])
@pytest.mark.parametrize("slots_list", [[0, -1, 5, -7, 11], [-1, 5, -1]])
def test_device_scatter_matches_slot_oracle(
    cache_writer, dtype, cache_dtype, layout, slots_list
):
    """The real Triton launch lands every mapped token in its logical slot."""
    key, value = _inputs(dtype)
    key_cache = _cache(layout, dtype, -13)
    # Independent value-cache stride order keeps the two launches distinct.
    value_cache = _cache("HND" if layout == "NHD" else "NHD", dtype, -17)
    slots = torch.tensor(slots_list, device="cuda", dtype=torch.int64)
    expected_key = _expected(key_cache, key, slots)
    expected_value = _expected(value_cache, value, slots)

    _write(cache_writer, key, value, key_cache, value_cache, slots, cache_dtype)
    torch.cuda.synchronize()

    torch.testing.assert_close(key_cache.cpu(), expected_key, rtol=0, atol=0)
    torch.testing.assert_close(value_cache.cpu(), expected_value, rtol=0, atol=0)


@gpu
@pytest.mark.gpu
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("layout", ["NHD", "HND"])
@pytest.mark.parametrize("slot_dtype", [torch.int32, torch.int64])
def test_zero_tokens_early_return_leaves_cache_untouched(
    cache_writer, dtype, layout, slot_dtype
):
    """``num_tokens == 0`` must return before any kernel launch.

    The same-device check precedes this branch, so a CPU tensor can never
    reach it — only a real (or emulated) device run covers the early return.
    Passing invalid NaN payloads proves the kernel never executed.
    """
    key, value = _inputs(dtype)
    key.fill_(float("nan"))
    value.fill_(float("nan"))
    key_cache = _cache(layout, dtype, 7)
    value_cache = _cache(layout, dtype, 9)
    initial_key = key_cache.cpu().clone()
    initial_value = value_cache.cpu().clone()

    slots = torch.empty((0,), device="cuda", dtype=slot_dtype)
    _write(cache_writer, key, value, key_cache, value_cache, slots)
    torch.cuda.synchronize()

    torch.testing.assert_close(key_cache.cpu(), initial_key, rtol=0, atol=0)
    torch.testing.assert_close(value_cache.cpu(), initial_value, rtol=0, atol=0)


@gpu
@pytest.mark.gpu
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_multiple_element_tiles_and_out_of_range_slots(cache_writer, dtype):
    """645 elements per token need three 256-element programs; large out-of-range
    slots must be masked rather than corrupting neighbouring blocks."""
    shape = (5, 3, 215)  # 645 elements -> tiles_per_token == 3
    blocks, block_size = 2, 8
    key = (
        torch.arange(shape[0] * shape[1] * shape[2], device="cuda") % 97
    ).reshape(shape).to(dtype)
    value = -key
    key_cache = torch.full((blocks, block_size, shape[1], shape[2]), -5, device="cuda", dtype=dtype)
    value_cache = torch.full_like(key_cache, -9)
    # Slot 31 is out of range (2 blocks * 8 tokens = 16 slots total).
    slots = torch.tensor([0, 31, 15, -1, 7], device="cuda", dtype=torch.int64)
    expected_key = _expected(key_cache, key, slots)
    expected_value = _expected(value_cache, value, slots)

    _write(cache_writer, key, value, key_cache, value_cache, slots)
    torch.cuda.synchronize()

    torch.testing.assert_close(key_cache.cpu(), expected_key, rtol=0, atol=0)
    torch.testing.assert_close(value_cache.cpu(), expected_value, rtol=0, atol=0)
