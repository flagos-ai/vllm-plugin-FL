# Copyright (c) 2026 BAAI. All rights reserved.

import os

import pytest
import torch

_test_platform = os.environ.get("FL_TEST_PLATFORM")
if _test_platform and _test_platform != "enflame":
    pytest.skip("GCU-specific unit tests only run on Enflame", allow_module_level=True)

from vllm_fl.dispatch.backends.vendor.gcu.impl import reshape_and_cache as impl


def test_slot_conversion_preserves_integer_precision_and_all_negative_sentinels():
    slots = torch.tensor(
        [-(2**63), -(2**32), -(2**31), -1, 0, 2**24 + 1, 2**31 - 1],
        dtype=torch.int64,
    )
    before = slots.clone()
    converted = impl._slots_as_int32(slots)
    assert converted.dtype == torch.int32
    assert converted.tolist() == [-1, -1, -1, -1, 0, 2**24 + 1, 2**31 - 1]
    torch.testing.assert_close(slots, before)


def _inputs(*, tokens=7, blocks=16, device="cpu"):
    source = torch.arange((tokens + 2) * 3 * 8, dtype=torch.float32, device=device)
    source = source.reshape(tokens + 2, 3, 2, 4)
    backing = torch.full((blocks * 67 + 2,), -91.0, device=device)
    cache = backing.as_strided((blocks, 4, 2, 4), (67, 8, 4, 1), 1)
    value_cache = backing.as_strided(cache.shape, cache.stride(), 33)
    return {
        "key": source[:, 0],
        "value": source[:, 2],
        "key_cache": cache,
        "value_cache": value_cache,
        "slot_mapping": torch.arange(tokens, dtype=torch.int64, device=device),
        "kv_cache_dtype": "auto",
        "k_scale": torch.tensor(1.0, device=device),
        "v_scale": torch.tensor(1.0, device=device),
    }, backing


@pytest.mark.parametrize("index_dtype", [torch.int32, torch.int64])
def test_padding_is_skipped_and_interleaved_storage_is_preserved(
    monkeypatch, index_dtype
):
    args, backing = _inputs()
    padding = -(2**32) if index_dtype == torch.int64 else -(2**31)
    original_slots = torch.tensor([0, 1, 18, -1, padding, 63, 30], dtype=index_dtype)
    args["slot_mapping"] = original_slots.clone()
    expected = backing.clone()
    for token, block, offset in (
        (0, 0, 0),
        (1, 0, 1),
        (2, 4, 2),
        (5, 15, 3),
        (6, 7, 2),
    ):
        for name, plane in (("key", 1), ("value", 33)):
            start = block * 67 + offset * 8 + plane
            expected[start : start + 8] = args[name][token].flatten()

    calls = []

    def provider(key, value, key_cache, value_cache, slots, *rest):
        calls.append(slots.clone())
        # Model the provider's int32 conversion and negative-slot contract.
        # This exercises the wrapper on CPU; native graph correctness is
        # validated separately on S60 using the real FlagGems provider.
        for token, slot in enumerate(slots.to(torch.int32).tolist()):
            if slot >= 0:
                block, offset = divmod(slot, 4)
                key_cache[block, offset].copy_(key[token])
                value_cache[block, offset].copy_(value[token])

    monkeypatch.setattr(impl, "_gems_reshape_and_cache_flash", provider)
    impl.reshape_and_cache_flash(**args)
    torch.testing.assert_close(backing, expected, rtol=0, atol=0)
    torch.testing.assert_close(args["slot_mapping"], original_slots)
    assert len(calls) == 1
    assert calls[0][4].item() == (-1 if index_dtype == torch.int64 else padding)


def test_long_prefill_splits_at_hardware_grid_limit_without_losing_tokens(monkeypatch):
    count = 65538
    args, _ = _inputs(tokens=count, blocks=(count + 3) // 4)
    calls = []

    def provider(key, value, key_cache, value_cache, slots, *rest):
        calls.append((key, value, slots))
        assert key_cache is args["key_cache"]
        assert value_cache is args["value_cache"]

    monkeypatch.setattr(impl, "_gems_reshape_and_cache_flash", provider)
    impl.reshape_and_cache_flash(**args)
    assert [part[2].numel() for part in calls] == [65535, 3]
    for index, name in enumerate(("key", "value", "slot_mapping")):
        joined = torch.cat([part[index] for part in calls])
        expected = args[name][:count]
        if name == "slot_mapping":
            assert joined.dtype == torch.int32
            expected = expected.to(torch.int32)
        torch.testing.assert_close(joined, expected, rtol=0, atol=0)


def test_empty_slots_do_not_launch(monkeypatch):
    args, _ = _inputs(tokens=0)
    monkeypatch.setattr(
        impl, "_gems_reshape_and_cache_flash", lambda *a: pytest.fail("launch")
    )
    impl.reshape_and_cache_flash(**args)


@pytest.mark.parametrize(
    "mutation, error, message",
    [
        (lambda a: a.update(kv_cache_dtype="fp8"), NotImplementedError, "quantized"),
        (lambda a: a.update(value=a["value"].half()), ValueError, "dtypes"),
        (lambda a: a.update(value=a["value"].to("meta")), ValueError, "same device"),
        (lambda a: a.update(key=a["key"].flatten()), ValueError, "rank 3"),
        (lambda a: a.update(value_cache=a["value_cache"][0]), ValueError, "rank-4"),
        (
            lambda a: a.update(slot_mapping=a["slot_mapping"].float()),
            ValueError,
            "int32",
        ),
        (
            lambda a: a.update(slot_mapping=torch.arange(14)[::2]),
            ValueError,
            "contiguous",
        ),
        (lambda a: a.update(key=a["key"][:6]), ValueError, "within tokens"),
        (lambda a: a.update(key=a["key"].transpose(1, 2)), ValueError, "within tokens"),
        (
            lambda a: a.update(value_cache=a["value_cache"].contiguous()),
            ValueError,
            "matching strides",
        ),
        (
            lambda a: a.update(
                key_cache=a["key_cache"].transpose(1, 3),
                value_cache=a["value_cache"].transpose(1, 3),
            ),
            ValueError,
            "contiguous blocks",
        ),
    ],
)
def test_unsupported_metadata_is_rejected_before_launch(
    monkeypatch, mutation, error, message
):
    args, _ = _inputs()
    mutation(args)
    monkeypatch.setattr(
        impl, "_gems_reshape_and_cache_flash", lambda *a: pytest.fail("launch")
    )
    with pytest.raises(error, match=message):
        impl.reshape_and_cache_flash(**args)


def test_cache_address_overflow_is_rejected_without_allocating_device_memory(
    monkeypatch,
):
    args, _ = _inputs(device="meta")
    huge = torch.empty_strided((16, 4, 2, 4), (2**28, 8, 4, 1), device="meta")
    args.update(key_cache=huge, value_cache=huge)
    monkeypatch.setattr(
        impl, "_gems_reshape_and_cache_flash", lambda *a: pytest.fail("launch")
    )
    with pytest.raises(ValueError, match="int32 range"):
        impl.reshape_and_cache_flash(**args)


@pytest.mark.parametrize("head_major", [True, False])
def test_standardized_cache_device_readback(head_major):
    if not hasattr(torch, "gcu") or not torch.gcu.is_available():
        pytest.skip("Requires real GCU for strided cache kernel")
    args, _ = _inputs(device="gcu")
    if head_major:
        cache = torch.full((16, 2, 4, 8), -91.0, device="gcu")
    else:
        cache = torch.full((16, 4, 2, 8), -91.0, device="gcu").transpose(1, 2)
    k, v = cache.transpose(1, 2).split(4, dim=-1)
    args.update(key_cache=k, value_cache=v)
    args["slot_mapping"] = torch.tensor(
        [0, 1, 18, -1, -(2**32), 63, 30], dtype=torch.int64, device="gcu"
    )
    expected = cache.cpu().clone()
    for token, slot in enumerate([0, 1, 18, -1, -1, 63, 30]):
        if slot >= 0:
            block, offset = divmod(slot, 4)
            expected[block, :, offset, :4] = args["key"][token].cpu()
            expected[block, :, offset, 4:] = args["value"][token].cpu()
    impl.reshape_and_cache_flash(**args)
    torch.testing.assert_close(cache.cpu(), expected, rtol=0, atol=0)
