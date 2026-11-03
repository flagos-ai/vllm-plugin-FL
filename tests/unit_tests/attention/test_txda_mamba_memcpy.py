# Copyright (c) 2026 BAAI. All rights reserved.
"""Numerical state-copy regressions for TXDA's native tensor fallback."""

import importlib.util
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[3]
PATCH = ROOT / "vllm_fl/dispatch/backends/vendor/txda/impl/patchs/mamba_memcpy.py"


@pytest.fixture
def patched(monkeypatch):
    worker = ModuleType("vllm.v1.worker")
    mamba = ModuleType("vllm.v1.worker.mamba_utils")
    worker.mamba_utils = mamba
    monkeypatch.setitem(sys.modules, worker.__name__, worker)
    monkeypatch.setitem(sys.modules, mamba.__name__, mamba)

    def collect(bufs, config, funcs, groups, src, dst, bias, request, context):
        if src == dst and bias == 0:
            return
        offset = bufs.offset
        for group in groups:
            blocks = request.block_ids[group]
            for name in config.kv_cache_groups[group].layer_names:
                for state, copy_func in zip(context[name].kv_cache, funcs):
                    spec = copy_func(state, blocks, src, bias + 1)
                    bufs.src_ptrs.np[offset] = spec.start_addr
                    bufs.dst_ptrs.np[offset] = state[blocks[dst]].data_ptr()
                    bufs.sizes.np[offset] = spec.num_elements * state.element_size()
                    offset += 1
        bufs.offset = offset

    mamba.collect_mamba_copy_meta = collect
    mamba.do_mamba_copy_block = lambda _: pytest.fail("raw-pointer kernel was used")
    mamba.MambaCopyBuffers = SimpleNamespace(create=object())
    descriptor = mamba.MambaCopyBuffers.create
    exec(
        "def preprocess_alias(*args):\n"
        "    collect_mamba_copy_meta(*args)\n"
        "    do_mamba_copy_block(args[0])\n",
        mamba.__dict__,
    )
    imported_alias = mamba.preprocess_alias
    spec = importlib.util.spec_from_file_location("txda_native_mamba_test", PATCH)
    patch = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(patch)
    return SimpleNamespace(
        mamba=mamba,
        patch=patch,
        alias=imported_alias,
        descriptor=descriptor,
        original_collect=collect,
    )


def buffers():
    def buffer(dtype):
        return SimpleNamespace(
            np=np.zeros(16, dtype=dtype),
            copy_to_gpu=lambda _: pytest.fail("native copies must not upload pointers"),
        )

    return SimpleNamespace(
        src_ptrs=buffer(np.uint64),
        dst_ptrs=buffer(np.uint64),
        sizes=buffer(np.int32),
        offset=0,
    )


def padded_state(dtype=torch.float32):
    storage = torch.arange(100, dtype=torch.float32).to(dtype)
    state = storage.as_strided((3, 2, 4), (16, 4, 1), storage_offset=7)
    return storage, state


def copy_spec(state, blocks, src, accepted):
    shift = accepted - 1
    return SimpleNamespace(
        start_addr=state[blocks[src]].data_ptr() + shift * state.element_size(),
        num_elements=8 - shift,
    )


def arguments(bufs, states, *, src=0, dst=1, bias=0, function=copy_spec):
    names = [f"layer.{i}" for i in range(len(states))]
    config = SimpleNamespace(kv_cache_groups=[SimpleNamespace(layer_names=names)])
    context = {
        name: SimpleNamespace(kv_cache=[state]) for name, state in zip(names, states)
    }
    return (
        bufs,
        config,
        (function,),
        [0],
        src,
        dst,
        bias,
        SimpleNamespace(block_ids=[[0, 1, 2]]),
        context,
    )


@pytest.mark.parametrize(
    "dtype", [torch.uint8, torch.float16, torch.bfloat16, torch.float32, torch.int64]
)
def test_padded_storage_and_nonzero_offset_copy_exactly(patched, dtype):
    backing, state = padded_state(dtype)
    before = backing.clone()
    bufs = buffers()
    patched.alias(*arguments(bufs, [state]))
    expected = before.clone()
    expected[23:31] = before[7:15]
    torch.testing.assert_close(backing, expected, rtol=0, atol=0)
    assert bufs.offset == 1
    assert bufs.src_ptrs.np[0] == state[0].data_ptr()
    assert bufs.dst_ptrs.np[0] == state[1].data_ptr()
    assert bufs.sizes.np[0] == 8 * state.element_size()
    assert bufs.src_ptrs.np.dtype == np.uint64


def test_multiple_collections_and_reset_preserve_copy_specs(patched):
    backing, state = padded_state()
    second_backing, second = padded_state()
    second_backing.add_(100)
    before, second_before = backing.clone(), second_backing.clone()
    bufs = buffers()
    calls = []

    def tracked(*args):
        calls.append(args[-1])
        return copy_spec(*args)

    patched.mamba.collect_mamba_copy_meta(
        *arguments(bufs, [state, second], function=tracked)
    )
    patched.mamba.collect_mamba_copy_meta(
        *arguments(bufs, [state], dst=2, bias=2, function=tracked)
    )
    assert calls == [1, 1, 3]
    assert bufs.offset == 3
    patched.mamba.do_mamba_copy_block(bufs)
    expected = before.clone()
    expected[23:31] = before[7:15]
    expected[39:45] = before[9:15]
    expected_second = second_before.clone()
    expected_second[23:31] = second_before[7:15]
    torch.testing.assert_close(backing, expected, rtol=0, atol=0)
    torch.testing.assert_close(second_backing, expected_second, rtol=0, atol=0)
    bufs.offset = 0
    patched.mamba.collect_mamba_copy_meta(*arguments(bufs, [state], src=0, dst=0))
    patched.mamba.do_mamba_copy_block(bufs)
    assert not getattr(bufs, patched.patch._PLAN_ATTRIBUTE)
    torch.testing.assert_close(backing, expected, rtol=0, atol=0)
    patched.alias(*arguments(bufs, [state], dst=2))
    expected[39:47] = before[7:15]
    torch.testing.assert_close(backing, expected, rtol=0, atol=0)


def test_overlapping_shift_clones_source(patched):
    backing, state = padded_state()
    expected = backing.clone()
    expected[7:14] = backing[8:15].clone()
    patched.alias(*arguments(buffers(), [state], src=0, dst=0, bias=1))
    torch.testing.assert_close(backing, expected, rtol=0, atol=0)


def test_zero_length_does_not_dereference_null_source(patched):
    backing, state = padded_state()
    before = backing.clone()
    bufs = buffers()
    patched.alias(
        *arguments(
            bufs,
            [state],
            function=lambda *args: SimpleNamespace(start_addr=0, num_elements=0),
        )
    )
    assert bufs.offset == 1
    torch.testing.assert_close(backing, before, rtol=0, atol=0)


@pytest.mark.parametrize("kind", ["unaligned", "before_storage", "after_storage"])
def test_invalid_range_rejected_before_copy_and_offset_restored(patched, kind):
    backing, state = padded_state()
    before = backing.clone()
    address = {
        "unaligned": state.data_ptr() + 1,
        "before_storage": backing.data_ptr() - state.element_size(),
        "after_storage": backing.data_ptr() + backing.untyped_storage().nbytes(),
    }[kind]
    bufs = buffers()
    with pytest.raises(ValueError, match="aligned|outside"):
        patched.alias(
            *arguments(
                bufs,
                [state],
                function=lambda *args: SimpleNamespace(
                    start_addr=address, num_elements=1
                ),
            )
        )
    assert bufs.offset == 0
    torch.testing.assert_close(backing, before, rtol=0, atol=0)


@pytest.mark.parametrize("size", [-4, 3])
def test_invalid_byte_size_rejected(patched, size):
    _, state = padded_state()
    with pytest.raises(ValueError, match="size is not element-aligned"):
        patched.patch._state_view(state, state.data_ptr(), size)


def test_unowned_metadata_fails_clearly(patched):
    bufs = buffers()
    bufs.offset = 1
    with pytest.raises(RuntimeError, match="owning tensor"):
        patched.mamba.do_mamba_copy_block(bufs)


def test_patch_idempotent_and_preserves_buffer_creation(patched):
    collect = patched.mamba.collect_mamba_copy_meta
    execute = patched.mamba.do_mamba_copy_block
    patched.patch.patch_mamba_state_copy()
    assert patched.mamba.collect_mamba_copy_meta is collect
    assert patched.mamba.do_mamba_copy_block is execute
    assert collect.__wrapped__ is patched.original_collect
    assert patched.mamba.MambaCopyBuffers.create is patched.descriptor
