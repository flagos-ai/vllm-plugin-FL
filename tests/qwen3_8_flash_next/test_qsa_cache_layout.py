"""QSA KV-cache compatibility tests for the supported vLLM ABIs."""

from __future__ import annotations

import pytest
import torch

# The reference package intentionally has model -> qsa -> model ownership.  Keep
# the model import here so the test mirrors package registration instead of
# importing qsa as a standalone leaf module.
from vllm_fl.models.qwen3_8_flash_next.common.qsa_cache import QSAStateBackend
from vllm_fl.models.qwen3_8_flash_next.gpu import model as _model  # noqa: F401
from vllm_fl.models.qwen3_8_flash_next.gpu.qsa import (
    Qwen3_8FlashNextQSAAttentionBackend,
    _unpack_qsa_kv_cache,
)


@pytest.mark.parametrize(
    "backend,expected_order,expected_layered_order,expected_block_stride",
    [
        (
            Qwen3_8FlashNextQSAAttentionBackend,
            (0, 1, 2, 3, 4),
            (0, 1, 2, 3, 4, 5),
            False,
        ),
        (QSAStateBackend, (0, 1, 2, 3), (0, 1, 2, 3, 4), True),
    ],
)
def test_qsa_backends_use_layered_identity_layout(
    backend, expected_order, expected_layered_order, expected_block_stride
):
    # The main QSA cache keeps its legacy layout.  The side cache opts into
    # vLLM's block-stride padding contract so its page can be unified with the
    # other attention layers.
    assert backend.indexes_kv_by_block_stride() is expected_block_stride
    assert backend.get_kv_cache_stride_order() == expected_order
    assert backend.get_kv_cache_stride_order(True) == expected_layered_order


def test_qsa_backend_owns_vendor_neutral_legacy_layout():
    assert Qwen3_8FlashNextQSAAttentionBackend.get_kv_cache_shape(
        3, 16, 2, 8
    ) == (3, 2, 16, 2, 8)
    assert Qwen3_8FlashNextQSAAttentionBackend.get_kv_cache_stride_order() == (
        0,
        1,
        2,
        3,
        4,
    )
    assert Qwen3_8FlashNextQSAAttentionBackend.get_kv_cache_stride_order(True) == (
        0,
        1,
        2,
        3,
        4,
        5,
    )


def test_unpack_legacy_vllm_024_cache_layout():
    cache = torch.zeros(3, 2, 16, 2, 8)
    key, value = _unpack_qsa_kv_cache(cache, 8)
    assert key.shape == value.shape == (3, 16, 2, 8)
    assert key.untyped_storage().data_ptr() == cache.untyped_storage().data_ptr()
    assert value.untyped_storage().data_ptr() == cache.untyped_storage().data_ptr()
    assert key.storage_offset() == cache.storage_offset()
    assert value.storage_offset() == cache.storage_offset() + cache[0, 0].numel()

    # The QSA cache-update Triton kernel flattens only the contiguous head/dim
    # tail.  This must remain a view and writes must reach the allocator-owned
    # backing storage so no vendor performs a hidden full-cache copy during
    # decode.
    flat_key = key.reshape(3, 16, 1, 16)
    flat_value = value.reshape(3, 16, 1, 16)
    assert (
        flat_key.untyped_storage().data_ptr() == cache.untyped_storage().data_ptr()
    )
    assert (
        flat_value.untyped_storage().data_ptr() == cache.untyped_storage().data_ptr()
    )
    assert flat_key.storage_offset() == key.storage_offset()
    assert flat_value.storage_offset() == value.storage_offset()

    key_update = torch.arange(flat_key.numel(), dtype=cache.dtype).reshape_as(flat_key)
    value_update = -torch.arange(
        flat_value.numel(), dtype=cache.dtype
    ).reshape_as(flat_value)
    flat_key.copy_(key_update)
    flat_value.copy_(value_update)
    torch.testing.assert_close(cache[:, 0], key_update.reshape_as(cache[:, 0]))
    torch.testing.assert_close(cache[:, 1], value_update.reshape_as(cache[:, 1]))


def test_unpack_rejects_packed_vllm_cache_layout():
    cache = torch.arange(3 * 2 * 16 * 16).reshape(3, 2, 16, 16)
    with pytest.raises(ValueError, match="packed 4-D"):
        _unpack_qsa_kv_cache(cache, 8)


@pytest.mark.parametrize(
    "shape,head_size",
    [((3, 3, 16, 2, 8), 8), ((3, 16, 8), 8)],
)
def test_unpack_rejects_unknown_cache_layout(shape, head_size):
    with pytest.raises(ValueError, match="QSA KV cache"):
        _unpack_qsa_kv_cache(torch.empty(shape), head_size)


def test_block_stride_flag_matches_installed_abi():
    """The opt-in flag is only published when the ABI carries the field."""

    from dataclasses import fields

    from vllm.v1.kv_cache_interface import AttentionSpec

    from vllm_fl.models.qwen3_8_flash_next.common.qsa_cache import (
        _BLOCK_STRIDE_SPEC_FLAG,
    )

    abi_has_field = any(
        field.name == "indexes_kv_by_block_stride" for field in fields(AttentionSpec)
    )
    assert _BLOCK_STRIDE_SPEC_FLAG is abi_has_field
    if abi_has_field:
        spec = AttentionSpec(
            block_size=16,
            num_kv_heads=1,
            head_size=8,
            dtype=torch.bfloat16,
            indexes_kv_by_block_stride=True,
        )
        assert spec.indexes_kv_by_block_stride is True


def test_side_cache_binds_padded_block_stride_view():
    """A padded page view keeps the logical QSA shape and padded block stride.

    vLLM 0.24's ``_reshape_attention_kv_cache`` hands a backend a strided view
    whose block stride is the padded page size.  The QSA side cache must slice
    that view without flattening it and must keep writes inside the logical
    page so the padding stays untouched.
    """

    from torch import nn

    from vllm_fl.models.qwen3_8_flash_next.common.qsa_cache import QSAKeyStateCache

    block_size, width, num_blocks, pad = 4, 8, 3, 5
    logical_page = block_size * width
    padded_page = logical_page + pad

    raw = torch.zeros(num_blocks, padded_page, dtype=torch.bfloat16)
    padded_view = torch.as_strided(
        raw,
        size=(num_blocks, block_size, 1, width),
        stride=(padded_page, width, width, 1),
    )
    assert padded_view.stride(0) == padded_page
    assert padded_view.is_contiguous() is False

    cache = object.__new__(QSAKeyStateCache)
    nn.Module.__init__(cache)
    cache.key_head_size = width
    cache.cache_rope_positions = False
    cache.rope_position_offset = width
    cache.head_size = width
    cache.dtype = torch.bfloat16
    cache.bind_kv_cache(padded_view)

    key_cache = cache.key_cache
    assert key_cache.shape == (num_blocks, block_size, 1, width)
    assert key_cache.stride(0) == padded_page
    assert key_cache.untyped_storage().data_ptr() == raw.untyped_storage().data_ptr()

    # Writes must land only in the logical page of each padded block.
    key_cache[1].fill_(1.0)
    torch.testing.assert_close(raw[1, :logical_page], torch.ones(logical_page, dtype=torch.bfloat16))
    assert raw[1, logical_page:].count_nonzero().item() == 0
    assert raw[0].count_nonzero().item() == 0
    assert raw[2].count_nonzero().item() == 0


def test_padded_slot_mapping_decomposes_to_physical_block_and_offset():
    """Flat QSA slots recover the block/offset the strided store kernel needs."""

    from vllm_fl.models.qwen3_8_flash_next.common.qsa_cache import (
        _logical_to_physical_qsa_slots,
    )

    block_size = 16
    block_table = torch.tensor([[2, 0, 5]], dtype=torch.int32)
    positions = torch.tensor([0, 15, 16, 31, 32], dtype=torch.int64)
    requests = torch.zeros(5, dtype=torch.int64)

    slots = _logical_to_physical_qsa_slots(
        block_table, requests, positions, block_size
    )
    expected_blocks = torch.tensor([2, 2, 0, 0, 5])
    expected_offsets = torch.tensor([0, 15, 0, 15, 0])
    torch.testing.assert_close(slots // block_size, expected_blocks)
    torch.testing.assert_close(slots % block_size, expected_offsets)
