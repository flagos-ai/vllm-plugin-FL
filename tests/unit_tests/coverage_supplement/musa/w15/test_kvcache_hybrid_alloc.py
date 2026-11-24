# SPDX-License-Identifier: Apache-2.0
"""CPU-only coverage for ``allocate_ascend_kv_cache`` and block plane ops.

Real PyTorch tensors back every pool; ``allocate_kv_cache`` (the upstream
allocator) is replaced by a stub that hands back pre-built manager views.
"""

import os
import sys
from types import SimpleNamespace

import pytest
import torch

import vllm_fl.kv_cache.ascend.hybrid as hybrid

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from test_kvcache_hybrid_config import (
    FakeAttentionSpec,
    FakeMambaSpec,
    FakeUniformSpecs,
    UnsupportedSpec,
)


@pytest.fixture(autouse=True)
def fake_specs(monkeypatch):
    monkeypatch.setattr(hybrid, "AttentionSpec", FakeAttentionSpec)
    monkeypatch.setattr(hybrid, "MambaSpec", FakeMambaSpec)
    monkeypatch.setattr(hybrid, "UniformTypeKVCacheSpecs", FakeUniformSpecs)


@pytest.fixture
def allocate(monkeypatch):
    def stub(kv_cache_config, device, layout, kernel_block_sizes=None):
        return stub.result

    stub.result = {}
    monkeypatch.setitem(
        sys.modules,
        "vllm.v1.worker.utils",
        SimpleNamespace(allocate_kv_cache=stub),
    )
    return stub


def tensor_cfg(
    size, names, layer_stride, block_stride, offset=0, host_resident=False
):
    return SimpleNamespace(
        size=size,
        layers=list(names),
        layer_stride=layer_stride,
        block_stride=block_stride,
        offset=offset,
        host_resident=host_resident,
    )


def group_cfg(spec, names):
    return SimpleNamespace(kv_cache_spec=spec, layer_names=list(names))


def cache_cfg(num_blocks, tensors, groups):
    return SimpleNamespace(
        num_blocks=num_blocks,
        kv_cache_tensors=tensors,
        kv_cache_groups=groups,
        num_blocks_of=lambda _tensor: num_blocks,
    )


def attn_group(spec, group_id, kernel_size, stride=None):
    return SimpleNamespace(
        kv_cache_spec=spec,
        kv_cache_group_id=group_id,
        kernel_block_size=kernel_size,
        kernel_block_stride=stride,
    )


def layout(name):
    return SimpleNamespace(name=name)


PAGE = 76
CONV = 12
STATE = 32


def dense_config(num_blocks=4, attention_only=False, page=PAGE):
    attention = FakeAttentionSpec(
        dtype=torch.float16,
        block_size=4,
        num_kv_heads=2,
        head_size=2,
        page_size_padded=page,
    )
    mamba = FakeMambaSpec(
        shapes=((3,), (4, 2)),
        dtypes=(torch.float32, torch.float32),
        page_size_padded=page,
    )
    pool = num_blocks * page
    groups = [group_cfg(attention, ["a0", "a1"])]
    tensors = [tensor_cfg(2 * pool, ["a0", "a1"], pool, page)]
    if not attention_only:
        groups.append(group_cfg(mamba, ["m0", "m1"]))
        tensors.append(tensor_cfg(2 * pool, ["m0", "m1"], pool, page))
    return cache_cfg(num_blocks, tensors, groups), attention, mamba


def mamba_only_config(
    page=PAGE,
    shapes=((3,), (4, 2)),
    dtypes=(torch.float32, torch.float32),
    num_blocks=4,
    layers=("m0",),
):
    spec = FakeMambaSpec(shapes=shapes, dtypes=dtypes, page_size_padded=page)
    pool = num_blocks * page
    tensor = tensor_cfg(len(layers) * pool, layers, pool, page)
    return cache_cfg(num_blocks, [tensor], [group_cfg(spec, layers)]), spec


# ---------------------------------------------------------------------------
# early exits
# ---------------------------------------------------------------------------
def test_empty_pools_return_empty_views():
    config = cache_cfg(0, [], [])
    assert hybrid.allocate_ascend_kv_cache(
        config, "cpu", layout("LHBNC"), [], [], False
    ) == ({}, {}, [])


def test_host_resident_pools_are_rejected():
    config = cache_cfg(1, [tensor_cfg(8, ["a"], 8, 8, host_resident=True)], [])
    with pytest.raises(NotImplementedError, match="host-resident"):
        hybrid.allocate_ascend_kv_cache(config, "cpu", layout("LHBNC"), [], [], False)


# ---------------------------------------------------------------------------
# planar (non-dense) allocation
# ---------------------------------------------------------------------------
def _planar_manager(blocks=4, block=4, heads=2, head=2):
    planar = torch.arange(
        2 * blocks * block * heads * head, dtype=torch.float32
    ).view(2, blocks, block, heads * head)
    return planar, planar.permute(1, 0, 2, 3)


def test_planar_attention_builds_kernel_view_and_planes(allocate):
    spec = FakeAttentionSpec(
        dtype=torch.float32, block_size=4, num_kv_heads=2, head_size=2
    )
    planar, manager = _planar_manager()
    allocate.result = {"a0": manager}
    config = cache_cfg(
        4,
        [tensor_cfg(planar.nbytes, ["a0"], planar.nbytes, 128)],
        [group_cfg(spec, ["a0"])],
    )
    groups = [
        attn_group(spec, 0, 2),
        attn_group(spec, 0, None, stride=99),
    ]

    bound, managers, planes = hybrid.allocate_ascend_kv_cache(
        config, "cpu", layout("LHBNC"), [2], groups, False
    )

    assert list(managers) == ["a0"]
    assert managers["a0"] is manager
    assert tuple(bound["a0"].shape) == (2, 8, 2, 2, 2)
    assert bound["a0"].is_contiguous()
    assert bound["a0"].data_ptr() == planar.data_ptr()
    assert len(planes) == 2
    assert groups[0].kernel_block_stride == 2
    assert groups[1].kernel_block_stride == 99


def test_planar_mamba_binds_manager_directly(allocate):
    spec = FakeMambaSpec(page_size_padded=PAGE)
    pages = torch.zeros(4, 1, 1, PAGE, dtype=torch.uint8)
    allocate.result = {"m0": pages}
    config = cache_cfg(
        4,
        [tensor_cfg(pages.nbytes, ["m0"], pages.nbytes, PAGE)],
        [group_cfg(spec, ["m0"])],
    )

    bound, managers, planes = hybrid.allocate_ascend_kv_cache(
        config, "cpu", layout("LHBNC"), [4], [], False
    )

    assert bound["m0"] is pages
    assert managers["m0"] is pages
    assert len(planes) == 1 and planes[0].data_ptr() == pages.data_ptr()


def test_planar_skips_layout_name_check_without_attention(allocate):
    spec = FakeMambaSpec(page_size_padded=PAGE)
    pages = torch.zeros(4, 1, 1, PAGE, dtype=torch.uint8)
    allocate.result = {"m0": pages}
    config = cache_cfg(
        4,
        [tensor_cfg(pages.nbytes, ["m0"], pages.nbytes, PAGE)],
        [group_cfg(spec, ["m0"])],
    )

    bound, _, _ = hybrid.allocate_ascend_kv_cache(
        config, "cpu", layout("anything"), [4], [], False
    )
    assert bound["m0"] is pages


def test_planar_rejects_unknown_spec(allocate):
    spec = UnsupportedSpec()
    allocate.result = {"x0": torch.zeros(4, 8, dtype=torch.uint8)}
    config = cache_cfg(
        4, [tensor_cfg(32, ["x0"], 32, 8)], [group_cfg(spec, ["x0"])]
    )

    with pytest.raises(NotImplementedError, match="Unsupported Ascend KV spec"):
        hybrid.allocate_ascend_kv_cache(
            config, "cpu", layout("LHBNC"), [4], [], False
        )


def test_planar_rejects_mixed_attention_and_mamba():
    config, attention, mamba = dense_config()
    with pytest.raises(NotImplementedError, match="dense GDN"):
        hybrid.allocate_ascend_kv_cache(
            config, "cpu", layout("LHBNC"), [2, 4], [], False
        )


def test_mismatched_layout_names_are_rejected():
    config, attention, _ = dense_config(attention_only=True)
    with pytest.raises(NotImplementedError, match="planar"):
        hybrid.allocate_ascend_kv_cache(
            config, "cpu", layout("LBNHC"), [2], [], False
        )
    with pytest.raises(NotImplementedError, match="dense hybrid"):
        hybrid.allocate_ascend_kv_cache(
            config, "cpu", layout("LHBNC"), [2], [], True
        )


# ---------------------------------------------------------------------------
# dense allocation
# ---------------------------------------------------------------------------
def test_dense_overlaid_pools_share_planes(allocate):
    config, attention, mamba = dense_config()
    groups = [attn_group(attention, 0, 2), attn_group(mamba, 1, 4)]

    bound, managers, planes = hybrid.allocate_ascend_kv_cache(
        config, "cpu", layout("LBNHC"), [2, 4], groups, True
    )

    assert set(bound) == {"a0", "a1", "m0", "m1"}
    assert set(managers) == {"a0", "a1"}
    assert [tuple(p.shape) for p in planes] == [(4, CONV), (4, STATE), (4, STATE)] * 2
    assert groups[0].kernel_block_stride == 2
    assert groups[1].kernel_block_stride is None
    assert (
        bound["a0"].untyped_storage().data_ptr()
        == bound["m0"].untyped_storage().data_ptr()
    )
    assert bound["a1"].data_ptr() - bound["a0"].data_ptr() == 4 * PAGE
    assert tuple(managers["a0"].shape) == (4, 2, 4, 4)

    conv, state = hybrid.dense_mamba_views(
        bound["m0"].view(-1), ((3,), (4, 2)), (torch.float32, torch.float32), 4
    )
    assert conv.data_ptr() == planes[0].data_ptr()
    assert state.data_ptr() == bound["a0"][0].data_ptr()


def test_dense_attention_only_stage_derives_conv_prefix(allocate):
    config, attention, _ = dense_config(attention_only=True)

    bound, managers, planes = hybrid.allocate_ascend_kv_cache(
        config,
        "cpu",
        layout("LBNHC"),
        [2],
        [attn_group(attention, 0, 2)],
        True,
    )

    assert set(bound) == set(managers) == {"a0", "a1"}
    assert [p.shape[1] for p in planes] == [CONV, STATE, STATE] * 2


def test_dense_requires_single_backing_size(allocate):
    config, attention, mamba = dense_config()
    config.kv_cache_tensors[1].size = 640

    with pytest.raises(ValueError, match="share one backing allocation"):
        hybrid.allocate_ascend_kv_cache(
            config, "cpu", layout("LBNHC"), [2, 4], [], True
        )


@pytest.mark.parametrize("field", ["block_stride", "layer_stride"])
def test_dense_requires_layer_compact_pools(field):
    config, _, _ = dense_config()
    target = config.kv_cache_tensors[0]
    if field == "block_stride":
        target.block_stride = PAGE * 2
    else:
        target.layer_stride = 4 * PAGE - 1

    with pytest.raises(NotImplementedError, match="layer-compact"):
        hybrid.allocate_ascend_kv_cache(
            config, "cpu", layout("LBNHC"), [2, 4], [], True
        )


def test_dense_rejects_out_of_range_offsets():
    config, _, _ = dense_config()
    config.kv_cache_tensors[0].offset = -4

    with pytest.raises(ValueError, match="exceeds its allocation"):
        hybrid.allocate_ascend_kv_cache(
            config, "cpu", layout("LBNHC"), [2, 4], [], True
        )


def test_dense_attention_requires_conv_prefix():
    config, _, _ = dense_config(attention_only=True, page=64)

    with pytest.raises(ValueError, match="convolution prefix"):
        hybrid.allocate_ascend_kv_cache(
            config, "cpu", layout("LBNHC"), [2], [], True
        )


def test_dense_attention_rejects_indivisible_kernel():
    config, attention, _ = dense_config(attention_only=True)

    with pytest.raises(ValueError, match="kernel block size must divide"):
        hybrid.allocate_ascend_kv_cache(
            config, "cpu", layout("LBNHC"), [3], [attn_group(attention, 0, 3)], True
        )


@pytest.mark.parametrize(
    "shapes,dtypes,match",
    [
        (((3,),), (torch.float32,), "convolution \\+ GDN state"),
        (((4,), (4,)), (torch.float32, torch.float32), "convolution \\+ GDN state"),
    ],
)
def test_dense_mamba_rejects_bad_state_shapes(shapes, dtypes, match):
    config, _ = mamba_only_config(shapes=shapes, dtypes=dtypes)

    with pytest.raises(NotImplementedError, match=match):
        hybrid.allocate_ascend_kv_cache(
            config, "cpu", layout("LBNHC"), [4], [], True
        )


def test_dense_mamba_rejects_padding_mismatch():
    config, _ = mamba_only_config(page=80)

    with pytest.raises(ValueError, match="padding must match"):
        hybrid.allocate_ascend_kv_cache(
            config, "cpu", layout("LBNHC"), [4], [], True
        )


def test_dense_rejects_unknown_spec():
    spec = UnsupportedSpec(page_size_bytes=8)
    pages = torch.zeros(4, 8, dtype=torch.uint8)
    config = cache_cfg(
        4,
        [tensor_cfg(pages.nbytes, ["x0"], pages.nbytes, 8)],
        [group_cfg(spec, ["x0"])],
    )

    with pytest.raises(NotImplementedError, match="Unsupported Ascend KV spec"):
        hybrid.allocate_ascend_kv_cache(
            config, "cpu", layout("LBNHC"), [4], [], True
        )


def test_dense_rejects_overlaid_pools_with_different_byte_ownership():
    first = FakeMambaSpec(
        shapes=((3,), (4, 2)),
        dtypes=(torch.float32, torch.float32),
        page_size_padded=PAGE,
    )
    second = FakeMambaSpec(
        shapes=((5,), (7,)),
        dtypes=(torch.float32, torch.float32),
        page_size_padded=PAGE,
    )
    pool = 4 * PAGE
    config = cache_cfg(
        4,
        [
            tensor_cfg(pool, ["m0"], pool, PAGE),
            tensor_cfg(pool, ["m1"], pool, PAGE),
        ],
        [group_cfg(first, ["m0"]), group_cfg(second, ["m1"])],
    )

    with pytest.raises(ValueError, match="Overlaid Ascend dense pools"):
        hybrid.allocate_ascend_kv_cache(
            config, "cpu", layout("LBNHC"), [4], [], True
        )


def test_dense_mamba_only_binds_raw_pages(allocate):
    config, spec = mamba_only_config()

    bound, managers, planes = hybrid.allocate_ascend_kv_cache(
        config, "cpu", layout("LBNHC"), [4], [], True
    )

    assert managers == {}
    assert tuple(bound["m0"].shape) == (4, 1, 1, PAGE)
    assert [p.shape[1] for p in planes] == [CONV, STATE, STATE]


# ---------------------------------------------------------------------------
# copy_block_planes
# ---------------------------------------------------------------------------
def test_copy_block_planes_noops_on_empty_inputs():
    hybrid.copy_block_planes([], [(0, 1)])
    hybrid.copy_block_planes([torch.zeros(4, 2)], [])
    hybrid.copy_block_planes(None, None)


def test_copy_block_planes_snapshots_aliases_before_writing():
    raw = torch.arange(12, dtype=torch.int32).view(4, 3)
    before = raw.clone()

    hybrid.copy_block_planes([raw[:, :2], raw[:, 1:]], [(0, 1), (1, 2)])

    torch.testing.assert_close(raw[1], before[0])
    torch.testing.assert_close(raw[2], before[1])
    torch.testing.assert_close(raw[0], before[0])
    torch.testing.assert_close(raw[3], before[3])


def test_copy_block_planes_aligned_uses_index_copy(monkeypatch):
    raw = torch.arange(16, dtype=torch.float32).view(4, 4)
    before = raw.clone()
    calls = []
    original = torch.Tensor.index_copy_

    def tracked(destination, dim, indices, values):
        calls.append((destination.dtype, values.dtype))
        return original(destination, dim, indices, values)

    monkeypatch.setattr(torch.Tensor, "index_copy_", tracked)
    hybrid.copy_block_planes([raw], [(0, 2), (1, 3)])

    assert calls == [(torch.int32, torch.int32)]
    torch.testing.assert_close(raw[2], before[0])
    torch.testing.assert_close(raw[3], before[1])
    torch.testing.assert_close(raw[0], before[0])


def test_copy_block_planes_unaligned_copies_rows_only(monkeypatch):
    raw = torch.arange(20, dtype=torch.uint8).view(4, 5)
    before = raw.clone()
    row_sizes = []
    original = torch.Tensor.copy_

    def tracked(destination, values, *args, **kwargs):
        row_sizes.append(destination.numel())
        return original(destination, values, *args, **kwargs)

    monkeypatch.setattr(torch.Tensor, "copy_", tracked)
    hybrid.copy_block_planes([raw], [(0, 1), (1, 2)])

    assert row_sizes == [5, 5]
    torch.testing.assert_close(raw[1], before[0])
    torch.testing.assert_close(raw[2], before[1])
    torch.testing.assert_close(raw[3], before[3])


# ---------------------------------------------------------------------------
# zero_block_planes
# ---------------------------------------------------------------------------
def test_zero_block_planes_noops_on_empty_inputs():
    hybrid.zero_block_planes([], [0])
    hybrid.zero_block_planes([torch.zeros(4, 2)], [])
    hybrid.zero_block_planes(None, None)


def test_zero_block_planes_aligned_uses_index_fill():
    raw = torch.ones(4, 4, dtype=torch.int32)
    extra = torch.ones(4, 4, dtype=torch.int32)
    pointer = raw.data_ptr()

    hybrid.zero_block_planes([raw, extra], [1, 3])

    assert raw.data_ptr() == pointer
    assert torch.count_nonzero(raw[[1, 3]]).item() == 0
    assert torch.all(raw[[0, 2]] == 1).item()
    assert torch.count_nonzero(extra[[1, 3]]).item() == 0
    assert torch.all(extra[[0, 2]] == 1).item()


def test_zero_block_planes_unaligned_uses_row_copy():
    raw = torch.full((4, 5), 7, dtype=torch.uint8)
    pointer = raw.data_ptr()

    hybrid.zero_block_planes([raw], [1, 3])

    assert raw.data_ptr() == pointer
    assert torch.count_nonzero(raw[[1, 3]]).item() == 0
    assert torch.all(raw[[0, 2]] == 7).item()
