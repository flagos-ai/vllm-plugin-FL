# SPDX-License-Identifier: Apache-2.0
"""CPU-only coverage for the Ascend hybrid cache configuration/view helpers.

The real module is imported.  Only the vLLM spec classes are replaced with
local dataclasses so the layout math can be exercised without an NPU; the
helpers under test, ``get_dtype_size`` and all tensor operations are real.
"""

import math
from dataclasses import dataclass
from types import SimpleNamespace

import pytest
import torch

import vllm_fl.kv_cache.ascend.hybrid as hybrid


# ---------------------------------------------------------------------------
# vLLM spec doubles
# ---------------------------------------------------------------------------
@dataclass
class FakeAttentionSpec:
    dtype: torch.dtype = torch.float32
    block_size: int = 4
    num_kv_heads: int = 2
    head_size: int = 2
    page_size_padded: int | None = None
    tokens_per_state: int = 1

    @property
    def head_size_v(self):
        return self.head_size

    @property
    def page_size_bytes(self):
        if self.page_size_padded is not None:
            return self.page_size_padded
        return (
            2
            * self.block_size
            * self.num_kv_heads
            * self.head_size
            * hybrid.get_dtype_size(self.dtype)
        )


@dataclass
class FakeMambaSpec:
    shapes: tuple = ((3,), (4, 2))
    dtypes: tuple = (torch.float32, torch.float32)
    page_size_padded: int | None = None

    @property
    def page_size_bytes(self):
        if self.page_size_padded is not None:
            return self.page_size_padded
        sizes = [
            math.prod(shape) * hybrid.get_dtype_size(dtype)
            for shape, dtype in zip(self.shapes, self.dtypes)
        ]
        return sizes[0] + 2 * sizes[1]


class FakeUniformSpecs:
    def __init__(self, kv_cache_specs):
        self.kv_cache_specs = kv_cache_specs


@dataclass
class UnsupportedSpec:
    page_size_bytes: int = 8


@pytest.fixture
def fake_specs(monkeypatch):
    monkeypatch.setattr(hybrid, "AttentionSpec", FakeAttentionSpec)
    monkeypatch.setattr(hybrid, "MambaSpec", FakeMambaSpec)
    monkeypatch.setattr(hybrid, "UniformTypeKVCacheSpecs", FakeUniformSpecs)
    return SimpleNamespace(
        attention=FakeAttentionSpec,
        mamba=FakeMambaSpec,
        uniform=FakeUniformSpecs,
        other=UnsupportedSpec,
    )


# ---------------------------------------------------------------------------
# uses_dense_hybrid_cache
# ---------------------------------------------------------------------------
def _model(model_type="qwen3_5_text", is_hybrid=True):
    return SimpleNamespace(
        is_hybrid=is_hybrid,
        hf_text_config=SimpleNamespace(model_type=model_type),
    )


@pytest.mark.parametrize("model_type", ["qwen3_5_text", "qwen3_5_moe_text"])
def test_uses_dense_hybrid_cache_accepts_gdn_family(model_type):
    assert hybrid.uses_dense_hybrid_cache(_model(model_type)) is True


@pytest.mark.parametrize(
    "model_type,is_hybrid",
    [("qwen2", True), ("qwen3_5_text", False), ("mamba", False)],
)
def test_uses_dense_hybrid_cache_rejects_other_models(model_type, is_hybrid):
    assert hybrid.uses_dense_hybrid_cache(_model(model_type, is_hybrid)) is False


def test_uses_dense_hybrid_cache_tolerates_missing_text_config():
    model = SimpleNamespace(is_hybrid=True, hf_text_config=SimpleNamespace())
    assert hybrid.uses_dense_hybrid_cache(model) is False


# ---------------------------------------------------------------------------
# align_hybrid_block_size
# ---------------------------------------------------------------------------
def _align_config(
    shapes=((512,), (1024,)),
    dtypes=(torch.float32, torch.float32),
    *,
    use_mla=False,
    dtype=torch.float32,
    head_size=2,
    num_kv_heads=2,
    cache_dtype="auto",
    user_block=None,
    mamba_cache_mode="align",
    max_model_len=8192,
):
    cache = SimpleNamespace(
        cache_dtype=cache_dtype,
        user_specified_block_size=user_block is not None,
        block_size=16 if user_block is None else user_block,
        mamba_cache_mode=mamba_cache_mode,
    )
    model = SimpleNamespace(
        architecture="FakeArch",
        use_mla=use_mla,
        dtype=dtype,
        get_head_size=lambda: head_size,
        get_num_kv_heads=lambda parallel_config: num_kv_heads,
        max_model_len=max_model_len,
    )
    config = SimpleNamespace(
        cache_config=cache,
        model_config=model,
        parallel_config=SimpleNamespace(),
    )
    model_cls = SimpleNamespace(
        get_mamba_state_shape_from_config=lambda _cfg: shapes,
        get_mamba_state_dtype_from_config=lambda _cfg: dtypes,
    )
    return config, model_cls


def _patch_resolution(monkeypatch, model_cls):
    monkeypatch.setattr(
        hybrid,
        "ModelRegistry",
        SimpleNamespace(
            resolve_model_cls=lambda architecture, model_config=None: (model_cls, None)
        ),
    )


def test_align_hybrid_block_size_sets_pages(monkeypatch):
    config, model_cls = _align_config()
    _patch_resolution(monkeypatch, model_cls)

    assert hybrid.align_hybrid_block_size(config) is None

    # conv=2048B, state=4096B, key token=16B -> page=256 blocks.
    assert config.cache_config.block_size == 256
    assert config.cache_config.mamba_page_size_padded == 2 * 4096 + 2048
    assert config.cache_config.mamba_block_size == 256


def test_align_hybrid_block_size_unaligned_mode_uses_model_len(monkeypatch):
    config, model_cls = _align_config(mamba_cache_mode="strict")
    _patch_resolution(monkeypatch, model_cls)

    hybrid.align_hybrid_block_size(config)

    assert config.cache_config.mamba_block_size == 8192


def test_align_hybrid_block_size_explicit_cache_dtype(monkeypatch):
    monkeypatch.setattr(
        hybrid, "STR_DTYPE_TO_TORCH_DTYPE", {"half": torch.float16}
    )
    config, model_cls = _align_config(cache_dtype="half")
    _patch_resolution(monkeypatch, model_cls)

    hybrid.align_hybrid_block_size(config)

    # key token = 2 * 2 * 2 = 8B, cdiv(4096, 1024) = 4 -> 512.
    assert config.cache_config.block_size == 512


def test_align_hybrid_block_size_allows_small_requested_block(monkeypatch):
    config, model_cls = _align_config(user_block=128)
    _patch_resolution(monkeypatch, model_cls)

    hybrid.align_hybrid_block_size(config)

    assert config.cache_config.block_size == 256


@pytest.mark.parametrize(
    "kwargs",
    [
        {"use_mla": True},
        {"shapes": ((512,),)},
        {"shapes": ((2048,), (1024,))},
    ],
)
def test_align_hybrid_block_size_rejects_unsupported_states(monkeypatch, kwargs):
    config, model_cls = _align_config(**kwargs)
    _patch_resolution(monkeypatch, model_cls)

    with pytest.raises(NotImplementedError, match="convolution \\+ GDN state"):
        hybrid.align_hybrid_block_size(config)


def test_align_hybrid_block_size_rejects_unalignable_pages(monkeypatch):
    config, model_cls = _align_config(shapes=((512,), (1025,)))
    _patch_resolution(monkeypatch, model_cls)

    with pytest.raises(ValueError, match="align Ascend key and GDN state pages"):
        hybrid.align_hybrid_block_size(config)


def test_align_hybrid_block_size_rejects_larger_user_block(monkeypatch):
    config, model_cls = _align_config(user_block=4096)
    _patch_resolution(monkeypatch, model_cls)

    with pytest.raises(ValueError, match="requires block_size="):
        hybrid.align_hybrid_block_size(config)


# ---------------------------------------------------------------------------
# pad_attention_specs
# ---------------------------------------------------------------------------
def test_pad_attention_specs_pads_only_small_attention(fake_specs):
    small = fake_specs.attention(block_size=4, page_size_padded=1024)
    large = fake_specs.attention(block_size=4, page_size_padded=16384)
    mamba = fake_specs.mamba(page_size_padded=8192)

    result = hybrid.pad_attention_specs(
        {"small": small, "large": large, "mamba": mamba}, 8192
    )

    assert result["small"].page_size_bytes == 8192
    assert result["small"] is not small
    assert small.page_size_padded == 1024
    assert result["large"] is large
    assert result["mamba"] is mamba


# ---------------------------------------------------------------------------
# dense_attention_view
# ---------------------------------------------------------------------------
def test_dense_attention_view_returns_tail_alias():
    spec = FakeAttentionSpec(dtype=torch.float16)
    raw = torch.arange(40, dtype=torch.uint8)
    shape = (2, 4, 1, 2)  # 16 elements * 2B = 32B payload

    view = hybrid.dense_attention_view(raw, spec, shape)

    assert view.dtype == torch.float16
    assert tuple(view.shape) == shape
    assert view.data_ptr() == raw.data_ptr() + 8


def test_dense_attention_view_rejects_non_byte_backing():
    spec = FakeAttentionSpec(dtype=torch.float16)
    with pytest.raises(ValueError, match="Invalid Ascend hybrid attention backing"):
        hybrid.dense_attention_view(torch.zeros(8, dtype=torch.float32), spec, (2, 2))


def test_dense_attention_view_rejects_negative_offset():
    spec = FakeAttentionSpec(dtype=torch.float16)
    with pytest.raises(ValueError, match="Invalid Ascend hybrid attention backing"):
        hybrid.dense_attention_view(torch.zeros(4, dtype=torch.uint8), spec, (2, 4, 1, 2))


def test_dense_attention_view_rejects_misaligned_offset():
    spec = FakeAttentionSpec(dtype=torch.float32)
    # payload = 1 * 4B, offset 6 is not a multiple of 4.
    with pytest.raises(ValueError, match="Invalid Ascend hybrid attention backing"):
        hybrid.dense_attention_view(torch.zeros(10, dtype=torch.uint8), spec, (1,))


# ---------------------------------------------------------------------------
# dense_mamba_views
# ---------------------------------------------------------------------------
def test_dense_mamba_views_splits_conv_then_state():
    shapes = ((2, 3), (4, 4))
    dtypes = (torch.float16, torch.float32)
    num_blocks = 3
    raw = torch.zeros(3 * (12 + 64), dtype=torch.uint8)

    conv, state = hybrid.dense_mamba_views(raw, shapes, dtypes, num_blocks)

    assert tuple(conv.shape) == (3, 2, 3) and conv.dtype == torch.float16
    assert tuple(state.shape) == (3, 4, 4) and state.dtype == torch.float32
    assert conv.data_ptr() == raw.data_ptr()
    assert state.data_ptr() == raw.data_ptr() + 3 * 12


def test_dense_mamba_views_single_state():
    raw = torch.zeros(8, dtype=torch.uint8)
    (only,) = hybrid.dense_mamba_views(raw, ((2,),), (torch.uint8,), 4)
    assert tuple(only.shape) == (4, 2)
    assert only.data_ptr() == raw.data_ptr()


def test_dense_mamba_views_rejects_non_byte_backing():
    with pytest.raises(ValueError, match="requires byte backing"):
        hybrid.dense_mamba_views(
            torch.zeros(8, dtype=torch.float32), ((2,),), (torch.float32,), 1
        )


def test_dense_mamba_views_rejects_misaligned_state():
    raw = torch.zeros(64, dtype=torch.uint8)
    with pytest.raises(ValueError, match="Invalid Ascend hybrid state backing"):
        hybrid.dense_mamba_views(raw, ((1,), (1,)), (torch.uint8, torch.float32), 1)


def test_dense_mamba_views_rejects_overflow():
    raw = torch.zeros(8, dtype=torch.uint8)
    with pytest.raises(ValueError, match="Invalid Ascend hybrid state backing"):
        hybrid.dense_mamba_views(raw, ((16,),), (torch.uint8,), 1)


# ---------------------------------------------------------------------------
# _deduplicate_block_planes
# ---------------------------------------------------------------------------
def test_deduplicate_block_planes_merges_byte_equivalent_views():
    raw = torch.arange(64, dtype=torch.uint8).view(4, 16)
    planes = hybrid._deduplicate_block_planes([raw, raw.view(torch.float32)])

    assert len(planes) == 1
    assert planes[0].dtype == torch.uint8
    assert planes[0].data_ptr() == raw.data_ptr()


def test_deduplicate_block_planes_keeps_distinct_offsets():
    raw = torch.arange(64, dtype=torch.uint8).view(4, 16)
    planes = hybrid._deduplicate_block_planes([raw[:, :8], raw[:, 8:]])

    assert len(planes) == 2
    assert planes[0].data_ptr() != planes[1].data_ptr()


# ---------------------------------------------------------------------------
# _layer_specs
# ---------------------------------------------------------------------------
@dataclass
class FakeGroup:
    kv_cache_spec: object
    layer_names: list


def test_layer_specs_flattens_uniform_specs(fake_specs):
    first = fake_specs.attention()
    second = fake_specs.mamba()
    uniform = fake_specs.uniform({"u0": first, "u1": second})
    config = SimpleNamespace(
        kv_cache_groups=[
            FakeGroup(first, ["a0", "a1"]),
            FakeGroup(second, ["m0"]),
            FakeGroup(uniform, ["u0", "u1"]),
        ]
    )

    specs, group_ids = hybrid._layer_specs(config)

    assert specs == {
        "a0": first,
        "a1": first,
        "m0": second,
        "u0": first,
        "u1": second,
    }
    assert group_ids == {"a0": 0, "a1": 0, "m0": 1, "u0": 2, "u1": 2}


# ---------------------------------------------------------------------------
# _attention_kernel_view
# ---------------------------------------------------------------------------
def _planar_manager(blocks=4, block=4, heads=2, head=2):
    planar = torch.arange(
        2 * blocks * block * heads * head, dtype=torch.float32
    ).view(2, blocks, block, heads * head)
    return planar, planar.permute(1, 0, 2, 3)


def test_attention_kernel_view_splits_manager_blocks(fake_specs):
    spec = fake_specs.attention(block_size=4, num_kv_heads=2, head_size=2)
    planar, manager = _planar_manager()

    view = hybrid._attention_kernel_view(manager, spec, 2)

    assert tuple(view.shape) == (2, 8, 2, 2, 2)
    assert view.data_ptr() == planar.data_ptr()
    assert view.is_contiguous()


def _kernel_spec(**overrides):
    fields = {
        "block_size": 4,
        "num_kv_heads": 2,
        "head_size": 2,
        "head_size_v": 2,
        "tokens_per_state": 1,
    }
    fields.update(overrides)
    return SimpleNamespace(**fields)


@pytest.mark.parametrize(
    "manager_shape,overrides",
    [
        ((4, 2, 4), {}),
        ((4, 3, 4, 4), {}),
        ((4, 2, 3, 4), {}),
        ((4, 2, 4, 5), {}),
        ((4, 2, 4, 4), {"tokens_per_state": 2}),
        ((4, 2, 4, 4), {"head_size_v": 4}),
    ],
)
def test_attention_kernel_view_rejects_unsupported_manager(
    manager_shape, overrides
):
    with pytest.raises(NotImplementedError, match="Unsupported Ascend manager"):
        hybrid._attention_kernel_view(
            torch.zeros(manager_shape), _kernel_spec(**overrides), 2
        )


def test_attention_kernel_view_rejects_indivisible_kernel(fake_specs):
    spec = fake_specs.attention(block_size=4, num_kv_heads=2, head_size=2)
    _, manager = _planar_manager()

    with pytest.raises(ValueError, match="kernel block size must divide"):
        hybrid._attention_kernel_view(manager, spec, 3)
