# SPDX-License-Identifier: Apache-2.0
"""CPU byte-address oracles for Ascend's native vLLM 0.32 layer binding.

The view tests also run without vLLM installed; the final allocator contract
uses real public specs/AttentionGroup/allocation when that dependency exists.
No NPU operations, native accelerator libraries, or model construction are used.
"""

import ast
import importlib.util
from dataclasses import dataclass, replace
from enum import IntEnum
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import pytest
import torch

SOURCE = Path(__file__).resolve().parents[2] / "vllm_fl/kv_cache/ascend/native_032.py"
HAS_VLLM = importlib.util.find_spec("vllm") is not None

if HAS_VLLM:
    from vllm.v1.kv_cache_interface import (
        AttentionSpec,
        FullAttentionSpec,
        KVQuantMode,
        MambaSpec,
    )
else:
    # Public spec fields needed by the real view code, isolated from sys.modules.
    class KVQuantMode(IntEnum):
        NONE = 0
        FP8_PER_TENSOR = 1

    @dataclass(frozen=True)
    class AttentionSpec:
        block_size: int
        num_kv_heads: int
        head_size: int
        dtype: torch.dtype
        page_size_padded: int | None = None
        head_size_v: int | None = None
        tokens_per_state: int = 1
        num_head_slots: int | None = None
        state_content_bytes: int | None = None
        kv_quant_mode: KVQuantMode = KVQuantMode.NONE

        def __post_init__(self):
            if self.head_size_v is None:
                object.__setattr__(self, "head_size_v", self.head_size)

        @property
        def num_heads(self):
            return self.num_head_slots or self.num_kv_heads

    FullAttentionSpec = AttentionSpec

    @dataclass(frozen=True)
    class MambaSpec:
        block_size: int
        shapes: tuple
        dtypes: tuple
        page_size_padded: int | None = None


@pytest.fixture(scope="module")
def adapt():
    # Execute the complete real module except its vLLM spec import. This avoids
    # importing plugin/device initialization just to exercise CPU tensor views;
    # installed vLLM supplies the real public classes above when available.
    tree = ast.parse(SOURCE.read_text(encoding="utf-8"))
    tree.body = [
        node
        for node in tree.body
        if not (
            isinstance(node, ast.ImportFrom)
            and node.module == "vllm.v1.kv_cache_interface"
        )
    ]
    namespace = dict(
        AttentionSpec=AttentionSpec, MambaSpec=MambaSpec, KVQuantMode=KVQuantMode
    )
    exec(compile(tree, str(SOURCE), "exec"), namespace)
    return namespace["adapt_layer_kv_caches"]


def _group(spec, *names):
    return SimpleNamespace(kv_cache_spec=spec, layer_names=list(names))


def _descriptor(cache):
    return (
        cache.data_ptr(),
        cache.storage_offset(),
        cache.shape,
        cache.stride(),
        cache.untyped_storage()._cdata,
    )


def _write_oracle(raw, offset, value, dtype):
    encoded = torch.tensor([value], dtype=dtype).view(torch.int8)
    raw[offset : offset + encoded.numel()] = encoded


def _ascend_backend_contract():
    """Read the real size declaration without importing torch_npu consumers."""
    path = SOURCE.parents[2] / "dispatch/backends/vendor/ascend/impl/attention.py"
    tree = ast.parse(path.read_text(encoding="utf-8"))
    backend = next(
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == "AscendAttentionBackend"
    )
    method = next(
        node
        for node in backend.body
        if isinstance(node, ast.FunctionDef)
        and node.name == "get_supported_kernel_block_sizes"
    )
    cls = ast.ClassDef(
        name="BackendContract", bases=[], keywords=[], body=[method], decorator_list=[]
    )
    namespace = {}
    exec(
        compile(
            ast.fix_missing_locations(ast.Module(body=[cls], type_ignores=[])),
            str(path),
            "exec",
        ),
        namespace,
    )
    backend = namespace["BackendContract"]
    backend.get_name = staticmethod(lambda: "ASCEND")
    # The real backend inherits the upstream identity customize_spec method.
    backend.customize_spec = staticmethod(lambda spec: spec)
    return backend


def test_032_platform_inherits_native_hybrid_alignment_and_keeps_kernel_128():
    path = SOURCE.parents[2] / "platform.py"
    tree = ast.parse(path.read_text(encoding="utf-8"))
    platform = next(
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == "PlatformFL"
    )
    assert not any(
        isinstance(node, ast.FunctionDef) and node.name == "_align_hybrid_block_size"
        for node in platform.body
    )
    backend = _ascend_backend_contract()
    assert backend.get_supported_kernel_block_sizes() == [128]
    assert backend.get_supported_kernel_block_sizes(
        FullAttentionSpec(
            block_size=1024, num_kv_heads=2, head_size=4, dtype=torch.float16
        )
    ) == [128]


def test_runner_keeps_native_attention_spec_dense_despite_mamba_page_padding():
    path = SOURCE.parents[2] / "worker/model_runner.py"
    tree = ast.parse(path.read_text(encoding="utf-8"))
    runner = next(
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == "ModelRunnerFL"
    )
    method = next(
        node
        for node in runner.body
        if isinstance(node, ast.FunctionDef) and node.name == "get_kv_cache_spec"
    )
    method.decorator_list = []
    spec = FullAttentionSpec(
        block_size=256, num_kv_heads=2, head_size=4, dtype=torch.float16
    )
    config = SimpleNamespace()
    scope = dict(
        Any=Any,
        cast=cast,
        KVCacheSpec=object,
        AttentionSpec=AttentionSpec,
        AttentionLayerBase=object,
        Attention=type("OrdinaryAttention", (), {}),
        has_ec_transfer=lambda: False,
        get_layers_from_vllm_config=lambda *_: {
            "attention": SimpleNamespace(get_kv_cache_spec=lambda _: spec)
        },
        customize_attention_spec=lambda backend, candidate: candidate,
    )
    scope["get_layers_from_vllm_config"] = lambda *_: {
        "attention": SimpleNamespace(
            get_kv_cache_spec=lambda _: spec, get_attn_backend=lambda: object
        )
    }
    module = ast.Module(
        body=[
            ast.ImportFrom(
                module="__future__", names=[ast.alias(name="annotations")], level=0
            ),
            method,
        ],
        type_ignores=[],
    )
    exec(compile(ast.fix_missing_locations(module), str(path), "exec"), scope)
    instance = SimpleNamespace(
        vllm_config=config,
        cache_config=SimpleNamespace(mamba_page_size_padded=32768),
        # The removed legacy flag must not re-pad an otherwise dense spec.
        use_ascend_hybrid_cache=True,
    )
    assert scope["get_kv_cache_spec"](instance)["attention"] is spec
    assert spec.page_size_padded is None


@pytest.mark.parametrize("layout", ["LBNHC", "LBHNC", "BLNHC"])
@pytest.mark.parametrize("dtype", [torch.float16, torch.float32])
def test_attention_preserves_layer_offsets_padding_strides_and_shared_alias(
    adapt, layout, dtype
):
    blocks, heads, tokens, dim = 3, 2, 4, 3
    itemsize = dtype.itemsize
    content = 2 * dim * itemsize
    page = heads * tokens * content + 32
    block_stride = 2 * page if layout == "BLNHC" else page
    layer_stride = page if layout == "BLNHC" else blocks * page
    head_stride, token_stride = (
        (content, heads * content) if layout != "LBHNC" else (tokens * content, content)
    )
    guard = 64
    raw = torch.full((2 * blocks * page + 2 * guard,), 31, dtype=torch.int8)
    manager = {
        name: raw.as_strided(
            (blocks, heads, tokens, content),
            (block_stride, head_stride, token_stride, 1),
            guard + i * layer_stride,
        ).view(dtype)
        for i, name in enumerate(("untouched", "target"))
    }
    manager["shared"] = manager["target"]
    originals = dict(manager)
    descriptors = {name: _descriptor(cache) for name, cache in manager.items()}
    expected = raw.clone()
    spec = FullAttentionSpec(
        block_size=tokens,
        num_kv_heads=heads,
        head_size=dim,
        dtype=dtype,
        page_size_padded=page,
    )
    bound = adapt(manager, [_group(spec, *manager)])
    torch.testing.assert_close(raw, expected, rtol=0, atol=0)
    assert bound is not manager
    assert all(manager[name] is originals[name] for name in manager)
    assert {name: _descriptor(cache) for name, cache in manager.items()} == descriptors
    key, value = bound["target"]
    assert key.shape == value.shape == (blocks, tokens, heads, dim)
    assert (
        key.stride()
        == value.stride()
        == (
            block_stride // itemsize,
            token_stride // itemsize,
            head_stride // itemsize,
            1,
        )
    )
    assert key.data_ptr() == manager["target"].data_ptr()
    assert value.data_ptr() == key.data_ptr() + dim * itemsize
    assert all(
        view.untyped_storage()._cdata == raw.untyped_storage()._cdata
        for views in bound.values()
        for view in views
    )
    key[1, 2, 1, 2] = 3.5
    value[2, 1, 0, 0] = -4
    start = guard + layer_stride
    _write_oracle(
        expected,
        start + block_stride + 2 * token_stride + head_stride + 2 * itemsize,
        3.5,
        dtype,
    )
    _write_oracle(
        expected,
        start + 2 * block_stride + token_stride + dim * itemsize,
        -4,
        dtype,
    )
    torch.testing.assert_close(raw, expected, rtol=0, atol=0)
    assert bound["shared"][0][1, 2, 1, 2].item() == 3.5
    assert bound["shared"][1][2, 1, 0, 0].item() == -4


@pytest.mark.parametrize("blocks", [1, 3])
def test_mamba_slices_each_page_and_preserves_attention_overlay(adapt, blocks):
    shapes = ((2, 3), (2, 2, 3))
    dtypes = (torch.float16, torch.float32)
    payload, page, block_stride, guard = 60, 80, 192, 40
    raw = torch.full((guard + blocks * block_stride + 40,), 29, dtype=torch.int8)
    manager = {
        name: raw.as_strided(
            (blocks, 1, 1, payload),
            (block_stride, payload, payload, 1),
            guard + i * page,
        )
        for i, name in enumerate(("mamba0", "mamba1"))
    }
    manager["shared"] = manager["mamba1"]
    # Overlay an attention page on the SSM payload of mamba0, as native shared
    # pools can do. This must preserve the scheduler's byte-level aliasing.
    manager["attention"] = raw.as_strided(
        (blocks, 1, 2, 24), (block_stride, 48, 24, 1), guard + 12
    ).view(torch.float32)
    originals = dict(manager)
    descriptors = {name: _descriptor(cache) for name, cache in manager.items()}
    expected = raw.clone()
    mamba = MambaSpec(block_size=2, shapes=shapes, dtypes=dtypes, page_size_padded=page)
    attention = FullAttentionSpec(
        block_size=2, num_kv_heads=1, head_size=3, dtype=torch.float32
    )
    bound = adapt(
        manager,
        [_group(mamba, "mamba0", "mamba1", "shared"), _group(attention, "attention")],
    )
    torch.testing.assert_close(raw, expected, rtol=0, atol=0)
    assert all(manager[name] is originals[name] for name in manager)
    assert {name: _descriptor(cache) for name, cache in manager.items()} == descriptors
    conv, state = bound["mamba0"]
    assert conv.shape == (blocks, 2, 3)
    assert state.shape == (blocks, 2, 2, 3)
    assert conv.dtype == torch.float16 and state.dtype == torch.float32
    assert conv.stride(0) == block_stride // 2
    assert state.stride(0) == block_stride // 4
    assert conv.data_ptr() == raw.data_ptr() + guard
    assert state.data_ptr() == raw.data_ptr() + guard + 12
    b = blocks - 1
    bound["mamba1"][0][b, 1, 2] = 5
    state[b, 1, 0, 2] = 7
    _write_oracle(expected, guard + page + b * block_stride + 10, 5, torch.float16)
    _write_oracle(expected, guard + b * block_stride + 12 + 8 * 4, 7, torch.float32)
    torch.testing.assert_close(raw, expected, rtol=0, atol=0)
    assert bound["shared"][0][b, 1, 2].item() == 5
    assert bound["attention"][0][b, 1, 0, 2].item() == 7
    assert all(
        view.untyped_storage()._cdata == raw.untyped_storage()._cdata
        for views in bound.values()
        for view in views
    )


@pytest.mark.parametrize(
    "changes",
    [
        {"head_size_v": 2},
        {"tokens_per_state": 2},
        {"num_head_slots": 1},
        {"state_content_bytes": 16},
        {"kv_quant_mode": KVQuantMode.FP8_PER_TENSOR},
    ],
)
def test_attention_rejects_unsupported_storage_spec(adapt, changes):
    spec = FullAttentionSpec(
        block_size=4, num_kv_heads=2, head_size=3, dtype=torch.float16
    )
    cache = torch.empty(2, 2, 4, 6, dtype=torch.float16)
    with pytest.raises(NotImplementedError, match="ordinary, unquantized"):
        adapt({"layer": cache}, [_group(replace(spec, **changes), "layer")])


@pytest.mark.parametrize(
    "shape,dtype",
    [
        ((2, 2, 4, 5), torch.float16),
        ((2, 1, 4, 6), torch.float16),
        ((2, 2, 4, 6), torch.float32),
        ((2, 2, 4, 6, 1), torch.float16),
    ],
)
def test_attention_rejects_wrong_native_view(adapt, shape, dtype):
    spec = FullAttentionSpec(
        block_size=4, num_kv_heads=2, head_size=3, dtype=torch.float16
    )
    with pytest.raises(ValueError, match="typed \\[B, H, N"):
        adapt({"layer": torch.empty(shape, dtype=dtype)}, [_group(spec, "layer")])


@pytest.mark.parametrize(
    "case",
    [
        "dtype",
        "rank",
        "content_stride",
        "payload",
        "alignment",
        "block_stride",
        "shape_dtype_count",
        "empty_shape",
        "bad_dimension",
    ],
)
def test_mamba_rejects_incompatible_page_views(adapt, case):
    spec = MambaSpec(
        block_size=4, shapes=((2,), (2,)), dtypes=(torch.float16, torch.float32)
    )
    raw = torch.empty(128, dtype=torch.int8)
    cache = raw.as_strided((3, 1, 1, 12), (32, 12, 12, 1), 0)
    if case == "dtype":
        cache = cache.to(torch.uint8)
    elif case == "rank":
        cache = cache[:, 0]
    elif case == "content_stride":
        cache = raw.as_strided((3, 1, 1, 12), (32, 12, 12, 2), 0)
    elif case == "payload":
        cache = cache[..., :11]
    elif case == "alignment":
        cache = raw.as_strided(cache.shape, cache.stride(), 1)
    elif case == "block_stride":
        cache = raw.as_strided(cache.shape, (31, 12, 12, 1), 0)
    elif case == "shape_dtype_count":
        spec = replace(spec, dtypes=(torch.float16,))
    elif case == "empty_shape":
        spec = replace(spec, shapes=(), dtypes=())
    else:
        spec = replace(spec, shapes=((0,), (2,)))
    with pytest.raises((ValueError, TypeError)):
        adapt({"layer": cache}, [_group(spec, "layer")])


def test_groups_must_cover_each_cache_exactly_once(adapt):
    spec = FullAttentionSpec(
        block_size=4, num_kv_heads=2, head_size=3, dtype=torch.float16
    )
    cache = torch.empty(2, 2, 4, 6, dtype=torch.float16)
    group = _group(spec, "layer")
    with pytest.raises(ValueError, match="no attention group"):
        adapt({"layer": cache}, [])
    with pytest.raises(ValueError, match="multiple KV cache groups"):
        adapt({"layer": cache}, [group, group])
    with pytest.raises(NotImplementedError, match="does not support"):
        adapt({"layer": cache}, [_group(object(), "layer")])
    with pytest.raises(TypeError, match="native tensor"):
        adapt({"layer": (cache, cache)}, [group])
    assert adapt({}, [_group(spec, "encoder_without_cache")]) == {}


@pytest.mark.skipif(
    not HAS_VLLM, reason="Requires real vLLM 0.32 CPU allocator dependencies"
)
@pytest.mark.parametrize("layout_name", ["LBNHC", "LBHNC", "BLNHC"])
def test_real_032_allocator_manager_and_kernel_views_stay_aliased(adapt, layout_name):
    from vllm.v1.kv_cache_interface import (
        KVCacheConfig,
        KVCacheGroupSpec,
        KVCacheTensor,
    )
    from vllm.v1.kv_cache_layout import KVCacheLayout
    from vllm.v1.worker.utils import (
        AttentionGroup,
        allocate_kv_cache,
        map_kv_caches_to_kernel_blocks,
    )

    blocks, page, guard = 3, 160, 64
    attention = FullAttentionSpec(
        block_size=4,
        num_kv_heads=2,
        head_size=4,
        dtype=torch.float16,
        page_size_padded=page,
    )
    mamba = MambaSpec(
        block_size=4,
        shapes=((2, 4), (2, 3, 4)),
        dtypes=(torch.float16, torch.float32),
        page_size_padded=page,
    )
    layer_stride = page if layout_name == "BLNHC" else blocks * page
    block_stride = 2 * page if layout_name == "BLNHC" else page
    size = 2 * guard + 2 * blocks * page
    placements = [
        KVCacheTensor(
            size=size,
            layers=names,
            layer_stride=layer_stride,
            block_stride=block_stride,
            offset=guard,
        )
        for names in [["a0", "a1"], ["m0", "m1"]]
    ]
    specs = [
        KVCacheGroupSpec(["a0", "a1"], attention),
        KVCacheGroupSpec(["m0", "m1"], mamba),
    ]
    config = KVCacheConfig(
        num_blocks=blocks, kv_cache_tensors=placements, kv_cache_groups=specs
    )
    groups = [
        AttentionGroup(
            backend=object,
            layer_names=group.layer_names,
            kv_cache_spec=group.kv_cache_spec,
            kv_cache_group_id=i,
        )
        for i, group in enumerate(specs)
    ]
    manager = allocate_kv_cache(
        config, torch.device("cpu"), KVCacheLayout[layout_name], [4, 4]
    )
    kernel = map_kv_caches_to_kernel_blocks(manager, groups)
    descriptors = {name: _descriptor(cache) for name, cache in manager.items()}
    bound = adapt(kernel, groups)
    assert {name: _descriptor(cache) for name, cache in manager.items()} == descriptors
    raw = torch.empty(0, dtype=torch.int8).set_(
        manager["a0"].untyped_storage(), 0, (size,), (1,)
    )
    expected = raw.clone()
    bound["m1"][1][2, 1, 2, 3] = 9
    address = guard + layer_stride + 2 * block_stride + 16 + 23 * 4
    _write_oracle(expected, address, 9, torch.float32)
    torch.testing.assert_close(raw, expected, rtol=0, atol=0)
    assert all(
        view.untyped_storage()._cdata == raw.untyped_storage()._cdata
        for views in bound.values()
        for view in views
    )


@pytest.mark.skipif(
    not HAS_VLLM, reason="Requires real vLLM 0.32 alignment/allocator dependencies"
)
@pytest.mark.parametrize("layout_name", ["LBNHC", "LBHNC"])
@pytest.mark.parametrize("manager_block_size", [128, 256, 384, 1024])
def test_native_hybrid_align_prepare_allocate_and_bind_chain(
    adapt, monkeypatch, layout_name, manager_block_size
):
    from vllm.model_executor.models import ModelRegistry
    from vllm.platforms.interface import Platform
    from vllm.v1.kv_cache_interface import (
        KVCacheConfig,
        KVCacheGroupSpec,
        KVCacheTensor,
    )
    from vllm.v1.kv_cache_layout import KVCacheLayout
    from vllm.v1.worker.utils import (
        AttentionGroup,
        allocate_kv_cache,
        map_kv_caches_to_kernel_blocks,
        prepare_kernel_block_sizes,
    )

    backend = _ascend_backend_contract()
    # Native alignment counts BOTH K and V. Pick a recurrent state just below
    # the target attention page so the real upstream algorithm must round to
    # this manager block, then pad only Mamba to that dense attention page.
    page = manager_block_size * 2 * 2 * 4 * 2
    shapes = ((2, 4), (2, (page - 32) // 8))
    dtypes = (torch.float16, torch.float32)

    class ShapeOnlyModel:
        @staticmethod
        def get_mamba_state_shape_from_config(config):
            return shapes

        @staticmethod
        def get_mamba_state_dtype_from_config(config):
            return dtypes

    monkeypatch.setattr(
        ModelRegistry,
        "resolve_model_cls",
        lambda *args, **kwargs: (ShapeOnlyModel, "ShapeOnlyModel"),
    )

    class NativeAlignmentContract(Platform):
        @classmethod
        def _find_non_ssm_backends(cls, config):
            return [backend]

    config = SimpleNamespace(
        cache_config=SimpleNamespace(
            cache_dtype="auto",
            block_size=128,
            mamba_cache_mode="align",
            mamba_block_size=128,
            mamba_page_size_padded=None,
            kv_cache_dtype_skip_layers=[],
        ),
        model_config=SimpleNamespace(
            dtype=torch.float16,
            use_mla=False,
            architecture="ShapeOnlyModel",
            get_num_kv_heads=lambda _: 2,
            get_head_size=lambda: 4,
        ),
        parallel_config=SimpleNamespace(),
    )
    NativeAlignmentContract._align_hybrid_block_size(config, backend)
    assert config.cache_config.block_size == manager_block_size
    assert config.cache_config.mamba_block_size == manager_block_size
    assert config.cache_config.mamba_page_size_padded == page
    attention = FullAttentionSpec(
        block_size=manager_block_size,
        num_kv_heads=2,
        head_size=4,
        dtype=torch.float16,
    )
    mamba = MambaSpec(
        block_size=manager_block_size,
        shapes=shapes,
        dtypes=dtypes,
        page_size_padded=config.cache_config.mamba_page_size_padded,
    )
    assert attention.page_size_padded is None
    assert attention.page_size_bytes == mamba.page_size_bytes == page
    blocks, layers, guard = 3, 2, 64
    size = 2 * guard + blocks * layers * page
    placements = [
        KVCacheTensor(
            size=size,
            layers=names,
            layer_stride=blocks * page,
            block_stride=page,
            offset=guard,
        )
        for names in [["a0", "a1"], ["m0", "m1"]]
    ]
    specs = [
        KVCacheGroupSpec(["a0", "a1"], attention),
        KVCacheGroupSpec(["m0", "m1"], mamba),
    ]
    allocation = KVCacheConfig(
        num_blocks=blocks, kv_cache_tensors=placements, kv_cache_groups=specs
    )
    groups = [
        AttentionGroup(
            backend=backend,
            layer_names=group.layer_names,
            kv_cache_spec=group.kv_cache_spec,
            kv_cache_group_id=i,
        )
        for i, group in enumerate(specs)
    ]
    kernel_sizes = prepare_kernel_block_sizes(allocation, [[group] for group in groups])
    assert kernel_sizes == [128, manager_block_size]
    if manager_block_size != 128:
        groups[0].kernel_block_size = 128
    manager = allocate_kv_cache(
        allocation, torch.device("cpu"), KVCacheLayout[layout_name], kernel_sizes
    )
    descriptors = {name: _descriptor(cache) for name, cache in manager.items()}
    kernel = map_kv_caches_to_kernel_blocks(manager, groups)
    bound = adapt(kernel, groups)
    assert {name: _descriptor(cache) for name, cache in manager.items()} == descriptors
    ratio = manager_block_size // 128
    assert bound["a1"][0].shape == (blocks * ratio, 128, 2, 4)
    assert bound["m1"][1].shape == (blocks, *shapes[1])
    raw = torch.empty(0, dtype=torch.int8).set_(
        manager["a0"].untyped_storage(), 0, (size,), (1,)
    )
    expected = raw.clone()
    # Independent physical-byte oracle: touch the LAST kernel block of manager
    # block one, proving virtual subdivision preserves the manager block IDs.
    b, token, head, dim = 2 * ratio - 1, 127, 1, 3
    bound["a1"][0][b, token, head, dim] = 6
    kernel_page = 128 * 2 * 4 * 2 * 2
    if layout_name == "LBNHC":
        within = token * 2 * 8 * 2 + head * 8 * 2 + dim * 2
    else:
        within = head * 128 * 8 * 2 + token * 8 * 2 + dim * 2
    _write_oracle(
        expected, guard + blocks * page + b * kernel_page + within, 6, torch.float16
    )
    torch.testing.assert_close(raw, expected, rtol=0, atol=0)
    assert all(
        view.untyped_storage()._cdata == raw.untyped_storage()._cdata
        for views in bound.values()
        for view in views
    )
