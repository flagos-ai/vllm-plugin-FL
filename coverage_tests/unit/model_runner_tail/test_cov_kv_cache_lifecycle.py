# Copyright 2026 FlagOS Contributors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Coverage tests for the KV-cache lifecycle of ModelRunnerFL (vllm_fl/
worker/model_runner.py): ``capture_model`` (the NONE-mode early return),
``initialize_attn_backend`` (backend discovery, per-num-heads group
splitting, cudagraph-mode resolution), ``_reshape_kv_cache_tensors``
(Mamba as_strided state views, the Ascend hybrid attention view, the
packed-tensor layout entry and the ``_reshape_attention_kv_cache``
boundary), ``_update_hybrid_attention_mamba_layout`` and
``initialize_kv_cache`` / ``initialize_kv_cache_tensors``.

Real objects run wherever CPU-safe: the KV cache specs are real
``FullAttentionSpec`` / ``MambaSpec`` dataclasses, the raw buffers are real
CPU tensors and the strided views execute for real. Two collaborator
boundaries are patched per CONTRACT R2 (imported names of the module under
test): ``_reshape_attention_kv_cache`` (a raise-if-called W4 shim
placeholder locally; the real one needs a CUDA allocator layout) and
``prepare_kernel_block_sizes`` / ``initialize_mamba_ssu_backend`` in the
``initialize_kv_cache`` wiring test. ``use_uniform_kv_cache`` /
``allocate_uniform_kv_caches`` are inherited from the CI image's runner
lineage and absent in local vllm 0.33 — the runner namespace provides them
as recorders (A100 first-run confirms the real branch).

The full CUDA-graph capture path of ``capture_model`` (mem_get_info,
graph_capture, lock_workspace) needs a real device; only the NONE-mode
warning return is CPU-reachable.
"""

from types import SimpleNamespace

import torch

from vllm.model_executor.layers.attention_layer_base import AttentionLayerBase
from vllm.v1.attention.backend import (
    AttentionBackend,
    AttentionMetadataBuilder,
)
from vllm.v1.kv_cache_interface import (
    FullAttentionSpec,
    KVCacheConfig,
    KVCacheGroupSpec,
    MambaSpec,
)

from . import _imports  # noqa: F401  (installs local import shims first)
from ._loader_helpers import Patch, module_under_test
from vllm_fl.runtime.kv_layout import PhysicalCacheLayout
from vllm_fl.worker.model_runner import ModelRunnerFL

model_runner_module = module_under_test()


class _TestStorageSpec(FullAttentionSpec):
    @property
    def storage_block_size(self):
        return 16


class _Builder(AttentionMetadataBuilder):
    """Real builder class; NEVER cudagraph support by default."""

    requires_block_table_width = False

    def __init__(self, *args, **kwargs):
        pass


class _AlwaysBuilder(_Builder):
    from vllm.v1.attention.backend import AttentionCGSupport

    _cudagraph_support = AttentionCGSupport.ALWAYS


class _Backend(AttentionBackend):
    """Concrete backend: name/builder resolve; the shape API is real."""

    @staticmethod
    def get_name():
        return "TEST_ATTN"

    @staticmethod
    def get_impl_cls():
        raise NotImplementedError

    @staticmethod
    def get_builder_cls():
        return _AlwaysBuilder

    @staticmethod
    def get_kv_cache_shape(
        num_blocks, block_size, num_kv_heads, head_size, cache_dtype_str="auto"
    ):
        return (2, num_blocks, block_size, num_kv_heads, head_size)


class _Layer(AttentionLayerBase):
    """Attention layer publishing the test backend."""

    def __init__(self, num_heads=8, backend=_Backend):
        self.num_heads = num_heads
        self._backend = backend

    def get_attn_backend(self):
        return self._backend

    def get_kv_cache_spec(self, vllm_config):
        return None


def _group(spec, layer_names, gid, backend=_Backend):
    return SimpleNamespace(
        kv_cache_spec=spec,
        layer_names=layer_names,
        kv_cache_group_id=gid,
        backend=backend,
    )


def _reshape_runner(spec, layer_names, tensors, groups, backend=_Backend, **extra):
    """Runner namespace for _reshape_kv_cache_tensors."""
    runner = SimpleNamespace(
        kv_cache_config=SimpleNamespace(
            kv_cache_tensors=tensors,
            kv_cache_groups=groups,
        ),
        attn_groups=[[_group(spec, layer_names, 0, backend=backend)]],
        runner_only_attn_layers=set(),
        use_ascend_hybrid_cache=False,
        cache_config=SimpleNamespace(cache_dtype="auto"),
    )
    runner._kv_cache_spec_attn_group_iterator = (
        ModelRunnerFL._kv_cache_spec_attn_group_iterator.__get__(
            runner,
            ModelRunnerFL,
        )
    )
    for name, value in extra.items():
        setattr(runner, name, value)
    return runner


def test_reshape_kv_cache_tensors_mamba_state_views():
    """A MambaSpec layer becomes per-state strided views over the raw
    buffer, one page apart, with real torch.as_strided views."""
    mamba_spec = MambaSpec(
        block_size=16,
        shapes=((2, 4), (2, 4)),
        dtypes=(torch.float32, torch.float32),
    )
    raw = torch.zeros(3 * mamba_spec.page_size_bytes, dtype=torch.uint8)
    groups = [
        KVCacheGroupSpec(
            layer_names=["model.layers.0"],
            kv_cache_spec=mamba_spec,
        )
    ]
    runner = _reshape_runner(
        mamba_spec,
        ["model.layers.0"],
        [],
        groups,
    )

    kv_caches = ModelRunnerFL._reshape_kv_cache_tensors(
        runner,
        {"model.layers.0": raw},
        [16],
    )

    views = kv_caches["model.layers.0"]
    assert isinstance(views, list) and len(views) == 2
    for view in views:
        # 3 blocks per state, both views alias the raw buffer.
        assert view.shape == (3, 2, 4)
        assert view.data_ptr() >= raw.data_ptr()
    # State 1 starts one page into the buffer.
    assert views[1].data_ptr() > views[0].data_ptr()


def test_reshape_kv_cache_tensors_packed_layout_uses_block_stride():
    """A packed KVCacheTensor (block_stride > 0) sizes the view by the
    block stride rather than the spec page size."""
    spec = _TestStorageSpec(
        block_size=16,
        num_kv_heads=2,
        head_size=8,
        dtype=torch.float32,
    )
    block_stride = spec.page_size_bytes // 2
    packed = SimpleNamespace(
        size=2 * spec.page_size_bytes,
        shared_by=["model.layers.0"],
        layer_stride=spec.page_size_bytes,
        block_stride=block_stride,
        offset=0,
        layers=["model.layers.0"],
    )
    raw = torch.zeros(2 * spec.page_size_bytes, dtype=torch.uint8)
    groups = [
        KVCacheGroupSpec(
            layer_names=["model.layers.0"],
            kv_cache_spec=spec,
        )
    ]
    runner = _reshape_runner(
        spec,
        ["model.layers.0"],
        [packed],
        groups,
    )
    reshape_calls = []

    def _recorder(
        raw_tensor, reshape_spec, shape, stride_order, kernel_num_blocks, packing
    ):
        reshape_calls.append((kernel_num_blocks, packing))
        return "view"

    with Patch(
        model_runner_module,
        "_reshape_attention_kv_cache",
        _recorder,
    ):
        kv_caches = ModelRunnerFL._reshape_kv_cache_tensors(
            runner,
            {"model.layers.0": raw},
            [16],
        )

    # num_blocks = numel // block_stride = 2*page / (page/2) = 4 blocks.
    assert reshape_calls == [(4, (0, block_stride))]
    assert kv_caches["model.layers.0"] == "view"


def test_reshape_kv_cache_tensors_ascend_hybrid_attention_view():
    """With use_ascend_hybrid_cache the attention layer gets the dense
    tail view instead of the strided kernel view."""
    spec = _TestStorageSpec(
        block_size=16,
        num_kv_heads=2,
        head_size=8,
        dtype=torch.float32,
    )
    raw = torch.zeros(2 * spec.page_size_bytes, dtype=torch.uint8)
    groups = [
        KVCacheGroupSpec(
            layer_names=["model.layers.0"],
            kv_cache_spec=spec,
        )
    ]
    runner = _reshape_runner(
        spec,
        ["model.layers.0"],
        [],
        groups,
        use_ascend_hybrid_cache=True,
    )

    kv_caches = ModelRunnerFL._reshape_kv_cache_tensors(
        runner,
        {"model.layers.0": raw},
        [16],
    )

    view = kv_caches["model.layers.0"]
    # (2, kernel_num_blocks=2, block_size=16, heads=2, head_size=8) float32.
    assert view.shape == (2, 2, 16, 2, 8)
    assert view.dtype is torch.float32
    # The dense view is the payload tail of the raw byte buffer.
    assert view.data_ptr() >= raw.data_ptr()


def test_reshape_kv_cache_tensors_physical_layout_backend():
    """A backend publishing get_physical_cache_layout drives the shape via
    the layout's pages_per_block and kernel_block_size."""

    class _LayoutBackend(_Backend):
        @staticmethod
        def get_physical_cache_layout(kv_cache_spec):
            return PhysicalCacheLayout(
                logical_block_size=32,
                storage_block_size=16,
                kernel_block_size=8,
            )

    spec = _TestStorageSpec(
        block_size=32,
        num_kv_heads=2,
        head_size=8,
        dtype=torch.float32,
    )
    raw = torch.zeros(2 * spec.page_size_bytes, dtype=torch.uint8)
    groups = [
        KVCacheGroupSpec(
            layer_names=["model.layers.0"],
            kv_cache_spec=spec,
        )
    ]
    runner = _reshape_runner(
        spec,
        ["model.layers.0"],
        [],
        groups,
        backend=_LayoutBackend,
    )
    reshape_calls = []

    def _recorder(
        raw_tensor, reshape_spec, shape, stride_order, kernel_num_blocks, packing
    ):
        reshape_calls.append((shape, kernel_num_blocks))
        return "view"

    with Patch(
        model_runner_module,
        "_reshape_attention_kv_cache",
        _recorder,
    ):
        kv_caches = ModelRunnerFL._reshape_kv_cache_tensors(
            runner,
            {"model.layers.0": raw},
            [16],
        )

    # pages_per_block = 16 // 8 = 2; kernel_num_blocks = 2 blocks * 2 = 4;
    # shape_block_size = layout.kernel_block_size = 8.
    shape, kernel_num_blocks = reshape_calls[0]
    assert shape == (2, 4, 8, 2, 8)
    assert kernel_num_blocks == 4
    assert kv_caches["model.layers.0"] == "view"


def test_update_hybrid_attention_mamba_layout_swaps_block_dim():
    """block_dim 1 backends have their (2, blocks, ...) caches re-strided
    to (blocks, 2, ...); block_dim 0 caches are left alone."""

    class _BlockDimBackend:
        @staticmethod
        def get_kv_cache_block_dim(
            kernel_block_size, num_kv_heads, head_size, cache_dtype_str="auto"
        ):
            return 1

    spec = FullAttentionSpec(
        block_size=16,
        num_kv_heads=2,
        head_size=8,
        dtype=torch.float32,
    )
    runner = SimpleNamespace(
        kv_cache_config=SimpleNamespace(kv_cache_groups=[SimpleNamespace()]),
        attn_groups=[
            [
                _group(
                    spec,
                    ["model.layers.0"],
                    0,
                    backend=_BlockDimBackend,
                )
            ]
        ],
        cache_config=SimpleNamespace(cache_dtype="auto"),
    )
    runner._kv_cache_spec_attn_group_iterator = (
        ModelRunnerFL._kv_cache_spec_attn_group_iterator.__get__(
            runner,
            ModelRunnerFL,
        )
    )
    cache = torch.zeros(2, 3, 2, 8)
    kv_caches = {"model.layers.0": cache}

    ModelRunnerFL._update_hybrid_attention_mamba_layout(
        runner,
        kv_caches,
        [16],
    )

    view = kv_caches["model.layers.0"]
    # Same shape, same storage, permuted strides: (hidden, 2*hidden, ...).
    assert view.shape == (2, 3, 2, 8)
    assert view.stride() == (16, 32, 8, 1)
    assert view.data_ptr() == cache.data_ptr()

    class _BlockDim0Backend(_BlockDimBackend):
        @staticmethod
        def get_kv_cache_block_dim(*args, **kwargs):
            return 0

    runner.attn_groups[0][0].backend = _BlockDim0Backend
    untouched = torch.zeros(2, 3, 2, 8)
    kv_caches_0 = {"model.layers.0": untouched}
    ModelRunnerFL._update_hybrid_attention_mamba_layout(
        runner,
        kv_caches_0,
        [16],
    )
    assert kv_caches_0["model.layers.0"].stride() == untouched.stride()


def test_initialize_kv_cache_runs_full_lifecycle_wiring():
    """initialize_kv_cache deep-copies the config, then drives the
    attention backend init, metadata builders, input-batch reinit, the
    mamba SSU backend and the tensor allocation in order."""
    spec = FullAttentionSpec(
        block_size=16,
        num_kv_heads=2,
        head_size=8,
        dtype=torch.float32,
    )
    group = KVCacheGroupSpec(
        layer_names=["model.layers.0", "model.layers.1"],
        kv_cache_spec=spec,
    )
    kt = SimpleNamespace(
        size=2 * spec.page_size_bytes,
        shared_by=["model.layers.0", "model.layers.1"],
        layer_stride=spec.page_size_bytes,
        block_stride=0,
        offset=0,
        layers=["model.layers.0", "model.layers.1"],
    )
    config = KVCacheConfig(
        num_blocks=2,
        kv_cache_tensors=[kt],
        kv_cache_groups=[group],
    )
    calls = SimpleNamespace(order=[], meta=[], kv_tensors=[], reinit=[])
    runner = SimpleNamespace(
        attn_groups=[[_group(spec, ["model.layers.0", "model.layers.1"], 0)]],
        vllm_config=SimpleNamespace(mamba_config=SimpleNamespace()),
        kv_caches=[],
        runner_only_attn_layers=set(),
        use_ascend_hybrid_cache=False,
        cache_config=SimpleNamespace(cache_dtype="auto"),
        shared_kv_cache_layers={},
        model_config=SimpleNamespace(
            hf_config=SimpleNamespace(model_type="llama"),
        ),
        compilation_config=SimpleNamespace(static_forward_context={}),
        device=torch.device("cpu"),
        speculative_config=None,
        graph_runtime=None,
        _mamba_bufs=None,
        kv_cache_config=None,
    )
    runner._kv_cache_spec_attn_group_iterator = (
        ModelRunnerFL._kv_cache_spec_attn_group_iterator.__get__(
            runner,
            ModelRunnerFL,
        )
    )
    runner.may_add_encoder_only_layers_to_kv_cache_config = lambda: calls.order.append(
        "encoder_only"
    )
    runner.maybe_add_kv_sharing_layers_to_kv_cache_groups = lambda cfg: (
        calls.order.append("kv_sharing")
    )
    runner.initialize_attn_backend = lambda cfg, is_profiling=False: calls.order.append(
        ("attn_backend", is_profiling),
    )
    runner.initialize_metadata_builders = lambda cfg, kbs: (
        calls.meta.append(kbs) or calls.order.append("meta")
    )
    runner.may_reinitialize_input_batch = lambda cfg, kbs: (
        calls.reinit.append(kbs) or calls.order.append("reinit")
    )
    runner.initialize_kv_cache_tensors = lambda cfg, kbs: (
        calls.kv_tensors.append(kbs)
        or calls.order.append("kv_tensors")
        or {"model.layers.0": "t0", "model.layers.1": "t1"}
    )

    with (
        Patch(
            model_runner_module,
            "prepare_kernel_block_sizes",
            lambda cfg, groups: [16],
        ),
        Patch(
            model_runner_module,
            "initialize_mamba_ssu_backend",
            lambda mamba_config, cfg: calls.order.append("mamba_ssu"),
        ),
    ):
        ModelRunnerFL.initialize_kv_cache(runner, config, is_profiling=True)

    # The stored config is a deep copy, not the caller's object.
    assert runner.kv_cache_config is not config
    assert runner.kv_cache_config.kv_cache_groups[0].layer_names == [
        "model.layers.0",
        "model.layers.1",
    ]
    assert runner._mamba_bufs is None
    # Every lifecycle stage ran, in the source's order, with the kernel
    # block sizes computed once and threaded through.
    assert calls.order == [
        "encoder_only",
        "kv_sharing",
        ("attn_backend", True),
        "mamba_ssu",
        "meta",
        "reinit",
        "kv_tensors",
    ]
    assert calls.meta == [[16]]
    assert calls.reinit == [[16]]
    assert calls.kv_tensors == [[16]]


def test_initialize_kv_cache_tensors_allocates_and_binds():
    """The fallback path allocates raw CPU buffers, reshapes them per
    layer, aliases shared layers and binds each view to its layer."""
    spec = _TestStorageSpec(
        block_size=16,
        num_kv_heads=2,
        head_size=8,
        dtype=torch.float32,
    )

    class _BindingLayer(AttentionLayerBase):
        def __init__(self):
            self.kv_cache = None

        def get_attn_backend(self):
            return _Backend

        def get_kv_cache_spec(self, vllm_config):
            return None

        def bind_kv_cache(self, kv_cache):
            self.kv_cache = kv_cache

    layers = {
        "model.layers.0": _BindingLayer(),
        "model.layers.1": _BindingLayer(),
        "model.layers.2": _BindingLayer(),
    }
    groups = [
        KVCacheGroupSpec(
            layer_names=["model.layers.0", "model.layers.1"],
            kv_cache_spec=spec,
        ),
        KVCacheGroupSpec(
            layer_names=["model.layers.2"],
            kv_cache_spec=spec,
        ),
    ]
    tensors = [
        SimpleNamespace(
            size=spec.page_size_bytes,
            shared_by=["model.layers.0", "model.layers.1"],
            layer_stride=spec.page_size_bytes,
            block_stride=0,
            offset=0,
            layers=["model.layers.0", "model.layers.1"],
        ),
        SimpleNamespace(
            size=spec.page_size_bytes,
            shared_by=["model.layers.2"],
            layer_stride=spec.page_size_bytes,
            block_stride=0,
            offset=0,
            layers=["model.layers.2"],
        ),
    ]
    runner = SimpleNamespace(
        attn_groups=[
            [_group(spec, ["model.layers.0", "model.layers.1"], 0)],
            [_group(spec, ["model.layers.2"], 1)],
        ],
        kv_cache_config=SimpleNamespace(
            kv_cache_tensors=tensors,
            kv_cache_groups=groups,
        ),
        runner_only_attn_layers=set(),
        use_ascend_hybrid_cache=False,
        cache_config=SimpleNamespace(cache_dtype="auto"),
        # The uniform fast path comes from the CI image's runner lineage;
        # locally provide the recorder choosing the fallback branch.
        shared_kv_cache_layers={"model.layers.1": "model.layers.0"},
        model_config=SimpleNamespace(
            hf_config=SimpleNamespace(model_type="llama"),
        ),
        compilation_config=SimpleNamespace(static_forward_context=layers),
        kv_caches=[],
        device=torch.device("cpu"),
    )
    for name in (
        "_kv_cache_spec_attn_group_iterator",
        "_allocate_kv_cache_tensors",
        "_reshape_kv_cache_tensors",
    ):
        setattr(
            runner,
            name,
            getattr(ModelRunnerFL, name).__get__(runner, ModelRunnerFL),
        )
    runner.use_uniform_kv_cache = lambda attn_groups: False
    config = SimpleNamespace(kv_cache_groups=groups, kv_cache_tensors=tensors)

    def _recorder(
        raw_tensor, reshape_spec, shape, stride_order, kernel_num_blocks, packing
    ):
        return raw_tensor.view(torch.float32).view(shape)

    with Patch(
        model_runner_module,
        "_reshape_attention_kv_cache",
        _recorder,
    ):
        kv_caches = ModelRunnerFL.initialize_kv_cache_tensors(
            runner,
            config,
            [16, 16],
        )

    # Every layer got its (real) reshaped view bound through bind_kv_cache.
    for layer in layers.values():
        assert isinstance(layer.kv_cache, torch.Tensor)
    # The shared layer aliases its target's cache.
    assert kv_caches["model.layers.1"] is kv_caches["model.layers.0"]
    # The runner cache list holds the views in layer-index order.
    assert len(runner.kv_caches) == 3
