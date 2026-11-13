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

"""GLM5 kpool cache hooks: grouping, runtime patch factories and idempotency.

Each patch factory is exercised by invoking its replacement closure with a
fake owner, after capturing a recording "pristine" callable.  Block-table
translation and zeroer filtering assert exact tensor contents derived from the
input table.
"""

from types import SimpleNamespace

import pytest
import torch

from vllm.v1.kv_cache_interface import (
    KVCacheGroupSpec,
    MambaSpec,
    MLAAttentionSpec,
    UniformTypeKVCacheSpecs,
)

from . import _import_shim  # noqa: F401  (idempotent; no-op on CI)
from vllm_fl.models.glm5 import cache as pc
from vllm_fl.models.glm5.kpool import KpoolTailSpec
from vllm_fl.runtime.kv_layout import PhysicalCacheLayout


def _mla_spec(*, block_size=512, compress_ratio=1):
    return MLAAttentionSpec(
        block_size=block_size,
        num_kv_heads=1,
        head_size=128,
        dtype=torch.bfloat16,
        compress_ratio=compress_ratio,
        indexes_kv_by_block_stride=True,
    )


def _tail_spec():
    return KpoolTailSpec(
        block_size=4,
        num_kv_heads=1,
        head_size=256,
        head_size_v=0,
        dtype=torch.bfloat16,
        sliding_window=4,
        indexes_kv_by_block_stride=True,
    )


def _mamba_spec():
    return MambaSpec(
        block_size=512,
        shapes=((2, 128), (128, 128)),
        dtypes=(torch.bfloat16, torch.float32),
        mamba_cache_mode="align",
    )


class TestInnerSpecs:
    def test_uniform_groups_yield_their_member_mapping(self, monkeypatch):
        import vllm.v1.kv_cache_interface as kvif

        class FakeUniform:
            def __init__(self, mapping):
                self.kv_cache_specs = mapping

        monkeypatch.setattr(kvif, "UniformTypeKVCacheSpecs", FakeUniform)
        spec_a, spec_b = object(), object()
        group = SimpleNamespace(kv_cache_spec=FakeUniform({"a": spec_a, "b": spec_b}))
        assert list(pc._inner_specs([group])) == [("a", spec_a), ("b", spec_b)]

    def test_plain_groups_repeat_the_group_spec_per_layer(self):
        spec = object()
        group = SimpleNamespace(kv_cache_spec=spec, layer_names=["x", "y"])
        assert list(pc._inner_specs([group])) == [("x", spec), ("y", spec)]

    def test_tail_specs_identify_kpool_groups(self):
        tail = _tail_spec()
        assert (
            pc._is_glm5_kpool_groups(
                [SimpleNamespace(kv_cache_spec=tail, layer_names=["tail"])]
            )
            is True
        )
        assert (
            pc._is_glm5_kpool_groups(
                [SimpleNamespace(kv_cache_spec=object(), layer_names=["x"])]
            )
            is False
        )


class TestGroupGlm5Kpool:
    def test_attention_tail_and_mamba_groups_are_separated(self):
        specs = {
            "layers.0.mla": _mla_spec(),
            "layers.0.indexer.k_cache": _mla_spec(compress_ratio=4),
            "layers.0.indexer.tail_cache": _tail_spec(),
            "layers.0.kda": _mamba_spec(),
        }
        groups = pc._group_glm5_kpool(None, specs)
        assert len(groups) == 3
        attn_group, tail_group, mamba_group = groups
        assert isinstance(attn_group, KVCacheGroupSpec)
        assert set(attn_group.layer_names) == {
            "layers.0.mla",
            "layers.0.indexer.k_cache",
        }
        assert isinstance(attn_group.kv_cache_spec, UniformTypeKVCacheSpecs)
        assert tail_group.layer_names == ["layers.0.indexer.tail_cache"]
        assert isinstance(tail_group.kv_cache_spec, UniformTypeKVCacheSpecs)
        tail = tail_group.kv_cache_spec.kv_cache_specs["layers.0.indexer.tail_cache"]
        assert tail.page_size_padded == _mla_spec(compress_ratio=4).page_size_bytes
        assert isinstance(mamba_group.kv_cache_spec, MambaSpec)

    def test_mamba_is_optional(self):
        specs = {
            "mla": _mla_spec(),
            "index": _mla_spec(compress_ratio=4),
            "tail": _tail_spec(),
        }
        groups = pc._group_glm5_kpool(None, specs)
        assert len(groups) == 2

    def test_non_mla_attention_is_rejected(self):
        specs = {
            "mla": _mla_spec(),
            "index": _mla_spec(compress_ratio=4),
            "tail": _tail_spec(),
            "dense": object(),
        }
        with pytest.raises(AssertionError):
            pc._group_glm5_kpool(None, specs)


class _Owner:
    value = "pristine"


class TestRuntimePatchRegistry:
    def test_capture_and_lookup_are_keyed_by_owner_and_attribute(self, monkeypatch):
        monkeypatch.setattr(pc, "_RUNTIME_BASELINES", {})
        pc._capture_runtime_baseline(_Owner, "value")
        assert pc._runtime_pristine(_Owner, "value") == "pristine"
        target = pc._kpool_target(_Owner, "value")
        assert target.endswith("_Owner.value")

    def test_runtime_patch_records_the_pristine_callable(self, monkeypatch):
        monkeypatch.setattr(pc, "_RUNTIME_BASELINES", {})
        pc._capture_runtime_baseline(_Owner, "value")

        def replacement(self, a, b):
            return a + b

        patch = pc._runtime_patch(
            _Owner, "value", replacement, "fp", ("self", "a", "b")
        )
        assert patch.owner is _Owner
        assert patch.attr == "value"
        assert patch.replacement is replacement
        assert patch.pristine == "pristine"
        assert patch.fingerprint == "fp"
        assert patch.expected_params == ("self", "a", "b")


def _capture(monkeypatch, owner, attr):
    monkeypatch.setattr(pc, "_RUNTIME_BASELINES", {})
    pc._capture_runtime_baseline(owner, attr)
    return pc._runtime_pristine(owner, attr)


def _install_layout(monkeypatch, layout):
    import vllm_fl.runtime.kv_layout as layout_module

    monkeypatch.setattr(
        layout_module,
        "get_physical_cache_layout",
        lambda backend, spec: layout,
    )


class TestMetadataBuilderPatch:
    def test_layout_none_delegates_to_the_pristine_builder(self, monkeypatch):
        from vllm.v1.worker import utils as worker_utils

        calls = []

        def pristine(self, config, device, kernel_block_size, count):
            calls.append((self, config, device, kernel_block_size, count))
            return "base"

        monkeypatch.setattr(
            worker_utils.AttentionGroup, "create_metadata_builders", pristine
        )
        _capture(monkeypatch, worker_utils.AttentionGroup, "create_metadata_builders")
        _install_layout(monkeypatch, None)
        patch = pc._create_metadata_builders_patch("fp")
        owner = SimpleNamespace(
            backend="backend",
            kv_cache_spec="spec",
            layer_names=["p"],
        )
        result = patch.replacement(owner, "config", "device", 8, 2)
        assert result == "base"
        assert calls == [(owner, "config", "device", 8, 2)]

    def test_layout_rewrites_the_builder_spec_and_records_the_layout(self, monkeypatch):
        from vllm.v1.worker import utils as worker_utils

        built = []

        class FakeBuilder:
            def __init__(self, spec, layer_names, config, device):
                self.spec = spec
                self.layer_names = layer_names
                self.kernel_block_size = None
                built.append(self)

        layout = PhysicalCacheLayout(64, 32, 32)
        _install_layout(monkeypatch, layout)
        spec = SimpleNamespace(copy_with_new_block_size=lambda size: ("spec", size))
        backend = SimpleNamespace(get_builder_cls=lambda: FakeBuilder)
        monkeypatch.setattr(
            worker_utils.AttentionGroup,
            "create_metadata_builders",
            lambda *args: pytest.fail("layout path must not delegate"),
        )
        _capture(monkeypatch, worker_utils.AttentionGroup, "create_metadata_builders")
        patch = pc._create_metadata_builders_patch("fp")
        owner = SimpleNamespace(backend=backend, kv_cache_spec=spec, layer_names=["p"])
        assert patch.replacement(owner, "config", "device", 8, 2) is None
        assert len(built) == 2
        assert built[0].spec == ("spec", 64)
        assert built[0]._glm5_physical_layout is layout
        assert built[0].kernel_block_size == 8


class TestIndexerBuildPatch:
    def _build(self, monkeypatch, *, physical=True, use_nvidia=False):
        import vllm.v1.attention.backends.mla.indexer as indexer_backend

        import vllm_fl.models.glm5.provider as provider

        calls = []

        def pristine(self, prefix, common, fast_build=False):
            calls.append((self, prefix, common, fast_build))
            return common

        monkeypatch.setattr(
            indexer_backend.DeepseekV32IndexerMetadataBuilder, "build", pristine
        )
        if physical:
            monkeypatch.setattr(indexer_backend, "has_deep_gemm", lambda: True)
        _capture(
            monkeypatch,
            indexer_backend.DeepseekV32IndexerMetadataBuilder,
            "build",
        )
        monkeypatch.setattr(provider, "use_nvidia_reference", lambda: use_nvidia)
        patch = pc._indexer_build_patch("fp")
        return patch, calls

    def _common(self, table):
        def replace(**changes):
            return SimpleNamespace(**changes)

        return SimpleNamespace(block_table_tensor=table, replace=replace)

    def test_block_table_is_downsampled_and_copied_into_a_stable_buffer(
        self, monkeypatch
    ):
        patch, calls = self._build(monkeypatch)
        table = torch.arange(8, dtype=torch.int32).reshape(2, 4)
        owner = SimpleNamespace(
            kv_cache_spec=SimpleNamespace(block_size=8),
            kernel_block_size=4,
            _glm5_physical_layout=object(),
            vllm_config=SimpleNamespace(
                scheduler_config=SimpleNamespace(max_num_batched_tokens=8)
            ),
        )
        common = self._common(table)
        result = patch.replacement(owner, 0, common, False)
        # Every second column, then divided by the layout factor.
        expected = table[:, ::2] // 2
        assert result.block_table_tensor.tolist() == expected.tolist()
        assert (
            result.block_table_tensor.data_ptr()
            == owner._glm5_indexer_block_table.data_ptr()
        )
        assert calls[0][2] is result
        assert calls[0][3] is False

    def test_buffer_is_reused_across_graph_buckets(self, monkeypatch):
        patch, _ = self._build(monkeypatch, physical=False)
        owner = SimpleNamespace(
            kv_cache_spec=SimpleNamespace(block_size=8),
            kernel_block_size=4,
            _glm5_physical_layout=object(),
            vllm_config=SimpleNamespace(
                scheduler_config=SimpleNamespace(max_num_batched_tokens=8)
            ),
        )
        first = patch.replacement(
            owner,
            0,
            self._common(torch.zeros(2, 4, dtype=torch.int32)),
            False,
        )
        second = patch.replacement(
            owner,
            0,
            self._common(torch.ones(1, 4, dtype=torch.int32)),
            False,
        )
        assert (
            first.block_table_tensor.data_ptr() == second.block_table_tensor.data_ptr()
        )

    def test_width_and_row_capacity_changes_are_rejected(self, monkeypatch):
        patch, _ = self._build(monkeypatch, physical=False)
        owner = SimpleNamespace(
            kv_cache_spec=SimpleNamespace(block_size=8),
            kernel_block_size=4,
            _glm5_physical_layout=object(),
            _glm5_indexer_block_table=torch.zeros(8, 3, dtype=torch.int32),
            vllm_config=SimpleNamespace(
                scheduler_config=SimpleNamespace(max_num_batched_tokens=8)
            ),
        )
        with pytest.raises(RuntimeError, match="width changed"):
            patch.replacement(
                owner,
                0,
                self._common(torch.zeros(2, 4, dtype=torch.int32)),
                False,
            )
        owner._glm5_indexer_block_table = torch.zeros(1, 2, dtype=torch.int32)
        with pytest.raises(RuntimeError, match="rows exceed"):
            patch.replacement(
                owner,
                0,
                self._common(torch.zeros(2, 4, dtype=torch.int32)),
                False,
            )

    def test_absent_kernel_block_size_skips_translation(self, monkeypatch):
        patch, _ = self._build(monkeypatch, physical=False)
        table = torch.arange(8, dtype=torch.int32).reshape(2, 4)
        owner = SimpleNamespace(
            kv_cache_spec=SimpleNamespace(block_size=8),
            kernel_block_size=None,
            _glm5_physical_layout=object(),
            vllm_config=SimpleNamespace(
                scheduler_config=SimpleNamespace(max_num_batched_tokens=8)
            ),
        )
        common = self._common(table)
        result = patch.replacement(owner, 0, common, False)
        assert result is common


class TestZeroerPatch:
    def test_groups_with_a_physical_layout_are_removed(self, monkeypatch):
        from vllm.v1.worker import utils as worker_utils

        calls = []

        def pristine(
            self,
            device,
            pin_memory,
            groups,
            kernel_block_sizes,
            cache_dtype,
            static_forward_context,
            runner_only_attn_layers,
        ):
            calls.append(list(groups))
            return "base"

        monkeypatch.setattr(worker_utils.KVBlockZeroer, "__init__", pristine)
        _capture(monkeypatch, worker_utils.KVBlockZeroer, "__init__")
        layout = PhysicalCacheLayout(64, 32, 32)
        import vllm_fl.runtime.kv_layout as layout_module

        monkeypatch.setattr(
            layout_module,
            "get_physical_cache_layout",
            lambda backend, spec: layout if backend == "layout" else None,
        )
        patch = pc._zeroer_init_patch("fp")
        groups = [
            SimpleNamespace(backend="layout", kv_cache_spec="a"),
            SimpleNamespace(backend="plain", kv_cache_spec="b"),
        ]
        assert (
            patch.replacement(
                self=None,
                device="cpu",
                pin_memory=False,
                attn_groups_iter=groups,
                kernel_block_sizes={},
                cache_dtype="auto",
                static_forward_context={},
                runner_only_attn_layers=None,
            )
            == "base"
        )
        assert [group.backend for group in calls[0]] == ["plain"]


class TestPatchBundle:
    def test_bundle_contains_all_worker_patches(self, monkeypatch):
        from vllm.v1.attention.backends.mla import indexer as indexer_backend
        from vllm.v1.worker import utils as worker_utils

        monkeypatch.setattr(pc, "_RUNTIME_BASELINES", {})
        pc._capture_runtime_baseline(
            worker_utils.AttentionGroup, "create_metadata_builders"
        )
        pc._capture_runtime_baseline(
            indexer_backend.DeepseekV32IndexerMetadataBuilder, "build"
        )
        pc._capture_runtime_baseline(worker_utils.KVBlockZeroer, "__init__")
        patches = pc.glm5_next_kpool_runtime_patches("fp")
        assert [patch.attr for patch in patches] == [
            "create_metadata_builders",
            "build",
            "__init__",
        ]
        assert all(patch.fingerprint == "fp" for patch in patches)

    def test_install_is_idempotent_after_the_first_transaction(self, monkeypatch):
        marker = object()
        bound = []
        monkeypatch.setattr(pc, "_EARLY_PATCHES", [marker])
        monkeypatch.setattr(pc, "bind_patches", lambda patches: bound.append(patches))
        pc.install_glm5_next_kpool()
        assert bound == [[marker]]
