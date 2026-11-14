# SPDX-License-Identifier: Apache-2.0
"""CPU tensor tests for Ascend's manager, kernel, and physical block views.

Lightweight vLLM metadata fixtures keep these layout tests runnable without
installing vLLM or an NPU runtime. All storage and indexing operations use
real PyTorch tensors.
"""

import importlib.util
import sys
import types
import unittest
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import torch


@dataclass
class AttentionSpec:
    block_size: int = 4
    num_kv_heads: int = 2
    head_size: int = 2
    dtype: torch.dtype = torch.float32
    page_size_padded: int | None = None
    tokens_per_state: int = 1

    @property
    def head_size_v(self):
        return self.head_size

    @property
    def page_size_bytes(self):
        return self.page_size_padded or (
            2
            * self.block_size
            * self.num_kv_heads
            * self.head_size
            * torch.empty((), dtype=self.dtype).element_size()
        )


@dataclass
class MambaSpec:
    shapes: tuple = ((3,), (4, 2))
    dtypes: tuple = (torch.float32, torch.float32)
    page_size_padded: int = 76

    @property
    def page_size_bytes(self):
        return self.page_size_padded


class UniformTypeKVCacheSpecs:
    pass


def metadata_group(spec, names):
    return SimpleNamespace(kv_cache_spec=spec, layer_names=names)


def tensor_config(size, names, layer_stride, block_stride, offset=0):
    return SimpleNamespace(
        size=size,
        layers=names,
        layer_stride=layer_stride,
        block_stride=block_stride,
        offset=offset,
        host_resident=False,
    )


def cache_config(num_blocks, tensors, groups):
    return SimpleNamespace(
        num_blocks=num_blocks,
        kv_cache_tensors=tensors,
        kv_cache_groups=groups,
        num_blocks_of=lambda _: num_blocks,
    )


class AscendKVLayoutTests(unittest.TestCase):
    def setUp(self):
        self.allocate = Mock()
        modules = {}
        for name in (
            "vllm",
            "vllm.model_executor",
            "vllm.model_executor.models",
            "vllm.utils",
            "vllm.utils.math_utils",
            "vllm.utils.torch_utils",
            "vllm.v1",
            "vllm.v1.kv_cache_interface",
            "vllm.v1.worker",
            "vllm.v1.worker.utils",
        ):
            module = types.ModuleType(name)
            module.__path__ = []
            modules[name] = module
        modules["vllm.model_executor.models"].ModelRegistry = object()
        modules["vllm.utils.math_utils"].cdiv = lambda a, b: (a + b - 1) // b
        torch_utils = modules["vllm.utils.torch_utils"]
        torch_utils.STR_DTYPE_TO_TORCH_DTYPE = {"float32": torch.float32}
        torch_utils.get_dtype_size = lambda dtype: torch.empty(
            (), dtype=dtype
        ).element_size()
        interface = modules["vllm.v1.kv_cache_interface"]
        interface.AttentionSpec = AttentionSpec
        interface.MambaSpec = MambaSpec
        interface.UniformTypeKVCacheSpecs = UniformTypeKVCacheSpecs
        modules["vllm.v1.worker.utils"].allocate_kv_cache = self.allocate
        module_patch = patch.dict(sys.modules, modules)
        module_patch.start()
        self.addCleanup(module_patch.stop)
        source = (
            Path(__file__).resolve().parents[2] / "vllm_fl/kv_cache/ascend/hybrid.py"
        )
        spec = importlib.util.spec_from_file_location(
            "ascend_layout_under_test", source
        )
        self.hybrid = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(self.hybrid)

    def dense_fixture(self, num_blocks=4, attention_only=False):
        # A 12-byte conv row and a 32-byte recurrent/K row intentionally differ.
        attention = AttentionSpec(dtype=torch.float16, page_size_padded=76)
        mamba = MambaSpec()
        pool_size = num_blocks * 76
        groups = [metadata_group(attention, ["a0", "a1"])]
        tensors = [tensor_config(2 * pool_size, ["a0", "a1"], pool_size, 76)]
        if not attention_only:
            groups.append(metadata_group(mamba, ["m0", "m1"]))
            tensors.append(tensor_config(2 * pool_size, ["m0", "m1"], pool_size, 76))
        config = cache_config(num_blocks, tensors, groups)
        attn_group = SimpleNamespace(
            kv_cache_spec=attention,
            kv_cache_group_id=0,
            kernel_block_size=2,
            kernel_block_stride=None,
        )
        return config, attn_group

    def test_planar_manager_maps_to_kernel_without_copy_or_virtual_allocation(self):
        spec = AttentionSpec()
        blocks = 4
        planar = torch.arange(2 * blocks * 4 * 4, dtype=torch.float32).view(
            2, blocks, 4, 4
        )
        manager = planar.permute(1, 0, 2, 3)
        self.allocate.return_value = {"a0": manager}
        config = cache_config(
            blocks,
            [tensor_config(planar.nbytes, ["a0"], planar.nbytes, 64)],
            [metadata_group(spec, ["a0"])],
        )
        group = SimpleNamespace(
            kv_cache_spec=spec,
            kv_cache_group_id=0,
            kernel_block_size=2,
            kernel_block_stride=None,
        )
        bound, managers, planes = self.hybrid.allocate_ascend_kv_cache(
            config,
            torch.device("cpu"),
            SimpleNamespace(name="LHBNC"),
            [2],
            [group],
            False,
        )
        self.allocate.assert_called_once_with(
            config,
            torch.device("cpu"),
            SimpleNamespace(name="LHBNC"),
            kernel_block_sizes=None,
        )
        self.assertIs(managers["a0"], manager)
        self.assertEqual(bound["a0"].shape, (2, 8, 2, 2, 2))
        self.assertEqual(bound["a0"].data_ptr(), planar.data_ptr())
        self.assertTrue(bound["a0"].is_contiguous())
        self.assertEqual(group.kernel_block_stride, 2)
        self.assertEqual(len(planes), 2)
        bound["a0"][1, 3, 0, 0, 0] = -17
        self.assertEqual(manager[1, 1, 2, 0].item(), -17)

    def test_dense_padded_pools_keep_state_key_aliases_and_layer_offsets(self):
        config, group = self.dense_fixture()
        bound, managers, planes = self.hybrid.allocate_ascend_kv_cache(
            config, "cpu", SimpleNamespace(name="LBNHC"), [2, 4], [group], True
        )
        self.assertEqual(set(bound), {"a0", "a1", "m0", "m1"})
        self.assertEqual(set(managers), {"a0", "a1"})
        # Every overlay uses the same three byte planes; no dtype aliases duplicate them.
        self.assertEqual(len(planes), 6)
        self.assertEqual(
            [tuple(p.shape) for p in planes], [(4, 12), (4, 32), (4, 32)] * 2
        )
        conv, ssm = self.hybrid.dense_mamba_views(
            bound["m0"].view(-1), ((3,), (4, 2)), (torch.float32, torch.float32), 4
        )
        self.assertEqual(conv.data_ptr(), planes[0].data_ptr())
        self.assertEqual(ssm.data_ptr(), bound["a0"][0].data_ptr())
        self.assertEqual(bound["a1"].data_ptr() - bound["a0"].data_ptr(), 4 * 76)
        self.assertEqual(
            bound["a0"].untyped_storage().data_ptr(),
            bound["m1"].untyped_storage().data_ptr(),
        )
        self.assertEqual(managers["a0"].shape, (4, 2, 4, 4))
        for plane in planes:
            for row in range(4):
                plane[row].fill_(row + 1)
        pointers = [plane.data_ptr() for plane in planes]
        self.hybrid.copy_block_planes(planes, [(0, 1), (1, 2)])
        self.hybrid.zero_block_planes(planes, [3])
        for plane, pointer in zip(planes, pointers):
            self.assertEqual(plane.data_ptr(), pointer)
            self.assertTrue(torch.all(plane[0] == 1).item())
            self.assertTrue(torch.all(plane[1] == 1).item())
            self.assertTrue(torch.all(plane[2] == 2).item())
            self.assertEqual(torch.count_nonzero(plane[3]).item(), 0)

    def test_attention_only_pipeline_stage_derives_conv_prefix(self):
        config, group = self.dense_fixture(attention_only=True)
        bound, managers, planes = self.hybrid.allocate_ascend_kv_cache(
            config, "cpu", SimpleNamespace(name="LBNHC"), [2], [group], True
        )
        self.assertEqual(set(bound), set(managers))
        self.assertEqual([p.shape[1] for p in planes], [12, 32, 32] * 2)

    def test_copy_snapshots_all_aliasing_planes_before_any_write(self):
        raw = torch.arange(12, dtype=torch.int32).view(4, 3)
        before = raw.clone()
        self.hybrid.copy_block_planes([raw[:, :2], raw[:, 1:]], [(0, 1), (1, 2)])
        torch.testing.assert_close(raw[1], before[0])
        torch.testing.assert_close(raw[2], before[1])
        torch.testing.assert_close(raw[0], before[0])
        torch.testing.assert_close(raw[3], before[3])

    def test_byte_equivalent_dtype_views_are_deduplicated(self):
        raw = torch.arange(64, dtype=torch.uint8).view(4, 16)
        planes = self.hybrid._deduplicate_block_planes([raw, raw.view(torch.float32)])
        self.assertEqual(len(planes), 1)
        self.assertEqual(planes[0].data_ptr(), raw.data_ptr())

    def test_zero_strided_rows_in_place_and_leaves_other_columns(self):
        raw = torch.ones(4, 5, dtype=torch.float32)
        pointer = raw.data_ptr()
        self.hybrid.zero_block_planes([raw[:, 1:4]], [1, 3])
        self.assertEqual(raw.data_ptr(), pointer)
        self.assertTrue(torch.all(raw[[0, 2]] == 1).item())
        self.assertTrue(torch.all(raw[:, [0, 4]] == 1).item())
        self.assertEqual(torch.count_nonzero(raw[[1, 3], 1:4]).item(), 0)

    def test_dense_packed_blocks_are_rejected(self):
        config, group = self.dense_fixture()
        config.kv_cache_tensors[0].block_stride *= 2
        with self.assertRaisesRegex(NotImplementedError, "layer-compact"):
            self.hybrid.allocate_ascend_kv_cache(
                config, "cpu", SimpleNamespace(name="LBNHC"), [2, 4], [group], True
            )

    def test_dense_mamba_padding_cannot_separate_its_state_from_key(self):
        config, group = self.dense_fixture()
        config.kv_cache_groups[1].kv_cache_spec.page_size_padded = 80
        config.kv_cache_tensors[1].block_stride = 80
        with self.assertRaises((ValueError, NotImplementedError)):
            self.hybrid.allocate_ascend_kv_cache(
                config, "cpu", SimpleNamespace(name="LBNHC"), [2, 4], [group], True
            )

    def test_non_dense_mixed_mamba_attention_is_rejected(self):
        config, group = self.dense_fixture()
        with self.assertRaisesRegex(NotImplementedError, "dense GDN"):
            self.hybrid.allocate_ascend_kv_cache(
                config, "cpu", SimpleNamespace(name="LHBNC"), [2, 4], [group], False
            )

    def test_pure_mamba_preserves_upstream_default_layout(self):
        spec = MambaSpec()
        pages = torch.zeros(4, 1, 1, 76, dtype=torch.int8)
        self.allocate.return_value = {"m0": pages}
        config = cache_config(
            4,
            [tensor_config(pages.nbytes, ["m0"], pages.nbytes, 76)],
            [metadata_group(spec, ["m0"])],
        )
        bound, managers, planes = self.hybrid.allocate_ascend_kv_cache(
            config, "cpu", SimpleNamespace(name="LBNHC"), [4], [], False
        )
        self.assertIs(bound["m0"], pages)
        self.assertIs(managers["m0"], pages)
        self.assertEqual(planes[0].data_ptr(), pages.data_ptr())

    def test_dense_overlay_rejects_different_byte_ownership(self):
        config, group = self.dense_fixture()
        # Same total page size but a different conv/SSM split would alias
        # different scheduler block IDs across the overlaid groups.
        config.kv_cache_groups[1].kv_cache_spec.shapes = ((5,), (7,))
        with self.assertRaisesRegex(ValueError, "Overlaid Ascend dense pools"):
            self.hybrid.allocate_ascend_kv_cache(
                config, "cpu", SimpleNamespace(name="LBNHC"), [2, 4], [group], True
            )

    def test_unaligned_uint8_zero_uses_row_copy_without_index_fill(self):
        sentinel = 127
        raw = torch.full((4, 5), sentinel, dtype=torch.uint8)
        pointer = raw.data_ptr()
        calls = []
        original = torch.Tensor.copy_

        def tracked_copy(destination, values, *args, **kwargs):
            calls.append((destination.dtype, tuple(destination.shape)))
            return original(destination, values, *args, **kwargs)

        with (
            patch.object(
                torch.Tensor,
                "index_fill_",
                side_effect=AssertionError("uint8 index_fill_ is unavailable on NPU"),
            ),
            patch.object(torch.Tensor, "copy_", new=tracked_copy),
        ):
            self.hybrid.zero_block_planes([raw[:, 1:4]], [1, 3])
        self.assertEqual(calls, [(torch.uint8, (3,)), (torch.uint8, (3,))])
        self.assertEqual(raw.data_ptr(), pointer)
        self.assertTrue(torch.all(raw[[0, 2]] == sentinel).item())
        self.assertTrue(torch.all(raw[:, [0, 4]] == sentinel).item())
        self.assertEqual(torch.count_nonzero(raw[[1, 3], 1:4]).item(), 0)

    def test_aligned_offset_planes_use_bit_preserving_int32_operations(self):
        raw = torch.arange(40, dtype=torch.uint8)
        before = raw.clone()
        plane = raw[4:36].view(4, 8)
        expected = before.clone()
        expected_plane = expected[4:36].view(4, 8)
        expected_plane[1] = before[4:12]
        expected_plane[2] = before[12:20]
        expected_plane[3] = 0
        copies = []
        zeros = []
        index_copy = torch.Tensor.index_copy_
        index_fill = torch.Tensor.index_fill_

        def tracked_index_copy(destination, dim, indices, values):
            copies.append((destination.dtype, values.dtype))
            return index_copy(destination, dim, indices, values)

        def tracked_index_fill(destination, dim, indices, value):
            zeros.append(destination.dtype)
            return index_fill(destination, dim, indices, value)

        with (
            patch.object(torch.Tensor, "index_copy_", new=tracked_index_copy),
            patch.object(torch.Tensor, "index_fill_", new=tracked_index_fill),
        ):
            self.hybrid.copy_block_planes([plane], [(0, 1), (1, 2)])
            self.hybrid.zero_block_planes([plane], [3])
        self.assertEqual(copies, [(torch.int32, torch.int32)])
        self.assertEqual(zeros, [torch.int32])
        torch.testing.assert_close(raw, expected)

    def test_unaligned_offset_and_strided_planes_copy_only_requested_rows(self):
        for strided in (False, True):
            with self.subTest(strided=strided):
                raw = torch.arange(40, dtype=torch.uint8)
                before = raw.clone()
                plane = raw[:36].view(4, 9)[:, 1:8] if strided else raw[3:31].view(4, 7)
                source = (
                    before[:36].view(4, 9)[:, 1:8]
                    if strided
                    else before[3:31].view(4, 7)
                )
                expected = before.clone()
                target = (
                    expected[:36].view(4, 9)[:, 1:8]
                    if strided
                    else expected[3:31].view(4, 7)
                )
                target[1] = source[0]
                target[2] = source[1]
                target[3] = 0
                row_sizes = []
                original = torch.Tensor.copy_

                def tracked_copy(
                    destination,
                    values,
                    *args,
                    row_sizes=row_sizes,
                    original=original,
                    **kwargs,
                ):
                    row_sizes.append(destination.numel())
                    return original(destination, values, *args, **kwargs)

                with (
                    patch.object(
                        torch.Tensor,
                        "index_copy_",
                        side_effect=AssertionError("unsafe uint8 index_copy_"),
                    ),
                    patch.object(
                        torch.Tensor,
                        "index_fill_",
                        side_effect=AssertionError("unsupported uint8 index_fill_"),
                    ),
                    patch.object(torch.Tensor, "copy_", new=tracked_copy),
                ):
                    self.hybrid.copy_block_planes([plane], [(0, 1), (1, 2)])
                    self.hybrid.zero_block_planes([plane], [3])
                self.assertEqual(row_sizes, [7, 7, 7])
                torch.testing.assert_close(raw, expected)

    def test_empty_operations_are_noops(self):
        self.hybrid.copy_block_planes([], [(0, 1)])
        self.hybrid.zero_block_planes([], [0])
        self.hybrid.copy_block_planes([torch.zeros(4, 2)], [])
        self.hybrid.zero_block_planes([torch.zeros(4, 2)], [])


if __name__ == "__main__":
    unittest.main()
