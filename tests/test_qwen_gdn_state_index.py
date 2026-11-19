"""CPU-only boundaries, aliasing and installation checks for the new workaround."""

import importlib.util
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch

source = Path(__file__).resolve().parents[1] / "vllm_fl/patches/qwen_gdn_state_index.py"
spec = importlib.util.spec_from_file_location("state_index_tested", source)
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)


class Input:
    def __init__(
        self,
        shape=(256, 6, 128, 128),
        strides=(98304, 16384, 128, 1),
        dtype=torch.float32,
        device="cuda",
    ):
        self.shape = shape
        self._strides = strides
        self.ndim = len(shape)
        self.dtype = dtype
        self.device = SimpleNamespace(type=device)

    def stride(self):
        return self._strides


class Index:
    def __init__(self, dtype=torch.int32):
        self.dtype = dtype
        self.ndim = 1
        self.values = [1, 2, 3]
        self.conversions = []

    def to(self, *, dtype):
        self.conversions.append(dtype)
        new = Index(dtype)
        new.values = list(self.values)
        return new


class Tests(unittest.TestCase):
    def test_small_is_identity(self):
        ids = [Index()]
        self.assertIs(m.widen_state_indices(Input(), ids), ids)
        self.assertEqual(ids[0].conversions, [])

    def test_exact_signed32_boundary_is_identity(self):
        ids = [Index()]
        x = Input((32768, 1, 1, 65536), (65536, 65536, 65536, 1))
        self.assertIs(m.widen_state_indices(x, ids), ids)

    def test_large_widens_without_mutating_indices(self):
        ids = [Index()]
        x = Input((32768, 6, 128, 128))
        out = m.widen_state_indices(x, ids)
        self.assertIsNot(out, ids)
        self.assertEqual(out[0].dtype, torch.int64)
        self.assertEqual(out[0].values, ids[0].values)
        self.assertEqual(ids[0].dtype, torch.int32)

    def test_strided_offset_uses_layout_not_numel(self):
        ids = [Index()]
        out = m.widen_state_indices(
            Input((4096, 6, 128, 128), (1000000, 16384, 128, 1)), ids
        )
        self.assertEqual(out[0].dtype, torch.int64)

    def test_other_index_and_value_dtypes_keep_identity(self):
        for dtype in [torch.int64, torch.int16]:
            ids = [Index(dtype)]
            self.assertIs(m.widen_state_indices(Input((32768, 6, 128, 128)), ids), ids)
        ids = [Index()]
        self.assertIs(
            m.widen_state_indices(
                Input((32768, 6, 128, 128), dtype=torch.bfloat16), ids
            ),
            ids,
        )

    def test_cpu_empty_and_other_ranks_keep_identity(self):
        ids = [Index()]
        for x in [
            Input(device="cpu"),
            Input((0, 6, 128, 128)),
            Input((32768, 128, 128), (16384, 128, 1)),
        ]:
            self.assertIs(m.widen_state_indices(x, ids), ids)

    def test_non_target_architecture_does_not_import_gpu_module(self):
        with patch.object(m.importlib, "import_module") as imported:
            self.assertFalse(
                m.patch_qwen_gdn_state_indices(
                    SimpleNamespace(hf_config=SimpleNamespace(model_type="deepseek_v4"))
                )
            )
            imported.assert_not_called()

    def test_install_idempotence_and_forward_aliasing(self):
        calls = []

        class Function:
            def __call__(self, *args):
                calls.append(args)
                return args[0]

        module = SimpleNamespace(IndexPutFunction=Function)
        config = SimpleNamespace(hf_config=SimpleNamespace(model_type="qwen4_exp"))
        with patch.object(m.importlib, "import_module", return_value=module):
            self.assertTrue(m.patch_qwen_gdn_state_indices(config))
            self.assertFalse(m.patch_qwen_gdn_state_indices(config))
        x = Input((32768, 6, 128, 128))
        ids = [Index()]
        values = object()
        result = module.IndexPutFunction()(x, ids, values, False)
        self.assertIs(result, x)
        self.assertIs(calls[0][2], values)
        self.assertEqual(calls[0][1][0].dtype, torch.int64)
        self.assertEqual(ids[0].dtype, torch.int32)


if __name__ == "__main__":
    unittest.main(verbosity=2)
