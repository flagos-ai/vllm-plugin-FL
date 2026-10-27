"""Check legacy direct assignment and hook-based QSA cache ownership on CPU."""

import ast
import unittest
import weakref
from pathlib import Path

import torch

SOURCE = (
    Path(__file__).resolve().parents[2]
    / "vllm_fl/models/qwen3_8_flash_next/common/qsa_cache.py"
)


def cache_type():
    tree = ast.parse(SOURCE.read_text())
    cls = next(
        n
        for n in tree.body
        if isinstance(n, ast.ClassDef) and n.name == "QSAKeyStateCache"
    )

    class StateCache:
        def __init__(self, head_size, **kwargs):
            self.head_size = head_size
            self.kv_cache = torch.tensor([])

        def bind_kv_cache(self, tensor):
            self.kv_cache = tensor

    code = ast.Module(
        body=[
            ast.ImportFrom(
                module="__future__", names=[ast.alias(name="annotations")], level=0
            ),
            cls,
        ],
        type_ignores=[],
    )
    ast.fix_missing_locations(code)
    scope = {"torch": torch, "_QSAStateCache": StateCache}
    exec(compile(code, str(SOURCE), "exec"), scope)
    return scope["QSAKeyStateCache"]


class QSACacheBindingTest(unittest.TestCase):
    def test_legacy_assignment_views_track_rebinding_and_padding(self):
        cache = cache_type()(head_size=128, cache_rope_positions=True)
        storage = torch.zeros(3, 4, 1, 160, dtype=torch.bfloat16)
        bound = storage[..., :140]
        cache.kv_cache = bound  # Exact vLLM 0.24 binding ABI.
        self.assertEqual(cache.key_cache.shape, (3, 4, 1, 128))
        self.assertEqual(cache.rope_position_cache.shape, (3, 4, 1, 3))
        cache.key_cache[1, 2, 0, 7] = 5
        self.assertEqual(storage[1, 2, 0, 7].item(), 5)
        positions = torch.tensor([2**40 + 1, 123, -7], dtype=torch.int64)
        cache.rope_position_cache[1, 2, 0].copy_(positions)
        torch.testing.assert_close(
            storage[1, 2, 0, 128:140].view(torch.int64), positions
        )
        self.assertEqual(cache.key_cache.stride(0), storage.stride(0))
        previous = weakref.ref(bound)
        replacement = torch.zeros_like(storage)[..., :140]
        cache.kv_cache = replacement
        del bound
        self.assertIsNone(previous())
        self.assertEqual(cache.key_cache.data_ptr(), replacement.data_ptr())
        self.assertEqual(cache.key_cache[1, 2, 0, 7].item(), 0)
        cache.kv_cache = torch.tensor([])
        self.assertEqual(cache.key_cache.numel(), 0)
        self.assertEqual(cache.rope_position_cache.numel(), 0)

    def test_optional_hook_validates_and_text_cache_has_no_positions(self):
        cache = cache_type()(head_size=128, cache_rope_positions=False)
        data = torch.zeros(2, 4, 1, 128, dtype=torch.bfloat16)
        cache.bind_kv_cache(data)
        self.assertIsNone(cache.rope_position_cache)
        self.assertEqual(cache.key_cache.data_ptr(), data.data_ptr())
        with self.assertRaisesRegex(ValueError, "packed BF16"):
            cache.bind_kv_cache(data.float())
        with self.assertRaisesRegex(ValueError, "QSA raw cache"):
            cache.bind_kv_cache(data.squeeze(2))


if __name__ == "__main__":
    unittest.main()
