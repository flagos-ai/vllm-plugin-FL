# Copyright (c) 2026 BAAI. All rights reserved.

import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch

import vllm_fl


def _fake_cuda():
    """Stand-in for the torch.cuda namespace the patch binds from."""
    return SimpleNamespace(
        empty_cache=lambda: None,
        memory_reserved=lambda device=None: 0,
        memory_stats=lambda device=None: {},
        memory_allocated=lambda device=None: 0,
        max_memory_allocated=lambda device=None: 0,
        reset_peak_memory_stats=lambda device=None: None,
    )


class TestMetaxAcceleratorPatch(unittest.TestCase):
    def _run_patch(self, vendor_name, accel):
        with (
            patch("flag_gems.vendor_name", vendor_name),
            patch.object(torch, "accelerator", accel, create=True),
            patch.object(torch, "cuda", _fake_cuda(), create=True),
        ):
            vllm_fl._patch_torch_accelerator()

    def test_non_metax_vendor_accelerator_is_left_alone(self):
        for vendor_name in ("cuda", "ascend", "musa"):
            accel = SimpleNamespace()
            self._run_patch(vendor_name, accel)
            self.assertFalse(hasattr(accel, "memory_stats"), vendor_name)
            self.assertFalse(hasattr(accel, "empty_cache"), vendor_name)

    def test_metax_vendor_gets_the_missing_memory_api(self):
        accel = SimpleNamespace()
        self._run_patch("metax", accel)
        for name in (
            "empty_cache",
            "memory_reserved",
            "memory_stats",
            "memory_allocated",
            "max_memory_allocated",
            "reset_peak_memory_stats",
        ):
            self.assertTrue(hasattr(accel, name), name)

    def test_metax_accelerator_with_memory_stats_is_not_overwritten(self):
        def sentinel(device=None):
            return None

        accel = SimpleNamespace(
            memory_stats=lambda device=None: {}, empty_cache=sentinel
        )
        self._run_patch("metax", accel)
        self.assertIs(accel.empty_cache, sentinel)


if __name__ == "__main__":
    unittest.main()
