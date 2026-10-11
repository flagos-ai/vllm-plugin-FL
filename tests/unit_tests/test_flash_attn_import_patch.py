# Copyright (c) 2026 BAAI. All rights reserved.

import builtins
import os
import sys
import unittest
from contextlib import contextmanager
from types import ModuleType
from unittest.mock import Mock, patch

import vllm_fl

_PARENT = "vllm.vllm_flash_attn"
_INTERFACE = f"{_PARENT}.flash_attn_interface"
_OTHER_CHILD = f"{_PARENT}.other_child"
_MISSING = object()


@contextmanager
def _preserve_modules(*names):
    original = {name: sys.modules.get(name, _MISSING) for name in names}
    try:
        yield
    finally:
        for name, module in original.items():
            if module is _MISSING:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = module


class TestFlashAttnImportPatch(unittest.TestCase):
    def test_failed_probe_removes_orphan_and_installs_fallback(self):
        orphan = ModuleType(_INTERFACE)
        unrelated_child = ModuleType(_OTHER_CHILD)
        real_import = builtins.__import__
        probe_count = 0

        def import_with_failed_flash_attn_probe(
            name, globals=None, locals=None, fromlist=(), level=0
        ):
            nonlocal probe_count
            if name == _PARENT:
                probe_count += 1
                # Reproduce vLLM's failed package import: the parent is removed
                # while its successfully imported interface remains cached.
                sys.modules[_INTERFACE] = orphan
                raise ImportError("_vllm_fa2_C is unavailable")
            return real_import(name, globals, locals, fromlist, level)

        with (
            _preserve_modules(_PARENT, _INTERFACE, _OTHER_CHILD),
            patch(
                "builtins.__import__",
                side_effect=import_with_failed_flash_attn_probe,
            ),
        ):
            sys.modules.pop(_PARENT, None)
            sys.modules.pop(_INTERFACE, None)
            sys.modules[_OTHER_CHILD] = unrelated_child

            vllm_fl._patch_flash_attn_import()

            fallback = sys.modules[_PARENT]
            self.assertIsInstance(fallback, ModuleType)
            self.assertFalse(fallback.FA2_AVAILABLE)
            self.assertFalse(fallback.FA3_AVAILABLE)
            self.assertFalse(fallback.is_fa_version_supported(2))
            self.assertIsNone(fallback.flash_attn_varlen_func)
            self.assertIsNone(fallback.get_scheduler_metadata)
            self.assertIn(
                "not available",
                fallback.fa_version_unsupported_reason(2),
            )
            self.assertNotIn(_INTERFACE, sys.modules)
            self.assertIs(sys.modules[_OTHER_CHILD], unrelated_child)

            # The installed fallback makes subsequent calls idempotent and
            # prevents another probe of the unavailable CUDA extension.
            vllm_fl._patch_flash_attn_import()
            self.assertIs(sys.modules[_PARENT], fallback)
            self.assertEqual(probe_count, 1)

    def test_successful_probe_preserves_loaded_modules(self):
        parent = ModuleType(_PARENT)
        interface = ModuleType(_INTERFACE)
        real_import = builtins.__import__

        def import_with_successful_flash_attn_probe(
            name, globals=None, locals=None, fromlist=(), level=0
        ):
            if name == _PARENT:
                sys.modules[_PARENT] = parent
                sys.modules[_INTERFACE] = interface
            return real_import(name, globals, locals, fromlist, level)

        with (
            _preserve_modules(_PARENT, _INTERFACE),
            patch(
                "builtins.__import__",
                side_effect=import_with_successful_flash_attn_probe,
            ),
        ):
            sys.modules.pop(_PARENT, None)
            sys.modules.pop(_INTERFACE, None)

            vllm_fl._patch_flash_attn_import()

            self.assertIs(sys.modules[_PARENT], parent)
            self.assertIs(sys.modules[_INTERFACE], interface)


@unittest.skipUnless(os.environ.get("FL_TEST_PLATFORM") == "enflame", "Enflame only")
class TestGCUVisionAttention(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import torch

        import vllm.model_executor.layers.attention.mm_encoder_attention as mm
        from vllm.model_executor.custom_op import CustomOp
        from vllm.v1.attention.backends.registry import AttentionBackendEnum

        from vllm_fl.platform import PlatformFL

        cls.torch, cls.mm, cls.backend, cls.platform = (
            torch,
            mm,
            AttentionBackendEnum,
            PlatformFL,
        )
        cls.original_methods = (
            getattr(
                mm.MMEncoderAttention,
                "_fl_gcu_original_forward_native",
                mm.MMEncoderAttention.forward_native,
            ),
            CustomOp.forward_oot,
        )

    def setUp(self):
        from vllm.config import VllmConfig, set_current_vllm_config

        platform = self.platform()
        self.enterContext(patch.object(self.platform, "vendor_name", "enflame"))
        for module in (
            "vllm.platforms",
            "vllm.model_executor.custom_op",
            "vllm.model_executor.models.vision",
        ):
            self.enterContext(patch(f"{module}.current_platform", platform))
        self.config = VllmConfig()
        self.config.compilation_config.custom_ops = ["all"]
        self.enterContext(set_current_vllm_config(self.config))
        for name, method in zip(
            ("forward_native", "forward_oot"), self.original_methods
        ):
            self.enterContext(patch.object(self.mm.MMEncoderAttention, name, method))
        self.enterContext(
            patch.object(self.mm, "get_flash_attn_version", return_value=2)
        )
        self.enterContext(
            patch.object(
                self.torch, "get_default_dtype", return_value=self.torch.float16
            )
        )

    def _provider(self):
        provider = ModuleType("flash_attn.vllm_flash_attn")
        provider.flash_attn_varlen_func = Mock(side_effect=lambda **kw: kw["q"].clone())
        self.enterContext(patch.dict(sys.modules, {provider.__name__: provider}))
        return provider.flash_attn_varlen_func

    def test_default_and_explicit_backend_selection_avoid_cuda_capability(self):
        with patch.object(
            self.platform, "get_device_capability", side_effect=AssertionError
        ):
            self.assertEqual(
                self.platform.get_vit_attn_backend(64, self.torch.float16),
                self.backend.FLASH_ATTN,
            )
            self.assertEqual(
                self.platform.get_vit_attn_backend(64, self.torch.float32),
                self.backend.TORCH_SDPA,
            )
            self.assertEqual(
                self.platform.get_vit_attn_backend(
                    64, self.torch.float32, self.backend.TORCH_SDPA
                ),
                self.backend.TORCH_SDPA,
            )
            for head_size in (0, 7, 264):
                with self.subTest(head_size=head_size), self.assertRaises(ValueError):
                    self.platform.get_vit_attn_backend(
                        head_size, self.torch.float16, self.backend.FLASH_ATTN
                    )

    def test_first_instance_keeps_old_oot_binding_but_calls_native_provider(self):
        provider = self._provider()
        attention = self.mm.MMEncoderAttention(4, 64)
        self.assertIs(attention._forward_method.__func__, self.original_methods[1])
        query = self.torch.zeros(1, 2, 256, dtype=self.torch.float16)
        self.assertEqual(attention(query, query, query).shape, query.shape)
        provider.assert_called_once()

    def test_first_disabled_instance_is_patched_before_native_binding(self):
        provider = self._provider()
        self.config.compilation_config.custom_ops = ["none"]
        self.platform.check_and_update_config(self.config)
        attention = self.mm.MMEncoderAttention(4, 64)
        self.assertIs(
            attention._forward_method.__func__, attention.forward_native.__func__
        )
        self.assertIsNot(attention._forward_method.__func__, self.original_methods[0])
        query = self.torch.zeros(1, 2, 256, dtype=self.torch.float16)
        self.assertEqual(attention(query, query, query).shape, query.shape)
        provider.assert_called_once()

    def test_explicit_sdpa_routes_without_calling_native_provider(self):
        provider = self._provider()
        attention = self.mm.MMEncoderAttention(4, 64)
        attention.attn_backend = self.backend.TORCH_SDPA
        query = self.torch.zeros(1, 2, 256)
        with patch.object(attention, "_forward_sdpa", return_value=query) as sdpa:
            self.assertIs(attention(query, query, query), query)
            sdpa.assert_called_once_with(query, query, query, None)
        provider.assert_not_called()

    def test_batched_3d_and_4d_gqa_preserve_shapes_and_separate_lengths(self):
        provider = self._provider()
        attention = self.mm.MMEncoderAttention(4, 64, num_kv_heads=2)
        for flattened in (False, True):
            with self.subTest(flattened=flattened):
                query = self.torch.zeros(2, 3, 4, 64, dtype=self.torch.float16)
                key = self.torch.zeros(2, 5, 2, 64, dtype=self.torch.float16)
                if flattened:
                    query, key = query.flatten(2), key.flatten(2)
                self.assertEqual(attention(query, key, key).shape, query.shape)
                kwargs = provider.call_args.kwargs
                self.assertEqual(kwargs["q"].shape, (6, 4, 64))
                self.assertEqual(kwargs["k"].shape, (10, 2, 64))
                self.assertEqual(kwargs["cu_seqlens_q"].tolist(), [0, 3, 6])
                self.assertEqual(kwargs["cu_seqlens_k"].tolist(), [0, 5, 10])
                self.assertEqual(
                    (kwargs["max_seqlen_q"], kwargs["max_seqlen_k"]), (3, 5)
                )
                self.assertEqual(
                    (kwargs["dropout_p"], kwargs["causal"], kwargs["fa_version"]),
                    (0.0, False, 2),
                )
                self.assertEqual(kwargs["softmax_scale"], attention.scale)

    def test_packed_lengths_are_passed_unchanged_and_require_max_length(self):
        provider = self._provider()
        attention = self.mm.MMEncoderAttention(4, 64)
        query = self.torch.zeros(1, 8, 256, dtype=self.torch.float16)
        cu = self.torch.tensor([0, 3, 8], dtype=self.torch.int32)
        maximum = self.torch.tensor(5)
        with patch("vllm.utils.gpu_sync_debug.gpu_sync_allowed") as sync:
            self.assertEqual(
                attention(query, query, query, cu, maximum).shape, query.shape
            )
            sync.return_value.__enter__.assert_called_once()
        kwargs = provider.call_args.kwargs
        self.assertIs(kwargs["cu_seqlens_q"], cu)
        self.assertIs(kwargs["cu_seqlens_k"], cu)
        self.assertEqual((kwargs["max_seqlen_q"], kwargs["max_seqlen_k"]), (5, 5))
        for args in ((cu, None), (None, maximum)):
            with self.subTest(args=args), self.assertRaises(AssertionError):
                attention(query, query, query, *args)
        self.assertEqual(provider.call_count, 1)


if __name__ == "__main__":
    unittest.main()
