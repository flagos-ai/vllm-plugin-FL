"""MLA configuration and metadata contracts without accelerator imports."""

import importlib.util
import sys
import unittest
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import patch


class TheadMLAPrefillTests(unittest.TestCase):
    def setUp(self):
        self.config = SimpleNamespace(
            attention_config=SimpleNamespace(mla_prefill_backend=None)
        )
        self.registered = []
        config = ModuleType("vllm.config")
        config.get_current_vllm_config = lambda: self.config
        registry = ModuleType("vllm.v1.attention.backends.mla.prefill.registry")
        registry.MLAPrefillBackendEnum = SimpleNamespace(CUSTOM="custom")
        registry.register_mla_prefill_backend = lambda *args: self.registered.append(
            args
        )
        base = ModuleType("vllm.v1.attention.backends.mla.prefill.base")
        base.MLAPrefillBackend = type("Prefill", (), {})
        triton = ModuleType("vllm.v1.attention.backends.mla.triton_mla")
        triton.TritonMLABackend = type("Backend", (), {})
        triton.TritonMLAImpl = type("Impl", (), {})
        torch = ModuleType("torch")
        torch.ops = SimpleNamespace(
            _C_cache_ops=SimpleNamespace(gather_and_maybe_dequant_cache=object())
        )
        modules = patch.dict(
            sys.modules,
            {
                "vllm.config": config,
                registry.__name__: registry,
                base.__name__: base,
                triton.__name__: triton,
                "torch": torch,
            },
        )
        modules.start()
        self.addCleanup(modules.stop)
        path = (
            Path(__file__).resolve().parents[2]
            / "vllm_fl/dispatch/backends/vendor/thead/impl/mla.py"
        )
        spec = importlib.util.spec_from_file_location("thead_mla_contract", path)
        self.module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(self.module)

    def test_default_prefill_is_registered_before_native_initialization(self):
        self.module.TheadMLAImpl()
        self.assertEqual(self.config.attention_config.mla_prefill_backend, "custom")
        self.assertEqual(
            self.registered, [("custom", "thead_mla_contract.TheadMLAPrefillBackend")]
        )

    def test_explicit_user_prefill_is_preserved(self):
        self.config.attention_config.mla_prefill_backend = "explicit-user-backend"
        self.module.TheadMLAImpl()
        self.assertEqual(
            self.config.attention_config.mla_prefill_backend, "explicit-user-backend"
        )
        self.assertEqual(self.registered, [])

    def test_missing_native_gather_uses_portable_reader(self):
        torch = sys.modules["torch"]
        torch.ops._C_cache_ops = SimpleNamespace()
        vllm = ModuleType("vllm")
        ops = ModuleType("vllm._custom_ops")
        vllm._custom_ops = ops
        helper = ModuleType("vllm_fl.dispatch.backends.mla_cache")
        helper.torch_gather_and_maybe_dequant_cache = object()
        with patch.dict(
            sys.modules,
            {"vllm": vllm, "vllm._custom_ops": ops, helper.__name__: helper},
        ):
            self.module.TheadMLAImpl()
        self.assertIs(
            ops.gather_and_maybe_dequant_cache,
            helper.torch_gather_and_maybe_dequant_cache,
        )

    def prefill(self):
        instance = object.__new__(self.module.TheadMLAPrefillBackend)
        instance.scale = 0.125
        self.calls = []
        instance.flash_attn_varlen_func = lambda **kwargs: (
            self.calls.append(kwargs) or "result"
        )
        instance._prefill_metadata = SimpleNamespace(
            query_start_loc="new-starts",
            max_query_len=5,
            chunked_context=SimpleNamespace(
                cu_seq_lens=["cached-starts"], max_seq_lens=[31]
            ),
        )
        return instance

    def test_new_tokens_use_causal_attention_and_caller_output(self):
        instance = self.prefill()
        instance.run_prefill_new_tokens("q", "k", "v", True, out="output")
        call = self.calls[0]
        self.assertTrue(call["causal"])
        self.assertTrue(call["return_softmax_lse"])
        self.assertEqual(call["cu_seqlens_k"], "new-starts")
        self.assertEqual(call["out"], "output")

    def test_cached_context_uses_its_own_offsets_and_lse(self):
        instance = self.prefill()
        instance.run_prefill_context_chunk(0, "q", "k", "v")
        call = self.calls[0]
        self.assertFalse(call["causal"])
        self.assertTrue(call["return_softmax_lse"])
        self.assertEqual(call["cu_seqlens_q"], "new-starts")
        self.assertEqual(call["cu_seqlens_k"], "cached-starts")
        self.assertEqual(call["max_seqlen_k"], 31)

    def test_quantized_output_is_rejected_before_kernel_execution(self):
        instance = self.prefill()
        with self.assertRaises(NotImplementedError):
            instance.run_prefill_new_tokens("q", "k", "v", False, output_scale="scale")
        self.assertEqual(self.calls, [])

    def test_cache_update_uses_public_writer_and_normalizes_layout(self):
        instance = object.__new__(self.module.TheadMLAImpl)
        calls = []
        gems = ModuleType("flag_gems")
        gems.concat_and_cache_mla = lambda *args: calls.append(args)
        rope = SimpleNamespace(squeeze=lambda dim: "squeezed-rope")
        slots = SimpleNamespace(flatten=lambda: "flat-slots")
        cache = SimpleNamespace(numel=lambda: 32)
        with patch.dict(sys.modules, {"flag_gems": gems}):
            instance.do_kv_cache_update("latent", rope, cache, slots, "auto", "scale")
        self.assertEqual(
            calls, [("latent", "squeezed-rope", cache, "flat-slots", "auto", "scale")]
        )

    def test_empty_profile_cache_does_not_call_writer(self):
        instance = object.__new__(self.module.TheadMLAImpl)
        instance.do_kv_cache_update(
            None, None, SimpleNamespace(numel=lambda: 0), None, "auto", None
        )


if __name__ == "__main__":
    unittest.main()
