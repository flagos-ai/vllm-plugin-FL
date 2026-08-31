# Copyright (c) 2026 BAAI. All rights reserved.

import os
import sys
import types
import unittest
from unittest.mock import Mock, patch

import torch

from vllm.v1.core.block_pool import BlockPool
from vllm.v1.core.single_type_kv_cache_manager import SlidingWindowManager
from vllm.v1.kv_cache_interface import FullAttentionSpec, SlidingWindowSpec
from vllm.v1.worker.utils import KVBlockZeroer

from vllm_fl.ops import cpu_qwen_gdn, cpu_qwen_runtime, cpu_w8_greedy
from vllm_fl.patches.arm_cpu_vllm_0240 import (
    _cpu_attention_scheduler_rebuild_policy,
    _install_cpu_hybrid_kv_zeroing,
    _install_cpu_spec_decode_compat,
    _zero_cpu_attention_blocks,
)


class TestCpuQwenRuntime(unittest.TestCase):
    def test_apple_arm_split_kv_metadata_is_rebuilt_without_split(self):
        scheduler = torch.zeros(96, dtype=torch.uint8)
        header = scheduler[64:88].view(torch.int32)
        header[0] = 3  # NEON ISA
        header[3] = 1  # reduction_split_num

        self.assertFalse(
            _cpu_attention_scheduler_rebuild_policy(
                scheduler,
                expected_isa=3,
                requested_split_kv=True,
                apple_arm=True,
            )
        )

        header[3] = 0
        self.assertIsNone(
            _cpu_attention_scheduler_rebuild_policy(
                scheduler,
                expected_isa=3,
                requested_split_kv=True,
                apple_arm=True,
            )
        )

    def test_runtime_registration_does_not_set_machine_policy(self):
        q4_module = types.ModuleType("flag_gems.runtime.backend._arm.q4")
        q4_module.enable_vllm_q4_codegen = Mock()
        integration_module = types.ModuleType("flag_gems.integrations.vllm")
        integration_module.maybe_install_kernel_coverage = Mock()
        compatibility = Mock()
        gdn_bridge = Mock()
        cached_greedy = Mock()

        compatibility_module = types.ModuleType(
            "vllm_fl.patches.arm_cpu_vllm_0240"
        )
        compatibility_module.install_arm_cpu_vllm_0240_compat = compatibility

        policy_names = {
            "FLAGGEMS_VENDOR",
            "TRITON_CPU_BACKEND",
            "TRITON_LOCAL_LIBOMP_PATH",
            "FLAGGEMS_GDN_TRITON_THREADS",
            "OMP_NUM_THREADS",
        }
        with (
            patch.dict(os.environ, {}, clear=True),
            patch.dict(
                sys.modules,
                {
                    compatibility_module.__name__: compatibility_module,
                    q4_module.__name__: q4_module,
                    integration_module.__name__: integration_module,
                },
            ),
            patch.object(
                cpu_qwen_gdn,
                "install_vllm_gdn_bridge",
                gdn_bridge,
            ),
            patch.object(
                cpu_w8_greedy,
                "install_w8_cached_greedy",
                cached_greedy,
            ),
            patch.object(cpu_qwen_runtime, "_ACTIVE", False),
        ):
            self.assertTrue(cpu_qwen_runtime.enable_qwen_runtime(verbose=False))
            self.assertTrue(policy_names.isdisjoint(os.environ))

        compatibility.assert_called_once_with()
        gdn_bridge.assert_called_once_with()
        cached_greedy.assert_called_once_with()
        q4_module.enable_vllm_q4_codegen.assert_called_once_with(
            verbose=False,
            runtime="libtriton_jit",
        )
        integration_module.maybe_install_kernel_coverage.assert_called_once_with()

    def test_gdn_bridge_delegates_to_flaggems(self):
        integration = types.ModuleType("flag_gems.integrations.vllm")
        integration.install_qwen_gdn = Mock()
        with patch.dict(sys.modules, {integration.__name__: integration}):
            cpu_qwen_gdn.install_vllm_gdn_bridge()
        integration.install_qwen_gdn.assert_called_once_with()

    def test_hybrid_cpu_kv_zeroing_delegates_to_vllm_zeroer(self):
        runner = types.SimpleNamespace(_kv_block_zeroer=Mock())
        _zero_cpu_attention_blocks(runner, [2, 7])
        runner._kv_block_zeroer.zero_block_ids.assert_called_once_with([2, 7])

    def test_hybrid_cpu_kv_zeroing_tolerates_uninitialized_zeroer(self):
        _zero_cpu_attention_blocks(types.SimpleNamespace(), [2, 7])

    def test_cpu_kv_zeroer_clears_requested_host_page(self):
        _install_cpu_hybrid_kv_zeroing()
        cache = torch.full((2, 3, 1, 4, 2), 7, dtype=torch.float16)
        backend = Mock()
        backend.get_kv_cache_block_dim.return_value = 1
        group = types.SimpleNamespace(
            kv_cache_spec=FullAttentionSpec(
                block_size=4,
                num_kv_heads=1,
                head_size=2,
                dtype=torch.float16,
            ),
            kv_cache_group_id=0,
            backend=backend,
            layer_names=["target.attn"],
        )
        zeroer = KVBlockZeroer(
            torch.device("cpu"),
            pin_memory=False,
            attn_groups_iter=[group],
            kernel_block_sizes=[4],
            cache_dtype="auto",
            runner_only_attn_layers=set(),
            static_forward_context={
                "target.attn": types.SimpleNamespace(kv_cache=cache)
            },
        )

        zeroer.zero_block_ids([1])

        torch.testing.assert_close(cache[:, 0], torch.full_like(cache[:, 0], 7))
        torch.testing.assert_close(cache[:, 1], torch.zeros_like(cache[:, 1]))
        torch.testing.assert_close(cache[:, 2], torch.full_like(cache[:, 2], 7))

    def test_cpu_kv_zeroer_includes_sliding_window_attention(self):
        _install_cpu_hybrid_kv_zeroing()
        cache = torch.full((2, 3, 1, 4, 2), 7, dtype=torch.float16)
        backend = Mock()
        backend.get_kv_cache_block_dim.return_value = 1
        group = types.SimpleNamespace(
            kv_cache_spec=SlidingWindowSpec(
                block_size=4,
                num_kv_heads=1,
                head_size=2,
                dtype=torch.float16,
                sliding_window=16,
            ),
            kv_cache_group_id=0,
            backend=backend,
            layer_names=["draft.attn"],
        )
        context = {
            "draft.attn": types.SimpleNamespace(kv_cache=cache),
        }
        zeroer = KVBlockZeroer(
            torch.device("cpu"),
            pin_memory=False,
            attn_groups_iter=[group],
            kernel_block_sizes=[4],
            cache_dtype="auto",
            runner_only_attn_layers=set(),
            static_forward_context=context,
        )
        zeroer.zero_block_ids([1])

        torch.testing.assert_close(cache[:, 0], torch.full_like(cache[:, 0], 7))
        torch.testing.assert_close(cache[:, 1], torch.zeros_like(cache[:, 1]))
        torch.testing.assert_close(cache[:, 2], torch.full_like(cache[:, 2], 7))

    def test_sliding_window_manager_reports_new_pages_for_zeroing(self):
        spec = SlidingWindowSpec(
            block_size=4,
            num_kv_heads=1,
            head_size=2,
            dtype=torch.float16,
            sliding_window=16,
        )
        pool = BlockPool(
            num_gpu_blocks=8,
            enable_caching=False,
            hash_block_size=4,
        )
        manager = SlidingWindowManager(
            spec,
            block_pool=pool,
            enable_caching=False,
            kv_cache_group_id=0,
            scheduler_block_size=4,
            max_admission_blocks_per_request=8,
        )

        blocks = manager.allocate_new_blocks(
            request_id="request",
            num_tokens=4,
            num_tokens_main_model=4,
        )

        self.assertEqual(
            manager.take_new_block_ids(),
            [block.block_id for block in blocks],
        )

    def test_cpu_spec_decode_compat_accepts_current_greedy_abi(self):
        import vllm.utils.cpu_triton_utils as cpu_tl

        kernel = cpu_tl.rejection_greedy_sample_kernel
        random_kernel = cpu_tl.rejection_random_sample_kernel
        original = kernel.func
        original_random = random_kernel.func
        legacy = Mock()
        legacy._vllm_fl_spec_decode_abi = False
        kernel.func = legacy
        try:
            _install_cpu_spec_decode_compat()
            args = [object() for _ in range(7)]
            kernel.func(
                *args,
                object(),
                object(),
                SYNTHETIC_MODE=False,
            )
            legacy.assert_called_once_with(*args)
            with self.assertRaisesRegex(NotImplementedError, "synthetic"):
                kernel.func(
                    *args,
                    None,
                    None,
                    SYNTHETIC_MODE=True,
                )
        finally:
            kernel.func = original
            random_kernel.func = original_random

    def test_cached_greedy_accepts_only_safe_processor_state(self):
        LogitBias = type("LogitBiasLogitsProcessor", (), {})
        MinTokens = type("MinTokensLogitsProcessor", (), {})
        ThinkingBudget = type("ThinkingTokenBudgetLogitsProcessor", (), {})
        bias = LogitBias()
        bias.biases = {}
        min_tokens = MinTokens()
        min_tokens.min_toks = {0: (128, [], {1, 2})}
        thinking = ThinkingBudget()
        thinking.is_enabled = False
        processors = types.SimpleNamespace(
            non_argmax_invariant=[bias, min_tokens, thinking]
        )
        metadata = types.SimpleNamespace(
            logitsprocs=processors,
            temperature=None,
            all_random=False,
            all_greedy=True,
            max_num_logprobs=None,
            logprob_token_ids={},
            allowed_token_ids_mask=None,
            bad_words_token_ids=[],
            no_penalties=True,
        )
        logits = torch.zeros((1, 32), dtype=torch.bfloat16)
        with patch.object(cpu_w8_greedy, "_ENABLED", True):
            self.assertTrue(cpu_w8_greedy._eligible(logits, metadata))
            self.assertEqual(
                cpu_w8_greedy._masked_stop_tokens(metadata), {1, 2}
            )
            min_tokens.min_toks[1] = (129, [], {3})
            self.assertEqual(
                cpu_w8_greedy._all_masked_stop_tokens(metadata), {1, 2, 3}
            )
            bias.biases = {0: {5: 1.0}}
            self.assertFalse(cpu_w8_greedy._eligible(logits, metadata))

    def test_cached_greedy_recognizes_zero_temperature_mixed_batch(self):
        metadata = types.SimpleNamespace(
            logitsprocs=types.SimpleNamespace(non_argmax_invariant=[]),
            temperature=torch.tensor([0.0]),
            all_random=False,
            all_greedy=False,
            max_num_logprobs=None,
            logprob_token_ids={},
            allowed_token_ids_mask=None,
            bad_words_token_ids=[],
            no_penalties=True,
        )
        logits = torch.zeros((1, 32), dtype=torch.bfloat16)
        with patch.object(cpu_w8_greedy, "_ENABLED", True):
            self.assertTrue(cpu_w8_greedy._eligible(logits, metadata))
            metadata.temperature.fill_(0.5)
            self.assertFalse(cpu_w8_greedy._eligible(logits, metadata))

    def test_cached_spec_greedy_requires_exact_mtp1_semantics(self):
        sampling = types.SimpleNamespace(
            logitsprocs=types.SimpleNamespace(non_argmax_invariant=[]),
            temperature=torch.tensor([0.0]),
            all_random=False,
            all_greedy=True,
            max_num_logprobs=None,
            logprob_token_ids={},
            allowed_token_ids_mask=None,
            bad_words_token_ids=[],
            no_penalties=True,
        )
        metadata = types.SimpleNamespace(
            max_spec_len=1,
            draft_token_ids=torch.tensor([7], dtype=torch.int32),
            num_draft_tokens=[1],
            target_logits_indices=torch.tensor([0]),
            bonus_logits_indices=torch.tensor([1]),
        )
        logits = torch.zeros((2, 32), dtype=torch.bfloat16)
        with patch.object(cpu_w8_greedy, "_ENABLED", True):
            self.assertTrue(
                cpu_w8_greedy._spec_eligible(logits, metadata, sampling)
            )
            metadata.max_spec_len = 2
            self.assertFalse(
                cpu_w8_greedy._spec_eligible(logits, metadata, sampling)
            )
            metadata.max_spec_len = 1
            MinTokens = type("MinTokensLogitsProcessor", (), {})
            sampling.logitsprocs.non_argmax_invariant = [MinTokens()]
            self.assertTrue(
                cpu_w8_greedy._spec_eligible(logits, metadata, sampling)
            )
            Unsupported = type("UnsupportedLogitsProcessor", (), {})
            sampling.logitsprocs.non_argmax_invariant = [Unsupported()]
            self.assertFalse(
                cpu_w8_greedy._spec_eligible(logits, metadata, sampling)
            )

    def test_multi_spec_greedy_reduces_bf16_rows_without_fp32_copy(self):
        sampling = types.SimpleNamespace(
            logitsprocs=types.SimpleNamespace(non_argmax_invariant=[]),
            temperature=torch.tensor([0.0]),
            all_random=False,
            all_greedy=True,
            max_num_logprobs=None,
            logprob_token_ids={},
            allowed_token_ids_mask=None,
            bad_words_token_ids=[],
            no_penalties=True,
        )
        metadata = types.SimpleNamespace(
            max_spec_len=3,
            draft_token_ids=torch.tensor([2, 4, 1], dtype=torch.int32),
            num_draft_tokens=[3],
            target_logits_indices=torch.tensor([0, 1, 2]),
            bonus_logits_indices=torch.tensor([3]),
        )
        logits = torch.full((4, 8), -4, dtype=torch.bfloat16)
        logits[0, 2] = 5
        logits[1, 4] = 5
        logits[2, 6] = 5
        logits[3, 7] = 5
        with (
            patch.dict(
                os.environ,
                {"VLLM_FL_FAST_GREEDY_SPEC_REJECTION": "1"},
            ),
            patch.object(cpu_w8_greedy, "_ENABLED", True),
            patch.object(cpu_w8_greedy, "_SPEC_ENABLED", True),
        ):
            self.assertTrue(
                cpu_w8_greedy._multi_spec_eligible(
                    logits, metadata, sampling
                )
            )
            sampled = cpu_w8_greedy._multi_spec_greedy_sample(
                logits, metadata, sampling
            )
            self.assertEqual(sampled.tolist(), [[2, 4, 6, -1]])

            metadata.draft_token_ids[2] = 6
            sampled = cpu_w8_greedy._multi_spec_greedy_sample(
                logits, metadata, sampling
            )
            self.assertEqual(sampled.tolist(), [[2, 4, 6, 7]])

            MinTokens = type("MinTokensLogitsProcessor", (), {})
            min_tokens = MinTokens()
            min_tokens.min_toks = {0: (128, [], {7})}
            sampling.logitsprocs.non_argmax_invariant = [min_tokens]
            self.assertIsNone(
                cpu_w8_greedy._multi_spec_greedy_sample(
                    logits, metadata, sampling
                )
            )


if __name__ == "__main__":
    unittest.main()
