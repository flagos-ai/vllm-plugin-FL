from types import SimpleNamespace

import torch

from vllm.v1.sample.ops.topk_topp_sampler import apply_top_k_top_p_pytorch
from vllm.v1.sample.rejection_sampler import (
    PLACEHOLDER_TOKEN_ID,
    apply_sampling_constraints,
    expand_batch_to_tokens,
    rejection_sample,
    sample_recovered_tokens,
)

from vllm_fl.patches.arm_cpu_vllm_0240 import (
    _apply_top_k_top_p_small_k_cpu,
    _install_cpu_spec_decode_compat,
    _install_cpu_spec_decode_kernels,
)


def setup_module() -> None:
    # Direct unit imports bypass platform-plugin initialization. Install the
    # same official CPU wrappers used by the real vLLM 0.24 engine so these
    # tests do not compile the generic GPU-oriented Triton control kernels.
    _install_cpu_spec_decode_kernels()
    _install_cpu_spec_decode_compat()


def test_topk_vocab_size_disables_filter_with_top_p() -> None:
    """vLLM 0.24 represents public top_k=0 as the vocabulary size."""
    logits = torch.tensor([[0.1, 0.2, 0.3, 0.4]], dtype=torch.float32)
    result = apply_top_k_top_p_pytorch(
        logits.clone(),
        k=torch.tensor([logits.shape[1]], dtype=torch.int32),
        p=torch.tensor([1.0], dtype=torch.float32),
    )

    torch.testing.assert_close(result, logits)


def test_small_topk_topp_matches_reference_masks_and_logits() -> None:
    torch.manual_seed(20260822)
    logits = torch.randn(4, 257, dtype=torch.float32)
    top_k = torch.tensor([1, 5, 20, 127], dtype=torch.int32)
    top_p = torch.tensor([0.8, 0.95, 1.0, 0.5], dtype=torch.float32)

    expected = apply_top_k_top_p_pytorch(logits.clone(), top_k.clone(), top_p.clone())
    actual = _apply_top_k_top_p_small_k_cpu(
        logits.clone(), top_k.clone(), top_p.clone()
    )

    assert actual is not None
    torch.testing.assert_close(actual, expected, atol=0, rtol=0)


def test_small_topk_topp_retains_all_topk_boundary_ties() -> None:
    logits = torch.tensor([[4.0, 3.0, 2.0, 2.0, 1.0]], dtype=torch.float32)

    actual = _apply_top_k_top_p_small_k_cpu(
        logits,
        torch.tensor([3], dtype=torch.int32),
        torch.tensor([0.95], dtype=torch.float32),
    )

    expected = torch.tensor([[4.0, 3.0, 2.0, 2.0, -float("inf")]])
    assert actual is not None
    torch.testing.assert_close(actual, expected, atol=0, rtol=0)


def test_small_topk_topp_uses_triton_token_order_for_topp_ties() -> None:
    logits = torch.tensor([[4.0, 4.0, 3.0, 2.0, 1.0]], dtype=torch.float32)

    actual = _apply_top_k_top_p_small_k_cpu(
        logits,
        torch.tensor([4], dtype=torch.int32),
        torch.tensor([0.3], dtype=torch.float32),
    )

    expected = torch.tensor(
        [[4.0, -float("inf"), -float("inf"), -float("inf"), -float("inf")]]
    )
    assert actual is not None
    torch.testing.assert_close(actual, expected, atol=0, rtol=0)


def test_small_topk_topp_keeps_fast_path_when_topp_does_not_split_tie() -> None:
    logits = torch.tensor([[4.0, 4.0, 3.0, 2.0, 1.0]], dtype=torch.float32)

    actual = _apply_top_k_top_p_small_k_cpu(
        logits,
        torch.tensor([4], dtype=torch.int32),
        torch.tensor([0.5], dtype=torch.float32),
    )

    assert actual is not None
    expected = apply_top_k_top_p_pytorch(
        logits.clone(),
        torch.tensor([4], dtype=torch.int32),
        torch.tensor([0.5], dtype=torch.float32),
    )
    torch.testing.assert_close(actual, expected, atol=0, rtol=0)


def test_small_topk_topp_falls_back_for_nonpositive_k() -> None:
    logits = torch.tensor([[4.0, 3.0, 2.0], [3.0, 2.0, 1.0]], dtype=torch.float32)

    actual = _apply_top_k_top_p_small_k_cpu(
        logits,
        torch.tensor([2, 0], dtype=torch.int32),
        torch.tensor([0.95, 0.95], dtype=torch.float32),
    )

    assert actual is None


def test_cpu_expand_batch_to_tokens_is_initialized() -> None:
    expanded = expand_batch_to_tokens(
        torch.tensor([1.0, 0.5], dtype=torch.float32),
        torch.tensor([3, 5], dtype=torch.int32),
        num_tokens=5,
    )

    torch.testing.assert_close(expanded, torch.tensor([1.0, 1.0, 1.0, 0.5, 0.5]))


def test_cpu_speculative_sampling_constraints_remain_finite() -> None:
    logits = torch.arange(64, dtype=torch.float32).view(4, 16)
    constrained = apply_sampling_constraints(
        logits,
        torch.tensor([4], dtype=torch.int32),
        SimpleNamespace(
            all_greedy=False,
            temperature=torch.tensor([1.0]),
            top_k=torch.tensor([5], dtype=torch.int32),
            top_p=torch.tensor([0.95]),
        ),
    )

    assert not torch.isnan(constrained).any()
    assert torch.isfinite(constrained).any(dim=-1).all()


def test_cpu_recovered_token_uses_positive_residual_mass() -> None:
    vocab_size = 16
    draft_probs = torch.zeros((1, vocab_size), dtype=torch.float32)
    draft_probs[0, 1] = 1.0
    target_probs = torch.zeros((1, vocab_size), dtype=torch.float32)
    target_probs[0, 5] = 1.0

    recovered = sample_recovered_tokens(
        max_spec_len=1,
        num_draft_tokens=[1],
        cu_num_draft_tokens=torch.tensor([1], dtype=torch.int32),
        draft_token_ids=torch.tensor([1], dtype=torch.int32),
        draft_probs=draft_probs,
        target_probs=target_probs,
        sampling_metadata=SimpleNamespace(
            generators={0: torch.Generator().manual_seed(7)}
        ),
        device=torch.device("cpu"),
    )

    torch.testing.assert_close(recovered, torch.tensor([5], dtype=torch.int32))


def test_cpu_random_rejection_emits_recovered_token() -> None:
    vocab_size = 16
    draft_probs = torch.zeros((1, vocab_size), dtype=torch.float32)
    draft_probs[0, 1] = 1.0
    target_logits = torch.full((1, vocab_size), -100.0, dtype=torch.float32)
    target_logits[0, 5] = 100.0

    output = rejection_sample(
        draft_token_ids=torch.tensor([1], dtype=torch.int32),
        num_draft_tokens=[1],
        max_spec_len=1,
        cu_num_draft_tokens=torch.tensor([1], dtype=torch.int32),
        draft_probs=draft_probs,
        target_logits=target_logits,
        bonus_token_ids=torch.tensor([[9]], dtype=torch.int32),
        sampling_metadata=SimpleNamespace(
            all_greedy=False,
            all_random=True,
            temperature=torch.tensor([1.0]),
            generators={0: torch.Generator().manual_seed(11)},
        ),
    )

    expected = torch.tensor([[5, PLACEHOLDER_TOKEN_ID]], dtype=torch.int32)
    torch.testing.assert_close(output, expected)
