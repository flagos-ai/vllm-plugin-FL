# SPDX-License-Identifier: Apache-2.0

import os
import sys
from types import ModuleType, SimpleNamespace

import pytest
import torch

from vllm_fl.models.qwen3_dflash2 import (
    _grouped_dynamic_conv_sliced,
    dflash2_keyed_uniform,
    grouped_dynamic_conv,
    select_greedy_path,
    select_stochastic_path,
    validate_dflash2_block_size,
)
from vllm_fl.spec_decode.dflash2 import (
    DFlash2Proposer,
    compute_dflash2_anchor_positions,
    dflash_target_rope_is_neox_style,
)


def test_dflash2_registers_before_registry_probe_returns(monkeypatch):
    import vllm.model_executor.models as models
    import vllm.platforms as platforms

    import vllm_fl

    registrations = []
    monkeypatch.setattr(
        models.ModelRegistry,
        "register_model",
        lambda architecture, model: registrations.append((architecture, model)),
    )
    monkeypatch.setattr(
        platforms, "current_platform", SimpleNamespace(device_type="cpu")
    )
    monkeypatch.setattr(vllm_fl, "_is_arm_cpu_build", lambda: True)
    monkeypatch.setattr("sys.argv", ["registry.py"])

    vllm_fl.register_model()

    dflash_registration = (
        "DFlash2DraftModel",
        "vllm_fl.models.qwen3_dflash2:DFlash2Qwen3ForCausalLM",
    )
    assert registrations.count(dflash_registration) == 1


def test_arm_cpu_bf16_installs_dflash2_compat(monkeypatch):
    import vllm.model_executor.models as models
    import vllm.platforms as platforms

    import vllm_fl

    calls = []
    compat = ModuleType("vllm_fl.patches.arm_cpu_vllm_0240")
    compat.install_arm_cpu_vllm_0240_compat = lambda: calls.append(True)
    monkeypatch.setitem(sys.modules, compat.__name__, compat)
    monkeypatch.setattr(models.ModelRegistry, "register_model", lambda *args: None)
    monkeypatch.setattr(
        platforms,
        "current_platform",
        SimpleNamespace(device_type="cpu"),
    )
    monkeypatch.setattr(vllm_fl, "_is_arm_cpu_build", lambda: True)
    monkeypatch.setattr("sys.argv", ["worker.py"])
    monkeypatch.delenv("FL_CPU_INT8", raising=False)
    monkeypatch.delenv("FL_CPU_INT4", raising=False)

    vllm_fl.register_model()

    assert calls == [True]


class _FakeSelector:
    def select_greedy(
        self,
        candidate_ids,
        unary_logits,
        hidden_states,
        anchor_token_ids,
    ):
        del unary_logits, hidden_states, anchor_token_ids
        return candidate_ids[..., 0]

    def select_stochastic(
        self,
        candidate_ids,
        unary_logits,
        hidden_states,
        anchor_token_ids,
        temperatures,
        seeds,
        anchor_positions,
    ):
        del (
            unary_logits,
            hidden_states,
            anchor_token_ids,
            temperatures,
            seeds,
            anchor_positions,
        )
        probabilities = torch.zeros_like(candidate_ids, dtype=torch.float32)
        probabilities[..., 1] = 1.0
        return candidate_ids[..., 1], probabilities


class _FakeDFlash2Model:
    def __init__(self):
        self.config = SimpleNamespace(vocab_size=11)
        self.model = SimpleNamespace(candidate_selector=_FakeSelector())

    def compute_candidates(self, hidden_states):
        num_tokens = hidden_states.shape[0]
        ids = torch.tensor([2, 5], dtype=torch.int64).expand(num_tokens, -1)
        logits = torch.tensor([0.25, 0.75]).expand(num_tokens, -1)
        return ids, logits


def _make_proposer():
    from vllm_fl.spec_decode.dflash2 import DFlash2Proposer

    proposer = object.__new__(DFlash2Proposer)
    proposer.model = _FakeDFlash2Model()
    proposer.num_speculative_tokens = 2
    proposer._anchor_token_ids = torch.tensor([7])
    proposer._anchor_positions = torch.tensor([10])
    proposer.runner = None
    proposer._request_sampling_seeds = {}
    return proposer


def test_dflash2_vllm024_greedy_selector_path():
    proposer = _make_proposer()
    sampling = SimpleNamespace(all_greedy=True)

    token_ids, draft_probs = proposer._sample_draft_tokens(torch.zeros(2, 3), sampling)

    assert token_ids.tolist() == [2, 2]
    assert draft_probs is None


def test_dflash2_vllm024_stochastic_selector_probability_rows():
    proposer = _make_proposer()
    sampling = SimpleNamespace(
        all_greedy=False,
        temperature=torch.tensor([0.8]),
        generators={},
    )

    token_ids, draft_probs = proposer._sample_draft_tokens(torch.zeros(2, 3), sampling)

    assert token_ids.tolist() == [5, 5]
    assert draft_probs.shape == (2, 11)
    assert torch.equal(draft_probs[:, 5], torch.ones(2))
    assert torch.count_nonzero(draft_probs).item() == 2


def test_runtime_block_size_stays_within_checkpoint_contract() -> None:
    draft_config = {"block_size": 8}

    assert validate_dflash2_block_size(draft_config, 3) == 4
    assert validate_dflash2_block_size(draft_config, 7) == 8
    with pytest.raises(ValueError, match="checkpoint block size"):
        validate_dflash2_block_size(draft_config, 8)


def _reference_grouped_dynamic_conv(
    hidden_states: torch.Tensor,
    delta: torch.Tensor,
    base: torch.Tensor,
    block_size: int,
    group_size: int,
) -> torch.Tensor:
    hidden_size = hidden_states.shape[-1]
    num_groups = hidden_size // group_size
    output = torch.empty_like(hidden_states)
    for row in range(hidden_states.shape[0]):
        block_position = row % block_size
        for group in range(num_groups):
            start = group * group_size
            end = start + group_size
            value = torch.zeros(group_size, dtype=hidden_states.dtype)
            for tap in range(base.shape[0]):
                if tap <= block_position:
                    source = hidden_states[row - tap, start:end]
                    coefficient = base[tap, start:end] + delta[row, tap, group]
                    value += coefficient * source
            output[row, start:end] = value
    return output


def _load_native_grouped_conv_or_skip() -> None:
    if hasattr(torch.ops.triton_jit_cpu, "dflash2_grouped_conv"):
        return
    library = os.environ.get("FLAGGEMS_LIBTRITON_JIT_Q4_OP")
    if library is None or not os.path.isfile(library):
        pytest.skip("FlagGems ARM operator bundle is not built")
    torch.ops.load_library(os.path.realpath(library))
    if not hasattr(torch.ops.triton_jit_cpu, "dflash2_grouped_conv"):
        pytest.skip("FlagGems ARM bundle lacks DFlash2 grouped convolution")


def test_grouped_dynamic_conv_matches_scalar_reference(monkeypatch) -> None:
    monkeypatch.setenv("VLLM_FL_DFLASH2_NATIVE_CONV", "0")
    monkeypatch.setenv("VLLM_FL_DFLASH2_SLICED_CONV", "0")
    torch.manual_seed(7)
    block_size = 4
    group_size = 2
    hidden_states = torch.randn(3 * block_size, 8)
    delta = torch.randn(3 * block_size, 3, 4)
    base = torch.randn(3, 8)

    actual = grouped_dynamic_conv(
        hidden_states,
        delta,
        base,
        block_size,
        group_size,
    )
    expected = _reference_grouped_dynamic_conv(
        hidden_states,
        delta,
        base,
        block_size,
        group_size,
    )
    torch.testing.assert_close(actual, expected)


def test_grouped_dynamic_conv_does_not_cross_draft_blocks(monkeypatch) -> None:
    monkeypatch.setenv("VLLM_FL_DFLASH2_NATIVE_CONV", "0")
    monkeypatch.setenv("VLLM_FL_DFLASH2_SLICED_CONV", "0")
    hidden_states = torch.tensor([[1.0], [2.0], [100.0], [200.0]])
    delta = torch.zeros(4, 2, 1)
    base = torch.ones(2, 1)

    actual = grouped_dynamic_conv(
        hidden_states,
        delta,
        base,
        block_size=2,
        group_size=1,
    )
    torch.testing.assert_close(
        actual[:, 0],
        torch.tensor([1.0, 3.0, 100.0, 300.0]),
    )


def test_sliced_grouped_dynamic_conv_is_bf16_bit_exact(monkeypatch) -> None:
    monkeypatch.setenv("VLLM_FL_DFLASH2_NATIVE_CONV", "0")
    monkeypatch.setenv("VLLM_FL_DFLASH2_SLICED_CONV", "0")
    torch.manual_seed(20260825)
    hidden_states = torch.randn(24, 5120, dtype=torch.bfloat16)
    delta = torch.randn(24, 2, 320, dtype=torch.bfloat16)
    base = torch.randn(2, 5120, dtype=torch.bfloat16)

    reference = grouped_dynamic_conv(
        hidden_states,
        delta,
        base,
        block_size=8,
        group_size=16,
    )
    direct_candidate = _grouped_dynamic_conv_sliced(
        hidden_states,
        delta,
        base,
        block_size=8,
        group_size=16,
    )
    monkeypatch.setenv("VLLM_FL_DFLASH2_SLICED_CONV", "1")
    routed_candidate = grouped_dynamic_conv(
        hidden_states,
        delta,
        base,
        block_size=8,
        group_size=16,
    )

    torch.testing.assert_close(direct_candidate, reference, atol=0, rtol=0)
    torch.testing.assert_close(routed_candidate, reference, atol=0, rtol=0)


def test_native_grouped_dynamic_conv_is_bf16_bit_exact(monkeypatch) -> None:
    _load_native_grouped_conv_or_skip()
    torch.manual_seed(20260826)
    hidden_states = torch.randn(24, 5120, dtype=torch.bfloat16)
    delta = torch.randn(24, 2, 320, dtype=torch.bfloat16)
    base = torch.randn(2, 5120, dtype=torch.bfloat16)
    monkeypatch.setenv("VLLM_FL_DFLASH2_NATIVE_CONV", "0")
    monkeypatch.setenv("VLLM_FL_DFLASH2_SLICED_CONV", "0")
    reference = grouped_dynamic_conv(
        hidden_states,
        delta,
        base,
        block_size=8,
        group_size=16,
    )
    candidate = torch.ops.triton_jit_cpu.dflash2_grouped_conv(
        hidden_states,
        delta,
        base,
        8,
        16,
    )
    torch.testing.assert_close(candidate, reference, atol=0, rtol=0)
    monkeypatch.setenv("VLLM_FL_DFLASH2_NATIVE_CONV", "1")
    routed = grouped_dynamic_conv(
        hidden_states,
        delta,
        base,
        block_size=8,
        group_size=16,
    )
    torch.testing.assert_close(routed, reference, atol=0, rtol=0)


def test_native_grouped_dynamic_conv_accepts_side_strides(monkeypatch) -> None:
    _load_native_grouped_conv_or_skip()
    torch.manual_seed(20260827)
    hidden_states = torch.randn(8, 5120, dtype=torch.bfloat16)
    all_delta = torch.randn(8, 2, 2, 320, dtype=torch.bfloat16)
    base = torch.randn(2, 5120, dtype=torch.bfloat16)
    delta = all_delta[:, 1]
    assert not delta.is_contiguous()
    monkeypatch.setenv("VLLM_FL_DFLASH2_NATIVE_CONV", "0")
    monkeypatch.setenv("VLLM_FL_DFLASH2_SLICED_CONV", "0")
    reference = grouped_dynamic_conv(
        hidden_states, delta, base, block_size=8, group_size=16
    )
    monkeypatch.setenv("VLLM_FL_DFLASH2_NATIVE_CONV", "1")
    candidate = grouped_dynamic_conv(
        hidden_states, delta, base, block_size=8, group_size=16
    )
    torch.testing.assert_close(candidate, reference, atol=0, rtol=0)


@pytest.mark.parametrize("batched_selector", ["0", "1"])
def test_select_greedy_path_matches_materialized_edge_scores(
    monkeypatch, batched_selector
) -> None:
    monkeypatch.setenv("VLLM_FL_DFLASH2_BATCHED_GREEDY_SELECTOR", batched_selector)
    torch.manual_seed(11)
    batch_size, positions, top_k, rank, vocab_size = 2, 5, 4, 3, 23
    candidate_ids = torch.stack(
        [torch.randperm(vocab_size)[: positions * top_k] for _ in range(batch_size)]
    ).reshape(batch_size, positions, top_k)
    unary_logits = torch.randn(batch_size, positions, top_k)
    hidden = torch.randn(batch_size, positions, rank)
    anchors = torch.tensor([21, 22])
    predecessor_codebook = torch.randn(vocab_size, rank)
    successor_codebook = torch.randn(vocab_size, rank)

    actual = select_greedy_path(
        candidate_ids,
        unary_logits,
        hidden,
        anchors,
        predecessor_codebook,
        successor_codebook,
    )

    predecessor = anchors
    expected = []
    for position in range(positions):
        candidates = candidate_ids[:, position]
        state = predecessor_codebook[predecessor] * hidden[:, position]
        scores = unary_logits[:, position] + torch.bmm(
            successor_codebook[candidates],
            state.unsqueeze(-1),
        ).squeeze(-1)
        selected = scores.argmax(dim=-1)
        predecessor = candidates.gather(1, selected[:, None]).squeeze(1)
        expected.append(predecessor)
    torch.testing.assert_close(actual, torch.stack(expected, dim=1))


def test_batched_greedy_selector_matches_reference_bf16(monkeypatch) -> None:
    torch.manual_seed(20260822)
    batch_size, positions, top_k, rank, vocab_size = 2, 7, 16, 256, 521
    candidate_ids = torch.randint(
        vocab_size, (batch_size, positions, top_k), dtype=torch.int64
    )
    unary_logits = torch.randn(batch_size, positions, top_k)
    hidden = torch.randn(batch_size, positions, rank, dtype=torch.bfloat16)
    anchors = torch.randint(vocab_size, (batch_size,), dtype=torch.int64)
    predecessor_codebook = torch.randn(vocab_size, rank, dtype=torch.bfloat16)
    successor_codebook = torch.randn(vocab_size, rank, dtype=torch.bfloat16)
    arguments = (
        candidate_ids,
        unary_logits,
        hidden,
        anchors,
        predecessor_codebook,
        successor_codebook,
    )
    monkeypatch.setenv("VLLM_FL_DFLASH2_BATCHED_GREEDY_SELECTOR", "0")
    reference = select_greedy_path(*arguments)
    monkeypatch.setenv("VLLM_FL_DFLASH2_BATCHED_GREEDY_SELECTOR", "1")
    candidate = select_greedy_path(*arguments)
    torch.testing.assert_close(candidate, reference, atol=0, rtol=0)


def test_select_stochastic_path_returns_reproducible_normalized_q() -> None:
    torch.manual_seed(13)
    batch_size, positions, top_k, rank, vocab_size = 1, 3, 4, 2, 19
    candidate_ids = torch.randperm(vocab_size)[: positions * top_k].reshape(
        batch_size,
        positions,
        top_k,
    )
    unary_logits = torch.randn(batch_size, positions, top_k)
    hidden = torch.randn(batch_size, positions, rank)
    anchors = torch.tensor([18])
    predecessor_codebook = torch.randn(vocab_size, rank)
    successor_codebook = torch.randn(vocab_size, rank)

    def run(seed: int):
        return select_stochastic_path(
            candidate_ids,
            unary_logits,
            hidden,
            anchors,
            predecessor_codebook,
            successor_codebook,
            torch.tensor([1.0]),
            [seed],
            torch.tensor([100]),
        )

    path, probabilities = run(29)
    repeated_path, repeated_probabilities = run(29)
    _, other_seed_probabilities = run(31)
    torch.testing.assert_close(
        probabilities.sum(dim=-1),
        torch.ones(batch_size, positions),
    )
    torch.testing.assert_close(probabilities, repeated_probabilities)
    # The first q row is seed-independent. Later rows legitimately change
    # when another seed chooses a different predecessor token in the walk.
    torch.testing.assert_close(probabilities[:, :1], other_seed_probabilities[:, :1])
    torch.testing.assert_close(path, repeated_path)
    assert torch.isin(path, candidate_ids.flatten()).all()


def test_select_stochastic_path_is_candidate_order_invariant() -> None:
    """The upstream sampler keys Gumbel noise by token id, not top-k slot."""
    candidate_ids = torch.tensor([[[2, 5, 7, 11]]])
    unary_logits = torch.zeros(1, 1, 4)
    hidden = torch.zeros(1, 1, 2)
    anchors = torch.tensor([13])
    predecessor_codebook = torch.zeros(17, 2)
    successor_codebook = torch.zeros(17, 2)

    def run(ids: torch.Tensor, logits: torch.Tensor) -> torch.Tensor:
        path, _ = select_stochastic_path(
            ids,
            logits,
            hidden,
            anchors,
            predecessor_codebook,
            successor_codebook,
            torch.tensor([1.0]),
            [29],
            torch.tensor([100]),
        )
        return path

    reference = run(candidate_ids, unary_logits)
    permutation = torch.tensor([3, 2, 1, 0])
    permuted = run(
        candidate_ids.index_select(-1, permutation),
        unary_logits.index_select(-1, permutation),
    )
    torch.testing.assert_close(permuted, reference)


def test_dflash2_keyed_uniform_matches_upstream_triton_vector() -> None:
    token_ids = torch.tensor(
        [
            2,
            5,
            7,
            11,
            17,
            29,
            31,
            101,
            127,
            255,
            1024,
            4095,
            65535,
            131071,
            248000,
            248319,
        ]
    )
    expected = torch.tensor(
        [
            0.46762946248054504,
            0.38040393590927124,
            0.9446861147880554,
            0.3120281398296356,
            0.05598408728837967,
            0.3419990539550781,
            0.1893262267112732,
            0.6072098612785339,
            0.29684576392173767,
            0.4915401339530945,
            0.352323442697525,
            0.6909090876579285,
            0.6584656834602356,
            0.32988160848617554,
            0.35414284467697144,
            0.923816442489624,
        ],
        dtype=torch.float32,
    )

    actual = dflash2_keyed_uniform(29, 100, token_ids)

    torch.testing.assert_close(actual, expected, atol=0, rtol=0)
    assert not torch.equal(actual, dflash2_keyed_uniform(29, 101, token_ids))


def test_dflash2_anchor_positions_skip_rejected_padding() -> None:
    target_positions = torch.tensor(
        [
            [0, 1, 2, 3, 4, 0, 1, 2],
            [0, 1, 2, 3, 4, 0, 1, 2],
            [0, 1, 2, 3, 4, 0, 1, 2],
        ]
    )

    actual = compute_dflash2_anchor_positions(
        target_positions,
        torch.tensor([0, 5, 8]),
        torch.tensor([2, 1]),
    )

    torch.testing.assert_close(actual, torch.tensor([3, 2]))


@pytest.mark.parametrize("style", [False, True])
def test_dflash2_inherits_target_rope_layout(monkeypatch, style: bool) -> None:
    target = torch.nn.Module()
    target.rotary_emb = torch.nn.Module()
    target.rotary_emb.is_neox_style = style
    proposer = object.__new__(DFlash2Proposer)
    proposer.draft_model_config = SimpleNamespace(hf_config=SimpleNamespace())
    observed = []

    def load_model(_self, _target_model):
        observed.append(proposer.draft_model_config.hf_config.is_neox_style)

    from vllm.v1.spec_decode.dflash import DFlashProposer

    monkeypatch.setattr(DFlashProposer, "load_model", load_model)

    proposer.load_model(target)

    assert dflash_target_rope_is_neox_style(target) is style
    assert observed == [style]
