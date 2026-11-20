# Copyright (c) 2026 BAAI. All rights reserved.

"""CPU numeric tests for the PyTorch reference dispatch backend.

Every expectation is computed with an independent oracle written directly in
torch (softmax/topk/cumsum arithmetic spelled out element-wise or with
deliberately different ops than the implementation), then compared with
``torch.testing.assert_close`` at explicit tolerances.
"""

import math

import pytest
import torch

from vllm_fl.dispatch.backends.reference import ReferenceBackend


@pytest.fixture
def backend():
    """A fresh ReferenceBackend instance per test."""
    return ReferenceBackend()


class TestTopkSoftmax:
    """ReferenceBackend.topk_softmax (impl/fused_moe.py delegate)."""

    @pytest.mark.supplementary
    def test_basic_routing_matches_softmax_topk_oracle(self, backend):
        """Weights come from raw softmax over the logits, gathered at top-k."""
        gating = torch.tensor(
            [[0.1, 2.0, -1.0, 0.5], [1.0, 1.0, 1.0, 1.0]], dtype=torch.float32
        )
        topk = 2
        weights = torch.zeros(2, topk, dtype=torch.float32)
        indices = torch.zeros(2, topk, dtype=torch.int32)
        src_rows = torch.zeros(2, topk, dtype=torch.int32)

        out_w, out_i = backend.topk_softmax(weights, indices, src_rows, gating)

        # Independent oracle: manual softmax then top-2 selection.
        for row in range(2):
            exps = [math.exp(v) for v in gating[row].tolist()]
            denom = sum(exps)
            probs = [e / denom for e in exps]
            order = sorted(range(4), key=lambda j: -probs[j])[:topk]
            for slot, j in enumerate(order):
                assert out_i[row, slot].item() == j
                torch.testing.assert_close(
                    out_w[row, slot].item(), probs[j], rtol=1e-5, atol=1e-6
                )

    @pytest.mark.supplementary
    def test_bias_changes_selection_but_not_weights(self, backend):
        """e_score_correction_bias only skews selection; weights stay raw."""
        gating = torch.tensor([[0.0, 0.05, 0.1, -3.0]])
        bias = torch.tensor([0.0, 0.0, -1.0, 5.0])
        weights = torch.zeros(1, 2)
        indices = torch.zeros(1, 2, dtype=torch.int64)
        src_rows = torch.zeros(1, 2, dtype=torch.int64)

        backend.topk_softmax(weights, indices, src_rows, gating, False, bias)

        # probs ≈ [0.3119, 0.3279, 0.3447, 0.0155]; scores = probs + bias
        # ≈ [0.3119, 0.3279, -0.6553, 5.0155] → top-2 selection is {3, 1}.
        assert set(indices[0].tolist()) == {3, 1}
        # Raw softmax probabilities (unbiased) gathered at those indices.
        exps = [math.exp(v) for v in gating[0].tolist()]
        denom = sum(exps)
        probs = {j: exps[j] / denom for j in range(4)}
        for slot in range(2):
            torch.testing.assert_close(
                weights[0, slot].item(),
                probs[indices[0, slot].item()],
                rtol=1e-5,
                atol=1e-6,
            )

    @pytest.mark.supplementary
    def test_renormalize_sums_to_one(self, backend):
        """Renormalize rescales selected weights to sum to 1 per row."""
        gating = torch.tensor([[3.0, 1.0, 0.0]])
        weights = torch.zeros(1, 2)
        indices = torch.zeros(1, 2, dtype=torch.int64)
        src_rows = torch.zeros(1, 2, dtype=torch.int64)

        backend.topk_softmax(weights, indices, src_rows, gating, renormalize=True)
        torch.testing.assert_close(weights.sum().item(), 1.0, rtol=1e-5, atol=1e-6)
        # Ratio between the two selected weights is preserved.
        exps = [math.exp(v) for v in gating[0].tolist()]
        ratio = exps[0] / exps[1]
        torch.testing.assert_close(
            weights[0, 0].item() / weights[0, 1].item(),
            ratio,
            rtol=1e-4,
            atol=1e-6,
        )

    @pytest.mark.supplementary
    def test_source_rows_use_slot_major_layout(self, backend):
        """token_expert_indices follow k*num_tokens + token (slot-major)."""
        num_tokens, topk = 3, 2
        gating = torch.zeros(num_tokens, 4)
        weights = torch.zeros(num_tokens, topk)
        indices = torch.zeros(num_tokens, topk, dtype=torch.int64)
        src_rows = torch.zeros(num_tokens, topk, dtype=torch.int64)

        backend.topk_softmax(weights, indices, src_rows, gating)
        expected = torch.tensor(
            [[0, 3], [1, 4], [2, 5]], dtype=torch.int64
        )  # rows + slot*num_tokens
        torch.testing.assert_close(src_rows, expected)

    @pytest.mark.supplementary
    def test_outputs_are_written_in_place(self, backend):
        """The same tensors passed in are returned after being filled."""
        gating = torch.randn(4, 5)
        weights = torch.zeros(4, 2)
        indices = torch.zeros(4, 2, dtype=torch.int64)
        src_rows = torch.zeros(4, 2, dtype=torch.int64)
        out_w, out_i = backend.topk_softmax(weights, indices, src_rows, gating)
        assert out_w is weights and out_i is indices


class TestMoeAlignBlockSize:
    """ReferenceBackend.moe_align_block_size against independent bucket oracle."""

    @staticmethod
    def _oracle(topk_ids, block_size, num_experts):
        """Independent bucket construction with plain Python loops."""
        flat = [int(v) for v in topk_ids.reshape(-1).tolist()]
        counts = [0] * num_experts
        for eid in flat:
            counts[eid] += 1
        padded = [((c + block_size - 1) // block_size) * block_size for c in counts]
        offsets, run = [], 0
        for p in padded:
            offsets.append(run)
            run += p
        total = sum(padded)
        sorted_ids = {}
        rank_within = [0] * num_experts
        for token, eid in enumerate(flat):
            sorted_ids[offsets[eid] + rank_within[eid]] = token
            rank_within[eid] += 1
        return counts, padded, offsets, total, sorted_ids

    @pytest.mark.supplementary
    def test_basic_alignment(self, backend):
        """Buckets group tokens per expert with block padding."""
        topk_ids = torch.tensor([[0, 1], [1, 2], [2, 0]])
        block_size, num_experts = 4, 3
        counts, padded, offsets, total, sorted_ids = self._oracle(
            topk_ids, block_size, num_experts
        )

        sorted_out, expert_ids, num_post = backend.moe_align_block_size(
            topk_ids, block_size, num_experts
        )

        # Static buffer: num_valid(6) + 3*(4-1) + 1 = 16 → round up to 16.
        assert sorted_out.numel() == 16
        torch.testing.assert_close(num_post, torch.tensor([total], dtype=torch.int32))
        assert total == 12  # 3 experts padded to 4 each
        for dest, token in sorted_ids.items():
            assert sorted_out[dest].item() == token
        # Unfilled tail slots carry the sentinel num_valid.
        for i in range(total, sorted_out.numel()):
            assert sorted_out[i].item() == 6
        # Every block of 4 before `total` maps to its owning expert.
        for block in range(total // block_size):
            torch.testing.assert_close(
                expert_ids[block], torch.tensor(block, dtype=torch.int32)
            )
        assert (expert_ids[total // block_size :] == -1).all()

    @pytest.mark.supplementary
    def test_pad_sorted_ids_rounds_total_up(self, backend):
        """pad_sorted_ids rounds the block count up to a full block."""
        topk_ids = torch.tensor([[0, 0], [0, 0], [1, 1]])
        sorted_out, expert_ids, num_post = backend.moe_align_block_size(
            topk_ids, block_size=4, num_experts=2, pad_sorted_ids=True
        )
        # expert0: 4 tokens → 4; expert1: 2 → 4; total 8 → already block aligned.
        torch.testing.assert_close(num_post, torch.tensor([8], dtype=torch.int32))
        # With 6 valid tokens and 2 experts: 6 + 2*3 + 1 = 13 → round to 16.
        assert sorted_out.numel() == 16
        assert (expert_ids[:2] == torch.tensor([0, 1], dtype=torch.int32)).all()
        assert (expert_ids[2:] == -1).all()

    @pytest.mark.supplementary
    def test_expert_map_without_ignore_remaps_output(self, backend):
        """Without ignore_invalid_experts, expert_ids are remapped through the map."""
        topk_ids = torch.tensor([[0, 1], [1, 0]])
        expert_map = torch.tensor([5, 7])
        sorted_out, expert_ids, num_post = backend.moe_align_block_size(
            topk_ids, block_size=2, num_experts=2, expert_map=expert_map
        )
        # Both experts own 2 tokens → one block each, remapped to 5 and 7.
        torch.testing.assert_close(
            expert_ids[:2], torch.tensor([5, 7], dtype=torch.int32)
        )
        torch.testing.assert_close(num_post, torch.tensor([4], dtype=torch.int32))

    @pytest.mark.supplementary
    def test_ignore_invalid_experts_drops_remote_ids(self, backend):
        """With ignore_invalid_experts, ids mapped to -1 are dropped."""
        # expert_map[3] = -1 → global id 3 belongs to another rank.
        topk_ids = torch.tensor([[0, 1], [1, 3]])
        expert_map = torch.tensor([0, 1, -1, -1])
        sorted_out, expert_ids, num_post = backend.moe_align_block_size(
            topk_ids,
            block_size=2,
            num_experts=2,
            expert_map=expert_map,
            ignore_invalid_experts=True,
        )
        # Only tokens 0,1,2 are local: expert0 gets 1, expert1 gets 2.
        torch.testing.assert_close(num_post, torch.tensor([4], dtype=torch.int32))
        assert (expert_ids[0] == 0).item()
        assert (expert_ids[1] == 1).item()
        assert (expert_ids[2:] == -1).all()

    @pytest.mark.supplementary
    def test_negative_ids_are_screened(self, backend):
        """Upstream-style -1 padding never indexes into any expert bucket."""
        topk_ids = torch.tensor([[0, -1], [-1, 1]])
        sorted_out, expert_ids, num_post = backend.moe_align_block_size(
            topk_ids, block_size=2, num_experts=2
        )
        # Valid tokens: 0 (expert0) and 3 (expert1); each pads to 2.
        torch.testing.assert_close(num_post, torch.tensor([4], dtype=torch.int32))
        # The dropped ids land in the sink slot (last buffer position).
        assert sorted_out[-1].item() == 4  # sentinel value = num_valid
        assert (expert_ids[:2] == torch.tensor([0, 1], dtype=torch.int32)).all()

    @pytest.mark.supplementary
    def test_out_of_range_ids_without_map_are_dropped(self, backend):
        """Ids >= num_experts (no map) are treated as invalid, not crashed on."""
        topk_ids = torch.tensor([[0, 99]])
        sorted_out, expert_ids, num_post = backend.moe_align_block_size(
            topk_ids, block_size=2, num_experts=2
        )
        # Only token 0 is valid: expert0 gets 1 token padded to a block of 2.
        torch.testing.assert_close(num_post, torch.tensor([2], dtype=torch.int32))
        assert (expert_ids[0] == 0).item()
        assert (expert_ids[1:] == -1).all()


class TestMoeSum:
    """ReferenceBackend.moe_sum reduces over the expert axis."""

    @pytest.mark.supplementary
    def test_sum_over_topk_axis_into_out(self, backend):
        """Summation lands in the provided out tensor."""
        inp = torch.tensor(
            [[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]]
        )  # [tokens, topk, dim]
        out = torch.zeros(2, 2)
        backend.moe_sum(inp, out)
        torch.testing.assert_close(out, torch.tensor([[4.0, 6.0], [12.0, 14.0]]))

    @pytest.mark.supplementary
    def test_sum_independent_oracle(self, backend):
        """Result equals a per-token loop over experts (different op mix)."""
        inp = torch.randn(5, 3, 7)
        out = torch.zeros(5, 7)
        backend.moe_sum(inp, out)
        for token in range(5):
            expected = inp[token, 0].clone()
            for expert in range(1, 3):
                expected = expected + inp[token, expert]
            torch.testing.assert_close(out[token], expected, rtol=1e-6, atol=1e-6)


class TestDynamicPerTokenQuantInt8:
    """ReferenceBackend.dynamic_per_token_quant_int8 numeric contract."""

    @pytest.mark.supplementary
    def test_symmetric_per_token_scale(self, backend):
        """Quantization is symmetric: q = round(x * 127 / max|x|), clamped."""
        x = torch.tensor([[1.0, -2.0], [0.0, 0.5]])
        q, s = backend.dynamic_per_token_quant_int8(x)

        # Row 0: max|x| = 2 → scale 2/127; q = round(x*127/2) = [64, -127].
        torch.testing.assert_close(q[0], torch.tensor([64, -127], dtype=q.dtype))
        torch.testing.assert_close(s[0].item(), 2.0 / 127.0, rtol=1e-6, atol=1e-8)

        # Row 1: max|x| = 0.5 → scale 0.5/127; q = [0, 127].
        torch.testing.assert_close(q[1], torch.tensor([0, 127], dtype=q.dtype))
        torch.testing.assert_close(s[1].item(), 0.5 / 127.0, rtol=1e-6, atol=1e-8)

    @pytest.mark.supplementary
    def test_zero_row_produces_zero_scale(self, backend):
        """An all-zero row quantizes to zeros with a zero scale (no NaN)."""
        x = torch.zeros(1, 4)
        q, s = backend.dynamic_per_token_quant_int8(x)
        assert (q == 0).all()
        assert s.item() == 0.0

    @pytest.mark.supplementary
    def test_reconstruction_error_bound(self, backend):
        """Dequantized values stay within half a scale step of the input."""
        x = torch.randn(16, 32) * 3.0
        q, s = backend.dynamic_per_token_quant_int8(x)
        dequant = q.float() * s
        # |x - dequant| <= scale/2 (+ float rounding slack).
        step = s.squeeze(-1) if s.dim() > 1 else s
        bound = step.unsqueeze(-1) * 0.5 + 1e-6
        assert ((x - dequant).abs() <= bound).all()


class TestBackendMetadata:
    """ReferenceBackend identity and availability."""

    @pytest.mark.supplementary
    def test_name_and_availability(self, backend):
        """The backend is named 'reference' and available with torch present."""
        assert backend.name == "reference"
        assert backend.is_available() is True

    @pytest.mark.supplementary
    def test_availability_is_cached_on_class(self, backend):
        """is_available memoizes the probe result on the class."""
        backend.is_available()
        assert ReferenceBackend._available is True

    @pytest.mark.supplementary
    def test_implemented_ops_delegate_to_impls(self, backend):
        """Activation/norm ops exist on the backend and return tensors."""

        class Obj:
            approximate = "none"
            variance_epsilon = 1e-6
            weight = torch.ones(4)
            pass_weight = True
            pass_weight_add = True

        obj = Obj()
        out = backend.silu_and_mul(obj, torch.randn(2, 8))
        assert out.shape == (2, 4)
        out = backend.gelu_and_mul(obj, torch.randn(2, 8))
        assert out.shape == (2, 4)
        out = backend.rms_norm(obj, torch.randn(2, 4))
        assert out.shape == (2, 4)
