# Copyright 2025 BAAI. All rights reserved.
"""Supplementary coverage for the Iluvatar MoE routing adapters.

Covers ``vllm_fl/dispatch/backends/vendor/iluvatar/moe_router.py``:

* ``topk_softmax`` no-bias path: results returned by ``call_op`` (fresh
  tensors) are copied into the caller's out params; the token-expert index
  slot-major layout is written by the reference impl.
* bias path: numeric oracle — selection by softmax(gating)+bias, weights
  gathered from the UNBIASED softmax, renormalization with the
  zero-denominator guard.
* padding mask: monkeypatched ``_get_padding_mask`` produces -1 fills on
  padded rows.
* ``patch_moe_router``: both real vllm router modules get
  ``vllm_topk_softmax is our fn``; originals are saved and restored;
  repeated calls are safe.

CPU-only. Vendor-gated to the iluvatar platform.
"""

from __future__ import annotations

import pytest
import torch

pytestmark = [pytest.mark.supplementary, pytest.mark.platforms("iluvatar")]

ROUTER_MODULES = (
    "vllm.model_executor.layers.fused_moe.router.fused_topk_router",
    "vllm.model_executor.layers.fused_moe.router.fused_topk_bias_router",
)


def _make_case(num_tokens=5, num_experts=7, topk=3, seed=0):
    torch.manual_seed(seed)
    return (
        torch.randn(num_tokens, num_experts),  # gating_output
        torch.zeros(num_tokens, topk, dtype=torch.float32),  # topk_weights
        torch.zeros(num_tokens, topk, dtype=torch.int32),  # topk_indices
        torch.zeros(num_tokens, topk, dtype=torch.int32),  # token_expert_indices
    )


@pytest.fixture
def restore_call_op(monkeypatch):
    """Patch vllm_fl.dispatch.call_op and restore it afterwards.

    moe_router imports ``call_op`` from ``vllm_fl.dispatch`` inside the
    function body at call time, so patching the package attribute is enough.
    """
    import vllm_fl.dispatch

    original = vllm_fl.dispatch.call_op
    yield original
    vllm_fl.dispatch.call_op = original


class TestTopkSoftmaxNoBias:
    def test_copies_fresh_tensors_into_out_params(self, monkeypatch):
        import vllm_fl.dispatch
        import vllm_fl.dispatch.backends.vendor.iluvatar.moe_router as mr

        gating, topk_weights, topk_indices, token_expert_indices = _make_case()
        scores = torch.softmax(gating.float(), dim=-1)
        ref_indices = torch.topk(scores, topk_indices.shape[1], dim=-1).indices
        ref_weights = scores.gather(1, ref_indices)

        call_names = []

        def fake_call_op(op_name, *args, **kwargs):
            call_names.append(op_name)
            # a backend that returns NEW tensors instead of mutating
            return ref_weights.clone(), ref_indices.clone()

        monkeypatch.setattr(vllm_fl.dispatch, "call_op", fake_call_op)
        w, i = mr.topk_softmax(
            topk_weights, topk_indices, token_expert_indices, gating, False
        )
        assert call_names == ["topk_softmax"]
        # out params themselves received the values via copy_
        assert w is topk_weights
        assert i is topk_indices
        assert torch.equal(topk_weights, ref_weights)
        assert torch.equal(topk_indices, ref_indices)

    def test_out_params_are_mutated_in_place(self, monkeypatch):
        import vllm_fl.dispatch
        import vllm_fl.dispatch.backends.vendor.iluvatar.moe_router as mr

        gating, topk_weights, topk_indices, token_expert_indices = _make_case()
        scores = torch.softmax(gating.float(), dim=-1)
        ref_indices = torch.topk(scores, topk_indices.shape[1], dim=-1).indices
        ref_weights = scores.gather(1, ref_indices)
        # distinctive garbage: proves copy_ overwrote it
        topk_weights.fill_(-123.0)
        topk_indices.fill_(-7)

        monkeypatch.setattr(
            vllm_fl.dispatch,
            "call_op",
            lambda *a, **k: (ref_weights.clone(), ref_indices.clone()),
        )
        mr.topk_softmax(
            topk_weights, topk_indices, token_expert_indices, gating, False
        )
        assert not torch.any(topk_weights == -123.0)
        assert not torch.any(topk_indices == -7)
        assert torch.equal(topk_weights, ref_weights)
        assert torch.equal(topk_indices, ref_indices)

    def test_token_expert_indices_untouched_by_no_bias_branch(
        self, monkeypatch
    ):
        """The no-bias branch delegates to call_op and only copies weights
        and indices back; ``token_expert_indices`` is left to the underlying
        op (not written by this adapter)."""
        import vllm_fl.dispatch
        import vllm_fl.dispatch.backends.vendor.iluvatar.moe_router as mr

        num_tokens, topk = 5, 3
        gating, topk_weights, topk_indices, token_expert_indices = _make_case(
            num_tokens=num_tokens, topk=topk
        )
        scores = torch.softmax(gating.float(), dim=-1)
        ref_indices = torch.topk(scores, topk, dim=-1).indices
        ref_weights = scores.gather(1, ref_indices)
        sentinel = token_expert_indices.clone()
        token_expert_indices.fill_(-99)

        monkeypatch.setattr(
            vllm_fl.dispatch,
            "call_op",
            lambda *a, **k: (ref_weights.clone(), ref_indices.clone()),
        )
        mr.topk_softmax(
            topk_weights, topk_indices, token_expert_indices, gating, False
        )
        # adapter copies weights/indices but never writes token_expert_indices
        assert torch.equal(token_expert_indices, torch.full_like(sentinel, -99))


class TestTopkSoftmaxBias:
    def _oracle(self, gating, bias, topk, renormalize):
        scores = torch.softmax(gating.float(), dim=-1)
        indices = torch.topk(scores + bias.float(), topk, dim=-1).indices
        weights = scores.gather(1, indices)
        if renormalize:
            denominator = weights.sum(dim=-1, keepdim=True)
            weights = weights / torch.where(
                denominator > 0, denominator, torch.ones_like(denominator)
            )
        return weights, indices

    @pytest.mark.parametrize("renormalize", [False, True])
    def test_selection_by_bias_weights_from_unbiased_softmax(
        self, renormalize
    ):
        import vllm_fl.dispatch.backends.vendor.iluvatar.moe_router as mr

        gating, topk_weights, topk_indices, token_expert_indices = _make_case(
            seed=4
        )
        bias = torch.randn(gating.shape[-1])
        topk = topk_indices.shape[1]

        mr.topk_softmax(
            topk_weights,
            topk_indices,
            token_expert_indices,
            gating,
            renormalize,
            bias,
        )
        ref_weights, ref_indices = self._oracle(gating, bias, topk, renormalize)
        assert torch.allclose(topk_weights, ref_weights)
        assert torch.equal(topk_indices, ref_indices)

    def test_bias_changes_selection_but_not_weight_source(self):
        """A large bias must reroute selection; weights still from softmax."""
        import vllm_fl.dispatch.backends.vendor.iluvatar.moe_router as mr

        gating = torch.tensor([[0.1, 0.5, -0.2, 0.3]])
        bias = torch.tensor([-100.0, 0.0, 0.0, 0.0])  # expert 0 excluded
        topk_weights = torch.zeros(1, 2)
        topk_indices = torch.zeros(1, 2, dtype=torch.int32)
        token_expert_indices = torch.zeros(1, 2, dtype=torch.int32)

        mr.topk_softmax(topk_weights, topk_indices, token_expert_indices, gating, False, bias)
        ref_weights, ref_indices = self._oracle(gating, bias, 2, False)
        assert torch.equal(topk_indices, ref_indices)
        assert 0 not in topk_indices.tolist()[0]
        # weights are softmax probs of the selected experts (unbiased)
        scores = torch.softmax(gating.float(), dim=-1)
        assert torch.allclose(topk_weights, scores.gather(1, ref_indices))

    def test_zero_denominator_guard(self):
        """Underflowed softmax weights renormalize via the ones-guard."""
        import vllm_fl.dispatch.backends.vendor.iluvatar.moe_router as mr

        # extreme logits: one expert takes all probability, others underflow
        gating = torch.tensor([[100.0, -100.0, -100.0, -100.0]])
        bias = torch.tensor([-5.0, 0.0, 0.0, 0.0])  # bias picks underflowed
        topk = 2
        topk_weights = torch.zeros(1, topk)
        topk_indices = torch.zeros(1, topk, dtype=torch.int32)
        token_expert_indices = torch.zeros(1, topk, dtype=torch.int32)

        mr.topk_softmax(
            topk_weights, topk_indices, token_expert_indices, gating, True, bias
        )
        ref_weights, ref_indices = self._oracle(gating, bias, topk, True)
        assert torch.equal(topk_indices, ref_indices)
        # denominator is exactly zero -> weights stay 0 (no NaN/inf)
        assert torch.all(torch.isfinite(topk_weights))
        assert torch.allclose(topk_weights, ref_weights)

    def test_bias_path_skips_call_op(self, monkeypatch):
        """With a bias, the reference impl runs; call_op is never consulted."""
        import vllm_fl.dispatch
        import vllm_fl.dispatch.backends.vendor.iluvatar.moe_router as mr

        def explode(*args, **kwargs):
            raise AssertionError("call_op must not be used on the bias path")

        monkeypatch.setattr(vllm_fl.dispatch, "call_op", explode)
        gating, topk_weights, topk_indices, token_expert_indices = _make_case(
            seed=9
        )
        bias = torch.randn(gating.shape[-1])
        mr.topk_softmax(
            topk_weights, topk_indices, token_expert_indices, gating, False, bias
        )
        ref_weights, ref_indices = self._oracle(gating, bias, topk_indices.shape[1], False)
        assert torch.allclose(topk_weights, ref_weights)


class TestPaddingMask:
    def test_padded_rows_filled_with_minus_one(self, monkeypatch):
        import vllm_fl.dispatch
        import vllm.model_executor.layers.fused_moe.router.fused_topk_router as ftr
        import vllm_fl.dispatch.backends.vendor.iluvatar.moe_router as mr

        num_tokens = 5
        gating, topk_weights, topk_indices, token_expert_indices = _make_case(
            num_tokens=num_tokens
        )
        scores = torch.softmax(gating.float(), dim=-1)
        ref_indices = torch.topk(scores, topk_indices.shape[1], dim=-1).indices
        ref_weights = scores.gather(1, ref_indices)

        mask = torch.tensor([False, True, False, True, False])
        monkeypatch.setattr(
            ftr, "_get_padding_mask", lambda n: mask[:n]
        )
        monkeypatch.setattr(
            vllm_fl.dispatch,
            "call_op",
            lambda *a, **k: (ref_weights.clone(), ref_indices.clone()),
        )
        w, i = mr.topk_softmax(
            topk_weights, topk_indices, token_expert_indices, gating, False
        )
        # rows 1 and 3 are padding: fully -1
        assert torch.all(i[1] == -1)
        assert torch.all(i[3] == -1)
        # non-padding rows keep the routed indices (all != -1 since experts
        # are in [0, num_experts))
        for row in (0, 2, 4):
            assert torch.all(i[row] >= 0)
        assert torch.equal(i[0], ref_indices[0])

    def test_no_mask_leaves_indices_untouched(self, monkeypatch):
        import vllm_fl.dispatch
        import vllm.model_executor.layers.fused_moe.router.fused_topk_router as ftr
        import vllm_fl.dispatch.backends.vendor.iluvatar.moe_router as mr

        gating, topk_weights, topk_indices, token_expert_indices = _make_case()
        scores = torch.softmax(gating.float(), dim=-1)
        ref_indices = torch.topk(scores, topk_indices.shape[1], dim=-1).indices
        ref_weights = scores.gather(1, ref_indices)

        monkeypatch.setattr(ftr, "_get_padding_mask", lambda n: None)
        monkeypatch.setattr(
            vllm_fl.dispatch,
            "call_op",
            lambda *a, **k: (ref_weights.clone(), ref_indices.clone()),
        )
        _, i = mr.topk_softmax(
            topk_weights, topk_indices, token_expert_indices, gating, False
        )
        assert torch.equal(i, ref_indices)


class TestPatchMoeRouter:
    @pytest.fixture(autouse=True)
    def _save_and_restore_router_modules(self):
        """Save vllm_topk_softmax on both real router modules; restore after."""
        import importlib

        saved = {}
        for name in ROUTER_MODULES:
            module = importlib.import_module(name)
            saved[name] = module.vllm_topk_softmax
        yield saved
        for name, original in saved.items():
            module = importlib.import_module(name)
            module.vllm_topk_softmax = original

    def test_both_router_modules_patched(self, _save_and_restore_router_modules):
        import importlib

        import vllm_fl.dispatch.backends.vendor.iluvatar.moe_router as mr

        mr.patch_moe_router()
        for name in ROUTER_MODULES:
            module = importlib.import_module(name)
            assert module.vllm_topk_softmax is mr.topk_softmax, name

    def test_repeated_patch_is_safe(self, _save_and_restore_router_modules):
        import importlib

        import vllm_fl.dispatch.backends.vendor.iluvatar.moe_router as mr

        mr.patch_moe_router()
        first = {}
        for name in ROUTER_MODULES:
            first[name] = importlib.import_module(name).vllm_topk_softmax
        mr.patch_moe_router()
        for name in ROUTER_MODULES:
            assert importlib.import_module(name).vllm_topk_softmax is first[name]

    def test_patched_fn_is_the_module_level_topk_softmax(
        self, _save_and_restore_router_modules
    ):
        import importlib

        import vllm_fl.dispatch.backends.vendor.iluvatar.moe_router as mr

        mr.patch_moe_router()
        module = importlib.import_module(ROUTER_MODULES[0])
        # exactly our function object, not a copy or wrapper
        assert module.vllm_topk_softmax.__module__ == mr.topk_softmax.__module__
        assert module.vllm_topk_softmax.__name__ == "topk_softmax"
