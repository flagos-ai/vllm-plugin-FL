# Copyright 2025 BAAI. All rights reserved.

"""CPU contracts for the Kunlunxin MoE / FLA / causal-conv1d implementations.

Covers, through ``xtorch_ops`` stubs plus pure-torch oracles:

* ``fused_moe/fused_moe.py::fused_experts_impl`` -- rejection paths, expert
  remapping, inplace contract, and the ``_klx_fused_experts`` pipeline
  (call order, buffer shapes, activation switch);
* ``fused_moe/experts_selector.py`` -- ``vllm_topk_softmax``,
  ``fused_topk``, ``fused_topk_bias`` (pure torch, independent oracles) and
  ``grouped_topk`` (kernel stub whose input is the oracle);
* ``fla/chunk.py`` and ``fla/fused_recurrent.py`` -- top-level validations
  and the kernel state-layout contracts;
* ``causal_conv1d.py`` -- activation rejections, host lod forwarding, and
  the decode-path layout round-trip.

Order independence: all ``sys.modules`` manipulation happens inside context
managers that restore every touched entry, and module-scope ``xtorch_ops``
importers are re-imported through their real package paths inside the
manager.
"""

from __future__ import annotations

import contextlib
import importlib
import sys
from types import ModuleType

import pytest
import torch
import torch.nn.functional as F

pytestmark = [pytest.mark.supplementary]

_FUSED_MOE = "vllm_fl.dispatch.backends.vendor.kunlunxin.impl.fused_moe.fused_moe"
_EXPERTS = "vllm_fl.dispatch.backends.vendor.kunlunxin.impl.fused_moe.experts_selector"
_CHUNK = "vllm_fl.dispatch.backends.vendor.kunlunxin.impl.fla.chunk"
_RECURRENT = "vllm_fl.dispatch.backends.vendor.kunlunxin.impl.fla.fused_recurrent"
_CONV1D = "vllm_fl.dispatch.backends.vendor.kunlunxin.impl.causal_conv1d"
_GDN = "vllm_fl.dispatch.backends.vendor.kunlunxin.impl.fused_gdn_gating"

#: impl modules that import xtorch_ops at MODULE scope; they can only be
#: (re-)imported while a stub is installed.
_STUB_REQUIRED = [_FUSED_MOE, _EXPERTS, _CHUNK, _RECURRENT, _CONV1D]


@contextlib.contextmanager
def _xtorch_ops_reimport(module_names, **ops):
    """Install an xtorch_ops stub and (re)import the given impl modules.

    Saves every touched ``sys.modules`` entry, pops the target modules,
    installs the stub, imports the targets through their real package path,
    and restores all saved entries on exit.  Nothing leaks.
    """
    stub = ModuleType("xtorch_ops")
    for name, fn in ops.items():
        setattr(stub, name, fn)
    names = [*module_names, "xtorch_ops"]
    saved = {name: sys.modules.get(name) for name in names}
    for name in module_names:
        sys.modules.pop(name, None)
    sys.modules["xtorch_ops"] = stub
    try:
        modules = {name: importlib.import_module(name) for name in module_names}
        yield stub, modules
    finally:
        for name, entry in saved.items():
            if entry is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = entry


# ---------------------------------------------------------------------------
# fused_moe.fused_experts_impl: rejection paths
# ---------------------------------------------------------------------------


class TestFusedExpertsRejections:
    @pytest.mark.parametrize(
        ("kwargs", "message"),
        [
            ({"use_fp8_w8a8": True}, "does not support fp8_w8a8"),
            ({"use_int8_w8a16": True}, "does not support fp8_w8a8/int8_w8a16"),
            ({"use_int4_w4a16": True}, "does not support fp8_w8a8/int8_w8a16"),
            (
                {"apply_router_weight_on_input": True},
                "does not support apply_router_weight_on_input",
            ),
            (
                {"activation": "gelu_tanh"},
                "does not support activation 'gelu_tanh'",
            ),
        ],
        ids=["fp8_w8a8", "int8_w8a16", "int4_w4a16", "router_weight_on_input", "activation"],
    )
    @pytest.mark.platforms("kunlunxin")
    def test_unsupported_features_raise_not_implemented(self, kwargs, message):
        with _xtorch_ops_reimport(_STUB_REQUIRED) as (_, modules):
            fn = modules[_FUSED_MOE].fused_experts_impl
            hidden = torch.randn(2, 4)
            w1 = torch.randn(3, 8)
            w2 = torch.randn(3, 4)
            with pytest.raises(NotImplementedError, match=message):
                fn(hidden, w1, w2, torch.ones(2, 2), torch.zeros(2, 2, dtype=torch.int32), **kwargs)

    @pytest.mark.platforms("kunlunxin")
    def test_supported_activations_are_accepted(self):
        # every documented activation clears stage-1 validation and reaches
        # the kernel pipeline (stubbed to a no-op).
        def noop(*args, **kwargs):
            return None

        with _xtorch_ops_reimport(
            _STUB_REQUIRED,
            gen_block_statistic=noop,
            moe_pre_sorted=noop,
            moe_fc=noop,
            swiglu=noop,
            gelu=lambda t: t,
            relu=lambda t: t,
            silu=lambda t: t,
            moe_post=noop,
        ) as (_, modules):
            fn = modules[_FUSED_MOE].fused_experts_impl
            hidden = torch.randn(2, 4)
            w1 = torch.randn(3, 8)
            w2 = torch.randn(3, 4)
            weights = torch.ones(2, 2)
            ids = torch.zeros(2, 2, dtype=torch.int32)
            for activation in ["silu", "gelu", "relu", "gelu_no_mul", "silu_no_mul"]:
                out = fn(hidden, w1, w2, weights, ids, activation=activation)
                assert out.shape == hidden.shape


class TestFusedExpertsOutputContract:
    @pytest.fixture
    def moe_setup(self):
        torch.manual_seed(23)
        seq_num, hidden_dim, experts, topk, ffn = 3, 4, 5, 2, 6
        hidden = torch.randn(seq_num, hidden_dim)
        w1 = torch.randn(experts, 2 * ffn)
        w2 = torch.randn(experts, ffn, hidden_dim)
        weights = torch.rand(seq_num, topk)
        ids = torch.randint(0, experts, (seq_num, topk), dtype=torch.int32)
        return hidden, w1, w2, weights, ids, (seq_num, hidden_dim, experts, topk, ffn)

    @pytest.mark.platforms("kunlunxin")
    def test_inplace_true_writes_into_hidden_states(self, moe_setup):
        def noop(*args, **kwargs):
            return None

        with _xtorch_ops_reimport(
            _STUB_REQUIRED,
            gen_block_statistic=noop,
            moe_pre_sorted=noop,
            moe_fc=noop,
            swiglu=noop,
            moe_post=noop,
        ) as (_, modules):
            fn = modules[_FUSED_MOE].fused_experts_impl
            hidden, w1, w2, weights, ids, _ = moe_setup
            result = fn(hidden, w1, w2, weights, ids, inplace=True)
            assert result is hidden

    @pytest.mark.platforms("kunlunxin")
    def test_inplace_false_allocates_fresh_zeroed_output(self, moe_setup):
        def noop(*args, **kwargs):
            return None

        with _xtorch_ops_reimport(
            _STUB_REQUIRED,
            gen_block_statistic=noop,
            moe_pre_sorted=noop,
            moe_fc=noop,
            swiglu=noop,
            moe_post=noop,
        ) as (_, modules):
            fn = modules[_FUSED_MOE].fused_experts_impl
            hidden, w1, w2, weights, ids, _ = moe_setup
            result = fn(hidden, w1, w2, weights, ids, inplace=False)
            assert result is not hidden
            assert torch.equal(result, torch.zeros_like(hidden))

    @pytest.mark.platforms("kunlunxin")
    def test_expert_map_remaps_global_ids_to_local(self, moe_setup):
        seen = {}

        def moe_pre_sorted(hidden_states, topk_ids, *rest):
            seen["topk_ids"] = topk_ids.clone()

        def noop(*args, **kwargs):
            return None

        with _xtorch_ops_reimport(
            _STUB_REQUIRED,
            gen_block_statistic=noop,
            moe_pre_sorted=moe_pre_sorted,
            moe_fc=noop,
            swiglu=noop,
            moe_post=noop,
        ) as (_, modules):
            fn = modules[_FUSED_MOE].fused_experts_impl
            hidden, w1, w2, weights, ids, _ = moe_setup
            # global ids 2,3,4 map to local 0,1,2 (1:1 EP shard); 0,1 dropped
            # by mapping to -1 is a real-world case but the impl just
            # gathers, so use a permutation mapping that is easy to verify.
            expert_map = torch.tensor([-1, -1, 0, 1, 2], dtype=torch.int32)
            # restrict ids to the mappable range so the gather is defined
            ids = torch.full_like(ids, 3)
            ids[0, :] = 4
            fn(
                hidden, w1, w2, weights, ids,
                inplace=False, expert_map=expert_map,
            )
            expected = expert_map[ids.long()].to(ids.dtype)
            torch.testing.assert_close(seen["topk_ids"], expected)
            assert seen["topk_ids"].dtype == ids.dtype


class TestKlxFusedExpertsPipeline:
    """_klx_fused_experts plumbing via full xtorch_ops stubs."""

    @pytest.mark.platforms("kunlunxin")
    @pytest.mark.parametrize(
        ("activation", "expected_op"),
        [
            ("silu", "swiglu"),
            ("gelu", "gelu"),
            ("relu", "relu"),
            ("gelu_no_mul", "gelu"),
            ("silu_no_mul", "silu"),
        ],
    )
    def test_activation_switch_selects_the_right_kernel(
        self, activation, expected_op
    ):
        torch.manual_seed(29)
        seq_num, hidden_dim, experts, topk, ffn = 2, 4, 3, 2, 6
        hidden = torch.randn(seq_num, hidden_dim)
        w1 = torch.randn(experts, 2 * ffn)
        w2 = torch.randn(experts, ffn, hidden_dim)
        weights = torch.rand(seq_num, topk)
        ids = torch.randint(0, experts, (seq_num, topk), dtype=torch.int32)

        ops_used = []
        order = []
        counters = {
            "gen_block_statistic": 0,
            "moe_pre_sorted": 0,
            "moe_fc": 0,
            "swiglu": 0,
            "gelu": 0,
            "relu": 0,
            "silu": 0,
            "moe_post": 0,
        }

        def gen_block_statistic(topk_ids, block_statistic):
            counters["gen_block_statistic"] += 1
            order.append("gen_block_statistic")
            assert topk_ids.dtype == torch.int32
            assert block_statistic.shape == (12, experts)
            assert block_statistic.dtype == torch.int32

        def moe_pre_sorted(hs, topk_ids, block_statistic, expand, index, expert_m, lod):
            counters["moe_pre_sorted"] += 1
            order.append("moe_pre_sorted")
            assert expand.shape == (seq_num * topk, hidden_dim)
            assert index.shape == (seq_num * topk,)
            assert expert_m.shape == (experts,)
            assert lod.shape == (experts + 1,)

        def moe_fc(x, w, lod, index, moe_topk, out, bias=None):
            counters["moe_fc"] += 1
            order.append(f"moe_fc")
            return None

        def swiglu(x, out):
            counters["swiglu"] += 1
            order.append("swiglu")
            assert out.shape == (seq_num * topk, ffn)
            out.copy_(torch.ones_like(out))

        def gelu(x):
            counters["gelu"] += 1
            order.append("gelu")
            ops_used.append("gelu")
            return x

        def relu(x):
            counters["relu"] += 1
            order.append("relu")
            ops_used.append("relu")
            return x

        def silu(x):
            counters["silu"] += 1
            order.append("silu")
            ops_used.append("silu")
            return x

        def moe_post(x, index_2d, topk_weights, dequant_scale, output):
            counters["moe_post"] += 1
            order.append("moe_post")
            assert index_2d.shape == (seq_num, topk)
            assert topk_weights.shape == (seq_num, topk)
            assert dequant_scale.shape == (seq_num, topk)
            assert output.shape == (seq_num, hidden_dim)

        with _xtorch_ops_reimport(
            _STUB_REQUIRED,
            gen_block_statistic=gen_block_statistic,
            moe_pre_sorted=moe_pre_sorted,
            moe_fc=moe_fc,
            swiglu=swiglu,
            gelu=gelu,
            relu=relu,
            silu=silu,
            moe_post=moe_post,
        ) as (_, modules):
            fn = modules[_FUSED_MOE].fused_experts_impl
            fn(hidden, w1, w2, weights, ids, activation=activation, inplace=False)

        # exact pipeline shape and order
        assert order == [
            "gen_block_statistic",
            "moe_pre_sorted",
            "moe_fc",
            expected_op,
            "moe_fc",
            "moe_post",
        ]
        assert counters["gen_block_statistic"] == 1
        assert counters["moe_pre_sorted"] == 1
        assert counters["moe_fc"] == 2
        assert counters["moe_post"] == 1
        assert counters[expected_op] == 1
        for other in ("swiglu", "gelu", "relu", "silu"):
            if other != expected_op:
                assert counters[other] == 0, f"{other} kernel was invoked"

    @pytest.mark.platforms("kunlunxin")
    def test_unknown_activation_raises_value_error(self):
        torch.manual_seed(31)
        hidden = torch.randn(2, 4)
        w1 = torch.randn(3, 8)
        w2 = torch.randn(3, 4)
        weights = torch.rand(2, 2)
        ids = torch.zeros(2, 2, dtype=torch.int32)

        def noop(*args, **kwargs):
            return None

        with _xtorch_ops_reimport(
            _STUB_REQUIRED,
            gen_block_statistic=noop,
            moe_pre_sorted=noop,
            moe_fc=noop,
            moe_post=noop,
        ) as (_, modules):
            # fused_experts_impl rejects unknown activations in stage 1
            # already; drive _klx_fused_experts directly to cover the
            # in-pipeline ValueError.
            fn = modules[_FUSED_MOE]._klx_fused_experts
            with pytest.raises(ValueError, match="Unsupported activation 'gelu_tanh'"):
                fn(
                    hidden_states=hidden,
                    output=torch.zeros_like(hidden),
                    w1=w1,
                    w2=w2,
                    topk_weights=weights,
                    topk_ids=ids,
                    activation="gelu_tanh",
                )

    @pytest.mark.platforms("kunlunxin")
    def test_int8_quantization_rejected_in_pipeline(self):
        hidden = torch.randn(2, 4)
        w1 = torch.randn(3, 8)
        w2 = torch.randn(3, 4)
        weights = torch.rand(2, 2)
        ids = torch.zeros(2, 2, dtype=torch.int32)

        def noop(*args, **kwargs):
            return None

        with _xtorch_ops_reimport(
            _STUB_REQUIRED,
            gen_block_statistic=noop,
            moe_pre_sorted=noop,
            moe_fc=noop,
            moe_post=noop,
        ) as (_, modules):
            fn = modules[_FUSED_MOE]._klx_fused_experts
            for flag in ("use_int8_w8a8", "use_int8_w4a8"):
                with pytest.raises(NotImplementedError, match="int8 w8a8 and w4a8"):
                    fn(
                        hidden_states=hidden,
                        output=torch.zeros_like(hidden),
                        w1=w1,
                        w2=w2,
                        topk_weights=weights,
                        topk_ids=ids,
                        **{flag: True},
                    )


# ---------------------------------------------------------------------------
# experts_selector: pure-torch routing functions
# ---------------------------------------------------------------------------


def _manual_softmax(x):
    """softmax computed without torch.softmax (independent oracle)."""
    shifted = x - x.max(dim=-1, keepdim=True).values
    exp = shifted.exp()
    return exp / exp.sum(dim=-1, keepdim=True)


def _manual_topk(scores, k):
    """top-k indices via argsort (independent of torch.topk)."""
    order = torch.argsort(scores, dim=-1, descending=True)
    return order[..., :k]


class TestVllmTopkSoftmax:
    @pytest.mark.parametrize(
        ("n", "e", "k", "renormalize"),
        [
            (4, 8, 2, False),
            (4, 8, 2, True),
            (7, 16, 5, True),
            (1, 3, 3, False),
            (6, 5, 1, True),
        ],
    )
    @pytest.mark.platforms("kunlunxin")
    def test_matches_independent_oracle(self, n, e, k, renormalize):
        with _xtorch_ops_reimport(_STUB_REQUIRED) as (_, modules):
            fn = modules[_EXPERTS].vllm_topk_softmax
            torch.manual_seed(37)
            gating = torch.randn(n, e)

            topk_weights = torch.empty(n, k, dtype=torch.float32)
            topk_ids = torch.empty(n, k, dtype=torch.int32)
            token_expert_indices = torch.empty(n, k, dtype=torch.int32)

            w, i = fn(
                topk_weights, topk_ids, token_expert_indices, gating, renormalize
            )

            # independent oracle
            scores = _manual_softmax(gating.float())
            expect_ids = _manual_topk(scores, k)
            expect_w = scores.gather(1, expect_ids)
            if renormalize:
                expect_w = expect_w / expect_w.sum(-1, keepdim=True)

            assert w is topk_weights and i is topk_ids
            assert w.dtype == torch.float32
            # identical selected sets (order unspecified: sorted=False)
            assert torch.equal(
                torch.sort(i.long(), dim=-1).values,
                torch.sort(expect_ids, dim=-1).values,
            )
            # and identical weight multisets per row (order unspecified)
            torch.testing.assert_close(
                torch.sort(w, dim=-1).values, torch.sort(expect_w, dim=-1).values
            )
            # weight at each position matches the renormalized score of the
            # chosen expert
            positional = scores.gather(1, i.long())
            if renormalize:
                positional = positional / positional.sum(-1, keepdim=True)
            torch.testing.assert_close(w, positional)
            # the ABI compatibility output is initialized deterministically:
            # row id broadcast across the topk columns
            torch.testing.assert_close(
                token_expert_indices,
                torch.arange(n, dtype=torch.int32)[:, None].expand(n, k),
            )

    @pytest.mark.platforms("kunlunxin")
    def test_writes_into_caller_buffers(self):
        with _xtorch_ops_reimport(_STUB_REQUIRED) as (_, modules):
            fn = modules[_EXPERTS].vllm_topk_softmax
            gating = torch.tensor([[1.0, 2.0, 3.0]])
            topk_weights = torch.full((1, 2), 99.0)
            topk_ids = torch.full((1, 2), 99, dtype=torch.int32)
            token_expert_indices = torch.zeros(1, 2, dtype=torch.int32)
            fn(topk_weights, topk_ids, token_expert_indices, gating, False)
            # selected experts are 2 and 1 (descending softmax scores)
            assert set(topk_ids[0].tolist()) == {2, 1}
            assert 99.0 not in topk_weights[0].tolist()

    @pytest.mark.platforms("kunlunxin")
    def test_softmax_computed_in_fp32(self):
        with _xtorch_ops_reimport(_STUB_REQUIRED) as (_, modules):
            fn = modules[_EXPERTS].vllm_topk_softmax
            # bf16 logits: the FP32 softmax path keeps tiny probabilities
            # distinguishable where a bf16 softmax would collapse them.
            gating = torch.tensor([[1.0, 1.0, 1.0001]], dtype=torch.bfloat16)
            topk_weights = torch.empty(1, 2, dtype=torch.float32)
            topk_ids = torch.empty(1, 2, dtype=torch.int32)
            token_expert_indices = torch.empty(1, 2, dtype=torch.int32)
            fn(topk_weights, topk_ids, token_expert_indices, gating, False)
            scores = _manual_softmax(gating.float())
            torch.testing.assert_close(topk_weights, scores.gather(1, topk_ids.long()))


class TestFusedTopk:
    @pytest.mark.parametrize(
        ("n", "e", "k", "renormalize", "indices_type"),
        [
            (5, 9, 3, True, None),
            (5, 9, 3, False, None),
            (2, 4, 4, True, torch.int64),
            (3, 6, 1, False, torch.int32),
        ],
    )
    @pytest.mark.platforms("kunlunxin")
    def test_allocates_outputs_and_matches_oracle(
        self, n, e, k, renormalize, indices_type
    ):
        with _xtorch_ops_reimport(_STUB_REQUIRED) as (_, modules):
            fn = modules[_EXPERTS].fused_topk
            torch.manual_seed(41)
            hidden = torch.randn(n, e)
            gating = torch.randn(n, e)

            w, i, tei = fn(hidden, gating, k, renormalize, indices_type)

            assert w.shape == (n, k) and w.dtype == torch.float32
            expect_dtype = torch.int32 if indices_type is None else indices_type
            assert i.dtype == expect_dtype
            assert tei.dtype == torch.int32

            scores = _manual_softmax(gating.float())
            expect_ids = _manual_topk(scores, k)
            expect_w = scores.gather(1, expect_ids)
            if renormalize:
                expect_w = expect_w / expect_w.sum(-1, keepdim=True)
            assert torch.equal(
                torch.sort(i.long(), dim=-1).values,
                torch.sort(expect_ids, dim=-1).values,
            )
            # positional check against the impl's own selection (with the
            # same renormalization the impl applies)
            positional = scores.gather(1, i.long())
            if renormalize:
                positional = positional / positional.sum(-1, keepdim=True)
            torch.testing.assert_close(w, positional)
            # renormalized weights sum to one per row
            if renormalize:
                torch.testing.assert_close(
                    w.sum(-1), torch.ones(n), rtol=1e-5, atol=1e-5
                )

    @pytest.mark.platforms("kunlunxin")
    def test_mismatched_token_counts_assert(self):
        with _xtorch_ops_reimport(_STUB_REQUIRED) as (_, modules):
            fn = modules[_EXPERTS].fused_topk
            hidden = torch.randn(3, 4)
            gating = torch.randn(2, 4)
            with pytest.raises(AssertionError, match="Number of tokens mismatch"):
                fn(hidden, gating, 2, False)


class TestFusedTopkBias:
    @pytest.mark.parametrize(
        ("n", "e", "k", "renormalize"),
        [
            (4, 8, 2, False),
            (4, 8, 2, True),
            (6, 10, 4, True),
        ],
    )
    @pytest.mark.platforms("kunlunxin")
    def test_selection_uses_bias_weights_use_raw_scores(
        self, n, e, k, renormalize
    ):
        with _xtorch_ops_reimport(_STUB_REQUIRED) as (_, modules):
            fn = modules[_EXPERTS].fused_topk_bias
            torch.manual_seed(43)
            hidden = torch.randn(n, e)
            gating = torch.randn(n, e)
            bias = torch.randn(e)

            w, i = fn(hidden, gating, bias, k, renormalize)

            # oracle: selection on softmax+bias, weights gathered from the
            # unbiased softmax scores
            scores = _manual_softmax(gating)
            choice = scores + bias.unsqueeze(0)
            expect_ids = _manual_topk(choice, k)
            expect_w = scores.gather(1, expect_ids)
            if renormalize:
                expect_w = expect_w / expect_w.sum(-1, keepdim=True)

            assert w.dtype == torch.float32
            assert i.dtype == torch.int32
            assert torch.equal(
                torch.sort(i.long(), dim=-1).values,
                torch.sort(expect_ids, dim=-1).values,
            )
            # weights match the oracle as a multiset (selection order is
            # unspecified: torch.topk(sorted=False)) and positionally against
            # the impl's own selected ids
            torch.testing.assert_close(
                torch.sort(w, dim=-1).values, torch.sort(expect_w, dim=-1).values,
                rtol=1e-5, atol=1e-5,
            )
            # positional check against the impl's own selection (with the
            # same renormalization)
            positional = scores.gather(1, i.long())
            if renormalize:
                positional = positional / positional.sum(-1, keepdim=True)
            torch.testing.assert_close(w, positional, rtol=1e-5, atol=1e-5)

    @pytest.mark.platforms("kunlunxin")
    def test_bias_moves_selection_off_the_softmax_argmax(self):
        with _xtorch_ops_reimport(_STUB_REQUIRED) as (_, modules):
            fn = modules[_EXPERTS].fused_topk_bias
            # expert 0 has the largest score; a large negative bias on it
            # must push it out of the selection entirely.
            gating = torch.tensor([[3.0, 1.0, 0.5]])
            bias = torch.tensor([-100.0, 0.0, 0.0])
            _, i = fn(torch.randn(1, 3), gating, bias, 2, False)
            selected = i.reshape(-1).tolist()
            assert 0 not in selected
            assert set(selected) == {1, 2}


class TestGroupedTopk:
    @pytest.mark.platforms("kunlunxin")
    def test_kernel_input_is_bias_corrected_scores(self):
        received = []

        def moe_group_topk(scores, n_group, topk_group, topk_weights, topk_ids, extra):
            received.append(
                {
                    "scores": scores.clone(),
                    "n_group": n_group,
                    "topk_group": topk_group,
                    "extra": extra,
                }
            )
            # selection oracle implemented with the plugin's own primitives:
            # pick topk experts from the top topk_group groups
            e = scores.shape[-1]
            per = e // n_group
            grouped = scores.view(-1, n_group, per)
            group_scores = grouped.max(dim=-1).values
            top_groups = _manual_topk(group_scores, topk_group)
            mask = torch.zeros_like(scores, dtype=torch.bool)
            for g in range(n_group):
                if g in top_groups[0].tolist():
                    mask[0, g * per : (g + 1) * per] = True
            masked = scores.masked_fill(~mask, float("-inf"))
            sel = _manual_topk(masked, topk_weights.shape[-1])
            topk_ids.copy_(sel.to(torch.int32))
            topk_weights.copy_(scores.gather(1, sel))

        with _xtorch_ops_reimport(
            _STUB_REQUIRED, moe_group_topk=moe_group_topk
        ) as (_, modules):
            fn = modules[_EXPERTS].grouped_topk
            torch.manual_seed(47)
            n, e, n_group, topk_group, topk = 1, 8, 4, 2, 3
            scores = torch.rand(n, e)
            bias = torch.randn(e)

            w, i = fn(
                scores, n_group, topk_group, topk, False, 1.0, bias, 0
            )

            call = received[0]
            assert call["n_group"] == n_group
            assert call["topk_group"] == topk_group
            assert call["extra"] is None
            # the kernel sees scores + bias (selection domain)
            torch.testing.assert_close(call["scores"], scores + bias.unsqueeze(0))
            # weights come from the ORIGINAL (unbiased) scores
            torch.testing.assert_close(w, scores.gather(1, i.long()))

    @pytest.mark.parametrize("renormalize", [False, True])
    @pytest.mark.platforms("kunlunxin")
    def test_renormalize_and_scaling_applied_after_gather(self, renormalize):
        def moe_group_topk(scores, n_group, topk_group, topk_weights, topk_ids, extra):
            sel = _manual_topk(scores, topk_weights.shape[-1])
            topk_ids.copy_(sel.to(torch.int32))
            topk_weights.copy_(scores.gather(1, sel))

        with _xtorch_ops_reimport(
            _STUB_REQUIRED, moe_group_topk=moe_group_topk
        ) as (_, modules):
            fn = modules[_EXPERTS].grouped_topk
            torch.manual_seed(51)
            n, e, k = 2, 6, 3
            scores = torch.rand(n, e)
            scaling = 2.5
            w, i = fn(
                scores, 2, 1, k, renormalize, scaling,
                torch.zeros(e), 0,
            )
            gathered = scores.gather(1, i.long())
            if renormalize:
                expect = gathered / gathered.sum(-1, keepdim=True) * scaling
            else:
                expect = gathered * scaling
            torch.testing.assert_close(w, expect, rtol=1e-5, atol=1e-6)

    @pytest.mark.platforms("kunlunxin")
    def test_scoring_func_one_applies_sigmoid_first(self):
        received = []

        def moe_group_topk(scores, n_group, topk_group, topk_weights, topk_ids, extra):
            received.append(scores.clone())
            sel = _manual_topk(scores, topk_weights.shape[-1])
            topk_ids.copy_(sel.to(torch.int32))
            topk_weights.copy_(scores.gather(1, sel))

        with _xtorch_ops_reimport(
            _STUB_REQUIRED, moe_group_topk=moe_group_topk
        ) as (_, modules):
            fn = modules[_EXPERTS].grouped_topk
            torch.manual_seed(53)
            logits = torch.randn(2, 5) * 3
            w, i = fn(
                logits, 1, 1, 3, False, 1.0, torch.zeros(5), 1
            )
            torch.testing.assert_close(received[0], torch.sigmoid(logits))

    @pytest.mark.platforms("kunlunxin")
    def test_empty_bias_bypasses_correction(self):
        received = []

        def moe_group_topk(scores, n_group, topk_group, topk_weights, topk_ids, extra):
            received.append(scores.clone())
            sel = _manual_topk(scores, topk_weights.shape[-1])
            topk_ids.copy_(sel.to(torch.int32))
            topk_weights.copy_(scores.gather(1, sel))

        with _xtorch_ops_reimport(
            _STUB_REQUIRED, moe_group_topk=moe_group_topk
        ) as (_, modules):
            fn = modules[_EXPERTS].grouped_topk
            torch.manual_seed(59)
            scores = torch.rand(2, 6)
            w, i = fn(scores, 2, 1, 3, False, 1.0, torch.empty(0), 0)
            # no bias: the kernel receives the raw scores unchanged
            torch.testing.assert_close(received[0], scores)
            torch.testing.assert_close(w, scores.gather(1, i.long()))


# ---------------------------------------------------------------------------
# fla/chunk.py
# ---------------------------------------------------------------------------


def _fla_inputs(dtype=torch.bfloat16, batch=1, seq=6, heads=2, k=4, v=4):
    torch.manual_seed(61)
    q = torch.randn(batch, seq, heads, k, dtype=dtype)
    k_t = torch.randn(batch, seq, heads, k, dtype=dtype)
    v_t = torch.randn(batch, seq, heads, v, dtype=dtype)
    g = torch.randn(batch, seq, heads, dtype=torch.float32)
    beta = torch.rand(batch, seq, heads, dtype=torch.float32)
    return q, k_t, v_t, g, beta


class TestChunkGatedDeltaRule:
    @pytest.mark.platforms("kunlunxin")
    def test_cu_seqlens_requires_batch_one(self):
        with _xtorch_ops_reimport(_STUB_REQUIRED) as (_, modules):
            fn = modules[_CHUNK].chunk_gated_delta_rule
            q, k_t, v_t, g, beta = _fla_inputs(batch=2)
            with pytest.raises(ValueError, match="batch size is expected to be 1"):
                fn(q, k_t, v_t, g, beta, cu_seqlens=torch.tensor([0, 3, 6]))

    @pytest.mark.platforms("kunlunxin")
    def test_cu_seqlens_requires_matching_initial_states(self):
        with _xtorch_ops_reimport(_STUB_REQUIRED) as (_, modules):
            fn = modules[_CHUNK].chunk_gated_delta_rule
            q, k_t, v_t, g, beta = _fla_inputs(batch=1)
            # two sequences in cu_seqlens but only one initial state
            state = torch.randn(1, 2, 4, 4)
            with pytest.raises(ValueError, match="number of initial states"):
                fn(
                    q, k_t, v_t, g, beta,
                    initial_state=state,
                    cu_seqlens=torch.tensor([0, 3, 6]),
                )

    @pytest.mark.platforms("kunlunxin")
    def test_head_first_is_rejected_before_any_work(self):
        with _xtorch_ops_reimport(_STUB_REQUIRED) as (_, modules):
            fn = modules[_CHUNK].chunk_gated_delta_rule
            q, k_t, v_t, g, beta = _fla_inputs()
            # On this Python the raise expression itself is malformed
            # (DeprecationWarning takes no stacklevel), so any exception
            # aborts the call before the kernel is reached; the guaranteed
            # contract is that head_first never reaches the kernel and the
            # call never returns normally.
            kernel_called = []
            modules[_CHUNK].__dict__["xtorch_ops"] = type(
                "Probe", (), {"chunk_gated_delta_rule": lambda *a: kernel_called.append(1)}
            )()
            with pytest.raises((DeprecationWarning, TypeError)):
                fn(q, k_t, v_t, g, beta, head_first=True)
            assert kernel_called == []

    @pytest.mark.platforms("kunlunxin")
    def test_float32_inputs_rejected(self):
        with _xtorch_ops_reimport(_STUB_REQUIRED) as (_, modules):
            fn = modules[_CHUNK].chunk_gated_delta_rule
            q, k_t, v_t, g, beta = _fla_inputs(dtype=torch.float32)
            with pytest.raises(AssertionError, match="does not support float32"):
                fn(q, k_t, v_t, g, beta)

    @pytest.mark.platforms("kunlunxin")
    def test_scale_defaults_to_inverse_sqrt_k(self):
        received = {}

        def chunk_gated_delta_rule(
            q, k, v, g, beta, scale, state, o, final_state, cu_seqlens,
            head_first=False, use_qk_l2norm_in_kernel=True,
        ):
            received["scale"] = scale
            received["use_qk_l2norm_in_kernel"] = use_qk_l2norm_in_kernel
            o.copy_(torch.ones_like(o))
            final_state.copy_(torch.ones_like(final_state))

        with _xtorch_ops_reimport(
            _STUB_REQUIRED, chunk_gated_delta_rule=chunk_gated_delta_rule
        ) as (_, modules):
            fn = modules[_CHUNK].chunk_gated_delta_rule
            q, k_t, v_t, g, beta = _fla_inputs()
            initial = torch.randn(1, 2, 4, 4, dtype=torch.bfloat16)
            o, final_state = fn(q, k_t, v_t, g, beta, initial_state=initial)
            # K = 4 -> scale = 4 ** -0.5 = 0.5
            assert received["scale"] == pytest.approx(4 ** -0.5)
            # explicit scale is forwarded verbatim
            o, final_state = fn(q, k_t, v_t, g, beta, scale=0.25, initial_state=initial)
            assert received["scale"] == 0.25

    @pytest.mark.platforms("kunlunxin")
    def test_fwd_state_layout_contract(self):
        """The kernel sees [N,H,K,V] and the caller gets back [N,H,V,K]."""
        received = {}

        def chunk_gated_delta_rule(
            q, k, v, g, beta, scale, state, o, final_state, cu_seqlens,
            head_first=False, use_qk_l2norm_in_kernel=True,
        ):
            received["initial_state"] = state.clone()
            received["cu_seqlens"] = cu_seqlens
            received["head_first"] = head_first
            received["l2norm"] = use_qk_l2norm_in_kernel
            # echo the (transposed) initial state into final_state, so the
            # caller-side transpose is observable
            final_state.copy_(state)

        with _xtorch_ops_reimport(
            _STUB_REQUIRED, chunk_gated_delta_rule=chunk_gated_delta_rule
        ) as (_, modules):
            fwd = modules[_CHUNK].chunk_gated_delta_rule_fwd
            torch.manual_seed(67)
            B, T, H, K, V = 1, 5, 2, 3, 4
            q = torch.randn(B, T, H, K, dtype=torch.bfloat16)
            k_t = torch.randn(B, T, H, K, dtype=torch.bfloat16)
            v_t = torch.randn(B, T, H, V, dtype=torch.bfloat16)
            g = torch.randn(B, T, H, dtype=torch.float32)
            beta = torch.rand(B, T, H, dtype=torch.float32)
            # vLLM paged cache layout: [N, H, V, K]
            initial = torch.arange(H * V * K, dtype=torch.float32).reshape(1, H, V, K)
            cu = torch.tensor([0, 2, 5])

            g_out, o, A, final_state, w, h, v_new = fwd(
                q, k_t, v_t, g, beta, scale=0.5,
                initial_state=initial,
                output_final_state=True,
                cu_seqlens=cu,
            )

            # the kernel received the [N, H, K, V] transpose of the cache
            # block, made contiguous
            expect_kernel_state = initial.transpose(2, 3).contiguous()
            assert received["initial_state"].shape == (1, H, K, V)
            torch.testing.assert_close(received["initial_state"], expect_kernel_state)
            assert received["cu_seqlens"] is not None
            # cu_seqlens was moved to host; on a CPU tensor .cpu() is the
            # identity, so the value is unchanged either way
            torch.testing.assert_close(
                received["cu_seqlens"], torch.tensor([0, 2, 5])
            )
            assert received["head_first"] is False
            assert received["l2norm"] is True
            # caller gets [N, H, V, K] back: the kernel's [K, V] output
            # re-transposed
            assert final_state.shape == (1, H, V, K)
            torch.testing.assert_close(final_state, expect_kernel_state.transpose(2, 3).contiguous())
            # 7-tuple ABI: (g, o, A, final_state, w, h, v_new)
            assert g_out is g
            assert o.shape == v_t.shape
            assert A is None and w is None and h is None and v_new is None

    @pytest.mark.platforms("kunlunxin")
    def test_fwd_bf16_inputs_promoted_to_fp32_for_kernel(self):
        received = {}

        def chunk_gated_delta_rule(
            q, k, v, g, beta, scale, state, o, final_state, cu_seqlens,
            head_first=False, use_qk_l2norm_in_kernel=True,
        ):
            received["g_dtype"] = g.dtype
            received["beta_dtype"] = beta.dtype
            received["state_dtype"] = state.dtype
            final_state.copy_(state)

        with _xtorch_ops_reimport(
            _STUB_REQUIRED, chunk_gated_delta_rule=chunk_gated_delta_rule
        ) as (_, modules):
            fwd = modules[_CHUNK].chunk_gated_delta_rule_fwd
            q, k_t, v_t, g, beta = _fla_inputs()
            initial = torch.randn(1, 2, 4, 4, dtype=torch.bfloat16)
            fwd(
                q, k_t, v_t, g, beta, scale=0.5,
                initial_state=initial, output_final_state=True,
            )
            assert received["g_dtype"] == torch.float32
            assert received["beta_dtype"] == torch.float32
            assert received["state_dtype"] == torch.float32

    @pytest.mark.platforms("kunlunxin")
    def test_fwd_none_scale_passes_minus_one_to_kernel(self):
        received = {}

        def chunk_gated_delta_rule(
            q, k, v, g, beta, scale, state, o, final_state, cu_seqlens,
            head_first=False, use_qk_l2norm_in_kernel=True,
        ):
            received["scale"] = scale
            final_state.copy_(state)

        with _xtorch_ops_reimport(
            _STUB_REQUIRED, chunk_gated_delta_rule=chunk_gated_delta_rule
        ) as (_, modules):
            fwd = modules[_CHUNK].chunk_gated_delta_rule_fwd
            q, k_t, v_t, g, beta = _fla_inputs()
            initial = torch.randn(1, 2, 4, 4, dtype=torch.bfloat16)
            fwd(
                q, k_t, v_t, g, beta, scale=None,
                initial_state=initial, output_final_state=True,
            )
            # -1 is the kernel's "auto scale" sentinel
            assert received["scale"] == -1

    @pytest.mark.platforms("kunlunxin")
    def test_top_level_passes_l2norm_flag_and_returns_o(self):
        received = {}

        def chunk_gated_delta_rule(
            q, k, v, g, beta, scale, state, o, final_state, cu_seqlens,
            head_first=False, use_qk_l2norm_in_kernel=True,
        ):
            received["l2norm"] = use_qk_l2norm_in_kernel
            o.copy_(torch.full_like(o, 2.0))
            final_state.copy_(torch.full_like(final_state, 7.0))

        with _xtorch_ops_reimport(
            _STUB_REQUIRED, chunk_gated_delta_rule=chunk_gated_delta_rule
        ) as (_, modules):
            fn = modules[_CHUNK].chunk_gated_delta_rule
            q, k_t, v_t, g, beta = _fla_inputs()
            initial = torch.randn(1, 2, 4, 4, dtype=torch.bfloat16)
            o, final_state = fn(
                q, k_t, v_t, g, beta,
                initial_state=initial, use_qk_l2norm_in_kernel=True,
            )
            assert received["l2norm"] is True
            assert o.dtype == torch.bfloat16
            torch.testing.assert_close(o, torch.full_like(o, 2.0))
            assert final_state is not None
            # core_attn_out receives a flattened copy of the output
            core_attn_out = torch.zeros(64, dtype=torch.bfloat16)
            o2, _ = fn(
                q, k_t, v_t, g, beta,
                initial_state=initial,
                use_qk_l2norm_in_kernel=True, core_attn_out=core_attn_out,
            )
            flat = o2.reshape(-1)
            torch.testing.assert_close(
                core_attn_out[: flat.numel()], flat
            )


# ---------------------------------------------------------------------------
# fla/fused_recurrent.py
# ---------------------------------------------------------------------------


class TestFusedRecurrent:
    @pytest.mark.platforms("kunlunxin")
    def test_cu_seqlens_requires_batch_one(self):
        with _xtorch_ops_reimport(_STUB_REQUIRED) as (_, modules):
            fn = modules[_RECURRENT].fused_recurrent_gated_delta_rule
            q, k_t, v_t, g, beta = _fla_inputs(batch=2)
            with pytest.raises(ValueError, match="batch size is expected to be 1"):
                fn(q, k_t, v_t, g, beta, cu_seqlens=torch.tensor([0, 3, 6]))

    @pytest.mark.platforms("kunlunxin")
    def test_scale_defaults_and_positive_check(self):
        seen = {}

        def fwd_v2(q, k, v, g, beta, scale, initial_state, inplace, **kwargs):
            seen["scale"] = scale
            return torch.ones_like(v), initial_state

        with _xtorch_ops_reimport(
            _STUB_REQUIRED, fused_recurrent_gated_delta_rule_fwdv2=fwd_v2
        ) as (_, modules):
            fn = modules[_RECURRENT].fused_recurrent_gated_delta_rule
            q, k_t, v_t, g, beta = _fla_inputs()
            # K = 4 -> default scale = 4 ** -0.5
            fn(q, k_t, v_t, g, beta)
            assert seen["scale"] == pytest.approx(4 ** -0.5)
            # explicit scale is used verbatim
            fn(q, k_t, v_t, g, beta, scale=0.125)
            assert seen["scale"] == 0.125
            # non-positive scale trips the assert
            with pytest.raises(AssertionError, match="scale must be positive"):
                fn(q, k_t, v_t, g, beta, scale=-1.0)

    @pytest.mark.platforms("kunlunxin")
    def test_beta_none_becomes_ones(self):
        seen = {}

        def fwd_v2(q, k, v, g, beta, scale, initial_state, inplace, **kwargs):
            seen["beta"] = beta.clone()
            return torch.ones_like(v), initial_state

        with _xtorch_ops_reimport(
            _STUB_REQUIRED, fused_recurrent_gated_delta_rule_fwdv2=fwd_v2
        ) as (_, modules):
            fn = modules[_RECURRENT].fused_recurrent_gated_delta_rule
            q, k_t, v_t, g, beta = _fla_inputs()
            fn(q, k_t, v_t, g, beta=None)
            assert seen["beta"].shape == q[..., 0].shape
            assert torch.all(seen["beta"] == 1.0)
            assert seen["beta"].dtype == q.dtype

    @pytest.mark.platforms("kunlunxin")
    def test_fwd_kernel_contract(self):
        seen = {}

        def fwd_v2(
            q, k, v, g, beta, scale, initial_state, inplace,
            cu_seqlens=None, h0_indices=None, num_accepted_tokens=None,
            use_qk_l2norm_in_kernel=False, is_h0_transposed=True,
        ):
            seen.update(
                {
                    "q": q,
                    "k": k,
                    "v": v,
                    "g": g,
                    "beta": beta,
                    "scale": scale,
                    "initial_state": initial_state,
                    "inplace": inplace,
                    "cu_seqlens": cu_seqlens,
                    "h0_indices": h0_indices,
                    "num_accepted_tokens": num_accepted_tokens,
                    "l2norm": use_qk_l2norm_in_kernel,
                    "is_h0_transposed": is_h0_transposed,
                }
            )
            return torch.ones_like(v), initial_state

        with _xtorch_ops_reimport(
            _STUB_REQUIRED, fused_recurrent_gated_delta_rule_fwdv2=fwd_v2
        ) as (_, modules):
            fwd = modules[_RECURRENT].fused_recurrent_gated_delta_rule_fwd
            q, k_t, v_t, g, beta = _fla_inputs()
            # make q non-contiguous to observe the .contiguous() calls
            q_nc_base = torch.randn(1, 6, 2, 4, dtype=torch.bfloat16)
            q_nc = q_nc_base.transpose(1, 2).transpose(1, 2)  # still contiguous
            q_strided = q_nc_base[:, ::2]
            assert not q_strided.is_contiguous()
            initial = torch.randn(1, 2, 4, 4, dtype=torch.bfloat16)
            cu = torch.tensor([0, 3, 6])
            idx = torch.tensor([0, 1])
            nat = torch.tensor([1, 2])

            o, ht = fwd(
                q_strided, k_t, v_t, g, beta, scale=0.5,
                initial_state=initial, inplace_final_state=False,
                cu_seqlens=cu, ssm_state_indices=idx,
                num_accepted_tokens=nat, use_qk_l2norm_in_kernel=True,
            )

            # contiguity enforced for all five tensors
            for name in ("q", "k", "v", "g", "beta"):
                assert seen[name].is_contiguous(), f"{name} not contiguous"
            torch.testing.assert_close(seen["q"], q_strided.contiguous())
            assert seen["scale"] == 0.5
            assert seen["initial_state"] is initial
            assert seen["inplace"] is False
            assert seen["cu_seqlens"] is cu
            assert seen["h0_indices"] is idx
            assert seen["num_accepted_tokens"] is nat
            assert seen["l2norm"] is True
            # the cache-side layout flag is fixed by this impl
            assert seen["is_h0_transposed"] is True
            # outputs forwarded verbatim from the kernel
            assert o.shape == v_t.shape
            assert ht is initial


# ---------------------------------------------------------------------------
# causal_conv1d
# ---------------------------------------------------------------------------


class TestCausalConv1dFn:
    @pytest.mark.platforms("kunlunxin")
    def test_unsupported_activation_raises(self):
        with _xtorch_ops_reimport(_STUB_REQUIRED) as (_, modules):
            fn = modules[_CONV1D].causal_conv1d_fn_kunlunxin
            x = torch.randn(8, 4)
            w = torch.randn(4, 3)
            with pytest.raises(NotImplementedError, match="activation must be None, silu, or swish"):
                fn(
                    x, w, None, torch.randn(2, 4, 8),
                    torch.tensor([0, 4, 8]), activation="gelu",
                )

    @pytest.mark.platforms("kunlunxin")
    @pytest.mark.parametrize("activation", [None, "silu", "swish"])
    def test_host_lod_and_layout_forwarded(self, activation):
        seen = {}

        def causal_conv1d_fwd(
            x, weight, bias, conv_states, query_start_loc,
            cache_indices, has_initial_state, use_activation, is_ncw,
            query_start_loc_cpu, pad_slot_id,
        ):
            seen["x"] = x
            seen["bias"] = bias
            seen["use_activation"] = use_activation
            seen["is_ncw"] = is_ncw
            seen["lod"] = query_start_loc_cpu
            seen["pad_slot_id"] = pad_slot_id
            seen["query_start_loc"] = query_start_loc

        with _xtorch_ops_reimport(
            _STUB_REQUIRED, causal_conv1d_fwd=causal_conv1d_fwd
        ) as (_, modules):
            fn = modules[_CONV1D].causal_conv1d_fn_kunlunxin
            x = torch.randn(8, 4)
            weight = torch.randn(4, 3)
            bias = torch.randn(4)
            conv_states = torch.randn(2, 4, 8)
            qsl = torch.tensor([0, 4, 8])

            result = fn(
                x, weight, bias, conv_states, qsl, activation=activation,
                pad_slot_id=-2,
            )

            assert result is x
            assert seen["x"] is x
            assert seen["x"].is_contiguous()
            assert seen["bias"] is bias
            assert seen["bias"].is_contiguous()
            assert seen["use_activation"] is (activation in ["silu", "swish"])
            assert seen["is_ncw"] is False
            # host-side lod: python list from query_start_loc.cpu().tolist()
            assert seen["lod"] == [0, 4, 8]
            assert not torch.is_tensor(seen["lod"])
            assert seen["pad_slot_id"] == -2
            assert seen["query_start_loc"] is qsl

    @pytest.mark.platforms("kunlunxin")
    def test_noncontiguous_x_gets_contiguous_copy(self):
        seen = {}

        def causal_conv1d_fwd(
            x, weight, bias, conv_states, query_start_loc,
            cache_indices, has_initial_state, use_activation, is_ncw,
            query_start_loc_cpu, pad_slot_id,
        ):
            seen["x"] = x

        with _xtorch_ops_reimport(
            _STUB_REQUIRED, causal_conv1d_fwd=causal_conv1d_fwd
        ) as (_, modules):
            fn = modules[_CONV1D].causal_conv1d_fn_kunlunxin
            base = torch.randn(4, 8)
            x = base.t()  # (8, 4) but non-contiguous
            assert not x.is_contiguous()
            x_snapshot = x.clone()
            result = fn(
                x, torch.randn(4, 3), None, torch.randn(2, 4, 8),
                torch.tensor([0, 8]), activation=None,
            )
            assert seen["x"].is_contiguous()
            assert seen["x"] is not x
            torch.testing.assert_close(seen["x"], x.contiguous())
            # the returned tensor carries the (unchanged) input values with
            # the input's shape; on this path the kernel is a no-op stub, so
            # the wrapper hands back the contiguous copy it made
            assert result.shape == x.shape
            torch.testing.assert_close(result, x_snapshot)


class TestCausalConv1dUpdate:
    def _noop_kernel(self, seen):
        def causal_conv1d_update(
            x, conv_state, weight, bias, use_activation,
            cache_seqlens, conv_state_indices, intermediate_conv_window,
            is_ncw, pad_slot_id,
        ):
            seen["x"] = x
            seen["conv_state"] = conv_state
            seen["weight"] = weight
            seen["bias"] = bias
            seen["use_activation"] = use_activation
            seen["cache_seqlens"] = cache_seqlens
            seen["conv_state_indices"] = conv_state_indices
            seen["intermediate_conv_window"] = intermediate_conv_window
            seen["is_ncw"] = is_ncw
            seen["pad_slot_id"] = pad_slot_id

        return causal_conv1d_update

    @pytest.mark.platforms("kunlunxin")
    def test_rejects_speculative_decoding(self):
        with _xtorch_ops_reimport(_STUB_REQUIRED) as (_, modules):
            fn = modules[_CONV1D].causal_conv1d_update_kunlunxin
            with pytest.raises(NotImplementedError, match="speculative decoding"):
                fn(
                    torch.randn(2, 4), torch.randn(2, 4, 3), torch.randn(4, 3),
                    num_accepted_tokens=torch.tensor([1, 1]),
                )

    @pytest.mark.platforms("kunlunxin")
    def test_rejects_varlen_query_start_loc(self):
        with _xtorch_ops_reimport(_STUB_REQUIRED) as (_, modules):
            fn = modules[_CONV1D].causal_conv1d_update_kunlunxin
            with pytest.raises(NotImplementedError, match="varlen with query_start_loc"):
                fn(
                    torch.randn(2, 4), torch.randn(2, 4, 3), torch.randn(4, 3),
                    query_start_loc=torch.tensor([0, 1, 2]), max_query_len=4,
                )

    @pytest.mark.platforms("kunlunxin")
    def test_varlen_without_max_query_len_is_allowed(self):
        seen = {}
        kernel = self._noop_kernel(seen)
        with _xtorch_ops_reimport(
            _STUB_REQUIRED, causal_conv1d_update=kernel
        ) as (_, modules):
            fn = modules[_CONV1D].causal_conv1d_update_kunlunxin
            # query_start_loc set but max_query_len unset (-1/0): not varlen
            fn(
                torch.randn(2, 4), torch.randn(2, 4, 3), torch.randn(4, 3),
                query_start_loc=torch.tensor([0, 1, 2]), max_query_len=-1,
            )
            assert "x" in seen

    @pytest.mark.platforms("kunlunxin")
    def test_rejects_bad_activation(self):
        with _xtorch_ops_reimport(_STUB_REQUIRED) as (_, modules):
            fn = modules[_CONV1D].causal_conv1d_update_kunlunxin
            with pytest.raises(NotImplementedError, match="actual: gelu"):
                fn(
                    torch.randn(2, 4), torch.randn(2, 4, 3), torch.randn(4, 3),
                    activation="gelu",
                )

    @pytest.mark.platforms("kunlunxin")
    def test_dim2_unsqueezes_and_restores_values(self):
        seen = {}
        kernel = self._noop_kernel(seen)
        with _xtorch_ops_reimport(
            _STUB_REQUIRED, causal_conv1d_update=kernel
        ) as (_, modules):
            fn = modules[_CONV1D].causal_conv1d_update_kunlunxin
            x = torch.randn(3, 5)  # [batch, dim], dim()==2
            x_snapshot = x.clone()
            conv_state = torch.randn(4, 5, 3)
            weight = torch.randn(5, 3)
            bias = torch.randn(5)
            result = fn(x, conv_state, weight, bias, activation="silu")

            # kernel saw the [batch, 1, dim] unsqueeze
            assert seen["x"].shape == (3, 1, 5)
            assert seen["x"].is_contiguous()
            assert seen["conv_state"] is conv_state
            assert seen["weight"] is weight
            assert seen["bias"] is bias
            assert seen["use_activation"] is True
            assert seen["is_ncw"] is False
            # the caller gets the squeezed shape back with unchanged values
            # (unsqueeze->kernel->squeeze produces a fresh but equal tensor)
            assert result.shape == (3, 5)
            torch.testing.assert_close(result, x_snapshot)

    @pytest.mark.platforms("kunlunxin")
    def test_dim3_transpose_round_trip(self):
        seen = {}
        kernel = self._noop_kernel(seen)
        with _xtorch_ops_reimport(
            _STUB_REQUIRED, causal_conv1d_update=kernel
        ) as (_, modules):
            fn = modules[_CONV1D].causal_conv1d_update_kunlunxin
            # [batch, dim, seqlen] input; kernel receives [batch, seqlen, dim]
            x = torch.randn(3, 5, 2)
            conv_state = torch.randn(4, 5, 3)
            weight = torch.randn(5, 3)
            result = fn(x, conv_state, weight, activation=None)

            assert seen["x"].shape == (3, 2, 5)
            assert seen["x"].is_contiguous()
            # the returned tensor is the transposed-back layout
            assert result.shape == (3, 5, 2)
            torch.testing.assert_close(result, x)
            assert seen["use_activation"] is False

    @pytest.mark.platforms("kunlunxin")
    def test_noncontiguous_3d_gets_contiguous_transposed_copy(self):
        seen = {}

        def causal_conv1d_update(
            x, conv_state, weight, bias, use_activation,
            cache_seqlens, conv_state_indices, intermediate_conv_window,
            is_ncw, pad_slot_id,
        ):
            seen["x"] = x

        with _xtorch_ops_reimport(
            _STUB_REQUIRED, causal_conv1d_update=causal_conv1d_update
        ) as (_, modules):
            fn = modules[_CONV1D].causal_conv1d_update_kunlunxin
            base = torch.randn(3, 2, 5)
            x = base.transpose(1, 2)  # [3, 5, 2], non-contiguous
            assert not x.is_contiguous()
            result = fn(x, torch.randn(4, 5, 3), torch.randn(5, 3), activation=None)
            # kernel saw a dense [batch, seqlen, dim] tensor
            assert seen["x"].is_contiguous()
            assert seen["x"].shape == (3, 2, 5)
            torch.testing.assert_close(seen["x"], x.transpose(1, 2).contiguous())
            # result is transposed back and numerically equal
            assert result.shape == (3, 5, 2)
            torch.testing.assert_close(result, x)
