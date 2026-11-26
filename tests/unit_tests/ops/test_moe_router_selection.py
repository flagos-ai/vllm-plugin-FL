# Copyright (c) 2026 BAAI. All rights reserved.

"""Router math and dispatch tests for ``vllm_fl.ops.fused_moe.router``.

``_sqrtsoftplus_topk``'s DSV4 guard edges (NaN selection, the zero
denominator guard, indices dtype, hash-table dtype casting), ``_has_native_moe_topk``,
``_fl_grouped_topk`` numeric math (checked against independent plain-Python
oracles), both FL router ``_compute_routing`` methods, and
``replace_router_with_fl`` — the parts of the router module the core suite
leaves uncovered.

The sqrtsoftplus basics (biased top-k oracle, renormalize + scaling, and the
hash-table selection values) are already covered by
``tests/unit_tests/ops/test_fused_moe_router.py`` and are not duplicated here.

The module imports ``vllm`` at module level.  On machines without the real
package a stub ``vllm`` graph is installed first (external boundary only);
with the real ``vllm`` present the module imports against it.  Either
way, every platform probe, kernel class, and dispatched op is swapped at
test time via ``monkeypatch`` on module attributes, so the code under test
always runs its own logic.  Numeric expectations are computed independently
with plain-Python softmax / softplus arithmetic; where logits tie exactly
only the contract — distinct indices sharing the tied probability — is
asserted, never the order.
"""

import math
import sys
import types
from typing import Any

import pytest
import torch

# ---------------------------------------------------------------------------
# Windows (no real vllm): install the external vllm boundary the fused_moe
# modules read at import time.  With the real vllm installed the modules
# import against it directly and this block is skipped.
# ---------------------------------------------------------------------------

_HAS_VLLM = True
try:  # pragma: no cover - dual-environment guard
    import vllm  # noqa: F401
except ImportError:  # pragma: no cover - Windows machines without vllm
    _HAS_VLLM = False


def _make_module(name: str, **attrs: Any) -> types.ModuleType:
    """Create a plain module with the given attributes attached."""
    module = types.ModuleType(name)
    module.__path__ = []
    for key, value in attrs.items():
        setattr(module, key, value)
    return module


class _StubLogger:
    """vllm.logger stand-in supporting the *_once APIs the selector uses.

    Records are forwarded to the standard library's module-level logger so
    pytest's ``caplog`` can observe the selection messages.
    """

    def __init__(self):
        import logging

        self._log = logging.getLogger("vllm_stub")

    def info(self, msg, *args):
        self._log.info(msg % args if args else msg)

    def debug(self, msg, *args):
        self._log.debug(msg % args if args else msg)

    def warning(self, msg, *args):
        self._log.warning(msg % args if args else msg)

    def info_once(self, msg, *args):
        self._log.info(msg % args if args else msg)

    def debug_once(self, msg, *args):
        self._log.debug(msg % args if args else msg)

    def warning_once(self, msg, *args):
        self._log.warning(msg % args if args else msg)


def _install_stub_vllm_graph() -> None:
    """Install the fake vllm graph the target modules import at module level.

    Only names read at import time are provided (enums, config classes,
    helper signatures); every function under test still runs its own logic.
    """
    import enum

    class _StubMoEActivation(enum.Enum):
        SILU = "silu"
        GELU = "gelu"
        SILU_NO_MUL = "silu_no_mul"
        GELU_NO_MUL = "gelu_no_mul"
        GELU_TANH = "gelu_tanh"
        RELU2 = "relu2"
        RELU2_NO_MUL = "relu2_no_mul"
        SITU = "situ"
        SWIGLUOAI = "swigluoai"
        SWIGLUOAI_UNINTERLEAVE = "swigluoai_uninterleave"
        SWIGLUSTEP = "swiglustep"

        @property
        def is_gated(self):
            return not self.name.endswith("NO_MUL")

    class _StubBackend(enum.Enum):
        AITER = "ROCm AITER"
        TRITON = "TRITON"
        BATCHED_TRITON = "BATCHED_TRITON"
        FLASHINFER_TRTLLM = "FlashInfer TRTLLM"
        FLASHINFER_CUTLASS = "FlashInfer CUTLASS"
        XPU = "XPU"
        CPU = "CPU"
        TPU = "TPU"

    class _StubActivationFormat(enum.Enum):
        BatchedExperts = "batched"
        Standard = "standard"

    vllm = _make_module("vllm")
    platforms = _make_module(
        "vllm.platforms",
        current_platform=types.SimpleNamespace(
            device_type="cpu", vendor_name=None, is_cuda=lambda: False
        ),
    )
    logger_mod = _make_module("vllm.logger", init_logger=lambda name: _StubLogger())
    envs_mod = _make_module(
        "vllm.envs",
        is_set=lambda name: False,
        VLLM_ROCM_USE_AITER=False,
        VLLM_ROCM_USE_AITER_MOE=False,
    )
    aiter_ops = _make_module(
        "vllm._aiter_ops",
        # The namespace the FL router probes at call time; parity with the
        # real vllm._aiter_ops surface so monkeypatch targets always exist.
        rocm_aiter_ops=types.SimpleNamespace(
            is_rdna_aiter_enabled=lambda: False,
            is_fused_moe_enabled=lambda: False,
            is_fusion_moe_shared_experts_enabled=lambda: False,
        ),
    )
    me = _make_module("vllm.model_executor")
    layers = _make_module("vllm.model_executor.layers")
    fm = _make_module("vllm.model_executor.layers.fused_moe")
    fm.FusedMoE = object
    fm.FusedMoEFactory = None
    experts_dir = _make_module("vllm.model_executor.layers.fused_moe.experts")
    router_dir = _make_module("vllm.model_executor.layers.fused_moe.router")
    runner_dir = _make_module("vllm.model_executor.layers.fused_moe.runner")
    oracle_dir = _make_module("vllm.model_executor.layers.fused_moe.oracle")
    fm_config = _make_module(
        "vllm.model_executor.layers.fused_moe.config",
        FusedMoEConfig=object,
        FUSED_MOE_UNQUANTIZED_CONFIG=None,
        FusedMoEQuantConfig=object,
        _get_config_dtype_str=lambda **kw: "float32",
    )
    fm_runner = _make_module(
        "vllm.model_executor.layers.fused_moe.runner.moe_runner", MoERunner=object
    )
    unq_method = _make_module(
        "vllm.model_executor.layers.fused_moe.unquantized_fused_moe_method",
        UnquantizedFusedMoEMethod=object,
        select_unquantized_moe_backend=lambda *a, **k: None,
    )
    fm_utils_v = _make_module(
        "vllm.model_executor.layers.fused_moe.utils",
        _resize_cache=lambda t, shape: t.view(*shape),
        moe_kernel_quantize_input=lambda A, *a, **kw: (A, None),
    )
    fm_activation = _make_module(
        "vllm.model_executor.layers.fused_moe.activation",
        MoEActivation=_StubMoEActivation,
        ApplyMoEActivationConfig=object,
        apply_moe_activation=lambda *a, **k: None,
    )
    fm_fused_v = _make_module(
        "vllm.model_executor.layers.fused_moe.fused_moe",
        try_get_optimal_moe_config=lambda *a, **k: {"BLOCK_SIZE_M": 16},
        _get_config_quant_dtype=lambda **kw: None,
        invoke_fused_moe_triton_kernel=lambda *a, **k: None,
    )
    oracle_unq = _make_module(
        "vllm.model_executor.layers.fused_moe.oracle.unquantized",
        UnquantizedMoeBackend=_StubBackend,
        _trtllm_bf16_lora_supported=lambda *a, **k: False,
        backend_to_kernel_cls=lambda backend: [],
        map_unquantized_backend=lambda name: None,
        select_unquantized_moe_backend=lambda *a, **k: None,
    )
    modular = _make_module(
        "vllm.model_executor.layers.fused_moe.modular_kernel",
        FusedMoEExperts=object,
        ExpertTokensMetadata=object,
        FusedMoEActivationFormat=_StubActivationFormat,
    )

    class _StubTritonExperts:
        """Upstream TritonExperts stand-in accepting the constructor args."""

        def __init__(self, moe_config, *args, **kwargs):
            self.moe_config = moe_config

        def workspace_shapes(self, *args, **kwargs):
            raise AssertionError("upstream workspace_shapes reached")

        def apply(self, *args, **kwargs):
            raise AssertionError("upstream apply reached")

    experts_triton = _make_module(
        "vllm.model_executor.layers.fused_moe.experts.triton_moe",
        TritonExperts=_StubTritonExperts,
    )
    experts_trtllm = _make_module(
        "vllm.model_executor.layers.fused_moe.experts.trtllm_lora_moe",
        TrtLlmBf16LoRAExperts=type("TrtLlmBf16LoRAExperts", (object,), {}),
    )
    aiter_rocm = _make_module(
        "vllm.model_executor.layers.fused_moe.experts.rocm_aiter_moe",
        rocm_aiter_grouped_topk=lambda *a, **k: None,
    )

    class _StubUpstreamRouter:
        """Upstream router stand-in: replace_router_with_fl overwrites this."""

        def _compute_routing(self, *args, **kwargs):
            raise AssertionError("unpatched upstream router reached")

    router_fused = _make_module(
        "vllm.model_executor.layers.fused_moe.router.fused_topk_router",
        FusedTopKRouter=type("FusedTopKRouter", (_StubUpstreamRouter,), {}),
        _get_padding_mask=lambda n: None,
        fused_topk=lambda **k: None,
    )
    router_bias = _make_module(
        "vllm.model_executor.layers.fused_moe.router.fused_topk_bias_router",
        FusedTopKBiasRouter=type("FusedTopKBiasRouter", (_StubUpstreamRouter,), {}),
        fused_topk_bias=lambda **k: None,
    )
    router_grouped = _make_module(
        "vllm.model_executor.layers.fused_moe.router.grouped_topk_router",
        GroupedTopKRouter=type("GroupedTopKRouter", (_StubUpstreamRouter,), {}),
    )
    triton_utils = _make_module(
        "vllm.triton_utils",
        tl=types.SimpleNamespace(bfloat16="bf16", float16="f16", float32="f32"),
        triton=None,
    )

    vllm.platforms = platforms
    vllm.model_executor = me
    vllm.envs = envs_mod
    me.layers = layers
    layers.fused_moe = fm
    fm.experts = experts_dir
    fm.router = router_dir
    fm.config = fm_config
    fm.runner = runner_dir
    runner_dir.moe_runner = fm_runner
    fm.unquantized_fused_moe_method = unq_method
    fm.utils = fm_utils_v
    fm.activation = fm_activation
    fm.oracle = oracle_dir
    oracle_dir.unquantized = oracle_unq
    fm.fused_moe = fm_fused_v

    for name, module in {
        "vllm": vllm,
        "vllm.platforms": platforms,
        "vllm.logger": logger_mod,
        "vllm.envs": envs_mod,
        "vllm._aiter_ops": aiter_ops,
        "vllm.model_executor": me,
        "vllm.model_executor.layers": layers,
        "vllm.model_executor.layers.fused_moe": fm,
        "vllm.model_executor.layers.fused_moe.config": fm_config,
        "vllm.model_executor.layers.fused_moe.runner": runner_dir,
        "vllm.model_executor.layers.fused_moe.runner.moe_runner": fm_runner,
        "vllm.model_executor.layers.fused_moe.unquantized_fused_moe_method": unq_method,
        "vllm.model_executor.layers.fused_moe.utils": fm_utils_v,
        "vllm.model_executor.layers.fused_moe.activation": fm_activation,
        "vllm.model_executor.layers.fused_moe.oracle": oracle_dir,
        "vllm.model_executor.layers.fused_moe.oracle.unquantized": oracle_unq,
        "vllm.model_executor.layers.fused_moe.modular_kernel": modular,
        "vllm.model_executor.layers.fused_moe.fused_moe": fm_fused_v,
        "vllm.model_executor.layers.fused_moe.experts": experts_dir,
        "vllm.model_executor.layers.fused_moe.experts.triton_moe": experts_triton,
        "vllm.model_executor.layers.fused_moe.experts.trtllm_lora_moe": experts_trtllm,
        "vllm.model_executor.layers.fused_moe.experts.rocm_aiter_moe": aiter_rocm,
        "vllm.model_executor.layers.fused_moe.router": router_dir,
        "vllm.model_executor.layers.fused_moe.router.fused_topk_router": router_fused,
        "vllm.model_executor.layers.fused_moe.router.fused_topk_bias_router": router_bias,
        "vllm.model_executor.layers.fused_moe.router.grouped_topk_router": router_grouped,
        "vllm.triton_utils": triton_utils,
    }.items():
        sys.modules[name] = module


# Snapshot every pre-existing vllm/vllm_fl entry, install the stub graph
# when the real vllm is absent, import the module under test, then
# restore. The imports stay alive after the restore because the tested code
# lives in the vllm_fl modules themselves, which remain in sys.modules; but
# nothing of the stub graph leaks into sys.modules for later files — without
# the restore, a later file's ``import vllm`` probe would succeed against the
# stubs and files gated on HAS_VLLM would collect stale/partial modules
# (e.g. vllm.model_executor.layers without rotary_embedding).
_STUB_ROOTS = ("vllm", "vllm_fl")
_SAVED = {
    name: sys.modules.get(name, None)
    for name in list(sys.modules)
    if name.split(".")[0] in _STUB_ROOTS
}
if not _HAS_VLLM:  # pragma: no cover - Windows machines without vllm
    _install_stub_vllm_graph()
try:
    import vllm_fl.ops.fused_moe.router as fl_router  # noqa: E402
finally:
    if not _HAS_VLLM:  # pragma: no cover - restore only what stubs touched
        for name in [
            name
            for name in list(sys.modules)
            if name.split(".")[0] in _STUB_ROOTS and name not in _SAVED
        ]:
            sys.modules.pop(name, None)
        for name, module in _SAVED.items():
            if module is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = module


class _RecordingCachedOp:
    """A stand-in for CachedOp instances recording every dispatched call.

    The modules under test call their module-level ``CachedOp`` handles as
    plain callables (``_moe_sum(input, output)``), so replacing the module
    attribute with an instance of this class preserves the call contract.
    """

    def __init__(self, impl):
        self.calls = []
        self._impl = impl

    def __call__(self, *args, **kwargs):
        self.calls.append((args, kwargs))
        if self._impl is None:
            return None
        return self._impl(*args, **kwargs)


def _replace_cached_op(monkeypatch, module, attr, impl):
    """Swap a module's CachedOp attribute for a recording stand-in."""
    recorder = _RecordingCachedOp(impl)
    monkeypatch.setattr(module, attr, recorder, raising=False)
    return recorder

# ---------------------------------------------------------------------------
# _sqrtsoftplus_topk
# ---------------------------------------------------------------------------


class TestSqrtsoftplusTopk:
    """The pure-torch DSV4 sqrt(softplus) routing fallback."""

    @staticmethod
    def _call(logits, **kwargs):
        return fl_router._sqrtsoftplus_topk(
            gating_output=logits,
            e_score_correction_bias=kwargs.get("bias"),
            topk=kwargs.get("topk", 2),
            renormalize=kwargs.get("renormalize", False),
            indices_type=kwargs.get("indices_type"),
            input_tokens=kwargs.get("input_tokens"),
            hash_indices_table=kwargs.get("hash_indices_table"),
            routed_scaling_factor=kwargs.get("routed_scaling_factor", 1.0),
        )

    def test_nan_logits_score_last_for_selection_only(self):
        """NaN logits score as -1e30 for selection; weights stay raw scores."""
        logits = torch.tensor([[float("nan"), 1.0, 2.0]], dtype=torch.float32)
        weights, ids = self._call(logits, topk=2)
        assert set(ids[0].tolist()) == {1, 2}  # the NaN expert is never selected
        sp = [math.log1p(math.exp(v)) for v in (1.0, 2.0)]
        expected = sorted([math.sqrt(v) for v in sp], reverse=True)
        got = sorted(weights[0].tolist(), reverse=True)
        for exp, got_v in zip(expected, got):
            torch.testing.assert_close(got_v, exp, rtol=1e-5, atol=1e-6)

    def test_zero_denominator_guard(self):
        """A non-positive denominator is replaced by 1, not NaN."""
        # softplus(-inf) = 0, so the renormalize denominator is 0 and the
        # torch.where guard replaces it with 1: weights stay 0, not NaN.
        logits = torch.tensor([[-float("inf"), -float("inf")]])
        weights, ids = self._call(logits, topk=2, renormalize=True)
        assert len(set(ids[0].tolist())) == 2
        torch.testing.assert_close(weights, torch.zeros(1, 2))

    def test_indices_dtype_default_and_override(self):
        """indices_type=None writes int32; an explicit dtype is honoured."""
        logits = torch.tensor([[0.0, 1.0]])
        _w, ids_default = self._call(logits, topk=1)
        assert ids_default.dtype == torch.int32
        _w, ids_i64 = self._call(logits, topk=1, indices_type=torch.int64)
        assert ids_i64.dtype == torch.int64

    def test_hash_table_casts_input_token_dtype(self):
        """input_tokens are cast to the hash table's dtype before lookup."""
        hash_table = torch.tensor([[1, 0], [2, 0]], dtype=torch.int32)
        weights, ids = self._call(
            torch.tensor([[0.0, 9.0, 4.0]]),
            topk=2,
            input_tokens=torch.tensor([1], dtype=torch.int64),
            hash_indices_table=hash_table,
        )
        assert ids.dtype == torch.int32
        assert set(ids[0].tolist()) == {2, 0}
        assert weights.shape == (1, 2)


# ---------------------------------------------------------------------------
# _has_native_moe_topk
# ---------------------------------------------------------------------------


class TestHasNativeMoeTopk:
    """The torch.ops._moe_C.topk_softmax capability probe."""

    def test_reports_false_without_topk_softmax(self, monkeypatch):
        """A _moe_C namespace lacking topk_softmax keeps the probe False."""
        monkeypatch.setattr(torch.ops, "_moe_C", types.SimpleNamespace(), raising=False)
        assert fl_router._has_native_moe_topk() is False

    def test_reports_true_with_topk_softmax(self, monkeypatch):
        """A _moe_C namespace exposing topk_softmax makes the probe True."""
        monkeypatch.setattr(
            torch.ops,
            "_moe_C",
            types.SimpleNamespace(topk_softmax=lambda *a: None),
            raising=False,
        )
        assert fl_router._has_native_moe_topk() is True


# ---------------------------------------------------------------------------
# _fl_grouped_topk: numeric math (pure torch, no dispatch)
# ---------------------------------------------------------------------------


class TestFlGroupedTopk:
    """The grouped top-k router math against independent plain-Python oracles."""

    def test_softmax_no_bias_matches_group_oracle(self):
        """softmax scoring without bias matches the plain-loop oracle."""
        torch.manual_seed(7)
        gating = torch.randn(4, 8, dtype=torch.float32) * 1.5
        weights, ids = fl_router._fl_grouped_topk(
            hidden_states=torch.zeros(4, 3),
            gating_output=gating,
            topk=2,
            renormalize=True,
            num_expert_group=2,
            topk_group=1,
        )
        rows = gating.tolist()
        for row in range(4):
            # Independent oracle: softmax, per-group max, mask, topk.
            exps = [math.exp(v) for v in rows[row]]
            denom = sum(exps)
            probs = [e / denom for e in exps]
            group_scores = [max(probs[:4]), max(probs[4:])]
            chosen = 0 if group_scores[0] >= group_scores[1] else 1
            allowed = set(range(chosen * 4, chosen * 4 + 4))
            picked = sorted(((probs[i], i) for i in allowed), key=lambda t: -t[0])[:2]
            assert set(ids[row].tolist()) == {t[1] for t in picked}
            total = sum(t[0] for t in picked)
            got = {ids[row, s].item(): weights[row, s].item() for s in range(2)}
            for prob, idx in picked:
                torch.testing.assert_close(got[idx], prob / total, rtol=1e-4, atol=1e-6)

    def test_sigmoid_no_bias_matches_group_oracle(self):
        """sigmoid scoring without bias matches the plain-loop oracle."""
        torch.manual_seed(11)
        gating = torch.randn(3, 9, dtype=torch.float32)
        weights, ids = fl_router._fl_grouped_topk(
            hidden_states=torch.zeros(3, 3),
            gating_output=gating,
            topk=3,
            renormalize=False,
            num_expert_group=3,
            topk_group=2,
            scoring_func="sigmoid",
        )
        rows = gating.tolist()
        for row in range(3):
            probs = [1.0 / (1.0 + math.exp(-v)) for v in rows[row]]
            group_scores = [max(probs[g * 3 : g * 3 + 3]) for g in range(3)]
            chosen = sorted(range(3), key=lambda g: -group_scores[g])[:2]
            allowed = {g * 3 + j for g in chosen for j in range(3)}
            picked = sorted(((probs[i], i) for i in allowed), key=lambda t: -t[0])[:3]
            assert set(ids[row].tolist()) == {t[1] for t in picked}
            got = {ids[row, s].item(): weights[row, s].item() for s in range(3)}
            for prob, idx in picked:
                torch.testing.assert_close(got[idx], prob, rtol=1e-4, atol=1e-6)

    def test_unknown_scoring_func_raises(self):
        """An unsupported scoring function raises ValueError (no bias)."""
        with pytest.raises(ValueError, match="Unsupported scoring function"):
            fl_router._fl_grouped_topk(
                hidden_states=torch.zeros(1, 3),
                gating_output=torch.zeros(1, 8),
                topk=1,
                renormalize=False,
                num_expert_group=2,
                topk_group=1,
                scoring_func="bogus",
            )

    def test_unknown_scoring_func_with_bias_raises(self):
        """The biased path also rejects unsupported scoring functions."""
        with pytest.raises(ValueError, match="Unsupported scoring function"):
            fl_router._fl_grouped_topk(
                hidden_states=torch.zeros(1, 3),
                gating_output=torch.zeros(1, 8),
                topk=1,
                renormalize=False,
                num_expert_group=2,
                topk_group=1,
                scoring_func="bogus",
                e_score_correction_bias=torch.zeros(8),
            )

    def test_routed_scaling_factor_scales_weights(self):
        """routed_scaling_factor multiplies the selected weights."""
        torch.manual_seed(13)
        gating = torch.randn(2, 8, dtype=torch.float32)
        base_w, base_i = fl_router._fl_grouped_topk(
            hidden_states=torch.zeros(2, 3),
            gating_output=gating,
            topk=2,
            renormalize=False,
            num_expert_group=2,
            topk_group=1,
        )
        scaled_w, scaled_i = fl_router._fl_grouped_topk(
            hidden_states=torch.zeros(2, 3),
            gating_output=gating,
            topk=2,
            renormalize=False,
            num_expert_group=2,
            topk_group=1,
            routed_scaling_factor=2.5,
        )
        # Tie order is unspecified: compare value multisets per row.
        for row in range(2):
            assert set(scaled_i[row].tolist()) == set(base_i[row].tolist())
            for b, s in zip(
                sorted(base_w[row].tolist()), sorted(scaled_w[row].tolist())
            ):
                torch.testing.assert_close(s, b * 2.5, rtol=1e-4, atol=1e-6)

    def test_token_count_mismatch_raises(self):
        """hidden_states / gating_output row mismatch fails the assert."""
        with pytest.raises(AssertionError, match="Number of tokens mismatch"):
            fl_router._fl_grouped_topk(
                hidden_states=torch.zeros(2, 3),
                gating_output=torch.zeros(3, 8),
                topk=1,
                renormalize=False,
                num_expert_group=2,
                topk_group=1,
            )

    def test_output_dtypes_are_normalized(self):
        """Weights come back float32 and ids int32 regardless of input dtype."""
        weights, ids = fl_router._fl_grouped_topk(
            hidden_states=torch.zeros(1, 3),
            gating_output=torch.zeros(1, 8, dtype=torch.float64),
            topk=1,
            renormalize=False,
            num_expert_group=2,
            topk_group=1,
        )
        assert weights.dtype == torch.float32
        assert ids.dtype == torch.int32


# ---------------------------------------------------------------------------
# _fl_grouped_topk biased path (dispatched through the CachedOp boundary)
# ---------------------------------------------------------------------------


class TestFlGroupedTopkBiased:
    """The e_score_correction_bias path routed through the grouped_topk op."""

    def test_sigmoid_bias_forwards_to_grouped_topk_dispatch(self, monkeypatch):
        """sigmoid+bias hands the raw logits to grouped_topk with mode 1."""
        captured = {}

        def fake_grouped_topk(
            scores,
            num_expert_group,
            topk_group,
            topk,
            renormalize,
            scaling,
            bias,
            mode,
        ):
            captured.update(
                scores=scores,
                num_expert_group=num_expert_group,
                topk_group=topk_group,
                topk=topk,
                renormalize=renormalize,
                scaling=scaling,
                bias=bias,
                mode=mode,
            )
            return (
                torch.tensor([[0.5, 0.25]]),
                torch.tensor([[1, 0]], dtype=torch.int32),
            )

        _replace_cached_op(monkeypatch, fl_router, "_grouped_topk", fake_grouped_topk)
        bias = torch.tensor([0.1, -0.1])
        weights, ids = fl_router._fl_grouped_topk(
            hidden_states=torch.zeros(1, 3),
            gating_output=torch.tensor([[0.3, 0.7]]),
            topk=2,
            renormalize=False,
            num_expert_group=1,
            topk_group=1,
            scoring_func="sigmoid",
            e_score_correction_bias=bias,
        )
        torch.testing.assert_close(weights, torch.tensor([[0.5, 0.25]]))
        assert ids.tolist() == [[1, 0]]
        # Sigmoid mode is encoded as 1; the raw logits are handed over.
        assert captured["mode"] == 1
        assert captured["num_expert_group"] == 1
        assert captured["bias"] is bias
        torch.testing.assert_close(captured["scores"], torch.tensor([[0.3, 0.7]]))

    def test_softmax_bias_applies_softmax_first(self, monkeypatch):
        """softmax+bias softmaxes the logits before the dispatched call."""
        captured = {}

        def fake_grouped_topk(
            scores,
            num_expert_group,
            topk_group,
            topk,
            renormalize,
            scaling,
            bias,
            mode,
        ):
            captured.update(scores=scores, mode=mode)
            return torch.tensor([[0.5]]), torch.tensor([[0]], dtype=torch.int32)

        _replace_cached_op(monkeypatch, fl_router, "_grouped_topk", fake_grouped_topk)
        fl_router._fl_grouped_topk(
            hidden_states=torch.zeros(1, 3),
            gating_output=torch.tensor([[0.0, math.log(3.0)]]),
            topk=1,
            renormalize=False,
            num_expert_group=1,
            topk_group=1,
            scoring_func="softmax",
            e_score_correction_bias=torch.zeros(2),
        )
        assert captured["mode"] == 0
        # Independent oracle: softmax([0, ln3]) = [1/4, 3/4].
        torch.testing.assert_close(
            captured["scores"], torch.tensor([[0.25, 0.75]]), rtol=1e-6, atol=1e-6
        )

    def test_dispatched_result_is_returned_verbatim(self, monkeypatch):
        """The dispatched kernel's outputs are returned unchanged."""
        sentinel_w = torch.tensor([[0.125]])
        sentinel_i = torch.tensor([[3]], dtype=torch.int32)
        _replace_cached_op(
            monkeypatch,
            fl_router,
            "_grouped_topk",
            lambda *a, **k: (sentinel_w, sentinel_i),
        )
        weights, ids = fl_router._fl_grouped_topk(
            hidden_states=torch.zeros(1, 3),
            gating_output=torch.zeros(1, 4),
            topk=1,
            renormalize=False,
            num_expert_group=1,
            topk_group=1,
            scoring_func="sigmoid",
            e_score_correction_bias=torch.zeros(4),
        )
        assert weights is sentinel_w
        assert ids is sentinel_i


# ---------------------------------------------------------------------------
# GroupedTopKRouterFL._compute_routing
# ---------------------------------------------------------------------------


class TestGroupedTopKRouterComputeRouting:
    """GroupedTopKRouterFL's grouping-validity gate and dispatch choice."""

    def _router(self, **overrides):
        """A router instance answering only the routing probes."""
        params = dict(
            top_k=2,
            renormalize=True,
            num_expert_group=2,
            topk_group=1,
            scoring_func="softmax",
            routed_scaling_factor=1.0,
            e_score_correction_bias=None,
            num_fused_shared_experts=0,
        )
        params.update(overrides)
        router = types.SimpleNamespace(**params)
        # Invoke the FL method unbound, exactly as the class-level patch does.
        router._compute_routing = (
            fl_router.GroupedTopKRouterFL._compute_routing.__get__(router)
        )
        return router

    def test_invalid_grouping_without_bias_uses_fused_topk(self, monkeypatch):
        """num_experts <= num_expert_group routes through plain fused_topk."""
        router = self._router()
        recorded = {}

        def fake_fused_topk(**kwargs):
            recorded.update(kwargs)
            return (
                torch.tensor([[0.5, 0.5]]),
                torch.tensor([[0, 1]], dtype=torch.int32),
                None,
            )

        monkeypatch.setattr(fl_router, "fused_topk", fake_fused_topk)
        # 2 logits columns <= num_expert_group=2 makes the grouping invalid.
        weights, ids = router._compute_routing(
            torch.zeros(1, 3), torch.zeros(1, 2), None
        )
        assert recorded["topk"] == 2
        assert recorded["renormalize"] is True
        torch.testing.assert_close(weights, torch.tensor([[0.5, 0.5]]))
        assert ids.tolist() == [[0, 1]]

    def test_invalid_grouping_with_bias_uses_fused_topk_bias(self, monkeypatch):
        """A bias present on invalid grouping routes through fused_topk_bias."""
        bias = torch.tensor([0.1, 0.2, 0.3, 0.4])
        router = self._router(e_score_correction_bias=bias)
        recorded = {}

        def fake_fused_topk_bias(**kwargs):
            recorded.update(kwargs)
            return (
                torch.tensor([[0.6, 0.4]]),
                torch.tensor([[3, 0]], dtype=torch.int32),
            )

        monkeypatch.setattr(fl_router, "fused_topk_bias", fake_fused_topk_bias)
        # 3 logits columns % num_expert_group=2 != 0 makes the grouping
        # invalid, exercising the bias branch of the fallback.
        weights, ids = router._compute_routing(
            torch.zeros(1, 3), torch.zeros(1, 3), None
        )
        # .data detaches into a fresh view each access, so the recorded value
        # is checked for shared storage against the tensor's own data.
        assert recorded["e_score_correction_bias"].data_ptr() == bias.data_ptr()
        assert recorded["topk"] == 2
        torch.testing.assert_close(weights, torch.tensor([[0.6, 0.4]]))

    def test_invalid_grouping_bias_applies_scaling_after(self, monkeypatch):
        """routed_scaling_factor is applied on top of the bias-router result."""
        bias = torch.tensor([0.1, 0.2])
        router = self._router(e_score_correction_bias=bias, routed_scaling_factor=2.0)
        monkeypatch.setattr(
            fl_router,
            "fused_topk_bias",
            lambda **k: (torch.tensor([[0.5, 0.25]]), torch.tensor([[0, 1]])),
        )
        weights, _ids = router._compute_routing(
            torch.zeros(1, 3), torch.zeros(1, 2), None
        )
        torch.testing.assert_close(weights, torch.tensor([[1.0, 0.5]]))

    def test_aiter_enabled_routes_to_rocm_partial(self, monkeypatch):
        """AITER-enabled builds route to rocm_aiter_grouped_topk."""
        router = self._router()
        monkeypatch.setattr(
            fl_router.rocm_aiter_ops, "is_fused_moe_enabled", lambda: True
        )
        monkeypatch.setattr(
            fl_router.rocm_aiter_ops,
            "is_fusion_moe_shared_experts_enabled",
            lambda: True,
        )
        recorded = {}

        def fake_aiter_topk(**kwargs):
            recorded.update(kwargs)
            return torch.tensor([[0.9]]), torch.tensor([[1]], dtype=torch.int32)

        monkeypatch.setattr(fl_router, "rocm_aiter_grouped_topk", fake_aiter_topk)
        weights, ids = router._compute_routing(
            torch.zeros(1, 3), torch.zeros(1, 8), None
        )
        assert recorded["num_fused_shared_experts"] == 0
        assert recorded["num_expert_group"] == 2
        torch.testing.assert_close(weights, torch.tensor([[0.9]]))
        assert ids.tolist() == [[1]]

    def test_aiter_shared_experts_mismatch_asserts(self, monkeypatch):
        """AITER with shared experts disabled asserts num_fused_shared_experts==0."""
        router = self._router(num_fused_shared_experts=2)
        monkeypatch.setattr(
            fl_router.rocm_aiter_ops, "is_fused_moe_enabled", lambda: True
        )
        monkeypatch.setattr(
            fl_router.rocm_aiter_ops,
            "is_fusion_moe_shared_experts_enabled",
            lambda: False,
        )
        with pytest.raises(AssertionError):
            router._compute_routing(torch.zeros(1, 3), torch.zeros(1, 8), None)

    def test_aiter_disabled_routes_to_fl_grouped_topk(self, monkeypatch):
        """AITER disabled routes valid groupings through _fl_grouped_topk."""
        router = self._router()
        monkeypatch.setattr(
            fl_router.rocm_aiter_ops, "is_fused_moe_enabled", lambda: False
        )
        recorded = {}

        def fake_fl_grouped_topk(**kwargs):
            recorded.update(kwargs)
            return (
                torch.tensor([[0.7, 0.3]]),
                torch.tensor([[2, 5]], dtype=torch.int32),
            )

        monkeypatch.setattr(fl_router, "_fl_grouped_topk", fake_fl_grouped_topk)
        weights, ids = router._compute_routing(
            torch.zeros(1, 3), torch.zeros(1, 8), None
        )
        assert recorded["num_expert_group"] == 2
        assert recorded["topk_group"] == 1
        torch.testing.assert_close(weights, torch.tensor([[0.7, 0.3]]))
        assert ids.tolist() == [[2, 5]]

    def test_indivisible_grouping_is_invalid(self, monkeypatch):
        """num_experts not divisible by num_expert_group is invalid grouping."""
        router = self._router(num_expert_group=3)
        recorded = {}

        def fake_fl_grouped_topk(**kwargs):
            recorded.update(kwargs)
            return torch.tensor([[1.0]]), torch.tensor([[0]], dtype=torch.int32)

        monkeypatch.setattr(
            fl_router.rocm_aiter_ops, "is_fused_moe_enabled", lambda: False
        )
        monkeypatch.setattr(fl_router, "_fl_grouped_topk", fake_fl_grouped_topk)
        # 4 experts / 3 groups is indivisible: valid_grouping() returns False,
        # so the no-bias fallback (plain fused_topk) runs instead.
        monkeypatch.setattr(
            fl_router,
            "fused_topk",
            lambda **k: (
                torch.tensor([[0.4]]),
                torch.tensor([[0]], dtype=torch.int32),
                None,
            ),
        )
        weights, _ids = router._compute_routing(
            torch.zeros(1, 3), torch.zeros(1, 4), None
        )
        assert recorded == {}  # _fl_grouped_topk was never reached
        torch.testing.assert_close(weights, torch.tensor([[0.4]]))


# ---------------------------------------------------------------------------
# FusedTopKBiasRouterFL._compute_routing
# ---------------------------------------------------------------------------


class TestFusedTopKBiasRouterComputeRouting:
    """FusedTopKBiasRouterFL's sqrtsoftplus-ascend and shared-expert paths."""

    def _router(self, **overrides):
        params = dict(
            top_k=2,
            renormalize=False,
            scoring_func="softmax",
            routed_scaling_factor=1.0,
            e_score_correction_bias=None,
            num_fused_shared_experts=0,
            global_num_experts=8,
            shared_expert_weight=0.5,
            _hash_indices_table=None,
            bias_vl=None,
            image_sentinel_lo=None,
        )
        params.update(overrides)
        router = types.SimpleNamespace(**params)
        router._compute_routing = (
            fl_router.FusedTopKBiasRouterFL._compute_routing.__get__(router)
        )
        return router

    def test_softmax_routes_through_fused_topk_bias(self, monkeypatch):
        """Non-sqrtsoftplus scoring routes through upstream fused_topk_bias."""
        router = self._router()
        recorded = {}

        def fake_fused_topk_bias(**kwargs):
            recorded.update(kwargs)
            return (
                torch.tensor([[0.5, 0.5]]),
                torch.tensor([[0, 1]], dtype=torch.int32),
            )

        monkeypatch.setattr(fl_router, "fused_topk_bias", fake_fused_topk_bias)
        weights, ids = router._compute_routing(
            torch.zeros(1, 3), torch.zeros(1, 8), None
        )
        assert recorded["topk"] == 2
        assert recorded["bias_vl"] is None
        torch.testing.assert_close(weights, torch.tensor([[0.5, 0.5]]))
        assert ids.tolist() == [[0, 1]]

    def test_correction_bias_data_is_forwarded(self, monkeypatch):
        """The registered nn.Parameter .data is what reaches fused_topk_bias."""
        bias = torch.nn.Parameter(torch.tensor([0.1, 0.2, 0.3, 0.4]))
        router = self._router(e_score_correction_bias=bias)
        recorded = {}

        def fake_fused_topk_bias(**kwargs):
            recorded.update(kwargs)
            return (
                torch.tensor([[0.5, 0.5]]),
                torch.tensor([[0, 1]], dtype=torch.int32),
            )

        monkeypatch.setattr(fl_router, "fused_topk_bias", fake_fused_topk_bias)
        router._compute_routing(torch.zeros(1, 3), torch.zeros(1, 4), None)
        # .data detaches into a fresh view each access, so the recorded value
        # is checked for shared storage against the parameter's own data.
        torch.testing.assert_close(recorded["e_score_correction_bias"], bias.data)
        assert recorded["e_score_correction_bias"].data_ptr() == bias.data_ptr()

    def test_shared_experts_are_appended(self, monkeypatch):
        """num_fused_shared_experts > 0 appends the shared expert slots."""
        router = self._router(
            num_fused_shared_experts=2,
            global_num_experts=8,
            shared_expert_weight=0.5,
        )
        monkeypatch.setattr(
            fl_router,
            "fused_topk_bias",
            lambda **k: (
                torch.tensor([[0.25, 0.75]]),
                torch.tensor([[1, 0]], dtype=torch.int32),
            ),
        )
        weights, ids = router._compute_routing(
            torch.zeros(1, 3), torch.zeros(1, 8), None
        )
        # Independent oracle: ids gain [8, 9], weights gain [0.5, 0.5].
        assert ids.tolist() == [[1, 0, 8, 9]]
        torch.testing.assert_close(weights, torch.tensor([[0.25, 0.75, 0.5, 0.5]]))


# ---------------------------------------------------------------------------
# replace_router_with_fl
# ---------------------------------------------------------------------------


class TestReplaceRouterWithFl:
    """The class-level monkey-patch installation of the FL router methods."""

    def test_patch_assigns_fl_compute_routing_to_all_three(self, monkeypatch):
        """The three upstream router classes receive the FL _compute_routing."""
        # The upstream classes are bound in fl_router's own namespace at its
        # import (the same objects replace_router_with_fl patches), so they
        # are reached through it rather than a fresh vllm import — which on
        # stub hosts would fail after the file-scope restore removed the
        # stub graph.
        m1 = fl_router
        m2 = fl_router
        m3 = fl_router

        # Pin the originals through monkeypatch so the patch is auto-reversed.
        for module, cls_name in (
            (m1, "FusedTopKRouter"),
            (m2, "FusedTopKBiasRouter"),
            (m3, "GroupedTopKRouter"),
        ):
            cls = getattr(module, cls_name)
            monkeypatch.setattr(
                cls, "_compute_routing", lambda self, *a, **k: ("orig", "orig")
            )
        fl_router.replace_router_with_fl()
        assert (
            m1.FusedTopKRouter._compute_routing
            is fl_router.FusedTopKRouterFL._compute_routing
        )
        assert (
            m3.GroupedTopKRouter._compute_routing
            is fl_router.GroupedTopKRouterFL._compute_routing
        )
        assert (
            m2.FusedTopKBiasRouter._compute_routing
            is fl_router.FusedTopKBiasRouterFL._compute_routing
        )
