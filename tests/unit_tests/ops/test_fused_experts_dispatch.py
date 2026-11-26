# Copyright (c) 2026 BAAI. All rights reserved.

"""``fused_experts_impl`` dispatch-contract tests for ``vllm_fl.ops.fused_moe.fused_moe``.

``fused_experts_impl``'s constraint assertions, activation normalization,
and the boundary calls it hands to the dispatch system — the parts of the
fused_moe module the core suite leaves uncovered.

The Triton GEMM kernels and the config lookup are external boundaries;
recorders replace them so the tests assert what the function hands to the
dispatch system, not the kernels themselves.

The module imports ``vllm`` at module level.  On machines without the real
package a stub ``vllm`` graph is installed first (external boundary only);
with the real ``vllm`` present the module imports against it.  Either
way, every kernel class, dispatched op, and config lookup is swapped at
test time via ``monkeypatch`` on module attributes, so the code under test
always runs its own logic.
"""

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
    import vllm_fl.ops.fused_moe.fused_moe as fl_fused  # noqa: E402
    import vllm_fl.ops.fused_moe.fused_moe_utils as _fmu  # noqa: E402
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



def _fl_enum(name: str):
    """Fetch a MoEActivation member the module under test normalizes.

    The enum type was bound during the file-scope import above (against the
    stub graph on vllm-less hosts, against the real vllm otherwise); the
    module object stays alive after the restore, so no test-time vllm import
    is needed.
    """
    return getattr(_fmu.MoEActivation, name)

# ---------------------------------------------------------------------------
# fused_experts_impl dispatch contracts
# ---------------------------------------------------------------------------


class TestFusedExpertsImplContracts:
    """fused_experts_impl shape assertions and boundary-call contracts.

    The Triton GEMM kernels and the config lookup are external boundaries;
    recorders replace them so the tests assert what the function hands to the
    dispatch system, not the kernels themselves.
    """

    E, N, K, TOPK = 2, 8, 4, 2

    def _make_inputs(self, num_tokens=3, dtype=torch.float32):
        hidden = torch.randn(num_tokens, self.K, dtype=dtype)
        w1 = torch.randn(self.E, 2 * self.N, self.K, dtype=dtype)
        w2 = torch.randn(self.E, self.K, self.N, dtype=dtype)
        tw = torch.rand(num_tokens, self.TOPK, dtype=dtype)
        ti = torch.randint(0, self.E, (num_tokens, self.TOPK))
        return hidden, w1, w2, tw, ti

    @pytest.fixture
    def recorders(self, monkeypatch):
        """Record every boundary call fused_experts_impl makes."""
        calls = {"align": [], "kernel": [], "moe_sum": [], "config": []}

        def fake_align(ids, block_size, num_experts, expert_map, **kwargs):
            calls["align"].append((tuple(ids.shape), block_size, num_experts))
            return (
                torch.zeros(8, dtype=torch.int32),
                torch.zeros(4, dtype=torch.int32),
                torch.tensor([8], dtype=torch.int32),
            )

        def fake_kernel(*args, **kwargs):
            # Positional contract of _invoke_fused_moe_triton_kernel:
            # (A, B, C, A_scale, B_scale, topk_weights, sorted_ids,
            #  expert_ids, num_post_pad, mul_routed_weight, top_k, config)
            calls["kernel"].append(
                {
                    "B_shape": tuple(args[1].shape),
                    "C_shape": tuple(args[2].shape),
                    "mul_routed_weight": args[9],
                    "top_k": args[10],
                    "compute_type": kwargs.get("compute_type"),
                    "use_fp8_w8a8": kwargs.get("use_fp8_w8a8"),
                    "per_channel_quant": kwargs.get("per_channel_quant"),
                }
            )

        def fake_moe_sum(inp, out):
            calls["moe_sum"].append((tuple(inp.shape), tuple(out.shape)))
            torch.sum(inp, dim=1, out=out)

        def fake_silu_and_mul(unused, x):
            # Independent oracle: silu over the first half, multiplied by the
            # second half — the contract the real dispatched kernel provides.
            d = x.shape[-1] // 2
            return torch.nn.functional.silu(x[..., :d]) * x[..., d:]

        def fake_config(*args, **kwargs):
            calls["config"].append(args)
            return {"BLOCK_SIZE_M": 16}

        monkeypatch.setattr(fl_fused, "_moe_align_block_size", fake_align)
        monkeypatch.setattr(fl_fused, "_invoke_fused_moe_triton_kernel", fake_kernel)
        _replace_cached_op(monkeypatch, fl_fused, "_moe_sum", fake_moe_sum)
        # _silu_and_mul is a CachedOp too: resolving it initializes the
        # dispatch system, which probes vllm.platforms and cannot run on a
        # stub host. Replacing it with the numeric oracle keeps the dispatch
        # boundary (like the other kernels) out of the contract under test.
        _replace_cached_op(monkeypatch, fl_fused, "_silu_and_mul", fake_silu_and_mul)
        # vLLM 0.32's _get_config_quant_dtype lacks the ocp_mx_scheme kwarg
        # the FL call site passes, and the config lookup signature differs per
        # vLLM version: pin both to the contract fused_experts_impl expects.
        monkeypatch.setattr(
            fl_fused, "_get_config_quant_dtype", lambda **kwargs: None, raising=False
        )
        monkeypatch.setattr(
            fl_fused, "try_get_optimal_moe_config", fake_config, raising=False
        )
        return calls

    def test_happy_path_dispatch_contract(self, recorders):
        """Both GEMMs, the activation, and moe_sum receive proper shapes."""
        hidden, w1, w2, tw, ti = self._make_inputs()
        out = fl_fused.fused_experts_impl(hidden, w1, w2, tw, ti)
        assert out.shape == hidden.shape
        assert len(recorders["kernel"]) == 2
        first, second = recorders["kernel"]
        # The implementation reads N from w1's second dim — the fused
        # gate+up width, i.e. 2 * the per-gate N the inputs are built with.
        fused_n = 2 * self.N
        assert first["B_shape"] == (self.E, fused_n, self.K)
        assert first["C_shape"] == (hidden.size(0), self.TOPK, fused_n)
        assert second["B_shape"] == (self.E, self.K, self.N)
        assert second["C_shape"] == (hidden.size(0), self.TOPK, self.K)
        # First GEMM multiplies routed weights iff requested on input;
        # the second GEMM applies them iff the first did not.
        assert first["mul_routed_weight"] is False
        assert second["mul_routed_weight"] is True
        assert second["top_k"] == 1
        # Alignment gets the block size from the config lookup.
        assert recorders["align"][0] == (ti.shape, 16, self.E)
        assert recorders["moe_sum"][0] == (
            (hidden.size(0), self.TOPK, self.K),
            (hidden.size(0), self.K),
        )

    def test_inplace_writes_into_hidden_states(self, recorders):
        """inplace=True writes the result into the hidden_states tensor."""
        hidden, w1, w2, tw, ti = self._make_inputs()
        out = fl_fused.fused_experts_impl(hidden, w1, w2, tw, ti, inplace=True)
        assert out is hidden

    def test_router_weight_on_input_flips_gemm_flags(self, recorders):
        """apply_router_weight_on_input inverts both GEMM weight flags."""
        hidden, w1, w2, tw, ti = self._make_inputs()
        fl_fused.fused_experts_impl(
            hidden, w1, w2, tw, ti, apply_router_weight_on_input=True
        )
        first, second = recorders["kernel"]
        assert first["mul_routed_weight"] is True
        assert second["mul_routed_weight"] is False

    def test_compute_type_follows_dtype(self, recorders):
        """The Triton compute type is chosen from the activation dtype."""
        # fl_fused binds the upstream tl namespace at its own import; the
        # stub graph is gone by test time on vllm-less hosts, so the bound
        # reference is used instead of importing vllm.triton_utils.
        _vtu = fl_fused

        hidden, w1, w2, tw, ti = self._make_inputs(dtype=torch.float16)
        fl_fused.fused_experts_impl(hidden, w1, w2, tw, ti)
        assert recorders["kernel"][0]["compute_type"] is _vtu.tl.float16
        recorders["kernel"].clear()
        hidden, w1, w2, tw, ti = self._make_inputs(dtype=torch.bfloat16)
        fl_fused.fused_experts_impl(hidden, w1, w2, tw, ti)
        assert recorders["kernel"][0]["compute_type"] is _vtu.tl.bfloat16

    def test_gelu_activation_routes_to_gelu_and_mul(self, recorders, monkeypatch):
        """activation='gelu' dispatches gelu_and_mul instead of silu_and_mul.

        The gelu oracle's operand is checked against the first GEMM's cache
        contract (the flattened [M*top_k, N] view of the first kernel call's
        C operand), the gelu result is provably the silu-free path, and the
        second GEMM plus moe_sum ran — the dispatch contract the string
        routing promises, not just a call count.
        """
        hidden, w1, w2, tw, ti = self._make_inputs()
        gelu_inputs = []

        def fake_gelu(out, x):
            gelu_inputs.append(x)
            d = x.shape[-1] // 2
            return torch.nn.functional.gelu(x[..., :d]) * x[..., d:]

        silu_rec = _replace_cached_op(
            monkeypatch,
            fl_fused,
            "_silu_and_mul",
            lambda *a, **k: pytest.fail("silu dispatched for activation='gelu'"),
        )
        gelu_rec = _replace_cached_op(monkeypatch, fl_fused, "_gelu_and_mul", fake_gelu)
        out = fl_fused.fused_experts_impl(hidden, w1, w2, tw, ti, activation="gelu")
        assert gelu_rec.calls
        assert not silu_rec.calls
        # The gelu oracle received the flattened first-GEMM cache.
        assert len(gelu_inputs) == 1
        first_c = recorders["kernel"][0]["C_shape"]
        assert tuple(gelu_inputs[0].shape) == (
            first_c[0] * first_c[1],
            first_c[2],
        )
        # The gelu product flowed into the second GEMM and moe_sum.
        assert len(recorders["kernel"]) == 2
        assert recorders["moe_sum"]
        assert out.shape == hidden.shape

    def test_no_mul_activations_stay_in_torch(self, recorders):
        """silu_no_mul keeps the activation inside torch (F.silu path)."""
        hidden, w1, w2, tw, ti = self._make_inputs()
        out = fl_fused.fused_experts_impl(
            hidden, w1, w2, tw, ti, activation="silu_no_mul"
        )
        assert len(recorders["kernel"]) == 2  # GEMMs still ran
        assert out.shape == hidden.shape

    def test_unsupported_activation_raises(self, recorders):
        """An unknown activation name raises ValueError."""
        hidden, w1, w2, tw, ti = self._make_inputs()
        with pytest.raises(ValueError, match="Unsupported FusedMoe activation"):
            fl_fused.fused_experts_impl(hidden, w1, w2, tw, ti, activation="relu")

    def test_hidden_size_mismatch_raises(self, recorders):
        """A hidden size inconsistent with w1 fails the shape assertion."""
        hidden, w1, w2, tw, ti = self._make_inputs()
        bad_w1 = torch.randn(self.E, 2 * self.N, self.K + 2)
        with pytest.raises(AssertionError, match="Hidden size mismatch"):
            fl_fused.fused_experts_impl(hidden, bad_w1, w2, tw, ti)

    def test_topk_shape_mismatch_raises(self, recorders):
        """topk_weights / topk_ids shape disagreement fails the assertion."""
        hidden, w1, w2, tw, ti = self._make_inputs()
        with pytest.raises(AssertionError, match="topk shape mismatch"):
            fl_fused.fused_experts_impl(hidden, w1, w2, tw[:, :1], ti)

    def test_int4_hidden_size_halved_assertion(self, recorders):
        """use_int4_w4a16 checks hidden // 2 == w1.size(2)."""
        hidden, w1, w2, tw, ti = self._make_inputs()
        # int4 packs 2 weights per element: w1 must see K columns packed
        # from K/2 hidden elements. A matching w1 passes; a mismatched one
        # raises.
        ok_w1 = torch.randn(self.E, 2 * self.N, self.K // 2)
        fl_fused.fused_experts_impl(hidden, ok_w1, w2, tw, ti, use_int4_w4a16=True)
        bad_w1 = torch.randn(self.E, 2 * self.N, self.K // 2 + 1)
        with pytest.raises(AssertionError, match="Hidden size mismatch"):
            fl_fused.fused_experts_impl(hidden, bad_w1, w2, tw, ti, use_int4_w4a16=True)

    def test_enum_activation_is_normalized_to_string(self, recorders, monkeypatch):
        """A MoEActivation enum is normalized via its .value attribute.

        The normalized string must select the silu kernel (and only it):
        the gelu dispatch is a tripwire, and the silu oracle's operands are
        checked against the first GEMM's cache contract — the flattened
        [M*top_k, N] view of the first kernel call's C operand — instead of
        asserting only that some call happened.
        """
        hidden, w1, w2, tw, ti = self._make_inputs()
        silu_inputs = []

        def fake_silu(out, x):
            silu_inputs.append(x)
            d = x.shape[-1] // 2
            return torch.nn.functional.silu(x[..., :d]) * x[..., d:]

        gelu_rec = _replace_cached_op(
            monkeypatch,
            fl_fused,
            "_gelu_and_mul",
            lambda *a, **k: pytest.fail("gelu dispatched for a SILU enum"),
        )
        silu_rec = _replace_cached_op(monkeypatch, fl_fused, "_silu_and_mul", fake_silu)
        out = fl_fused.fused_experts_impl(
            hidden, w1, w2, tw, ti, activation=_fl_enum("SILU")
        )
        assert silu_rec.calls
        assert not gelu_rec.calls
        assert out.shape == hidden.shape
        # The silu oracle received the flattened first-GEMM cache: the same
        # element count as the first kernel call's C operand reshaped to
        # [-1, N], i.e. the M*top_k rows of width 2*N the cache promises.
        assert len(silu_inputs) == 1
        first_c = recorders["kernel"][0]["C_shape"]
        assert tuple(silu_inputs[0].shape) == (
            first_c[0] * first_c[1],
            first_c[2],
        )

    def test_global_num_experts_defaults_from_w1(self, recorders):
        """global_num_experts=-1 derives the expert count from w1.size(0)."""
        hidden, w1, w2, tw, ti = self._make_inputs()
        fl_fused.fused_experts_impl(hidden, w1, w2, tw, ti)
        assert recorders["align"][0][2] == self.E

    def test_explicit_global_num_experts_forwarded(self, recorders):
        """An explicit global_num_experts reaches the alignment boundary."""
        hidden, w1, w2, tw, ti = self._make_inputs()
        fl_fused.fused_experts_impl(hidden, w1, w2, tw, ti, global_num_experts=17)
        assert recorders["align"][0][2] == 17

    def test_quantized_activation_paths_dispatch_kernel(
        self, recorders, monkeypatch
    ):
        """fp8 flags flow through to the kernel boundary unchanged.

        Both GEMMs must also receive the fp8 / per-channel flags verbatim,
        and the activation-quantize boundary must run with the per-channel
        flag and the (pinned) config-quant dtype, so the quantization path
        is observable, not just the call count.
        """
        hidden, w1, w2, tw, ti = self._make_inputs()
        quantized = []

        def fake_quantize_input(A, *, A_scale, quant_dtype, per_act_token_quant, block_shape):
            quantized.append(
                {
                    "quant_dtype": quant_dtype,
                    "per_act_token_quant": per_act_token_quant,
                    "block_shape": block_shape,
                }
            )
            return A, None

        monkeypatch.setattr(fl_fused, "moe_kernel_quantize_input", fake_quantize_input)
        fl_fused.fused_experts_impl(
            hidden, w1, w2, tw, ti, use_fp8_w8a8=True, per_channel_quant=True
        )
        assert len(recorders["kernel"]) == 2
        # Both GEMMs carried the fp8 + per-channel flags to the kernel.
        for call in recorders["kernel"]:
            assert call["use_fp8_w8a8"] is True
            assert call["per_channel_quant"] is True
        # Both activation-quantize boundaries ran with the per-channel flag
        # and the pinned config-quant dtype (None in this fixture).
        assert len(quantized) == 2
        assert all(q["per_act_token_quant"] is True for q in quantized)
        assert all(q["quant_dtype"] is None for q in quantized)
        assert all(q["block_shape"] is None for q in quantized)

    def test_single_chunk_runs_one_config_lookup(self, recorders):
        """num_tokens < CHUNK_SIZE runs one chunk and one config lookup."""
        hidden, w1, w2, tw, ti = self._make_inputs(num_tokens=3)
        fl_fused.fused_experts_impl(hidden, w1, w2, tw, ti)
        assert len(recorders["config"]) == 1
        assert len(recorders["moe_sum"]) == 1

    def test_out_dtype_matches_input(self, recorders):
        """float16 and bfloat16 inputs flow through with matching dtypes."""
        for dtype in (torch.float16, torch.bfloat16):
            recorders["kernel"].clear()
            recorders["moe_sum"].clear()
            hidden, w1, w2, tw, ti = self._make_inputs(dtype=dtype)
            out = fl_fused.fused_experts_impl(hidden, w1, w2, tw, ti)
            assert out.dtype == dtype
            assert len(recorders["kernel"]) == 2
