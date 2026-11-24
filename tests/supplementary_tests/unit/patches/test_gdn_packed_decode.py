# Copyright 2025 BAAI. All rights reserved.
"""Supplementary coverage for ``vllm_fl.patches.gdn_packed_decode``.

``tests/unit_tests`` never executes the FL packed-GDN decode kernel body
(the coverage report shows the Triton kernel plus the detection/replacement
helpers missing), so this file runs the kernel for real on CPU tensors
through Triton's interpreter (``TRITON_INTERPRET=1``) and checks the
recurrence against an independent plain-Python oracle, then drives every
``_kernel_needs_beta_patch`` decision row and the
``patch_vllm_packed_gdn_beta`` replacement flow against fake target modules.

The module under test imports ``exp`` from
``vllm.third_party.flash_linear_attention.ops.op`` and ``tl``/``triton``
from ``vllm.triton_utils``; both external boundaries are faked here so the
real source file loads on machines without a vLLM install. All
``sys.modules`` changes go through ``monkeypatch.setitem`` so they are
restored automatically, and the fixture also undoes the re-interpretation of
triton builtins (restoring the original ``JITFunction`` objects) so tests
that later import triton normally are unaffected.
"""

from __future__ import annotations

import importlib.util
import inspect
import linecache
import math
import sys
from pathlib import Path
from types import ModuleType

import pytest
import torch

pytestmark = [pytest.mark.supplementary]

_ROOT = Path(__file__).resolve().parents[4]
_GDN_SOURCE = _ROOT / "vllm_fl/patches/gdn_packed_decode.py"
_GDN_MODULE = "supp_gdn_packed_decode_under_test"

_TARGET_MODULE = "vllm.third_party.flash_linear_attention.ops.fused_recurrent"
_TARGET_NAME = "fused_recurrent_gated_delta_rule_packed_decode_kernel"


class _ExpDispatcher:
    """``exp`` stand-in used by the interpreted Triton kernel.

    The module under test imports ``exp`` from
    ``vllm.third_party.flash_linear_attention.ops.op``. The Triton
    interpreter only patches module objects (``tl``, ``tl.core``) found in
    the kernel's ``__globals__``, so a bare ``exp`` global must be a callable
    object dispatching back into ``tl.exp`` for the kernel to run on CPU.
    """

    def __call__(self, x):
        import triton.language as tl

        return tl.exp(x)


def _reinterp_tl_builtins(tl, std) -> dict:
    """Re-decorate ``tl`` functions that were jitted before interpret mode.

    ``@triton.jit`` decides at decoration time between ``JITFunction`` and
    ``InterpretedFunction`` by reading ``triton.knobs.runtime.interpret``.
    When an earlier test in the same pytest session imported triton without
    ``TRITON_INTERPRET=1``, module-scope decorations in
    ``triton.language.standard`` already produced plain ``JITFunction``
    objects (e.g. ``tl.sigmoid``); setting the env var afterwards cannot
    change them. Inside an interpreted kernel those objects raise
    ``Cannot call @triton.jit'd outside of the scope of a kernel`` when the
    kernel body calls them, so they are re-wrapped as
    ``InterpretedFunction`` here — the same class the clean import order
    would have produced. ``tl`` re-exports the same objects, so both
    namespaces are rebound. The original objects are returned so the fixture
    can restore them.
    """

    from triton.runtime.interpreter import InterpretedFunction

    replaced = {}
    for holder in (std, tl):
        for name in list(vars(holder)):
            member = getattr(holder, name, None)
            if type(member).__name__ == "JITFunction":
                replaced[(holder, name)] = member
                setattr(holder, name, InterpretedFunction(member.fn))
    return replaced


@pytest.fixture
def gdn(monkeypatch):
    """Load the GDN patch module with its vllm import stubs installed.

    Triton's interpreter mode is enabled before the module under test loads
    so the ``@triton.jit`` decorator produces an ``InterpretedFunction``
    that executes on CPU torch tensors. The interpreter's ``_patch_builtin``
    scans every ``tl`` member with ``tl.core.is_builtin``; on stacks where a
    ``tl`` attribute getter raises RuntimeError (not AttributeError) the
    scan itself crashes, so ``is_builtin`` is wrapped to treat that as
    "not a builtin" — a no-op pass-through on upstream triton.
    """
    monkeypatch.setenv("TRITON_INTERPRET", "1")
    triton = pytest.importorskip("triton")
    import triton.language as tl

    real_is_builtin = tl.core.is_builtin

    def _safe_is_builtin(fn):
        try:
            return real_is_builtin(fn)
        except RuntimeError:
            return False

    monkeypatch.setattr(tl.core, "is_builtin", _safe_is_builtin)

    replaced: dict = {}
    if not triton.knobs.runtime.interpret or type(tl.sigmoid).__name__ != (
        "InterpretedFunction"
    ):
        import triton.language.standard as std

        replaced = _reinterp_tl_builtins(tl, std)

    vllm_pkg = ModuleType("vllm")
    vllm_pkg.__path__ = []
    triton_utils = ModuleType("vllm.triton_utils")
    triton_utils.tl = tl
    triton_utils.triton = triton
    fla_op = ModuleType("vllm.third_party.flash_linear_attention.ops.op")
    fla_op.exp = _ExpDispatcher()
    monkeypatch.setitem(sys.modules, "vllm", vllm_pkg)
    monkeypatch.setitem(sys.modules, "vllm.triton_utils", triton_utils)
    monkeypatch.setitem(
        sys.modules, "vllm.third_party.flash_linear_attention.ops.op", fla_op
    )

    spec = importlib.util.spec_from_file_location(_GDN_MODULE, _GDN_SOURCE)
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, _GDN_MODULE, module)
    spec.loader.exec_module(module)

    yield module

    # Restore the module-scope tl builtins the re-interpretation replaced.
    import triton.language as current_tl
    import triton.language.standard as current_std

    for (holder, name), original in replaced.items():
        holder_ref = current_std if holder is current_std or holder is std else current_tl
        try:
            setattr(holder_ref, name, original)
        except Exception:  # pragma: no cover - defensive restore
            pass
    sys.modules.pop(_GDN_MODULE, None)


# ---------------------------------------------------------------------------
# GDN kernel numerics (Triton interpreter)
# ---------------------------------------------------------------------------


def _gdn_oracle(
    *,
    qkv,
    a,
    b,
    A_log,
    dt_bias,
    h0,
    indices,
    scale,
    use_qk_l2norm,
    softplus_threshold,
    num_tokens,
    num_heads,
    num_kv_heads,
    head_k,
    head_v,
):
    """Independent plain-Python reference for one packed GDN decode step.

    Implements the gated delta rule recurrence the kernel encodes:
    FP32 softplus/sigmoid gating, decayed state, delta-rule value update and
    the output projection. ``NULL_BLOCK_ID`` rows (state index <= 0)
    produce zero output and leave the state untouched.
    """
    H, HV, K, V = num_heads, num_kv_heads, head_k, head_v
    output = torch.zeros(num_tokens, HV, V)
    final_state = h0.clone()
    for n in range(num_tokens):
        state_idx = int(indices[n])
        if state_idx <= 0:
            continue  # NULL_BLOCK_ID: zero output, no state touch
        for i_hv in range(HV):
            i_h = i_hv // (HV // H)
            state = h0[state_idx, i_hv].clone().double()
            q = qkv[n, i_h * K : (i_h + 1) * K].double()
            k = qkv[n, H * K + i_h * K : H * K + (i_h + 1) * K].double()
            v = qkv[n, 2 * H * K + i_hv * V : 2 * H * K + (i_hv + 1) * V].double()
            if use_qk_l2norm:
                q = q / torch.sqrt((q * q).sum() + 1e-6)
                k = k / torch.sqrt((k * k).sum() + 1e-6)
            q = q * scale
            x = float(a[n, i_hv] + dt_bias[i_hv])
            softplus = math.log1p(math.exp(x)) if x <= softplus_threshold else x
            g = -math.exp(float(A_log[i_hv])) * softplus
            beta = 1.0 / (1.0 + math.exp(-float(b[n, i_hv])))
            state = state * math.exp(g)
            v = v - (state * k[None, :]).sum(1)
            v = v * beta
            state = state + v[:, None] * k[None, :]
            output[n, i_hv] = (state * q[None, :]).sum(1).float()
            final_state[state_idx, i_hv] = state.float()
    return output, final_state


def _launch_gdn_kernel(
    module,
    *,
    qkv,
    a,
    b,
    A_log,
    dt_bias,
    h0,
    ht,
    output,
    indices,
    scale,
    num_heads,
    num_kv_heads,
    head_k,
    head_v,
    block_k,
    block_v,
    softplus_threshold,
    use_qk_l2norm,
    split_batch_head_grid,
):
    """Invoke the real kernel through the Triton interpreter."""
    import triton

    num_tokens, HV = a.shape
    if split_batch_head_grid:
        grid = (triton.cdiv(head_v, block_v), HV, num_tokens)
    else:
        grid = (triton.cdiv(head_v, block_v), num_tokens * HV)
    kernel = module._fused_recurrent_gated_delta_rule_packed_decode_kernel_fp32_beta
    kernel[grid](
        qkv,
        a,
        b,
        A_log,
        dt_bias,
        output,
        h0,
        ht,
        indices,
        scale,
        qkv.stride(0),
        a.stride(0),
        b.stride(0),
        h0.stride(0),
        ht.stride(0),
        indices.stride(0),
        H=num_heads,
        HV=HV,
        K=head_k,
        V=head_v,
        BK=block_k,
        BV=block_v,
        SOFTPLUS_THRESHOLD=softplus_threshold,
        USE_QK_L2NORM_IN_KERNEL=use_qk_l2norm,
        SPLIT_BATCH_HEAD_GRID=split_batch_head_grid,
    )


class TestGdnKernelNumerics:
    """The FP32-beta kernel body, executed via the Triton interpreter."""

    # (use_qk_l2norm, split_batch_head_grid)
    @pytest.mark.parametrize(
        ("use_l2norm", "split_grid"),
        [(False, False), (True, False), (False, True), (True, True)],
    )
    def test_kernel_matches_independent_oracle(self, gdn, use_l2norm, split_grid):
        """The recurrence output and written state match a plain-Python oracle."""
        generator = torch.Generator().manual_seed(17)
        num_tokens, H, HV, K, V = 2, 2, 4, 4, 6
        BK, BV, S = 4, 4, 5
        qkv = torch.randn(num_tokens, 2 * H * K + HV * V, generator=generator)
        a = torch.randn(num_tokens, HV, generator=generator)
        b = torch.randn(num_tokens, HV, generator=generator)
        A_log = torch.randn(HV, generator=generator)
        dt_bias = torch.randn(HV, generator=generator)
        scale = K**-0.5
        h0 = torch.randn(S, HV, V, K, generator=generator)
        ht = torch.zeros(S, HV, V, K)
        output = torch.zeros(num_tokens, HV, V)
        indices = torch.tensor([2, 0])  # row 1 is NULL_BLOCK_ID

        _launch_gdn_kernel(
            gdn,
            qkv=qkv,
            a=a,
            b=b,
            A_log=A_log,
            dt_bias=dt_bias,
            h0=h0,
            ht=ht,
            output=output,
            indices=indices,
            scale=scale,
            num_heads=H,
            num_kv_heads=HV,
            head_k=K,
            head_v=V,
            block_k=BK,
            block_v=BV,
            softplus_threshold=20.0,
            use_qk_l2norm=use_l2norm,
            split_batch_head_grid=split_grid,
        )

        expected_output, expected_state = _gdn_oracle(
            qkv=qkv,
            a=a,
            b=b,
            A_log=A_log,
            dt_bias=dt_bias,
            h0=h0,
            indices=indices,
            scale=scale,
            use_qk_l2norm=use_l2norm,
            softplus_threshold=20.0,
            num_tokens=num_tokens,
            num_heads=H,
            num_kv_heads=HV,
            head_k=K,
            head_v=V,
        )
        torch.testing.assert_close(output, expected_output, rtol=1e-4, atol=1e-4)
        # Only state slots referenced by a valid index are written; slot 0
        # stays at its zero init and untouched slots keep h0's values.
        torch.testing.assert_close(ht[2], expected_state[2], rtol=1e-4, atol=1e-4)
        assert torch.all(ht[0] == 0)

    def test_null_state_index_zeroes_output_only(self, gdn):
        """Rows with state_idx <= 0 store zeros and never touch the state."""
        generator = torch.Generator().manual_seed(5)
        num_tokens, H, HV, K, V = 3, 1, 2, 2, 4
        S = 4
        qkv = torch.randn(num_tokens, 2 * H * K + HV * V, generator=generator)
        a = torch.randn(num_tokens, HV, generator=generator)
        b = torch.randn(num_tokens, HV, generator=generator)
        A_log = torch.randn(HV, generator=generator)
        dt_bias = torch.randn(HV, generator=generator)
        h0 = torch.randn(S, HV, V, K, generator=generator)
        ht = torch.zeros(S, HV, V, K)
        output = torch.full((num_tokens, HV, V), float("nan"))
        indices = torch.tensor([0, -3, 2])  # two NULL rows, one valid

        _launch_gdn_kernel(
            gdn,
            qkv=qkv,
            a=a,
            b=b,
            A_log=A_log,
            dt_bias=dt_bias,
            h0=h0,
            ht=ht,
            output=output,
            indices=indices,
            scale=1.0,
            num_heads=H,
            num_kv_heads=HV,
            head_k=K,
            head_v=V,
            block_k=2,
            block_v=4,
            softplus_threshold=20.0,
            use_qk_l2norm=False,
            split_batch_head_grid=True,
        )

        # NULL rows were overwritten with zeros (not left as NaN).
        assert torch.all(output[0] == 0)
        assert torch.all(output[1] == 0)
        assert not torch.isnan(output[2]).any()
        # No state slot was touched except the one the valid row writes.
        valid_state = ht[2]
        assert torch.any(valid_state != 0)
        assert torch.all(ht[0] == 0) and torch.all(ht[1] == 0)
        assert torch.all(ht[3] == 0)

    def test_softplus_threshold_passthrough(self, gdn):
        """x above SOFTPLUS_THRESHOLD keeps x itself instead of log1p(exp(x))."""
        generator = torch.Generator().manual_seed(11)
        num_tokens, H, HV, K, V = 1, 1, 1, 2, 2
        qkv = torch.randn(num_tokens, 2 * H * K + HV * V, generator=generator)
        a = torch.tensor([[60.0]])  # far above the threshold below
        b = torch.tensor([[0.3]])
        A_log = torch.tensor([0.1])
        dt_bias = torch.tensor([0.0])
        h0 = torch.randn(2, 1, V, K, generator=generator)
        ht = torch.zeros(2, 1, V, K)
        output = torch.zeros(1, 1, V)
        indices = torch.tensor([1])

        threshold = -1.0  # every x passes through un-softplus'd
        _launch_gdn_kernel(
            gdn,
            qkv=qkv,
            a=a,
            b=b,
            A_log=A_log,
            dt_bias=dt_bias,
            h0=h0,
            ht=ht,
            output=output,
            indices=indices,
            scale=1.0,
            num_heads=H,
            num_kv_heads=HV,
            head_k=K,
            head_v=V,
            block_k=2,
            block_v=2,
            softplus_threshold=threshold,
            use_qk_l2norm=False,
            split_batch_head_grid=True,
        )

        expected_output, expected_state = _gdn_oracle(
            qkv=qkv,
            a=a,
            b=b,
            A_log=A_log,
            dt_bias=dt_bias,
            h0=h0,
            indices=indices,
            scale=1.0,
            use_qk_l2norm=False,
            softplus_threshold=threshold,
            num_tokens=num_tokens,
            num_heads=H,
            num_kv_heads=HV,
            head_k=K,
            head_v=V,
        )
        torch.testing.assert_close(output, expected_output, rtol=1e-4, atol=1e-4)
        torch.testing.assert_close(ht[1], expected_state[1], rtol=1e-4, atol=1e-4)

    def test_kernel_carries_fp32_beta_marker(self, gdn):
        """The replacement kernel is flagged so detection skips it."""
        kernel = gdn._fused_recurrent_gated_delta_rule_packed_decode_kernel_fp32_beta
        assert getattr(kernel, "_fl_fp32_beta", False) is True


# ---------------------------------------------------------------------------
# Detection and patching
# ---------------------------------------------------------------------------


def _function_with_source(source: str):
    """Compile a function whose inspect.getsource returns ``source``.

    ``inspect.getsource`` reads from ``linecache`` using the code object's
    filename; registering the source under a fake file makes the round trip
    work on any platform.
    """
    filename = f"<supp_gdn_source_{abs(hash(source))}>"
    namespace: dict = {}
    compiled = compile(source, filename, "exec")
    exec(compiled, namespace)
    function = namespace[source.split("(")[0].split()[-1]]  # first def name
    linecache.cache[filename] = (len(source), None, source.splitlines(True), filename)
    return function


class TestKernelNeedsBetaPatch:
    """``_kernel_needs_beta_patch`` decision table."""

    def test_own_marked_kernel_is_not_flagged(self, gdn):
        """The marker short-circuits detection before source inspection."""
        kernel = gdn._fused_recurrent_gated_delta_rule_packed_decode_kernel_fp32_beta
        assert gdn._kernel_needs_beta_patch(kernel) is False

    def test_sourceless_function_is_flagged(self, gdn):
        """A function without retrievable source defaults to needing the fix.

        Builtins have no ``__module__``/source, so ``inspect.getsource``
        raises TypeError — the conservative branch returns True.
        """
        assert gdn._kernel_needs_beta_patch(len) is True

    def test_vulnerable_expression_is_flagged(self, gdn):
        """A source containing the round-to-input-dtype beta is flagged."""
        vulnerable = _function_with_source(
            "def vulnerable_kernel():\n"
            "    beta = tl.sigmoid(b_val).to(b.dtype.element_ty).to(tl.float32)\n"
        )
        assert gdn._kernel_needs_beta_patch(vulnerable) is True

    def test_fixed_source_is_not_flagged(self, gdn):
        """A source without the vulnerable expression reports False."""
        fixed = _function_with_source(
            "def fixed_kernel():\n    beta = tl.sigmoid(b_val)\n"
        )
        assert gdn._kernel_needs_beta_patch(fixed) is False


class TestPatchVllmPackedGdnBeta:
    """``patch_vllm_packed_gdn_beta`` replacement flow."""

    def test_missing_target_module_returns_false(self, gdn, monkeypatch):
        """Without the vLLM fused_recurrent module the patch is a no-op."""
        # The stub graph deliberately does not install _TARGET_MODULE.
        monkeypatch.setitem(sys.modules, _TARGET_MODULE, None)
        assert gdn.patch_vllm_packed_gdn_beta() is False

    def test_missing_kernel_symbol_returns_false(self, gdn, monkeypatch):
        """A target module without the kernel symbol is a no-op."""
        target = ModuleType(_TARGET_MODULE)
        monkeypatch.setitem(sys.modules, _TARGET_MODULE, target)
        assert gdn.patch_vllm_packed_gdn_beta() is False

    def test_vulnerable_kernel_is_replaced(self, gdn, monkeypatch):
        """A vulnerable kernel is swapped for the FP32-beta one."""
        vulnerable = _function_with_source(
            "def packed_kernel():\n"
            "    beta = tl.sigmoid(b_val).to(b.dtype.element_ty).to(tl.float32)\n"
        )
        target = ModuleType(_TARGET_MODULE)
        setattr(target, _TARGET_NAME, vulnerable)
        monkeypatch.setitem(sys.modules, _TARGET_MODULE, target)

        assert gdn.patch_vllm_packed_gdn_beta() is True
        assert (
            getattr(target, _TARGET_NAME)
            is gdn._fused_recurrent_gated_delta_rule_packed_decode_kernel_fp32_beta
        )

    def test_replacement_is_idempotent(self, gdn, monkeypatch):
        """A second call on the already-patched module reports False."""
        vulnerable = _function_with_source(
            "def packed_kernel():\n"
            "    beta = tl.sigmoid(b_val).to(b.dtype.element_ty).to(tl.float32)\n"
        )
        target = ModuleType(_TARGET_MODULE)
        setattr(target, _TARGET_NAME, vulnerable)
        monkeypatch.setitem(sys.modules, _TARGET_MODULE, target)

        assert gdn.patch_vllm_packed_gdn_beta() is True
        assert gdn.patch_vllm_packed_gdn_beta() is False

    def test_already_fixed_kernel_is_left_alone(self, gdn, monkeypatch):
        """An upstream kernel without the vulnerable expression stays."""
        fixed = _function_with_source(
            "def packed_kernel():\n    beta = tl.sigmoid(b_val).to(tl.float32)\n"
        )
        target = ModuleType(_TARGET_MODULE)
        setattr(target, _TARGET_NAME, fixed)
        monkeypatch.setitem(sys.modules, _TARGET_MODULE, target)

        assert gdn.patch_vllm_packed_gdn_beta() is False
        assert getattr(target, _TARGET_NAME) is fixed


class TestSourceContract:
    """Source-pinned contracts that do not need a kernel launch."""

    def test_module_targets_fused_recurrent_module(self, gdn):
        """The patch addresses the documented vLLM symbol."""
        assert gdn._TARGET_MODULE == (
            "vllm.third_party.flash_linear_attention.ops.fused_recurrent"
        )
        assert gdn._TARGET_NAME == (
            "fused_recurrent_gated_delta_rule_packed_decode_kernel"
        )

    def test_vulnerable_expression_constant(self, gdn):
        """The detection expression matches the documented upstream bug."""
        assert gdn._VULNERABLE_BETA_EXPRESSION == (
            "tl.sigmoid(b_val).to(b.dtype.element_ty).to(tl.float32)"
        )

    def test_kernel_signature_shape(self, gdn):
        """The shipped kernel keeps the documented parameter surface."""
        kernel = gdn._fused_recurrent_gated_delta_rule_packed_decode_kernel_fp32_beta
        python_fn = getattr(kernel, "fn", kernel)
        parameters = list(inspect.signature(python_fn).parameters)
        for required in (
            "mixed_qkv",
            "a",
            "b",
            "A_log",
            "dt_bias",
            "o",
            "h0",
            "ht",
            "ssm_state_indices",
            "scale",
            "H",
            "HV",
            "K",
            "V",
            "BK",
            "BV",
            "SOFTPLUS_THRESHOLD",
            "USE_QK_L2NORM_IN_KERNEL",
            "SPLIT_BATCH_HEAD_GRID",
        ):
            assert required in parameters, f"kernel lost parameter {required}"
