# Copyright (c) 2026 BAAI. All rights reserved.

"""CPU tests for the GDN beta patch and QSA cache-hint patch.

``vllm_fl/patches/gdn_packed_decode.py`` ships a corrected Triton kernel for
vLLM's packed GDN decode (beta kept in FP32) plus the detection/replacement
helpers; ``vllm_fl/patches/qsa_cache_hint.py`` strips an optional
``eviction_policy`` keyword from QSA K-cache loads on Hopper + FlagTree 3.6.

The GDN kernel body never executes under the normal CI stubs (the coverage
report shows the whole kernel missing), so these tests run it for real on
CPU tensors through Triton's interpreter (``TRITON_INTERPRET=1``) and check
the recurrence against an independent plain-Python oracle.  The QSA side is
driven through AST-level helpers and a fake ``qsa_indexer`` module, covering
the full gate matrix (architecture, triton version, flagtree presence,
device type, HIP build, compute capability) plus the validation failures.
"""

import importlib.util
import inspect
import math
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import Any

import pytest
import torch


_ROOT = Path(__file__).resolve().parents[4]
_GDN_SOURCE = _ROOT / "vllm_fl/patches/gdn_packed_decode.py"
_QSA_SOURCE = _ROOT / "vllm_fl/patches/qsa_cache_hint.py"

_GDN_MODULE = "supp_gdn_packed_decode_under_test"
_QSA_MODULE = "supp_qsa_cache_hint_under_test"

_TARGET_MODULE = "vllm.third_party.flash_linear_attention.ops.fused_recurrent"
_TARGET_NAME = "fused_recurrent_gated_delta_rule_packed_decode_kernel"


# ---------------------------------------------------------------------------
# Shared loading helpers
# ---------------------------------------------------------------------------


def _load(source: Path, module_name: str) -> Any:
    """Exec a plugin source file once under a private module name."""
    spec = importlib.util.spec_from_file_location(module_name, source)
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    try:
        spec.loader.exec_module(module)
    except BaseException:
        sys.modules.pop(module_name, None)
        raise
    return module


class _ExpDispatcher:
    """``exp`` stand-in used by the interpreted Triton kernel.

    The module under test imports ``exp`` from
    ``vllm.third_party.flash_linear_attention.ops.op``.  The Triton
    interpreter only patches module objects (``tl``, ``tl.core``) found in
    the kernel's ``__globals__``, so a bare ``exp`` global must be a callable
    object dispatching back into ``tl.exp`` for the kernel to run on CPU.
    """

    def __call__(self, x):
        import triton.language as tl

        return tl.exp(x)


def _reinterp_tl_builtins(tl, std) -> None:
    """Re-decorate ``tl`` functions that were jitted before interpret mode.

    ``@triton.jit`` decides at decoration time between ``JITFunction`` and
    ``InterpretedFunction`` by reading ``triton.knobs.runtime.interpret``.
    When an earlier test in the same pytest session imported triton without
    ``TRITON_INTERPRET=1``, module-scope decorations in
    ``triton.language.standard`` already produced plain ``JITFunction``
    objects (e.g. ``tl.sigmoid``); setting the env var afterwards cannot
    change them.  Inside an interpreted kernel those objects raise
    ``Cannot call @triton.jit'd outside of the scope of a kernel`` when the
    kernel body calls them, so they are re-wrapped as
    ``InterpretedFunction`` here — the same class the clean import order
    would have produced.  ``tl`` re-exports the same objects, so both
    namespaces are rebound.
    """
    from triton.runtime.interpreter import InterpretedFunction

    for holder in (std, tl):
        for name in list(vars(holder)):
            member = getattr(holder, name, None)
            if type(member).__name__ == "JITFunction":
                setattr(holder, name, InterpretedFunction(member.fn))


@pytest.fixture
def gdn(monkeypatch):
    """Load the GDN patch module with its two vllm import stubs installed.

    ``vllm.third_party.flash_linear_attention.ops.op`` (the ``exp`` builtin)
    and ``vllm.triton_utils`` (``tl`` / ``triton``) are external boundaries;
    both are faked here so the real module imports on machines without
    vLLM.  Triton's interpreter mode is enabled before importing triton so
    the ``@triton.jit`` decorator produces an ``InterpretedFunction`` that
    executes on CPU torch tensors.  If triton was already imported by an
    earlier test in this session (without the env var), the module-scope
    ``tl`` builtins are re-wrapped as interpreted functions.  All
    ``sys.modules`` changes go through ``monkeypatch.setitem`` for
    automatic restoration.
    """
    monkeypatch.setenv("TRITON_INTERPRET", "1")
    # The kernel under test executes through triton's interpreter mode; on
    # hosts without the package (CPU-only Windows) the whole fixture's
    # subject matter is unavailable, so the tests skip rather than error.
    triton = pytest.importorskip("triton")
    import triton.language as tl

    # FlagTree's build wires a lazy ``tl.ext`` symbol dispatcher whose
    # ``__getattr__`` raises RuntimeError (not AttributeError) for unknown
    # names.  The interpreter's ``_patch_builtin`` scans every ``tl`` member
    # with ``tl.core.is_builtin`` -> ``getattr(fn, "__triton_builtin__")``,
    # and ``getattr``'s default only swallows AttributeError, so the scan
    # itself crashes on the dispatcher.  Wrapping ``is_builtin`` to treat
    # that RuntimeError as "not a builtin" (which the dispatcher indeed is
    # not) keeps the kernel executable under FlagTree; upstream triton
    # never raises here, so the wrapper is a no-op pass-through there.
    real_is_builtin = tl.core.is_builtin

    def _safe_is_builtin(fn):
        try:
            return real_is_builtin(fn)
        except RuntimeError:
            return False

    monkeypatch.setattr(tl.core, "is_builtin", _safe_is_builtin)

    # FlagTree's language/semantic layer passes trailing load/store options
    # (e.g. ``flagtree_hints``) that its own InterpreterBuilder methods do
    # not accept — an internal mismatch in that wheel.  The interpreter
    # ignores cache/eviction hints anyway, so each builder method is wrapped
    # to drop positional arguments beyond its declared arity.  On upstream
    # triton the arities match and the wrappers are pass-throughs.
    from triton.runtime.interpreter import InterpreterBuilder

    def _truncate_positionals(name):
        orig = getattr(InterpreterBuilder, name, None)
        if orig is None:
            return
        params = inspect.signature(orig).parameters
        if any(p.kind is inspect.Parameter.VAR_POSITIONAL for p in params.values()):
            return
        n_pos = (
            sum(
                p.kind
                in (
                    inspect.Parameter.POSITIONAL_ONLY,
                    inspect.Parameter.POSITIONAL_OR_KEYWORD,
                )
                for p in params.values()
            )
            - 1  # exclude self
        )

        def wrapper(self, *args, **kwargs):
            return orig(self, *args[:n_pos], **kwargs)

        monkeypatch.setattr(InterpreterBuilder, name, wrapper)

    for name in (
        "create_load",
        "create_masked_load",
        "create_store",
        "create_masked_store",
    ):
        _truncate_positionals(name)

    if not triton.knobs.runtime.interpret or type(tl.sigmoid).__name__ != (
        "InterpretedFunction"
    ):
        import triton.language.standard as std

        _reinterp_tl_builtins(tl, std)

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

    module = _load(_GDN_SOURCE, _GDN_MODULE)
    yield module
    sys.modules.pop(_GDN_MODULE, None)


@pytest.fixture
def qsa(monkeypatch):
    """Load the QSA patch module (pure stdlib imports at module scope)."""
    module = _load(_QSA_SOURCE, _QSA_MODULE)
    yield module
    sys.modules.pop(_QSA_MODULE, None)


# ---------------------------------------------------------------------------
# GDN kernel numerics
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
    the output projection.  ``NULL_BLOCK_ID`` rows (state index <= 0)
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
            v = qkv[
                n,
                2 * H * K + i_hv * V : 2 * H * K + (i_hv + 1) * V,
            ].double()
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
        # No state slot was touched at all (the only valid row writes slot 2).
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


class TestGdnKernelMarker:
    """The FL marker on the shipped kernel."""

    def test_kernel_carries_fp32_beta_marker(self, gdn):
        """The replacement kernel is flagged so detection skips it."""
        kernel = gdn._fused_recurrent_gated_delta_rule_packed_decode_kernel_fp32_beta
        assert getattr(kernel, "_fl_fp32_beta", False) is True


# ---------------------------------------------------------------------------
# GDN detection and patching
# ---------------------------------------------------------------------------


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


def _function_with_source(source: str):
    """Compile a function whose inspect.getsource returns ``source``.

    ``inspect.getsource`` reads from ``linecache`` using the code object's
    filename; registering the source under a fake file makes the round trip
    work on any platform.
    """
    import linecache

    filename = f"<supp_gdn_source_{abs(hash(source))}>"
    namespace: dict = {}
    compiled = compile(source, filename, "exec")
    exec(compiled, namespace)
    function = namespace[source.split("(")[0].split()[-1]]  # first def name in source
    linecache.cache[filename] = (len(source), None, source.splitlines(True), filename)
    return function


class TestPatchVllmPackedGdnBeta:
    """``patch_vllm_packed_gdn_beta`` replacement flow."""

    def test_missing_target_module_returns_false(self, gdn):
        """Without the vLLM fused_recurrent module the patch is a no-op."""
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


# ---------------------------------------------------------------------------
# QSA cache hint: AST helper
# ---------------------------------------------------------------------------


class _FakeQsaKernel:
    """Kernel object mimicking a Triton JITFunction's source surface."""

    def __init__(self, src, device_caches=None):
        self.src = src
        self.device_caches = {} if device_caches is None else device_caches
        self.updates = []
        self.original_src = None

    def _unsafe_update_src(self, source):
        self.updates.append(source)
        self.src = source  # the real primitive replaces the cached source

    def _fl_original_src_marker(self):
        return self.original_src


_HINTED_SRC = (
    "def _qsa_mqa_paged_prefill_kernel(k_cache_ptr, other):\n"
    '    keys = tl.load(k_cache_ptr + other, eviction_policy="evict_first")\n'
)
_UNHINTED_SRC = (
    "def _qsa_mqa_paged_prefill_kernel(k_cache_ptr, other):\n"
    "    keys = tl.load(k_cache_ptr + other)\n"
)


class TestWithoutKCacheEvictionHint:
    """``_without_k_cache_eviction_hint`` AST surgery."""

    def test_removes_the_eviction_hint(self, qsa):
        """The eviction_policy keyword is removed; everything else survives."""
        result = qsa._without_k_cache_eviction_hint(_FakeQsaKernel(_HINTED_SRC))
        assert result is not None
        assert "eviction_policy" not in result
        # The load itself is intact, minus only the keyword.
        assert "keys = tl.load(k_cache_ptr + other)" in result
        assert "k_cache_ptr" in result and "other" in result

    def test_returns_none_when_already_patched(self, qsa):
        """A source without the hint reports None (already fixed)."""
        assert qsa._without_k_cache_eviction_hint(_FakeQsaKernel(_UNHINTED_SRC)) is None

    def test_rejects_missing_k_cache_load(self, qsa):
        """Zero matching K-cache loads raise the revalidation error."""
        src = "def _k(other):\n    keys = tl.load(other)\n"
        with pytest.raises(RuntimeError, match="K-cache load changed"):
            qsa._without_k_cache_eviction_hint(_FakeQsaKernel(src))

    def test_rejects_multiple_k_cache_loads(self, qsa):
        """Two matching loads raise instead of guessing which to change."""
        src = (
            "def _k(k_cache_ptr, other):\n"
            '    keys = tl.load(k_cache_ptr + 1, eviction_policy="evict_first")\n'
            '    keys = tl.load(k_cache_ptr + 2, eviction_policy="evict_first")\n'
        )
        with pytest.raises(RuntimeError, match="K-cache load changed"):
            qsa._without_k_cache_eviction_hint(_FakeQsaKernel(src))

    def test_rejects_wrong_policy_value(self, qsa):
        """A different eviction policy value raises."""
        src = (
            "def _k(k_cache_ptr, other):\n"
            '    keys = tl.load(k_cache_ptr + other, eviction_policy="evict_last")\n'
        )
        with pytest.raises(RuntimeError, match="eviction policy changed"):
            qsa._without_k_cache_eviction_hint(_FakeQsaKernel(src))

    def test_rejects_non_constant_policy(self, qsa):
        """A non-literal policy expression raises."""
        src = (
            "def _k(k_cache_ptr, other):\n"
            "    keys = tl.load(k_cache_ptr + other, eviction_policy=policy)\n"
        )
        with pytest.raises(RuntimeError, match="eviction policy changed"):
            qsa._without_k_cache_eviction_hint(_FakeQsaKernel(src))


# ---------------------------------------------------------------------------
# QSA cache hint: gate matrix
# ---------------------------------------------------------------------------


class _QsaPatchEnv:
    """Install the full external boundary the patch function consults."""

    ARCHITECTURE = "Qwen4ExpForConditionalGeneration"

    def __init__(self, monkeypatch):
        self.monkeypatch = monkeypatch
        self.kernels = {
            name: _FakeQsaKernel(_HINTED_SRC) for name in qsa_kernel_names()
        }
        self.qsa_indexer = ModuleType("vllm.models.qwen4_exp.nvidia.ops.qsa_indexer")
        for name, kernel in self.kernels.items():
            setattr(self.qsa_indexer, name, kernel)

        self.triton_stub = ModuleType("triton")
        self.triton_stub.__version__ = "3.6.0"

        vllm_pkg = ModuleType("vllm")
        vllm_pkg.__path__ = []
        parent = ModuleType("vllm.models.qwen4_exp.nvidia.ops")
        parent.__path__ = []
        monkeypatch.setitem(sys.modules, "vllm", vllm_pkg)
        monkeypatch.setitem(sys.modules, "vllm.models.qwen4_exp.nvidia.ops", parent)
        monkeypatch.setitem(
            sys.modules,
            "vllm.models.qwen4_exp.nvidia.ops.qsa_indexer",
            self.qsa_indexer,
        )
        monkeypatch.setitem(sys.modules, "triton", self.triton_stub)

        self._original_capability = torch.cuda.get_device_capability
        self._original_version = qsa_version_fn()
        self.config = SimpleNamespace(architecture=self.ARCHITECTURE)
        self.device = SimpleNamespace(type="cuda")

    def set_triton_version(self, version: str):
        self.triton_stub.__version__ = version

    def set_capability(self, capability):
        torch.cuda.get_device_capability = (
            (lambda device=None: capability)
            if capability is not None
            else self._original_capability
        )

    def set_flagtree_version(self, resolver):
        """Point the module-level ``version`` at a custom resolver."""
        module = sys.modules[_QSA_MODULE]
        module.version = resolver

    def restore(self):
        torch.cuda.get_device_capability = self._original_capability
        module = sys.modules.get(_QSA_MODULE)
        if module is not None:
            module.version = self._original_version


def qsa_kernel_names():
    """The two kernel names the patch targets."""
    return (
        "_qsa_mqa_paged_prefill_kernel",
        "_qsa_mqa_paged_uniform_kernel",
    )


def qsa_version_fn():
    """The pristine ``importlib.metadata.version`` reference."""
    from importlib.metadata import version

    return version


@pytest.fixture
def qsa_env(monkeypatch):
    """Install the patch's external boundaries with defaults that succeed."""
    env = _QsaPatchEnv(monkeypatch)
    env.set_flagtree_version(lambda name: "0.1.0")
    env.set_capability((9, 0))
    try:
        yield env
    finally:
        env.restore()


class TestPatchQsaCacheHintGates:
    """Every early-return gate of ``patch_qsa_cache_hint``."""

    def test_success_patches_both_kernels(self, qsa, qsa_env):
        """All gates open: both kernels lose their hint and keep a backup."""
        assert qsa.patch_qsa_cache_hint(qsa_env.config, qsa_env.device) is True
        for kernel in qsa_env.kernels.values():
            assert len(kernel.updates) == 1
            assert "eviction_policy" not in kernel.updates[0]
            # The original source is preserved before mutation.
            assert getattr(kernel, "_fl_qsa_original_src", None) == _HINTED_SRC

    def test_second_call_reports_no_updates(self, qsa, qsa_env):
        """After patching, the changed sources yield no further updates."""
        assert qsa.patch_qsa_cache_hint(qsa_env.config, qsa_env.device) is True
        assert qsa.patch_qsa_cache_hint(qsa_env.config, qsa_env.device) is False
        for kernel in qsa_env.kernels.values():
            assert len(kernel.updates) == 1

    def test_triton_36_patch_series_proceeds(self, qsa, qsa_env):
        """Any 3.6.x micro version passes the version gate."""
        for version in ("3.6.0", "3.6.2"):
            qsa_env.set_triton_version(version)
            assert qsa.patch_qsa_cache_hint(qsa_env.config, qsa_env.device) is True
            # Reset for the next iteration of the parametrized style loop.
            for kernel in qsa_env.kernels.values():
                kernel.updates.clear()
                kernel.src = _HINTED_SRC

    def test_compiled_kernel_raises(self, qsa, qsa_env):
        """A kernel with live device caches refuses the source change."""
        first, second = qsa_kernel_names()
        qsa_env.kernels[first].device_caches = {"cuda:0": object()}
        with pytest.raises(RuntimeError, match="already compiled"):
            qsa.patch_qsa_cache_hint(qsa_env.config, qsa_env.device)
        # The untouched second kernel was never mutated either: both are
        # validated before either is changed.
        assert qsa_env.kernels[second].updates == []

    def test_architectures_both_accepted(self, qsa, qsa_env):
        """Both Qwen4Exp architecture spellings pass the arch gate."""
        qsa.patch_qsa_cache_hint(qsa_env.config, qsa_env.device)
        for kernel in qsa_env.kernels.values():
            kernel.updates.clear()
            kernel.src = _HINTED_SRC
        causal = SimpleNamespace(architecture="Qwen4ExpForCausalLM")
        assert qsa.patch_qsa_cache_hint(causal, qsa_env.device) is True
