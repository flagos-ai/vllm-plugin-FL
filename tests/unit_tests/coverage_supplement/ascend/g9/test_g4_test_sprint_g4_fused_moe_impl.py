"""Sprint G4 coverage tests for ``vllm_fl/ops/fused_moe/fused_moe.py``.

Target branch: 814b.  ``fused_experts_impl`` was entirely unexecuted there
(miss 57 lines), so these tests drive the whole chunked two-GEMM pipeline on
CPU with every kernel/quantization boundary stubbed:

* ``try_get_optimal_moe_config``      -> returns a fixed BLOCK_SIZE_M
* ``moe_kernel_quantize_input``       -> identity (returns A, None)
* ``moe_align_block_size``            -> placeholder tuple
* ``invoke_fused_moe_triton_kernel``  -> recorder
* ``silu_and_mul`` / ``gelu_and_mul`` -> shape-correct empty tensors
* ``moe_sum``                         -> real sum over the top-k axis

Covered 814b fused_moe.py lines: 71-76,81,82 (problem size + chunking),
84,94,98,107 (config/quant dtype + cached config), 111-117,120 (workspace
caches), 124-129,131 (compute type + unsupported dtype), 133-139 (inplace
and chunk loop), 143-149 (chunk slicing + empty-chunk break), 154-159
(last-chunk cache resize), 161,162,164,172,180 (quantize/align/GEMM1),
206-209,212,213,215-218,221 (activation dispatch), 223,231 (quantize/GEMM2),
254,259 (moe_sum + return).  274-302 (the ``fused_experts`` wrapper) is also
exercised.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

try:  # pragma: no cover - import guard is environment specific
    import vllm_fl.ops.fused_moe.fused_moe as Mod
except Exception as exc:  # pragma: no cover
    pytest.skip(f"cannot import fused_moe: {exc}", allow_module_level=True)


class _ActivationEnum:
    """Stand-in for ``MoEActivation`` exposing a ``.value`` attribute."""

    def __init__(self, value: str):
        self.value = value


@pytest.fixture(autouse=True)
def _stub_kernels(monkeypatch):
    """Replace every GPU/triton/quantization boundary with a recorder."""
    calls = {
        "config": [],
        "align": [],
        "invoke": [],
        "quant": [],
        "sum": [],
        "act": [],
    }

    def _config(w1_size, w2_size, top_k, dtype, M, block_shape=None):
        calls["config"].append(M)
        return {"BLOCK_SIZE_M": 16}

    def _align(topk_ids, block, num_experts, expert_map, ignore_invalid_experts=False):
        calls["align"].append(
            (tuple(topk_ids.shape), block, num_experts, ignore_invalid_experts)
        )
        return None, None, None

    def _invoke(*args, **kwargs):
        calls["invoke"].append((args, kwargs))

    def _quant(
        A,
        A_scale,
        quant_dtype,
        per_act_token_quant,
        block_shape,
        quantization_emulation=None,
    ):
        calls["quant"].append(tuple(A.shape))
        return A, None

    def _silu(_dummy, x):
        calls["act"].append("silu")
        return torch.empty((x.shape[0], x.shape[1] // 2), dtype=x.dtype)

    def _gelu(_dummy, x):
        calls["act"].append("gelu")
        return torch.empty((x.shape[0], x.shape[1] // 2), dtype=x.dtype)

    def _moe_sum(input_t, output_t):
        calls["sum"].append((tuple(input_t.shape), tuple(output_t.shape)))

    # The installed triton.language stand-in may not expose the scalar dtype
    # handles that fused_experts_impl reads (tl.float32/float16/bfloat16), so
    # give the module a complete namespace for the compute_type dispatch.
    monkeypatch.setattr(
        Mod,
        "tl",
        SimpleNamespace(
            float32="tl.float32",
            float16="tl.float16",
            bfloat16="tl.bfloat16",
        ),
    )
    monkeypatch.setattr(Mod, "try_get_optimal_moe_config", _config)
    monkeypatch.setattr(Mod, "_moe_align_block_size", _align)
    monkeypatch.setattr(Mod, "_invoke_fused_moe_triton_kernel", _invoke)
    monkeypatch.setattr(Mod, "moe_kernel_quantize_input", _quant)
    monkeypatch.setattr(Mod, "_silu_and_mul", _silu)
    monkeypatch.setattr(Mod, "_gelu_and_mul", _gelu)
    monkeypatch.setattr(Mod, "_moe_sum", _moe_sum)
    if hasattr(Mod, "_get_config_dtype_str"):
        monkeypatch.setattr(Mod, "_get_config_dtype_str", lambda **kwargs: "dummy")
    if hasattr(Mod, "_get_config_quant_dtype"):
        monkeypatch.setattr(Mod, "_get_config_quant_dtype", lambda **kwargs: None)
    return calls


def _inputs(num_tokens=8, top_k=2, N=6, K=4, E=2, dtype=torch.float32):
    hidden = torch.randn(num_tokens, K, dtype=dtype)
    w1 = torch.randn(E, N, K, dtype=dtype)
    w2 = torch.randn(E, K, N, dtype=dtype)
    topk_ids = torch.randint(0, E, (num_tokens, top_k), dtype=torch.int32)
    topk_weights = torch.rand(num_tokens, top_k, dtype=dtype)
    return hidden, w1, w2, topk_weights, topk_ids


def test_fused_experts_impl_silu_out_of_place(_stub_kernels):
    """Checks the happy path: shapes, config calls and two GEMM invocations."""
    hidden, w1, w2, topk_weights, topk_ids = _inputs()
    out = Mod.fused_experts_impl(
        hidden, w1, w2, topk_weights, topk_ids, activation="silu"
    )
    assert out.shape == hidden.shape
    assert out is not hidden
    assert _stub_kernels["config"] == [8]
    assert _stub_kernels["align"] == [((8, 2), 16, 2, True)]
    assert len(_stub_kernels["invoke"]) == 2
    assert _stub_kernels["sum"] == [((8, 2, 4), (8, 4))]
    assert _stub_kernels["act"] == ["silu"]


def test_fused_experts_impl_inplace_returns_input(_stub_kernels):
    """Checks inplace=True reuses the input tensor as the output buffer."""
    hidden, w1, w2, topk_weights, topk_ids = _inputs()
    out = Mod.fused_experts_impl(
        hidden, w1, w2, topk_weights, topk_ids, inplace=True, activation="silu"
    )
    assert out is hidden


def test_fused_experts_impl_counter_op_default(_stub_kernels):
    """Checks global_num_experts=-1 is replaced by the local expert count."""
    hidden, w1, w2, topk_weights, topk_ids = _inputs(E=3)
    Mod.fused_experts_impl(hidden, w1, w2, topk_weights, topk_ids, activation="silu")
    assert _stub_kernels["align"][0][2] == 3


@pytest.mark.parametrize(
    ("activation", "expected_mul"),
    [("gelu", True), ("silu_no_mul", False), ("gelu_no_mul", False)],
)
def test_fused_experts_impl_activation_variants(
    _stub_kernels, activation, expected_mul
):
    """Checks the gelu/no-mul activation dispatch branches."""
    hidden, w1, w2, topk_weights, topk_ids = _inputs()
    out = Mod.fused_experts_impl(
        hidden, w1, w2, topk_weights, topk_ids, activation=activation
    )
    assert out.shape == hidden.shape
    assert len(_stub_kernels["invoke"]) == 2
    if expected_mul:
        assert _stub_kernels["act"] == ["gelu"]
    else:
        assert _stub_kernels["act"] == []


def test_fused_experts_impl_accepts_activation_enum(_stub_kernels):
    """Checks an enum-like activation is normalized through ``.value``."""
    hidden, w1, w2, topk_weights, topk_ids = _inputs()
    out = Mod.fused_experts_impl(
        hidden, w1, w2, topk_weights, topk_ids, activation=_ActivationEnum("silu")
    )
    assert out.shape == hidden.shape
    assert _stub_kernels["act"] == ["silu"]


def test_fused_experts_impl_rejects_unknown_activation(_stub_kernels):
    """Checks an unsupported activation string raises ValueError."""
    hidden, w1, w2, topk_weights, topk_ids = _inputs()
    with pytest.raises(ValueError, match="Unsupported FusedMoe activation"):
        Mod.fused_experts_impl(
            hidden, w1, w2, topk_weights, topk_ids, activation="relu"
        )


@pytest.mark.parametrize(
    ("dtype", "tl_name"),
    [(torch.bfloat16, "bfloat16"), (torch.float16, "float16")],
)
def test_fused_experts_impl_compute_type_mapping(_stub_kernels, dtype, tl_name):
    """Checks the bf16/fp16 branches of the compute_type dispatch."""
    hidden, w1, w2, topk_weights, topk_ids = _inputs(dtype=dtype)
    out = Mod.fused_experts_impl(
        hidden, w1, w2, topk_weights, topk_ids, activation="silu"
    )
    assert out.shape == hidden.shape
    assert _stub_kernels["invoke"][0][1]["compute_type"] == getattr(Mod.tl, tl_name)


def test_fused_experts_impl_breaks_on_empty_chunk(_stub_kernels):
    """Checks an exact multiple of the chunk size breaks on the empty tail."""
    num_tokens = 65536
    hidden, w1, w2, topk_weights, topk_ids = _inputs(
        num_tokens=num_tokens, top_k=1, N=4, K=2
    )
    out = Mod.fused_experts_impl(
        hidden, w1, w2, topk_weights, topk_ids, activation="silu"
    )
    assert out.shape == (num_tokens, 2)
    assert _stub_kernels["config"] == [num_tokens]
    assert len(_stub_kernels["invoke"]) == 2
    assert len(_stub_kernels["sum"]) == 1


def test_fused_experts_impl_chunk_boundary_and_last_chunk_resize(_stub_kernels):
    """Checks the second chunk hits the resize branch and a third breaks."""
    num_tokens = 65537  # one full chunk + a 1-token tail + an empty iteration
    hidden, w1, w2, topk_weights, topk_ids = _inputs(
        num_tokens=num_tokens, top_k=1, N=4, K=2
    )
    out = Mod.fused_experts_impl(
        hidden, w1, w2, topk_weights, topk_ids, activation="silu"
    )
    assert out.shape == (num_tokens, 2)
    assert _stub_kernels["config"] == [65536, 1]
    assert len(_stub_kernels["invoke"]) == 4
    assert len(_stub_kernels["sum"]) == 2
    assert _stub_kernels["sum"][-1] == ((1, 1, 2), (1, 2))


def test_fused_experts_impl_int4_halves_hidden_size(_stub_kernels):
    """Checks the int4 path compares hidden/2 against w1's last dim."""
    hidden = torch.randn(4, 4, dtype=torch.float32)
    w1 = torch.randn(2, 6, 2, dtype=torch.float32)
    w2 = torch.randn(2, 4, 6, dtype=torch.float32)
    topk_ids = torch.zeros(4, 1, dtype=torch.int32)
    topk_weights = torch.rand(4, 1)
    out = Mod.fused_experts_impl(
        hidden,
        w1,
        w2,
        topk_weights,
        topk_ids,
        activation="silu",
        use_int4_w4a16=True,
    )
    assert out.shape == hidden.shape


class _ScriptedDtypeTensor(torch.Tensor):
    """Tensor whose ``dtype`` reads a scripted sequence (branch 131 probe)."""

    @property
    def dtype(self):
        sequence = self.__dict__.get("_dtype_sequence")
        if sequence is None:
            return super().dtype
        index = self.__dict__.get("_dtype_cursor", 0)
        self.__dict__["_dtype_cursor"] = index + 1
        return sequence[min(index, len(sequence) - 1)]


def test_fused_experts_impl_rejects_unsupported_compute_dtype(_stub_kernels):
    """Checks the else-branch at the compute_type dispatch is reachable.

    The public assertion restricts dtypes to fp32/fp16/bf16, so the only way
    to reach line 131 is a tensor whose ``dtype`` changes between the
    assertion and the dispatch.  This test scripts that; if the torch build
    does not honour the property override it skips instead of failing.
    """
    hidden, w1, w2, topk_weights, topk_ids = _inputs()
    try:
        scripted = hidden.as_subclass(_ScriptedDtypeTensor)
        scripted.__dict__["_dtype_sequence"] = [
            torch.float32,
            torch.float32,
            torch.float32,
            torch.float32,
            torch.float64,
        ]
        scripted.__dict__["_dtype_cursor"] = 0
    except Exception as exc:  # pragma: no cover - torch build specific
        pytest.skip(f"tensor subclasses cannot carry state: {exc}")

    try:
        Mod.fused_experts_impl(
            scripted, w1, w2, topk_weights, topk_ids, inplace=True, activation="silu"
        )
    except ValueError as exc:
        assert "Unsupported compute_type" in str(exc)
    else:  # pragma: no cover - torch build specific
        pytest.skip("dtype override was not observed")


@pytest.mark.parametrize(
    ("hidden_kwargs", "w1_shape", "w2_shape"),
    [
        ({"num_tokens": 4, "K": 4}, (2, 6, 5), (2, 4, 6)),
        ({"num_tokens": 4, "K": 4, "dtype": torch.float64}, (2, 6, 4), (2, 4, 6)),
    ],
)
def test_fused_experts_impl_rejects_bad_inputs(
    _stub_kernels, hidden_kwargs, w1_shape, w2_shape
):
    """Checks the shape/dtype assertions guard the kernel entry."""
    hidden = torch.randn(
        hidden_kwargs["num_tokens"], hidden_kwargs["K"],
        dtype=hidden_kwargs.get("dtype", torch.float32),
    )
    w1 = torch.randn(*w1_shape)
    w2 = torch.randn(*w2_shape)
    topk_ids = torch.zeros(4, 1, dtype=torch.int32)
    topk_weights = torch.rand(4, 1)
    with pytest.raises(AssertionError):
        Mod.fused_experts_impl(hidden, w1, w2, topk_weights, topk_ids)


def test_fused_experts_wrapper_forwards_quant_config(monkeypatch):
    """Checks the public wrapper forwards a explicit quant config verbatim."""
    captured = {}

    def _impl(**kwargs):
        captured.update(kwargs)
        return "forwarded"

    monkeypatch.setattr(Mod, "fused_experts_impl", _impl)
    hidden, w1, w2, topk_weights, topk_ids = _inputs()
    quant_config = SimpleNamespace(
        use_fp8_w8a8=False,
        use_int8_w8a8=True,
        use_int8_w8a16=False,
        use_int4_w4a16=False,
        per_act_token_quant=True,
        w1_scale=None,
        w2_scale=None,
        w1_zp=None,
        w2_zp=None,
        a1_scale=None,
        a2_scale=None,
        block_shape=None,
        w1_bias=None,
        w2_bias=None,
    )
    out = Mod.fused_experts(
        hidden,
        w1,
        w2,
        topk_weights,
        topk_ids,
        activation="gelu",
        quant_config=quant_config,
        apply_router_weight_on_input=True,
        global_num_experts=5,
    )
    assert out == "forwarded"
    assert captured["activation"] == "gelu"
    assert captured["use_int8_w8a8"] is True
    assert captured["per_channel_quant"] is True
    assert captured["apply_router_weight_on_input"] is True
    assert captured["global_num_experts"] == 5


def test_fused_experts_wrapper_defaults_to_unquantized_config(monkeypatch):
    """Checks quant_config=None falls back to the unquantized config."""
    monkeypatch.setattr(Mod, "fused_experts_impl", lambda **kwargs: kwargs)
    hidden, w1, w2, topk_weights, topk_ids = _inputs()
    try:
        captured = Mod.fused_experts(hidden, w1, w2, topk_weights, topk_ids)
    except AttributeError as exc:  # pragma: no cover - vllm build specific
        pytest.skip(f"default unquantized config lacks an attribute: {exc}")
    assert captured["use_fp8_w8a8"] is False
    assert captured["use_int4_w4a16"] is False
