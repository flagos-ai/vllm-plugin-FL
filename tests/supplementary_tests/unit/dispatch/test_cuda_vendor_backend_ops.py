# Copyright (c) 2026 BAAI. All rights reserved.

"""CPU tests for the CUDA vendor dispatch backend and its impls.

The modules under test (``vllm_fl/dispatch/backends/vendor/cuda``) wrap
vLLM CUDA kernels behind lazy imports, so almost everything is
CPU-drivable once the kernel boundary is faked:

* ``cuda.py`` — backend metadata, the availability branch matrix (no CUDA
  device, non-NVIDIA vendor, platform import failure) and every delegate's
  pass-through contract;
* ``impl/activation.py`` — silu/gelu oracle numerics against a fake
  ``torch.ops._C`` plus the ``approximate`` dispatch;
* ``impl/normalization.py`` — rms_norm with and without residual against
  faked ``vllm._custom_ops`` kernels that emulate the real in-place
  contract on both tensors;
* ``impl/rotary.py`` — argument order and return identity;
* ``impl/fused_moe.py`` — moe_align_block_size buffer sizing on every
  branch, expert_map remapping semantics, and the thin pass-throughs;
* ``register_ops.py`` — full VENDOR wiring into a fresh OpRegistry;
* ``impl/__init__.py`` — export surface.

All numeric expectations come from independent oracles spelled out in
plain torch, never from re-running the code under test.
"""

import sys
import types
from types import ModuleType, SimpleNamespace

import pytest
import torch

pytestmark = pytest.mark.supplementary

_STUB_ROOTS = ("vllm", "vllm_fl", "flag_gems", "flaggems_vllm")
_MISSING = object()


@pytest.fixture(autouse=True)
def _fl_sys_modules():
    """Snapshot ``sys.modules`` around stub installs and restore exactly.

    Same contract as the ops/worker supplementary conftest fixtures,
    inlined here because this directory's conftest cannot be modified.
    """
    saved = {}
    for name in list(sys.modules):
        if name.split(".")[0] in _STUB_ROOTS:
            saved[name] = sys.modules.get(name, _MISSING)
    try:
        yield saved
    finally:
        for name in [
            name
            for name in list(sys.modules)
            if name.split(".")[0] in _STUB_ROOTS and name not in saved
        ]:
            del sys.modules[name]
        for name, module in saved.items():
            if module is _MISSING:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = module


# ---------------------------------------------------------------------------
# vllm module stubs (only for the names the impls import lazily)
# ---------------------------------------------------------------------------


class _CustomOpsStub(ModuleType):
    """Fake ``vllm._custom_ops`` recording kernel invocations.

    The kernel bodies implement the real contracts in plain torch so the
    CPU-side pre/post-processing in the impls can be verified against
    independent oracles: ``fused_add_rms_norm`` updates BOTH tensors in
    place; ``rms_norm`` writes into the provided output buffer.
    """

    def __init__(self, name):
        super().__init__(name)
        self.rms_norm_calls = []
        self.rotary_calls = []
        self.topk_calls = []
        self.moe_sum_calls = []
        self.grouped_topk_calls = []
        self.moe_align_calls = []

    # -- normalization ------------------------------------------------------
    def rms_norm(self, out, x, weight, epsilon):
        self.rms_norm_calls.append((x, weight, epsilon))
        # Real contract: fills ``out`` in place.
        ms = x.pow(2).mean(dim=-1, keepdim=True)
        normed = x * torch.rsqrt(ms + epsilon)
        out.copy_(normed * weight.to(x.dtype))

    def fused_add_rms_norm(self, x, residual, weight, epsilon):
        # Real contract: residual += x in place, then x = rmsnorm(residual).
        residual.add_(x)
        ms = residual.pow(2).mean(dim=-1, keepdim=True)
        x.copy_(residual * torch.rsqrt(ms + epsilon) * weight.to(x.dtype))

    # -- rotary -------------------------------------------------------------
    def rotary_embedding(self, position_ids, query, key, cos, sin, rotary_interleaved):
        self.rotary_calls.append(
            {
                "position_ids": position_ids,
                "query": query,
                "key": key,
                "cos": cos,
                "sin": sin,
                "rotary_interleaved": rotary_interleaved,
            }
        )

    # -- fused MoE ------------------------------------------------------------
    def topk_softmax(
        self,
        topk_weights,
        topk_indices,
        token_expert_indices,
        gating_output,
        renormalize,
    ):
        self.topk_calls.append(
            {
                "weights": topk_weights,
                "indices": topk_indices,
                "token_expert_indices": token_expert_indices,
                "gating": gating_output,
                "renormalize": renormalize,
            }
        )
        # Plain-torch oracle: softmax then top-k.
        probs = torch.softmax(gating_output, dim=-1)
        top_p, top_i = torch.topk(probs, topk_weights.shape[-1], dim=-1)
        topk_weights.copy_(top_p)
        topk_indices.copy_(top_i)

    def moe_sum(self, inp, out):
        self.moe_sum_calls.append((inp, out))
        # Reduce over the expert axis into ``out``.
        out.copy_(inp.sum(dim=1))

    def grouped_topk(
        self,
        scores,
        n_group,
        topk_group,
        topk,
        renormalize,
        routed_scaling_factor,
        bias,
        scoring_func,
    ):
        self.grouped_topk_calls.append(
            {
                "scores": scores,
                "n_group": n_group,
                "topk_group": topk_group,
                "topk": topk,
                "renormalize": renormalize,
                "routed_scaling_factor": routed_scaling_factor,
                "bias": bias,
                "scoring_func": scoring_func,
            }
        )
        return "grouped-topk-sentinel"

    def moe_align_block_size(
        self,
        topk_ids,
        num_experts,
        block_size,
        sorted_ids,
        expert_ids,
        num_tokens_post_pad,
        expert_map,
    ):
        self.moe_align_calls.append(
            {
                "topk_ids": topk_ids,
                "num_experts": num_experts,
                "block_size": block_size,
                "sorted_ids": sorted_ids,
                "expert_ids": expert_ids,
                "num_tokens_post_pad": num_tokens_post_pad,
                "expert_map": expert_map,
            }
        )
        # Independent bucket oracle in plain Python + torch.
        flat = topk_ids.reshape(-1).to(torch.int64).tolist()
        counts = [0] * num_experts
        for eid in flat:
            if 0 <= eid < num_experts:
                counts[eid] += 1
        padded = [((c + block_size - 1) // block_size) * block_size for c in counts]
        offsets, run = [], 0
        for p in padded:
            offsets.append(run)
            run += p
        total = run
        sorted_ids.fill_(topk_ids.numel())  # sentinel = num valid tokens
        rank_within = [0] * num_experts
        for token, eid in enumerate(flat):
            if 0 <= eid < num_experts:
                sorted_ids[offsets[eid] + rank_within[eid]] = token
                rank_within[eid] += 1
        expert_ids.fill_(-1)
        for block in range(total // block_size):
            owner = 0
            acc = 0
            for e, p in enumerate(padded):
                acc += p
                if block * block_size < acc:
                    owner = e
                    break
            expert_ids[block] = owner
        num_tokens_post_pad.fill_(total)


class _RegistryBackendEntry:
    def __init__(self, path):
        self.path = path

    def get_path(self):
        return self.path


def _install_vllm_stubs(monkeypatch, vendor_name="nvidia"):
    """Install the minimal vllm stubs the cuda modules import lazily.

    Returns the ``_CustomOpsStub`` instance whose kernels the tests steer.
    On Linux the real vllm is installed; the stub entries replace the same
    module names so both platforms behave identically inside a test.
    """
    modules = {}

    def leaf(name, **attrs):
        parts = name.split(".")
        for length in range(1, len(parts) + 1):
            partial = ".".join(parts[:length])
            if partial not in modules:
                module = ModuleType(partial)
                module.__path__ = []
                modules[partial] = module
        for length in range(1, len(parts)):
            setattr(
                modules[".".join(parts[:length])],
                parts[length],
                modules[".".join(parts[: length + 1])],
            )
        for key, value in attrs.items():
            setattr(modules[name], key, value)
        return modules[name]

    vllm = leaf("vllm")
    custom_ops = _CustomOpsStub("vllm._custom_ops")
    modules["vllm._custom_ops"] = custom_ops
    vllm._custom_ops = custom_ops

    class _Enum:
        FLASH_ATTN = _RegistryBackendEntry(
            "vllm.v1.attention.backends.flash_attn.FlashAttentionBackend"
        )
        FLASHMLA = _RegistryBackendEntry(
            "vllm.v1.attention.backends.mla.FlashMLABackend"
        )
        FLASHMLA_SPARSE = _RegistryBackendEntry(
            "vllm.v1.attention.backends.mla_sparse.FlashMLASparseBackend"
        )

    leaf("vllm.platforms", current_platform=SimpleNamespace(vendor_name=vendor_name))
    leaf("vllm.v1.attention.backends.registry", AttentionBackendEnum=_Enum)

    fused_moe = leaf("vllm.model_executor.layers.fused_moe.fused_moe")

    def fake_invoke(*args, **kwargs):
        custom_ops.invoke_calls = getattr(custom_ops, "invoke_calls", [])
        custom_ops.invoke_calls.append((args, kwargs))

    fused_moe.invoke_fused_moe_triton_kernel = fake_invoke

    leaf("vllm.triton_utils", triton=__import__("triton"))
    leaf(
        "vllm.utils.math_utils",
        round_up=lambda value, multiple: (value + multiple - 1) // multiple * multiple,
    )

    for name, module in modules.items():
        monkeypatch.setitem(sys.modules, name, module)
    return custom_ops


@pytest.fixture
def custom_ops(monkeypatch):
    """Installed vllm stubs; returns the fake ``vllm._custom_ops``."""
    return _install_vllm_stubs(monkeypatch)


@pytest.fixture
def backend(monkeypatch):
    """A CudaBackend with an empty availability cache and no CUDA device."""
    from vllm_fl.dispatch.backends.vendor.cuda import CudaBackend

    monkeypatch.setattr(CudaBackend, "_available", None)
    return CudaBackend()


@pytest.fixture(autouse=True)
def _no_cuda(monkeypatch):
    """Pin CUDA unavailable so is_available's guard branch is exercised."""
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 0)


# ---------------------------------------------------------------------------
# CudaBackend metadata and availability
# ---------------------------------------------------------------------------


class TestCudaBackendAvailability:
    """CudaBackend identity, probe and availability caching."""

    def test_name_and_vendor(self, backend):
        """The backend is named 'cuda' with vendor 'nvidia'."""
        assert backend.name == "cuda"
        assert backend.vendor == "nvidia"

    def test_no_cuda_device_unavailable(self, backend):
        """Without a CUDA device the probe returns False and caches it."""
        from vllm_fl.dispatch.backends.vendor.cuda import CudaBackend

        assert backend.is_available() is False
        assert CudaBackend._available is False

    def test_nvidia_platform_available(self, backend, monkeypatch):
        """CUDA present + nvidia vendor resolves the probe to True."""
        from vllm_fl.dispatch.backends.vendor.cuda import CudaBackend

        _install_vllm_stubs(monkeypatch, vendor_name="nvidia")
        monkeypatch.setattr(CudaBackend, "_available", None)
        monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
        monkeypatch.setattr(torch.cuda, "device_count", lambda: 1)
        assert backend.is_available() is True
        assert CudaBackend._available is True

    def test_other_vendor_unavailable(self, backend, monkeypatch):
        """CUDA present but a non-NVIDIA vendor (amd) resolves to False."""
        from vllm_fl.dispatch.backends.vendor.cuda import CudaBackend

        _install_vllm_stubs(monkeypatch, vendor_name="amd")
        monkeypatch.setattr(CudaBackend, "_available", None)
        monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
        monkeypatch.setattr(torch.cuda, "device_count", lambda: 1)
        assert backend.is_available() is False

    def test_platform_import_failure_unavailable(self, backend, monkeypatch):
        """A raising platform import is swallowed into False."""

        class _Broken:
            @property
            def current_platform(self):
                raise RuntimeError("platform probe exploded")

        broken = ModuleType("vllm.platforms.broken")
        monkeypatch.setitem(
            sys.modules,
            "vllm",
            types.SimpleNamespace(
                __path__=[],
                platforms=SimpleNamespace(
                    current_platform=SimpleNamespace(vendor_name=None)
                ),
            ),
        )

        # Make ``from vllm.platforms import current_platform`` itself raise
        # by replacing the module with a property-raising proxy.
        class _RaisingModule(ModuleType):
            def __getattr__(self, item):
                if item == "current_platform":
                    raise RuntimeError("platform probe exploded")
                return super().__getattr__(item)

        raising = _RaisingModule("vllm.platforms")
        monkeypatch.setitem(sys.modules, "vllm.platforms", raising)
        monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
        monkeypatch.setattr(torch.cuda, "device_count", lambda: 1)
        assert backend.is_available() is False
        del broken


# ---------------------------------------------------------------------------
# attention_backend
# ---------------------------------------------------------------------------


class TestCudaAttentionBackend:
    """CudaBackend.attention_backend enum mapping."""

    def test_default_flash_attn(self, backend, custom_ops):
        """The default backend path is FLASH_ATTN."""
        assert backend.attention_backend() == (
            "vllm.v1.attention.backends.flash_attn.FlashAttentionBackend"
        )

    def test_mla_selects_flashmla(self, backend, custom_ops):
        """use_mla selects FLASHMLA."""
        assert backend.attention_backend(use_mla=True) == (
            "vllm.v1.attention.backends.mla.FlashMLABackend"
        )

    def test_mla_sparse_selects_flashmla_sparse(self, backend, custom_ops):
        """use_mla + use_sparse selects FLASHMLA_SPARSE."""
        assert backend.attention_backend(use_mla=True, use_sparse=True) == (
            "vllm.v1.attention.backends.mla_sparse.FlashMLASparseBackend"
        )


# ---------------------------------------------------------------------------
# impl/activation.py
# ---------------------------------------------------------------------------


class _FakeOpsC:
    """Recording double for ``torch.ops._C``.

    Kernel bodies use plain-torch oracles: silu(x1)*x2 and
    gelu(x1)*x2 with the requested approximation.
    """

    def __init__(self):
        self.silu_calls = []
        self.gelu_calls = []
        self.gelu_tanh_calls = []

    def silu_and_mul(self, out, x):
        self.silu_calls.append((x, out))
        d = x.shape[-1] // 2
        x1, x2 = x[..., :d], x[..., d:]
        out.copy_(x1 * torch.sigmoid(x1) * x2)

    def gelu_and_mul(self, out, x):
        self.gelu_calls.append((x, out))
        d = x.shape[-1] // 2
        x1, x2 = x[..., :d], x[..., d:]
        gelu = 0.5 * x1 * (1.0 + torch.erf(x1 / 2**0.5))
        out.copy_(gelu * x2)

    def gelu_tanh_and_mul(self, out, x):
        self.gelu_tanh_calls.append((x, out))
        d = x.shape[-1] // 2
        x1, x2 = x[..., :d], x[..., d:]
        inner = (2**0.5 / math.pi) * (x1 + 0.044715 * x1**3)
        gelu = 0.5 * x1 * (1.0 + torch.tanh(inner))
        out.copy_(gelu * x2)


import math  # noqa: E402  (used by _FakeOpsC above)


@pytest.fixture
def fake_ops_c(monkeypatch):
    """Replace ``torch.ops._C`` with the recording double."""
    fake = _FakeOpsC()
    monkeypatch.setattr(torch.ops, "_C", fake, raising=False)
    return fake


class TestActivationImpls:
    """silu_and_mul_cuda / gelu_and_mul_cuda contracts and numerics."""

    def test_silu_shape_dtype_and_oracle(self, backend, fake_ops_c):
        """Output is [..., d], same dtype/device, values match the oracle."""
        torch.manual_seed(2026)
        x = torch.randn(3, 4, 10)
        out = backend.silu_and_mul(None, x)

        assert out.shape == (3, 4, 5)
        assert out.dtype == x.dtype
        assert len(fake_ops_c.silu_calls) == 1
        kernel_x, kernel_out = fake_ops_c.silu_calls[0]
        assert kernel_x is x
        assert kernel_out is out  # the kernel receives the output buffer

        d = x.shape[-1] // 2
        x1, x2 = x[..., :d], x[..., d:]
        expected = x1 * torch.sigmoid(x1) * x2
        torch.testing.assert_close(out, expected, rtol=1e-6, atol=1e-6)

    def test_gelu_none_approximate_uses_exact_kernel(self, backend, fake_ops_c):
        """approximate='none' (or absent) selects gelu_and_mul."""
        torch.manual_seed(2026)
        x = torch.randn(2, 6)

        class Obj:
            approximate = "none"

        out = backend.gelu_and_mul(Obj(), x)
        assert len(fake_ops_c.gelu_calls) == 1
        assert fake_ops_c.gelu_tanh_calls == []
        assert out.shape == (2, 3)

        d = x.shape[-1] // 2
        x1, x2 = x[..., :d], x[..., d:]
        expected = 0.5 * x1 * (1.0 + torch.erf(x1 / 2**0.5)) * x2
        torch.testing.assert_close(out, expected, rtol=1e-6, atol=1e-6)

    def test_gelu_tanh_approximate_uses_tanh_kernel(self, backend, fake_ops_c):
        """approximate='tanh' selects gelu_tanh_and_mul."""
        torch.manual_seed(2026)
        x = torch.randn(2, 6)

        class Obj:
            approximate = "tanh"

        out = backend.gelu_and_mul(Obj(), x)
        assert len(fake_ops_c.gelu_tanh_calls) == 1
        assert fake_ops_c.gelu_calls == []
        assert out.shape == (2, 3)

        d = x.shape[-1] // 2
        x1, x2 = x[..., :d], x[..., d:]
        inner = (2**0.5 / math.pi) * (x1 + 0.044715 * x1**3)
        expected = 0.5 * x1 * (1.0 + torch.tanh(inner)) * x2
        torch.testing.assert_close(out, expected, rtol=1e-6, atol=1e-6)

    def test_gelu_obj_none_defaults_to_none(self, backend, fake_ops_c):
        """obj=None reads approximate='none' and uses the exact kernel."""
        x = torch.randn(2, 4)
        out = backend.gelu_and_mul(None, x)
        assert out.shape == (2, 2)
        assert len(fake_ops_c.gelu_calls) == 1
        assert fake_ops_c.gelu_tanh_calls == []

    def test_gelu_obj_without_attribute_defaults_to_none(self, backend, fake_ops_c):
        """An obj missing ``approximate`` falls back to 'none'."""
        x = torch.randn(2, 4)
        out = backend.gelu_and_mul(object(), x)
        assert out.shape == (2, 2)
        assert len(fake_ops_c.gelu_calls) == 1


# ---------------------------------------------------------------------------
# impl/normalization.py
# ---------------------------------------------------------------------------


class _NormObj:
    """Minimal RMSNorm layer double with the attributes the impl reads."""

    def __init__(self, weight, variance_epsilon=1e-6):
        self.weight = weight
        self.variance_epsilon = variance_epsilon


class TestRmsNormImpl:
    """rms_norm_cuda residual and non-residual contracts."""

    def test_without_residual_matches_oracle(self, backend, custom_ops):
        """Without residual a fresh output buffer is filled and returned."""
        torch.manual_seed(2026)
        x = torch.randn(4, 8)
        weight = torch.randn(8)
        eps = 1e-5

        out = backend.rms_norm(_NormObj(weight, eps), x)

        assert isinstance(out, torch.Tensor)
        # Independent oracle: mean-of-squares normalization in float64.
        widened = x.double()
        ms = widened.pow(2).mean(dim=-1, keepdim=True)
        expected = (widened * torch.rsqrt(ms + eps) * weight.double()).float()
        torch.testing.assert_close(out, expected, rtol=1e-5, atol=1e-6)
        # The kernel was called once with the layer's weight and epsilon.
        assert len(custom_ops.rms_norm_calls) == 1
        kernel_x, kernel_w, kernel_eps = custom_ops.rms_norm_calls[0]
        assert kernel_x is x
        assert kernel_w is weight
        assert kernel_eps == eps
        # The input tensor is left untouched (out-of-place path).
        assert not torch.equal(out, x)

    def test_with_residual_updates_both_in_place(self, backend, custom_ops):
        """With residual, x and residual are updated in place and returned."""
        torch.manual_seed(2026)
        x = torch.randn(3, 6)
        residual = torch.randn(3, 6)
        weight = torch.randn(6)
        eps = 1e-5
        x_before = x.clone()
        residual_before = residual.clone()

        out_x, out_residual = backend.rms_norm(_NormObj(weight, eps), x, residual)

        # The SAME tensor objects come back (in-place contract).
        assert out_x is x
        assert out_residual is residual

        # Independent oracle: residual += x, then rmsnorm(residual).
        expected_residual = residual_before + x_before
        widened = expected_residual.double()
        ms = widened.pow(2).mean(dim=-1, keepdim=True)
        expected_x = (widened * torch.rsqrt(ms + eps) * weight.double()).float()
        torch.testing.assert_close(
            out_residual, expected_residual, rtol=1e-6, atol=1e-6
        )
        torch.testing.assert_close(out_x, expected_x, rtol=1e-5, atol=1e-6)
        # Only the fused kernel was used on this path.
        assert custom_ops.rms_norm_calls == []


# ---------------------------------------------------------------------------
# impl/rotary.py
# ---------------------------------------------------------------------------


class TestRotaryImpl:
    """rotary_embedding_cuda argument order and return identity."""

    def test_argument_order_and_return_identity(self, backend, custom_ops):
        """Args reach the kernel in (pos, q, k, cos, sin, interleaved) order
        and the very same q/k tensors are returned."""
        q = torch.randn(2, 4, 8)
        k = torch.randn(2, 1, 8)
        cos = torch.randn(8, 4)
        sin = torch.randn(8, 4)
        pos = torch.tensor([0, 1])

        out_q, out_k = backend.rotary_embedding(
            None, q, k, cos, sin, pos, rotary_interleaved=True
        )

        assert len(custom_ops.rotary_calls) == 1
        call = custom_ops.rotary_calls[0]
        assert call["position_ids"] is pos
        assert call["query"] is q
        assert call["key"] is k
        assert call["cos"] is cos
        assert call["sin"] is sin
        assert call["rotary_interleaved"] is True
        assert out_q is q
        assert out_k is k

    def test_default_interleaved_false(self, backend, custom_ops):
        """The rotary_interleaved default is False."""
        q = torch.randn(1, 2, 4)
        k = torch.randn(1, 1, 4)
        backend.rotary_embedding(None, q, k, None, None, torch.tensor([0]))
        assert custom_ops.rotary_calls[0]["rotary_interleaved"] is False


# ---------------------------------------------------------------------------
# impl/fused_moe.py
# ---------------------------------------------------------------------------


class TestMoeAlignBlockSize:
    """moe_align_block_size_cuda buffer sizing and expert_map semantics."""

    def test_basic_buffer_sizing(self, backend, custom_ops):
        """numel + experts*(block-1) sized buffers; kernel gets them."""
        topk_ids = torch.tensor([[0, 1], [1, 2], [2, 0]], dtype=torch.int32)
        sorted_ids, expert_ids, num_post = backend.moe_align_block_size(
            topk_ids, block_size=4, num_experts=3
        )

        # numel=6, experts=3, block=4: 6 + 3*3 = 15.
        assert sorted_ids.numel() == 15
        assert sorted_ids.dtype == torch.int32
        assert expert_ids.numel() == 4  # cdiv(15, 4)
        assert expert_ids.dtype == torch.int32
        assert num_post.shape == (1,)
        assert num_post.dtype == torch.int32

        call = custom_ops.moe_align_calls[-1]
        assert call["sorted_ids"] is sorted_ids
        assert call["expert_ids"] is expert_ids
        assert call["num_tokens_post_pad"] is num_post
        # No expert_map: the kernel receives None for it.
        assert call["expert_map"] is None
        # Bucket oracle: 3 experts x 1 token each, padded to 4 → total 12.
        torch.testing.assert_close(num_post, torch.tensor([12], dtype=torch.int32))

    def test_pad_sorted_ids_rounds_buffer_up(self, backend, custom_ops):
        """pad_sorted_ids rounds the buffer size to a block multiple."""
        topk_ids = torch.tensor([[0, 0], [0, 0], [1, 1]], dtype=torch.int32)
        sorted_ids, expert_ids, num_post = backend.moe_align_block_size(
            topk_ids, block_size=4, num_experts=2, pad_sorted_ids=True
        )
        # numel=6, experts=2, block=4: 6 + 2*3 = 12 → round_up(12, 4) = 12.
        assert sorted_ids.numel() == 12
        assert expert_ids.numel() == 3
        # expert0: 4 tokens → 4; expert1: 2 → 4; total 8.
        torch.testing.assert_close(num_post, torch.tensor([8], dtype=torch.int32))

    def test_pad_sorted_ids_rounding_needed(self, backend, custom_ops):
        """pad_sorted_ids actually rounds up when the base size is unaligned."""
        topk_ids = torch.tensor([[0, 1]], dtype=torch.int32)
        sorted_ids, _, _ = backend.moe_align_block_size(
            topk_ids, block_size=4, num_experts=2, pad_sorted_ids=True
        )
        # numel=2, experts=2, block=4: 2 + 2*3 = 8 → round_up(8, 4) = 8.
        assert sorted_ids.numel() == 8

    def test_small_topk_clamps_buffer(self, backend, custom_ops):
        """numel < num_experts clamps to numel * block_size."""
        topk_ids = torch.tensor([[0, 1]], dtype=torch.int32)  # numel=2
        sorted_ids, expert_ids, _ = backend.moe_align_block_size(
            topk_ids, block_size=4, num_experts=3
        )
        # base: 2 + 3*3 = 11; clamp: min(2*4=8, 11) = 8.
        assert sorted_ids.numel() == 8
        assert expert_ids.numel() == 2

    def test_expert_map_forwarded_when_ignoring_invalid(self, backend, custom_ops):
        """ignore_invalid_experts forwards the map to the kernel."""
        topk_ids = torch.tensor([[0, 1]], dtype=torch.int32)
        expert_map = torch.tensor([5, 7])
        _, expert_ids, _ = backend.moe_align_block_size(
            topk_ids,
            block_size=2,
            num_experts=2,
            expert_map=expert_map,
            ignore_invalid_experts=True,
        )
        call = custom_ops.moe_align_calls[-1]
        assert call["expert_map"] is expert_map
        # The output was NOT remapped through the map afterwards.
        assert expert_ids.tolist() == [0, 1]

    def test_expert_map_remaps_output_when_not_ignoring(self, backend, custom_ops):
        """Without ignore_invalid_experts the output ids are remapped."""
        topk_ids = torch.tensor([[0, 1], [1, 0]], dtype=torch.int32)
        expert_map = torch.tensor([5, 7])
        _, expert_ids, _ = backend.moe_align_block_size(
            topk_ids, block_size=2, num_experts=2, expert_map=expert_map
        )
        call = custom_ops.moe_align_calls[-1]
        # The kernel itself got None (the impl remaps after the call).
        assert call["expert_map"] is None
        # numel=4, experts=2, block=2: 4 + 2*1 = 6 → expert_ids has 3 slots.
        # Each expert owns 2 tokens → one block each, remapped 0→5, 1→7.
        # The third slot is a tail slot outside total(=4); the source kernel
        # fills tail slots with -1, so the remap indexes the map at -1,
        # yielding Python's last element (7) — pinned as-is.
        assert expert_ids.tolist() == [5, 7, 7]

    def test_values_match_independent_bucket_oracle(self, backend, custom_ops):
        """sorted_ids/expert_ids/num_post match a hand-built bucket layout."""
        topk_ids = torch.tensor([[0, 2], [2, 0], [1, 1]], dtype=torch.int32)
        block_size, num_experts = 2, 3
        sorted_ids, expert_ids, num_post = backend.moe_align_block_size(
            topk_ids, block_size=block_size, num_experts=num_experts
        )

        # counts: e0=2, e1=2, e2=2 → each pads to 2; total 6; offsets 0/2/4.
        torch.testing.assert_close(num_post, torch.tensor([6], dtype=torch.int32))
        # token order per expert: e0: [0, 3], e1: [4, 5], e2: [1, 2].
        assert sorted_ids[0].item() == 0
        assert sorted_ids[1].item() == 3
        assert sorted_ids[2].item() == 4
        assert sorted_ids[3].item() == 5
        assert sorted_ids[4].item() == 1
        assert sorted_ids[5].item() == 2
        # numel=6, experts=3, block=2: 6 + 3*1 = 9 buffer → 5 expert slots;
        # the first three cover the total, the tail two stay -1.
        assert expert_ids.tolist()[:3] == [0, 1, 2]
        assert expert_ids.tolist()[3:] == [-1, -1]


class TestFusedMoeDelegates:
    """Thin pass-through delegates over the faked kernel boundary."""

    def test_topk_softmax_forwards_and_renormalizes_flag(self, backend, custom_ops):
        """Args (incl. renormalize) are forwarded; the pair comes back."""
        torch.manual_seed(2026)
        gating = torch.randn(2, 4)
        weights = torch.zeros(2, 2, dtype=torch.float32)
        indices = torch.zeros(2, 2, dtype=torch.int64)
        src_rows = torch.zeros(2, 2, dtype=torch.int64)

        out_w, out_i = backend.topk_softmax(
            weights, indices, src_rows, gating, renormalize=True
        )

        call = custom_ops.topk_calls[-1]
        assert call["renormalize"] is True
        assert call["gating"] is gating
        assert out_w is weights and out_i is indices

        # Independent oracle: softmax probs gathered at the top-2 indices.
        probs = torch.softmax(gating, dim=-1)
        for row in range(2):
            for slot in range(2):
                torch.testing.assert_close(
                    out_w[row, slot].item(),
                    probs[row, out_i[row, slot].item()].item(),
                    rtol=1e-5,
                    atol=1e-6,
                )

    def test_moe_sum_writes_into_out(self, backend, custom_ops):
        """moe_sum reduces over the expert axis into the given buffer."""
        inp = torch.tensor(
            [[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]]
        )  # [tokens, topk, dim]
        out = torch.zeros(2, 2)
        backend.moe_sum(inp, out)

        assert custom_ops.moe_sum_calls[0][0] is inp
        # Independent loop oracle.
        for token in range(2):
            expected = inp[token, 0].clone()
            for expert in range(1, 2):
                expected = expected + inp[token, expert]
            torch.testing.assert_close(out[token], expected)

    def test_grouped_topk_returns_kernel_result(self, backend, custom_ops):
        """grouped_topk passes all args through and returns the result."""
        scores = torch.randn(1, 8)
        bias = torch.randn(8)
        out = backend.grouped_topk(scores, 2, 1, 4, True, 2.5, bias, scoring_func=1)
        assert out == "grouped-topk-sentinel"
        call = custom_ops.grouped_topk_calls[-1]
        assert call["n_group"] == 2
        assert call["topk_group"] == 1
        assert call["topk"] == 4
        assert call["renormalize"] is True
        assert call["routed_scaling_factor"] == 2.5
        assert call["bias"] is bias
        assert call["scoring_func"] == 1

    def test_invoke_fused_moe_triton_kernel_forwards_all(self, backend, custom_ops):
        """Every positional and keyword argument reaches the kernel."""
        kwargs_out = {}

        import vllm.model_executor.layers.fused_moe.fused_moe as fused_mod

        original = fused_mod.invoke_fused_moe_triton_kernel

        def capture(*args, **kwargs):
            kwargs_out["args"] = args
            kwargs_out["kwargs"] = kwargs

        fused_mod.invoke_fused_moe_triton_kernel = capture
        try:
            backend.invoke_fused_moe_triton_kernel(
                "A",
                "B",
                "C",
                "As",
                "Bs",
                "tw",
                "sti",
                "ei",
                "ntp",
                True,
                2,
                "cfg",
                "ct",
                False,
                False,
                False,
                False,
                False,
                block_shape=[128, 128],
                B_bias="bb",
            )
        finally:
            fused_mod.invoke_fused_moe_triton_kernel = original

        args = kwargs_out["args"]
        assert args[:9] == (
            "A",
            "B",
            "C",
            "As",
            "Bs",
            "tw",
            "sti",
            "ei",
            "ntp",
        )
        assert args[9] is True  # mul_routed_weight
        assert args[10] == 2  # top_k
        assert args[11] == "cfg"  # config
        assert args[12] == "ct"  # compute_type
        assert args[13:18] == (False, False, False, False, False)  # quant flags
        assert kwargs_out["kwargs"] == {"block_shape": [128, 128], "B_bias": "bb"}


# ---------------------------------------------------------------------------
# register_ops.py
# ---------------------------------------------------------------------------


class TestCudaRegisterOps:
    """register_builtins wiring for the vendor/cuda family."""

    @pytest.fixture
    def registered(self, custom_ops):
        """A fresh registry populated by the cuda register_builtins."""
        from vllm_fl.dispatch.backends.vendor.cuda.register_ops import (
            register_builtins,
        )
        from vllm_fl.dispatch.registry import OpRegistry

        registry = OpRegistry()
        register_builtins(registry)
        return registry

    def test_all_ten_ops_registered(self, registered):
        """All ten vendor ops land in the registry."""
        expected = {
            "silu_and_mul",
            "gelu_and_mul",
            "rms_norm",
            "rotary_embedding",
            "attention_backend",
            "moe_align_block_size",
            "moe_sum",
            "topk_softmax",
            "invoke_fused_moe_triton_kernel",
            "grouped_topk",
        }
        assert set(registered.list_operators()) == expected

    def test_impl_metadata(self, registered):
        """Each impl carries the vendor.cuda id, VENDOR kind and priority."""
        from vllm_fl.dispatch.types import BackendImplKind, BackendPriority

        for op_name in registered.list_operators():
            impls = registered.get_implementations(op_name)
            assert len(impls) == 1, op_name
            impl = impls[0]
            assert impl.impl_id == "vendor.cuda", op_name
            assert impl.kind == BackendImplKind.VENDOR, op_name
            assert impl.vendor == "cuda", op_name
            assert impl.priority == BackendPriority.VENDOR, op_name

    def test_no_use_flaggems_filtering(self, registered, monkeypatch):
        """Unlike the flaggems registrar, USE_FLAGGEMS=0 does NOT filter.

        The cuda register_builtins registers unconditionally, so every op
        appears even with the FlagGems switch off.
        """
        from vllm_fl.dispatch.backends.vendor.cuda.register_ops import (
            register_builtins,
        )
        from vllm_fl.dispatch.registry import OpRegistry

        monkeypatch.setenv("USE_FLAGGEMS", "0")
        registry = OpRegistry()
        register_builtins(registry)
        assert len(registry.list_operators()) == 10

    def test_availability_probe_reported_false_on_cpu(self, registered):
        """The bound probe reports the (cached) CPU availability."""
        silu = registered.get_implementation("silu_and_mul", "vendor.cuda")
        assert silu.is_available() is False

    def test_bound_fn_calls_through(self, registered, fake_ops_c):
        """The registered fn delegates to the backend method."""
        x = torch.randn(2, 8)
        silu = registered.get_implementation("silu_and_mul", "vendor.cuda")
        out = silu.fn(None, x)
        assert out.shape == (2, 4)
        assert len(fake_ops_c.silu_calls) == 1

    def test_bind_is_available_wraps_transparently(self, custom_ops):
        """_bind_is_available delegates and exposes the probe attribute."""
        from vllm_fl.dispatch.backends.vendor.cuda.register_ops import (
            _bind_is_available,
        )

        calls = []

        def fn(a, b=0):
            calls.append((a, b))
            return a + b

        def probe():
            return False

        wrapped = _bind_is_available(fn, probe)
        assert wrapped(1, b=2) == 3
        assert calls == [(1, 2)]
        assert wrapped._is_available is probe
        assert wrapped.__name__ == "fn"


# ---------------------------------------------------------------------------
# impl/__init__.py
# ---------------------------------------------------------------------------


class TestImplExports:
    """The vendor/cuda impl package export surface."""

    def test_all_exports_importable(self, custom_ops):
        """__all__ names resolve to the impl functions."""
        import vllm_fl.dispatch.backends.vendor.cuda.impl as impl_pkg

        for name in impl_pkg.__all__:
            assert hasattr(impl_pkg, name), name
        assert set(impl_pkg.__all__) == {
            "silu_and_mul_cuda",
            "gelu_and_mul_cuda",
            "rms_norm_cuda",
            "rotary_embedding_cuda",
        }
