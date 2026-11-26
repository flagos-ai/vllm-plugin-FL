# Copyright (c) 2026 BAAI. All rights reserved.

"""CPU numerics for the vendor multimodal encoder attention adapters.

``vllm_fl/attention/utils.py`` installs per-vendor attention adapters through
``patch_mm_encoder_attention``:

* **hygon** — replace ``vllm.v1.attention.ops.vit_attn_wrappers.apply_sdpa``
  with a BHSD-contiguous wrapper that pads head dim 72 to 128 (``F.pad`` to
  72+56) around the SDPA call, cropping the output back to 72;
* **kunlunxin** — replace ``MMEncoderAttention._forward_sdpa`` with a
  query-chunked SDPA (chunk 512) that is exact but bounds the score matrix;
* **gcu/enflame** — replace ``forward_native``/``forward_oot`` with a
  varlen flash-attention path;
* plus an NVIDIA ``forward_oot -> forward_cuda`` binding and a
  ``maybe_get_vit_flash_attn_backend`` override.

The core hygon numerics are covered by
``tests/unit_tests/attention/test_hygon_sdpa_patch_contract.py`` (which needs
the vLLM source wheel). These tests load the real
``vllm_fl/attention/utils.py`` under stubbed vllm modules (no vLLM install
needed) and verify the adapter wiring plus the numeric behavior of the
patched callables on CPU with an independent softmax oracle.
"""

import importlib.util
import math
import sys
from contextlib import nullcontext
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest
import torch
import torch.nn.functional as F

_SOURCE = Path(__file__).resolve().parents[3] / "vllm_fl/attention/utils.py"
_MODULE_NAME = "supp_attention_utils_under_test"

_VIT_LEAVES = ("vllm.v1.attention.ops.vit_attn_wrappers",)
_MM_LEAVES = ("vllm.model_executor.layers.attention.mm_encoder_attention",)


def _install_vllm_stubs(monkeypatch, vendor_name, is_cuda=False):
    """Create the stubbed vllm module tree the adapters patch into.

    Returns the three modules whose attributes the production code mutates:
    (vit_wrappers, mm_encoder_attention, platforms).
    """
    modules = {}

    def leaf(name):
        parts = name.split(".")
        for length in range(1, len(parts) + 1):
            partial = ".".join(parts[:length])
            if partial not in modules:
                module = ModuleType(partial)
                module.__path__ = []
                modules[partial] = module
        for length in range(1, len(parts)):
            parent, attribute = ".".join(parts[:length]), parts[length]
            setattr(modules[parent], attribute, modules[".".join(parts[: length + 1])])
        return modules[name]

    for name in _VIT_LEAVES + _MM_LEAVES + ("vllm.platforms", "vllm.logger"):
        leaf(name)

    # --- vllm.logger (the module's only top-level import) ---
    logger_module = modules["vllm.logger"]
    logged = []

    class _Logger:
        def info_once(self, msg, *args):
            logged.append(msg % args if args else msg)

        def info(self, msg, *args):
            logged.append(msg % args if args else msg)

        def warning(self, msg, *args):
            logged.append(msg % args if args else msg)

    logger_module.init_logger = lambda name: _Logger()
    modules.setdefault("logs", SimpleNamespace(logged=logged))

    # --- vllm.v1.attention.ops.vit_attn_wrappers ---
    vit = modules[_VIT_LEAVES[0]]
    vit.einops = SimpleNamespace(rearrange=_rearrange_bshd_bhsd)
    vit.F = SimpleNamespace(
        pad=F.pad, scaled_dot_product_attention=F.scaled_dot_product_attention
    )
    original_calls = []

    def original_apply_sdpa(q, k, v, scale=None, enable_gqa=False):
        original_calls.append((q.shape, scale, enable_gqa))
        return F.scaled_dot_product_attention(
            *(t.transpose(1, 2) for t in (q, k, v)),
            dropout_p=0.0,
            scale=scale,
            enable_gqa=enable_gqa,
        ).transpose(1, 2)

    vit.apply_sdpa = original_apply_sdpa
    vit.original_calls = original_calls

    # --- vllm.model_executor.layers.attention.mm_encoder_attention ---
    mm = modules[_MM_LEAVES[0]]

    class MMEncoderAttention:
        def forward_native(self, query, key, value, cu_seqlens=None, max_seqlen=None):
            raise AssertionError("original forward_native must not run")

        def forward_oot(self, *args, **kwargs):
            raise AssertionError("original forward_oot must not run")

        def forward_cuda(self, *args, **kwargs):
            raise AssertionError("original forward_cuda must not run")

    mm.MMEncoderAttention = MMEncoderAttention

    def original_maybe_get(attn_backend):
        raise AssertionError("original maybe_get must be replaced")

    mm.maybe_get_vit_flash_attn_backend = original_maybe_get

    # --- vllm.platforms with a vendor-switchable current_platform ---
    platforms = modules["vllm.platforms"]
    platform = SimpleNamespace(
        vendor_name=vendor_name,
        is_cuda=(lambda: is_cuda)
        if not callable(getattr(platforms, "is_cuda", None))
        else None,
    )
    platforms.current_platform = platform

    # --- misc imports used by the gcu path ---
    gpu_sync = leaf("vllm.utils.gpu_sync_debug")
    gpu_sync.gpu_sync_allowed = nullcontext
    registry = leaf("vllm.v1.attention.backends.registry")

    class AttentionBackendEnum:
        FLASH_ATTN = "FLASH_ATTN"
        ROCM_AITER_FA = "ROCM_AITER_FA"

    registry.AttentionBackendEnum = AttentionBackendEnum

    for name, module in modules.items():
        monkeypatch.setitem(sys.modules, name, module)

    return SimpleNamespace(
        vit=vit,
        mm=mm,
        platforms=platforms,
        platform=platform,
        registry=registry,
    )


def _rearrange_bshd_bhsd(x, pattern):
    """einops.rearrange stand-in for the two patterns the adapter uses."""
    if pattern.replace(" ", "") == "bshd->bhsd":
        return x.transpose(1, 2)
    if pattern.replace(" ", "") == "bhsd->bshd":
        return x.transpose(1, 2)
    raise AssertionError(f"unexpected rearrange pattern {pattern!r}")


@pytest.fixture
def load_adapter(monkeypatch):
    """Load a fresh copy of vllm_fl/attention/utils.py per test."""

    def load(vendor_name, is_cuda=False):
        stubs = _install_vllm_stubs(monkeypatch, vendor_name, is_cuda)
        spec = importlib.util.spec_from_file_location(_MODULE_NAME, _SOURCE)
        module = importlib.util.module_from_spec(spec)
        monkeypatch.setitem(sys.modules, _MODULE_NAME, module)
        spec.loader.exec_module(module)
        module.patch_mm_encoder_attention()
        return module, stubs

    return load


def _oracle_bshd(q, k, v, scale=None, enable_gqa=False):
    """Independent SDPA oracle in BSHD layout, explicit softmax in float64."""
    if scale is None:
        scale = q.shape[-1] ** -0.5
    qf = q.transpose(1, 2).double()  # B,H,S,D
    kf = k.transpose(1, 2).double()
    vf = v.transpose(1, 2).double()
    if enable_gqa:
        repeats = qf.shape[1] // kf.shape[1]
        kf = kf.repeat_interleave(repeats, dim=1)
        vf = vf.repeat_interleave(repeats, dim=1)
    scores = torch.einsum("bhsd,bhtd->bhst", qf, kf) * scale
    weights = torch.softmax(scores, dim=-1)
    out = torch.einsum("bhst,bhtd->bhsd", weights, vf)
    return out.transpose(1, 2).to(q.dtype)


def test_hygon_patch_replaces_apply_sdpa_with_flagged_wrapper(load_adapter):
    """patch_mm_encoder_attention on hygon rebinds vit apply_sdpa once."""
    module, stubs = load_adapter("hygon")

    patched = stubs.vit.apply_sdpa
    assert patched is not module  # replaced on the stub module
    assert getattr(patched, "_fl_hygon_sdpa_layout", None) == "bhsd-pad-d72"

    # A second patch call must be idempotent (guarded by the layout flag).
    module.patch_mm_encoder_attention()
    assert stubs.vit.apply_sdpa is patched


@pytest.mark.parametrize("head_dim", [72, 64])
@pytest.mark.parametrize("scale", [None, 0.125])
def test_hygon_patched_sdpa_matches_bshd_oracle(load_adapter, head_dim, scale):
    """The hygon wrapper is numerically exact for D72 (padded) and D64.

    For head_dim=72 the adapter pads q/k/v to 128 with F.pad and crops the
    output; for other dims it must behave like plain SDPA. Both are compared
    against an independent explicit-softmax oracle.
    """
    _, stubs = load_adapter("hygon")
    generator = torch.Generator().manual_seed(31)
    b, s, h, kv = 2, 5, 4, 2
    q = torch.randn(b, s, h, head_dim, generator=generator)
    k = torch.randn(b, s, kv, head_dim, generator=generator)
    v = torch.randn(b, s, kv, head_dim, generator=generator)

    output = stubs.vit.apply_sdpa(q, k, v, scale=scale, enable_gqa=True)
    expected = _oracle_bshd(q, k, v, scale=scale, enable_gqa=True)

    assert output.shape == (b, s, h, head_dim)
    assert output.dtype == q.dtype
    torch.testing.assert_close(output, expected, rtol=1e-5, atol=1e-5)


def test_hygon_d72_default_scale_uses_72_rsqrt(load_adapter):
    """With scale=None on D72 inputs, the adapter uses 72**-0.5, not 128**-0.5.

    Padding the head dim changes what a naive SDPA default scale would be;
    the adapter compensates. An oracle with the un-padded default scale
    (head_dim**-0.5 = 72**-0.5) must match.
    """
    _, stubs = load_adapter("hygon")
    generator = torch.Generator().manual_seed(57)
    q = torch.randn(1, 3, 2, 72, generator=generator)
    k = torch.randn(1, 3, 2, 72, generator=generator)
    v = torch.randn(1, 3, 2, 72, generator=generator)

    output = stubs.vit.apply_sdpa(q, k, v, scale=None, enable_gqa=False)

    expected_scale = 72**-0.5
    assert abs(expected_scale - math.sqrt(1.0 / 72.0)) < 1e-15
    expected = _oracle_bshd(q, k, v, scale=expected_scale, enable_gqa=False)
    torch.testing.assert_close(output, expected, rtol=1e-5, atol=1e-5)


def test_hygon_wrapper_inputs_become_contiguous_bhsd(load_adapter):
    """The wrapper forces BHSD contiguity before calling torch SDPA.

    A stubbed scaled_dot_product_attention records the memory format of its
    inputs; non-contiguous BSHD views must arrive contiguous in BHSD.
    """
    _, stubs = load_adapter("hygon")
    seen = []

    def recording_sdpa(q, k, v, dropout_p=0.0, scale=None, enable_gqa=False):
        seen.append(
            (
                tuple(q.shape),
                q.is_contiguous(),
                k.is_contiguous(),
                v.is_contiguous(),
                scale,
            )
        )
        return torch.randn(q.shape[0], q.shape[2], q.shape[1], v.shape[-1])

    stubs.vit.F = SimpleNamespace(
        pad=F.pad, scaled_dot_product_attention=recording_sdpa
    )

    generator = torch.Generator().manual_seed(3)
    q = torch.randn(2, 4, 3, 72, generator=generator).transpose(1, 2)  # BHSD non-contig
    k = torch.randn(2, 4, 2, 72, generator=generator).transpose(1, 2)
    v = torch.randn(2, 4, 2, 72, generator=generator).transpose(1, 2)
    assert not q.is_contiguous()

    stubs.vit.apply_sdpa(q, k, v, scale=None, enable_gqa=True)

    assert len(seen) == 1
    shape, q_contig, k_contig, v_contig, scale = seen[0]
    # BHSD layout with D72 padded to 128: (B=2, H=3, S=4, D=128).
    assert shape == (2, 4, 3, 128)
    assert (q_contig, k_contig, v_contig) == (True, True, True)
    assert scale == 72**-0.5


def test_gcu_patch_binds_forward_native_and_oot(load_adapter):
    """On gcu, MMEncoderAttention.forward_native and forward_oot are replaced."""
    module, stubs = load_adapter("gcu")
    cls = stubs.mm.MMEncoderAttention

    assert cls.forward_native is module._gcu_forward_mm_attention
    assert cls.forward_oot is module._gcu_forward_mm_attention
    assert hasattr(cls, "_fl_gcu_original_forward_native")


def test_gcu_patch_is_idempotent(load_adapter):
    """Applying the gcu patch twice keeps exactly one original forward saved."""
    module, stubs = load_adapter("enflame")
    cls = stubs.mm.MMEncoderAttention
    first_original = cls._fl_gcu_original_forward_native

    module.patch_mm_encoder_attention()

    assert cls._fl_gcu_original_forward_native is first_original
    assert cls.forward_native is module._gcu_forward_mm_attention


def test_gcu_forward_native_delegates_when_backend_not_flash(load_adapter, monkeypatch):
    """With a non-FLASH_ATTN backend the original native forward is restored."""
    module, stubs = load_adapter("gcu")
    cls = stubs.mm.MMEncoderAttention

    calls = []

    def original_forward(self, *args, **kwargs):
        calls.append((args, kwargs))
        return "original-result"

    monkeypatch.setattr(cls, "_fl_gcu_original_forward_native", original_forward)

    attention = object.__new__(cls)
    attention.attn_backend = stubs.registry.AttentionBackendEnum.ROCM_AITER_FA

    result = module._gcu_forward_mm_attention(attention, "q", "k", "v")

    assert result == "original-result"
    # All six forward_native arguments are forwarded positionally (the
    # production delegation passes cu_seqlens as the fourth positional arg,
    # never as a keyword).
    assert calls == [(("q", "k", "v", None, None, None), {})]
    assert len(calls) == 1


def test_kunlunxin_patch_replaces_forward_sdpa(load_adapter):
    """On kunlunxin, MMEncoderAttention._forward_sdpa is the chunked version."""
    module, stubs = load_adapter("kunlunxin")
    cls = stubs.mm.MMEncoderAttention

    assert cls._forward_sdpa is not None
    # The patched symbol name pins the vLLM API the adapter relies on.
    assert hasattr(cls, "_forward_sdpa")
    source = _SOURCE.read_text(encoding="utf-8")
    assert source.count("_forward_sdpa = _kunlunxin_forward_sdpa") == 1


def test_nvidia_binds_forward_oot_to_forward_cuda(load_adapter):
    """On CUDA platforms, forward_oot must alias vLLM's forward_cuda."""
    module, stubs = load_adapter("nvidia", is_cuda=True)
    cls = stubs.mm.MMEncoderAttention

    sentinel = object()
    cls.forward_cuda = sentinel
    module.patch_mm_encoder_attention()
    assert cls.forward_oot is sentinel


def test_patch_replaces_maybe_get_vit_backend_for_hygon_kunlunxin_nvidia(load_adapter):
    """maybe_get_vit_flash_attn_backend is replaced on non-GCU vendor paths.

    The gcu/enflame branch of patch_mm_encoder_attention returns early after
    binding forward_native/forward_oot, so the vit backend selector override
    only applies to hygon, kunlunxin and nvidia (and any other vendor).
    """
    for vendor in ("hygon", "kunlunxin", "nvidia"):
        module, stubs = load_adapter(vendor)
        replaced = stubs.mm.maybe_get_vit_flash_attn_backend
        assert replaced is not None
        assert callable(replaced)
        # The replacement is defined inside the production module's
        # patch_mm_encoder_attention body.
        assert replaced.__name__ == "_patched_maybe_get_vit_flash_attn_backend"


def test_gcu_patch_leaves_maybe_get_backend_unreplaced(load_adapter):
    """The gcu/enflame early return skips the vit backend selector override."""
    _, stubs = load_adapter("gcu")
    selector = stubs.mm.maybe_get_vit_flash_attn_backend
    assert selector.__name__ == "original_maybe_get"


def test_replaced_maybe_get_flash_backend_returns_varlen_func(
    load_adapter, monkeypatch
):
    """The replaced backend selector maps FLASH_ATTN to a varlen callable."""
    module, stubs = load_adapter("hygon")
    enum = stubs.registry.AttentionBackendEnum

    fake_varlen = object()
    fake_pkg = ModuleType("vllm.vllm_flash_attn")
    fake_pkg.flash_attn_varlen_func = fake_varlen
    monkeypatch.setitem(sys.modules, "vllm.vllm_flash_attn", fake_pkg)
    monkeypatch.setitem(sys.modules, "vllm", sys.modules["vllm"])

    selector = stubs.mm.maybe_get_vit_flash_attn_backend
    assert selector(enum.FLASH_ATTN) is fake_varlen
    # The ROCM_AITER_FA branch imports from aiter (absent here): ImportError,
    # matching upstream behaviour, not a silent None.
    with pytest.raises(ImportError):
        selector(enum.ROCM_AITER_FA)
    # Unknown backends map to None.
    assert selector("UNKNOWN") is None
