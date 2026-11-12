# Copyright 2026 FlagOS Contributors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""CPU tests for vllm_fl.attention.utils — MM-encoder and rotary patch wiring.

``patch_mm_encoder_attention`` rebinds methods on vllm's
``mm_encoder_attention`` module. That module's import chain pulls Linux-only
packages (uvloop, xgrammar) in this venv, so the tests install an inert leaf
module under the real vllm package before the patched function imports it —
an absent-optional-package boundary; the patch targets vllm's module, never
the module under test. The kunlunxin SDPA path is validated numerically
against an independently written chunked-math reference.

C1 tranche of the supplementary coverage suite (flagcicd_cov).
"""

from __future__ import annotations

import types

import pytest
import torch

from . import _flaggems_shim  # noqa: F401  (must run before vllm_fl import)

_MM_PATH = "vllm.model_executor.layers.attention.mm_encoder_attention"
_MM_PARENTS = (
    "vllm.model_executor",
    "vllm.model_executor.layers",
    "vllm.model_executor.layers.attention",
)


class _MMEncoderAttention:
    """Capture of the class the patch rebinds methods on."""

    def forward_cuda(self, *args, **kwargs):
        return "cuda-dispatch"


@pytest.fixture()
def mm_stub(monkeypatch):
    """Install an inert vllm mm_encoder_attention leaf module (uvloop chain).

    The real module's package __init__ pulls Linux-only deps in this venv;
    the leaf stub carries exactly the two names the patch targets. A
    vllm.vllm_flash_attn stub provides the CUDA-extension-free import the
    patched selector needs (both are absent optional packages locally).
    """
    import sys

    import vllm  # noqa: F401  (ensure the real package owns the top namespace)

    fake_mm = types.ModuleType(_MM_PATH)
    fake_mm.MMEncoderAttention = _MMEncoderAttention
    fake_mm.maybe_get_vit_flash_attn_backend = lambda *a, **k: "original"

    fake_fa = types.ModuleType("vllm.vllm_flash_attn")

    def flash_attn_varlen_func(*args, **kwargs):
        return "flash"

    fake_fa.flash_attn_varlen_func = flash_attn_varlen_func

    for parent in _MM_PARENTS:
        if parent not in sys.modules:
            monkeypatch.setitem(sys.modules, parent, types.ModuleType(parent))
    # ``import a.b as m`` binds through getattr(parent, "b") first (bpo-30024),
    # so a sys.modules setitem alone leaves a previously-imported real leaf
    # unreachable-only-by-luck: whenever another test module imports the
    # vllm_fl.worker chain first, the real leaf lands in sys.modules and the
    # parent attribute shadows the stub. Rebinding the parent package attribute
    # keeps the stub the copy this fixture's assertions observe; monkeypatch
    # restores the previous value (or removes the stub) afterwards.
    monkeypatch.setattr(
        sys.modules["vllm.model_executor.layers.attention"],
        "mm_encoder_attention",
        fake_mm,
        raising=False,
    )
    monkeypatch.setitem(sys.modules, _MM_PATH, fake_mm)
    monkeypatch.setitem(sys.modules, "vllm.vllm_flash_attn", fake_fa)
    monkeypatch.setattr(
        fake_fa, "flash_attn_varlen_func", flash_attn_varlen_func, raising=False
    )
    yield fake_mm


def _reference_chunked_sdpa(q, k, v, scale, num_heads, num_kv_heads):
    """Independent per-batch SDPA reference with GQA head expansion."""
    batch, q_len, heads, head_dim = q.shape
    _, kv_len, kv_heads, _ = k.shape
    q4 = q.permute(0, 2, 1, 3)
    k4 = k.permute(0, 2, 1, 3)
    v4 = v.permute(0, 2, 1, 3)
    repeat = num_heads // num_kv_heads
    outputs = []
    for b in range(batch):
        kb = k4[b : b + 1].repeat_interleave(repeat, dim=1)
        vb = v4[b : b + 1].repeat_interleave(repeat, dim=1)
        outputs.append(
            torch.nn.functional.scaled_dot_product_attention(
                q4[b : b + 1], kb, vb, dropout_p=0.0, scale=scale
            )
        )
    return torch.cat(outputs, dim=0).permute(0, 2, 1, 3)


class _FakeSelf:
    """MMEncoderAttention instance stand-in used by the rebound _forward_sdpa."""

    scale = 0.3
    num_heads = 8
    num_kv_heads = 2

    def view_qkv_to_4d(self, q, k, v, bsz, q_len, kv_len):
        return q, k, v


def test_patch_installs_vit_backend_selector_without_cuda(mm_stub, monkeypatch):
    """Off-CUDA, non-kunlunxin platforms only get the vit backend selector."""
    from vllm.platforms import current_platform

    # Pin the platform probe: the A100 gate runs with is_cuda() True, the
    # authoring host False. Patching current_platform is a sanctioned
    # boundary, so both hosts take the identical off-CUDA branch here.
    monkeypatch.setattr(current_platform, "is_cuda", lambda: False)
    import vllm_fl.attention.utils as au

    au.patch_mm_encoder_attention()
    # CUDA-only forward_oot binding did not fire (is_cuda pinned False).
    assert "forward_oot" not in vars(mm_stub.MMEncoderAttention)
    # The vit backend selector was replaced wholesale.
    assert mm_stub.maybe_get_vit_flash_attn_backend("anything") is None


def test_patch_binds_cuda_dispatch_for_nvidia_platforms(mm_stub, monkeypatch):
    """On CUDA platforms (non-kunlunxin) forward_oot binds the CUDA dispatch."""
    from vllm.platforms import current_platform

    monkeypatch.setattr(current_platform, "is_cuda", lambda: True)
    import vllm_fl.attention.utils as au

    au.patch_mm_encoder_attention()
    bound = vars(mm_stub.MMEncoderAttention).get("forward_oot")
    assert bound == mm_stub.MMEncoderAttention.forward_cuda


def test_patch_installs_kunlunxin_sdpa_and_vendor_oot(mm_stub, monkeypatch):
    """The kunlunxin branch installs the chunked SDPA and vendor forward_oot."""
    from vllm.platforms import current_platform

    monkeypatch.setattr(current_platform, "vendor_name", "kunlunxin", raising=False)
    import vllm_fl.attention.utils as au

    au.patch_mm_encoder_attention()
    forward_sdpa = vars(mm_stub.MMEncoderAttention).get("_forward_sdpa")
    assert forward_sdpa.__name__ == "_kunlunxin_forward_sdpa"
    forward_oot = vars(mm_stub.MMEncoderAttention).get("forward_oot")
    assert forward_oot.__module__ == (
        "vllm_fl.dispatch.backends.vendor.kunlunxin.impl.mm_encoder_attention"
    )


def test_kunlunxin_sdpa_matches_reference_across_chunk_boundary():
    """The installed SDPA matches an independent reference past the 512 chunk.

    The query length (600) exceeds the 512-token chunk boundary, exercising
    the multi-chunk concatenation inside _apply_chunked_sdpa.
    """
    sdpa = _installed_kunlunxin_sdpa()
    torch.manual_seed(7)
    batch, q_len, kv_len, heads, head_dim, kv_heads = 2, 600, 700, 8, 16, 2
    q = torch.randn(batch, q_len, heads, head_dim)
    k = torch.randn(batch, kv_len, kv_heads, head_dim)
    v = torch.randn(batch, kv_len, kv_heads, head_dim)
    out = sdpa(_FakeSelf(), q, k, v)
    assert out.shape == q.shape
    reference = _reference_chunked_sdpa(q, k, v, 0.3, heads, kv_heads)
    torch.testing.assert_close(out, reference, rtol=0, atol=0)


def test_kunlunxin_sdpa_handles_single_chunk_batches():
    """Short sequences below the chunk boundary produce exact SDPA output."""
    sdpa = _installed_kunlunxin_sdpa()
    torch.manual_seed(11)
    q = torch.randn(1, 64, 8, 16)
    k = torch.randn(1, 64, 2, 16)
    v = torch.randn(1, 64, 2, 16)
    out = sdpa(_FakeSelf(), q, k, v)
    reference = _reference_chunked_sdpa(q, k, v, 0.3, 8, 2)
    torch.testing.assert_close(out, reference, rtol=0, atol=0)


def test_kunlunxin_sdpa_splits_varlen_sequences_on_cu_seqlens():
    """cu_seqlens splits q/k/v rows per sequence; 600-row seq crosses the chunk."""
    sdpa = _installed_kunlunxin_sdpa()
    torch.manual_seed(13)
    heads, head_dim, kv_heads = 8, 16, 2
    q = torch.randn(1, 900, heads, head_dim)
    k = torch.randn(1, 900, kv_heads, head_dim)
    v = torch.randn(1, 900, kv_heads, head_dim)
    cu_seqlens = torch.tensor([0, 300, 900])
    out = sdpa(_FakeSelf(), q, k, v, cu_seqlens=cu_seqlens)
    assert out.shape == q.shape

    outputs = []
    for start, end in ((0, 300), (300, 900)):
        q_part = q[:, start:end]
        k_part = k[:, start:end]
        v_part = v[:, start:end]
        outputs.append(
            _reference_chunked_sdpa(q_part, k_part, v_part, 0.3, heads, kv_heads)
        )
    torch.testing.assert_close(out, torch.cat(outputs, dim=1), rtol=0, atol=0)


def test_kunlunxin_sdpa_returns_3d_output_for_3d_input():
    """3D (B, L, H*D-free) inputs keep the (B, L, heads*head_dim) shape contract."""
    sdpa = _installed_kunlunxin_sdpa()
    torch.manual_seed(17)
    q = torch.randn(2, 100, 8, 16)
    k = torch.randn(2, 120, 2, 16)
    v = torch.randn(2, 120, 2, 16)
    out = sdpa(_FakeSelf(), q, k, v)
    assert out.dim() == 4
    assert out.shape == q.shape


def _installed_kunlunxin_sdpa():
    """Patch the kunlunxin branch on a stub module and return the installed SDPA."""
    import sys

    import vllm  # noqa: F401

    fake_mm = types.ModuleType(_MM_PATH)
    fake_mm.MMEncoderAttention = _MMEncoderAttention
    fake_mm.maybe_get_vit_flash_attn_backend = lambda *a, **k: "original"
    saved_modules = {name: sys.modules.get(name) for name in (*_MM_PARENTS, _MM_PATH)}
    try:
        for parent in _MM_PARENTS:
            if parent not in sys.modules:
                sys.modules[parent] = types.ModuleType(parent)
        sys.modules[_MM_PATH] = fake_mm

        from vllm.platforms import current_platform

        had_vendor = "vendor_name" in vars(current_platform) or hasattr(
            current_platform, "vendor_name"
        )
        original_vendor = getattr(current_platform, "vendor_name", None)
        current_platform.vendor_name = "kunlunxin"
        try:
            import vllm_fl.attention.utils as au

            au.patch_mm_encoder_attention()
        finally:
            if had_vendor:
                current_platform.vendor_name = original_vendor
            elif "vendor_name" in vars(current_platform):
                del current_platform.vendor_name
        return vars(fake_mm.MMEncoderAttention)["_forward_sdpa"]
    finally:
        for name, module in saved_modules.items():
            if module is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = module


def test_vit_backend_selector_routes_flash_attn_to_vllm_shim(mm_stub):
    """The patched selector maps FLASH_ATTN to a callable varlen function."""
    import vllm_fl.attention.utils as au

    au.patch_mm_encoder_attention()
    from vllm.v1.attention.backends.registry import AttentionBackendEnum

    selector = mm_stub.maybe_get_vit_flash_attn_backend
    resolved = selector(AttentionBackendEnum.FLASH_ATTN)
    assert callable(resolved)
    assert resolved.__name__ == "flash_attn_varlen_func"
    assert selector(AttentionBackendEnum.FLASH_ATTN) is resolved  # module cached


def test_vit_backend_selector_returns_none_for_unsupported(mm_stub):
    """Backends other than FLASH_ATTN/ROCM_AITER_FA resolve to None."""
    import vllm_fl.attention.utils as au

    au.patch_mm_encoder_attention()
    from vllm.v1.attention.backends.registry import AttentionBackendEnum

    selector = mm_stub.maybe_get_vit_flash_attn_backend
    assert selector(AttentionBackendEnum.TORCH_SDPA) is None
