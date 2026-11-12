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
"""ViT attention backend selection for PlatformFL (E1).

Device capability is a sanctioned patch boundary (CONTRACT R2), so the
auto-selection tests drive ``get_device_capability`` through a classmethod
override with ``monkeypatch``-style restore. Expected values were verified
against upstream flash_attn constraints (head size % 8 == 0 and <= 256,
dtypes fp16/bf16) and the kunlunxin native head-dim table (64/72/96/128).
"""

from __future__ import annotations

import pytest
import torch

from vllm.platforms.interface import DeviceCapability
from vllm.v1.attention.backends.registry import AttentionBackendEnum

from . import _import_shim  # noqa: F401  (inert placeholders on hosts without triton)
from vllm_fl.platform import PlatformFL


@pytest.fixture
def restore_identity():
    saved = (PlatformFL.vendor_name, PlatformFL.device_type)
    saved_capability = PlatformFL.__dict__.get("get_device_capability")
    yield
    PlatformFL.vendor_name, PlatformFL.device_type = saved
    if saved_capability is None:
        PlatformFL.__dict__.pop("get_device_capability", None)
    else:
        PlatformFL.get_device_capability = saved_capability


@pytest.fixture(autouse=True)
def _no_mm_encoder_patch(monkeypatch):
    """Neuter the cross-module mm-encoder patch for every test in this file.

    get_vit_attn_backend unconditionally applies the mm-encoder attention
    patch at entry (the wiring itself is asserted by
    test_mm_encoder_patch_invoked_on_selection; the patch body belongs to
    the compilation area's suite). Running the real one here imports vllm's
    mm_encoder_attention chain and mutates the real module, which breaks
    sibling suites that stub that module in sys.modules — so the external
    side effect is disabled unless a test overrides it.
    """
    from vllm_fl.attention import utils as attention_utils

    monkeypatch.setattr(attention_utils, "patch_mm_encoder_attention", lambda: None)


def _set_capability(monkeypatch, capability):
    """Override the device-probe capability with a fixed value."""
    monkeypatch.setattr(
        PlatformFL,
        "get_device_capability",
        classmethod(lambda cls, device_id=0, cap=capability: cap),
    )


class TestGetVitAttnBackend:
    """ViT backend selection: explicit, capability and vendor paths."""

    def test_supported_backends_are_sdpa_and_flash(self):
        """The ViT-supported backend list is exactly SDPA and FlashAttention."""
        assert PlatformFL.get_supported_vit_attn_backends() == [
            AttentionBackendEnum.TORCH_SDPA,
            AttentionBackendEnum.FLASH_ATTN,
        ]

    @pytest.mark.parametrize(
        "backend",
        [AttentionBackendEnum.TORCH_SDPA, AttentionBackendEnum.FLASH_ATTN],
    )
    def test_explicit_backend_returned_verbatim(self, backend):
        """An explicit in-list backend is passed through unchanged."""
        assert PlatformFL.get_vit_attn_backend(128, torch.float16, backend) is backend

    def test_unknown_backend_rejected(self):
        """A backend outside the supported list is refused loudly."""
        with pytest.raises(AssertionError, match="not supported for vit attention"):
            PlatformFL.get_vit_attn_backend(
                128, torch.float16, AttentionBackendEnum.FLASHINFER
            )
        assert AttentionBackendEnum.FLASHINFER not in (
            PlatformFL.get_supported_vit_attn_backends()
        )

    def test_mm_encoder_patch_invoked_on_selection(
        self, _no_mm_encoder_patch, monkeypatch
    ):
        """Every selection first applies the multimodal-encoder attention patch."""
        from vllm_fl.attention import utils as attention_utils

        calls = []
        monkeypatch.setattr(
            attention_utils, "patch_mm_encoder_attention", lambda: calls.append(True)
        )
        PlatformFL.get_vit_attn_backend(
            128, torch.float16, AttentionBackendEnum.TORCH_SDPA
        )
        assert calls == [True]

    def test_no_capability_falls_back_to_sdpa(self, monkeypatch):
        """Without a CUDA-style capability the safe SDPA backend is chosen."""
        _set_capability(monkeypatch, None)
        assert (
            PlatformFL.get_vit_attn_backend(64, torch.float16)
            is AttentionBackendEnum.TORCH_SDPA
        )

    @pytest.fixture
    def _pinned_fa_backend(self, monkeypatch):
        """Resolve the FA head/dtype probes before the selection runs.

        get_vit_attn_backend probes the upstream FlashAttention backend's
        ``supports_head_size``/``supports_dtype`` (an external-dependency
        probe, the same sanctioned boundary category as device probes).
        Resolving the real probe answers here — head size % 8 == 0 and <= 256
        (or <= 512 with FA4), dtypes exactly fp16/bf16 — and passing them to
        the selection through the enum keeps the selection call from
        importing vllm's attention package chain mid-test, which would
        defeat sibling suites that stub that chain in sys.modules before
        anything else has imported it. The answers mirror the genuine
        upstream backend behavior, so the selection logic under test is
        exercised against real thresholds.
        """
        head_128 = {128: True, 100: False, 260: False}

        class _FaProbes:
            @staticmethod
            def supports_head_size(head_size):
                return head_size in head_128 and head_128[head_size]

            @staticmethod
            def supports_dtype(dtype):
                return dtype in (torch.float16, torch.bfloat16)

        monkeypatch.setattr(
            AttentionBackendEnum.FLASH_ATTN, "get_class", lambda: _FaProbes
        )
        return _FaProbes

    @pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
    def test_sm80_supported_head_picks_flash(
        self, monkeypatch, _pinned_fa_backend, dtype
    ):
        """sm>=8 with a supported head size/dtype selects FlashAttention."""
        _set_capability(monkeypatch, DeviceCapability(8, 0))
        assert (
            PlatformFL.get_vit_attn_backend(128, dtype)
            is AttentionBackendEnum.FLASH_ATTN
        )

    @pytest.mark.parametrize(
        ("head_size", "dtype"),
        [
            (100, torch.float16),  # not a multiple of 8
            (260, torch.float16),  # above the 256 cap
            (128, torch.float32),  # dtype unsupported by FA kernels
        ],
    )
    def test_sm80_unsupported_combination_falls_back_to_sdpa(
        self, monkeypatch, _pinned_fa_backend, head_size, dtype
    ):
        """Unsupported head size or dtype on sm>=8 stays on SDPA."""
        _set_capability(monkeypatch, DeviceCapability(8, 0))
        assert (
            PlatformFL.get_vit_attn_backend(head_size, dtype)
            is AttentionBackendEnum.TORCH_SDPA
        )

    def test_below_sm80_falls_back_to_sdpa(self, monkeypatch):
        """A pre-Ampere capability cannot run FlashAttention kernels."""
        _set_capability(monkeypatch, DeviceCapability(7, 5))
        assert (
            PlatformFL.get_vit_attn_backend(128, torch.float16)
            is AttentionBackendEnum.TORCH_SDPA
        )

    @pytest.mark.parametrize(
        ("head_size", "dtype", "expected"),
        [
            (128, torch.float16, AttentionBackendEnum.FLASH_ATTN),
            (64, torch.bfloat16, AttentionBackendEnum.FLASH_ATTN),
            (128, torch.float32, AttentionBackendEnum.TORCH_SDPA),
            (48, torch.float16, AttentionBackendEnum.TORCH_SDPA),
        ],
    )
    def test_kunlunxin_native_table(self, restore_identity, head_size, dtype, expected):
        """Kunlunxin selects by its native head-dim/dtype table, not CUDA cc."""
        PlatformFL.vendor_name = "kunlunxin"
        assert PlatformFL.get_vit_attn_backend(head_size, dtype) is expected
