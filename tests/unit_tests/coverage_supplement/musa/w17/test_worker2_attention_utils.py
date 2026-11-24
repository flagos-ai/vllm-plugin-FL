# SPDX-License-Identifier: Apache-2.0
"""Real-import contracts for remaining vllm_fl.attention.utils branches.

The real module is imported (its top level only needs ``vllm.logger``).  Vendor
modules and the vLLM platform/registry lookups are replaced via monkeypatch.
"""

from types import SimpleNamespace


def _ensure_pkg(monkeypatch, name):
    import sys
    import types

    existing = sys.modules.get(name)
    if isinstance(existing, types.ModuleType):
        return existing
    module = types.ModuleType(name)
    module.__path__ = []
    monkeypatch.setitem(sys.modules, name, module)
    if "." in name:
        parent_name, leaf = name.rsplit(".", 1)
        parent = _ensure_pkg(monkeypatch, parent_name)
        monkeypatch.setattr(parent, leaf, module, raising=False)
    return module


def _install(monkeypatch, name, **attrs):
    import sys
    import types

    module = types.ModuleType(name)
    for key, value in attrs.items():
        setattr(module, key, value)
    if "." in name:
        parent_name, leaf = name.rsplit(".", 1)
        parent = _ensure_pkg(monkeypatch, parent_name)
        monkeypatch.setattr(parent, leaf, module, raising=False)
    monkeypatch.setitem(sys.modules, name, module)
    return module


def _mm_module():
    class MMEncoderAttention:
        def forward_native(self, *args, **kwargs):
            return "native"

        def forward_cuda(self, *args, **kwargs):
            return "cuda"

        def forward_oot(self, *args, **kwargs):
            return "oot"

    return SimpleNamespace(MMEncoderAttention=MMEncoderAttention)


def _enum():
    return SimpleNamespace(FLASH_ATTN=object(), ROCM_AITER_FA=object())


def test_gcu_patch_keeps_existing_original(monkeypatch):
    import vllm_fl.attention.utils as a

    mm = _mm_module()
    sentinel = object()
    mm.MMEncoderAttention._fl_gcu_original_forward_native = sentinel
    _install(
        monkeypatch,
        "vllm.model_executor.layers.attention.mm_encoder_attention",
        **vars(mm),
    )
    _install(
        monkeypatch,
        "vllm.v1.attention.backends.registry",
        AttentionBackendEnum=_enum(),
    )
    _install(monkeypatch, "vllm.platforms", current_platform=SimpleNamespace(
        vendor_name="gcu", is_cuda=lambda: False
    ))

    a.patch_mm_encoder_attention()
    assert mm.MMEncoderAttention._fl_gcu_original_forward_native is sentinel
    assert mm.MMEncoderAttention.forward_native is a._gcu_forward_mm_attention
    assert mm.MMEncoderAttention.forward_oot is a._gcu_forward_mm_attention


def test_hygon_vendor_triggers_sdpa_patch(monkeypatch):
    import vllm_fl.attention.utils as a

    hygon_calls = []
    monkeypatch.setattr(
        a, "_patch_hygon_sdpa_contiguous", lambda: hygon_calls.append("hygon")
    )
    mm = _mm_module()
    installed = _install(
        monkeypatch,
        "vllm.model_executor.layers.attention.mm_encoder_attention",
        **vars(mm),
    )
    enum = _enum()
    _install(
        monkeypatch,
        "vllm.v1.attention.backends.registry",
        AttentionBackendEnum=enum,
    )
    _install(monkeypatch, "vllm.platforms", current_platform=SimpleNamespace(
        vendor_name="hygon", is_cuda=lambda: False
    ))

    a.patch_mm_encoder_attention()
    assert hygon_calls == ["hygon"]
    assert callable(installed.maybe_get_vit_flash_attn_backend)


def test_non_cuda_platform_leaves_forward_oot(monkeypatch):
    import vllm_fl.attention.utils as a

    mm = _mm_module()
    original = mm.MMEncoderAttention.forward_oot
    _install(
        monkeypatch,
        "vllm.model_executor.layers.attention.mm_encoder_attention",
        **vars(mm),
    )
    _install(
        monkeypatch,
        "vllm.v1.attention.backends.registry",
        AttentionBackendEnum=_enum(),
    )
    _install(monkeypatch, "vllm.platforms", current_platform=SimpleNamespace(
        vendor_name="npu", is_cuda=lambda: False
    ))

    a.patch_mm_encoder_attention()
    assert mm.MMEncoderAttention.forward_oot is original
