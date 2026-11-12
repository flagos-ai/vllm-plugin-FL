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

"""CPU tests for the ARM CPU GDN state-index compatibility installer.

The target builder class and its ``vllm.v1.attention.backends.gdn_attn``
module are the external boundary: the tests install a recording builder class
(fresh per test) and exercise the replacement ``build`` method.  Tensor
branching (CPU/strided/rank) is checked against explicit references; the
non-CPU and zero-rank branches use tensor-shaped fakes because this host has
no CUDA device.
"""

from __future__ import annotations

import sys
import types
from types import SimpleNamespace

import torch

import vllm_fl.patches.arm_cpu_gdn as patch_module


def _builder_class(metadata):
    """Return a fresh builder class plus the calls of its original build."""
    calls = []

    def build(self, *args, **kwargs):
        calls.append((args, kwargs))
        return metadata

    return type("_RecordingBuilder", (), {"build": build}), calls


def _install(monkeypatch, metadata):
    builder_cls, calls = _builder_class(metadata)
    original_build = builder_cls.build
    assert patch_module.apply_arm_cpu_gdn_state_indices_patch(builder_cls) is True
    return builder_cls, calls, original_build


def test_default_install_resolves_vllm_builder_class(monkeypatch):
    """With no argument the installer patches the vllm GDN builder class."""
    builder_cls, _ = _builder_class(SimpleNamespace(non_spec_state_indices_tensor=None))
    fake = types.ModuleType("vllm.v1.attention.backends.gdn_attn")
    fake.GDNAttentionMetadataBuilder = builder_cls
    monkeypatch.setitem(sys.modules, "vllm.v1.attention.backends.gdn_attn", fake)

    assert patch_module.apply_arm_cpu_gdn_state_indices_patch() is True
    assert builder_cls.build._vllm_fl_arm_cpu_gdn_state_indices_patch is True


def test_install_is_idempotent_and_keeps_original(monkeypatch):
    """A second install returns False and leaves the first wrapper in place."""
    metadata = SimpleNamespace(non_spec_state_indices_tensor=None)
    builder_cls, calls, original_build = _install(monkeypatch, metadata)

    first_wrapper = builder_cls.build
    assert patch_module.apply_arm_cpu_gdn_state_indices_patch(builder_cls) is False
    assert builder_cls.build is first_wrapper
    assert first_wrapper._vllm_fl_original is original_build
    # @wraps preserved the original callable's identity metadata.
    assert first_wrapper.__name__ == "build"
    assert first_wrapper.__wrapped__ is original_build

    instance = builder_cls()
    assert instance.build("arg", key="value") is metadata
    assert calls == [(("arg",), {"key": "value"})]


def test_wrapper_densifies_strided_cpu_indices(monkeypatch):
    """A strided CPU index tensor is replaced by a contiguous clone."""
    base = torch.arange(12, dtype=torch.int64).reshape(3, 4)
    strided = base[:, ::2]
    assert strided.stride(-1) == 2
    assert not strided.is_contiguous()

    metadata = SimpleNamespace(non_spec_state_indices_tensor=strided)
    builder_cls, _, _ = _install(monkeypatch, metadata)

    result = builder_cls().build()
    densified = result.non_spec_state_indices_tensor
    assert result is metadata
    assert densified is not strided
    assert densified.is_contiguous()
    assert densified.stride(-1) == 1
    assert torch.equal(densified, strided)
    # The builder-owned source tensor is not mutated in place.
    assert strided.stride(-1) == 2


def test_wrapper_leaves_contiguous_cpu_indices_untouched(monkeypatch):
    """A tensor that already has last-dim stride one is returned unchanged."""
    contiguous = torch.arange(6, dtype=torch.int64).view(2, 3)
    metadata = SimpleNamespace(non_spec_state_indices_tensor=contiguous)
    builder_cls, _, _ = _install(monkeypatch, metadata)

    result = builder_cls().build()
    assert result.non_spec_state_indices_tensor is contiguous


def test_wrapper_ignores_missing_indices(monkeypatch):
    """A None index tensor skips the device/stride probe entirely."""
    metadata = SimpleNamespace(non_spec_state_indices_tensor=None)
    builder_cls, _, _ = _install(monkeypatch, metadata)

    assert builder_cls().build() is metadata
    assert metadata.non_spec_state_indices_tensor is None


def test_wrapper_ignores_non_cpu_indices(monkeypatch):
    """A tensor reporting a non-CPU device is never cloned."""
    cuda_like = SimpleNamespace(
        device=SimpleNamespace(type="cuda"),
        ndim=2,
        stride=lambda dim: 7,
    )
    metadata = SimpleNamespace(non_spec_state_indices_tensor=cuda_like)
    builder_cls, _, _ = _install(monkeypatch, metadata)

    assert builder_cls().build() is metadata
    assert metadata.non_spec_state_indices_tensor is cuda_like


def test_wrapper_skips_zero_rank_indices(monkeypatch):
    """A rank-0 CPU tensor has no last dimension to check."""
    scalar_like = SimpleNamespace(
        device=SimpleNamespace(type="cpu"),
        ndim=0,
        stride=lambda dim: 3,
    )
    metadata = SimpleNamespace(non_spec_state_indices_tensor=scalar_like)
    builder_cls, _, _ = _install(monkeypatch, metadata)

    assert builder_cls().build() is metadata
    assert metadata.non_spec_state_indices_tensor is scalar_like


def test_install_on_real_class_like_builder(monkeypatch):
    """A builder whose build signature carries arguments forwards them exactly."""
    received = []
    metadata = SimpleNamespace(non_spec_state_indices_tensor=None)

    def build(self, common_attn_metadata, *, fast_build=False):
        received.append((common_attn_metadata, fast_build))
        return metadata

    builder_cls = type("_KeywordBuilder", (), {"build": build})
    assert patch_module.apply_arm_cpu_gdn_state_indices_patch(builder_cls) is True

    common = object()
    result = builder_cls().build(common, fast_build=True)
    assert result is metadata
    assert received == [(common, True)]


def test_install_returns_true_only_once(monkeypatch):
    """The returned flag distinguishes the installing call from no-ops."""
    builder_cls, _ = _builder_class(SimpleNamespace(non_spec_state_indices_tensor=None))
    assert patch_module.apply_arm_cpu_gdn_state_indices_patch(builder_cls) is True
    assert patch_module.apply_arm_cpu_gdn_state_indices_patch(builder_cls) is False
    assert patch_module.apply_arm_cpu_gdn_state_indices_patch(builder_cls) is False


