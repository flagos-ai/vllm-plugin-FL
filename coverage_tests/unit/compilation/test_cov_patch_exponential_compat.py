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

"""CPU tests for the exponential_ generator-compatibility shim.

``torch.ops.aten.exponential_`` is the external boundary: every test installs
a recorder shaped like the aten kernel, so the shim's generator-aware call and
its generator-less fallback are asserted against the exact keyword arguments
the sampler passes.
"""

from __future__ import annotations

import logging
from types import SimpleNamespace

import torch

import vllm_fl.patches.exponential_compat as patch_module


def _install_fake_aten(monkeypatch, kernel):
    """Replace torch.ops with a namespace exposing one aten kernel."""
    monkeypatch.setattr(
        patch_module.torch,
        "ops",
        SimpleNamespace(aten=SimpleNamespace(exponential_=kernel)),
    )


class _AtenRecorder:
    """Record aten exponential_ calls; optionally reject ``generator``."""

    def __init__(self, reject_generator=False):
        self.calls = []
        self.reject_generator = reject_generator

    def __call__(self, tensor, **kwargs):
        self.calls.append((tensor, dict(kwargs)))
        if self.reject_generator and "generator" in kwargs:
            raise TypeError("exponential_() got an unexpected keyword argument")
        return "ATEN-OUT"


def test_apply_installs_exactly_once(monkeypatch):
    """The first install binds the Python wrapper; later calls are no-ops."""
    monkeypatch.setattr(torch.Tensor, "exponential_", torch.Tensor.exponential_)
    monkeypatch.setattr(patch_module, "_installed", False)

    assert patch_module.apply_exponential_compat() is True
    assert torch.Tensor.exponential_ is patch_module._exponential_compat
    assert patch_module.apply_exponential_compat() is False
    assert torch.Tensor.exponential_ is patch_module._exponential_compat


def test_without_generator_calls_aten_with_default_lambd(monkeypatch):
    """No generator means a plain generator-less aten call with lambd=1.0."""
    recorder = _AtenRecorder()
    _install_fake_aten(monkeypatch, recorder)
    tensor = torch.zeros(3)

    assert patch_module._exponential_compat(tensor) == "ATEN-OUT"
    assert recorder.calls == [(tensor, {"lambd": 1.0})]


def test_with_generator_forwards_lambd_and_generator(monkeypatch):
    """Both the rate and the per-request generator reach the aten kernel."""
    recorder = _AtenRecorder()
    _install_fake_aten(monkeypatch, recorder)
    tensor = torch.zeros(2)
    generator = torch.Generator()

    assert patch_module._exponential_compat(tensor, 2.5, generator=generator) == (
        "ATEN-OUT"
    )
    assert recorder.calls == [(tensor, {"lambd": 2.5, "generator": generator})]


def test_generator_rejection_falls_back_and_warns_once(monkeypatch, caplog):
    """A vendor kernel without ``generator`` falls back and warns one time."""
    recorder = _AtenRecorder(reject_generator=True)
    _install_fake_aten(monkeypatch, recorder)
    monkeypatch.setattr(patch_module, "_warned", False)
    tensor = torch.zeros(2)
    generator = torch.Generator()

    with caplog.at_level(logging.WARNING, logger=patch_module.__name__):
        first = patch_module._exponential_compat(tensor, generator=generator)
        second = patch_module._exponential_compat(tensor, generator=generator)

    assert first == second == "ATEN-OUT"
    # Each call first tried the generator-aware form, then the fallback.
    assert recorder.calls == [
        (tensor, {"lambd": 1.0, "generator": generator}),
        (tensor, {"lambd": 1.0}),
        (tensor, {"lambd": 1.0, "generator": generator}),
        (tensor, {"lambd": 1.0}),
    ]
    assert patch_module._warned is True
    warnings = [
        record
        for record in caplog.records
        if record.name == patch_module.__name__ and record.levelno == logging.WARNING
    ]
    assert len(warnings) == 1
    assert "ignores the generator" in warnings[0].getMessage()


def test_successful_generator_call_does_not_warn(monkeypatch, caplog):
    """An accepting kernel keeps ``_warned`` false."""
    recorder = _AtenRecorder()
    _install_fake_aten(monkeypatch, recorder)
    monkeypatch.setattr(patch_module, "_warned", False)

    with caplog.at_level(logging.WARNING, logger=patch_module.__name__):
        patch_module._exponential_compat(torch.zeros(1), generator=torch.Generator())

    assert patch_module._warned is False
    assert not [
        record
        for record in caplog.records
        if record.name == patch_module.__name__ and record.levelno == logging.WARNING
    ]


def test_generatorless_call_after_install_uses_aten(monkeypatch):
    """The installed method works through normal tensor attribute lookup."""
    recorder = _AtenRecorder()
    _install_fake_aten(monkeypatch, recorder)
    monkeypatch.setattr(torch.Tensor, "exponential_", torch.Tensor.exponential_)
    monkeypatch.setattr(patch_module, "_installed", False)
    assert patch_module.apply_exponential_compat() is True

    tensor = torch.zeros(4)
    assert tensor.exponential_(0.75) == "ATEN-OUT"
    # Attribute lookup supplies ``self`` positionally; lambd stays keyword.
    assert recorder.calls == [(tensor, {"lambd": 0.75})]
