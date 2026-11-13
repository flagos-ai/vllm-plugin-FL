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

"""CPU tests for the FL RMSNorm wrapper (thin dispatch only)."""

from __future__ import annotations

import torch

from . import _imports  # noqa: F401  (installs CPU-only import placeholders)
from vllm_fl.ops import layernorm as Mod


class _Recorder:
    def __init__(self, result="norm"):
        self.calls = []
        self.result = result

    def __call__(self, *args, **kwargs):
        self.calls.append((args, kwargs))
        return self.result


def _bare(**attrs):
    instance = object.__new__(Mod.RMSNormFL)
    torch.nn.Module.__init__(instance)
    for name, value in attrs.items():
        setattr(instance, name, value)
    return instance


def test_forward_oot_forwards_instance_input_and_residual(monkeypatch):
    recorder = _Recorder("dispatched")
    monkeypatch.setattr(Mod, "_rms_norm", recorder)
    instance = _bare()
    x = torch.zeros(2, 8)
    residual = torch.ones(2, 8)

    assert instance.forward_oot(x, residual) == "dispatched"
    args, kwargs = recorder.calls[0]
    assert args[0] is instance
    assert args[1] is x
    assert args[2] is residual
    assert kwargs == {}


def test_forward_oot_passes_none_residual(monkeypatch):
    recorder = _Recorder("none")
    monkeypatch.setattr(Mod, "_rms_norm", recorder)
    instance = _bare()
    x = torch.zeros(2, 8)

    assert instance.forward_oot(x) == "none"
    assert recorder.calls[0][0][2] is None


def test_constructor_forwards_to_upstream_rms_norm(monkeypatch):
    init_calls = []
    monkeypatch.setattr(
        Mod.RMSNorm,
        "__init__",
        lambda self, *args, **kwargs: init_calls.append((args, kwargs)),
    )
    Mod.RMSNormFL(16, eps=1e-5, var_hidden_size=8, has_weight=False)
    assert init_calls == [((16, 1e-5, 8, False, None), {})]
