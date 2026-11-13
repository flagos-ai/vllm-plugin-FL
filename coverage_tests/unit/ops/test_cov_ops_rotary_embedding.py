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

"""CPU tests for the FL rotary-embedding wrapper.

``_rotary_embedding`` is a recorder; the expected query/key are rebuilt from
plain ``torch.cat`` over the pass-through dimensions.  Instances are created
with ``object.__new__`` because only the attributes read by ``forward_oot``
are needed.
"""

from __future__ import annotations

import sys
import types

import torch

from . import _imports  # noqa: F401  (installs CPU-only import placeholders)
from vllm_fl.ops import rotary_embedding as Mod


class _Recorder:
    def __init__(self, result=None):
        self.calls = []
        self.result = result

    def __call__(self, *args, **kwargs):
        self.calls.append((args, kwargs))
        return self.result


def _bare(**attrs):
    instance = object.__new__(Mod.RotaryEmbeddingFL)
    torch.nn.Module.__init__(instance)
    for name, value in attrs.items():
        setattr(instance, name, value)
    return instance


def _cache(rows, rotary_dim):
    values = torch.arange(rows * rotary_dim, dtype=torch.float32).reshape(
        rows, rotary_dim
    )
    return values / 100.0


def test_forward_oot_splits_rotary_and_pass_dims(monkeypatch):
    recorder = _Recorder()
    monkeypatch.setattr(Mod, "_rotary_embedding", recorder)
    rotary_dim = 4
    head_size = 8
    heads = 2
    tokens = 3
    cache = _cache(8, rotary_dim)
    instance = _bare(
        cos_sin_cache=cache,
        head_size=head_size,
        rotary_dim=rotary_dim,
        is_neox_style=False,
    )

    positions = torch.tensor([0, 2, 4])
    query = torch.arange(tokens * heads * head_size, dtype=torch.float32).reshape(
        tokens, heads, head_size
    )
    key = query + 1000.0
    q_embed = torch.full((tokens, heads, rotary_dim), -1.0)
    k_embed = torch.full((tokens, heads, rotary_dim), -2.0)
    recorder.result = (q_embed, k_embed)

    out_q, out_k = instance.forward_oot(positions, query, key)

    cos, sin = cache.chunk(2, dim=-1)
    args, kwargs = recorder.calls[0]
    assert args[0] is instance
    assert torch.equal(args[1], query[..., :rotary_dim])
    assert torch.equal(args[2], key[..., :rotary_dim])
    assert torch.equal(args[3], cos)
    assert torch.equal(args[4], sin)
    assert torch.equal(args[5], positions)
    assert args[6] is True  # rotary_interleaved = not is_neox_style
    assert args[7] is True  # inplace
    assert kwargs == {}

    expected_q = torch.cat((q_embed, query[..., rotary_dim:]), dim=-1)
    expected_k = torch.cat((k_embed, key[..., rotary_dim:]), dim=-1)
    assert out_q.shape == query.shape
    assert out_k.shape == key.shape
    assert torch.equal(out_q, expected_q)
    assert torch.equal(out_k, expected_k)
    assert instance.cos_sin_cache is cache


def test_forward_oot_without_pass_dims_reshapes_kernel_output(monkeypatch):
    recorder = _Recorder()
    monkeypatch.setattr(Mod, "_rotary_embedding", recorder)
    cache = _cache(4, 4)
    instance = _bare(
        cos_sin_cache=cache,
        head_size=4,
        rotary_dim=4,
        is_neox_style=True,
    )
    query = torch.arange(2 * 3 * 4, dtype=torch.float32).reshape(2, 3, 4)
    key = query * 2.0
    q_embed = torch.full_like(query, 5.0)
    k_embed = torch.full_like(key, 6.0)
    recorder.result = (q_embed, k_embed)
    positions = torch.tensor([0, 1])

    out_q, out_k = instance.forward_oot(positions, query, key)

    assert recorder.calls[0][0][6] is False  # neox style => not interleaved
    assert torch.equal(out_q, q_embed)
    assert torch.equal(out_k, k_embed)


def test_forward_oot_npu_compile_boundary_keeps_buffer(monkeypatch):
    recorder = _Recorder()
    monkeypatch.setattr(Mod, "_rotary_embedding", recorder)
    original_to = torch.Tensor.to

    def clone_to(self, *args, **kwargs):
        return original_to(self, *args, **kwargs).clone()

    monkeypatch.setattr(torch.Tensor, "to", clone_to)
    monkeypatch.setattr(Mod.current_platform, "device_type", "npu")
    monkeypatch.setattr(torch.compiler, "is_compiling", lambda: True)

    cache = _cache(4, 4)
    instance = _bare(
        cos_sin_cache=cache,
        head_size=4,
        rotary_dim=4,
        is_neox_style=True,
    )
    positions = torch.tensor([0, 1])
    query = torch.zeros(2, 2, 4)
    recorder.result = (torch.zeros_like(query), torch.zeros_like(query))

    instance.forward_oot(positions, query, query)

    assert instance.cos_sin_cache is cache


def test_forward_oot_updates_buffer_outside_npu_compile(monkeypatch):
    recorder = _Recorder()
    monkeypatch.setattr(Mod, "_rotary_embedding", recorder)
    original_to = torch.Tensor.to

    def clone_to(self, *args, **kwargs):
        return original_to(self, *args, **kwargs).clone()

    monkeypatch.setattr(torch.Tensor, "to", clone_to)
    monkeypatch.setattr(Mod.current_platform, "device_type", "cuda")

    cache = _cache(4, 4)
    instance = _bare(
        cos_sin_cache=cache,
        head_size=4,
        rotary_dim=4,
        is_neox_style=True,
    )
    positions = torch.tensor([0, 1])
    query = torch.zeros(2, 2, 4)
    recorder.result = (torch.zeros_like(query), torch.zeros_like(query))

    instance.forward_oot(positions, query, query)

    assert instance.cos_sin_cache is not cache
    assert torch.equal(instance.cos_sin_cache, cache)


def test_forward_hip_without_key_uses_base_implementation(monkeypatch):
    base_calls = []
    monkeypatch.setattr(
        Mod.RotaryEmbedding,
        "forward_hip",
        lambda self, positions, query, key=None: (
            base_calls.append((positions, query, key)) or "base-hip"
        ),
        raising=False,
    )
    instance = _bare(
        cos_sin_cache=_cache(4, 4),
        head_size=4,
        rotary_dim=4,
        is_neox_style=True,
    )
    positions = torch.tensor([0, 1])
    query = torch.zeros(2, 2, 4)
    assert instance.forward_hip(positions, query, None) == "base-hip"
    assert base_calls == [(positions, query, None)]


def test_forward_hip_with_key_delegates_to_oot(monkeypatch):
    calls = []
    monkeypatch.setattr(
        Mod.RotaryEmbeddingFL,
        "forward_oot",
        lambda self, positions, query, key=None: (
            calls.append((positions, query, key)) or "oot"
        ),
    )
    instance = _bare(
        cos_sin_cache=_cache(4, 4),
        head_size=4,
        rotary_dim=4,
        is_neox_style=True,
    )
    positions = torch.tensor([0, 1])
    query = torch.zeros(2, 2, 4)
    key = torch.ones(2, 2, 4)
    assert instance.forward_hip(positions, query, key) == "oot"
    assert calls[0][2] is key


def test_init_forwards_geometry_and_initializes_ascend_on_npu(monkeypatch):
    init_calls = []
    ascend_calls = []
    fake_module = types.ModuleType(
        "vllm_fl.dispatch.backends.vendor.ascend.impl.rotary"
    )
    fake_module.initialize_rotary_embedding_ascend = lambda: ascend_calls.append(True)
    monkeypatch.setitem(
        sys.modules,
        "vllm_fl.dispatch.backends.vendor.ascend.impl.rotary",
        fake_module,
    )
    monkeypatch.setattr(Mod.current_platform, "device_type", "npu")
    monkeypatch.setattr(
        Mod.RotaryEmbedding,
        "__init__",
        lambda self, *args, **kwargs: init_calls.append((args, kwargs)),
    )

    Mod.RotaryEmbeddingFL(8, 4, 16, 10000.0, True, torch.float32)

    assert init_calls == [((8, 4, 16, 10000.0, True, torch.float32), {})]
    assert ascend_calls == [True]


def test_init_skips_ascend_initialization_off_npu(monkeypatch):
    init_calls = []
    monkeypatch.setattr(Mod.current_platform, "device_type", "cuda")
    monkeypatch.setattr(
        Mod.RotaryEmbedding,
        "__init__",
        lambda self, *args, **kwargs: init_calls.append((args, kwargs)),
    )

    Mod.RotaryEmbeddingFL(8, 4, 16, 10000.0, True, torch.float32)

    assert len(init_calls) == 1
