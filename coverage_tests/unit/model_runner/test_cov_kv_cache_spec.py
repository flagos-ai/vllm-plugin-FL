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

"""Coverage tests for ModelRunnerFL.get_kv_cache_spec.

The method walks the layers registered in the static forward context:
cross-layer KV-sharing layers are recorded and skipped, layers without a
spec are skipped, non-attention specs pass through, and attention specs
get their block-stride index order resolved through the backend. The EC
transfer producer emits no spec, and the Ascend dense hybrid layout pads
the assembled specs.

Layer modules are fake objects handed in as registry contents; the vllm
context helper and EC-transfer probe are module boundaries.
"""

from contextlib import nullcontext  # noqa: I001  (shim import must precede vllm)
from dataclasses import dataclass
from types import SimpleNamespace

import torch

from . import _imports  # noqa: F401  (installs local import shims first)
from ._imports import mr_module
from vllm.model_executor.layers.attention import Attention  # noqa: E402
from vllm.v1.kv_cache_interface import AttentionSpec  # noqa: E402

from vllm_fl.worker.model_runner import ModelRunnerFL  # noqa: E402


class _Patch:
    """Deterministic set/restore of a module attribute (input boundary)."""

    def __init__(self, module, name, value):
        self.module, self.name, self.value = module, name, value

    def __enter__(self):
        self.saved = getattr(self.module, self.name)
        setattr(self.module, self.name, self.value)
        return self

    def __exit__(self, *exc):
        setattr(self.module, self.name, self.saved)
        return False


def _indexable_spec_class():
    """AttentionSpec subclass carrying the stride-order replacement field.

    Frozen-ness is inherited from the installed vllm so the class is a
    valid dataclass on either vllm line.
    """
    frozen = AttentionSpec.__dataclass_params__.frozen

    @dataclass(frozen=frozen)
    class _IndexableSpec(AttentionSpec):
        indexes_kv_by_block_stride: tuple = ()

    return _IndexableSpec


_IndexableSpec = _indexable_spec_class()


def _runner():
    return SimpleNamespace(
        vllm_config=object(),
        shared_kv_cache_layers={},
        use_ascend_hybrid_cache=False,
        cache_config=SimpleNamespace(mamba_page_size_padded=None),
    )


def _attention_layer(*, target=None, spec=None, indexes=("k", "v")):
    """A real Attention instance with the read attributes filled in."""
    layer = object.__new__(Attention)
    layer.kv_sharing_target_layer_name = target
    layer.get_kv_cache_spec = lambda cfg: spec
    layer.get_attn_backend = lambda: SimpleNamespace(
        indexes_kv_by_block_stride=lambda: indexes
    )
    return layer


def test_get_kv_cache_spec_ec_producer_has_no_layers():
    """A non-consumer EC-transfer rank reports an empty spec map."""
    runner = _runner()
    with (
        _Patch(mr_module, "has_ec_transfer", lambda: True),
        _Patch(
            mr_module,
            "get_ec_transfer",
            lambda: SimpleNamespace(is_consumer=False),
        ),
    ):
        assert ModelRunnerFL.get_kv_cache_spec(runner) == {}


def test_get_kv_cache_spec_skips_shared_and_spectless_layers():
    """Shared-target and spec-less layers produce no KV cache spec."""
    runner = _runner()
    base_spec = _IndexableSpec(
        block_size=16, num_kv_heads=2, head_size=8, dtype=torch.float16
    )
    non_attn_spec = SimpleNamespace(kind="linear-attention")
    layers = {
        "target": _attention_layer(spec=base_spec),
        "shared": _attention_layer(target="target", spec=base_spec),
        "none": _attention_layer(spec=None),
        "linear": SimpleNamespace(get_kv_cache_spec=lambda cfg: non_attn_spec),
    }
    calls = []

    def _current_config(cfg):
        calls.append(cfg)
        return nullcontext()

    with (
        _Patch(mr_module, "has_ec_transfer", lambda: False),
        _Patch(mr_module, "get_layers_from_vllm_config", lambda cfg, t: layers),
        _Patch(mr_module, "set_current_vllm_config", _current_config),
    ):
        specs = ModelRunnerFL.get_kv_cache_spec(runner)

    # The sharing layer is recorded as an alias, not given its own spec.
    assert runner.shared_kv_cache_layers == {"shared": "target"}
    assert set(specs) == {"target", "linear"}
    assert specs["linear"] is non_attn_spec
    # Attention specs are replaced with the backend's stride-order indices.
    assert specs["target"].indexes_kv_by_block_stride == ("k", "v")
    assert specs["target"] is not base_spec
    # The backend resolution ran under the runner's vllm config.
    assert calls == [runner.vllm_config]


def test_get_kv_cache_spec_ascend_hybrid_pads_assembled_specs():
    """The Ascend dense hybrid cache pads the final spec map."""
    runner = _runner()
    runner.use_ascend_hybrid_cache = True
    runner.cache_config.mamba_page_size_padded = 512
    spec = _IndexableSpec(
        block_size=16, num_kv_heads=2, head_size=8, dtype=torch.float16
    )
    layers = {"attn": _attention_layer(spec=spec)}
    padded_calls = []
    padded = {"padded": True}

    def _pad(specs, page_size):
        padded_calls.append((specs, page_size))
        return padded

    import vllm_fl.kv_cache.ascend.hybrid as hybrid

    with (
        _Patch(mr_module, "has_ec_transfer", lambda: False),
        _Patch(mr_module, "get_layers_from_vllm_config", lambda cfg, t: layers),
        _Patch(mr_module, "set_current_vllm_config", lambda cfg: nullcontext()),
        _Patch(hybrid, "pad_attention_specs", _pad),
    ):
        specs = ModelRunnerFL.get_kv_cache_spec(runner)

    assert specs is padded
    assert len(padded_calls) == 1
    padded_map, page_size = padded_calls[0]
    assert page_size == 512
    assert set(padded_map) == {"attn"}
    assert padded_map["attn"].indexes_kv_by_block_stride == ("k", "v")
