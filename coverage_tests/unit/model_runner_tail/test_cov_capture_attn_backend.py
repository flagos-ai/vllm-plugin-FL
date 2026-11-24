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

"""Coverage tests for ModelRunnerFL.capture_model and
initialize_attn_backend (vllm_fl/worker/model_runner.py).

capture_model's full CUDA-graph capture path (graph_capture context,
mem_get_info, empty_cache, lock_workspace) needs a real accelerator; on
CPU only the cudagraph_mode=NONE early return is reachable, which logs
the skip warning and returns 0 captured bytes.

initialize_attn_backend runs for real: the layers are real
``AttentionLayerBase`` subclasses publishing a concrete test backend, the
KV cache groups are real ``KVCacheGroupSpec`` dataclasses, and
``_check_and_update_cudagraph_mode`` is the real bound method driving the
real builder-class cudagraph-support resolution. The compilation config's
mode resolver and the cudagraph dispatcher are recorders (CONTRACT R2).
"""

from types import SimpleNamespace

import torch

from vllm.config import CUDAGraphMode
from vllm.model_executor.layers.attention_layer_base import AttentionLayerBase
from vllm.v1.attention.backend import (
    AttentionBackend,
    AttentionCGSupport,
    AttentionMetadataBuilder,
)
from vllm.v1.kv_cache_interface import KVCacheGroupSpec

from . import _imports  # noqa: F401  (installs local import shims first)
from vllm_fl.worker.model_runner import ModelRunnerFL


class _AlwaysBuilder(AttentionMetadataBuilder):
    """Real builder class publishing ALWAYS cudagraph support."""

    requires_block_table_width = False

    _cudagraph_support = AttentionCGSupport.ALWAYS

    def __init__(self, *args, **kwargs):
        pass


class _Backend(AttentionBackend):
    """Concrete backend: name / builder resolve for real."""

    @staticmethod
    def get_name():
        return "TEST_ATTN"

    @staticmethod
    def get_impl_cls():
        raise NotImplementedError

    @staticmethod
    def get_builder_cls():
        return _AlwaysBuilder


class _Layer(AttentionLayerBase):
    """Attention layer publishing the test backend."""

    def __init__(self, num_heads=8, backend=_Backend):
        self.num_heads = num_heads
        self._backend = backend

    def get_attn_backend(self):
        return self._backend

    def get_kv_cache_spec(self, vllm_config):
        return None


def test_capture_model_none_mode_warns_and_returns_zero(caplog):
    """cudagraph_mode NONE logs the skip warning and returns 0 captured
    bytes."""
    import logging

    runner = SimpleNamespace(
        compilation_config=SimpleNamespace(cudagraph_mode=CUDAGraphMode.NONE),
    )

    with caplog.at_level(
        logging.WARNING,
        logger="vllm_fl.worker.model_runner",
    ):
        captured = ModelRunnerFL.capture_model(runner)

    assert captured == 0
    skipped = [
        r for r in caplog.records if "Skipping CUDA graph capture" in r.getMessage()
    ]
    assert len(skipped) == 1


def _attn_runner(layers, groups):
    """Runner namespace for initialize_attn_backend."""

    class _ResolveConfig:
        def resolve_cudagraph_mode_and_sizes(self, *args, **kwargs):
            resolve_calls.append((args, kwargs))
            return CUDAGraphMode.NONE

    resolve_calls = []
    key_calls = []
    runner = SimpleNamespace(
        attn_groups=[],
        vllm_config=SimpleNamespace(
            compilation_config=SimpleNamespace(
                static_forward_context=layers,
            ),
            parallel_config=SimpleNamespace(
                prefill_context_parallel_size=1,
                decode_context_parallel_size=1,
                cp_kv_cache_interleave_size=1,
            ),
        ),
        compilation_config=_ResolveConfig(),
        parallel_config=SimpleNamespace(tensor_parallel_size=1),
        kv_sharing_fast_prefill_eligible_layers=set(),
        speculative_config=None,
        cudagraph_dispatcher=SimpleNamespace(
            initialize_cudagraph_keys=(lambda mode, q: key_calls.append((mode, q))),
        ),
        uniform_decode_query_len=3,
        kv_cache_config=SimpleNamespace(kv_cache_groups=groups),
        max_num_reqs=7,
    )
    runner.resolve_calls = resolve_calls
    runner.key_calls = key_calls
    # Real mode-resolution: drives the builder's cudagraph support.
    runner._check_and_update_cudagraph_mode = (
        ModelRunnerFL._check_and_update_cudagraph_mode.__get__(
            runner,
            ModelRunnerFL,
        )
    )
    return runner


def test_initialize_attn_backend_dedupes_within_group():
    """Layers with the same backend, spec and head count share one
    AttentionGroup carrying both layer names."""
    from vllm.v1.kv_cache_interface import FullAttentionSpec

    spec = FullAttentionSpec(
        block_size=16,
        num_kv_heads=2,
        head_size=8,
        dtype=torch.float32,
    )
    group = KVCacheGroupSpec(
        layer_names=["model.layers.0", "model.layers.1"],
        kv_cache_spec=spec,
    )
    config = SimpleNamespace(kv_cache_groups=[group])
    runner = _attn_runner(
        {"model.layers.0": _Layer(), "model.layers.1": _Layer()},
        [group],
    )

    ModelRunnerFL.initialize_attn_backend(runner, config)

    assert len(runner.attn_groups) == 1
    subgroups = runner.attn_groups[0]
    assert len(subgroups) == 1
    attn_group = subgroups[0]
    assert attn_group.backend is _Backend
    assert attn_group.layer_names == ["model.layers.0", "model.layers.1"]
    assert attn_group.kv_cache_spec is spec
    assert attn_group.kv_cache_group_id == 0
    # The resolved cudagraph mode reached the dispatcher with the uniform
    # decode query len.
    assert runner.key_calls == [(CUDAGraphMode.NONE, 3)]


def test_initialize_attn_backend_splits_on_num_heads():
    """Layers with different Q-head counts split into separate attention
    groups (spec-decode draft-vs-target head mismatch)."""
    from vllm.v1.kv_cache_interface import FullAttentionSpec

    spec = FullAttentionSpec(
        block_size=16,
        num_kv_heads=2,
        head_size=8,
        dtype=torch.float32,
    )
    group = KVCacheGroupSpec(
        layer_names=["model.layers.0", "model.layers.2"],
        kv_cache_spec=spec,
    )
    config = SimpleNamespace(kv_cache_groups=[group])
    runner = _attn_runner(
        {
            "model.layers.0": _Layer(num_heads=8),
            "model.layers.2": _Layer(num_heads=4),
        },
        [group],
    )

    ModelRunnerFL.initialize_attn_backend(runner, config)

    subgroups = runner.attn_groups[0]
    assert len(subgroups) == 2
    by_heads = {g.layer_names[0]: g for g in subgroups}
    assert by_heads["model.layers.0"].layer_names == ["model.layers.0"]
    assert by_heads["model.layers.2"].layer_names == ["model.layers.2"]
    # Same spec object shared by both groups.
    assert by_heads["model.layers.0"].kv_cache_spec is spec


def test_initialize_attn_backend_resolves_cudagraph_mode_with_context():
    """The mode resolution sees the weakest backend support, the spec, the
    runner's kv config and the profiling flag."""
    from vllm.v1.kv_cache_interface import FullAttentionSpec

    spec = FullAttentionSpec(
        block_size=16,
        num_kv_heads=2,
        head_size=8,
        dtype=torch.float32,
    )
    group = KVCacheGroupSpec(
        layer_names=["model.layers.0"],
        kv_cache_spec=spec,
    )
    config = SimpleNamespace(kv_cache_groups=[group])
    runner = _attn_runner({"model.layers.0": _Layer()}, [group])

    ModelRunnerFL.initialize_attn_backend(runner, config, is_profiling=True)

    args, kwargs = runner.resolve_calls[0]
    # _AlwaysBuilder publishes ALWAYS support, so min_cg_support is ALWAYS.
    assert args[0] == AttentionCGSupport.ALWAYS
    assert args[1] is None
    assert args[2] == 3  # uniform_decode_query_len
    assert args[3] == 1  # tensor_parallel_size
    assert args[4] is runner.kv_cache_config
    assert args[5] == 7  # max_num_reqs
    assert kwargs == {"is_profiling": True}
