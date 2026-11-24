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

"""Coverage tests for the optional paths of ModelRunnerFL._execute_mm_encoder.

The base batching/caching flow is covered in test_cov_mm_paths.py. This
file drives the remaining branches: per-operation encoder timing stats,
the encoder CUDA-graph manager, the sequential video micro-batch loop,
and tower/connector LoRA mapping construction.

The model and manager are recorders (CONTRACT R2); mm kwargs items are
real MultiModalKwargsItem objects so the grouping semantics are the
production ones.
"""

import threading
from types import SimpleNamespace

import numpy as np
import torch

from . import _imports  # noqa: F401  (installs local import shims first)
from vllm_fl.worker.model_runner import ModelRunnerFL


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


def _range(**kw):
    from vllm.multimodal.inputs import PlaceholderRange

    return PlaceholderRange(**kw)


def _mm_item(key, value, *, shared=False):
    from vllm.multimodal.inputs import (
        MultiModalBatchedField,
        MultiModalFieldElem,
        MultiModalKwargsItem,
        MultiModalSharedField,
    )

    field = MultiModalSharedField(batch_size=1) if shared else MultiModalBatchedField()
    return MultiModalKwargsItem({key: MultiModalFieldElem(data=value, field=field)})


class _Model:
    """Encoder recorder returning one (2, 4) output per batched item."""

    def __init__(self):
        self.calls = []

    def embed_multimodal(self, **kwargs):
        first = next(iter(kwargs.values()))
        num = first.shape[0] if isinstance(first, torch.Tensor) else 1
        self.calls.append(num)
        return [torch.full((2, 4), float(i + 1)) for i in range(num)]


class _LoraModel(_Model):
    """Encoder recorder also publishing the encoder-token count."""

    def get_num_mm_encoder_tokens(self, num_embeds):
        return num_embeds * 2


def _sched(encoder_inputs=None, total=0):
    return SimpleNamespace(
        scheduled_encoder_inputs=encoder_inputs or {},
        total_num_scheduled_tokens=total,
        num_scheduled_tokens={},
    )


def _feature(req_id, item, offset, length, identifier, modality="image"):
    return SimpleNamespace(
        req_id=req_id,
        data=item,
        modality=modality,
        identifier=identifier,
        mm_position=_range(offset=offset, length=length),
    )


def _runner(model, **overrides):
    runner = SimpleNamespace(
        requests={},
        input_batch=SimpleNamespace(
            req_id_to_index={},
            request_lora_mapping=np.zeros(4, dtype=np.int64),
            lora_id_to_lora_request={},
        ),
        encoder_cache={},
        model=model,
        device=torch.device("cpu"),
        pin_memory=False,
        lora_config=None,
        lora_manager=SimpleNamespace(
            supports_tower_connector_lora=lambda: False,
            set_active_adapters=None,
        ),
        observability_config=None,
        is_multimodal_pruning_enabled=False,
        requires_sequential_video_encoding=False,
        encoder_cudagraph_manager=None,
        encoder_timing_registry={},
        _encoder_timing_lock=threading.Lock(),
    )
    runner.__dict__.update(overrides)
    runner._batch_mm_inputs_from_scheduler = (
        ModelRunnerFL._batch_mm_inputs_from_scheduler.__get__(runner, ModelRunnerFL)
    )
    runner.timed_encoder_operation = ModelRunnerFL.timed_encoder_operation.__get__(
        runner, ModelRunnerFL
    )
    from vllm.v1.worker.ec_connector_model_runner_mixin import (
        ECConnectorModelRunnerMixin,
    )

    runner.maybe_save_ec_to_connector = (
        ECConnectorModelRunnerMixin.maybe_save_ec_to_connector
    )
    return runner


def test_execute_mm_encoder_records_timing_per_request():
    """Enabled processor stats time the encoder call per request."""
    item = _mm_item("image", torch.zeros(2, 3))
    model = _Model()
    runner = _runner(
        model,
        observability_config=SimpleNamespace(enable_mm_processor_stats=True),
    )
    runner.requests["r0"] = SimpleNamespace(
        mm_features=[_feature("r0", item, 0, 2, "h0")]
    )

    with _Patch(torch.accelerator, "synchronize", lambda: None):
        out = ModelRunnerFL._execute_mm_encoder(runner, _sched({"r0": [0]}, total=2))

    assert len(out) == 1
    # The timing context manager recorded one call for the request.
    stats = runner.encoder_timing_registry["r0"]
    assert stats.num_encoder_calls == 1
    assert stats.encoder_forward_secs >= 0.0


def test_execute_mm_encoder_uses_cudagraph_manager_when_supported():
    """A modality-supporting encoder graph manager replaces the model call."""
    item = _mm_item("image", torch.zeros(2, 3))
    model = _Model()
    captured = []
    graph_out = [torch.full((2, 4), 9.0)]

    def _execute(batch):
        captured.append(batch)
        return graph_out

    manager = SimpleNamespace(
        supports_modality=lambda modality: modality == "image",
        execute=_execute,
    )
    runner = _runner(model, encoder_cudagraph_manager=manager)
    runner.requests["r0"] = SimpleNamespace(
        mm_features=[_feature("r0", item, 0, 2, "h0")]
    )

    out = ModelRunnerFL._execute_mm_encoder(runner, _sched({"r0": [0]}, total=2))

    assert len(out) == 1
    torch.testing.assert_close(out[0], graph_out[0])
    assert model.calls == []  # the model's encoder never ran
    assert len(captured) == 1
    torch.testing.assert_close(runner.encoder_cache["h0"], graph_out[0])


def test_execute_mm_encoder_encodes_videos_one_at_a_time_when_pruning():
    """Pruning-enabled video groups are split into single-item encoder calls."""
    item0 = _mm_item("pixel_values_videos", torch.zeros(1, 2, 3))
    item1 = _mm_item("pixel_values_videos", torch.ones(1, 2, 3))

    class _VideoModel:
        def __init__(self):
            self.calls = []

        def embed_multimodal(self, **kwargs):
            # One output per micro-batch call, increasing per call.
            self.calls.append(1)
            return [torch.full((2, 4), float(len(self.calls)))]

    model = _VideoModel()
    runner = _runner(model, is_multimodal_pruning_enabled=True)
    runner.requests["r0"] = SimpleNamespace(
        mm_features=[
            _feature("r0", item0, 0, 2, "v0", modality="video"),
            _feature("r0", item1, 2, 2, "v1", modality="video"),
        ]
    )

    out = ModelRunnerFL._execute_mm_encoder(runner, _sched({"r0": [0, 1]}, total=4))

    # One encoder call per video item, each producing one output.
    assert model.calls == [1, 1]
    assert len(out) == 2
    torch.testing.assert_close(out[0], torch.full((2, 4), 1.0))
    torch.testing.assert_close(out[1], torch.full((2, 4), 2.0))


def test_execute_mm_encoder_builds_tower_lora_mapping():
    """Tower LoRA support maps each encoder token to the request's adapter."""
    item = _mm_item("image", torch.zeros(2, 3))
    model = _LoraModel()
    mapping_calls = []
    lora_request = object()

    runner = _runner(model, lora_config=object())
    runner.lora_manager = SimpleNamespace(
        supports_tower_connector_lora=lambda: True,
        set_active_adapters=lambda reqs, mapping: mapping_calls.append((reqs, mapping)),
    )
    runner.input_batch.req_id_to_index = {"r0": 0}
    runner.input_batch.request_lora_mapping = np.array([3, 0, 0, 0], dtype=np.int64)
    runner.input_batch.lora_id_to_lora_request = {3: lora_request}
    runner.requests["r0"] = SimpleNamespace(
        mm_features=[_feature("r0", item, 0, 2, "h0")]
    )

    ModelRunnerFL._execute_mm_encoder(runner, _sched({"r0": [0]}, total=2))

    # 2 embed positions x 2 encoder tokens per embed = 4 tower tokens.
    assert len(mapping_calls) == 1
    reqs, mapping = mapping_calls[0]
    assert reqs == {lora_request}
    assert mapping.index_mapping == (3, 3, 3, 3)
    assert mapping.prompt_mapping == (3,)
    assert mapping.is_prefill is True
