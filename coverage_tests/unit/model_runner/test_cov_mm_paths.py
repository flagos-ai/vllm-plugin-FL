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

"""Coverage tests for ModelRunnerFL._execute_mm_encoder.

The encoder path batches scheduler inputs (real vllm
``group_and_batch_mm_kwargs`` groups them), passes ``prompt_embeds``
items straight into the encoder cache, dispatches the grouped batch to
the model, and caches each output under its mm hash. The model is a
recorder (CONTRACT R2); mm kwargs items are real
``MultiModalKwargsItem`` objects so the grouping semantics exercised
are the production ones. The KV/EC connector mixins are bound from the
real class (no transfer group is active, so their calls are no-ops).
"""

from types import SimpleNamespace

import torch

from . import _imports  # noqa: F401  (installs local import shims first)
from vllm_fl.worker.model_runner import ModelRunnerFL


def _range(**kw):
    """A real PlaceholderRange (lazy import keeps the shim first)."""
    from vllm.multimodal.inputs import PlaceholderRange

    return PlaceholderRange(**kw)


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


def _mm_item(key, value, *, shared=False):
    """A real MultiModalKwargsItem around a plain tensor."""
    from vllm.multimodal.inputs import (
        MultiModalFieldElem,
        MultiModalKwargsItem,
        MultiModalSharedField,
    )

    if shared:
        field = MultiModalSharedField(batch_size=1)
    else:
        from vllm.multimodal.inputs import MultiModalBatchedField

        field = MultiModalBatchedField()
    return MultiModalKwargsItem({key: MultiModalFieldElem(data=value, field=field)})


def _sched(**kw):
    """SchedulerOutput-shaped namespace with encoder inputs."""
    return SimpleNamespace(
        scheduled_encoder_inputs=kw.get("encoder_inputs", {}),
        total_num_scheduled_tokens=kw.get("total", 0),
        num_scheduled_tokens=kw.get("num_scheduled_tokens", {}),
    )


class _Model:
    """Recorder standing in for the multimodal model.

    ``embed_multimodal`` returns exactly as many outputs as the grouped
    batch demands (the real models' contract checked by
    ``sanity_check_mm_encoder_outputs``).
    """

    def __init__(self):
        self.calls = []

    def embed_multimodal(self, **kwargs):
        # The grouped batch stacks per-item tensors: item count is the
        # leading batch dimension of the first field.
        first = next(iter(kwargs.values()))
        num = first.shape[0] if isinstance(first, torch.Tensor) else 1
        self.calls.append(num)
        return [torch.full((2, 4), float(i + 1)) for i in range(num)]


def _runner(model, encoder_cache=None):
    """Namespace runner with the fields _execute_mm_encoder reads."""
    runner = SimpleNamespace(
        requests={},
        input_batch=SimpleNamespace(req_id_to_index={}),
        encoder_cache=encoder_cache if encoder_cache is not None else {},
        model=model,
        device=torch.device("cpu"),
        pin_memory=False,
        lora_config=None,
        lora_manager=SimpleNamespace(
            supports_tower_connector_lora=lambda: False,
        ),
        observability_config=None,
        is_multimodal_pruning_enabled=False,
        requires_sequential_video_encoding=False,
        encoder_cudagraph_manager=None,
    )
    # Bind the real scheduler-input batching helper to this runner.
    runner._batch_mm_inputs_from_scheduler = (
        ModelRunnerFL._batch_mm_inputs_from_scheduler.__get__(runner, ModelRunnerFL)
    )
    # Bind the real EC-connector mixin helpers (no transfer group is
    # active: both are no-ops) and the timing context manager.
    from vllm.v1.worker.ec_connector_model_runner_mixin import (
        ECConnectorModelRunnerMixin,
    )

    # Static methods need no binding: the production code calls
    # self.maybe_save_ec_to_connector(cache, hash) directly.
    runner.maybe_save_ec_to_connector = (
        ECConnectorModelRunnerMixin.maybe_save_ec_to_connector
    )
    runner.timed_encoder_operation = ModelRunnerFL.timed_encoder_operation.__get__(
        runner, ModelRunnerFL
    )
    return runner


def _feature(req_id, item, offset, length, identifier, modality="image"):
    """A MultiModalFeatureSpec-shaped namespace."""
    return SimpleNamespace(
        req_id=req_id,
        data=item,
        modality=modality,
        identifier=identifier,
        mm_position=_range(offset=offset, length=length),
    )


def test_execute_mm_encoder_empty_inputs_return_empty_list():
    """No scheduled encoder inputs: no encoder call and an empty output list."""
    runner = _runner(_Model())
    out = ModelRunnerFL._execute_mm_encoder(runner, _sched())
    assert out == []
    assert runner.model.calls == []


def test_execute_mm_encoder_encodes_and_caches_by_hash():
    """Each scheduled feature is encoded once and cached under its mm hash."""
    item0 = _mm_item("image", torch.zeros(2, 3))
    item1 = _mm_item("image", torch.ones(2, 3))
    model = _Model()
    runner = _runner(model)
    runner.requests["r0"] = SimpleNamespace(
        mm_features=[_feature("r0", item0, 0, 2, "h0")]
    )
    runner.requests["r1"] = SimpleNamespace(
        mm_features=[_feature("r1", item1, 0, 2, "h1")]
    )

    out = ModelRunnerFL._execute_mm_encoder(
        runner,
        _sched(encoder_inputs={"r0": [0], "r1": [0]}, total=4),
    )
    # Two items, each producing one (2, 4) output tensor.
    assert len(out) == 2
    torch.testing.assert_close(out[0], torch.full((2, 4), 1.0))
    torch.testing.assert_close(out[1], torch.full((2, 4), 2.0))
    # The encoder ran once per grouped batch.
    assert len(model.calls) >= 1
    # Outputs are cached under their mm hashes.
    assert set(runner.encoder_cache) == {"h0", "h1"}
    torch.testing.assert_close(runner.encoder_cache["h0"], out[0])


def test_execute_mm_encoder_prompt_embeds_passthrough_skips_encoder():
    """prompt_embeds items go straight into the encoder cache."""
    embed = torch.full((2, 4), 7.0)
    item = _mm_item("embedding", embed, shared=True)
    model = _Model()
    runner = _runner(model)
    runner.requests["r0"] = SimpleNamespace(
        mm_features=[_feature("r0", item, 0, 2, "pe0", modality="prompt_embeds")]
    )

    out = ModelRunnerFL._execute_mm_encoder(
        runner, _sched(encoder_inputs={"r0": [0]}, total=2)
    )
    # Nothing left to encode after filtering prompt_embeds: empty list
    # and the encoder never ran.
    assert out == []
    assert model.calls == []
    # The raw tensor was placed in the cache for the gather to splice.
    assert runner.encoder_cache["pe0"] is embed


def test_execute_mm_encoder_prompt_embeds_mixed_with_real_modality():
    """prompt_embeds passthrough and image encoding both land in the cache."""
    embed = torch.full((2, 4), 5.0)
    pe_item = _mm_item("embedding", embed, shared=True)
    img_item = _mm_item("image", torch.zeros(2, 3))
    model = _Model()
    runner = _runner(model)
    runner.requests["r0"] = SimpleNamespace(
        mm_features=[
            _feature("r0", pe_item, 0, 2, "pe0", modality="prompt_embeds"),
            _feature("r0", img_item, 2, 2, "im0"),
        ]
    )

    out = ModelRunnerFL._execute_mm_encoder(
        runner, _sched(encoder_inputs={"r0": [0, 1]}, total=4)
    )
    # Only the image item went through the encoder.
    assert len(out) == 1
    assert len(model.calls) == 1
    # Both hashes live in the cache: passthrough + encoded.
    assert set(runner.encoder_cache) == {"pe0", "im0"}
    assert runner.encoder_cache["pe0"] is embed
    torch.testing.assert_close(runner.encoder_cache["im0"], out[0])


def test_execute_mm_encoder_batches_same_modality_items_together():
    """Two same-modality items share a single grouped encoder call."""
    item0 = _mm_item("image", torch.zeros(2, 3))
    item1 = _mm_item("image", torch.ones(2, 3))
    model = _Model()
    runner = _runner(model)
    runner.requests["r0"] = SimpleNamespace(
        mm_features=[
            _feature("r0", item0, 0, 2, "h0"),
            _feature("r0", item1, 2, 2, "h1"),
        ]
    )

    out = ModelRunnerFL._execute_mm_encoder(
        runner, _sched(encoder_inputs={"r0": [0, 1]}, total=4)
    )
    # BatchedField items of the same modality group into one encoder call
    # returning both outputs.
    assert len(model.calls) == 1
    assert len(out) == 2
    torch.testing.assert_close(out[0], torch.full((2, 4), 1.0))
    torch.testing.assert_close(out[1], torch.full((2, 4), 2.0))


def test_execute_mm_encoder_skips_features_without_data():
    """A feature whose data was consumed upstream is skipped entirely."""
    item0 = _mm_item("image", torch.zeros(2, 3))
    model = _Model()
    runner = _runner(model)
    # First feature carries no data (cached processor output): skipped.
    runner.requests["r0"] = SimpleNamespace(
        mm_features=[
            _feature("r0", None, 0, 2, "gone"),
            _feature("r0", item0, 2, 2, "h0"),
        ]
    )
    out = ModelRunnerFL._execute_mm_encoder(
        runner, _sched(encoder_inputs={"r0": [0, 1]}, total=4)
    )
    assert len(out) == 1
    assert set(runner.encoder_cache) == {"h0"}
