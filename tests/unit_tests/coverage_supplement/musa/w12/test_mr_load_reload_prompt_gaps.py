"""Gap-filling coverage for reload / prompt-logprobs / eagle3 helpers.

Complements the scenarios already covered in ``cov_musa_stage/w3d`` without
repeating them.  Source: model_runner.py @ d12e9d9e.  Everything is mocked; no
weights are read, no device transfer is performed.
"""

from contextlib import nullcontext
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from vllm_fl.worker import model_runner as mr

MR = mr.ModelRunnerFL


def new_runner(**attrs):
    runner = object.__new__(MR)
    for name, value in attrs.items():
        setattr(runner, name, value)
    return runner


def _reload_runner(model, **attrs):
    base = dict(
        load_config=SimpleNamespace(load_format="auto"),
        lora_config=None,
        model_config=SimpleNamespace(
            quantization=None, model="orig", revision="rev"
        ),
    )
    base.update(attrs)
    runner = new_runner(**base)
    runner.get_model = lambda: model
    runner.reset_lora_state = MagicMock()
    runner.reset_encoder_cache = MagicMock()
    runner.reset_mm_cache = MagicMock()
    return runner


def test_reload_from_disk_kernel_format_warns_and_copies(monkeypatch):
    param = SimpleNamespace(copy_=MagicMock())
    model = SimpleNamespace(named_parameters=lambda: [("w", object())])
    loader = SimpleNamespace(
        get_all_weights=MagicMock(return_value=[("w", "value")])
    )
    monkeypatch.setattr(mr, "logger", MagicMock())
    monkeypatch.setattr(mr, "get_model_loader", lambda cfg: loader)
    monkeypatch.setattr(
        mr, "_get_parameter_for_reload", lambda mod, name: param
    )
    runner = _reload_runner(model)

    MR.reload_weights(
        runner, weights_iterator=None, is_checkpoint_format=False
    )

    warning = mr.logger.warning.call_args[0][0]
    assert "checkpoint format" in warning
    assert mr.logger.warning_once.called
    loader.get_all_weights.assert_called_once_with(runner.model_config, model)
    param.copy_.assert_called_once_with("value")
    runner.reset_lora_state.assert_called_once()
    runner.reset_encoder_cache.assert_called_once()
    runner.reset_mm_cache.assert_called_once()


def test_reload_quantized_model_skips_missing_weight_warning(monkeypatch):
    model = SimpleNamespace(
        named_parameters=lambda: [("a", object()), ("b", object())],
        load_weights=MagicMock(return_value={"a"}),
    )
    monkeypatch.setattr(mr, "logger", MagicMock())
    monkeypatch.setattr(mr, "initialize_layerwise_reload", MagicMock())
    monkeypatch.setattr(mr, "finalize_layerwise_reload", MagicMock())
    runner = _reload_runner(
        model,
        model_config=SimpleNamespace(
            quantization="fp8", model="orig", revision="rev"
        ),
    )

    MR.reload_weights(
        runner, weights_iterator=[("a", 1)], is_checkpoint_format=True
    )

    mr.logger.warning.assert_not_called()


def test_reload_reports_unloaded_weights(monkeypatch):
    model = SimpleNamespace(
        named_parameters=lambda: [("a", object()), ("b", object())],
        load_weights=MagicMock(return_value={"a"}),
    )
    monkeypatch.setattr(mr, "logger", MagicMock())
    monkeypatch.setattr(mr, "initialize_layerwise_reload", MagicMock())
    monkeypatch.setattr(mr, "finalize_layerwise_reload", MagicMock())
    runner = _reload_runner(model)

    MR.reload_weights(
        runner, weights_iterator=[("a", 1)], is_checkpoint_format=True
    )

    mr.logger.warning.assert_called_once()
    assert mr.logger.warning.call_args[0][1] == {"b"}


def test_reload_none_loaded_weights_skips_validation(monkeypatch):
    model = SimpleNamespace(
        named_parameters=lambda: [("a", object())],
        load_weights=MagicMock(return_value=None),
    )
    monkeypatch.setattr(mr, "logger", MagicMock())
    monkeypatch.setattr(mr, "initialize_layerwise_reload", MagicMock())
    monkeypatch.setattr(mr, "finalize_layerwise_reload", MagicMock())
    runner = _reload_runner(model)

    MR.reload_weights(
        runner, weights_iterator=[("a", 1)], is_checkpoint_format=True
    )

    mr.logger.warning.assert_not_called()
    runner.reset_encoder_cache.assert_called_once()


class _FakeIndexable:
    def __getitem__(self, item):
        return self


class _RecordingSlot:
    def __init__(self):
        self.copies = []
        self.slices = []

    def __getitem__(self, item):
        self.slices.append(item)
        return self

    def copy_(self, src, non_blocking=False):
        self.copies.append((src, non_blocking))
        return self


class _RecordingTensors:
    def __init__(self):
        self.logprob_token_ids = _RecordingSlot()
        self.logprobs = _RecordingSlot()
        self.selected_token_ranks = _RecordingSlot()


def _prompt_runner(**attrs):
    base = dict(
        num_prompt_logprobs={},
        requests={},
        device="dev",
        input_batch=SimpleNamespace(req_id_to_index={}),
        model_config=SimpleNamespace(logprobs_mode="raw_logits"),
    )
    base.update(attrs)
    return new_runner(**base)


def test_get_prompt_logprobs_chunked_reuses_tensors(monkeypatch):
    tensors = _RecordingTensors()
    request = SimpleNamespace(
        prompt_token_ids=[10, 11, 12, 13, 14, 15],
        in_progress_prompt_logprobs_cpu=tensors,
        num_computed_tokens=2,
    )
    logits = SimpleNamespace(
        to=MagicMock(return_value="scores-frame")
    )
    gather = MagicMock(return_value=("tok", "lp", "rank", None))
    runner = _prompt_runner(
        num_prompt_logprobs={"r": 1},
        requests={"r": request},
        input_batch=SimpleNamespace(req_id_to_index={"r": 0}),
        query_start_loc=SimpleNamespace(
            np=[SimpleNamespace(item=lambda: 1)]
        ),
        model=SimpleNamespace(compute_logits=MagicMock(return_value=logits)),
        model_config=SimpleNamespace(logprobs_mode="processed_logits"),
        sampler=SimpleNamespace(gather_logprobs=gather),
    )
    runner._sync_device = MagicMock()
    fake_ids = _FakeIndexable()
    monkeypatch.setattr(
        mr, "async_tensor_h2d", MagicMock(return_value=fake_ids)
    )
    monkeypatch.setattr(
        mr, "LogprobsTensors", SimpleNamespace(empty_cpu=MagicMock())
    )

    out = MR._get_prompt_logprobs_dict(
        runner, mr.torch.zeros(8, 4), {"r": 2}
    )

    assert out == {}
    assert runner.num_prompt_logprobs == {"r": 1}
    mr.async_tensor_h2d.assert_called_once_with(
        request.prompt_token_ids, device="dev"
    )
    logits.to.assert_called_once_with(mr.torch.float32)
    gather.assert_called_once_with("scores-frame", 1, fake_ids)
    mr.LogprobsTensors.empty_cpu.assert_not_called()
    assert len(tensors.logprob_token_ids.copies) == 1
    assert len(tensors.logprobs.copies) == 1
    assert len(tensors.selected_token_ranks.copies) == 1
    assert all(
        copy[1] is True
        for copy in tensors.logprob_token_ids.copies
    )
    runner._sync_device.assert_not_called()
    assert request.in_progress_prompt_logprobs_cpu is tensors


def test_get_prompt_logprobs_compute_logprobs_branch(monkeypatch):
    created = []

    def empty_cpu(num_rows, num_cols):
        tensors = _RecordingTensors()
        created.append((num_rows, num_cols, tensors))
        return tensors

    request = SimpleNamespace(
        prompt_token_ids=[1, 2, 3, 4],
        in_progress_prompt_logprobs_cpu=None,
        num_computed_tokens=0,
    )
    logits = object()
    sampler = SimpleNamespace(
        compute_logprobs=MagicMock(return_value="computed-scores"),
        gather_logprobs=MagicMock(return_value=("tok", "lp", "rank", None)),
    )
    runner = _prompt_runner(
        num_prompt_logprobs={"r": 2},
        requests={"r": request},
        input_batch=SimpleNamespace(req_id_to_index={"r": 0}),
        query_start_loc=SimpleNamespace(
            np=[SimpleNamespace(item=lambda: 0)]
        ),
        model=SimpleNamespace(compute_logits=MagicMock(return_value=logits)),
        model_config=SimpleNamespace(logprobs_mode="processed_tokens"),
        sampler=sampler,
    )
    runner._sync_device = MagicMock()
    fake_ids = _FakeIndexable()
    monkeypatch.setattr(
        mr, "async_tensor_h2d", MagicMock(return_value=fake_ids)
    )
    monkeypatch.setattr(
        mr, "LogprobsTensors", SimpleNamespace(empty_cpu=empty_cpu)
    )

    out = MR._get_prompt_logprobs_dict(
        runner, mr.torch.zeros(8, 4), {"r": 5}
    )

    assert set(out) == {"r"}
    assert created[0][0] == 3
    assert created[0][1] == 3
    sampler.compute_logprobs.assert_called_once_with(logits)
    sampler.gather_logprobs.assert_called_once_with(
        "computed-scores", 2, fake_ids
    )
    assert runner.num_prompt_logprobs == {}
    assert request.in_progress_prompt_logprobs_cpu is None
    runner._sync_device.assert_called_once()


def test_get_prompt_logprobs_zero_logits_not_computed(monkeypatch):
    tensors = _RecordingTensors()
    request = SimpleNamespace(
        prompt_token_ids=[1, 2, 3],
        in_progress_prompt_logprobs_cpu=None,
        num_computed_tokens=2,
    )
    compute_logits = MagicMock()
    runner = _prompt_runner(
        num_prompt_logprobs={"r": 1},
        requests={"r": request},
        input_batch=SimpleNamespace(req_id_to_index={"r": 0}),
        query_start_loc=SimpleNamespace(
            np=[SimpleNamespace(item=lambda: 0)]
        ),
        model=SimpleNamespace(compute_logits=compute_logits),
        sampler=SimpleNamespace(gather_logprobs=MagicMock()),
    )
    runner._sync_device = MagicMock()
    fake_ids = _FakeIndexable()
    monkeypatch.setattr(
        mr, "async_tensor_h2d", MagicMock(return_value=fake_ids)
    )
    monkeypatch.setattr(
        mr, "LogprobsTensors", SimpleNamespace(empty_cpu=lambda r, c: tensors)
    )

    out = MR._get_prompt_logprobs_dict(
        runner, mr.torch.zeros(8, 4), {"r": 1}
    )

    assert out == {"r": tensors}
    compute_logits.assert_not_called()
    runner.sampler.gather_logprobs.assert_not_called()
    assert request.in_progress_prompt_logprobs_cpu is None
    runner._sync_device.assert_called_once()


def test_get_nans_in_logits_none_no_raise(monkeypatch):
    monkeypatch.setattr(mr, "gpu_sync_allowed", lambda: nullcontext())
    monkeypatch.setattr(
        mr, "nans_to_dict", lambda counts, mapping: {"r": len(counts)}
    )
    monkeypatch.setattr(mr.envs, "VLLM_RAISE_ON_LOGIT_NANS", False)
    runner = new_runner(input_batch=SimpleNamespace(req_id_to_index={}))

    assert MR._get_nans_in_logits(runner, None) == {"r": 0}


def test_get_nans_in_logits_counts_and_raises(monkeypatch):
    monkeypatch.setattr(mr, "gpu_sync_allowed", lambda: nullcontext())
    monkeypatch.setattr(
        mr,
        "count_nans_per_row",
        lambda logits: SimpleNamespace(tolist=lambda: [1, 0, 3]),
    )
    monkeypatch.setattr(
        mr,
        "nans_to_dict",
        lambda counts, mapping: {"a": counts[0], "b": counts[2]},
    )
    raiser = MagicMock()
    monkeypatch.setattr(mr, "raise_if_nan_logits", raiser)
    monkeypatch.setattr(mr.envs, "VLLM_RAISE_ON_LOGIT_NANS", True)
    runner = new_runner(
        input_batch=SimpleNamespace(req_id_to_index={"a": 0, "b": 2})
    )

    assert MR._get_nans_in_logits(runner, object()) == {"a": 1, "b": 3}
    raiser.assert_called_once_with({"a": 1, "b": 3})


def test_get_nans_in_logits_index_error_returns_empty(monkeypatch):
    monkeypatch.setattr(mr, "gpu_sync_allowed", lambda: nullcontext())

    def boom(logits):
        raise IndexError("bad mapping")

    monkeypatch.setattr(mr, "count_nans_per_row", boom)
    runner = new_runner(input_batch=SimpleNamespace(req_id_to_index={}))

    assert MR._get_nans_in_logits(runner, object()) == {}


def test_setup_eagle3_aux_returns_early_when_disabled(monkeypatch):
    called = []
    monkeypatch.setattr(
        mr, "supports_eagle3", lambda model: called.append(model) or True
    )
    runner = new_runner(use_aux_hidden_state_outputs=False)
    runner.get_model = lambda: "model"

    MR._setup_eagle3_aux_hidden_state_outputs(runner)

    assert called == []


def test_setup_eagle3_aux_requires_model_support(monkeypatch):
    monkeypatch.setattr(mr, "supports_eagle3", lambda model: False)
    runner = new_runner(use_aux_hidden_state_outputs=True)
    runner.get_model = lambda: "model"

    with pytest.raises(RuntimeError, match="does not support EAGLE3"):
        MR._setup_eagle3_aux_hidden_state_outputs(runner)


def test_setup_eagle3_aux_prefers_config_then_default(monkeypatch):
    monkeypatch.setattr(mr, "supports_eagle3", lambda model: True)
    model = SimpleNamespace(
        set_aux_hidden_state_layers=MagicMock(),
        get_eagle3_default_aux_hidden_state_layers=MagicMock(return_value=(7, 8)),
    )
    runner = new_runner(use_aux_hidden_state_outputs=True)
    runner.model = model
    runner.get_model = lambda: model
    runner._get_eagle3_aux_layers_from_config = MagicMock(return_value=(1, 2))

    MR._setup_eagle3_aux_hidden_state_outputs(runner)
    model.set_aux_hidden_state_layers.assert_called_once_with((1, 2))

    runner._get_eagle3_aux_layers_from_config = MagicMock(return_value=None)
    MR._setup_eagle3_aux_hidden_state_outputs(runner)
    model.set_aux_hidden_state_layers.assert_called_with((7, 8))
    model.get_eagle3_default_aux_hidden_state_layers.assert_called_once_with()


def test_eagle3_aux_layers_ignores_non_dict_and_missing():
    runner = new_runner(
        speculative_config=SimpleNamespace(draft_model_config=None)
    )
    assert MR._get_eagle3_aux_layers_from_config(runner) is None

    hf_missing = SimpleNamespace()
    runner = new_runner(
        speculative_config=SimpleNamespace(
            draft_model_config=SimpleNamespace(hf_config=hf_missing)
        )
    )
    assert MR._get_eagle3_aux_layers_from_config(runner) is None

    hf_non_dict = SimpleNamespace(
        eagle_aux_hidden_state_layer_ids=None,
        dflash_config=["not", "a", "dict"],
        eagle_config="also-not-a-dict",
    )
    runner = new_runner(
        speculative_config=SimpleNamespace(
            draft_model_config=SimpleNamespace(hf_config=hf_non_dict)
        )
    )
    assert MR._get_eagle3_aux_layers_from_config(runner) is None

    hf_empty_targets = SimpleNamespace(
        eagle_aux_hidden_state_layer_ids=None,
        dflash_config={"target_layer_ids": []},
    )
    runner = new_runner(
        speculative_config=SimpleNamespace(
            draft_model_config=SimpleNamespace(hf_config=hf_empty_targets)
        )
    )
    assert MR._get_eagle3_aux_layers_from_config(runner) is None

    hf_tuple = SimpleNamespace(eagle_aux_hidden_state_layer_ids=(5, 6))
    runner = new_runner(
        speculative_config=SimpleNamespace(
            draft_model_config=SimpleNamespace(hf_config=hf_tuple)
        )
    )
    assert MR._get_eagle3_aux_layers_from_config(runner) == (5, 6)
