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

"""G1b sprint coverage: the load/reload tail of ModelRunnerFL.

Targets ``vllm_fl/worker/model_runner.py`` (814b line) methods:
``load_model`` (happy path, dummy-weight, drafter), ``reload_weights``
(kernel-format copy, checkpoint-format layerwise, disk source, missing
loader, not-loaded warning), ``_get_parameter_for_reload``,
``_setup_eagle3_aux_hidden_state_outputs`` and
``_get_eagle3_aux_layers_from_config``.

The runner is an input namespace; the loaded model is a real nn.Module so
weight copies run on CPU. Collaborators imported by the module under test
(``get_model_loader``, ``DeviceMemoryProfiler``, the io-dump / dispatch
warmup helpers, the offloader and the layerwise-reload helpers) are patched
at the module boundary. Every patch is scoped with ExitStack.
"""

from __future__ import annotations

import contextlib
from types import SimpleNamespace

import torch
import torch.nn as nn

import vllm_fl.worker.model_runner as model_runner_module
from vllm_fl.worker.model_runner import ModelRunnerFL


class _Patch:
    """Deterministic set/restore of a module attribute."""

    def __init__(self, module, name, value):
        self.module, self.name, self.value = module, name, value

    def __enter__(self):
        self.saved = getattr(self.module, self.name)
        setattr(self.module, self.name, self.value)
        return self

    def __exit__(self, *exc):
        setattr(self.module, self.name, self.saved)
        return False


def _stack(*pairs):
    """Enter a sequence of (module, name, value) patches on one ExitStack."""
    stack = contextlib.ExitStack()
    for module, name, value in pairs:
        stack.enter_context(_Patch(module, name, value))
    return stack


class _Model(nn.Module):
    """Real module: named_parameters / get_submodule / copy_ all work."""

    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(2, 2, bias=False)
        self.linear.weight.requires_grad_(False)

    def load_weights(self, weights):
        loaded = set()
        for name, tensor in weights:
            self.get_parameter(name).copy_(tensor)
            loaded.add(name)
        return loaded


class _Loader:
    """Recording stand-in for the vllm model loader."""

    def __init__(self, calls, model, all_weights=None):
        self._calls, self._model = calls, model
        self._all_weights = all_weights

    def load_model(self, *, vllm_config, model_config):
        self._calls.append(("load_model", vllm_config, model_config))
        return self._model

    def get_all_weights(self, model_config, model):
        self._calls.append(("get_all_weights", model_config, model))
        return self._all_weights


class _MemoryProfilerFactory:
    def __init__(self, calls, consumed_memory=1024):
        self._calls = calls
        self.consumed_memory = consumed_memory

    def __call__(self):
        return self

    def __enter__(self):
        self._calls.append("enter")
        return self

    def __exit__(self, *exc):
        self._calls.append("exit")
        return False


class _Offloader:
    def __init__(self, calls):
        self._calls = calls

    def post_init(self):
        self._calls.append("post_init")


def _base_runner(model, **extra):
    from vllm.config import CompilationMode, CUDAGraphMode

    vllm_config = SimpleNamespace(
        compilation_config=SimpleNamespace(
            mode=CompilationMode.NONE,
            cudagraph_mode=CUDAGraphMode.NONE,
        ),
        speculative_config=None,
    )
    calls = SimpleNamespace(
        loader=[], comm=[], hooks=[], warmup=[], io=[], offload=[],
        resets=[],
    )
    runner = SimpleNamespace(
        model_config=SimpleNamespace(
            model="test/model",
            enforce_eager=True,
            quantization=None,
            multimodal_config=None,
            revision="rev0",
        ),
        load_config=SimpleNamespace(load_format="safetensors"),
        vllm_config=vllm_config,
        parallel_config=SimpleNamespace(
            enable_eplb=False,
            enable_elastic_ep=False,
            use_ubatching=False,
        ),
        lora_config=None,
        device=torch.device("cpu"),
        model=model,
        _moe_model=None,
        compilation_config=vllm_config.compilation_config,
        graph_runtime=None,
        use_aux_hidden_state_outputs=False,
        speculative_config=None,
        jit_warmup_registry=SimpleNamespace(
            activate=lambda: contextlib.nullcontext()
        ),
        calls=calls,
    )
    runner._setup_eagle3_aux_hidden_state_outputs = (
        ModelRunnerFL._setup_eagle3_aux_hidden_state_outputs.__get__(
            runner, ModelRunnerFL
        )
    )
    runner.get_model = lambda: runner.model
    runner.reset_lora_state = lambda: calls.resets.append("lora")
    runner.reset_encoder_cache = lambda: calls.resets.append("encoder")
    runner.reset_mm_cache = lambda: calls.resets.append("mm")
    for key, value in extra.items():
        setattr(runner, key, value)
    return runner


def _load_env(runner, loader):
    calls = runner.calls
    return _stack(
        (model_runner_module, "init_io_dump_from_env",
         lambda *a, **k: calls.io.append((a, k))),
        (model_runner_module, "get_model_loader", lambda cfg: loader),
        (model_runner_module, "DeviceMemoryProfiler",
         _MemoryProfilerFactory([])),
        (model_runner_module, "warmup_cached_ops",
         lambda *a, **k: calls.warmup.append((a, k))),
        (model_runner_module, "_prepare_comm_buffer",
         lambda m: calls.comm.append(m)),
        (model_runner_module, "register_io_module_hooks",
         lambda m: calls.hooks.append(m)),
        (model_runner_module, "supports_multimodal_pruning",
         lambda m: False),
        (model_runner_module, "get_offloader",
         lambda: _Offloader(calls.offload)),
    )


def test_load_model_happy_path_stores_model_and_flags():
    model = _Model()
    runner = _base_runner(model)
    loader = _Loader(runner.calls.loader, model)

    with _load_env(runner, loader):
        ModelRunnerFL.load_model(runner)

    assert ("load_model", runner.vllm_config, runner.model_config) in (
        runner.calls.loader
    )
    assert runner.model is model
    assert runner.model_memory_usage == 1024
    assert runner.is_multimodal_pruning_enabled is False
    assert runner.requires_sequential_video_encoding is False
    assert runner.calls.comm == [model]
    assert runner.calls.hooks == [model]
    assert len(runner.calls.warmup) == 1
    assert runner.calls.offload == ["post_init"]
    assert runner.calls.io and runner.calls.io[0][0] == (True,)


def test_load_model_dummy_weights_skips_comm_and_io_hooks():
    model = _Model()
    runner = _base_runner(model)
    loader = _Loader(runner.calls.loader, model)

    with _load_env(runner, loader):
        ModelRunnerFL.load_model(runner, load_dummy_weights=True)

    assert runner.load_config.load_format == "dummy"
    assert runner.calls.comm == []
    assert runner.calls.hooks == []
    assert runner._moe_model is None
    assert runner.calls.offload == ["post_init"]


def test_load_model_drafter_loads_and_gets_comm_buffer():
    model = _Model()
    drafter_model = _Model()
    drafter_calls = []
    runner = _base_runner(
        model,
        drafter=SimpleNamespace(
            model=drafter_model,
            load_model=lambda m: drafter_calls.append(m),
        ),
    )
    loader = _Loader(runner.calls.loader, model)

    with _load_env(runner, loader):
        ModelRunnerFL.load_model(runner)

    assert drafter_calls == [model]
    assert runner.calls.comm == [model, drafter_model]
    assert runner.calls.hooks == [model]


# ---------------------------------------------------------------- reload


def _reload_runner(model, **extra):
    runner = _base_runner(model, **extra)
    return runner


def test_reload_weights_kernel_format_copies_parameters():
    model = _Model()
    runner = _reload_runner(model)

    ModelRunnerFL.reload_weights(
        runner,
        weights_iterator=[("linear.weight", torch.full((2, 2), 7.0))],
        is_checkpoint_format=False,
    )

    assert model.linear.weight.flatten().tolist() == [7.0, 7.0, 7.0, 7.0]
    assert runner.calls.resets == ["lora", "encoder", "mm"]


def test_reload_weights_checkpoint_format_layerwise_helpers():
    model = _Model()
    runner = _reload_runner(model)
    layerwise = []

    with _stack(
        (model_runner_module, "initialize_layerwise_reload",
         lambda m: layerwise.append(("init", m))),
        (model_runner_module, "finalize_layerwise_reload",
         lambda m, mc: layerwise.append(("finalize", m, mc))),
    ):
        ModelRunnerFL.reload_weights(
            runner,
            weights_iterator=[("linear.weight", torch.full((2, 2), 3.0))],
            is_checkpoint_format=True,
        )

    assert layerwise == [
        ("init", model),
        ("finalize", model, runner.model_config),
    ]
    assert model.linear.weight.flatten().tolist() == [3.0, 3.0, 3.0, 3.0]
    assert runner.calls.resets == ["lora", "encoder", "mm"]


def test_reload_weights_kernel_format_from_disk_warns(caplog):
    import logging

    model = _Model()
    runner = _reload_runner(model)
    loader = _Loader(
        runner.calls.loader,
        model,
        all_weights=[("linear.weight", torch.full((2, 2), 5.0))],
    )

    with (
        _Patch(model_runner_module, "get_model_loader", lambda cfg: loader),
        caplog.at_level(
            logging.WARNING, logger="vllm_fl.worker.model_runner"
        ),
    ):
        ModelRunnerFL.reload_weights(runner, is_checkpoint_format=False)

    warned = [r for r in caplog.records if "checkpoint format" in r.getMessage()]
    assert len(warned) == 1
    assert model.linear.weight.flatten().tolist() == [5.0] * 4


def test_reload_weights_missing_disk_support_raises():
    import pytest

    model = _Model()
    runner = _reload_runner(model)

    class _NoDiskLoader:
        pass

    with (
        _Patch(model_runner_module, "get_model_loader",
               lambda cfg: _NoDiskLoader()),
        pytest.raises(NotImplementedError, match="safetensors"),
    ):
        ModelRunnerFL.reload_weights(runner)


def test_reload_weights_from_disk_rewrites_model_path():
    model = _Model()
    runner = _reload_runner(model)
    layerwise = []
    loader = _Loader(
        runner.calls.loader,
        model,
        all_weights=[("linear.weight", torch.full((2, 2), 11.0))],
    )

    with _stack(
        (model_runner_module, "get_model_loader", lambda cfg: loader),
        (model_runner_module, "initialize_layerwise_reload",
         lambda m: layerwise.append("init")),
        (model_runner_module, "finalize_layerwise_reload",
         lambda m, mc: layerwise.append("finalize")),
    ):
        ModelRunnerFL.reload_weights(runner, weights_path="/tmp/w")

    assert runner.model_config.model == "/tmp/w"
    assert runner.model_config.revision is None
    assert ("get_all_weights", runner.model_config, model) in (
        runner.calls.loader
    )
    assert layerwise == ["init", "finalize"]
    assert model.linear.weight.flatten().tolist() == [11.0] * 4


def test_reload_weights_warns_on_unloaded_checkpoint_weights(caplog):
    import logging

    model = _Model()
    runner = _reload_runner(model)

    with caplog.at_level(
        logging.WARNING, logger="vllm_fl.worker.model_runner"
    ):
        ModelRunnerFL.reload_weights(
            runner,
            weights_iterator=[],
            is_checkpoint_format=False,
        )

    not_loaded = [r for r in caplog.records if "not loaded" in r.getMessage()]
    assert len(not_loaded) == 1
    assert "linear.weight" in not_loaded[0].getMessage()


def test_reload_weights_quantized_skips_missing_check(caplog):
    import logging

    model = _Model()
    runner = _reload_runner(model)
    runner.model_config.quantization = "fp8"

    with caplog.at_level(
        logging.WARNING, logger="vllm_fl.worker.model_runner"
    ):
        ModelRunnerFL.reload_weights(
            runner,
            weights_iterator=[],
            is_checkpoint_format=False,
        )

    not_loaded = [r for r in caplog.records if "not loaded" in r.getMessage()]
    assert not_loaded == []


def test_get_parameter_for_reload_resolves_nested_parameter():
    import pytest

    helper = getattr(model_runner_module, "_get_parameter_for_reload", None)
    if helper is None:  # pragma: no cover - 814b only
        pytest.skip("_get_parameter_for_reload not present on this vllm line")

    model = _Model()
    param = helper(model, "linear.weight")
    assert param is model.linear.weight
    param.copy_(torch.full((2, 2), 2.5))
    assert model.linear.weight.flatten().tolist() == [2.5] * 4


def test_setup_eagle3_aux_noop_when_disabled():
    runner = SimpleNamespace(use_aux_hidden_state_outputs=False)
    ModelRunnerFL._setup_eagle3_aux_hidden_state_outputs(runner)


def test_get_eagle3_aux_layers_prefers_none_and_dflash():
    assert (
        ModelRunnerFL._get_eagle3_aux_layers_from_config(
            SimpleNamespace(speculative_config=None)
        )
        is None
    )

    hf_config = SimpleNamespace(
        eagle_aux_hidden_state_layer_ids=None,
        dflash_config={"target_layer_ids": [0, 2]},
        eagle_config=None,
    )
    runner = SimpleNamespace(
        speculative_config=SimpleNamespace(
            draft_model_config=SimpleNamespace(hf_config=hf_config)
        )
    )
    assert ModelRunnerFL._get_eagle3_aux_layers_from_config(runner) == (1, 3)
