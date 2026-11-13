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

"""Coverage tests for ModelRunnerFL.load_model (vllm_fl/worker/
model_runner.py).

load_model orchestrates the model-loading stack: loader invocation,
drafter loading, MoE/EPLB registration (unwrapping VLM wrappers),
post-load flags, the stock-torch-compile path and the cudagraph wrapper
selection. The loader, memory profiler, graph wrappers and EPLB state
are inputs replaced with recorders (CONTRACT R2); the loaded model is a
real ``nn.Module`` subclass. Patch boundaries are imported collaborator
names of the module under test (``get_model_loader``,
``DeviceMemoryProfiler`` — raises NotImplementedError without CUDA —
``GraphWrapper`` — constructor needs CUDA graphs — and ``EplbState``),
never the module's own functions.
"""

from types import SimpleNamespace

from vllm.config import CompilationMode, CUDAGraphMode
from vllm.model_executor.models.interfaces import (
    MixtureOfExperts,
    SupportsMultiModal,
)

from . import _imports  # noqa: F401  (installs local import shims first)
from ._loader_helpers import (
    Loader,
    Model,
    Patch,
    base_runner,
    module_under_test,
    patched_loader_env,
)
from vllm_fl.worker.model_runner import ModelRunnerFL

model_runner_module = module_under_test()


def test_load_model_happy_path_loads_and_finalizes():
    """The default path: loader invoked with the runner configs, model
    stored, memory usage recorded, derived flags computed afterwards."""
    model = Model()
    runner = base_runner(model)
    loader = Loader(runner.calls, model)

    patches = patched_loader_env(runner, loader)
    with patches[0], patches[1]:
        ModelRunnerFL.load_model(runner)

    assert ("load_model", runner.vllm_config, runner.model_config) in (runner.calls)
    assert runner.model is model
    assert runner.model_memory_usage == 1024
    # Multimodal pruning disabled: a plain module supports neither
    # pruning nor multimodal inputs.
    assert runner.is_multimodal_pruning_enabled is False
    assert runner.requires_sequential_video_encoding is False


def test_load_model_dummy_weights_and_drafter():
    """load_dummy_weights flips the load format; a drafter with its own
    model gets the loaded target model."""
    model = Model()
    runner = base_runner(model)
    loader = Loader(runner.calls, model)
    drafter_calls = []
    runner.drafter = SimpleNamespace(
        model=Model(),
        load_model=lambda m: drafter_calls.append(m),
    )

    patches = patched_loader_env(runner, loader)
    with patches[0], patches[1]:
        ModelRunnerFL.load_model(runner, load_dummy_weights=True)

    # Dummy loading rewrote the load format before the loader ran.
    assert runner.load_config.load_format == "dummy"
    assert drafter_calls == [model]


def test_load_model_stock_torch_compile_compiles_model():
    """STOCK_TORCH_COMPILE mode compiles the model with the configured
    backend and returns before any cudagraph wrapper."""
    model = Model()
    compiled = []
    model.compile = lambda **kw: compiled.append(kw)
    runner = base_runner(model)
    runner.vllm_config.compilation_config.mode = CompilationMode.STOCK_TORCH_COMPILE
    init_backend_calls = []
    backend = object()
    runner.vllm_config.compilation_config.init_backend = lambda vc: (
        init_backend_calls.append(vc) or backend
    )
    loader = Loader(runner.calls, model)

    from vllm.compilation.counter import compilation_counter

    before = compilation_counter.stock_torch_compile_count
    patches = patched_loader_env(runner, loader)
    with patches[0], patches[1]:
        ModelRunnerFL.load_model(runner)

    assert init_backend_calls == [runner.vllm_config]
    assert len(compiled) == 1
    assert compiled[0]["fullgraph"] is True
    assert compiled[0]["backend"] is backend
    assert compilation_counter.stock_torch_compile_count == before + 1
    # Returned before the wrapper stage: the raw model is untouched.
    assert runner.model is model


def test_load_model_wraps_in_cudagraph_wrapper_when_full_mode():
    """A full-cudagraph mode wraps the loaded model in GraphWrapper with
    the FULL runtime mode and the runner's graph runtime."""
    model = Model()
    runner = base_runner(model)
    runner.vllm_config.compilation_config.cudagraph_mode = CUDAGraphMode.FULL
    wrapped = []
    loader = Loader(runner.calls, model)

    def _wrap(inner, vllm_config, *, runtime_mode, graph_runtime):
        wrapped.append((inner, vllm_config, runtime_mode, graph_runtime))
        return "wrapped-model"

    patches = patched_loader_env(runner, loader)
    with (
        patches[0],
        patches[1],
        Patch(
            module_under_test(),
            "GraphWrapper",
            _wrap,
        ),
    ):
        ModelRunnerFL.load_model(runner)

    assert wrapped == [
        (
            model,
            runner.vllm_config,
            CUDAGraphMode.FULL,
            None,
        )
    ]
    assert runner.model == "wrapped-model"


def _patch(name, value):
    return Patch(module_under_test(), name, value)


class _EplbState:
    """Recording stand-in for the EPLB state."""

    def __init__(self, parallel_config, device):
        self.model_states = []
        self.is_async = False
        self.added = []

    def add_model(self, moe_model, model_config):
        self.added.append((moe_model, model_config))


def test_load_model_eplb_registers_moe_model():
    """With EPLB enabled and a MoE target model, the model is registered
    with the EPLB state."""

    class _MoEModel(Model, MixtureOfExperts):
        num_moe_layers = 2

    model = _MoEModel()
    runner = base_runner(model)
    runner.parallel_config.enable_eplb = True
    loader = Loader(runner.calls, model)

    state = _EplbState(runner.parallel_config, runner.device)
    patches = patched_loader_env(runner, loader)
    with patches[0], patches[1], _patch("EplbState", lambda pc, d: state):
        ModelRunnerFL.load_model(runner)

    # The MoE model itself was registered with the EPLB state.
    assert state.added == [(model, runner.model_config)]
    assert runner.eplb_state is state
    assert runner._moe_model is model


def test_load_model_vlm_unwraps_language_model_for_moe():
    """A VLM wrapper around a MoE language model registers the inner
    language model."""

    class _MoEModel(Model, MixtureOfExperts):
        num_moe_layers = 2

    moe = _MoEModel()

    class _VLM(Model, SupportsMultiModal):
        def get_language_model(self):
            return moe

    vlm = _VLM()
    runner = base_runner(vlm)
    runner.parallel_config.enable_eplb = True
    loader = Loader(runner.calls, vlm)

    state = _EplbState(runner.parallel_config, runner.device)
    patches = patched_loader_env(runner, loader)
    with patches[0], patches[1], _patch("EplbState", lambda pc, d: state):
        ModelRunnerFL.load_model(runner)

    # The unwrapped language model was registered, not the VLM wrapper.
    assert state.added == [(moe, runner.model_config)]
    assert runner._moe_model is moe


def test_load_model_drafter_moe_with_eplb_registers_drafter():
    """A MoE drafter model under EPLB is registered too, and elastic EP
    is rejected for drafter models."""

    class _MoEModel(Model, MixtureOfExperts):
        num_moe_layers = 2

    model = Model()
    drafter_model = _MoEModel()
    drafter = SimpleNamespace(
        load_model=lambda m: None,
        model=drafter_model,
    )
    runner = base_runner(model, drafter=drafter)
    runner.parallel_config.enable_eplb = True
    # The target model is not MoE: only the drafter registers.
    runner.vllm_config.speculative_config = SimpleNamespace(
        draft_model_config=SimpleNamespace(model="draft/model"),
    )
    loader = Loader(runner.calls, model)

    state = _EplbState(runner.parallel_config, runner.device)
    patches = patched_loader_env(runner, loader)
    with patches[0], patches[1], _patch("EplbState", lambda pc, d: state):
        ModelRunnerFL.load_model(runner)

    assert state.added == [
        (drafter_model, runner.vllm_config.speculative_config.draft_model_config)
    ]


def test_load_model_drafter_moe_with_elastic_ep_rejected():
    """Elastic EP combined with a MoE drafter model is rejected."""
    import pytest

    class _MoEModel(Model, MixtureOfExperts):
        num_moe_layers = 2

    model = Model()
    drafter = SimpleNamespace(
        load_model=lambda m: None,
        model=_MoEModel(),
    )
    runner = base_runner(model, drafter=drafter)
    runner.parallel_config.enable_eplb = True
    runner.parallel_config.enable_elastic_ep = True
    loader = Loader(runner.calls, model)

    state = _EplbState(runner.parallel_config, runner.device)
    patches = patched_loader_env(runner, loader)
    with (
        patches[0],
        patches[1],
        _patch("EplbState", lambda pc, d: state),
        pytest.raises(AssertionError, match="Elastic EP"),
    ):
        ModelRunnerFL.load_model(runner)
