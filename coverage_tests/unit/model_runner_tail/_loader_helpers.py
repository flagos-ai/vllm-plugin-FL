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

"""Shared fixtures for the load_model / reload_weights coverage tests.

The runner is an input namespace (CONTRACT R2); the loaded model is a
real ``nn.Module`` subclass so weight copies run for real on CPU; the
loader / memory profiler are recording collaborators patched at module
boundaries (imported names of the module under test, never its own
functions).
"""

from __future__ import annotations

from types import SimpleNamespace

import torch
import torch.nn as nn

import vllm_fl.worker.model_runner as model_runner_module
from vllm_fl.worker.model_runner import ModelRunnerFL


class Patch:
    """Deterministic set/restore of a module attribute (R2 input boundary)."""

    def __init__(self, module, name, value):
        self.module, self.name, self.value = module, name, value

    def __enter__(self):
        self.saved = getattr(self.module, self.name)
        setattr(self.module, self.name, self.value)
        return self

    def __exit__(self, *exc):
        setattr(self.module, self.name, self.saved)
        return False


class Model(nn.Module):
    """Real module: named_parameters / get_parameter / copy_ all work."""

    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(2, 2, bias=False)
        # Inference models never track gradients on weights; this also
        # keeps the real kernel-format copy_ legal.
        self.linear.weight.requires_grad_(False)

    def load_weights(self, weights):
        loaded = set()
        for name, tensor in weights:
            self.get_parameter(name).copy_(tensor)
            loaded.add(name)
        return loaded


class Loader:
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


class MemoryProfilerFactory:
    """Stand-in for the CUDA DeviceMemoryProfiler class."""

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


def base_runner(model, **extra):
    """Runner namespace for load_model / reload_weights."""
    from vllm.config import CompilationMode, CUDAGraphMode

    vllm_config = SimpleNamespace(
        compilation_config=SimpleNamespace(
            mode=CompilationMode.NONE,
            cudagraph_mode=CUDAGraphMode.NONE,
        ),
        speculative_config=None,
    )
    runner = SimpleNamespace(
        model_config=SimpleNamespace(
            model="test/model",
            enforce_eager=True,
            quantization=None,
            multimodal_config=None,
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
    )
    # Real branch wiring: with aux outputs disabled this is a no-op.
    runner._setup_eagle3_aux_hidden_state_outputs = (
        ModelRunnerFL._setup_eagle3_aux_hidden_state_outputs.__get__(
            runner,
            ModelRunnerFL,
        )
    )
    runner.get_model = lambda: runner.model
    runner.calls = []
    for key, value in extra.items():
        setattr(runner, key, value)
    return runner


def patched_loader_env(runner, loader):
    """Common patched collaborators for load_model."""
    return [
        Patch(
            model_runner_module,
            "get_model_loader",
            lambda load_config: loader,
        ),
        Patch(
            model_runner_module,
            "DeviceMemoryProfiler",
            MemoryProfilerFactory(runner.calls),
        ),
    ]


def module_under_test():
    """The module under test (for further patch boundaries)."""
    return model_runner_module
