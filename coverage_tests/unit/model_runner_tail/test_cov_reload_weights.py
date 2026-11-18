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

"""Coverage tests for ModelRunnerFL.reload_weights (vllm_fl/worker/
model_runner.py): the argument validation, the kernel-format direct
parameter copy, the checkpoint-format layerwise path, disk reload and
the not-loaded-weights warning.

The model is a real ``nn.Module`` subclass so weight copies and the
loaded/missing bookkeeping run for real on CPU. ``get_model_loader`` and
the vllm layerwise-reload helpers are patched at module boundaries
(imported collaborator names of the module under test): the layerwise
machinery swaps parameters onto the meta device and expects vllm-quantized
layer types, so the branch wiring is the target here.
"""

from types import SimpleNamespace

import pytest
import torch

from . import _imports  # noqa: F401  (installs local import shims first)
from ._loader_helpers import (
    Loader,
    Model,
    Patch,
    base_runner,
    module_under_test,
)
from vllm_fl.worker.model_runner import ModelRunnerFL

model_runner_module = module_under_test()


def test_reload_weights_kernel_format_copies_into_parameters():
    """Kernel-format weights are copied directly into the model's
    parameters; caches reset afterwards."""
    model = Model()
    runner = base_runner(model)
    resets = []
    runner.reset_encoder_cache = lambda: resets.append("reset_enc")
    runner.reset_mm_cache = lambda: resets.append("reset_mm")

    ModelRunnerFL.reload_weights(
        runner,
        weights_iterator=[("linear.weight", torch.full((2, 2), 7.0))],
        is_checkpoint_format=False,
    )

    # The real parameter copy ran.
    assert model.linear.weight.flatten().tolist() == [7.0, 7.0, 7.0, 7.0]
    assert resets == ["reset_enc", "reset_mm"]


def test_reload_weights_kernel_format_with_disk_source_warns(caplog):
    """Kernel-format weights with no iterator warns that disk weights
    are checkpoint format (log-only branch)."""
    import logging

    model = Model()
    runner = base_runner(model)
    runner.reset_encoder_cache = lambda: None
    runner.reset_mm_cache = lambda: None
    loader = Loader(
        runner.calls,
        model,
        all_weights=[("linear.weight", torch.full((2, 2), 5.0))],
    )

    with (
        Patch(
            module_under_test(),
            "get_model_loader",
            lambda load_config: loader,
        ),
        caplog.at_level(
            logging.WARNING,
            logger="vllm_fl.worker.model_runner",
        ),
    ):
        ModelRunnerFL.reload_weights(runner, is_checkpoint_format=False)

    warned = [r for r in caplog.records if "checkpoint format" in r.getMessage()]
    assert len(warned) == 1
    # The disk weights were still loaded through the kernel path.
    assert model.linear.weight.flatten().tolist() == [5.0] * 4


def test_reload_weights_checkpoint_format_runs_layerwise_helpers():
    """Checkpoint-format weights run between the layerwise-reload
    helpers and flow into the model's own load_weights."""
    model = Model()
    runner = base_runner(model)
    layerwise = []
    recorder = SimpleNamespace(
        init=lambda m: layerwise.append(("init", m)),
        finalize=lambda m, mc: layerwise.append(("finalize", m, mc)),
    )
    runner.reset_encoder_cache = lambda: None
    runner.reset_mm_cache = lambda: None

    with (
        Patch(
            module_under_test(),
            "initialize_layerwise_reload",
            recorder.init,
        ),
        Patch(
            module_under_test(),
            "finalize_layerwise_reload",
            recorder.finalize,
        ),
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
    # The model's own load_weights performed the real copy.
    assert model.linear.weight.flatten().tolist() == [3.0, 3.0, 3.0, 3.0]


def test_reload_weights_from_disk_requires_loader_support():
    """A load format without get_all_weights raises NotImplementedError
    when reloading from disk."""
    model = Model()
    runner = base_runner(model)

    class _NoDiskLoader:
        pass

    with (
        Patch(
            module_under_test(),
            "get_model_loader",
            lambda load_config: _NoDiskLoader(),
        ),
        pytest.raises(NotImplementedError, match="safetensors"),
    ):
        ModelRunnerFL.reload_weights(runner)


def test_reload_weights_from_disk_uses_loader_weights():
    """Disk reload rewrites model_config.model to the weights path and
    feeds the loader's weights through the layerwise reload path."""
    model = Model()
    runner = base_runner(model)
    runner.reset_encoder_cache = lambda: None
    runner.reset_mm_cache = lambda: None
    loader = Loader(
        runner.calls,
        model,
        all_weights=[("linear.weight", torch.full((2, 2), 11.0))],
    )
    layerwise = []

    with (
        Patch(
            module_under_test(),
            "get_model_loader",
            lambda load_config: loader,
        ),
        Patch(
            module_under_test(),
            "initialize_layerwise_reload",
            lambda m: layerwise.append("init"),
        ),
        Patch(
            module_under_test(),
            "finalize_layerwise_reload",
            lambda m, mc: layerwise.append("finalize"),
        ),
    ):
        ModelRunnerFL.reload_weights(runner, weights_path="/tmp/w")

    assert runner.model_config.model == "/tmp/w"
    assert ("get_all_weights", runner.model_config, model) in runner.calls
    assert layerwise == ["init", "finalize"]
    assert model.linear.weight.flatten().tolist() == [11.0] * 4


def test_reload_weights_warns_on_unloaded_checkpoint_weights(caplog):
    """Weights that never load are reported as missing."""
    import logging

    model = Model()
    runner = base_runner(model)
    runner.reset_encoder_cache = lambda: None
    runner.reset_mm_cache = lambda: None

    # The iterator loads nothing: the parameter stays at its initial
    # value and is reported in the not-loaded warning.
    with caplog.at_level(
        logging.WARNING,
        logger="vllm_fl.worker.model_runner",
    ):
        ModelRunnerFL.reload_weights(
            runner,
            weights_iterator=[],
            is_checkpoint_format=False,
        )

    not_loaded = [r for r in caplog.records if "not loaded" in r.getMessage()]
    assert len(not_loaded) == 1
    assert "linear.weight" in not_loaded[0].getMessage()


def test_reload_weights_quantized_model_skips_missing_check(caplog):
    """A quantized model_config skips the not-loaded validation."""
    import logging

    model = Model()
    runner = base_runner(model)
    runner.model_config.quantization = "fp8"
    runner.reset_encoder_cache = lambda: None
    runner.reset_mm_cache = lambda: None

    with caplog.at_level(
        logging.WARNING,
        logger="vllm_fl.worker.model_runner",
    ):
        ModelRunnerFL.reload_weights(
            runner,
            weights_iterator=[],
            is_checkpoint_format=False,
        )

    # No not-loaded warning was emitted for the quantized model.
    not_loaded = [r for r in caplog.records if "not loaded" in r.getMessage()]
    assert not_loaded == []
