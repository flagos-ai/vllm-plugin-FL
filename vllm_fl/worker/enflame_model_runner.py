# Copyright (c) 2026 BAAI. All rights reserved.

"""Enflame integration with the upstream vLLM 0.32 V1 model runner."""

from functools import wraps

from vllm.v1.worker.gpu_model_runner import GPUModelRunner

from vllm_fl.dispatch.io_common import managed_inference_mode
from vllm_fl.dispatch.io_dumper import (
    advance_io_step,
    init_io_dump_from_env,
    register_io_module_hooks,
)


def _io_aware_inference(method):
    # These upstream entry points have one torch.inference_mode decorator.
    # Replace its context so dispatch-mode diagnostics can observe leaf ops.
    forward = method.__wrapped__

    @wraps(method)
    @managed_inference_mode()
    def call(*args, **kwargs):
        return forward(*args, **kwargs)

    return call


class EnflameModelRunner(GPUModelRunner):
    execute_model = _io_aware_inference(GPUModelRunner.execute_model)
    _dummy_run = _io_aware_inference(GPUModelRunner._dummy_run)

    @wraps(GPUModelRunner.sample_tokens)
    @managed_inference_mode()
    def sample_tokens(self, grammar_output):
        output = GPUModelRunner.sample_tokens.__wrapped__(self, grammar_output)
        if output is not None:
            advance_io_step()
        return output

    def load_model(self, load_dummy_weights=False, *, model=None):
        from vllm_fl.compilation.enflame_graph import adapt_enflame_graph_wrapper

        init_io_dump_from_env(self.model_config.enforce_eager)
        output = super().load_model(load_dummy_weights, model=model)
        self.model = adapt_enflame_graph_wrapper(getattr(self, "model", None))
        if not load_dummy_weights:
            register_io_module_hooks(self.get_model())
        return output
