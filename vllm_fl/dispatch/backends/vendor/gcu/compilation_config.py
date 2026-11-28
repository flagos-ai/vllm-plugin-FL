# Copyright (c) 2026 BAAI. All rights reserved.
"""Version the GCU compiler through the supported Inductor pass cache key."""

from vllm.compilation.passes.inductor_pass import CallableInductorPass, InductorPass


class GCUCodegenPass(InductorPass):
    def __init__(self, mode, previous=None):
        self.mode = mode
        self.previous = (
            CallableInductorPass(previous)
            if previous is not None and not isinstance(previous, InductorPass)
            else previous
        )

    def uuid(self):
        previous = None if self.previous is None else self.previous.uuid()
        return ("fl_gcu_codegen", self.mode, previous)

    def __call__(self, graph):
        if self.previous is not None:
            self.previous(graph)


def configure_codegen_cache(compilation_config):
    from .impl.inductor import gcu_codegen_mode

    options = getattr(compilation_config, "inductor_compile_config", None)
    if (
        options is None
        or getattr(compilation_config, "backend", None) != "inductor"
        or not getattr(compilation_config, "mode", None)
    ):
        return
    key = "post_grad_custom_post_pass"
    previous = options.get(key)
    if isinstance(previous, GCUCodegenPass):
        previous = previous.previous
    options[key] = GCUCodegenPass(gcu_codegen_mode(), previous)
