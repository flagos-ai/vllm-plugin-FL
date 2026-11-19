# SPDX-License-Identifier: Apache-2.0
"""Explicit worker-local GDN backend experiment; never auto-registered.

This is a validation helper, not a production dispatch policy. Native vLLM
also exposes the same backend choice through additional_config. Runtime
selection allows paired same-model checks without repeated weight loading.
"""


class GDNBackendProbe:
    def __init__(self, model):
        from vllm.model_executor.layers.mamba.gdn_linear_attn import (
            GatedDeltaNetAttention,
        )

        self.layers = [(name, layer) for name, layer in model.named_modules()
                       if type(layer) is GatedDeltaNetAttention]
        self.original = [layer.chunk_gated_delta_rule._forward_method
                         for _, layer in self.layers]

    def configure(self, backend):
        if backend not in {"flashinfer", "triton", "original"}:
            raise ValueError("GDN backend must be flashinfer, triton or original")
        if backend == "flashinfer":
            from vllm.platforms import current_platform
            if not (current_platform.is_cuda()
                    and current_platform.is_device_capability(90)):
                raise ValueError("This FlashInfer comparison requires Hopper")
        for (_, layer), saved in zip(self.layers, self.original, strict=True):
            op = layer.chunk_gated_delta_rule
            op._forward_method = (saved if backend == "original" else
                                  op.forward_cuda if backend == "flashinfer" else
                                  op.forward_native)
        return [{"layer": name,
                 "dispatch": layer.chunk_gated_delta_rule._forward_method.__qualname__}
                for name, layer in self.layers]

    def restore(self):
        return self.configure("original")
