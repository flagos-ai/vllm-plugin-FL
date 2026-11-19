# SPDX-License-Identifier: Apache-2.0
"""Explicit, worker-local Gemma Norm ablation; never enabled by registration.

Install after model loading. Both TP's cached CustomOp dispatch and PCP's
explicit prefill sites select the same native or fused implementation.
The GDN gated RMSNorm and every model parameter remain untouched.
"""

from functools import partial

import torch
from benchmark_fused_norm import prefill_gemma_norm


def _module_norm(module, x, residual=None):
    return module(x) if residual is None else module(x, residual)


class NormAblation:
    def __init__(self, model):
        from vllm.model_executor.layers.layernorm import GemmaRMSNorm

        self.modules = [
            (name, m, m._forward_method)
            for name, m in model.named_modules()
            if isinstance(m, GemmaRMSNorm)
        ]
        self.variant = None
        self.audit_rows = []
        self.auditing = False

    def configure(self, variant, audit=False):
        if variant not in {"native", "fused"}:
            raise ValueError("Norm variant must be native or fused")
        from vllm_fl.cp import prstyle

        self.variant, self.auditing = variant, audit
        self.audit_rows = []
        for name, module, native in self.modules:
            fused = partial(prefill_gemma_norm, module, native_forward=native)
            selected = native if variant == "native" else fused
            if audit:
                selected = self._audited(name, module, native, fused, variant)
            module._forward_method = selected
        # PCP bypasses module dispatch in its usual fused prefill path.
        # Route through modules only for native selection or diagnostic audit.
        prstyle.prefill_gemma_norm = (
            prefill_gemma_norm if variant == "fused" and not audit else _module_norm
        )
        return self.identity()

    def _audited(self, name, module, native, fused, variant):
        def forward(x, residual=None):
            expected = native(x, residual)
            actual = fused(x, residual)
            refs = expected if isinstance(expected, tuple) else (expected,)
            outs = actual if isinstance(actual, tuple) else (actual,)
            equal = [torch.equal(a, b) for a, b in zip(refs, outs)]
            row = {
                "name": name,
                "shape": list(x.shape),
                "stride": list(x.stride()),
                "dtype": str(x.dtype),
                "weight_dtype": str(module.weight.dtype),
                "residual": residual is not None,
                "equal": equal,
            }
            if not all(equal):
                row["max_abs_diff"] = [
                    (a.float() - b.float()).abs().max().item()
                    for a, b in zip(refs, outs)
                ]
                row["mismatch_count"] = [
                    (a != b).sum().item() for a, b in zip(refs, outs)
                ]
            self.audit_rows.append(row)
            return expected if variant == "native" else actual

        return forward

    def identity(self):
        return {
            "variant": self.variant,
            "auditing": self.auditing,
            "module_count": len(self.modules),
            "modules": [
                {
                    "name": name,
                    "weight_dtype": str(m.weight.dtype),
                    "native_dispatch": native.__qualname__,
                }
                for name, m, native in self.modules
            ],
        }
