"""Opt-in source-level NVTX annotations, no synchronization inside ranges."""

import functools
import importlib
import os

import torch
from timing_worker import TimingWorker


class TraceWorker(TimingWorker):
    def install_trace(self, backend):
        from vllm.distributed import get_tp_group

        from vllm_fl.cp import prstyle

        self._trace_sample = -1
        self._core_meta = {}
        self._core_calls = []

        def wrap(owner, name, label):
            original = getattr(owner, name)

            @functools.wraps(original)
            def annotated(*args, **kwargs):
                with torch.cuda.nvtx.range(label):
                    return original(*args, **kwargs)

            setattr(owner, name, annotated)

        for name, layer in self._gdn.layers:
            op = layer.chunk_gated_delta_rule

            def core_wrapper(name, original):
                @functools.wraps(original)
                def annotated(*args, **kwargs):
                    self._core_calls.append(
                        {"sample": self._trace_sample, "layer": name}
                    )
                    if name not in self._core_meta:
                        self._core_meta[name] = {
                            key: {
                                "shape": list(value.shape),
                                "stride": list(value.stride()),
                                "dtype": str(value.dtype),
                            }
                            for key, value in kwargs.items()
                            if isinstance(value, torch.Tensor)
                        }
                    with torch.cuda.nvtx.range("gdn.core:" + name):
                        return original(*args, **kwargs)

                return annotated

            op._forward_method = core_wrapper(name, op._forward_method)

        chunk = importlib.import_module("vllm.model_executor.layers.fla.ops.chunk")
        for name in (
            "l2norm_fwd",
            "chunk_local_cumsum",
            "chunk_scaled_dot_kkt_fwd",
            "solve_tril",
            "recompute_w_u_fwd",
            "chunk_gated_delta_rule_fwd_h",
            "chunk_fwd_o",
        ):
            wrap(chunk, name, "gdn.stage:" + name)
        gdn = importlib.import_module(
            "vllm.model_executor.layers.mamba.gdn_linear_attn"
        )
        wrap(gdn, "l2norm_fwd", "gdn.stage:l2norm_fwd")
        fi = importlib.import_module("flashinfer.gdn_prefill")
        wrap(fi, "chunk_gated_delta_rule", "gdn.stage:flashinfer_api")
        wrap(prstyle, "try_prefill_gdn", "model.stage:gdn_layer")
        wrap(prstyle, "_pr_moe", "model.stage:moe")
        wrap(prstyle.QwenPRStyleAttention, "forward", "model.stage:full_attention")
        for name in ("all_gather", "reduce_scatter", "all_reduce"):
            wrap(get_tp_group(), name, "comm:" + name)

        model = self.model_runner.model
        original = model.forward

        @functools.wraps(original)
        def forward(*args, **kwargs):
            with torch.cuda.nvtx.range(f"request:{self._trace_sample}"):
                return original(*args, **kwargs)

        model.forward = forward
        return {
            "pid": os.getpid(),
            "rank": get_tp_group().rank_in_group,
            "backend": backend,
            "sm_count": torch.cuda.get_device_properties(0).multi_processor_count,
        }

    def set_trace_sample(self, sample):
        self._trace_sample = sample

    def synchronize_trace(self):
        torch.cuda.synchronize()

    def start_trace(self):
        torch.cuda.synchronize()
        torch.cuda.cudart().cudaProfilerStart()

    def stop_trace(self):
        torch.cuda.cudart().cudaProfilerStop()

    def trace_metadata(self):
        return {
            "pid": os.getpid(),
            "core_meta": self._core_meta,
            "calls": self._core_calls,
        }
