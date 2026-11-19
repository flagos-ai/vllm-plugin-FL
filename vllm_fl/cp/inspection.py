# SPDX-License-Identifier: Apache-2.0
"""Local validation worker extension for the PR-style Qwen CP copy."""


class QwenCPInspectionWorker:
    def start_qwen_trace(self):
        from vllm.distributed import get_tp_group
        from vllm.model_executor.models.qwen3_next import Qwen3NextDecoderLayer

        self._qwen_trace = []
        self._qwen_trace_handles = []
        if get_tp_group().rank_in_group != 0:
            return

        def record(name, stage, stream):
            stream = stream.float()
            self._qwen_trace.append({
                "layer": name,
                "stage": stage,
                "rows": stream.shape[0],
                "first": stream[0, :16].cpu().tolist(),
                "last": stream[-1, :16].cpu().tolist(),
                "first_norm": stream[0].norm().item(),
                "last_norm": stream[-1].norm().item(),
            })

        def capture(name):
            def hook(module, args, result):
                hidden, residual = result
                record(name, "out", hidden + residual)
            return hook

        def layer_input(name):
            def hook(module, args, kwargs):
                if name.endswith("layers.0"):
                    record(name, "in", kwargs["hidden_states"])
            return hook

        def norm_input(name):
            def hook(module, args, kwargs):
                record(name, "attention_out", args[0])
            return hook

        for name, layer in self.model_runner.model.named_modules():
            if isinstance(layer, Qwen3NextDecoderLayer):
                self._qwen_trace_handles.append(
                    layer.register_forward_hook(capture(name)))
                if name.endswith("layers.0"):
                    self._qwen_trace_handles.append(
                        layer.register_forward_pre_hook(
                            layer_input(name), with_kwargs=True))
                    self._qwen_trace_handles.append(
                        layer.post_attention_layernorm.register_forward_pre_hook(
                            norm_input(name), with_kwargs=True))

    def get_qwen_trace(self):
        for handle in getattr(self, "_qwen_trace_handles", []):
            handle.remove()
        self._qwen_trace_handles = []
        return getattr(self, "_qwen_trace", [])

    def inspect_qwen_parallelism(self):
        from vllm.distributed import get_tp_group
        from vllm.model_executor.layers.mamba.gdn_linear_attn import (
            GatedDeltaNetAttention,
        )
        from vllm_fl.cp.prstyle import QwenPRStyleAttention

        pr_layers, gdn_layers = [], []
        for name, layer in self.model_runner.model.named_modules():
            if isinstance(layer, QwenPRStyleAttention):
                pr_layers.append({
                    "name": name,
                    "q_heads": layer.total_num_heads,
                    "kv_heads": layer.total_num_kv_heads,
                    "prefill_calls": layer.pr_prefill_calls,
                    "prefill_tokens": layer.pr_prefill_tokens,
                    "decode_calls": layer.tp_decode_calls,
                    "decode_tokens": layer.tp_decode_tokens,
                    "fallback_calls": layer.tp_fallback_calls,
                    "fallback_tokens": layer.tp_fallback_tokens,
                })
            if isinstance(layer, GatedDeltaNetAttention):
                gdn_layers.append({
                    "name": name,
                    "tp_size": layer.tp_size,
                    "out_proj_reduce_results": layer.out_proj.reduce_results,
                })
        return {"rank": get_tp_group().rank_in_group,
                "tp_size": get_tp_group().world_size,
                "pr_layers": pr_layers, "gdn_layers": gdn_layers}
