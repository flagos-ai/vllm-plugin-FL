"""Backend/Norm selectors; diagnostic GDN wrappers are removed before timing."""

import os

from benchmark_norm_ablation import NormAblation

from vllm_fl.cp.gdn_backend_probe import GDNBackendProbe
from vllm_fl.cp.inspection import QwenCPInspectionWorker


class TimingWorker(QwenCPInspectionWorker):
    def configure(self, norm, backend):
        from vllm.distributed import get_tp_group
        from vllm.model_executor.models import qwen3_next

        if not hasattr(self, "_gdn"):
            self._gdn = GDNBackendProbe(self.model_runner.model)
            self._norm = NormAblation(self.model_runner.model)
        gdn = self._gdn.configure(backend)
        norm_info = self._norm.configure(norm, audit=False)
        return {
            "pid": os.getpid(),
            "rank": get_tp_group().rank_in_group,
            "layer_forward": qwen3_next.Qwen3NextDecoderLayer.forward.__module__,
            "fa_versions": [
                m.attn.impl.vllm_flash_attn_version
                for n, m in self.model_runner.model.named_modules()
                if n.endswith("self_attn")
            ],
            "norm": norm_info,
            "gdn": gdn,
        }

    def begin_core_audit(self):
        self._core_calls = []
        for name, layer in self._gdn.layers:
            op = layer.chunk_gated_delta_rule

            def wrap(name, selected):
                def forward(*args, **kwargs):
                    self._core_calls.append(
                        {
                            "layer": name,
                            "dispatch": selected.__qualname__,
                            "tokens": kwargs["q"].shape[1],
                            "sequences": kwargs["cu_seqlens"].numel() - 1,
                            "chunk_metadata": kwargs.get("chunk_indices") is not None
                            and kwargs.get("chunk_offsets") is not None,
                        }
                    )
                    return selected(*args, **kwargs)

                return forward

            op._forward_method = wrap(name, op._forward_method)

    def core_audit(self):
        return self._core_calls
