"""Immutable K3 execution policy, derived from model and parallel configuration."""

from dataclasses import dataclass


@dataclass(frozen=True)
class K3ExecutionConfig:
    enabled: bool = False
    tp8_compute: bool = False
    host_pp_feedback: bool = False

    @classmethod
    def from_vllm(cls, config, *, vendor):
        model = config.model_config
        hf = getattr(model, "hf_config", None)
        model_type = getattr(hf, "model_type", None)
        # vLLM constructs K3's language submodel with its own kimi_linear HF
        # config. The selected quantization backend carries the family marker.
        family = getattr(getattr(config, "quant_config", None), "model_family", None)
        is_k3 = model_type == "kimi_k3" or (
            model_type == "kimi_linear" and family == "kimi_k3"
        )
        if vendor != "metax" or not is_k3:
            return cls()
        p = config.parallel_config
        options = config.additional_config.get("k3", {})
        if options:
            raise ValueError(f"Unknown K3 execution options: {sorted(options)}")
        if (
            p.tensor_parallel_size != 8
            or p.data_parallel_size != 1
            or p.enable_expert_parallel
        ):
            raise ValueError("K3 MetaX requires TP8/DP1 with EP disabled")
        return cls(
            enabled=True,
            tp8_compute=p.tensor_parallel_size == 8,
            host_pp_feedback=p.pipeline_parallel_size > 1,
        )
