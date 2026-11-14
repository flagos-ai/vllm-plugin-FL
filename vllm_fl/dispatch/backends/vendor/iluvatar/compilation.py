# Copyright (c) 2026 BAAI. All rights reserved.

from vllm.config import CompilationMode
from vllm.logger import init_logger

logger = init_logger(__name__)


def configure_compilation(compilation_config, model_config):
    compilation_config.cudagraph_mm_encoder = False
    compilation_config.use_inductor_graph_partition = True
    hf_text_config = getattr(model_config, "hf_text_config", None)
    if (
        getattr(hf_text_config, "model_type", None) == "qwen3_5_moe_text"
        and compilation_config.backend == "inductor"
        and compilation_config.mode != CompilationMode.NONE
        and compilation_config.cudagraph_mode.has_full_cudagraphs()
    ):
        compilation_config.inductor_compile_config["inplace_buffers"] = False
        logger.warning_once(
            "Disabling Inductor inplace buffers for Iluvatar Qwen3.5 MoE "
            "full graphs after output corruption; compilation and "
            "graph capture remain enabled."
        )
    _pc = getattr(compilation_config, "pass_config", None)
    if _pc is not None:
        _pc.fuse_rope_kvcache_cat_mla = False
        # OOT platforms do not import these CUDA-only fusion passes.
        # Explicit custom_ops=['all'] otherwise enables them on 0.32.
        _pc.fuse_norm_quant = False
        _pc.fuse_act_quant = False
        _pc.fuse_attn_quant = False
