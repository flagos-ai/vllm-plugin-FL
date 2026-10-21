# Copyright (c) 2026 BAAI. All rights reserved.

import torch
from torch._inductor.decomposition import select_decomp_table
from torch._subclasses.fake_tensor import FakeTensorMode
from torch.fx.experimental.proxy_tensor import make_fx

from vllm.compilation.passes.fusion.rms_quant_fusion import (
    RMSNormQuantFusionPass,
)
from vllm.config import VllmConfig, set_current_vllm_config

from vllm_fl.compilation.int8_rms_quant_fusion import (
    patch_vllm_int8_rms_quant_fusion,
)


def test_deepseek_v4_qr_int8_pass_replaces_unfused_subgraph():
    patch_vllm_int8_rms_quant_fusion()

    def fn(qr, kv, q_weight, kv_weight):
        return torch.ops.vllm.deepseek_v4_qr_int8.default(
            qr,
            kv,
            q_weight,
            kv_weight,
            1e-6,
        )

    fake_mode = FakeTensorMode()
    with fake_mode:
        args = (
            torch.empty(5, 16, device="cuda", dtype=torch.bfloat16),
            torch.empty(5, 8, device="cuda", dtype=torch.bfloat16),
            torch.empty(16, device="cuda", dtype=torch.bfloat16),
            torch.empty(8, device="cuda", dtype=torch.bfloat16),
        )
        graph_module = make_fx(
            fn,
            select_decomp_table(),
            tracing_mode="fake",
        )(*args)

    config = VllmConfig()
    with set_current_vllm_config(config):
        fusion_pass = RMSNormQuantFusionPass(config)
        fusion_pass(graph_module.graph)

    graph_module.graph.lint()
    targets = {node.target for node in graph_module.graph.nodes}
    assert fusion_pass.matched_count >= 1
    assert torch.ops.vllm.deepseek_v4_qr_int8_fusion.default in targets
    assert torch.ops.vllm.deepseek_v4_qr_int8.default not in targets
