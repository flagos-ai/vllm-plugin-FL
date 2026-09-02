# Copyright (c) 2025 BAAI. All rights reserved.

from vllm.config.compilation import CompilationConfig, CompilationMode

import vllm_fl


def test_register_deepseek_v4_fl_attention_as_splitting_op(monkeypatch):
    attention_ops = ["vllm::deepseek_v4_attention"]
    monkeypatch.setattr(CompilationConfig, "_attention_ops", attention_ops)

    vllm_fl._register_compilation_splitting_ops()
    vllm_fl._register_compilation_splitting_ops()

    assert attention_ops == [
        "vllm::deepseek_v4_attention",
        "vllm::deepseek_v4_fl_attention",
    ]

    config = CompilationConfig(mode=CompilationMode.VLLM_COMPILE)
    config.set_splitting_ops_for_v1("")
    assert config.splitting_ops.count("vllm::deepseek_v4_fl_attention") == 1
