# SPDX-License-Identifier: Apache-2.0
"""gpt-oss packed-expert names must follow vLLM 0.24's routed-experts layout.

vLLM 0.24 puts expert parameters under ``...experts.routed_experts.*`` while
some builds' ``GptOssModel._load_weights_other`` still look up
``...experts.w2_weight``; loading a standard HF gpt-oss checkpoint then fails
with ``KeyError: 'layers.0.mlp.experts.w2_weight'``. The plugin adapts the names
in ``load_weights`` and leaves already-correct layouts untouched.
"""

import importlib.util
from pathlib import Path

_SPEC = importlib.util.spec_from_file_location(
    "gpt_oss_weights_under_test",
    Path(__file__).resolve().parents[2] / "vllm_fl/patches/gpt_oss_weights.py",
)
_MODULE = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_MODULE)
remap_expert_param_name = _MODULE.remap_expert_param_name

PARAMS = {
    "layers.0.mlp.experts.routed_experts.w2_weight",
    "layers.0.mlp.experts.routed_experts.w13_weight",
    "layers.0.mlp.experts.routed_experts.w2_bias",
    "layers.0.mlp.router.weight",
}


def test_expert_names_move_under_routed_experts():
    assert (
        remap_expert_param_name("layers.0.mlp.experts.w2_weight", PARAMS)
        == "layers.0.mlp.experts.routed_experts.w2_weight"
    )
    assert (
        remap_expert_param_name("layers.0.mlp.experts.w13_weight", PARAMS)
        == "layers.0.mlp.experts.routed_experts.w13_weight"
    )
    assert (
        remap_expert_param_name("layers.0.mlp.experts.w2_bias", PARAMS)
        == "layers.0.mlp.experts.routed_experts.w2_bias"
    )


def test_other_parameters_are_untouched():
    assert remap_expert_param_name("layers.0.mlp.router.weight", PARAMS) == (
        "layers.0.mlp.router.weight"
    )
    assert remap_expert_param_name("layers.0.attn.sinks", PARAMS) == "layers.0.attn.sinks"


def test_names_without_legacy_target_are_kept():
    assert remap_expert_param_name(
        "layers.0.mlp.experts.w2_weight", {"layers.0.mlp.experts.w2_weight"}
    ) == "layers.0.mlp.experts.w2_weight"
