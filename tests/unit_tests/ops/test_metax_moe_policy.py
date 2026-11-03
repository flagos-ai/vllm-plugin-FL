# Copyright (c) 2026 BAAI. All rights reserved.
"""MetaX's validated MoE path must fail clearly when disabled by policy."""

import ast
from pathlib import Path
from types import SimpleNamespace

import pytest


def _selector(flag_gems_enabled, blacklist):
    source = (
        Path(__file__).resolve().parents[3] / "vllm_fl/ops/fused_moe/fused_moe_utils.py"
    )
    names = {
        "_should_use_fl_triton_experts",
        "_get_priority_backends",
        "select_unquantized_moe_backend_oot",
    }
    functions = [
        node
        for node in ast.parse(source.read_text(encoding="utf-8")).body
        if isinstance(node, ast.FunctionDef) and node.name in names
    ]
    future = ast.ImportFrom(
        module="__future__", names=[ast.alias(name="annotations")], level=0
    )
    experts = object()
    backend = object()
    namespace = {
        "current_platform": SimpleNamespace(
            vendor_name="metax",
            is_tpu=lambda: False,
            is_cuda=lambda: False,
            is_cpu=lambda: False,
            is_xpu=lambda: False,
            is_rocm=lambda: False,
            is_out_of_tree=lambda: True,
        ),
        "use_flaggems": lambda: flag_gems_enabled,
        "get_oot_blacklist": lambda: blacklist,
        "UnquantizedMoeBackend": SimpleNamespace(TRITON=backend),
        "TritonExpertsFL": experts,
    }
    module = ast.fix_missing_locations(
        ast.Module(body=[future, *functions], type_ignores=[])
    )
    exec(compile(module, str(source), "exec"), namespace)
    return namespace["select_unquantized_moe_backend_oot"], backend, experts


def test_default_metax_policy_keeps_flaggems_experts():
    select, backend, experts = _selector(True, [])
    assert select(SimpleNamespace(is_lora_enabled=False)) == (backend, experts)


@pytest.mark.parametrize(
    "flag_gems_enabled,blacklist", [(False, []), (True, ["fused_moe"])]
)
def test_disabled_metax_moe_policy_reports_supported_backend(
    flag_gems_enabled, blacklist
):
    select, _, _ = _selector(flag_gems_enabled, blacklist)
    with pytest.raises(NotImplementedError) as error:
        select(SimpleNamespace(is_lora_enabled=False))
    message = str(error.value)
    assert "FlagGems unquantized MoE backend" in message
    assert "USE_FLAGGEMS=1" in message
    assert "allow fused_moe" in message
    assert "No native MoE fallback is validated" in message
