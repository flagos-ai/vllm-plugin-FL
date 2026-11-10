"""Check OOT-only fusion defaults without loading model weights."""

import ast
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest


def _iluvatar_compilation_branch():
    path = Path(__file__).parents[2] / "vllm_fl/platform.py"
    tree = ast.parse(path.read_text())
    branch = next(
        n
        for n in ast.walk(tree)
        if isinstance(n, ast.If)
        and isinstance(n.test, ast.Compare)
        and ast.unparse(n.test) == "getattr(cls, 'vendor_name', None) == 'iluvatar'"
        and any(
            isinstance(c, ast.Name) and c.id == "configure_compilation"
            for c in ast.walk(n)
        )
    )
    return path, branch


@pytest.mark.parametrize("vendor,disabled", [("iluvatar", True), ("nvidia", False)])
def test_only_iluvatar_disables_unimported_cuda_passes(vendor, disabled):
    path, branch = _iluvatar_compilation_branch()
    flags = (
        "fuse_rope_kvcache_cat_mla",
        "fuse_norm_quant",
        "fuse_act_quant",
        "fuse_attn_quant",
    )
    pc = SimpleNamespace(**dict.fromkeys(flags, True))
    config = SimpleNamespace(pass_config=pc)
    code = ast.fix_missing_locations(ast.Module(body=[branch], type_ignores=[]))
    exec(
        compile(code, str(path), "exec"),
        {
            "cls": SimpleNamespace(vendor_name=vendor),
            "compilation_config": config,
            "model_config": None,
            "os": SimpleNamespace(environ={}),
        },
    )
    assert all(getattr(pc, name) is (not disabled) for name in flags)


@pytest.mark.parametrize(
    "vendor,model_type,backend,mode,graph_mode,expected",
    [
        ("iluvatar", "qwen3_5_moe_text", "inductor", 3, "FULL_AND_PIECEWISE", False),
        ("iluvatar", "qwen3_5_moe_text", "inductor", 3, "FULL", False),
        ("nvidia", "qwen3_5_moe_text", "inductor", 3, "FULL", True),
        ("iluvatar", "qwen3_5_text", "inductor", 3, "FULL", True),
        ("iluvatar", None, "inductor", 3, "FULL", True),
        ("iluvatar", "qwen3_5_moe_text", "inductor", 0, "FULL", True),
        ("iluvatar", "qwen3_5_moe_text", "eager", 3, "FULL", True),
        ("iluvatar", "qwen3_5_moe_text", "inductor", 3, "NONE", True),
        ("iluvatar", "qwen3_5_moe_text", "inductor", 3, "PIECEWISE", True),
    ],
)
@pytest.mark.parametrize("allow_buffer_reuse", [True, False])
def test_moe_full_graph_buffer_reuse_is_vendor_and_model_scoped(
    vendor, model_type, backend, mode, graph_mode, expected, allow_buffer_reuse
):
    path, branch = _iluvatar_compilation_branch()
    capture_sizes = [1, 2, 4, 8, 16, 32]
    config = SimpleNamespace(
        pass_config=None,
        backend=backend,
        mode=mode,
        cudagraph_mode=SimpleNamespace(
            has_full_cudagraphs=lambda: graph_mode in ("FULL", "FULL_AND_PIECEWISE")
        ),
        cudagraph_capture_sizes=capture_sizes,
        ir_enable_torch_wrap=True,
        inductor_compile_config={
            "allow_buffer_reuse": allow_buffer_reuse,
            "inplace_buffers": True,
            "combo_kernels": True,
        },
    )
    model = (
        SimpleNamespace(hf_text_config=SimpleNamespace(model_type=model_type))
        if model_type is not None
        else None
    )
    code = ast.fix_missing_locations(ast.Module(body=[branch], type_ignores=[]))
    exec(
        compile(code, str(path), "exec"),
        {
            "cls": SimpleNamespace(vendor_name=vendor),
            "compilation_config": config,
            "model_config": model,
            "CompilationMode": SimpleNamespace(NONE=0),
            "logger": Mock(),
        },
    )
    assert config.inductor_compile_config["allow_buffer_reuse"] is allow_buffer_reuse
    assert config.inductor_compile_config["inplace_buffers"] is expected
    assert config.inductor_compile_config["combo_kernels"] is True
    assert config.mode == mode
    assert config.cudagraph_capture_sizes is capture_sizes
    assert config.ir_enable_torch_wrap is True
