from types import SimpleNamespace

from vllm.model_executor.layers.fused_moe.oracle.unquantized import (
    UnquantizedMoeBackend,
)

from vllm_fl.ops.fused_moe import fused_moe_utils


def test_ascend_uses_fl_experts_when_flaggems_is_disabled(monkeypatch):
    """Ascend vendor kernels do not require global FlagGems ATen patches."""
    platform = SimpleNamespace(
        is_cpu=lambda: False,
        is_tpu=lambda: False,
        is_out_of_tree=lambda: True,
    )
    monkeypatch.setattr(fused_moe_utils, "current_platform", platform)
    monkeypatch.setattr(fused_moe_utils, "get_platform_name", lambda: "ascend")
    monkeypatch.setattr(fused_moe_utils, "use_flaggems", lambda: False)

    backend, experts_cls = fused_moe_utils.select_unquantized_moe_backend_oot(
        SimpleNamespace()
    )

    assert backend is UnquantizedMoeBackend.TRITON
    assert experts_cls is fused_moe_utils.TritonExpertsFL
