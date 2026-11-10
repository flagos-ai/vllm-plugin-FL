# Copyright (c) 2026 BAAI. All rights reserved.

"""Tests for dispatch builtin operator registration."""

from unittest.mock import MagicMock, patch

from vllm_fl.dispatch.builtin_ops import (
    _find_vendor_backend_dir,
    _register_vendor_backends,
)
from vllm_fl.dispatch.registry import OpRegistry


class TestFindVendorBackendDir:
    def test_maps_vendor_names(self):
        available = {"cuda", "ascend", "metax", "iluvatar"}
        assert _find_vendor_backend_dir("nvidia", available) == "cuda"
        assert _find_vendor_backend_dir("ascend", available) == "ascend"
        assert _find_vendor_backend_dir("metax", available) == "metax"
        assert _find_vendor_backend_dir("iluvatar", available) == "iluvatar"

    def test_maca_alias_resolves_to_metax(self):
        assert _find_vendor_backend_dir("maca", {"metax"}) == "metax"

    def test_unknown_vendor_returns_none(self):
        assert _find_vendor_backend_dir("unknown", {"cuda", "ascend"}) is None


class TestRegisterVendorBackends:
    def test_registers_only_current_vendor_backend(self):
        registry = MagicMock()
        module = MagicMock()

        with (
            patch("vllm_fl.dispatch.builtin_ops._VENDOR_BACKENDS_DIR", "/fake/vendor"),
            patch(
                "vllm_fl.dispatch.builtin_ops._get_current_vendor_backend_dirs",
                return_value="metax",
            ),
            patch("vllm_fl.dispatch.builtin_ops.os.path.isdir", return_value=True),
            patch(
                "vllm_fl.dispatch.builtin_ops.os.listdir",
                return_value=["metax", "cuda", "ascend"],
            ),
            patch("vllm_fl.dispatch.builtin_ops.os.path.isfile", return_value=True),
            patch(
                "vllm_fl.dispatch.builtin_ops.importlib.import_module",
                return_value=module,
            ) as import_module,
        ):
            _register_vendor_backends(registry)

        import_module.assert_called_once_with(
            ".backends.vendor.metax.register_ops",
            package="vllm_fl.dispatch",
        )
        module.register_builtins.assert_called_once_with(registry)


def test_reference_registration_keeps_implemented_fallbacks():
    """A missing optional method must not drop every reference fallback."""
    from vllm_fl.dispatch.backends.reference.register_ops import (
        register_builtins as register_reference,
    )

    registry = OpRegistry()
    register_reference(registry)

    implemented = {
        "dynamic_per_token_quant_int8",
        "silu_and_mul",
        "gelu_and_mul",
        "rms_norm",
        "rotary_embedding",
        "attention_backend",
        "invoke_fused_moe_triton_kernel",
        "moe_align_block_size",
        "moe_sum",
        "topk_softmax",
    }
    missing = {
        "grouped_topk",
    }
    for op_name in implemented:
        assert registry.get_implementation(op_name, "reference.torch") is not None
    for op_name in missing:
        assert registry.get_implementation(op_name, "reference.torch") is None


def test_reference_topk_softmax_matches_upstream_contract():
    """Weights/indices match torch, and the third output keeps upstream's order.

    Upstream's ``topkGating`` kernel writes ``k_idx * num_tokens + token_idx``
    into ``token_expert_indices``; the row-major ``token_idx * topk + k_idx``
    ordering is not interchangeable when ``topk > 1``.
    """
    import torch

    from vllm_fl.dispatch.backends.reference.reference import ReferenceBackend

    num_tokens, num_experts, topk = 3, 4, 2
    gating = torch.randn(num_tokens, num_experts)
    topk_weights = torch.empty(num_tokens, topk)
    topk_ids = torch.empty(num_tokens, topk, dtype=torch.int32)
    token_expert_indices = torch.empty(num_tokens, topk, dtype=torch.int32)

    ReferenceBackend().topk_softmax(
        topk_weights, topk_ids, token_expert_indices, gating, False
    )

    expected_indices = torch.tensor(
        [
            [0, num_tokens],
            [1, num_tokens + 1],
            [2, num_tokens + 2],
        ],
        dtype=torch.int32,
    )
    assert torch.equal(token_expert_indices, expected_indices)

    probs = torch.softmax(gating.float(), dim=-1)
    expected_weights, expected_ids = torch.topk(probs, topk, dim=-1)
    assert torch.allclose(topk_weights, expected_weights)
    assert torch.equal(topk_ids, expected_ids.to(torch.int32))


def test_reference_moe_align_block_size_drops_unowned_experts():
    """``expert_map`` entries mapping to -1 must not land in a real bucket."""
    import torch

    from vllm_fl.dispatch.backends.reference.reference import ReferenceBackend

    topk_ids = torch.tensor([[0, 1], [2, 3]], dtype=torch.int32)
    # Global 0 -> local 0, global 2 -> local 1; 1 and 3 belong to another rank.
    expert_map = torch.tensor([0, -1, 1, -1], dtype=torch.int32)

    sorted_ids, expert_ids, num_post = ReferenceBackend().moe_align_block_size(
        topk_ids, 2, 2, expert_map, ignore_invalid_experts=True
    )

    keep = int(num_post.item())
    # Only the owned pairs (slots 0 and 2) land in real buckets; 4 is the
    # padding sentinel (== number of top-k pairs) used for the gaps.
    placed = [slot for slot in sorted_ids[:keep].tolist() if slot < 4]
    assert placed == [0, 2]
    assert (sorted_ids[keep:] == 4).all()
    assert expert_ids[:2].tolist() == [0, 1]


def test_reference_moe_align_block_size_keeps_every_token():
    import torch

    from vllm_fl.dispatch.backends.reference.reference import ReferenceBackend

    topk_ids = torch.tensor([[0, 0], [1, 1]], dtype=torch.int32)

    sorted_ids, _, num_post = ReferenceBackend().moe_align_block_size(
        topk_ids, 2, 2
    )

    keep = int(num_post.item())
    assert sorted(sorted_ids[:keep].tolist()) == [0, 1, 2, 3]


def test_reference_moe_align_block_size_remaps_expert_ids_without_ignore_flag():
    """Without ``ignore_invalid_experts`` buckets stay global and ids are remapped."""
    import torch

    from vllm_fl.dispatch.backends.reference.reference import ReferenceBackend

    topk_ids = torch.tensor([[0, 1], [2, 3]], dtype=torch.int32)
    expert_map = torch.tensor([0, -1, 1, -1], dtype=torch.int32)

    sorted_ids, expert_ids, num_post = ReferenceBackend().moe_align_block_size(
        topk_ids, 2, 4, expert_map
    )

    # Buckets are built on global ids, so every pair is kept ...
    keep = int(num_post.item())
    placed = [slot for slot in sorted_ids[:keep].tolist() if slot < 4]
    assert placed == [0, 1, 2, 3]
    # ... and the returned expert ids are mapped into the local shard.
    assert expert_ids[:4].tolist() == [0, -1, 1, -1]

