"""T-Head launch metadata overrides preserve other shapes and are idempotent."""

from types import SimpleNamespace

import torch

from vllm_fl.dispatch.backends.vendor.thead.patch import (
    _patch_prefill_head_tile,
    _PrefillHeadTile,
)


def test_head_tile_override_and_fallback():
    native, jit = _Kernel("native"), _Kernel("padded")
    wrapper = _PrefillHeadTile(native, jit)
    args = (torch.empty(1, dtype=torch.bfloat16),) + (None,) * 9 + (8, 128, 16)
    opts = dict(BLOCK_SIZE_Q=1, BLOCK_SIZE_K=128, USE_FP8=False)
    assert wrapper[(1,)](*args, **opts) == "padded"
    assert jit.calls[-1][2] == {
        **opts,
        "BLOCK_SIZE_H": 16,
        "BLOCK_SIZE_QH": 16,
        "BLOCK_SIZE_D": 128,
        "BLOCK_SIZE_T": 16,
    }
    assert wrapper[(1,)](*(args[:10] + (4, 128, 16)), **opts) == "native"
    assert len(native.calls) == 1


def test_head_tile_hook_is_idempotent():
    names = ["q_ptr"] + ["unused"] * 9 + ["gqa_group_size", "head_dim", "max_topk"]
    attn = SimpleNamespace(
        _gqa_sparse_fwd_kernel=SimpleNamespace(
            fn=SimpleNamespace(__name__="_gqa_sparse_fwd_kernel", arg_names=names)
        )
    )
    assert _patch_prefill_head_tile(attn)
    saved = attn._gqa_sparse_fwd_kernel
    assert not _patch_prefill_head_tile(attn) and attn._gqa_sparse_fwd_kernel is saved


def test_worker_installs_launch_compat_before_model_load(monkeypatch):
    from contextlib import nullcontext

    import vllm_fl.worker.worker as worker_module
    from vllm_fl.dispatch.backends.vendor.thead import patch
    from vllm_fl.patches import minimax_m3, moe_sum

    events = []
    monkeypatch.setattr(
        worker_module, "set_current_vllm_config", lambda config: nullcontext()
    )
    monkeypatch.setattr(moe_sum, "patch_vllm_moe_sum", lambda: events.append("moe_sum"))
    monkeypatch.setattr(
        patch, "apply_thead_patches", lambda model_config: events.append("thead")
    )
    monkeypatch.setattr(
        minimax_m3, "prepare_minimax_m3_model", lambda config: events.append("model")
    )
    for vendor in ["thead", "nvidia"]:
        events.clear()
        monkeypatch.setattr(
            worker_module,
            "current_platform",
            SimpleNamespace(vendor_name=vendor, device_type="cuda"),
        )
        worker = SimpleNamespace(
            vllm_config=object(),
            model_config=object(),
            model_runner=SimpleNamespace(
                load_model=lambda **kwargs: events.append("load")
            ),
        )
        worker_module.WorkerFL.load_model(worker)
        assert events == (
            ["moe_sum", "model", "thead", "load"]
            if vendor == "thead"
            else ["moe_sum", "model", "load"]
        )


def test_other_models_do_not_import_or_patch_minimax(monkeypatch):
    import sys

    from vllm_fl.dispatch.backends.vendor.thead.patch import apply_thead_patches

    monkeypatch.setitem(sys.modules, "vllm.models.minimax_m3", None)
    for architectures in [["Qwen3ForCausalLM"], [], None]:
        config = SimpleNamespace(hf_config=SimpleNamespace(architectures=architectures))
        assert apply_thead_patches(config) is False


def test_model_architecture_selects_head_tile_and_preserves_topk(monkeypatch):
    import sys
    from unittest.mock import Mock

    from vllm_fl.dispatch.backends.vendor.thead import patch

    native_topk = object()
    topk = SimpleNamespace(_topk_index_kernel=native_topk)
    attention = object()
    monkeypatch.setitem(
        sys.modules,
        "vllm.models.minimax_m3.common.ops",
        SimpleNamespace(index_topk=topk, sparse_attn=attention),
    )
    attention_hook = Mock(return_value=True)
    monkeypatch.setattr(patch, "_patch_prefill_head_tile", attention_hook)
    config = SimpleNamespace(
        hf_config=SimpleNamespace(
            architectures=["MiniMaxM3SparseForConditionalGeneration"]
        )
    )
    assert patch.apply_thead_patches(config)
    assert topk._topk_index_kernel is native_topk
    attention_hook.assert_called_once_with(attention)


class _Kernel:
    def __init__(self, tag):
        self.tag = tag
        self.calls = []

    def __getitem__(self, grid):
        def launch(*args, **kwargs):
            self.calls.append((grid, args, kwargs))
            return self.tag

        return launch
