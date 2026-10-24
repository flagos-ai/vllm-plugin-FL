"""Model semantics do not install hardware launch patches or wrap constructors."""

import os
import sys
from types import SimpleNamespace

import pytest
import torch

from vllm_fl.patches import minimax_m3


def test_model_metadata_keeps_constructor_and_other_maps():
    class Model:
        packed_modules_mapping = {"other": ["part"]}

        def __init__(self):
            self.ready = True

    original = Model.__init__
    assert minimax_m3.install_minimax_m3_metadata((Model,))
    assert not minimax_m3.install_minimax_m3_metadata((Model,))
    assert Model.__init__ is original and Model().ready
    assert Model.packed_modules_mapping["other"] == ["part"]
    assert Model.packed_modules_mapping["qkv_proj"] == [
        "q_proj",
        "k_proj",
        "v_proj",
        "index_q_proj",
        "index_k_proj",
    ]
    assert Model.packed_modules_mapping["gate_up_proj"] == ["gate_proj", "up_proj"]


def test_model_module_absence_is_optional(monkeypatch):
    monkeypatch.setitem(sys.modules, "vllm.models.minimax_m3", None)
    assert minimax_m3.apply_minimax_m3_patches() is False


def test_missing_native_preprocess_calls_library_directly_and_is_idempotent(
    monkeypatch,
):
    from unittest.mock import Mock

    import vllm

    import vllm_fl.dispatch

    def native(*args, **kwargs):
        raise AssertionError("missing native op should not run")

    ops = SimpleNamespace(fused_minimax_m3_qknorm_rope_kv_insert=native)
    monkeypatch.setattr(vllm, "_custom_ops", ops)
    monkeypatch.setattr(
        minimax_m3, "torch", SimpleNamespace(ops=SimpleNamespace(_C=SimpleNamespace()))
    )
    monkeypatch.setattr(
        vllm_fl.dispatch,
        "get_default_manager",
        lambda: (_ for _ in ()).throw(AssertionError("unexpected dispatch")),
    )
    operator = Mock(return_value=None)
    monkeypatch.setitem(
        sys.modules,
        "flaggems_vllm",
        SimpleNamespace(fused_minimax_m3_qknorm_rope_kv_insert=operator),
    )
    assert minimax_m3.patch_minimax_m3_fused_preprocess()
    saved = ops.fused_minimax_m3_qknorm_rope_kv_insert
    assert not minimax_m3.patch_minimax_m3_fused_preprocess()
    assert ops.fused_minimax_m3_qknorm_rope_kv_insert is saved
    assert saved("qkv", positions="positions") is None
    operator.assert_called_once_with("qkv", positions="positions")


def test_native_preprocess_keeps_its_original_implementation(monkeypatch):
    import vllm

    native = lambda *args, **kwargs: None
    ops = SimpleNamespace(fused_minimax_m3_qknorm_rope_kv_insert=native)
    monkeypatch.setattr(vllm, "_custom_ops", ops)
    monkeypatch.setattr(
        minimax_m3,
        "torch",
        SimpleNamespace(
            ops=SimpleNamespace(
                _C=SimpleNamespace(fused_minimax_m3_qknorm_rope_kv_insert=object())
            )
        ),
    )
    monkeypatch.setitem(sys.modules, "flaggems_vllm", None)
    assert not minimax_m3.patch_minimax_m3_fused_preprocess()
    assert ops.fused_minimax_m3_qknorm_rope_kv_insert is native


def test_vision_mapping_keeps_three_parts_without_mutating_text_config():
    class Config:
        packed_modules_mapping = minimax_m3.PACKED_MODULES.copy()

        def get_scheme_dict(self, layer, layer_name=None):
            return self.packed_modules_mapping["qkv_proj"]

    original = Config.get_scheme_dict
    q, other = Config(), Config()
    assert minimax_m3.install_minimax_m3_vision_mapping(q)
    assert not minimax_m3.install_minimax_m3_vision_mapping(q)
    assert Config.get_scheme_dict is original
    assert q.get_scheme_dict(
        None, "vision_tower.vision_model.encoder.layers.0.self_attn.qkv_proj"
    ) == ["q_proj", "k_proj", "v_proj"]
    assert (
        q.get_scheme_dict(None, "language_model.model.layers.0.self_attn.qkv_proj")
        == (minimax_m3.PACKED_MODULES["qkv_proj"])
    )
    assert (
        other.get_scheme_dict(None, "vision_tower.layer.qkv_proj")
        == (minimax_m3.PACKED_MODULES["qkv_proj"])
    )
    assert q.packed_modules_mapping == minimax_m3.PACKED_MODULES


def test_preparation_leaves_other_architectures_untouched(monkeypatch):
    def unexpected(*args):
        raise AssertionError("unrelated model must not install MiniMax hooks")

    monkeypatch.setattr(minimax_m3, "install_minimax_m3_vision_mapping", unexpected)
    monkeypatch.setattr(minimax_m3, "patch_minimax_m3_fused_preprocess", unexpected)
    config = SimpleNamespace(
        model_config=SimpleNamespace(
            hf_config=SimpleNamespace(architectures=["OtherModel"])
        ),
        quant_config=object(),
    )
    minimax_m3.prepare_minimax_m3_model(config)


def test_metadata_registration_installs_no_execution_hooks(monkeypatch):
    class Model:
        pass

    monkeypatch.setitem(
        sys.modules,
        "vllm.models.minimax_m3",
        SimpleNamespace(
            MiniMaxM3SparseForCausalLM=Model,
            MiniMaxM3SparseForConditionalGeneration=Model,
        ),
    )

    def unexpected(*args):
        raise AssertionError("model inspection must not install execution hooks")

    monkeypatch.setattr(minimax_m3, "install_minimax_m3_vision_mapping", unexpected)
    monkeypatch.setattr(minimax_m3, "patch_minimax_m3_fused_preprocess", unexpected)
    assert minimax_m3.apply_minimax_m3_patches()


def test_prepare_binds_only_selected_quant_config(monkeypatch):
    from vllm.model_executor.layers.quantization.compressed_tensors.compressed_tensors import (
        CompressedTensorsConfig,
    )

    quant = object.__new__(CompressedTensorsConfig)
    quant.packed_modules_mapping = minimax_m3.PACKED_MODULES.copy()
    quant.ignore = ["re:vision_tower\\..*\\.(q|k|v)_proj"]
    quant.target_scheme_map = {}
    original = CompressedTensorsConfig.get_scheme_dict
    bridges = []
    monkeypatch.setattr(
        minimax_m3, "patch_minimax_m3_fused_preprocess", lambda: bridges.append(True)
    )
    config = SimpleNamespace(
        model_config=SimpleNamespace(
            hf_config=SimpleNamespace(
                architectures=["MiniMaxM3SparseForConditionalGeneration"]
            )
        ),
        quant_config=quant,
    )
    minimax_m3.prepare_minimax_m3_model(config)
    assert bridges == [True]
    assert CompressedTensorsConfig.get_scheme_dict is original
    assert quant.get_scheme_dict.__self__ is quant
    name = "vision_tower.vision_model.encoder.layers.0.self_attn.qkv_proj"
    layer = torch.nn.Linear(2, 2)
    with pytest.raises(ValueError, match="different quantization schemes"):
        original(quant, layer, name)
    assert quant.get_scheme_dict(layer, name) is None
    assert quant.packed_modules_mapping == minimax_m3.PACKED_MODULES


HEAD_DIM = 128


ROTARY_DIM = 64


def _make_cos_sin(max_pos=64):
    inv_freq = 1.0 / 5000000.0 ** (
        torch.arange(0, ROTARY_DIM, 2, device="cuda").float() / ROTARY_DIM
    )
    positions = torch.arange(max_pos, device="cuda").float()
    frequencies = torch.einsum("i,j->ij", positions, inv_freq)
    return torch.cat((frequencies.cos(), frequencies.sin()), dim=-1).bfloat16()


def _reference(x, weight, positions, cos_sin):
    x_float = x.float()
    x_float = x_float * torch.rsqrt(x_float.square().mean(dim=-1, keepdim=True) + 1e-06)
    x_float = x_float * (1.0 + weight.float())
    half = ROTARY_DIM // 2
    selected = cos_sin[positions].float()
    cos = selected[:, None, :half]
    sin = selected[:, None, half:]
    first = x_float[..., :half]
    second = x_float[..., half:ROTARY_DIM]
    out = x_float.clone()
    out[..., :half] = first * cos - second * sin
    out[..., half:ROTARY_DIM] = second * cos + first * sin
    return out.bfloat16()


def _patched_op():
    from vllm import _custom_ops as ops
    from vllm.plugins import load_general_plugins

    load_general_plugins()
    minimax_m3.prepare_minimax_m3_model(
        SimpleNamespace(
            model_config=SimpleNamespace(
                hf_config=SimpleNamespace(
                    architectures=["MiniMaxM3SparseForConditionalGeneration"]
                )
            ),
            quant_config=None,
        )
    )
    op = ops.fused_minimax_m3_qknorm_rope_kv_insert
    assert getattr(op, "_vllm_fl_minimax_m3_patch", False)
    return op


@pytest.mark.gpu
@pytest.mark.skipif(
    not torch.cuda.is_available() or "PPU_SDK" not in os.environ,
    reason="T-Head PPU is not available",
)
class TestTheadMiniMaxPreprocess:
    def test_patch_does_not_synthesize_private_vllm_abi(self):
        native_op_existed = hasattr(
            torch.ops._C, "fused_minimax_m3_qknorm_rope_kv_insert"
        )
        _patched_op()
        assert (
            hasattr(torch.ops._C, "fused_minimax_m3_qknorm_rope_kv_insert")
            is native_op_existed
        )

    def test_dense_matches_reference(self):
        fused_preprocess = _patched_op()
        torch.manual_seed(0)
        tokens, num_heads, num_kv_heads = (7, 8, 2)
        q_size = num_heads * HEAD_DIM
        kv_size = num_kv_heads * HEAD_DIM
        qkv = torch.randn(
            tokens, q_size + 2 * kv_size, device="cuda", dtype=torch.bfloat16
        )
        original = qkv.clone()
        q_weight = torch.randn(HEAD_DIM, device="cuda", dtype=torch.bfloat16) * 0.1
        k_weight = torch.randn(HEAD_DIM, device="cuda", dtype=torch.bfloat16) * 0.1
        cos_sin = _make_cos_sin()
        positions = torch.arange(tokens, device="cuda", dtype=torch.int64)
        fused_preprocess(
            qkv,
            q_weight,
            k_weight,
            cos_sin,
            positions,
            num_heads,
            num_kv_heads,
            ROTARY_DIM,
            1e-06,
            None,
            None,
            0,
            None,
            None,
            None,
            None,
            0,
            None,
            None,
            "auto",
        )
        q, k, value = qkv.split((q_size, kv_size, kv_size), dim=-1)
        q0, k0, value0 = original.split((q_size, kv_size, kv_size), dim=-1)
        q_ref = _reference(
            q0.view(tokens, num_heads, HEAD_DIM), q_weight, positions, cos_sin
        ).view(tokens, q_size)
        k_ref = _reference(
            k0.view(tokens, num_kv_heads, HEAD_DIM), k_weight, positions, cos_sin
        ).view(tokens, kv_size)
        torch.testing.assert_close(q, q_ref, rtol=0.01, atol=0.01)
        torch.testing.assert_close(k, k_ref, rtol=0.01, atol=0.01)
        torch.testing.assert_close(value, value0, rtol=0, atol=0)

    def test_sparse_outputs_and_cache_inserts(self):
        fused_preprocess = _patched_op()
        torch.manual_seed(1)
        tokens = 7
        num_heads, num_kv_heads, num_index_heads = (8, 2, 2)
        q_size = num_heads * HEAD_DIM
        kv_size = num_kv_heads * HEAD_DIM
        index_q_size = num_index_heads * HEAD_DIM
        sizes = (q_size, kv_size, kv_size, index_q_size, HEAD_DIM)
        qkv = torch.randn(tokens, sum(sizes), device="cuda", dtype=torch.bfloat16)
        original = qkv.clone()
        weights = [
            torch.randn(HEAD_DIM, device="cuda", dtype=torch.bfloat16) * 0.1
            for _ in range(4)
        ]
        cos_sin = _make_cos_sin()
        positions = torch.arange(tokens, device="cuda", dtype=torch.int64)
        block_size = 128
        slots = torch.tensor([2, 4, -1, 130, 131, 5, 6], device="cuda")
        index_slots = torch.tensor([9, 8, -1, 133, 134, 7, 3], device="cuda")
        kv_cache = torch.zeros(
            2,
            2,
            block_size,
            num_kv_heads,
            HEAD_DIM,
            device="cuda",
            dtype=torch.bfloat16,
        )
        index_cache = torch.zeros(
            2, block_size, HEAD_DIM, device="cuda", dtype=torch.bfloat16
        )
        q_out = torch.empty(tokens, q_size, device="cuda", dtype=torch.bfloat16)
        index_q_out = torch.empty(
            tokens, index_q_size, device="cuda", dtype=torch.bfloat16
        )
        fused_preprocess(
            qkv,
            weights[0],
            weights[1],
            cos_sin,
            positions,
            num_heads,
            num_kv_heads,
            ROTARY_DIM,
            1e-06,
            weights[2],
            weights[3],
            num_index_heads,
            slots,
            index_slots,
            kv_cache,
            index_cache,
            block_size,
            q_out,
            index_q_out,
            "auto",
        )
        _, k, value, _, index_k = qkv.split(sizes, dim=-1)
        q0, k0, value0, index_q0, index_k0 = original.split(sizes, dim=-1)
        expected = (
            _reference(
                q0.view(tokens, num_heads, HEAD_DIM), weights[0], positions, cos_sin
            ),
            _reference(
                k0.view(tokens, num_kv_heads, HEAD_DIM), weights[1], positions, cos_sin
            ),
            _reference(
                index_q0.view(tokens, num_index_heads, HEAD_DIM),
                weights[2],
                positions,
                cos_sin,
            ),
            _reference(
                index_k0.view(tokens, 1, HEAD_DIM), weights[3], positions, cos_sin
            ),
        )
        torch.testing.assert_close(
            q_out, expected[0].view_as(q_out), rtol=0.01, atol=0.01
        )
        torch.testing.assert_close(k, expected[1].view_as(k), rtol=0.01, atol=0.01)
        torch.testing.assert_close(
            index_q_out, expected[2].view_as(index_q_out), rtol=0.01, atol=0.01
        )
        torch.testing.assert_close(
            index_k, expected[3].view_as(index_k), rtol=0.01, atol=0.01
        )
        torch.testing.assert_close(value, value0, rtol=0, atol=0)
        for token, slot in enumerate(slots.tolist()):
            if slot < 0:
                continue
            block, offset = divmod(slot, block_size)
            torch.testing.assert_close(
                kv_cache[block, 0, offset],
                k[token].view(num_kv_heads, HEAD_DIM),
                rtol=0,
                atol=0,
            )
            torch.testing.assert_close(
                kv_cache[block, 1, offset],
                value0[token].view(num_kv_heads, HEAD_DIM),
                rtol=0,
                atol=0,
            )
        for token, slot in enumerate(index_slots.tolist()):
            if slot >= 0:
                block, offset = divmod(slot, block_size)
                torch.testing.assert_close(
                    index_cache[block, offset], index_k[token], rtol=0, atol=0
                )

    def test_dense_cuda_graph_replay(self):
        fused_preprocess = _patched_op()
        if not hasattr(torch.cuda, "CUDAGraph"):
            pytest.skip("CUDA-compatible graph capture is unavailable")
        torch.manual_seed(2)
        tokens, num_heads, num_kv_heads = (4, 4, 2)
        q_size = num_heads * HEAD_DIM
        kv_size = num_kv_heads * HEAD_DIM
        qkv = torch.randn(
            tokens, q_size + 2 * kv_size, device="cuda", dtype=torch.bfloat16
        )
        q_weight = torch.randn(HEAD_DIM, device="cuda", dtype=torch.bfloat16) * 0.1
        k_weight = torch.randn(HEAD_DIM, device="cuda", dtype=torch.bfloat16) * 0.1
        cos_sin = _make_cos_sin()
        positions = torch.arange(tokens, device="cuda", dtype=torch.int64)

        def invoke():
            fused_preprocess(
                qkv,
                q_weight,
                k_weight,
                cos_sin,
                positions,
                num_heads,
                num_kv_heads,
                ROTARY_DIM,
                1e-06,
                None,
                None,
                0,
                None,
                None,
                None,
                None,
                0,
                None,
                None,
                "auto",
            )

        initial = qkv.clone()
        invoke()
        torch.cuda.synchronize()
        qkv.copy_(initial)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            invoke()
        torch.cuda.synchronize()
        replay_input = torch.randn_like(qkv)
        qkv.copy_(replay_input)
        graph.replay()
        torch.cuda.synchronize()
        q, k, value = qkv.split((q_size, kv_size, kv_size), dim=-1)
        q0, k0, value0 = replay_input.split((q_size, kv_size, kv_size), dim=-1)
        q_ref = _reference(
            q0.view(tokens, num_heads, HEAD_DIM), q_weight, positions, cos_sin
        ).view(tokens, q_size)
        k_ref = _reference(
            k0.view(tokens, num_kv_heads, HEAD_DIM), k_weight, positions, cos_sin
        ).view(tokens, kv_size)
        torch.testing.assert_close(q, q_ref, rtol=0.01, atol=0.01)
        torch.testing.assert_close(k, k_ref, rtol=0.01, atol=0.01)
        torch.testing.assert_close(value, value0, rtol=0, atol=0)
