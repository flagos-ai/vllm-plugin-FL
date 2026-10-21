# Copyright (c) 2026 BAAI. All rights reserved.

"""The real DSV4.1 attention constructor must select the EMPTY FP8 provider."""

import inspect
from types import SimpleNamespace

import pytest
import torch

from tests.utils.device_utils import get_backend, requires_cuda_hopper

pytestmark = pytest.mark.skipif(
    get_backend() != "nvidia", reason="NVIDIA sparse indexer"
)


@pytest.fixture
def config(tmp_path, monkeypatch):
    from vllm.config import (
        AttentionConfig,
        KernelConfig,
        ModelConfig,
        SchedulerConfig,
        VllmConfig,
        set_current_vllm_config,
    )
    from vllm.model_executor.custom_op import op_registry_oot
    from vllm.model_executor.layers import sparse_attn_indexer
    from vllm.models.deepseek_v4.nvidia.ops import o_proj
    from vllm.models.deepseek_v41 import attention as dsv41_attention
    from vllm.models.deepseek_v41.nvidia import flashmla
    from vllm.transformers_utils.configs.deepseek_v41 import DeepseekV41Config

    from vllm_fl.ops import nvidia_dsv41

    # A real HF/ModelConfig with bounded projection sizes. The indexer head,
    # quantization and topology contracts match the selected checkpoint.
    hf = DeepseekV41Config(
        architectures=["DeepseekV41ForCausalLM"],
        dtype="bfloat16",
        text_config={
            "hidden_size": 128,
            "num_hidden_layers": 3,
            "num_attention_heads": 8,
            "num_key_value_heads": 1,
            "vocab_size": 128,
            "q_lora_rank": 128,
            "o_lora_rank": 128,
            "head_dim": 512,
            "qk_rope_head_dim": 64,
            "o_groups": 1,
            "sliding_window": 128,
            "rms_norm_eps": 1e-20,
            "index_head_dim": 128,
            "index_n_heads": 32,
            "index_topk": 512,
            "compress_ratios": [2, 1, 1],
            "kv_source_layer_ids": [0, 1],
            "index_source_layer_ids": [0, 1, 2],
            "candidate_source_layer_id": 1,
            "candidate_topk_blocks": 16,
            "candidate_block_size": 8,
            "max_position_embeddings": 1024,
            "rope_theta": 10000.0,
            "compress_rope_theta": 10000.0,
            "rope_scaling": {"rope_type": "default"},
            "n_routed_experts": 4,
            "n_shared_experts": 1,
            "num_experts_per_tok": 2,
            "moe_intermediate_size": 128,
        },
    )
    hf.save_pretrained(tmp_path)
    cfg = VllmConfig(
        model_config=ModelConfig(
            model=str(tmp_path),
            tokenizer=str(tmp_path),
            skip_tokenizer_init=True,
            dtype="bfloat16",
            max_model_len=1024,
            enforce_eager=True,
        ),
        attention_config=AttentionConfig(indexer_kv_dtype="fp8"),
        scheduler_config=SchedulerConfig(
            max_model_len=1024,
            max_num_seqs=4,
            max_num_batched_tokens=32,
            is_encoder_decoder=False,
        ),
        kernel_config=KernelConfig(enable_jit_warmup=False),
    )
    cfg.compilation_config.custom_ops = ["all"]
    monkeypatch.setattr(nvidia_dsv41, "_installed", False)
    monkeypatch.delitem(op_registry_oot, "SparseAttnIndexer", raising=False)
    for module, name in (
        (sparse_attn_indexer, "fp8_fp4_mqa_logits"),
        (sparse_attn_indexer, "fp8_fp4_paged_mqa_logits"),
        (o_proj, "fp8_einsum"),
        (dsv41_attention, "fused_indexer_q_rope_quant"),
        (flashmla, "flash_mla_sparse_fwd"),
        (flashmla, "flash_mla_with_kvcache"),
        (flashmla.DeepseekSparseSWAFlashMLAMetadataBuilder, "build_tile_scheduler"),
    ):
        monkeypatch.setattr(module, name, getattr(module, name))
    with set_current_vllm_config(cfg):
        yield cfg
    op_registry_oot.pop("SparseAttnIndexer", None)


def _kwargs():
    return dict(
        k_cache=SimpleNamespace(prefix="test.indexer", kv_cache=torch.empty(0)),
        quant_block_size=128,
        scale_fmt="ue8m0",
        topk_tokens=512,
        head_dim=128,
        max_model_len=512,
        max_total_seq_len=512,
        topk_indices_buffer=torch.empty(4, 512, dtype=torch.int32),
        skip_k_cache_insert=True,
        compress_ratio=2,
    )


def test_constructor_signature_and_inherited_forward():
    from vllm.model_executor.layers.sparse_attn_indexer import SparseAttnIndexer

    from vllm_fl.ops.nvidia_sparse_indexer import NvidiaSparseAttnIndexer

    assert inspect.signature(NvidiaSparseAttnIndexer.__init__) == inspect.signature(
        SparseAttnIndexer.__init__
    )
    assert NvidiaSparseAttnIndexer.forward_cuda is SparseAttnIndexer.forward_cuda
    assert NvidiaSparseAttnIndexer.forward_native is SparseAttnIndexer.forward_native


@pytest.mark.parametrize("enabled", [False, True])
@pytest.mark.usefixtures("mock_hopper_capability")
def test_original_compilation_op_name_is_preserved(config, enabled):
    from vllm.model_executor.layers.sparse_attn_indexer import SparseAttnIndexer

    from vllm_fl.ops.nvidia_sparse_indexer import (
        NvidiaSparseAttnIndexer,
        register_nvidia_sparse_indexer,
    )

    assert register_nvidia_sparse_indexer()
    assert (
        NvidiaSparseAttnIndexer.name == SparseAttnIndexer.name == "sparse_attn_indexer"
    )
    config.compilation_config.custom_ops = (
        ["none", "+sparse_attn_indexer"] if enabled else ["all", "-sparse_attn_indexer"]
    )
    assert NvidiaSparseAttnIndexer.enabled() is enabled
    layer = SparseAttnIndexer(**_kwargs())
    expected = (
        SparseAttnIndexer.forward_cuda if enabled else SparseAttnIndexer.forward_native
    )
    assert layer._forward_method.__func__ is expected


@pytest.mark.parametrize(
    "field,value,error",
    [
        ("use_fp4_cache", True, "FP8 128-dim"),
        ("head_dim", 64, "FP8 128-dim"),
        ("quant_block_size", 64, "FP8 128-dim"),
        ("scale_fmt", "float32", "FP8 128-dim"),
        ("topk_tokens", 2048, "topk=512"),
        ("skip_k_cache_insert", False, "separately inserted"),
        ("compress_ratio", 0, "separately inserted"),
    ],
)
@pytest.mark.usefixtures("mock_hopper_capability")
def test_dsv41_rejects_unimplemented_indexer_contract(config, field, value, error):
    from vllm.model_executor.layers.sparse_attn_indexer import SparseAttnIndexer

    from vllm_fl.ops.nvidia_dsv41 import install_dsv41_empty_adapters

    assert install_dsv41_empty_adapters(config)
    kwargs = _kwargs()
    kwargs[field] = value
    with pytest.raises(NotImplementedError, match=error):
        SparseAttnIndexer(**kwargs)


@pytest.mark.parametrize("major", [8, 10, None])
def test_indexer_rejects_unsupported_hardware(config, monkeypatch, major):
    from vllm.model_executor.layers.sparse_attn_indexer import SparseAttnIndexer
    from vllm.platforms import current_platform
    from vllm.platforms.interface import DeviceCapability

    from vllm_fl.ops.nvidia_sparse_indexer import register_nvidia_sparse_indexer

    capability = None if major is None else DeviceCapability(major, 0)
    monkeypatch.setattr(
        current_platform, "get_device_capability", lambda device_id=0: capability
    )
    assert register_nvidia_sparse_indexer()
    with pytest.raises(NotImplementedError, match="Hopper FP8 128-dim"):
        SparseAttnIndexer(**_kwargs())


@pytest.mark.parametrize("native", [False, True])
def test_other_models_and_native_use_original_constructor(config, monkeypatch, native):
    from vllm.model_executor.layers.sparse_attn_indexer import SparseAttnIndexer

    from vllm_fl.ops import nvidia_sparse_indexer

    assert nvidia_sparse_indexer.register_nvidia_sparse_indexer()
    if native:
        monkeypatch.setattr(
            nvidia_sparse_indexer, "has_native_vllm_kernels", lambda: True
        )
    else:
        monkeypatch.setattr(
            config, "model_config", SimpleNamespace(architectures=["OtherForCausalLM"])
        )
    original_calls = []
    monkeypatch.setattr(
        SparseAttnIndexer, "__init__", lambda self, *args: original_calls.append(args)
    )
    SparseAttnIndexer(**_kwargs())
    assert len(original_calls) == 1


@pytest.mark.usefixtures("mock_hopper_capability")
def test_user_provider_preserved(config, monkeypatch):
    from vllm.model_executor.custom_op import op_registry_oot
    from vllm.model_executor.layers.sparse_attn_indexer import SparseAttnIndexer

    from vllm_fl.ops.nvidia_dsv41 import install_dsv41_empty_adapters

    class UserIndexer(SparseAttnIndexer):
        pass

    monkeypatch.setitem(op_registry_oot, "SparseAttnIndexer", UserIndexer)
    assert not install_dsv41_empty_adapters(config)
    assert op_registry_oot["SparseAttnIndexer"] is UserIndexer


@pytest.mark.gpu
@requires_cuda_hopper
def test_real_dsv41_attention_constructs_compressed_and_candidate_indexers(
    config, tmp_path
):
    from vllm.distributed import init_distributed_environment, initialize_model_parallel
    from vllm.distributed.parallel_state import (
        destroy_distributed_environment,
        destroy_model_parallel,
    )
    from vllm.model_executor.layers.sparse_attn_indexer import SparseAttnIndexer
    from vllm.models.deepseek_v41.nvidia.flashmla import DeepseekV4FlashMLAAttention
    from vllm.utils.import_utils import has_deep_gemm
    from vllm.utils.torch_utils import set_default_torch_dtype
    from vllm.v1.attention.ops.flashmla import is_flashmla_sparse_supported

    from vllm_fl.ops.nvidia_dsv41 import install_dsv41_empty_adapters
    from vllm_fl.ops.nvidia_sparse_indexer import NvidiaSparseAttnIndexer

    before = has_deep_gemm(), is_flashmla_sparse_supported()
    assert install_dsv41_empty_adapters(config)
    owned_group = not torch.distributed.is_initialized()
    try:
        if owned_group:
            init_distributed_environment(
                world_size=1,
                rank=0,
                local_rank=0,
                distributed_init_method=(tmp_path / "dist").as_uri(),
                backend="nccl",
            )
            initialize_model_parallel(
                tensor_model_parallel_size=1, pipeline_model_parallel_size=1
            )
        with torch.device("cuda"), set_default_torch_dtype(torch.bfloat16):
            topk = torch.empty(4, 512, dtype=torch.int32)
            candidates = torch.empty(4, 16, dtype=torch.int32)
            layers = [
                DeepseekV4FlashMLAAttention(
                    config,
                    prefix=f"model.layers.{i}.self_attn",
                    topk_indices_buffer=topk,
                    candidate_block_buffer=candidates,
                )
                for i in range(3)
            ]
        for i, layer in enumerate(layers):
            indexer = layer.indexer.indexer_op
            assert type(indexer) is NvidiaSparseAttnIndexer
            assert indexer._forward_method.__func__ is SparseAttnIndexer.forward_cuda
            assert indexer.compress_ratio == (2 if i == 0 else 1)
            assert indexer.skip_k_cache_insert
            assert not indexer.use_fp4_cache
            assert indexer.candidate_blocks is (None if i == 0 else candidates)
            assert indexer.candidate_write == (i == 1)
            assert (indexer.dcp_world_size, indexer.dcp_rank, indexer.use_pcp) == (
                1,
                0,
                False,
            )
            assert layer.swa_bytes_per_token == 584
            assert layer.kv_cache_dtype == "fp8_ds_mla"
        from tests.unit_tests.ops.test_nvidia_dsv41_indexer_q import (
            assert_query_equal,
            query_reference,
        )

        # Exercise the actual upstream forward_q, including its real replicated
        # projection and the installer's scoped module binding. Identity blocks
        # make the linear result independently exact at BF16 precision.
        qr_cpu = torch.linspace(-2, 2, 3 * 128).reshape(3, 128).bfloat16()
        qr = qr_cpu.cuda()
        positions = torch.tensor([0, 7, 511], dtype=torch.int64, device="cuda")
        scores = torch.linspace(-1, 1, 3 * 32).reshape(3, 32).bfloat16().cuda()
        for layer in layers:
            indexer = layer.indexer
            with torch.no_grad():
                indexer.wq_b.weight.copy_(torch.eye(128).repeat(32, 1))
                query, scale, weights = indexer.forward_q(
                    qr, None, scores, positions, layer.indexer_rotary_emb
                )
            assert scale is None
            reference = query_reference(
                positions,
                qr_cpu[:, None, :].expand(3, 32, 128),
                layer.indexer_rotary_emb.cos_sin_cache,
                scores,
                indexer.softmax_scale,
                indexer.n_head**-0.5,
            )
            assert_query_equal((query, weights), reference)
        assert layers[2].indexer.k_cache is layers[1].indexer.k_cache
        assert (has_deep_gemm(), is_flashmla_sparse_supported()) == before
    finally:
        if owned_group:
            destroy_model_parallel()
            destroy_distributed_environment()
