# SPDX-License-Identifier: Apache-2.0
"""Cross-layer regressions from the GLM5-Next review at af0e829."""

import os
import subprocess
import sys
from types import SimpleNamespace

import pytest
import torch

from vllm_fl.kernels.glm5_next.indexer_backend import (
    Glm5NextIndexerBackend,
    _dequantize_grouped,
)
from vllm_fl.patches import glm5_next_v024 as patch
from vllm_fl.runtime.model_policy import ModelPolicyError, validate_model_config


@pytest.mark.parametrize(
    "arch", ["Glm5NextForCausalLM", "Glm5NextForConditionalGeneration"]
)
@pytest.mark.parametrize(
    "mode", ["pp", "spec", "eplb", "quant", "nested_quant", "runtime_quant"]
)
def test_unsupported_modes_rejected_at_early_and_final_config(arch, mode):
    hf = SimpleNamespace(architectures=[arch], model_type="glm5_next")
    text = SimpleNamespace(model_type="glm5_next_text")
    config = SimpleNamespace(
        model_config=SimpleNamespace(hf_config=hf, hf_text_config=text),
        parallel_config=SimpleNamespace(pipeline_parallel_size=1, enable_eplb=False),
        speculative_config=None,
    )
    if mode == "pp":
        config.parallel_config.pipeline_parallel_size = 2
    elif mode == "spec":
        config.speculative_config = SimpleNamespace(num_speculative_tokens=3)
    elif mode == "eplb":
        config.parallel_config.enable_eplb = True
    elif mode == "quant":
        config.model_config.quantization = "fp8"
    elif mode == "nested_quant":
        text.quantization_config = {"quant_method": "fp8"}
    else:
        config.quant_config = object()
    patch.apply_glm5_next_v024_patches()
    # The early hook must fail before accessing any hybrid/cache settings.
    with pytest.raises(ModelPolicyError):
        patch.Glm5NextForCausalLMConfig.verify_and_update_config(config)
    with pytest.raises(ModelPolicyError):
        validate_model_config(config)


@pytest.mark.parametrize("tokens", [128, 512, 2051])
@pytest.mark.parametrize("missing_op", [True, False])
def test_mqa_fallback_uses_real_prefill_scale_shape(tokens, missing_op, monkeypatch):
    backend = Glm5NextIndexerBackend()
    monkeypatch.setattr(backend, "is_nvidia", False)

    def reject(*args, **kwargs):
        raise NotImplementedError("unsupported shape")

    monkeypatch.setattr(backend, "_flag", lambda *args: None if missing_op else reject)
    # This is the caller's squeezed [N] scale, including N == head_dim where
    # accidental broadcasting used to succeed with the wrong values.
    keys = torch.ones(tokens, 128)
    scales = torch.arange(1, tokens + 1, dtype=torch.float32)
    query = torch.ones(2, 3, 128)
    weights = torch.ones(2, 3)
    starts, ends = torch.tensor([0, 1]), torch.tensor([tokens, tokens - 1])
    actual = backend.mqa_logits((query, None), (keys, scales), weights, starts, ends)
    expected = (384 * scales).expand(2, -1).clone()
    expected[1, 0] = expected[1, -1] = -torch.inf
    torch.testing.assert_close(actual, expected)


def test_grouped_scales_and_invalid_shape():
    values = torch.ones(2, 4)
    scales = torch.tensor([[2.0, 3.0], [4.0, 5.0]])
    torch.testing.assert_close(
        _dequantize_grouped(values, scales),
        torch.tensor([[2.0, 2.0, 3.0, 3.0], [4.0, 4.0, 5.0, 5.0]]),
    )
    with pytest.raises(ValueError, match="Scale shape"):
        _dequantize_grouped(values, torch.ones(3))


@pytest.mark.parametrize("provider", ["flaggems", "auto"])
def test_fresh_process_attention_dispatch_and_vision_import_isolation(provider):
    code = r"""
from types import SimpleNamespace
import vllm_fl
vllm_fl.register_model()
from vllm_fl.activation import activate_for_model
from vllm_fl.dispatch import SelectionPolicy, set_global_policy, get_default_manager
from vllm_fl.kernels.glm5_next import provider
from vllm_fl.patches import glm5_next_v024 as patch
from vllm_fl.platform import PlatformFL
from vllm_fl.dispatch.backends.flaggems.flaggems import FlagGemsBackend

# Force the auto -> portable condition without replacing dispatch or logger.
provider._has_nvidia_reference_kernels = lambda: False
set_global_policy(SelectionPolicy.from_dict(per_op_order={
    "attention_backend": ["flagos", "vendor:cuda"],
}))
from vllm_fl.dispatch.backends.vendor.cuda.cuda import CudaBackend
vendor = CudaBackend()
for sparse in (False, True):
    get_default_manager().clear_failed_impls("attention_backend")
    selector = SimpleNamespace(use_mla=True, use_sparse=sparse)
    expected = vendor.attention_backend(use_mla=True, use_sparse=sparse)
    assert PlatformFL.get_attn_backend_cls(None, selector) == expected
get_default_manager().clear_failed_impls("attention_backend")
selector = SimpleNamespace(use_mla=False, use_sparse=False)
assert PlatformFL.get_attn_backend_cls(None, selector) == FlagGemsBackend().attention_backend()

import vllm.model_executor.layers.attention.mm_encoder_attention as mm
import vllm.v1.attention.backends.fa_utils as fa
import vllm.v1.attention.ops.vit_attn_wrappers as vit
import vllm.vllm_flash_attn as native_fa
missing = object()
owners = (mm, fa, vit, native_fa)
before = [getattr(m, "flash_attn_varlen_func", missing) for m in owners]
import vllm_fl.models.glm5_next_multimodal
assert all(getattr(m, "flash_attn_varlen_func", missing) is old
           for m, old in zip(owners, before))

config = SimpleNamespace(model_config=SimpleNamespace(
    hf_config=SimpleNamespace(model_type="glm5_next_text")))
activate_for_model(config)
for sparse, expected in ((False, "MLAFLBackend"), (True, "FlagGemsSparseMLABackend")):
    selector = SimpleNamespace(use_mla=True, use_sparse=sparse)
    assert PlatformFL.get_attn_backend_cls(None, selector).endswith(expected)
print("review contracts passed")
"""
    env = dict(os.environ, VLLM_FL_GLM5_PROVIDER=provider, VLLM_PLUGINS="fl")
    result = subprocess.run(
        [sys.executable, "-c", code],
        env=env,
        text=True,
        capture_output=True,
        timeout=180,
    )
    assert result.returncode == 0, result.stdout + result.stderr


def test_private_vision_adapter_preserves_public_fa(monkeypatch):
    import flag_gems
    from vllm_fl.kernels.glm5_next.vision_attention import _vision_attention

    calls = []

    def fa(query, key, value, **kwargs):
        calls.append(kwargs)
        return query.clone(), torch.ones(1)

    monkeypatch.setattr(flag_gems, "flash_attn_varlen_func", fa)
    q = torch.randn(1, 7, 2, 8)
    cu = torch.tensor([0, 3, 7], dtype=torch.int32)
    torch.testing.assert_close(_vision_attention(q, q, q, cu, 0.5), q)
    assert calls[0]["fa_version"] == 2
    assert calls[0]["max_seqlen_q"] == 7
    assert calls[0]["cu_seqlens_q"] is cu


def test_subsecond_video_keeps_a_temporal_patch():
    from vllm_fl.transformers_utils.processors.glm5_next import glm_sample_frame_indices

    assert glm_sample_frame_indices(12, 30.0, 0.4) == [0, 0]
    assert glm_sample_frame_indices(1, 30.0, 1 / 30) == [0, 0]
