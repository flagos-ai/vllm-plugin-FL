# SPDX-License-Identifier: Apache-2.0
"""Cross-layer regressions from the GLM5-Next review at af0e829."""

import os
import subprocess
import sys
from types import SimpleNamespace

import pytest
import torch

from vllm_fl.kernels.glm5_next.indexer_backend import Glm5NextIndexerBackend
from vllm_fl.patches import glm5_next_runtime as patch
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
    patch.register_glm5_next_support()
    # The early hook must fail before accessing any hybrid/cache settings.
    with pytest.raises(ModelPolicyError):
        patch.Glm5NextForCausalLMConfig.verify_and_update_config(config)
    with pytest.raises(ModelPolicyError):
        validate_model_config(config)


@pytest.mark.parametrize("tokens", [128, 512, 2051])
def test_mqa_adapter_preserves_prefill_scale_shape(tokens, monkeypatch):
    backend = Glm5NextIndexerBackend()
    monkeypatch.setattr(backend, "is_nvidia", False)
    calls = []
    sentinel = object()

    def public_mqa(*args, **kwargs):
        calls.append((args, kwargs))
        return sentinel

    monkeypatch.setattr(backend, "_flag", lambda *args: public_mqa)
    # Stub the public operator boundary, retaining the squeezed [N] scale ABI.
    keys = torch.ones(tokens, 128)
    scales = torch.arange(1, tokens + 1, dtype=torch.float32)
    args = (
        (torch.ones(2, 3, 128), None),
        (keys, scales),
        torch.ones(2, 3),
        torch.tensor([0, 1]),
        torch.tensor([tokens, tokens - 1]),
    )
    assert backend.mqa_logits(*args) is sentinel
    assert len(calls) == 1
    assert calls[0][0][1][0] is keys
    assert calls[0][0][1][1] is scales
    assert scales.shape == (tokens,)
    assert calls[0][1] == {}


@pytest.mark.parametrize("missing_op", [False, True])
def test_mqa_unavailable_or_unsupported_has_no_numeric_reference(
    monkeypatch, missing_op
):
    backend = Glm5NextIndexerBackend()
    monkeypatch.setattr(backend, "is_nvidia", False)
    calls = []

    def unsupported(*args, **kwargs):
        calls.append("public")
        raise NotImplementedError("unsupported scale layout")

    monkeypatch.setattr(
        backend, "_flag", lambda *args: None if missing_op else unsupported
    )
    expected = RuntimeError if missing_op else NotImplementedError
    with pytest.raises(expected):
        backend.mqa_logits(
            (torch.ones(2, 3, 128), None),
            (torch.ones(128, 128), torch.ones(128)),
            torch.ones(2, 3),
            torch.zeros(2, dtype=torch.int32),
            torch.full((2,), 128, dtype=torch.int32),
        )
    assert calls == ([] if missing_op else ["public"])


def test_fresh_process_attention_dispatch_and_vision_import_isolation():
    code = r"""
from types import SimpleNamespace
import re
import pytest
import vllm_fl
vllm_fl.register_model()
from vllm_fl.activation import activate_for_model
from vllm_fl.dispatch import SelectionPolicy, set_global_policy, get_default_manager
from vllm_fl.kernels.glm5_next import provider
from vllm_fl.patches import glm5_next_runtime as patch
from vllm_fl.platform import PlatformFL
from vllm_fl.dispatch.backends.flaggems.flaggems import FlagGemsBackend

# Select the portable capability condition without replacing dispatch.
provider._has_nvidia_reference_kernels = lambda: False
set_global_policy(SelectionPolicy.from_dict(per_op_order={
    "attention_backend": ["flagos", "vendor:cuda"],
}))
from vllm.platforms import current_platform
from vllm_fl.dispatch.backends.vendor.cuda.cuda import CudaBackend
vendor = CudaBackend()
for sparse in (False, True):
    get_default_manager().clear_failed_impls("attention_backend")
    selector = SimpleNamespace(use_mla=True, use_sparse=sparse)
    if current_platform.is_cuda():
        expected = vendor.attention_backend(use_mla=True, use_sparse=sparse)
        assert PlatformFL.get_attn_backend_cls(None, selector) == expected
    else:
        # The forced CUDA fallback is unavailable here. Generic MLA must fail
        # before activation rather than borrow another model's portable plan.
        with pytest.raises(RuntimeError, match="requires a model runtime plan"):
            PlatformFL.get_attn_backend_cls(None, selector)
get_default_manager().clear_failed_impls("attention_backend")
selector = SimpleNamespace(use_mla=False, use_sparse=False)
# Compare the public candidate's capability contract on the actual platform.
# Some non-NVIDIA vendors provide this generic backend; others reject CUDA.
try:
    expected = FlagGemsBackend().attention_backend()
except RuntimeError as error:
    with pytest.raises(RuntimeError, match=re.escape(str(error))):
        PlatformFL.get_attn_backend_cls(None, selector)
else:
    assert PlatformFL.get_attn_backend_cls(None, selector) == expected

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
    env = dict(os.environ, VLLM_PLUGINS="fl")
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
