# Copyright 2026 FlagOS Contributors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""CPU tests for the Qwen3.8-Flash-Next / Qwen4Exp Day0 patch module.

Each installer is driven against the real plugin classes with the vllm
registries, the v0.24 hybrid-state class, the PLE builder module, and the
compilation config replaced by recorders.  ``verify_and_update_config`` runs
on plain config namespaces; the inherited Qwen3.5 verifier (an external vLLM
boundary) is recorded so the test isolates this module's contract.
"""

from __future__ import annotations

import sys
import types
from types import SimpleNamespace

import pytest
import torch

# The shim import is load-bearing: it must precede vllm_fl imports.
# isort: split
from . import _flaggems_shim  # noqa: F401

# isort: split
import vllm_fl.patches.qwen3_8_flash_next as patch_module

_ARCH = patch_module.Qwen3_8FlashNextForConditionalGenerationConfig
_CAUSAL_ARCH = patch_module.Qwen3_8FlashNextForCausalLMConfig


@pytest.fixture
def base_verify(monkeypatch):
    """Record the inherited Qwen3.5 verifier call (external boundary)."""
    calls = []

    def record(vllm_config):
        calls.append(vllm_config)

    monkeypatch.setattr(
        patch_module.Qwen3_5ForConditionalGenerationConfig,
        "verify_and_update_config",
        staticmethod(record),
    )
    return calls


def _make_config():
    """Minimal config namespace mirroring the fields the verifier reads."""
    text = SimpleNamespace(
        mamba_ssm_dtype=None,
        hc_count=4,
        ple_layer_ids=[],
        indexer_n_heads=None,
    )
    return SimpleNamespace(
        model_config=SimpleNamespace(
            hf_text_config=text,
            hf_config=None,
            multimodal_config=None,
        ),
        cache_config=SimpleNamespace(mamba_ssm_cache_dtype="auto"),
        parallel_config=SimpleNamespace(
            enable_dbo=False, ubatch_size=1, pipeline_parallel_size=1
        ),
        speculative_config=None,
    )


# ---------------------------------------------------------------------------
# Static helpers
# ---------------------------------------------------------------------------


def test_flag_gems_blacklist_is_exact_for_nvidia():
    """The NVIDIA blacklist is the measured Day0 list in order."""
    assert _ARCH.get_flag_gems_blacklist(vendor_name="nvidia") == [
        "index_select",
        "repeat_interleave_tensor",
        "repeat_interleave_self_tensor",
        "linear",
        "mm",
        "mm_out",
        "addmm",
        "addmm_out",
        "addmm_",
        "addmm_dtype",
        "addmm_dtype_out",
        "sort",
        "sort_stable",
    ]


@pytest.mark.parametrize("vendor", [None, "amd", "musa", "ascend"])
def test_flag_gems_blacklist_is_empty_for_other_vendors(vendor):
    """Non-NVIDIA vendors keep the generic FlagGems policy."""
    assert _ARCH.get_flag_gems_blacklist(vendor_name=vendor) == []


def test_strip_mrope_removes_only_mrope_keys():
    """Both mrope keys are popped; unrelated rope parameters survive."""
    config = SimpleNamespace(
        rope_parameters={
            "mrope_section": [1, 1, 1],
            "mrope_interleaved": True,
            "rope_theta": 10000.0,
        }
    )
    model_config = SimpleNamespace(hf_config=config, hf_text_config=config)

    patch_module._strip_mrope(model_config)

    assert config.rope_parameters == {"rope_theta": 10000.0}


def test_strip_mrope_handles_missing_configs_and_parameters():
    """A missing hf_config or a None rope_parameters mapping is harmless."""
    with_parameters = SimpleNamespace(rope_parameters={"keep": 1})
    without_parameters = SimpleNamespace(rope_parameters=None)
    model_config = SimpleNamespace(
        hf_config=None,
        hf_text_config=with_parameters,
    )
    patch_module._strip_mrope(model_config)
    assert with_parameters.rope_parameters == {"keep": 1}

    model_config = SimpleNamespace(
        hf_config=without_parameters,
        hf_text_config=without_parameters,
    )
    patch_module._strip_mrope(model_config)
    assert without_parameters.rope_parameters is None


# ---------------------------------------------------------------------------
# verify_and_update_config
# ---------------------------------------------------------------------------


def test_verify_resolves_auto_mamba_dtype(base_verify):
    """``auto`` adopts the checkpoint's FP32 recurrent-state dtype."""
    vllm_config = _make_config()
    vllm_config.model_config.hf_text_config.mamba_ssm_dtype = "float32"

    assert _ARCH.verify_and_update_config(vllm_config) is None
    assert vllm_config.cache_config.mamba_ssm_cache_dtype == "float32"
    assert base_verify == [vllm_config]


def test_verify_keeps_auto_when_checkpoint_has_no_dtype(base_verify):
    """With no requested dtype the cache stays on ``auto``."""
    vllm_config = _make_config()
    _ARCH.verify_and_update_config(vllm_config)
    assert vllm_config.cache_config.mamba_ssm_cache_dtype == "auto"


def test_verify_preserves_explicit_override_and_warns(base_verify, monkeypatch):
    """A conflicting explicit runtime dtype wins and logs one warning."""
    log = SimpleNamespace(calls=[])
    log.warning = lambda msg, *args: log.calls.append((msg, args))
    monkeypatch.setattr(patch_module, "logger", log)

    vllm_config = _make_config()
    vllm_config.model_config.hf_text_config.mamba_ssm_dtype = "float32"
    vllm_config.cache_config.mamba_ssm_cache_dtype = "float16"

    _ARCH.verify_and_update_config(vllm_config)

    assert vllm_config.cache_config.mamba_ssm_cache_dtype == "float16"
    assert len(log.calls) == 1
    message, args = log.calls[0]
    assert "mamba_ssm_dtype" in message
    assert args == ("float32", "float16")


def test_verify_does_not_warn_when_dtypes_agree(base_verify, monkeypatch):
    """Agreeing explicit and checkpoint dtypes produce no warning."""
    log = SimpleNamespace(calls=[])
    log.warning = lambda msg, *args: log.calls.append((msg, args))
    monkeypatch.setattr(patch_module, "logger", log)

    vllm_config = _make_config()
    vllm_config.model_config.hf_text_config.mamba_ssm_dtype = "float32"
    vllm_config.cache_config.mamba_ssm_cache_dtype = "float32"

    _ARCH.verify_and_update_config(vllm_config)

    assert log.calls == []
    assert vllm_config.cache_config.mamba_ssm_cache_dtype == "float32"


def test_verify_rejects_non_hybrid_hc_count(base_verify):
    """A single hyperconnection violates the Day0 hybrid contract."""
    vllm_config = _make_config()
    vllm_config.model_config.hf_text_config.hc_count = 1

    with pytest.raises(ValueError, match="hc_count > 1"):
        _ARCH.verify_and_update_config(vllm_config)


@pytest.mark.parametrize(
    "parallel",
    [
        {"enable_dbo": True, "ubatch_size": 1, "pipeline_parallel_size": 1},
        {"enable_dbo": False, "ubatch_size": 2, "pipeline_parallel_size": 1},
    ],
)
def test_verify_rejects_ple_with_dual_batch_or_microbatching(
    base_verify, parallel
):
    """PLE with DBO or microbatching is rejected."""
    vllm_config = _make_config()
    vllm_config.model_config.hf_text_config.ple_layer_ids = [2]
    vllm_config.parallel_config = SimpleNamespace(**parallel)

    with pytest.raises(NotImplementedError, match="dual-batch overlap"):
        _ARCH.verify_and_update_config(vllm_config)


def test_verify_rejects_qsa_with_microbatching(base_verify):
    """The sparse indexer (QSA) is covered by the same overlap gate."""
    vllm_config = _make_config()
    vllm_config.model_config.hf_text_config.indexer_n_heads = 4
    vllm_config.parallel_config = SimpleNamespace(
        enable_dbo=False, ubatch_size=2, pipeline_parallel_size=1
    )

    with pytest.raises(NotImplementedError, match="dual-batch overlap"):
        _ARCH.verify_and_update_config(vllm_config)


def test_verify_rejects_ple_with_pipeline_parallelism(base_verify):
    """PLE n-gram context cannot cross pipeline stages."""
    vllm_config = _make_config()
    vllm_config.model_config.hf_text_config.ple_layer_ids = [1]
    vllm_config.parallel_config = SimpleNamespace(
        enable_dbo=False, ubatch_size=1, pipeline_parallel_size=2
    )

    with pytest.raises(NotImplementedError, match="pipeline_parallel_size=1"):
        _ARCH.verify_and_update_config(vllm_config)


def test_verify_accepts_single_stage_ple(base_verify):
    """A single-stage PLE configuration passes every gate."""
    vllm_config = _make_config()
    vllm_config.model_config.hf_text_config.ple_layer_ids = [1]

    _ARCH.verify_and_update_config(vllm_config)
    assert base_verify == [vllm_config]


def test_verify_strips_mrope_for_language_model_only(base_verify):
    """Text-only multimodal use removes mrope keys from both configs."""
    vllm_config = _make_config()
    text_rope = {"mrope_section": [1], "rope_theta": 5000.0}
    hf_rope = {"mrope_interleaved": True, "keep": 2}
    vllm_config.model_config.hf_text_config.rope_parameters = text_rope
    vllm_config.model_config.hf_config = SimpleNamespace(rope_parameters=hf_rope)
    vllm_config.model_config.multimodal_config = SimpleNamespace(
        language_model_only=True
    )

    _ARCH.verify_and_update_config(vllm_config)

    assert text_rope == {"rope_theta": 5000.0}
    assert hf_rope == {"keep": 2}


def test_verify_keeps_mrope_for_full_multimodal(base_verify):
    """A non-text-only multimodal config keeps its mrope parameters."""
    vllm_config = _make_config()
    rope = {"mrope_section": [1]}
    vllm_config.model_config.hf_text_config.rope_parameters = rope
    vllm_config.model_config.multimodal_config = SimpleNamespace(
        language_model_only=False
    )

    _ARCH.verify_and_update_config(vllm_config)
    assert rope == {"mrope_section": [1]}


def test_verify_rejects_speculative_decoding(base_verify):
    """Native MTP/speculative decoding is a separate follow-up gate."""
    vllm_config = _make_config()
    vllm_config.speculative_config = SimpleNamespace(num_speculative_tokens=1)

    with pytest.raises(NotImplementedError, match="speculative"):
        _ARCH.verify_and_update_config(vllm_config)


def test_causal_verifier_always_strips_mrope(base_verify):
    """The causal variant strips mrope even without a multimodal config."""
    vllm_config = _make_config()
    text_rope = {"mrope_section": [1], "keep": 1}
    hf_rope = {"mrope_interleaved": True, "keep": 2}
    vllm_config.model_config.hf_text_config.rope_parameters = text_rope
    vllm_config.model_config.hf_config = SimpleNamespace(rope_parameters=hf_rope)

    _CAUSAL_ARCH.verify_and_update_config(vllm_config)

    assert text_rope == {"keep": 1}
    assert hf_rope == {"keep": 2}
    assert base_verify == [vllm_config]


# ---------------------------------------------------------------------------
# CommonAttentionMetadata token->request backport
# ---------------------------------------------------------------------------


def test_token_to_req_cache_noops_when_vllm_already_has_it(monkeypatch):
    """An upstream implementation is left untouched."""
    import vllm.v1.attention.backend as backend

    class _AlreadyBackported:
        @staticmethod
        def token_to_req_indices(self, buffer):
            return "UPSTREAM"

    monkeypatch.setattr(backend, "CommonAttentionMetadata", _AlreadyBackported)
    patch_module._patch_common_attention_token_to_req_cache()

    assert _AlreadyBackported.token_to_req_indices(None, None) == "UPSTREAM"


def test_token_to_req_cache_maps_and_pads(monkeypatch):
    """The mapping covers real tokens and zero-fills graph padding."""
    import vllm.v1.attention.backend as backend

    class _Metadata:
        pass

    monkeypatch.setattr(backend, "CommonAttentionMetadata", _Metadata)
    patch_module._patch_common_attention_token_to_req_cache()
    assert callable(_Metadata.token_to_req_indices)

    instance = SimpleNamespace(
        num_actual_tokens=7,
        query_start_loc_cpu=[0, 2, 2, 5],
        query_start_loc=torch.tensor([0, 2, 2, 5], dtype=torch.int32),
    )
    buffer = torch.full((8,), -9, dtype=torch.int32)

    mapped = _Metadata.token_to_req_indices(instance, buffer)

    # Query lens are [2, 0, 3] -> request ids [0, 0, 2, 2, 2].
    expected = torch.tensor([0, 0, 2, 2, 2, 0, 0], dtype=torch.int32)
    assert torch.equal(mapped, expected)
    assert mapped.dtype == torch.int32
    # Padding slots past the real tokens were zeroed; the tail was untouched.
    assert buffer.tolist()[:7] == [0, 0, 2, 2, 2, 0, 0]
    assert buffer.tolist()[7] == -9

    # The cache holds exactly max(mapped, actual) slots.
    cached = instance._token_to_req_indices_cache
    assert cached.shape == (7,)
    assert torch.equal(cached, expected)


def test_token_to_req_cache_reuses_cache(monkeypatch):
    """A later call returns the cached prefix without rebuilding."""
    import vllm.v1.attention.backend as backend

    class _Metadata:
        pass

    monkeypatch.setattr(backend, "CommonAttentionMetadata", _Metadata)
    patch_module._patch_common_attention_token_to_req_cache()

    instance = SimpleNamespace(
        num_actual_tokens=3,
        query_start_loc_cpu=[0, 1, 3],
        query_start_loc=torch.tensor([0, 1, 3], dtype=torch.int32),
    )
    first_buffer = torch.full((4,), -1, dtype=torch.int32)
    first = _Metadata.token_to_req_indices(instance, first_buffer)
    assert first.tolist() == [0, 1, 1]

    instance.num_actual_tokens = 2
    second_buffer = torch.full((4,), -7, dtype=torch.int32)
    second = _Metadata.token_to_req_indices(instance, second_buffer)

    assert second.tolist() == [0, 1]
    # The second buffer was never written; the cached prefix is returned.
    assert second_buffer.tolist() == [-7, -7, -7, -7]
    assert second.data_ptr() == first.data_ptr()


def test_token_to_req_cache_rejects_stale_cache(monkeypatch):
    """The cached prefix must cover the current actual-token count."""
    import vllm.v1.attention.backend as backend

    class _Metadata:
        pass

    monkeypatch.setattr(backend, "CommonAttentionMetadata", _Metadata)
    patch_module._patch_common_attention_token_to_req_cache()

    instance = SimpleNamespace(
        num_actual_tokens=2,
        query_start_loc_cpu=[0, 2],
        query_start_loc=torch.tensor([0, 2], dtype=torch.int32),
    )
    _Metadata.token_to_req_indices(instance, torch.zeros(3, dtype=torch.int32))

    # Growing the request's actual-token count beyond the cached length is a
    # contract violation (the cache is sized from the first call).
    instance.num_actual_tokens = 3
    with pytest.raises(AssertionError):
        _Metadata.token_to_req_indices(instance, torch.zeros(3, dtype=torch.int32))


def test_token_to_req_cache_rejects_short_buffer(monkeypatch):
    """Without a cache the scratch buffer must fit the mapped tokens."""
    import vllm.v1.attention.backend as backend

    class _Metadata:
        pass

    monkeypatch.setattr(backend, "CommonAttentionMetadata", _Metadata)
    patch_module._patch_common_attention_token_to_req_cache()

    instance = SimpleNamespace(
        num_actual_tokens=2,
        query_start_loc_cpu=[0, 2],
        query_start_loc=torch.tensor([0, 2], dtype=torch.int32),
    )
    with pytest.raises(AssertionError):
        _Metadata.token_to_req_indices(instance, torch.zeros(1, dtype=torch.int32))


# ---------------------------------------------------------------------------
# PLE metadata bridge
# ---------------------------------------------------------------------------


class _PleBuilder:
    pass


def _install_ple_bridge(monkeypatch):
    import vllm.v1.worker.gpu.model_states.mamba_hybrid as mamba_hybrid

    calls = []

    def original(self, builder, num_reqs):
        calls.append((self, builder, num_reqs))
        return ("ORIGINAL", builder, num_reqs)

    fake_mamba = type(
        "MambaHybridAttnMetadata", (), {"get_extra_attn_kwargs": original}
    )
    monkeypatch.setattr(mamba_hybrid, "MambaHybridAttnMetadata", fake_mamba)

    plugin_name = (
        "vllm_fl.models.qwen3_8_flash_next.common.short_conv_attn"
    )
    fake_short = types.ModuleType(plugin_name)
    fake_short.PleShortConvAttentionMetadataBuilder = _PleBuilder
    monkeypatch.setitem(sys.modules, plugin_name, fake_short)
    parent = sys.modules.get("vllm_fl.models.qwen3_8_flash_next.common")
    if parent is not None:
        monkeypatch.setattr(parent, "short_conv_attn", fake_short, raising=False)

    return fake_mamba, original, calls


def test_ple_bridge_slices_spec_fields_for_ple_builder(monkeypatch):
    """PLE builds receive request-sliced spec metadata; None stays None."""
    fake_mamba, original, calls = _install_ple_bridge(monkeypatch)
    patch_module._patch_ple_metadata_bridge()

    patched = fake_mamba.get_extra_attn_kwargs
    assert patched is not original
    assert patched._vllm_fl_qwen38_ple is True

    instance = SimpleNamespace(
        num_accepted_tokens=torch.tensor([3, 4, 5]),
        num_decode_draft_tokens_cpu=torch.tensor([0, 1, 2]),
    )
    kwargs = patched(instance, _PleBuilder(), 2)
    assert kwargs["num_accepted_tokens"].tolist() == [3, 4]
    assert kwargs["num_decode_draft_tokens_cpu"].tolist() == [0, 1]

    empty = SimpleNamespace(
        num_accepted_tokens=None, num_decode_draft_tokens_cpu=None
    )
    kwargs = patched(empty, _PleBuilder(), 1)
    assert kwargs == {
        "num_accepted_tokens": None,
        "num_decode_draft_tokens_cpu": None,
    }
    assert calls == []


def test_ple_bridge_delegates_for_other_builders(monkeypatch):
    """Non-PLE builders go through the original method unchanged."""
    fake_mamba, original, calls = _install_ple_bridge(monkeypatch)
    patch_module._patch_ple_metadata_bridge()
    patched = fake_mamba.get_extra_attn_kwargs

    instance = SimpleNamespace()
    builder = object()
    result = patched(instance, builder, 3)

    assert result == ("ORIGINAL", builder, 3)
    assert calls == [(instance, builder, 3)]


def test_ple_bridge_install_is_idempotent(monkeypatch):
    """A second install detects the marker and keeps the first wrapper."""
    fake_mamba, _, _ = _install_ple_bridge(monkeypatch)
    patch_module._patch_ple_metadata_bridge()
    first = fake_mamba.get_extra_attn_kwargs

    patch_module._patch_ple_metadata_bridge()
    assert fake_mamba.get_extra_attn_kwargs is first


# ---------------------------------------------------------------------------
# Compilation boundaries
# ---------------------------------------------------------------------------


def test_compilation_boundaries_are_appended_once(monkeypatch):
    """Both custom attention ops are registered without duplicates."""
    import vllm.config.compilation as compilation

    fake = type("CompilationConfig", (), {"_attention_ops": []})
    monkeypatch.setattr(compilation, "CompilationConfig", fake)

    patch_module._register_compilation_boundaries()
    patch_module._register_compilation_boundaries()

    assert fake._attention_ops == [
        "vllm::qwen3_8_flash_next_ple_short_conv",
        "vllm::qwen3_8_flash_next_qsa_with_output",
    ]


def test_compilation_boundaries_preserve_existing_ops(monkeypatch):
    """Pre-existing attention ops keep their order and position."""
    import vllm.config.compilation as compilation

    fake = type(
        "CompilationConfig",
        (),
        {"_attention_ops": ["vllm::existing", "vllm::qwen3_8_flash_next_qsa_with_output"]},
    )
    monkeypatch.setattr(compilation, "CompilationConfig", fake)

    patch_module._register_compilation_boundaries()

    assert fake._attention_ops == [
        "vllm::existing",
        "vllm::qwen3_8_flash_next_qsa_with_output",
        "vllm::qwen3_8_flash_next_ple_short_conv",
    ]


# ---------------------------------------------------------------------------
# apply_qwen3_8_flash_next_patches
# ---------------------------------------------------------------------------


class _ConfigSentinel:
    pass


def _install_registry_fakes(monkeypatch):
    """Install fake vllm registries and the fake plugin config module."""
    import vllm.model_executor.models as models_pkg
    import vllm.transformers_utils as transformers_utils_pkg

    register_calls = []
    state = SimpleNamespace(
        config_registry={},
        models_config_map={},
        text_models={},
        multimodal_models={},
        vllm_models={},
        register_calls=register_calls,
    )
    state.model_config = SimpleNamespace(MODELS_CONFIG_MAP=state.models_config_map)
    state.model_registry = SimpleNamespace(
        _TEXT_GENERATION_MODELS=state.text_models,
        _MULTIMODAL_MODELS=state.multimodal_models,
        _VLLM_MODELS=state.vllm_models,
        ModelRegistry=SimpleNamespace(
            register_model=lambda arch, path: register_calls.append((arch, path))
        ),
    )
    state.transformers_config = SimpleNamespace(_CONFIG_REGISTRY=state.config_registry)

    monkeypatch.setattr(models_pkg, "config", state.model_config, raising=False)
    monkeypatch.setattr(models_pkg, "registry", state.model_registry, raising=False)
    monkeypatch.setattr(
        transformers_utils_pkg,
        "config",
        state.transformers_config,
        raising=False,
    )

    fake_config_module = types.ModuleType(
        "vllm_fl.models.qwen3_8_flash_next.config"
    )
    fake_config_module.Qwen3_8FlashNextConfig = _ConfigSentinel
    fake_config_module.Qwen3_8FlashNextTextConfig = _ConfigSentinel
    monkeypatch.setitem(
        sys.modules, "vllm_fl.models.qwen3_8_flash_next.config", fake_config_module
    )
    import vllm_fl.models.qwen3_8_flash_next as plugin_pkg

    monkeypatch.setattr(
        plugin_pkg, "config", fake_config_module, raising=False
    )
    return state


def test_apply_registers_configs_models_and_helpers(monkeypatch):
    """Registries receive every architecture and the helpers run once."""
    state = _install_registry_fakes(monkeypatch)
    events = []
    monkeypatch.setattr(
        patch_module,
        "_patch_common_attention_token_to_req_cache",
        lambda: events.append("token_cache"),
    )
    monkeypatch.setattr(
        patch_module,
        "_patch_ple_metadata_bridge",
        lambda: events.append("ple"),
    )
    monkeypatch.setattr(
        patch_module,
        "_register_compilation_boundaries",
        lambda: events.append("compilation"),
    )

    assert patch_module.apply_qwen3_8_flash_next_patches() is True

    assert state.config_registry == {
        "qwen3_8_flash_next": _ConfigSentinel,
        "qwen3_8_flash_next_text": _ConfigSentinel,
        "qwen4_exp": _ConfigSentinel,
        "qwen4_exp_text": _ConfigSentinel,
    }

    expected_map = {}
    expected_text = {}
    expected_multimodal = {}
    expected_vllm = {}
    expected_calls = []
    for architecture, (class_name, verifier, family) in (
        patch_module._ARCHITECTURES.items()
    ):
        expected_map[architecture] = verifier
        module = "vllm_fl.models.qwen3_8_flash_next"
        if family == "multimodal":
            expected_multimodal[architecture] = (module, class_name)
        else:
            expected_text[architecture] = (module, class_name)
        expected_vllm[architecture] = (module, class_name)
        expected_calls.append(
            (architecture, f"{module}:{class_name}")
        )

    assert state.models_config_map == expected_map
    assert state.text_models == expected_text
    assert state.multimodal_models == expected_multimodal
    assert state.vllm_models == expected_vllm
    assert state.register_calls == expected_calls
    assert events == ["token_cache", "ple", "compilation"]


def test_apply_repeats_lazy_registration_and_keeps_setdefaults(monkeypatch):
    """Existing source-map entries survive; ModelRegistry registration repeats."""
    state = _install_registry_fakes(monkeypatch)
    monkeypatch.setattr(
        patch_module, "_patch_common_attention_token_to_req_cache", lambda: None
    )
    monkeypatch.setattr(patch_module, "_patch_ple_metadata_bridge", lambda: None)
    monkeypatch.setattr(
        patch_module, "_register_compilation_boundaries", lambda: None
    )
    state.config_registry["qwen4_exp"] = "PRESEEDED"
    state.text_models["Qwen3_8FlashNextForCausalLM"] = ("other", "Other")
    state.multimodal_models[
        "Qwen3_8FlashNextForConditionalGeneration"
    ] = ("other", "Other")

    assert patch_module.apply_qwen3_8_flash_next_patches() is True
    assert patch_module.apply_qwen3_8_flash_next_patches() is True

    assert state.config_registry["qwen4_exp"] == "PRESEEDED"
    assert state.text_models["Qwen3_8FlashNextForCausalLM"] == ("other", "Other")
    assert state.multimodal_models[
        "Qwen3_8FlashNextForConditionalGeneration"
    ] == ("other", "Other")
    assert len(state.register_calls) == 2 * len(patch_module._ARCHITECTURES)
