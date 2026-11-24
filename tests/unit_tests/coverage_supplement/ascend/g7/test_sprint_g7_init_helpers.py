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

"""G7-A (batch 2) coverage: init / cache / stream / state helpers of
ModelRunnerFL (814b line).

Targets (814b): update_max_model_len 1114-1119, reset_mm_cache 1121-1128,
reset_encoder_cache 1130-1137, post_kv_cache_wake_up 1139-1140,
init_fp8_kv_scales 1142-1191, _get_mamba_state_copy_funcs 1230-1247,
_init_model_kwargs 1267-1306, _zero_block_ids 1347-1350,
_init_device_properties 1353-1356, _sync_device 1359-1360,
_get_or_create_async_output_copy_stream 1362-1369,
_on_request_state_removed 1371-1377,
_process_encoder_cache_scheduler_output 1379-1385.

The runner is faked with SimpleNamespace; every module-level collaborator
(is_quantized_kv_cache, mamba_utils, num_compute_units, _accelerator_synchronize,
current_platform, Attention/MLAAttention) is patched at the module boundary.
"""

from __future__ import annotations

import contextlib
from types import SimpleNamespace

import numpy as np
import torch

from vllm_fl.worker import model_runner as mr


class _Patch:
    def __init__(self, module, name, value):
        self.module, self.name, self.value = module, name, value

    def __enter__(self):
        self.saved = getattr(self.module, self.name)
        setattr(self.module, self.name, self.value)
        return self

    def __exit__(self, *exc):
        setattr(self.module, self.name, self.saved)
        return False


def _stack(*pairs):
    stack = contextlib.ExitStack()
    for module, name, value in pairs:
        stack.enter_context(_Patch(module, name, value))
    return stack


class _FakeAttn:
    """Stand-in for Attention / MLAAttention to satisfy isinstance()."""


# --------------------------------------------------------------------------
# update_max_model_len / reset_* / post_kv_cache_wake_up
# --------------------------------------------------------------------------
def test_update_max_model_len_without_speculative_config():
    runner = SimpleNamespace(speculative_config=None)
    mr.ModelRunnerFL.update_max_model_len(runner, 4096)
    assert runner.max_model_len == 4096
    assert not hasattr(runner, "effective_drafter_max_model_len")


def test_update_max_model_len_with_speculative_draft_none():
    runner = SimpleNamespace(
        speculative_config=SimpleNamespace(draft_model_config=None)
    )
    mr.ModelRunnerFL.update_max_model_len(runner, 8192)
    assert runner.max_model_len == 8192
    assert runner.effective_drafter_max_model_len == 8192


def test_reset_mm_cache_resets_budget_and_late_interaction():
    calls = []
    runner = SimpleNamespace(
        mm_budget=SimpleNamespace(reset_cache=lambda: calls.append("budget")),
        late_interaction_runner=SimpleNamespace(clear=lambda: calls.append("li")),
    )
    mr.ModelRunnerFL.reset_mm_cache(runner)
    assert calls == ["budget", "li"]


def test_reset_mm_cache_skips_budget_when_none():
    calls = []
    runner = SimpleNamespace(
        mm_budget=None,
        late_interaction_runner=SimpleNamespace(clear=lambda: calls.append("li")),
    )
    mr.ModelRunnerFL.reset_mm_cache(runner)
    assert calls == ["li"]


def test_reset_encoder_cache_clears_cache_and_late_interaction():
    calls = []
    cache = {"h": 1}
    runner = SimpleNamespace(
        encoder_cache=cache,
        late_interaction_runner=SimpleNamespace(clear=lambda: calls.append("li")),
    )
    mr.ModelRunnerFL.reset_encoder_cache(runner)
    assert cache == {}
    assert calls == ["li"]


def test_post_kv_cache_wake_up_calls_init_fp8():
    calls = []
    runner = SimpleNamespace(init_fp8_kv_scales=lambda: calls.append("init"))
    mr.ModelRunnerFL.post_kv_cache_wake_up(runner)
    assert calls == ["init"]


# --------------------------------------------------------------------------
# init_fp8_kv_scales
# --------------------------------------------------------------------------
def test_init_fp8_kv_scales_early_return_when_not_quantized():
    runner = SimpleNamespace(cache_config=SimpleNamespace(cache_dtype="auto"))
    with _stack((mr, "is_quantized_kv_cache", lambda dtype: False)):
        mr.ModelRunnerFL.init_fp8_kv_scales(runner)
    # Early return before touching kv_caches.
    assert not hasattr(runner, "kv_caches")


def test_init_fp8_kv_scales_zeroes_cache_and_resets_scales():
    k_scale = torch.zeros(1)
    v_scale = torch.zeros(1)
    attn = _FakeAttn()
    attn._k_scale = k_scale
    attn._v_scale = v_scale
    attn.k_scale = torch.zeros(1)
    attn.v_scale = torch.zeros(1)

    kv_caches = [torch.ones(3), None, [torch.ones(2), torch.ones(2)]]
    runner = SimpleNamespace(
        cache_config=SimpleNamespace(cache_dtype="fp8"),
        kv_caches=kv_caches,
        compilation_config=SimpleNamespace(
            static_forward_context={"layer0": attn, "other": object()}
        ),
    )
    with _stack(
        (mr, "is_quantized_kv_cache", lambda dtype: True),
        (mr, "Attention", _FakeAttn),
        (mr, "MLAAttention", _FakeAttn),
    ):
        mr.ModelRunnerFL.init_fp8_kv_scales(runner)

    assert torch.count_nonzero(kv_caches[0]) == 0
    assert torch.count_nonzero(kv_caches[2][0]) == 0
    assert torch.count_nonzero(kv_caches[2][1]) == 0
    for t in (attn._k_scale, attn._v_scale, attn.k_scale, attn.v_scale):
        assert float(t.item()) == 1.0


# --------------------------------------------------------------------------
# _get_mamba_state_copy_funcs
# --------------------------------------------------------------------------
def _mamba_groups():
    return [
        SimpleNamespace(mamba_type="mamba2"),
        SimpleNamespace(mamba_type="gdn"),
    ]


def test_get_mamba_state_copy_funcs_new_api_caches_and_validates():
    calls = []
    groups = _mamba_groups()
    fake_mamba_utils = SimpleNamespace(
        get_mamba_groups=lambda cfg: groups,
        validate_mamba_state_copy_funcs=lambda g, c: calls.append((g, c)),
    )
    model = SimpleNamespace(
        get_mamba_state_copy_funcs=lambda types: {"types": sorted(types)}
    )
    runner = SimpleNamespace(kv_cache_config=object(), model=model)
    with _stack((mr, "mamba_utils", fake_mamba_utils)):
        result = mr.ModelRunnerFL._get_mamba_state_copy_funcs(runner)
    assert result == {"types": ["gdn", "mamba2"]}
    assert runner._mamba_state_copy_funcs is result
    assert len(calls) == 1


def test_get_mamba_state_copy_funcs_legacy_api():
    groups = _mamba_groups()
    fake_mamba_utils = SimpleNamespace(
        get_mamba_groups=lambda cfg: groups,
        validate_mamba_state_copy_funcs=lambda g, c: None,
    )
    model = SimpleNamespace(get_mamba_state_copy_func=lambda: "legacy")
    runner = SimpleNamespace(kv_cache_config=object(), model=model)
    with _stack((mr, "mamba_utils", fake_mamba_utils)):
        result = mr.ModelRunnerFL._get_mamba_state_copy_funcs(runner)
    assert result == {"mamba2": "legacy", "gdn": "legacy"}


def test_get_mamba_state_copy_funcs_uses_cache():
    calls = []
    fake_mamba_utils = SimpleNamespace(
        get_mamba_groups=lambda cfg: calls.append("groups") or [],
        validate_mamba_state_copy_funcs=lambda g, c: None,
    )
    runner = SimpleNamespace(
        kv_cache_config=object(),
        model=object(),
        _mamba_state_copy_funcs={"cached": True},
    )
    with _stack((mr, "mamba_utils", fake_mamba_utils)):
        result = mr.ModelRunnerFL._get_mamba_state_copy_funcs(runner)
    assert result == {"cached": True}
    assert calls == []


# --------------------------------------------------------------------------
# _init_model_kwargs
# --------------------------------------------------------------------------
def test_init_model_kwargs_non_pooling_returns_empty():
    runner = SimpleNamespace(is_pooling_model=False)
    assert mr.ModelRunnerFL._init_model_kwargs(runner) == {}




# --------------------------------------------------------------------------
# _zero_block_ids / _init_device_properties / _sync_device
# --------------------------------------------------------------------------
def test_zero_block_ids_delegates_to_zeroer():
    calls = []
    runner = SimpleNamespace(
        _kv_block_zeroer=SimpleNamespace(
            zero_block_ids=lambda ids: calls.append(ids)
        )
    )
    mr.ModelRunnerFL._zero_block_ids(runner, [3, 4])
    assert calls == [[3, 4]]


def test_zero_block_ids_no_zeroer_is_noop():
    runner = SimpleNamespace()
    mr.ModelRunnerFL._zero_block_ids(runner, [1])
    assert not hasattr(runner, "_kv_block_zeroer")


def test_init_device_properties_uses_num_compute_units():
    runner = SimpleNamespace(device=SimpleNamespace(index=2))
    with _stack((mr, "num_compute_units", lambda idx: 40 + idx)):
        mr.ModelRunnerFL._init_device_properties(runner)
    assert runner.num_sms == 42


def test_sync_device_delegates_to_accelerator_synchronize():
    calls = []
    with _stack((mr, "_accelerator_synchronize", lambda: calls.append("sync"))):
        mr.ModelRunnerFL._sync_device(SimpleNamespace())
    assert calls == ["sync"]


# --------------------------------------------------------------------------
# _get_or_create_async_output_copy_stream
# --------------------------------------------------------------------------
def test_get_or_create_async_output_copy_stream_creates_once():
    created = []

    def _make_stream():
        created.append(1)
        return f"stream-{len(created)}"

    fake_platform = SimpleNamespace(
        torch_device_fn=SimpleNamespace(Stream=_make_stream)
    )
    runner = SimpleNamespace(async_output_copy_stream=None)
    with _stack((mr, "current_platform", fake_platform)):
        first = mr.ModelRunnerFL._get_or_create_async_output_copy_stream(runner)
        second = mr.ModelRunnerFL._get_or_create_async_output_copy_stream(runner)
    assert first == second == "stream-1"
    assert runner.async_output_copy_stream == "stream-1"
    assert len(created) == 1


# --------------------------------------------------------------------------
# _on_request_state_removed / _process_encoder_cache_scheduler_output
# --------------------------------------------------------------------------
def test_on_request_state_removed_is_noop():
    assert mr.ModelRunnerFL._on_request_state_removed(
        SimpleNamespace(), "req", None
    ) is None


def test_process_encoder_cache_scheduler_output_pops_hashes():
    cache = {"a": 1, "b": 2}
    runner = SimpleNamespace(encoder_cache=cache)
    scheduler_output = SimpleNamespace(free_encoder_mm_hashes=["a", "missing"])
    mr.ModelRunnerFL._process_encoder_cache_scheduler_output(runner, scheduler_output)
    assert cache == {"b": 2}
