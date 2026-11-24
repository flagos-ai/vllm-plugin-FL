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

"""G9 gap closure for ``vllm_fl/ops/fused_moe/fused_moe_utils.py`` (814b).

Focuses on the ``TritonExpertsFL`` branch dispatcher (npu / kunlunxin /
flagGems-cuda / generic Triton step pipeline with and without LoRA),
``workspace_shapes`` and ``moe_sum``.  All kernels and quantization boundaries
are recording fakes; tensors are small and CPU-only.
"""

from __future__ import annotations

import sys
import types
from types import SimpleNamespace

import pytest
import torch

try:  # pragma: no cover - environment specific
    import vllm_fl.ops.fused_moe.fused_moe_utils as Mod
except Exception as exc:  # pragma: no cover
    pytest.skip(f"fused_moe_utils unavailable: {exc}", allow_module_level=True)


def _install_fake(monkeypatch, dotted, attrs):
    parts = dotted.split(".")
    for i in range(1, len(parts)):
        prefix = ".".join(parts[:i])
        if prefix not in sys.modules:
            pkg = types.ModuleType(prefix)
            pkg.__path__ = []
            monkeypatch.setitem(sys.modules, prefix, pkg)
    module = types.ModuleType(dotted)
    for name, value in attrs.items():
        setattr(module, name, value)
    monkeypatch.setitem(sys.modules, dotted, module)
    return module


def _platform(**attrs):
    base = dict(
        vendor_name="nvidia",
        device_type="cuda",
        is_cuda=lambda: False,
        is_out_of_tree=lambda: False,
        is_cpu=lambda: False,
    )
    base.update(attrs)
    return SimpleNamespace(**base)


def _quant(**attrs):
    base = dict(
        use_fp8_w8a8=False,
        use_int8_w8a8=False,
        use_int8_w8a16=False,
        use_int4_w4a16=False,
        quant_dtype=None,
        weight_quant_dtype=None,
        config_name=lambda dtype: "dummy",
    )
    base.update(attrs)
    return SimpleNamespace(**base)


def _experts(*, lora=None, quant=None, **attrs):
    experts = object.__new__(Mod.TritonExpertsFL)
    experts._lora_context = lora
    experts.quant_config = quant or _quant()
    experts.per_act_token_quant = False
    experts.block_shape = None
    experts.w1_scale = None
    experts.w2_scale = None
    experts.w1_bias = None
    experts.w2_bias = None
    experts.quant_dtype = None
    experts.quantization_emulation = None
    experts.activation_config = SimpleNamespace(clamp_limit=0.0)
    experts.moe_config = SimpleNamespace(max_num_tokens=64)
    for key, value in attrs.items():
        setattr(experts, key, value)
    return experts


@pytest.fixture(autouse=True)
def _tl_namespace(monkeypatch):
    # The host triton stand-in may lack scalar dtype handles.
    monkeypatch.setattr(
        Mod,
        "tl",
        SimpleNamespace(
            float32="tl.float32",
            float16="tl.float16",
            bfloat16="tl.bfloat16",
        ),
    )


# ---------------------------------------------------------------------------
# workspace_shapes
# ---------------------------------------------------------------------------




# ---------------------------------------------------------------------------
# apply: accelerator fast paths
# ---------------------------------------------------------------------------








# ---------------------------------------------------------------------------
# apply: generic Triton step pipeline
# ---------------------------------------------------------------------------


def _step_env(monkeypatch, *, with_lora):
    monkeypatch.setattr(
        Mod, "current_platform", _platform(is_cuda=lambda: False)
    )
    monkeypatch.setattr(
        Mod,
        "try_get_optimal_moe_config",
        lambda *a, **k: {"BLOCK_SIZE_M": 16},
        raising=False,
    )
    monkeypatch.setattr(
        Mod, "_resize_cache", lambda ws, shape: torch.zeros(*shape)
    )
    monkeypatch.setattr(
        Mod, "_prepare_expert_assignment", lambda *a, **k: (None, None, None)
    )
    invokes = []
    monkeypatch.setattr(
        Mod,
        "_invoke_fused_moe_triton_kernel",
        lambda *a, **k: invokes.append((a, k)),
    )
    acts = []
    monkeypatch.setattr(
        Mod,
        "apply_moe_activation",
        lambda activation, out, inp, **k: acts.append(activation.value),
    )
    monkeypatch.setattr(
        Mod,
        "moe_kernel_quantize_input",
        lambda A, *a, **k: (A, None),
    )
    sums = []
    monkeypatch.setattr(Mod, "_moe_sum", lambda i, o: sums.append(tuple(i.shape)))
    return invokes, acts, sums


def _run_step(experts, *, hidden_k, topk):
    output = torch.zeros(2, hidden_k)
    Mod.TritonExpertsFL.apply(
        experts,
        output,
        torch.zeros(2, hidden_k),
        torch.zeros(2, 8, hidden_k),
        torch.zeros(2, hidden_k, 8),
        torch.zeros(2, topk),
        torch.zeros(2, topk, dtype=torch.int32),
        SimpleNamespace(value="silu"),
        2,
        None,
        None,
        None,
        torch.zeros(2 * topk * 8),
        torch.zeros(2 * topk * 8),
        None,
        False,
    )
    return output






# ---------------------------------------------------------------------------
# moe_sum
# ---------------------------------------------------------------------------


