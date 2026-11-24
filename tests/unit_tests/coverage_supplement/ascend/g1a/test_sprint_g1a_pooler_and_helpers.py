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

"""Coverage for pooler-output copying and small module helpers (814b 304-370, 474-516)."""

import sys
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch

from vllm_fl.worker import model_runner as mr


def test_tensor_path_none_finished():
    raw = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    assert mr._copy_pooler_output_to_cpu(raw, [False, False]) == [None, None]


def test_tensor_path_all_finished():
    raw = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    result = mr._copy_pooler_output_to_cpu(raw, [True, True])
    assert [t.tolist() for t in result] == [[1.0, 2.0], [3.0, 4.0]]


def test_tensor_path_partial_finished_selects_rows():
    raw = torch.tensor([[1.0], [2.0], [3.0], [4.0]])
    result = mr._copy_pooler_output_to_cpu(raw, [False, True, False, True])
    assert result[0] is None
    assert result[1].tolist() == [2.0]
    assert result[2] is None
    assert result[3].tolist() == [4.0]


def test_tensor_path_rejects_mask_mismatch():
    with pytest.raises(ValueError, match="3 != 2"):
        mr._copy_pooler_output_to_cpu(torch.zeros(3, 2), [True, True])


def test_list_path_keeps_only_finished_non_none():
    raw = [torch.tensor([10.0]), torch.tensor([11.0]), None]
    result = mr._copy_pooler_output_to_cpu(raw, [True, False, True])
    assert result[0].tolist() == [10.0]
    assert result[1] is None
    assert result[2] is None


def test_list_path_rejects_mask_mismatch():
    with pytest.raises(ValueError, match="1 != 3"):
        mr._copy_pooler_output_to_cpu([torch.zeros(1)], [True, True, True])


def test_accelerator_synchronize_musa_branch():
    calls = []

    class _P:
        device_type = "musa"

    fake_musa = SimpleNamespace(synchronize=lambda: calls.append("musa"))
    with patch.object(mr, "current_platform", _P()), patch.dict(
        sys.modules, {"torch_musa": fake_musa}
    ):
        mr._accelerator_synchronize()
    assert calls == ["musa"]


def test_accelerator_synchronize_other_branch():
    calls = []

    class _P:
        device_type = "npu"

    fake_accelerator = SimpleNamespace(synchronize=lambda: calls.append("accel"))
    with patch.object(mr, "current_platform", _P()), patch.object(
        mr, "torch", SimpleNamespace(accelerator=fake_accelerator)
    ):
        mr._accelerator_synchronize()
    assert calls == ["accel"]


def test_get_parameter_for_reload_plain_submodule():
    model = torch.nn.Module()
    layer = torch.nn.Linear(2, 3)
    model.blocks = torch.nn.ModuleList([layer])
    param = mr._get_parameter_for_reload(model, "blocks.0.weight")
    assert param is layer.weight


def test_get_parameter_for_reload_unwraps_lora_layer():
    class _FakeLoraLayer(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.base_layer = torch.nn.Module()
            self.base_layer.register_parameter(
                "weight", torch.nn.Parameter(torch.zeros(2, 2))
            )

    model = torch.nn.Module()
    model.blocks = torch.nn.ModuleList([_FakeLoraLayer()])
    wrapper = model.blocks[0]
    with patch.object(mr, "BaseLayerWithLoRA", _FakeLoraLayer):
        param = mr._get_parameter_for_reload(model, "blocks.0.weight")
    assert param is wrapper.base_layer.weight


def test_get_parameter_for_reload_root_parameter():
    model = torch.nn.Module()
    model.register_parameter("weight", torch.nn.Parameter(torch.ones(1)))
    param = mr._get_parameter_for_reload(model, "weight")
    assert param is model.weight


def test_count_nans_per_row_counts_each_row():
    logits = torch.tensor(
        [[float("nan"), 1.0, 2.0], [3.0, 4.0, 5.0], [float("nan"), float("nan"), 0.0]]
    )
    counts = mr.count_nans_per_row(logits)
    assert counts.dtype == torch.int32
    assert counts.tolist() == [1, 0, 2]


def test_nans_to_dict_pads_missing_rows_with_zero():
    result = mr.nans_to_dict([1, 0], {"a": 0, "b": 1, "c": 2})
    assert result == {"a": 1, "b": 0, "c": 0}
