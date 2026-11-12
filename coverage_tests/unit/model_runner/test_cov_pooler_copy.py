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

"""Coverage tests for vllm_fl/worker/model_runner.py pooler-output copying.

``_copy_pooler_output_to_cpu`` is a module-level pure function with a
tensor path and a list path, each with batch-mismatch validation and the
0-finished / all-finished / partially-finished selection. Every test
asserts the selected values, not just shapes.
"""

import torch

from . import _imports  # noqa: F401  (installs local import shims first)
from vllm_fl.worker.model_runner import _copy_pooler_output_to_cpu


def test_tensor_path_returns_none_when_nothing_finished():
    """Tensor path: nothing finished means all rows are None."""
    raw = torch.tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
    result = _copy_pooler_output_to_cpu(raw, [False, False, False])
    assert result == [None, None, None]


def test_tensor_path_returns_all_rows_when_all_finished():
    """Tensor path: all finished returns every row's values."""
    raw = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    result = _copy_pooler_output_to_cpu(raw, [True, True])
    assert len(result) == 2
    assert result[0].tolist() == [1.0, 2.0]
    assert result[1].tolist() == [3.0, 4.0]


def test_tensor_path_selects_partial_finished_rows():
    """Tensor path: only finished rows are copied; others stay None."""
    raw = torch.tensor([[1.0], [2.0], [3.0], [4.0]])
    result = _copy_pooler_output_to_cpu(raw, [False, True, False, True])
    # Rows 1 and 3 are the finished requests; the rest stay None.
    assert result[0] is None
    assert result[1].tolist() == [2.0]
    assert result[2] is None
    assert result[3].tolist() == [4.0]


def test_tensor_path_rejects_batch_size_mismatch():
    """Tensor path: a mask shorter than the tensor is rejected."""
    raw = torch.zeros(3, 2)
    try:
        _copy_pooler_output_to_cpu(raw, [True, True])
    except ValueError as exc:
        assert "3 != 2" in str(exc)
        return
    raise AssertionError("tensor path did not reject a mismatched mask")


def test_list_path_keeps_only_finished_non_none_entries():
    """List path: unfinished entries become None."""
    t0 = torch.tensor([10.0])
    t1 = torch.tensor([11.0])
    t2 = torch.tensor([12.0])
    raw = [t0, t1, t2]
    result = _copy_pooler_output_to_cpu(raw, [True, False, True])
    assert result[0] is t0 or result[0].tolist() == [10.0]
    assert result[1] is None
    assert result[2].tolist() == [12.0]


def test_list_path_skips_finished_but_none_entries():
    """List path: finished-but-None outputs stay None."""
    raw = [torch.tensor([5.0]), None]
    result = _copy_pooler_output_to_cpu(raw, [True, True])
    assert result[0].tolist() == [5.0]
    assert result[1] is None


def test_list_path_rejects_batch_size_mismatch():
    """List path: a mask longer than the list is rejected."""
    try:
        _copy_pooler_output_to_cpu([torch.zeros(1)], [True, True, True])
    except ValueError as exc:
        assert "1 != 3" in str(exc)
        return
    raise AssertionError("list path did not reject a mismatched mask")
