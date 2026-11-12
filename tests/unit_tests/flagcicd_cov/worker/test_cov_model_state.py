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

"""Coverage tests for vllm_fl/worker/model_state.py.

``ModelRunnerState`` is the legacy-runner hook object: it stores the
runner it wraps, ignores request removals, and answers both prepare
calls with an empty mapping. Each test pins one of those behaviors.
"""

from . import _imports  # noqa: F401  (installs local import shims first)

from vllm_fl.worker.model_state import ModelRunnerState


class Sentinel:
    """Unique object proving the stored runner is the one passed in."""


def test_state_stores_the_runner_it_wraps():
    """The constructor records the runner on the .runner attribute."""
    runner = Sentinel()
    state = ModelRunnerState(runner)
    assert state.runner is runner


def test_remove_requests_is_a_no_op_for_any_ids():
    """remove_requests accepts arbitrary request ids and returns None."""
    state = ModelRunnerState(runner=object())
    assert state.remove_requests(["req-1", "req-2", None]) is None


def test_prepare_inputs_returns_empty_mapping():
    """prepare_inputs answers with an empty dict regardless of arguments."""
    state = ModelRunnerState(runner=object())
    result = state.prepare_inputs(num_reqs=3, num_reqs_padded=4, input_ids=[1, 2])
    assert result == {}
    assert isinstance(result, dict)


def test_prepare_dummy_inputs_returns_empty_mapping():
    """prepare_dummy_inputs answers with an empty dict regardless of arguments."""
    state = ModelRunnerState(runner=object())
    result = state.prepare_dummy_inputs(
        num_reqs=5, num_reqs_padded=8, num_scheduled_tokens=17
    )
    assert result == {}
    assert isinstance(result, dict)
