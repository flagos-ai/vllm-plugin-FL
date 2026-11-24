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

"""test_cov_probe.py — importability probe for the dispatch area.

Imports the three D3 target modules at module scope to verify the
collection gate passes locally (triton is absent on this Windows host,
so the shim installs an inert flag_gems placeholder first).
"""

# isort: split
from . import (
    _flaggems_shim,  # noqa: F401  (installs inert flag_gems if triton is absent)
)

# isort: split
from vllm_fl.dispatch import io_common, io_dumper
from vllm_fl.dispatch.config import utils as config_utils


def test_targets_importable():
    """The three D3 target modules import cleanly on a triton-less host."""
    assert io_common.__name__ == "vllm_fl.dispatch.io_common"
    assert io_dumper.__name__ == "vllm_fl.dispatch.io_dumper"
    assert config_utils.__name__ == "vllm_fl.dispatch.config.utils"
