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

"""Import shim for the W2b model_runner_tail area.

Reuses the W4 worker-area shim (which installs the uvloop / flag_gems /
xgrammar placeholders and the vllm 0.24-line symbol backfills, and imports
the ``vllm_fl.worker`` chain under a CUDA-like platform before restoring
the real one), then adds two local-only backfills:

* a placeholder for ``numba`` (present on the A100 CI image, absent on this
  dev host). Only the import of ``vllm.v1.spec_decode.ngram_proposer``
  (performed inside ``propose_draft_token_ids`` for the ``ngram`` method)
  reads it; the tests here never execute a numba-compiled kernel — the
  drafter is a recording subclass.
* a constructor wrapper letting the plugin's ``routed_experts=`` kwarg
  through ``ModelRunnerOutput`` on local vllm 0.33, where the field was
  dropped after the 0.24 line the CI image ships.

On the CI image both steps are no-ops guarded by a probe of the real
symbol/package.
"""

from __future__ import annotations

import sys
import types

from ..worker import _imports as _worker_imports


def _backfill_model_runner_output_routed_experts() -> None:
    """Accept the plugin's ``routed_experts=`` kwarg on local vllm 0.33.

    The CI image's vllm (0.24-line) ``ModelRunnerOutput`` dataclass has a
    ``routed_experts`` field that the plugin passes to the constructor;
    vllm 0.33 dropped it. On the CI image the field exists and this is a
    no-op. Locally, the constructor wrapper drops the (always-None on
    these CPU paths) kwarg so the plugin's call resolves; the sync-path
    payload assignment ``output.routed_experts = ...`` (field write, not
    a constructor call) is not exercised by these tests either way.
    """
    from vllm.v1.outputs import ModelRunnerOutput

    if "routed_experts" in ModelRunnerOutput.__dataclass_fields__:
        return
    original_init = ModelRunnerOutput.__init__

    def _init(self, *args, **kwargs):
        kwargs.pop("routed_experts", None)
        original_init(self, *args, **kwargs)

    ModelRunnerOutput.__init__ = _init
    # The async wrapper class forwards **kwargs of the same shape; it is
    # constructed with explicit keywords only, so it needs no backfill.


def _install_placeholder_numba() -> None:
    """Install an inert numba when absent (only its import surface is read)."""
    if "numba" in sys.modules:
        return
    try:
        __import__("numba")
        return
    except Exception:
        pass
    numba = types.ModuleType("numba")

    def _passthrough_decorator(*args, **kwargs):
        if len(args) == 1 and callable(args[0]) and not kwargs:
            return args[0]

        def _wrap(fn):
            return fn

        return _wrap

    numba.jit = _passthrough_decorator
    numba.njit = _passthrough_decorator
    numba.get_num_threads = lambda: 1
    numba.set_num_threads = lambda n: None
    numba.prange = range
    sys.modules["numba"] = numba


def ensure_importable_env() -> None:
    _worker_imports.ensure_importable_env()
    _install_placeholder_numba()
    _backfill_model_runner_output_routed_experts()


ensure_importable_env()
