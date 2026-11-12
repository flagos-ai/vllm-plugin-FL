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

"""Coverage tests for vllm_fl/worker/worker.py reinitialize_distributed.

``reinitialize_distributed`` sequences an elastic-EP resize: the optional
pre-shrink expert resharding, distributed-environment teardown, then
either a rank shutdown (evicted worker) or re-initialization plus the
MoE reconfiguration and post-scale-up EPLB hook. Per CONTRACT R2 the
vllm parallel-state groups, cleanup helpers, and set_current_vllm_config
are inputs replaced with recorders; the branch selection is driven only
by the EP world sizes and the request's rank fields. The unbound-method
pattern follows the dev tests in tests/unit_tests/worker/.
"""

import inspect
from contextlib import nullcontext
from types import SimpleNamespace

# The shim import is load-bearing: it must precede vllm_fl/vllm
# imports on hosts without triton. Do not alphabetize.
from . import _imports  # noqa: F401  (installs local import shims first)
import vllm.config as vllm_config_module
import vllm.distributed.parallel_state as parallel_state
from vllm.v1.engine import ReconfigureDistributedRequest, ReconfigureRankType

import vllm_fl.worker.worker as worker_module
from vllm_fl.worker.worker import WorkerFL


class _Patch:
    """Deterministic set/restore of a module attribute (R2 input boundary)."""

    def __init__(self, module, name, value):
        self.module = module
        self.name = name
        self.value = value

    def __enter__(self):
        self.saved = getattr(self.module, self.name)
        setattr(self.module, self.name, self.value)
        return self

    def __exit__(self, *exc):
        setattr(self.module, self.name, self.saved)
        return False


def _reconfig_request(**kwargs):
    """Build a ReconfigureDistributedRequest across vllm versions.

    The dev venv's msgspec struct requires two extra fields the CI
    image's older struct lacks; defaults cover the former, signature
    filtering the latter.
    """
    full = {
        "new_data_parallel_master_port_list": None,
        "coord_store_port": None,
    }
    full.update(kwargs)
    params = inspect.signature(ReconfigureDistributedRequest).parameters
    return ReconfigureDistributedRequest(
        **{k: v for k, v in full.items() if k in params}
    )


def _patch_ep_group(rank, world_size):
    return _Patch(
        parallel_state, "get_ep_group",
        lambda: SimpleNamespace(rank=rank, world_size=world_size),
    )


def test_reinitialize_distributed_shuts_down_evicted_rank():
    """A rank evicted by the shrink tears down without re-initializing."""
    calls = {"cleanup": 0, "init": 0, "scale_down": 0, "reconf_moe": 0}

    def scale_down(old, new):
        calls["scale_down"] += 1

    def reconf_moe(old, new):
        calls["reconf_moe"] += 1
        return None

    worker = SimpleNamespace(
        vllm_config=SimpleNamespace(
            parallel_config=SimpleNamespace(
                data_parallel_size=1,
                data_parallel_rank=0,
                data_parallel_rank_local=0,
                data_parallel_master_ip="i",
                data_parallel_master_port=1,
            )
        ),
        rank=3,
        distributed_init_method="tcp://",
        local_rank=0,
        _eplb_before_scale_down=scale_down,
        _reconfigure_moe=reconf_moe,
    )

    with _patch_ep_group(rank=3, world_size=4), _Patch(
        parallel_state, "cleanup_dist_env_and_memory",
        lambda **kw: calls.__setitem__("cleanup", calls["cleanup"] + 1),
    ), _Patch(
        parallel_state, "get_tp_group", lambda: SimpleNamespace(world_size=1),
    ), _Patch(
        parallel_state, "get_pp_group", lambda: SimpleNamespace(world_size=1),
    ), _Patch(
        worker_module, "get_tp_group", lambda: SimpleNamespace(world_size=1),
    ), _Patch(
        worker_module, "get_pp_group", lambda: SimpleNamespace(world_size=1),
    ), _Patch(
        vllm_config_module, "set_current_vllm_config", lambda cfg: nullcontext(),
    ), _Patch(
        worker_module, "init_worker_distributed_environment",
        lambda *a, **k: calls.__setitem__("init", calls["init"] + 1),
    ):
        WorkerFL.reinitialize_distributed(
            worker,
            _reconfig_request(
                new_data_parallel_size=2,
                new_data_parallel_rank=ReconfigureRankType.SHUTDOWN_CURRENT_RANK,
                new_data_parallel_rank_local=ReconfigureRankType.KEEP_CURRENT_RANK,
                new_data_parallel_master_ip="a",
                new_data_parallel_master_port=1,
            ),
        )

    assert calls["cleanup"] == 1
    assert calls["scale_down"] == 1
    assert calls["init"] == 0
    assert calls["reconf_moe"] == 0


def test_reinitialize_distributed_growth_reinitializes_and_scales_up():
    """A growing EP world tears down, re-inits, and runs the scale-up hook."""
    calls = {"cleanup": 0, "init": 0, "reconf_moe": 0, "scale_up": 0}

    def reconf_moe(old, new):
        calls["reconf_moe"] += 1
        return ["loads"]

    def scale_up(old, new, loads):
        calls["scale_up"] += 1
        assert (old, new, loads) == (4, 8, ["loads"])

    worker = SimpleNamespace(
        vllm_config=SimpleNamespace(
            parallel_config=SimpleNamespace(
                data_parallel_size=1,
                data_parallel_rank=0,
                data_parallel_rank_local=0,
                data_parallel_master_ip="i",
                data_parallel_master_port=1,
            )
        ),
        rank=0,
        distributed_init_method="tcp://",
        local_rank=0,
        _reconfigure_moe=reconf_moe,
        _eplb_after_scale_up=scale_up,
        _reconfigure_parallel_config=lambda req: calls.setdefault(
            "reconf_cfg", req
        ),
    )

    with _patch_ep_group(rank=0, world_size=4), _Patch(
        parallel_state, "cleanup_dist_env_and_memory",
        lambda **kw: calls.__setitem__("cleanup", calls["cleanup"] + 1),
    ), _Patch(
        parallel_state, "get_tp_group", lambda: SimpleNamespace(world_size=1),
    ), _Patch(
        parallel_state, "get_pp_group", lambda: SimpleNamespace(world_size=1),
    ), _Patch(
        worker_module, "get_tp_group", lambda: SimpleNamespace(world_size=1),
    ), _Patch(
        worker_module, "get_pp_group", lambda: SimpleNamespace(world_size=1),
    ), _Patch(
        vllm_config_module, "set_current_vllm_config", lambda cfg: nullcontext(),
    ), _Patch(
        worker_module, "init_worker_distributed_environment",
        lambda *a, **k: calls.__setitem__("init", calls["init"] + 1),
    ):
        WorkerFL.reinitialize_distributed(
            worker,
            _reconfig_request(
                new_data_parallel_size=8,
                new_data_parallel_rank=ReconfigureRankType.KEEP_CURRENT_RANK,
                new_data_parallel_rank_local=ReconfigureRankType.KEEP_CURRENT_RANK,
                new_data_parallel_master_ip="b",
                new_data_parallel_master_port=2,
            ),
        )

    assert calls["cleanup"] == 1
    assert calls["init"] == 1
    assert calls["reconf_moe"] == 1
    assert calls["scale_up"] == 1
    assert calls["reconf_cfg"] is not None
