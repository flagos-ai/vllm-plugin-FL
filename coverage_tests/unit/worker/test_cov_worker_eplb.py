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

"""Coverage tests for the vllm_fl/worker/worker.py elastic-EP paths.

``_eplb_before_scale_down`` / ``_eplb_after_scale_up`` compute the expert
rank mapping for an EP resize and drive the EPLB state;
``_reconfigure_parallel_config`` applies a ReconfigureDistributedRequest
to the parallel config; ``_reconfigure_moe`` resizes the MoE modules
(broadcasting the local expert count on scale-up, deriving redundant
experts from the EPLB state on scale-down).

Per CONTRACT R2 the vllm parallel-state groups, torch.distributed, the
EPLB state, and the fused-MoE modules are inputs replaced with recording
fakes; the branch selection itself is driven only by the old/new EP sizes
and the request's rank fields. The unbound-method-on-SimpleNamespace
pattern follows the dev tests in tests/unit_tests/worker/.
"""

import inspect
from types import SimpleNamespace

import torch
import torch.nn as nn

# The shim import is load-bearing: it must precede vllm_fl/vllm
# imports on hosts without triton. Do not alphabetize.
# isort: split
from . import _imports  # noqa: F401  (installs local import shims first)

# isort: split
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

    The dev venv's msgspec struct requires two extra fields
    (new_data_parallel_master_port_list, coord_store_port) the CI image's
    older struct lacks; defaults cover the former, signature filtering
    the latter.
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


class _RecordingEplb:
    """Fake EPLB state recording rearrange calls (args + kwargs)."""

    def __init__(self, result=None):
        self.calls = []
        self.result = result

    def rearrange(self, *args, **kwargs):
        self.calls.append((args, kwargs))
        return self.result


def _patch_ep_group(rank=0, world_size=2):
    return _Patch(
        parallel_state,
        "get_ep_group",
        lambda: SimpleNamespace(rank=rank, world_size=world_size),
    )


def test_eplb_before_scale_down_maps_old_ranks_to_new_ep_size():
    """Scale-down keeps ranks below the new size and evicts the rest."""
    eplb = _RecordingEplb()
    synced = []
    worker = SimpleNamespace(
        model_runner=SimpleNamespace(eplb_state=eplb, model="MODEL")
    )
    platform = SimpleNamespace(
        torch_device_fn=SimpleNamespace(synchronize=lambda: synced.append(1)),
        device_type="fake",
    )
    with _patch_ep_group(rank=0), _Patch(worker_module, "current_platform", platform):
        WorkerFL._eplb_before_scale_down(worker, old_ep_size=4, new_ep_size=2)

    assert len(eplb.calls) == 1
    args, kwargs = eplb.calls[0]
    assert args == (worker.model_runner.model,)
    assert kwargs["execute_shuffle"] is True
    assert kwargs["global_expert_loads"] is None
    assert kwargs["rank_mapping"] == {0: 0, 1: 1, 2: -1, 3: -1}
    assert synced == [1]


def test_eplb_after_scale_up_keeps_all_old_ranks():
    """Scale-up keeps every existing rank and forwards the expert loads."""
    eplb = _RecordingEplb()
    loads = torch.ones(2, 8)
    worker = SimpleNamespace(
        model_runner=SimpleNamespace(eplb_state=eplb, model="MODEL")
    )
    with _patch_ep_group(rank=0):
        WorkerFL._eplb_after_scale_up(
            worker, old_ep_size=2, new_ep_size=4, global_expert_loads=loads
        )

    assert len(eplb.calls) == 1
    _, kwargs = eplb.calls[0]
    assert kwargs["execute_shuffle"] is True
    assert kwargs["global_expert_loads"] is loads
    assert kwargs["rank_mapping"] == {0: 0, 1: 1}


def test_reconfigure_parallel_config_updates_all_fields():
    """A full request rewrites size, rank, local rank, ip, and port."""
    parallel_config = SimpleNamespace(
        data_parallel_size=1,
        data_parallel_rank=0,
        data_parallel_rank_local=0,
        data_parallel_master_ip="old-ip",
        data_parallel_master_port=1,
    )
    worker = SimpleNamespace(
        vllm_config=SimpleNamespace(parallel_config=parallel_config)
    )
    WorkerFL._reconfigure_parallel_config(
        worker,
        _reconfig_request(
            new_data_parallel_size=4,
            new_data_parallel_rank=3,
            new_data_parallel_rank_local=1,
            new_data_parallel_master_ip="new-ip",
            new_data_parallel_master_port=7777,
        ),
    )
    assert parallel_config.data_parallel_size == 4
    assert parallel_config.data_parallel_rank == 3
    assert parallel_config.data_parallel_rank_local == 1
    assert parallel_config.data_parallel_master_ip == "new-ip"
    assert parallel_config.data_parallel_master_port == 7777


def test_reconfigure_parallel_config_preserves_ranks_on_keep_current():
    """KEEP_CURRENT_RANK leaves rank and local rank untouched."""
    parallel_config = SimpleNamespace(
        data_parallel_size=1,
        data_parallel_rank=2,
        data_parallel_rank_local=1,
        data_parallel_master_ip="old-ip",
        data_parallel_master_port=1,
    )
    worker = SimpleNamespace(
        vllm_config=SimpleNamespace(parallel_config=parallel_config)
    )
    WorkerFL._reconfigure_parallel_config(
        worker,
        _reconfig_request(
            new_data_parallel_size=4,
            new_data_parallel_rank=ReconfigureRankType.KEEP_CURRENT_RANK,
            new_data_parallel_rank_local=ReconfigureRankType.KEEP_CURRENT_RANK,
            new_data_parallel_master_ip="new-ip",
            new_data_parallel_master_port=8888,
        ),
    )
    assert parallel_config.data_parallel_rank == 2
    assert parallel_config.data_parallel_rank_local == 1
    assert parallel_config.data_parallel_size == 4
    assert parallel_config.data_parallel_master_ip == "new-ip"
    assert parallel_config.data_parallel_master_port == 8888


class FusedMoE(nn.Module):
    """Fake fused-MoE layer matched by its class name (as the source does)."""

    def __init__(self, num_local, num_experts):
        super().__init__()
        self.moe_config = SimpleNamespace(
            num_local_experts=num_local,
            num_experts=num_experts,
            moe_parallel_config=None,
        )
        self.global_num_experts = num_experts
        self.moe_parallel_config = None


class SharedFusedMoE(FusedMoE):
    pass


class _MoEModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.moe1 = FusedMoE(4, 4)
        self.moe2 = SharedFusedMoE(4, 4)
        self.linear = nn.Module()

    def update_physical_experts_metadata(self, **kwargs):
        self.updates = kwargs


class _FakeGroups:
    tp = SimpleNamespace(world_size=2)
    pcp = SimpleNamespace(world_size=1)
    dp = SimpleNamespace(world_size=1, rank_in_group=0)
    ep = SimpleNamespace(world_size=2, cpu_group="cpu-group-obj")


def _patched_moe_modules(model, broadcast=None):
    """Patch every collaborator _reconfigure_moe reads (R2 inputs).

    worker.py imports get_pp/get_tp/get_pcp at module level; get_dp and
    get_ep are imported inside the method body from
    vllm.distributed.parallel_state, so both namespaces need patching.
    """
    import vllm.model_executor.layers.fused_moe.config as moe_config

    patches = [
        _Patch(worker_module, "get_tp_group", lambda: _FakeGroups.tp),
        _Patch(worker_module, "get_pcp_group", lambda: _FakeGroups.pcp),
        _Patch(parallel_state, "get_tp_group", lambda: _FakeGroups.tp),
        _Patch(parallel_state, "get_pcp_group", lambda: _FakeGroups.pcp),
        _Patch(parallel_state, "get_dp_group", lambda: _FakeGroups.dp),
        _Patch(parallel_state, "get_ep_group", lambda: _FakeGroups.ep),
        _Patch(
            parallel_state,
            "prepare_communication_buffer_for_model",
            lambda m: getattr(m, "prep_marker", None),
        ),
        _Patch(moe_config, "get_tensor_model_parallel_rank", lambda: 0),
        _Patch(moe_config, "get_dp_group", lambda: _FakeGroups.dp),
        _Patch(moe_config, "get_pcp_group", lambda: _FakeGroups.pcp),
    ]
    if broadcast is not None:
        patches.append(_Patch(torch.distributed, "broadcast", broadcast))
    return patches


def _enter_all(patches):
    for patch in patches:
        patch.__enter__()


def _exit_all(patches):
    for patch in reversed(patches):
        patch.__exit__()


def test_reconfigure_moe_scale_up_doubles_experts_and_broadcasts():
    """Scale-up updates MoE configs, broadcasts the local count, reshards."""
    model = _MoEModel()
    loads = [torch.zeros(1, 6), torch.zeros(1, 3)]
    eplb = _RecordingEplb(result=loads)
    parallel_config = SimpleNamespace(
        enable_expert_parallel=True,
        all2all_backend="none",
        enable_eplb=True,
        eplb_config=SimpleNamespace(num_redundant_experts=0),
    )
    worker = SimpleNamespace(
        vllm_config=SimpleNamespace(parallel_config=parallel_config),
        model_runner=SimpleNamespace(model=model, eplb_state=eplb),
    )

    broadcasts = []

    def fake_broadcast(tensor, group=None, group_src=0):
        tensor.fill_(4)
        broadcasts.append((tuple(tensor.shape), group, group_src))

    patches = _patched_moe_modules(model, fake_broadcast)
    _enter_all(patches)
    try:
        result = WorkerFL._reconfigure_moe(worker, old_ep_size=1, new_ep_size=2)
    finally:
        _exit_all(patches)

    assert result is loads
    for module in (model.moe1, model.moe2):
        assert module.moe_config.num_experts == 8
        assert module.global_num_experts == 8
        assert module.moe_parallel_config is not None
        assert module.moe_config.moe_parallel_config is module.moe_parallel_config
    # The local expert count was broadcast from ep rank 0 over the cpu group.
    assert broadcasts == [((1,), "cpu-group-obj", 0)]
    # eplb_state.rearrange ran without shuffling to compute the loads.
    assert eplb.calls == [((), {"execute_shuffle": False})]
    # new_physical = broadcast(4) * new_ep(2) = 8; redundant = 8 - loads[0].shape[1].
    assert parallel_config.eplb_config.num_redundant_experts == 2
    assert model.updates == {
        "num_physical_experts": 8,
        "num_local_physical_experts": 4,
    }


def test_reconfigure_moe_scale_down_derives_redundant_from_eplb_state():
    """Scale-down skips the broadcast and derives redundancy from maps."""
    model = _MoEModel()
    parallel_config = SimpleNamespace(
        enable_expert_parallel=True,
        all2all_backend="none",
        enable_eplb=True,
        eplb_config=SimpleNamespace(num_redundant_experts=99),
    )
    worker = SimpleNamespace(
        vllm_config=SimpleNamespace(parallel_config=parallel_config),
        model_runner=SimpleNamespace(
            model=model,
            eplb_state=SimpleNamespace(
                rearrange=lambda **kw: None,
                physical_to_logical_map=torch.zeros(2, 10),
                logical_replica_count=torch.zeros(2, 6),
            ),
        ),
    )

    broadcasts = []

    def fail_broadcast(*args, **kwargs):
        broadcasts.append("must-not-happen")

    patches = _patched_moe_modules(model, fail_broadcast)
    _enter_all(patches)
    try:
        result = WorkerFL._reconfigure_moe(worker, old_ep_size=4, new_ep_size=2)
    finally:
        _exit_all(patches)

    assert result is None
    assert broadcasts == []
    # redundant = physical_to_logical_map cols (10) - logical_replica cols (6).
    assert parallel_config.eplb_config.num_redundant_experts == 4
    assert model.updates == {
        "num_physical_experts": 10,
        "num_local_physical_experts": 4,
    }
    assert model.moe1.moe_config.num_experts == 8
