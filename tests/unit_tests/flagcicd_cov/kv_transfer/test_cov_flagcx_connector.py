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
"""Coverage tests for vllm_fl/distributed/kv_transfer/flagcx_connector.py.

The NVIDIA CI image ships no flagcx package, so the flagcx boundary is
faked with a wheel facade in sys.modules (the sanctioned mock style from
tests/unit_tests/distributed/test_flagcx.py) and every expected value is
derived in closed form from the inputs. These tests cover the import
ladder, the module-level transfer-planning helpers, the scheduler->worker
metadata records, and the scheduler-side bookkeeping.
"""

from __future__ import annotations

import importlib
import sys
import types

import msgspec
import pytest

from . import _flagcx_env
from . import _flagcx_helpers as helpers

_flagcx_env.ensure_importable_env()


@pytest.fixture()
def conn(monkeypatch):
    """Import the connector against the fake flagcx wheel facade.

    Drops any cached copy of the module first so the import ladder runs
    against this test's facade, and drops it again on teardown so the
    next test re-runs the ladder. Yields (module, call recorder).
    """
    recorder = _flagcx_env.install_fake_flagcx_wheel(monkeypatch)
    _flagcx_env.ensure_vllm_api_compat()
    _flagcx_env.drop_connector_module()
    import vllm_fl.distributed.kv_transfer.flagcx_connector as conn

    yield conn, recorder
    _flagcx_env.drop_connector_module()


def test_import_ladder_binds_wheel_facade(conn):
    """With a flagcx wheel present the ladder binds flagcx.api, marks
    the wheel path, and constructs FLAGCXLibrary with a None path."""
    conn, recorder = conn
    assert conn._flagcx_is_wheel is True
    assert conn._flagcx_module.__name__ == "flagcx.api"
    assert conn.FLAGCXLibrary is not None

    # Constructing the library through the bound symbol records the call
    # with library_path=None: the wheel resolves the .so itself.
    lib = conn.FLAGCXLibrary(None)
    assert recorder[0] == ("FLAGCXLibrary", None)
    assert lib.rpc_port == 4242


def _source_tree_wrapper(monkeypatch, root):
    """Install plugin.interservice.flagcx_wrapper as a source-tree path."""
    wrapper = types.ModuleType("plugin.interservice.flagcx_wrapper")
    wrapper.FLAGCXLibrary = lambda path=None: object()
    interservice = types.ModuleType("plugin.interservice")
    interservice.__path__ = []
    plugin = types.ModuleType("plugin")
    plugin.__path__ = []

    monkeypatch.setitem(sys.modules, "plugin", plugin)
    monkeypatch.setitem(sys.modules, "plugin.interservice", interservice)
    monkeypatch.setitem(sys.modules, "plugin.interservice.flagcx_wrapper", wrapper)
    return wrapper


def test_import_ladder_wheel_beats_source_tree(conn, monkeypatch, tmp_path):
    """A flagcx wheel takes precedence over FLAGCX_PATH source trees."""
    conn, _ = conn
    source = tmp_path / "flagcx-src"
    source.mkdir()
    _source_tree_wrapper(monkeypatch, source)
    monkeypatch.setenv("FLAGCX_PATH", str(source))

    _flagcx_env.drop_connector_module()
    try:
        fresh = importlib.import_module(
            "vllm_fl.distributed.kv_transfer.flagcx_connector"
        )
        assert fresh._flagcx_is_wheel is True
        assert fresh._flagcx_module.__name__ == "flagcx.api"
    finally:
        _flagcx_env.drop_connector_module()


def test_module_constants(conn):
    """The wire protocol status markers are fixed byte literals."""
    conn, _ = conn
    assert conn.TRANS_DONE == b"trans_done"
    assert conn.TRANS_ERROR == b"trans_error"
    assert conn.NULL_BLOCK_ID == 0


def test_agent_metadata_msgpack_roundtrip(conn):
    """FlagCXAgentMetadata round-trips through msgspec msgpack with the
    per-request block table preserved."""
    conn, _ = conn
    meta = conn.FlagCXAgentMetadata(
        remote_hostname="10.0.0.2",
        remote_port=4242,
        remote_tp_size=2,
        remote_tp_rank=1,
        req_blocks={
            "req-7": ("t-7", [[1, 2], [3]]),
            "req-8": ("t-8", []),
        },
        kv_caches_base_addr=[0x1000, 0x2000],
        block_lens=[4096, 4096],
        kv_block_lens=[2048, 2048],
        registered_layer_names=["model.layers.0.self_attn"],
        registered_layer_indices=[0],
        registered_group_indices=[0],
    )
    blob = msgspec.msgpack.encode(meta)
    loaded = msgspec.msgpack.decode(blob, type=conn.FlagCXAgentMetadata)
    assert loaded.remote_hostname == "10.0.0.2"
    assert loaded.req_blocks["req-7"] == ("t-7", [[1, 2], [3]])
    assert loaded.kv_caches_base_addr == [0x1000, 0x2000]
    assert loaded.registered_group_indices == [0]


def test_send_block_meta_defaults(conn):
    """A new SendBlockMeta starts un-armed and un-sent with no deadline."""
    conn, _ = conn
    m = conn.SendBlockMeta(
        p_req_id="p1", transfer_id="t1", local_block_ids=[[1]], ready=helpers.event()
    )
    assert m.expire_time == float("inf")
    assert m.need_send == 0
    assert m.sent == 0
    assert not m.ready.is_set()


def test_metadata_add_new_req_recv_branch(conn):
    """load_remote_cache=True records a RecvReqMeta with the remote
    endpoint and the transfer id from the scheduler params."""
    conn, _ = conn
    meta = conn.FlagCXConnectorMetadata()
    meta.add_new_req(
        request_id="r1",
        local_block_ids=[[1, 2]],
        kv_transfer_params={
            "transfer_id": "t1",
            "remote_host": "10.0.0.1",
            "remote_port": 9000,
            "remote_tp_size": 4,
        },
    )
    recv = meta.reqs_to_recv["r1"]
    assert recv.local_block_ids == [[1, 2]]
    assert recv.remote_host == "10.0.0.1"
    assert recv.remote_port == 9000
    assert recv.transfer_id == "t1"
    assert recv.remote_tp_size == 4
    assert meta.reqs_to_send == {}


def test_metadata_add_new_req_send_branch(conn):
    """load_remote_cache=False records a send tuple keyed by request."""
    conn, _ = conn
    meta = conn.FlagCXConnectorMetadata()
    meta.add_new_req(
        request_id="r2",
        local_block_ids=[[5], [6]],
        kv_transfer_params={"transfer_id": "t2"},
        load_remote_cache=False,
    )
    assert meta.reqs_to_send["r2"] == ("t2", [[5], [6]])
    assert meta.reqs_to_recv == {}


def test_get_required_kvcache_layout(conn):
    """The layout is HND for standard models, None for MLA or missing
    model configs."""
    conn, _ = conn
    assert conn.FlagCXConnector.get_required_kvcache_layout(
        helpers.vllm_config(model=None)
    ) is None
    assert conn.FlagCXConnector.get_required_kvcache_layout(
        helpers.vllm_config(model=helpers.ModelInfo(use_mla=True))
    ) is None
    assert (
        conn.FlagCXConnector.get_required_kvcache_layout(
            helpers.vllm_config(model=helpers.ModelInfo(use_mla=False))
        )
        == "HND"
    )


def test_get_side_channel_port_default_and_env(conn, monkeypatch):
    """The side-channel port is the bootstrap base plus dp_rank * tp_size,
    with the base overridable through FLAGCX_BOOTSTRAP_PORT."""
    conn, _ = conn
    monkeypatch.delenv("FLAGCX_BOOTSTRAP_PORT", raising=False)
    assert conn._get_side_channel_port(
        helpers.vllm_config(parallel=helpers.ParallelInfo(tp_size=2, dp_rank=0))
    ) == 8998
    assert conn._get_side_channel_port(
        helpers.vllm_config(parallel=helpers.ParallelInfo(tp_size=4, dp_rank=3))
    ) == 8998 + 12

    monkeypatch.setenv("FLAGCX_BOOTSTRAP_PORT", "7000")
    assert conn._get_side_channel_port(
        helpers.vllm_config(parallel=helpers.ParallelInfo(tp_size=1, dp_rank=1))
    ) == 7001
