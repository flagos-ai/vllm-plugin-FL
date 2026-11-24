# Copyright 2026 FlagOS Contributors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# SPDX-License-Identifier: Apache-2.0
"""Shared CPU bootstrap for the w5a FlagCX connector coverage tests.

The MUSA test container ships no real FlagCX shared object and no network
peer, so the native boundary is replaced by a recording facade installed
into ``sys.modules`` (the sanctioned external-resource mock used by
``tests/unit_tests/distributed/test_flagcx.py``).  Everything below is a
plain-CPU stand-in: config slices, KV-cache-spec markers, a fake
``TransferTopology`` and the per-test module loader.

The heavy production constructors are never skipped: vLLM's real
``RequestStatus``/msgspec/torch/zmq objects are used where the connector
logic reads them; only the parts that would need an accelerator, a
collective, or a native library are doubled.
"""

from __future__ import annotations

import importlib
import sys
import types

# ---------------------------------------------------------------------------
# Native FlagCX facade
# ---------------------------------------------------------------------------


class Recorder(list):
    """Ordered call log shared by the fake library/engine/connection."""

    def record(self, name, *args):
        self.append((name,) + args)


class FakeP2PEngine:
    def __init__(self, recorder, engine_id):
        self.recorder = recorder
        self.engine_id = engine_id


class FakeP2PConn:
    def __init__(self, session):
        self.session = session


class FakeFlagCXLibrary:
    """Recording stand-in for the connector's ``FLAGCXLibrary`` facade."""

    def __init__(self, recorder):
        self.recorder = recorder
        self.rpc_port = 4242
        self.write_should_raise = None
        self.library_path = None

    def flagcxP2pEngineCreate(self):
        self.recorder.record("flagcxP2pEngineCreate")
        return FakeP2PEngine(self.recorder, engine_id=len(self.recorder))

    def flagcxP2pEngineDestroy(self, engine):
        self.recorder.record("flagcxP2pEngineDestroy", engine.engine_id)

    def flagcxP2pGetConn(self, engine, session):
        self.recorder.record("flagcxP2pGetConn", session)
        return FakeP2PConn(session)

    def flagcxP2pGetRpcPort(self, engine):
        self.recorder.record("flagcxP2pGetRpcPort")
        return self.rpc_port

    def flagcxP2pStartRpcServer(self, engine):
        self.recorder.record("flagcxP2pStartRpcServer")

    def flagcxP2pRegister(self, engine, base_addr, size):
        self.recorder.record("flagcxP2pRegister", base_addr, size)

    def flagcxP2pBatchWriteSync(self, conn, source_addresses, target_addresses, sizes):
        self.recorder.record(
            "flagcxP2pBatchWriteSync",
            conn.session,
            list(source_addresses),
            list(sizes),
        )
        if self.write_should_raise is not None:
            raise self.write_should_raise


def install_fake_flagcx(monkeypatch) -> Recorder:
    """Install the recording facade under the plugin wrapper module path."""
    recorder = Recorder()

    def factory(library_path=None):
        recorder.record("FLAGCXLibrary", library_path)
        library = FakeFlagCXLibrary(recorder)
        library.library_path = library_path
        return library

    wrapper = types.ModuleType("plugin.interservice.flagcx_wrapper")
    wrapper.FLAGCXLibrary = factory
    interservice = types.ModuleType("plugin.interservice")
    interservice.__path__ = []
    interservice.flagcx_wrapper = wrapper
    plugin = types.ModuleType("plugin")
    plugin.__path__ = []
    plugin.interservice = interservice

    monkeypatch.setitem(sys.modules, "plugin", plugin)
    monkeypatch.setitem(sys.modules, "plugin.interservice", interservice)
    monkeypatch.setitem(
        sys.modules, "plugin.interservice.flagcx_wrapper", wrapper
    )
    return recorder


# ---------------------------------------------------------------------------
# Config / spec / request stand-ins
# ---------------------------------------------------------------------------


class FakeFullSpec:
    """Marker for FullAttentionSpec (only isinstance is consulted)."""


class FakeSlidingSpec:
    def __init__(self, sliding_window, block_size):
        self.sliding_window = sliding_window
        self.block_size = block_size


class FakeMambaSpec:
    def __init__(self, page_size_bytes=64, block_size=16):
        self.page_size_bytes = page_size_bytes
        self.block_size = block_size


class FakeMLASpec:
    def __init__(self, page_size_bytes=256, block_size=16):
        self.page_size_bytes = page_size_bytes
        self.block_size = block_size


class FakeSWMLASpec(FakeMLASpec):
    pass


class ModelInfo:
    def __init__(self, use_mla=False, total_num_kv_heads=8):
        self.use_mla = use_mla
        self._total_num_kv_heads = total_num_kv_heads

    def get_total_num_kv_heads(self):
        return self._total_num_kv_heads


class ParallelInfo:
    def __init__(self, tp_size=1, dp_rank=0):
        self.tensor_parallel_size = tp_size
        self.data_parallel_rank = dp_rank


class TransferInfo:
    def __init__(self, kv_role="kv_both", engine_id="eng-test", extra_config=None):
        self.kv_role = kv_role
        self.engine_id = engine_id
        self.kv_connector_extra_config = dict(extra_config or {})


class CacheInfo:
    def __init__(self, block_size=16):
        self.block_size = block_size


class SchedulerInfo:
    def __init__(self, disable_hybrid_kv_cache_manager=True):
        self.disable_hybrid_kv_cache_manager = disable_hybrid_kv_cache_manager


class CompilationInfo:
    def __init__(self, forward_context=None):
        self.static_forward_context = dict(forward_context or {})


class VllmConfigView:
    def __init__(self, parts):
        self.__dict__.update(parts)


def vllm_config(
    model="default",
    parallel=None,
    transfer=None,
    cache=None,
    scheduler=None,
    compilation=None,
):
    return VllmConfigView(
        {
            "model_config": ModelInfo() if model == "default" else model,
            "parallel_config": parallel if parallel is not None else ParallelInfo(),
            "kv_transfer_config": (
                transfer if transfer is not None else TransferInfo()
            ),
            "cache_config": cache if cache is not None else CacheInfo(),
            "scheduler_config": (
                scheduler if scheduler is not None else SchedulerInfo()
            ),
            "compilation_config": (
                compilation if compilation is not None else CompilationInfo()
            ),
        }
    )


class FakeKVGroup:
    def __init__(self, spec, layer_names):
        self.kv_cache_spec = spec
        self.layer_names = list(layer_names)


class FakeKVCacheConfig:
    def __init__(self, groups, has_mamba_layers=False):
        self.kv_cache_groups = list(groups)
        self.has_mamba_layers = has_mamba_layers


def full_group(layer=0):
    return FakeKVGroup(FakeFullSpec(), [f"model.layers.{layer}.self_attn"])


def sliding_group(layer=1, sliding_window=32, block_size=16):
    spec = FakeSlidingSpec(sliding_window, block_size)
    return FakeKVGroup(spec, [f"model.layers.{layer}.self_attn"])


def mamba_group(layer=2, page_size_bytes=64):
    return FakeKVGroup(FakeMambaSpec(page_size_bytes), [f"model.layers.{layer}.mamba"])


class FakeRequest:
    def __init__(
        self,
        request_id="req-1",
        prompt_token_ids="default",
        kv_transfer_params=None,
        status=None,
        num_prompt_tokens=None,
        max_tokens=16,
        prompt_embeds=None,
    ):
        self.request_id = request_id
        if prompt_token_ids == "default":
            prompt_token_ids = [1, 2, 3, 4]
        self.prompt_token_ids = prompt_token_ids
        self.prompt_embeds = prompt_embeds
        self.kv_transfer_params = kv_transfer_params
        self.status = status
        self.num_prompt_tokens = (
            num_prompt_tokens
            if num_prompt_tokens is not None
            else len(prompt_token_ids or [])
        )
        self.max_tokens = max_tokens
        self._all_token_ids = (
            [0] * self.num_prompt_tokens
            if self.prompt_token_ids is None
            else list(self.prompt_token_ids)
        )


class FakeBlocks:
    def __init__(self, unhashed_all_groups):
        self._unhashed = unhashed_all_groups

    def get_unhashed_block_ids_all_groups(self):
        return [list(group) for group in self._unhashed]


# ---------------------------------------------------------------------------
# Worker-side topology / backend stand-ins
# ---------------------------------------------------------------------------


class FakeBackend:
    @staticmethod
    def get_name():
        return "FAKE_ATTN"


# Mutable knobs read by the fake topology factory.  ``load`` resets them.
TOPO_STATE = {
    "blocks_first": False,
    "replicated": None,
    "targets": None,
    "regions": None,
}
TP_STATE = {"rank": 0, "size": 1}


class FakeTopology:
    def __init__(
        self,
        tp_rank,
        tp_size,
        block_size,
        engine_id,
        is_mla,
        is_mamba,
        total_num_kv_heads,
        attn_backends,
    ):
        self.tp_rank = tp_rank
        self.tp_size = tp_size
        self.block_size = block_size
        self.engine_id = engine_id
        self.is_mla = is_mla
        self.is_mamba = is_mamba
        self.attn_backends = attn_backends
        self.is_kv_layout_blocks_first = TOPO_STATE["blocks_first"]
        if TOPO_STATE["replicated"] is None:
            self.local_replicates_kv_cache = is_mla or total_num_kv_heads < tp_size
        else:
            self.local_replicates_kv_cache = TOPO_STATE["replicated"]

    def handshake_target_ranks(self, remote_tp_size):
        if TOPO_STATE["targets"] is not None:
            return TOPO_STATE["targets"](self.tp_rank, self.tp_size, remote_tp_size)
        if self.tp_size >= remote_tp_size:
            return [self.tp_rank % remote_tp_size]
        ratio = remote_tp_size // self.tp_size
        return [self.tp_rank * ratio + i for i in range(ratio)]

    def get_transfer_cache_regions(self, cache, spec):
        if TOPO_STATE["regions"] is not None:
            return TOPO_STATE["regions"](cache, spec, self)
        if isinstance(spec, FakeMambaSpec):
            return [cache[0]] if isinstance(cache, (list, tuple)) else [cache]
        if isinstance(spec, (FakeMLASpec, FakeSWMLASpec)):
            return [cache]
        if self.is_kv_layout_blocks_first:
            return [cache]
        if isinstance(cache, (list, tuple)):
            return list(cache)
        return [cache[0], cache[1]]


def load_connector(monkeypatch):
    """Import the connector fresh against the fake facade.

    Returns ``(conn, recorder)``.  After import the accelerator-only
    collaborators are rebound to CPU doubles so production ``__init__``
    bodies still execute.
    """
    recorder = install_fake_flagcx(monkeypatch)
    import vllm.v1.attention.backends.utils as _attn_utils
    if not hasattr(_attn_utils, "get_kv_cache_layout"):
        _attn_utils.get_kv_cache_layout = lambda *a, **k: "NHD"
    sys.modules.pop("vllm_fl.distributed.kv_transfer.flagcx_connector", None)
    conn = importlib.import_module(
        "vllm_fl.distributed.kv_transfer.flagcx_connector"
    )

    # Spec markers: the real classes are only used for isinstance and one
    # attribute read, so stable local markers exercise every branch.
    conn.FullAttentionSpec = FakeFullSpec
    conn.SlidingWindowSpec = FakeSlidingSpec
    conn.MambaSpec = FakeMambaSpec
    conn.MLAAttentionSpec = FakeMLASpec
    conn.SlidingWindowMLASpec = FakeSWMLASpec

    # CPU collaborators.
    conn.get_ip = lambda: "127.0.0.1"
    conn.get_tensor_model_parallel_rank = lambda: TP_STATE["rank"]
    conn.get_tensor_model_parallel_world_size = lambda: TP_STATE["size"]
    conn.get_current_attn_backends = lambda cfg: [FakeBackend]
    conn.select_common_block_size = lambda size, backends: size
    conn.get_kv_cache_layout = lambda: "NHD"
    conn.TransferTopology = FakeTopology
    return conn, recorder


def reset_topology_state(blocks_first=False):
    TOPO_STATE["blocks_first"] = blocks_first
    TOPO_STATE["replicated"] = None
    TOPO_STATE["targets"] = None
    TOPO_STATE["regions"] = None


def make_worker(conn, role="kv_both", groups=None, tp_size=1, tp_rank=0, **cfg_kw):
    """Construct a real FlagCXConnectorWorker with CPU doubles."""
    TP_STATE["rank"] = tp_rank
    TP_STATE["size"] = tp_size
    if groups is None:
        groups = [full_group(0)]
    cfg = vllm_config(
        model=ModelInfo(
            use_mla=cfg_kw.get("use_mla", False),
            total_num_kv_heads=cfg_kw.get("total_kv_heads", 8),
        ),
        parallel=ParallelInfo(tp_size=tp_size, dp_rank=cfg_kw.get("dp_rank", 0)),
        transfer=TransferInfo(
            kv_role=role,
            extra_config=cfg_kw.get("extra_config"),
        ),
        cache=CacheInfo(block_size=cfg_kw.get("block_size", 16)),
        scheduler=SchedulerInfo(),
    )
    return conn.FlagCXConnectorWorker(
        cfg, "eng-w", FakeKVCacheConfig(groups, cfg_kw.get("has_mamba", False))
    )
