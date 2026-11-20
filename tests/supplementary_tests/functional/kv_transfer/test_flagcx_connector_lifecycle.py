# Copyright (c) 2026 BAAI. All rights reserved.

"""GPU functional lifecycle tests for the FlagCX KV-transfer connector.

``vllm_fl/distributed/kv_transfer/flagcx_connector.py`` implements PD
disaggregation over the FlagCX P2P engine. Importing it requires the full
``vllm`` package and the FlagCX wrapper (``FLAGCX_PATH`` pointing at the
FlagCX repo root with ``plugin/interservice/flagcx_wrapper.py``), so this
file skips cleanly wherever either is missing (locally it always skips;
the A100 weekend workflow must set ``FLAGCX_PATH`` and provide
``build/lib/libflagcx.so`` or ``FLAGCX_LIB_PATH`` to run it).

Lifecycle stages covered, in the order the engine executes them:

1. **Layout contract** — ``FlagCXConnector.get_required_kvcache_layout``
   (HND for full attention, None for MLA) and the ZMQ side-channel port
   derivation;
2. **Scheduler-side accounting** — a real ``FlagCXConnectorScheduler``
   tracking one do_remote_prefill request through
   ``get_num_new_matched_tokens`` → ``update_state_after_alloc`` →
   ``build_connector_meta``, plus the Mamba/GDN N-1 token trick;
3. **Worker registration and shutdown** — a real FlagCX engine
   (``FLAGCXLibrary`` → ``flagcxP2pEngineCreate``) registering real GPU KV
   cache tensors via the production ``register_kv_caches`` and
   ``shutdown`` methods, for both the consumer (RPC server) and producer
   (sender thread) roles;
4. **Transfer planning** — the pure helpers
   (``group_concurrent_contiguous``, ``_get_tp_ratio``,
   ``_compute_sender_transfer_plan``, ``_expand_transfer_regions``,
   ``_align_transfer_regions``, ``_validate_asymmetric_region_lengths``,
   ``_build_transfer_params``) producing absolute-VA (src, dst, size)
   triples.

Design note: ``FlagCXConnectorWorker.__init__`` and the
``FlagCXConnector``/``KVConnectorBase_V1`` constructor read global engine
state (tensor-parallel rank, attention backend registry, KV cache layout)
that only exists inside a running vLLM engine, so the worker tests build
the component with ``object.__new__`` and populate exactly the state its
``__init__`` would leave behind, then drive the real production methods.
The FlagCX engine itself, the ZMQ contexts and the sender thread are all
real. The two-process handshake (``_receive_kv`` → ``_send_kv_to_decode``)
needs a live peer and is out of scope.
"""

import os
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

pytest.importorskip("vllm")
pytest.importorskip("zmq")

# Importing the connector requires the FlagCX wrapper; resolve it via
# FLAGCX_PATH exactly like the module does, and skip when unavailable.
_FLAGCX_PATH = os.getenv("FLAGCX_PATH")
if not (
    _FLAGCX_PATH
    and Path(_FLAGCX_PATH, "plugin", "interservice", "flagcx_wrapper.py").is_file()
):
    pytest.skip(
        "FlagCX wrapper not found: set FLAGCX_PATH to the FlagCX repo root",
        allow_module_level=True,
    )

import zmq  # noqa: E402
from plugin.interservice.flagcx_wrapper import FLAGCXLibrary  # noqa: E402

from vllm_fl.distributed.kv_transfer.flagcx_connector import (  # noqa: E402
    FlagCXConnector,
    FlagCXConnectorMetadata,
    FlagCXConnectorScheduler,
    FlagCXConnectorWorker,
    SendBlockMeta,
    TransferRegion,
    _align_transfer_regions,
    _can_coalesce_block_transfers,
    _compute_sender_transfer_plan,
    _expand_transfer_regions,
    _get_side_channel_port,
    _get_tp_ratio,
    _validate_asymmetric_region_lengths,
    group_concurrent_contiguous,
)

NUM_LAYERS = 2
NUM_KV_HEADS = 4
HEAD_SIZE = 64
BLOCK_SIZE = 16
LAYER_NAMES = [f"model.layers.{i}.self_attn.attn" for i in range(NUM_LAYERS)]


class _FakeRequest:
    """Minimal vLLM Request for the scheduler-side accounting methods."""

    def __init__(self, request_id, params, num_prompt_tokens, status=None):
        self.request_id = request_id
        self.kv_transfer_params = params
        self.num_prompt_tokens = num_prompt_tokens
        self.prompt_token_ids = list(range(num_prompt_tokens))
        self.prompt_embeds = None
        self._all_token_ids = list(range(num_prompt_tokens))
        self.max_tokens = 10
        self.status = status


class _FakeBlocks:
    """Stand-in for KVCacheBlocks.get_unhashed_block_ids_all_groups()."""

    def __init__(self, groups):
        self._groups = groups

    def get_unhashed_block_ids_all_groups(self):
        return self._groups


class _FakeTopo:
    """Duck-typed TransferTopology for the HND full-attention layout."""

    is_kv_layout_blocks_first = False
    local_replicates_kv_cache = False

    def get_transfer_cache_regions(self, caches, spec):
        del spec
        return [caches]


def _kv_cache_config():
    """Build the KVCacheConfig-shaped namespace the connector reads."""
    spec = SimpleNamespace(
        page_size_bytes=BLOCK_SIZE * NUM_KV_HEADS * HEAD_SIZE * 2,
        block_size=BLOCK_SIZE,
    )
    group = SimpleNamespace(
        kv_cache_spec=spec,
        layer_names=LAYER_NAMES,
    )
    return SimpleNamespace(
        kv_cache_groups=[group],
        has_mamba_layers=False,
    )


def _vllm_config(kv_role):
    """Build the VllmConfig-shaped namespace the connector reads."""
    return SimpleNamespace(
        model_config=SimpleNamespace(use_mla=False),
        cache_config=SimpleNamespace(block_size=BLOCK_SIZE),
        scheduler_config=SimpleNamespace(disable_hybrid_kv_cache_manager=True),
        parallel_config=SimpleNamespace(
            data_parallel_rank=0,
            tensor_parallel_size=1,
        ),
        kv_transfer_config=SimpleNamespace(
            engine_id="supp-test-engine",
            kv_role=kv_role,
            kv_connector_extra_config={},
        ),
    )


def _make_scheduler(kv_role="kv_consumer", has_mamba=False):
    """Build a real FlagCXConnectorScheduler for the given role."""
    kv_cache_config = _kv_cache_config()
    scheduler = FlagCXConnectorScheduler(
        _vllm_config(kv_role), "supp-test-engine", kv_cache_config
    )
    scheduler._has_mamba = has_mamba
    return scheduler


def _make_worker(kv_role, device):
    """Build a worker around a REAL FlagCX engine, without engine globals.

    ``FlagCXConnectorWorker.__init__`` reads vLLM engine-global state
    (tensor-parallel rank, attention backend registry, KV cache layout)
    that does not exist in a bare pytest process, so the component is
    created with ``object.__new__`` and given exactly the state its
    ``__init__`` produces. The FlagCX library, engine handle, ZMQ
    contexts and thread pool are all real.
    """
    library_path = os.getenv("FLAGCX_LIB_PATH") or os.path.join(
        _FLAGCX_PATH, "build", "lib", "libflagcx.so"
    )
    flagcx = FLAGCXLibrary(library_path)
    worker = object.__new__(FlagCXConnectorWorker)
    worker.vllm_config = _vllm_config(kv_role)
    worker.kv_cache_config = _kv_cache_config()
    worker.engine_id = "supp-test-engine"
    worker.hostname = "127.0.0.1"
    worker.flagcx = flagcx
    worker.engine = flagcx.flagcxP2pEngineCreate()
    worker.rpc_port = 0
    worker.kv_role = kv_role
    worker.tp_rank = 0
    worker.tp_size = 1
    worker.kv_cache_device = None
    worker.block_size = BLOCK_SIZE
    worker.use_mla = False
    worker._physical_blocks_per_logical_kv_block = 1
    worker._layer_specs = {name: SimpleNamespace() for name in LAYER_NAMES}
    worker._layer_group_indices = {name: 0 for name in LAYER_NAMES}
    worker.kv_topo = _FakeTopo()
    worker.block_len_per_layer = []
    worker.kv_block_len_per_layer = []
    worker.registered_layer_names = []
    worker.registered_layer_indices = []
    worker.registered_group_indices = []
    worker.kv_caches_base_addr = []
    worker.device_kv_caches = {}
    worker.side_channel_port = 8998
    worker.zmq_ctx = zmq.Context()
    worker.async_zmq_ctx = zmq.asyncio.Context()
    worker._sender_executor = ThreadPoolExecutor(
        max_workers=1, thread_name_prefix="supp-flagcx-sender"
    )
    worker._sender_t = None
    return worker


def _gpu_kv_cache(device, num_blocks=4):
    """One HND-layout 4D paged KV cache tensor on the device."""
    return torch.zeros(
        num_blocks,
        NUM_KV_HEADS,
        BLOCK_SIZE,
        2 * HEAD_SIZE,
        dtype=torch.bfloat16,
        device=device,
    )


# ---------------------------------------------------------------------------
# 1. Layout contract and side-channel port derivation
# ---------------------------------------------------------------------------


@pytest.mark.supplementary
@pytest.mark.gpu
def test_required_kvcache_layout_contract():
    """Full attention forces HND; MLA and model-less configs do not.

    ``get_required_kvcache_layout`` is a classmethod contract every KV
    connector must honor: FlagCX forces the HND layout for non-MLA models so
    heterogeneous-TP transfers stay uniform.
    """
    assert FlagCXConnector.get_required_kvcache_layout(_vllm_config("kv_consumer")) == (
        "HND"
    )

    mla_config = _vllm_config("kv_consumer")
    mla_config.model_config.use_mla = True
    assert FlagCXConnector.get_required_kvcache_layout(mla_config) is None

    assert (
        FlagCXConnector.get_required_kvcache_layout(SimpleNamespace(model_config=None))
        is None
    )


@pytest.mark.supplementary
@pytest.mark.gpu
def test_side_channel_port_scales_with_parallel_ranks():
    """The ZMQ side-channel port offsets by DP rank * TP size.

    Every (DP rank, TP rank) pair binds its own sender ROUTER socket, so
    the port must be unique per pair.
    """
    config = _vllm_config("kv_producer")
    assert _get_side_channel_port(config) == 8998

    config.parallel_config.data_parallel_rank = 1
    config.parallel_config.tensor_parallel_size = 2
    assert _get_side_channel_port(config) == 8998 + 2


# ---------------------------------------------------------------------------
# 2. Scheduler-side accounting for one remote-prefill request
# ---------------------------------------------------------------------------


@pytest.mark.supplementary
@pytest.mark.gpu
def test_scheduler_tracks_remote_prefill_request_lifecycle():
    """A do_remote_prefill request flows through the scheduler accounting.

    ``get_num_new_matched_tokens`` reports the remote prefill length (all
    prompt tokens, none computed yet); ``update_state_after_alloc`` moves
    it into ``_reqs_need_recv`` keyed by request id with the allocator's
    unhashed block ids and consumes the flag so a reschedule does not
    double-count; ``build_connector_meta`` drains the dict into a metadata
    object the worker consumes.
    """
    scheduler = _make_scheduler("kv_consumer")
    request = _FakeRequest(
        request_id="req-1",
        params={
            "do_remote_prefill": True,
            "remote_host": "10.0.0.2",
            "remote_port": 9000,
            "transfer_id": "t-1",
        },
        num_prompt_tokens=48,
    )

    matched, remote = scheduler.get_num_new_matched_tokens(request, 0)
    assert (matched, remote) == (48, True)

    blocks = _FakeBlocks([[7, 8, 9]])
    scheduler.update_state_after_alloc(request, blocks, num_external_tokens=48)
    assert "req-1" in scheduler._reqs_need_recv
    assert request.kv_transfer_params["do_remote_prefill"] is False

    metadata = scheduler.build_connector_meta(SimpleNamespace())
    assert isinstance(metadata, FlagCXConnectorMetadata)
    assert "req-1" in metadata.reqs_to_recv
    recv = metadata.reqs_to_recv["req-1"]
    assert recv.local_block_ids == [[7, 8, 9]]
    assert recv.remote_host == "10.0.0.2"
    assert recv.remote_port == 9000
    assert recv.transfer_id == "t-1"
    # The pending dict is drained after the metadata is built.
    assert scheduler._reqs_need_recv == {}


@pytest.mark.supplementary
@pytest.mark.gpu
def test_scheduler_consumer_role_fills_send_side_metadata():
    """The producer role emits send entries; the consumer does not.

    ``build_connector_meta`` gates each half by kv_role: a kv_consumer
    never emits reqs_to_send, a kv_producer never emits reqs_to_recv.
    """
    producer = _make_scheduler("kv_producer")
    producer._reqs_need_send["req-p"] = (
        _FakeRequest("req-p", {"transfer_id": "t-p"}, 8),
        [[1, 2]],
    )
    metadata = producer.build_connector_meta(SimpleNamespace())
    assert metadata.reqs_to_recv == {}
    assert "req-p" in metadata.reqs_to_send
    assert metadata.reqs_to_send["req-p"] == ("t-p", [[1, 2]])
    assert producer._reqs_need_send == {}

    consumer = _make_scheduler("kv_consumer")
    consumer._reqs_need_send["req-p"] = (
        _FakeRequest("req-p", {"transfer_id": "t-p"}, 8),
        [[1, 2]],
    )
    metadata = consumer.build_connector_meta(SimpleNamespace())
    assert metadata.reqs_to_send == {}


@pytest.mark.supplementary
@pytest.mark.gpu
def test_scheduler_mamba_prefill_asks_for_n_minus_1_tokens():
    """Mamba/GDN decode requests ask the prefill side for N-1 tokens.

    The decoder recomputes the last prompt token locally to obtain the
    first logits, so the remote prefill only needs h(N-2); non-mamba
    models ask for all N tokens.
    """
    mamba = _make_scheduler("kv_consumer", has_mamba=True)
    assert mamba._get_remote_prefill_token_count(50) == 49
    assert mamba._get_remote_prefill_token_count(1) == 1

    plain = _make_scheduler("kv_consumer", has_mamba=False)
    assert plain._get_remote_prefill_token_count(50) == 50


@pytest.mark.supplementary
@pytest.mark.gpu
def test_scheduler_truncates_mamba_request_for_prefill_once():
    """The P-side mamba truncation drops the last token exactly once.

    ``_truncate_mamba_request_for_prefill`` pops the last prompt token,
    records the truncation in kv_transfer_params, and is a no-op on a
    preempted+rescheduled request (``_p_side_truncated`` guard).
    """
    scheduler = _make_scheduler("kv_producer", has_mamba=True)
    request = _FakeRequest(
        request_id="req-m",
        params={"do_remote_decode": True, "transfer_id": "t-m"},
        num_prompt_tokens=4,
    )

    scheduler._truncate_mamba_request_for_prefill(request)

    assert request.num_prompt_tokens == 3
    assert len(request.prompt_token_ids) == 3
    assert len(request._all_token_ids) == 3
    assert request.max_tokens == 1
    assert request.kv_transfer_params["_p_side_truncated"] is True

    before = request.num_prompt_tokens
    scheduler._truncate_mamba_request_for_prefill(request)
    assert request.num_prompt_tokens == before


# ---------------------------------------------------------------------------
# 3. Worker registration and shutdown against real FlagCX engine
# ---------------------------------------------------------------------------


@pytest.mark.supplementary
@pytest.mark.gpu
def test_worker_consumer_registers_gpu_cache_and_shuts_down(device):
    """Consumer registration runs the real engine calls; shutdown is safe.

    ``register_kv_caches`` performs ``flagcxP2pRegister`` on the GPU cache
    storage, reads the RPC port, and (consumer role) starts the RPC accept
    daemon. ``shutdown`` then destroys the engine, terminates both ZMQ
    contexts and is idempotent.
    """
    worker = _make_worker("kv_consumer", device)
    try:
        kv_caches = {name: _gpu_kv_cache(device) for name in LAYER_NAMES}
        worker.register_kv_caches(kv_caches)

        assert len(worker.registered_layer_names) == NUM_LAYERS
        assert len(worker.kv_caches_base_addr) == NUM_LAYERS
        assert worker.kv_cache_device == device
        assert worker.rpc_port != 0
        # HND (token-first) layout: one region per layer, no K/V split.
        assert worker.kv_block_len_per_layer == worker.block_len_per_layer
    finally:
        worker.shutdown()

    assert worker.engine is None
    worker.shutdown()  # idempotent
    # The terminated ZMQ context rejects further socket creation.
    with pytest.raises(zmq.ZMQError):
        worker.zmq_ctx.socket(zmq.PULL)


@pytest.mark.supplementary
@pytest.mark.gpu
def test_worker_producer_starts_sender_thread_and_joins_it(device):
    """Producer registration starts the ZMQ sender thread; shutdown joins it.

    The producer role skips the RPC server but binds a ROUTER socket on the
    side-channel port and runs the poller loop until the ZMQ context is
    terminated by shutdown().
    """
    worker = _make_worker("kv_producer", device)
    try:
        worker.register_kv_caches(
            {"model.layers.0.self_attn.attn": _gpu_kv_cache(device)}
        )

        sender = worker._sender_t
        assert sender is not None
        assert sender.is_alive()
        assert sender.name == "flagcx_sender"
        assert len(worker.registered_layer_names) == 1
    finally:
        worker.shutdown()

    worker._sender_t.join(timeout=2)
    assert not worker._sender_t.is_alive()


@pytest.mark.supplementary
@pytest.mark.gpu
def test_worker_registration_requires_at_least_one_cache(device):
    """Registering no cache tensors is a hard RuntimeError.

    register_kv_caches raises when nothing was registered, so a
    misconfigured model (all layers filtered out) fails loudly instead of
    silently transferring nothing.
    """
    worker = _make_worker("kv_consumer", device)
    try:
        # A well-formed layer name with no KV cache spec is skipped by the
        # layer-spec filter, leaving zero registered regions.
        with pytest.raises(RuntimeError, match="No KV cache tensors"):
            worker.register_kv_caches(
                {"model.layers.99.self_attn.attn": _gpu_kv_cache(device)}
            )
        assert worker.registered_layer_names == []
    finally:
        worker.shutdown()


# ---------------------------------------------------------------------------
# 4. Transfer planning (pure helpers)
# ---------------------------------------------------------------------------


@pytest.mark.supplementary
@pytest.mark.gpu
def test_group_concurrent_contiguous_groups_runs():
    """Only runs contiguous in BOTH src and dst are merged."""
    src_groups, dst_groups = group_concurrent_contiguous(
        [0, 1, 2, 10, 11], [5, 6, 7, 40, 41]
    )
    assert src_groups == [[0, 1, 2], [10, 11]]
    assert dst_groups == [[5, 6, 7], [40, 41]]

    # A src run broken by dst discontinuity stays split.
    src_groups, dst_groups = group_concurrent_contiguous([0, 1], [3, 9])
    assert src_groups == [[0], [1]]
    assert dst_groups == [[3], [9]]

    assert group_concurrent_contiguous([], []) == ([], [])


@pytest.mark.supplementary
@pytest.mark.gpu
def test_tp_ratio_and_sender_transfer_plan_matrix():
    """The heterogeneous-TP planning table is exact for both directions."""
    # Homogeneous TP.
    assert _get_tp_ratio(1, 1) == 1
    assert _compute_sender_transfer_plan(
        0,
        1,
        0,
        1,
        local_kv_block_len=128,
        remote_kv_block_len=128,
        producer_cache_replicated=False,
    ) == (True, 0, 0, 128)

    # Local TP 4 -> remote TP 1: each rank sends its shard into its slot.
    assert _get_tp_ratio(4, 1) == 4
    assert _compute_sender_transfer_plan(
        0,
        4,
        0,
        1,
        local_kv_block_len=32,
        remote_kv_block_len=128,
        producer_cache_replicated=False,
    ) == (True, 0, 0, 32)
    assert _compute_sender_transfer_plan(
        3,
        4,
        0,
        1,
        local_kv_block_len=32,
        remote_kv_block_len=128,
        producer_cache_replicated=False,
    ) == (True, 0, 3 * 32, 32)

    # Replicated producer cache: only rank 0 sends, full block.
    assert _compute_sender_transfer_plan(
        1,
        4,
        0,
        1,
        local_kv_block_len=32,
        remote_kv_block_len=128,
        producer_cache_replicated=True,
    ) == (False, 0, 0, 32)
    assert _compute_sender_transfer_plan(
        0,
        4,
        0,
        1,
        local_kv_block_len=32,
        remote_kv_block_len=128,
        producer_cache_replicated=True,
    ) == (True, 0, 0, 32)

    # Local TP 1 -> remote TP 4: gather from the remote rank's shard.
    assert _get_tp_ratio(1, 4) == -4
    assert _compute_sender_transfer_plan(
        0,
        1,
        3,
        4,
        local_kv_block_len=128,
        remote_kv_block_len=32,
        producer_cache_replicated=False,
    ) == (True, 3 * 32, 0, 32)

    # Non-divisible TP sizes are rejected up front.
    with pytest.raises(AssertionError):
        _get_tp_ratio(3, 2)


@pytest.mark.supplementary
@pytest.mark.gpu
def test_expand_transfer_regions_splits_blocks_first_layout():
    """Blocks-first layouts emit a K and a V region per registered tensor."""
    regions = _expand_transfer_regions(
        base_addrs=[1000],
        block_lens=[256],
        kv_block_lens=[128],
        layer_names=["model.layers.0.self_attn.attn"],
        layer_indices=[0],
        is_kv_layout_blocks_first=True,
    )
    assert [(r.base_addr, r.block_len, r.kv_block_len) for r in regions] == [
        (1000, 256, 128),
        (1000 + 128, 256, 128),
    ]

    # Token-first (HND) layouts keep one region per tensor.
    regions = _expand_transfer_regions(
        base_addrs=[1000],
        block_lens=[256],
        kv_block_lens=[256],
        layer_names=["model.layers.0.self_attn.attn"],
        layer_indices=[0],
        is_kv_layout_blocks_first=False,
    )
    assert len(regions) == 1
    assert regions[0].kv_block_len == 256


@pytest.mark.supplementary
@pytest.mark.gpu
def test_align_and_validate_region_pairing():
    """Region alignment pairs by (layer, occurrence) and validates indices."""
    local = [
        TransferRegion("l0", 0, 1000, 256, 128, group_index=0),
        TransferRegion("l1", 1, 2000, 256, 128, group_index=1),
    ]
    # Remote lists layers in a different order with matching metadata.
    remote = [
        TransferRegion("l1", 1, 5000, 256, 128, group_index=1),
        TransferRegion("l0", 0, 6000, 256, 128, group_index=0),
    ]
    aligned_local, aligned_remote, err = _align_transfer_regions(local, remote)
    assert err is None
    assert [r.layer_name for r in aligned_local] == ["l0", "l1"]
    assert [r.layer_name for r in aligned_remote] == ["l0", "l1"]
    assert (
        _validate_asymmetric_region_lengths(
            aligned_local, aligned_remote, 1, 1, producer_cache_replicated=False
        )
        is None
    )

    # Missing consumer layer is a hard alignment error.
    _, _, err = _align_transfer_regions(local, [remote[0]])
    assert err is not None
    assert "no matching consumer occurrence" in err

    # Region count mismatch fails validation first.
    err = _validate_asymmetric_region_lengths(
        local, [local[0]], 1, 1, producer_cache_replicated=False
    )
    assert err is not None
    assert "matching KV region counts" in err


@pytest.mark.supplementary
@pytest.mark.gpu
def test_build_transfer_params_builds_absolute_va_triples():
    """_build_transfer_params maps regions to (src, dst, size) VA triples.

    Two layouts, both with one layer and one logical request whose local
    pages [0, 1, 2] map to remote pages [7, 8, 9]:

    * HND (token-first, what the connector itself enforces): one region
      whose contiguous run coalesces into a single 3-block descriptor;
    * blocks-first (split K/V): two regions that can NOT coalesce
      (transfer_len != block_len), so each block becomes its own
      descriptor of kv_block_len bytes.
    """

    class _PlanWorker:
        """The attributes _build_transfer_params reads."""

        kv_cache_config = _kv_cache_config()
        tp_rank = 0
        tp_size = 1
        _physical_blocks_per_logical_kv_block = 1

        def _producer_cache_is_replicated(self):
            return False

        def _logical_to_kernel_block_ids(self, block_ids):
            return block_ids

        def _get_sender_transfer_plan(
            self,
            local_kv_block_len=None,
            remote_kv_block_len=None,
            remote_tp_rank=None,
            remote_tp_size=None,
            **kwargs,
        ):
            del kwargs
            return _compute_sender_transfer_plan(
                local_tp_rank=self.tp_rank,
                local_tp_size=self.tp_size,
                remote_tp_rank=remote_tp_rank,
                remote_tp_size=remote_tp_size,
                local_kv_block_len=local_kv_block_len,
                remote_kv_block_len=remote_kv_block_len,
                producer_cache_replicated=False,
            )

    block_len = 256
    ready = [
        (
            "req-0",
            SendBlockMeta(
                p_req_id="p-0",
                transfer_id="t-0",
                local_block_ids=[[0, 1, 2]],
                ready=threading.Event(),
            ),
        )
    ]
    agent_meta = SimpleNamespace(
        remote_tp_rank=0,
        remote_tp_size=1,
        req_blocks={"req-0": ("t-0", [[7, 8, 9]])},
    )

    # --- HND layout: coalesced single descriptor covering the run ---
    hnd_local = _expand_transfer_regions(
        base_addrs=[100000],
        block_lens=[block_len],
        kv_block_lens=[block_len],
        layer_names=["l0"],
        layer_indices=[0],
        is_kv_layout_blocks_first=False,
        group_indices=[0],
    )
    hnd_remote = _expand_transfer_regions(
        base_addrs=[900000],
        block_lens=[block_len],
        kv_block_lens=[block_len],
        layer_names=["l0"],
        layer_indices=[0],
        is_kv_layout_blocks_first=False,
        group_indices=[0],
    )
    src, dst, sizes = FlagCXConnectorWorker._build_transfer_params(
        _PlanWorker(), ready, agent_meta, hnd_local, hnd_remote
    )
    assert sizes == [block_len * 3]
    assert src == [100000]
    assert dst == [900000 + 7 * block_len]
    assert _can_coalesce_block_transfers(block_len, block_len, 0, 0, block_len)

    # --- blocks-first layout: split K/V regions never coalesce ---
    kv_block_len = block_len // 2
    bf_local = _expand_transfer_regions(
        base_addrs=[100000],
        block_lens=[block_len],
        kv_block_lens=[kv_block_len],
        layer_names=["l0"],
        layer_indices=[0],
        is_kv_layout_blocks_first=True,
        group_indices=[0],
    )
    bf_remote = _expand_transfer_regions(
        base_addrs=[900000],
        block_lens=[block_len],
        kv_block_lens=[kv_block_len],
        layer_names=["l0"],
        layer_indices=[0],
        is_kv_layout_blocks_first=True,
        group_indices=[0],
    )
    src, dst, sizes = FlagCXConnectorWorker._build_transfer_params(
        _PlanWorker(), ready, agent_meta, bf_local, bf_remote
    )
    # 2 regions x 3 per-block descriptors, each kv_block_len bytes.
    assert len(sizes) == 6
    assert all(size == kv_block_len for size in sizes)
    assert src[0] == 100000
    assert src[3] == 100000 + kv_block_len  # V region starts mid-page
    assert dst[0] == 900000 + 7 * block_len
    assert not _can_coalesce_block_transfers(block_len, block_len, 0, 0, kv_block_len)
