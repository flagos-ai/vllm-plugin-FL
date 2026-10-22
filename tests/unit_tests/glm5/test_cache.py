# SPDX-License-Identifier: Apache-2.0
"""Cache contracts and regression coverage."""

import os
import subprocess
import sys
import textwrap
from types import SimpleNamespace

import pytest
import torch

from vllm import SamplingParams
from vllm.utils.hashing import sha256
from vllm.v1.attention.backend import CommonAttentionMetadata
from vllm.v1.core import kv_cache_utils, single_type_kv_cache_manager
from vllm.v1.core.kv_cache_coordinator import (
    HybridKVCacheCoordinator,
    KVCacheCoordinatorNoPrefixCache,
)
from vllm.v1.core.kv_cache_manager import KVCacheManager
from vllm.v1.core.sched.scheduler import Scheduler
from vllm.v1.kv_cache_interface import (
    KVCacheConfig,
    KVCacheGroupSpec,
    MambaSpec,
    MLAAttentionSpec,
    SlidingWindowSpec,
)
from vllm.v1.request import Request

from vllm_fl.dispatch.backends.flaggems.impl.mla import (
    _convert_request_to_physical_indices,
)
from vllm_fl.models.glm5 import cache as kpool_patch, kpool_metadata as portable
from vllm_fl.models.glm5.kpool import (
    Glm5NextIndexerAttentionBackend,
    KpoolTailManager,
    KpoolTailMetadataBuilder,
    KpoolTailSpec,
    compute_kpool_tail_slot_mapping,
)
from vllm_fl.models.glm5.kpool_metadata import (
    append_tail_to_topk,
    expand_pools_to_tokens,
)
from vllm_fl.models.glm5.sparse_indexer import (
    _decode_topk_seq_lens,
    _decode_write_layout,
)
from vllm_fl.runtime.kv_layout import get_physical_cache_layout


@pytest.fixture(scope="module")
def engine_hooks():
    kpool_patch.install_glm5_next_kpool()
    # Use the real upstream callable retained by the owned patch to reproduce
    # the regression, rather than reimplementing the upstream fallback.
    return next(
        patch.pristine
        for patch in kpool_patch._EARLY_PATCHES
        if patch.owner is kv_cache_utils
        and patch.attr == "resolve_kv_cache_block_sizes"
    )


@pytest.fixture(autouse=False)
def deterministic_hash_seed(monkeypatch):
    monkeypatch.setenv("PYTHONHASHSEED", "0")
    monkeypatch.setattr(
        kv_cache_utils,
        "NONE_HASH",
        getattr(kv_cache_utils, "NONE_HASH", None),
        raising=False,
    )
    kv_cache_utils.init_none_hash(sha256)


def _engine_config(
    *, prefix=True, hash_size=None, engine_block=64, dcp=1, pcp=1, connector=False
):
    # These are the already-resolved config values consumed by the public
    # resolver. Spec, coordinator, manager, Request and hashing stay real.
    return SimpleNamespace(
        cache_config=SimpleNamespace(
            block_size=engine_block,
            enable_prefix_caching=prefix,
            hash_block_size=hash_size,
        ),
        parallel_config=SimpleNamespace(
            decode_context_parallel_size=dcp,
            prefill_context_parallel_size=pcp,
        ),
        kv_transfer_config=SimpleNamespace() if connector else None,
    )


def _mla_spec(*, block_size=512, compress_ratio=1):
    return MLAAttentionSpec(
        block_size=block_size,
        num_kv_heads=1,
        head_size=128,
        dtype=torch.bfloat16,
        compress_ratio=compress_ratio,
        indexes_kv_by_block_stride=True,
    )


def _mamba_spec(mode="align"):
    return MambaSpec(
        block_size=512,
        shapes=((2, 128), (128, 128)),
        dtypes=(torch.bfloat16, torch.float32),
        mamba_cache_mode=mode,
    )


def _tail_cache_config(config, *, mamba_mode="align"):
    single_type_kv_cache_manager.register_all_kvcache_specs(config)
    specs = {
        "layers.0.mla": _mla_spec(),
        "layers.0.indexer.k_cache": _mla_spec(compress_ratio=4),
        "layers.0.indexer.tail_cache": KpoolTailSpec(
            block_size=4,
            num_kv_heads=1,
            head_size=256,
            head_size_v=0,
            dtype=torch.bfloat16,
            sliding_window=4,
            indexes_kv_by_block_stride=True,
        ),
        "layers.1.kda": _mamba_spec(mamba_mode),
    }
    groups = kv_cache_utils.get_kv_cache_groups(config, specs)
    # CPU bookkeeping requires groups and a block pool, not GPU cache tensors.
    worker_cache = KVCacheConfig(
        num_blocks=64, kv_cache_tensors=[], kv_cache_groups=groups
    )
    # EngineCore performs this real conversion before resolver/Scheduler:
    # worker UniformType groups become representative scheduler specs while
    # the main MLA and compressed indexer still share their block table.
    return kv_cache_utils.generate_scheduler_kv_cache_config([worker_cache])


def _new_manager(cache, config, *, sizes=None):
    if sizes is None:
        sizes = kv_cache_utils.resolve_kv_cache_block_sizes(cache, config)
    return KVCacheManager(
        kv_cache_config=cache,
        max_model_len=4096,
        scheduler_block_size=sizes[0],
        hash_block_size=sizes[1],
        max_num_batched_tokens=2048,
        enable_caching=config.cache_config.enable_prefix_caching,
    )


def _request(name, prompt, hash_size, *, prefix=True):
    # EngineCore creates a hasher only when prefix caching or a KV connector
    # is enabled. Follow that lifecycle for the disabled-prefix cases too.
    hasher = (
        kv_cache_utils.get_request_block_hasher(hash_size, sha256) if prefix else None
    )
    return Request(
        request_id=name,
        prompt_token_ids=prompt,
        sampling_params=SamplingParams(max_tokens=16),
        pooling_params=None,
        block_hasher=hasher,
    )


@pytest.mark.usefixtures("deterministic_hash_seed")
def test_upstream_mamba_fallback_reproduces_tail_coordinator_failure(engine_hooks):
    config = _engine_config(prefix=True, engine_block=64)
    cache = _tail_cache_config(config)
    sizes = engine_hooks(cache, config)
    # This is the actual serving shape: both large groups use 512 tokens,
    # whereas the ring uses 4. The upstream fallback ignores this ring.
    assert sizes == (512, 512)
    with pytest.raises(AssertionError, match="block_size must be divisible"):
        _new_manager(cache, config, sizes=sizes)


@pytest.mark.usefixtures("deterministic_hash_seed")
def test_resolved_worker_alignment_keeps_short_prefill_in_one_chunk(engine_hooks):
    config = _engine_config(engine_block=64)
    cache = _tail_cache_config(config)
    template = config.cache_config
    template.mamba_block_size = 512
    request = _request("video", list(range(189)), 4)
    scheduler = SimpleNamespace(cache_config=template, block_size=512, use_eagle=False)
    split = Scheduler._mamba_block_aligned_split

    # EngineCore has reset the configured size after worker alignment. The
    # real upstream splitter follows that stale field, despite block_size=512.
    assert split(scheduler, request, 189) == 128
    assert kv_cache_utils.resolve_kv_cache_block_sizes(cache, config) == (512, 4)
    scheduler.cache_config = config.cache_config
    assert split(scheduler, request, 189) == 189
    assert config.cache_config is not template
    assert template.block_size == 64
    assert config.cache_config.block_size == 512
    assert config.cache_config.mamba_block_size == template.mamba_block_size
    assert config.cache_config.hash_block_size is template.hash_block_size


@pytest.mark.usefixtures("deterministic_hash_seed")
def test_cold_and_cached_prefills_keep_identical_aligned_suffix_chunks(engine_hooks):
    config = _engine_config(engine_block=64)
    cache = _tail_cache_config(config)
    scheduler = SimpleNamespace(
        cache_config=config.cache_config, block_size=512, use_eagle=False
    )
    split = Scheduler._mamba_block_aligned_split

    def chunks(*, cached=0, common_prefix=0):
        request = _request("long", list(range(2813)), 4)
        computed = cached
        scheduled = split(
            scheduler,
            request,
            min(request.num_tokens - computed, 2048),
            num_new_local_computed_tokens=cached,
            num_uncached_common_prefix_tokens=common_prefix,
        )
        result = [scheduled]
        computed += scheduled
        while computed < request.num_tokens:
            request.num_computed_tokens = computed
            scheduled = split(scheduler, request, request.num_tokens - computed)
            assert scheduled > 0
            result.append(scheduled)
            computed += scheduled
        return result

    # Real cache admission sees another 512-token full-attention common block,
    # whereas a running request follows the stale 64-token last-cache boundary.
    assert chunks() == [2048, 704, 61]
    assert chunks(cached=2048, common_prefix=512) == [512, 192, 61]
    kv_cache_utils.resolve_kv_cache_block_sizes(cache, config)
    scheduler.cache_config = config.cache_config
    cold = chunks()
    warm = chunks(cached=2048, common_prefix=512)
    assert cold == [2048, 512, 253]
    assert warm == cold[1:]
    assert sum(cold[:-1]) == 2560
    assert sum(cold[:-1]) % _mamba_spec().block_size == 0


@pytest.mark.usefixtures("deterministic_hash_seed")
@pytest.mark.parametrize("prefix", [True, False])
@pytest.mark.parametrize("hash_size", [None, 1, 2, 4])
def test_real_manager_hash_allocate_decode_and_free(engine_hooks, prefix, hash_size):
    config = _engine_config(prefix=prefix, hash_size=hash_size)
    cache = _tail_cache_config(config)
    sizes = kv_cache_utils.resolve_kv_cache_block_sizes(cache, config)
    manager = _new_manager(cache, config, sizes=sizes)
    expected_coordinator = (
        HybridKVCacheCoordinator if prefix else KVCacheCoordinatorNoPrefixCache
    )
    assert isinstance(manager.coordinator, expected_coordinator)
    assert sizes == (512, 4 if hash_size is None else hash_size)
    tail_group, tail_manager = next(
        (index, candidate)
        for index, candidate in enumerate(manager.coordinator.single_type_managers)
        if isinstance(candidate, KpoolTailManager)
    )
    free_before = manager.block_pool.get_num_free_blocks()
    # Equal prompts must still receive distinct tail rings for simultaneous
    # requests; their equal hash chains exercise the strongest reuse case.
    prompt = [100 + index % 257 for index in range(3072)]
    requests = [_request(name, prompt, sizes[1], prefix=prefix) for name in ("a", "b")]
    assert requests[0].block_hashes == requests[1].block_hashes
    for request in requests:
        assert len(request.block_hashes) == (3072 // sizes[1] if prefix else 0)
    rings = {}
    replay = None
    try:
        for chunk in (512, 512, 1024, 1024):
            manager.new_step_starts()
            for request in requests:
                allocated = manager.allocate_slots(request, num_new_tokens=chunk)
                assert allocated is not None
                request.num_computed_tokens += chunk
                blocks = manager.get_blocks(request.request_id).blocks[tail_group]
                assert len(blocks) == 1
                ring = blocks[0]
                assert not ring.is_null and ring.block_hash is None
                assert ring.ref_cnt == 1
                if request.request_id in rings:
                    assert ring is rings[request.request_id]
                    assert not allocated.blocks[tail_group]
                else:
                    rings[request.request_id] = ring
                    assert len(allocated.blocks[tail_group]) == 1
            assert rings["a"] is not rings["b"]
        assert tail_manager.get_num_common_prefix_blocks("a") == 0
        # Real hashing grows through a decode boundary, preserving the prefix
        # chain. The ring remains one unhashed block after this growth.
        manager.new_step_starts()
        for request in requests:
            previous_hashes = request.block_hashes.copy()
            request.append_output_token_ids([700, 701, 702, 703])
            assert request.block_hashes[: len(previous_hashes)] == previous_hashes
            assert len(request.block_hashes) == (3076 // sizes[1] if prefix else 0)
            assert manager.allocate_slots(request, num_new_tokens=4) is not None
            request.num_computed_tokens += 4
            ring = manager.get_blocks(request.request_id).blocks[tail_group]
            assert len(ring) == 1 and ring[0] is rings[request.request_id]
            assert ring[0].block_hash is None
        # Replay through the actual coordinator must not return a tail page.
        manager.new_step_starts()
        replay = _request("replay", prompt, sizes[1], prefix=prefix)
        hit_blocks, hit_tokens = manager.get_computed_blocks(replay)
        assert not hit_blocks.blocks[tail_group]
        if prefix:
            assert hit_tokens > 0 and hit_tokens % sizes[0] == 0
            assert hit_blocks.blocks[0]
        else:
            assert hit_tokens == 0
        allocated = manager.allocate_slots(
            replay,
            num_new_tokens=min(512, replay.num_tokens - hit_tokens),
            num_new_computed_tokens=hit_tokens,
            new_computed_blocks=hit_blocks,
        )
        assert allocated is not None
        assert len(allocated.blocks[tail_group]) == 1
        replay_ring = manager.get_blocks(replay.request_id).blocks[tail_group]
        assert len(replay_ring) == 1
        assert all(replay_ring[0] is not ring for ring in rings.values())
        assert replay_ring[0].block_hash is None
    finally:
        if replay is not None:
            manager.free(replay)
        for request in requests:
            manager.free(request)
    assert manager.block_pool.get_num_free_blocks() == free_before
    assert all(ring.ref_cnt == 0 for ring in rings.values())
    assert all(ring.block_hash is None for ring in rings.values())
    # Freeing releases memory, not a reusable prefix entry for raw tail data.
    later = _request("later", prompt, sizes[1], prefix=prefix)
    later_hits, _ = manager.get_computed_blocks(later)
    assert not later_hits.blocks[tail_group]
    manager.new_step_starts()
    allocated = manager.allocate_slots(later, num_new_tokens=512)
    assert allocated is not None and len(allocated.blocks[tail_group]) == 1
    later_ring = manager.get_blocks(later.request_id).blocks[tail_group]
    assert len(later_ring) == 1 and later_ring[0].block_hash is None
    manager.free(later)
    assert manager.block_pool.get_num_free_blocks() == free_before


@pytest.mark.usefixtures("deterministic_hash_seed")
@pytest.mark.parametrize("mode", ["none", "all"])
@pytest.mark.parametrize("prefix,connector", [(True, False), (False, True)])
def test_kda_mode_without_aligned_snapshots_rejects_prefix_reuse(
    engine_hooks, mode, prefix, connector
):
    config = _engine_config(prefix=prefix, connector=connector)
    cache = _tail_cache_config(config, mamba_mode=mode)
    with pytest.raises(ValueError, match="requires aligned KDA state pages"):
        _new_manager(cache, config)


@pytest.mark.usefixtures("deterministic_hash_seed")
@pytest.mark.parametrize("prefix", [True, False])
def test_invalid_hash_override_rejected_before_manager_construction(
    engine_hooks, prefix
):
    config = _engine_config(prefix=prefix, hash_size=8)
    cache = _tail_cache_config(config)
    with pytest.raises(ValueError, match="Invalid hash_block_size=8"):
        _new_manager(cache, config)


@pytest.mark.usefixtures("deterministic_hash_seed")
@pytest.mark.parametrize("prefix", [True, False])
@pytest.mark.parametrize("dcp,pcp", [(2, 1), (1, 2), (2, 2)])
def test_tail_groups_reject_context_parallelism(engine_hooks, prefix, dcp, pcp):
    config = _engine_config(prefix=prefix, dcp=dcp, pcp=pcp)
    cache = _tail_cache_config(config)
    with pytest.raises(ValueError, match="context parallelism"):
        _new_manager(cache, config)


@pytest.mark.usefixtures("deterministic_hash_seed")
@pytest.mark.parametrize(
    "prefix,hash_size,engine_block,dcp,pcp,shape",
    [
        (True, None, 64, 1, 1, "mamba"),
        (False, 2, 64, 1, 1, "mamba"),
        (True, 8, 512, 1, 1, "mamba"),
        (True, 4, 64, 1, 1, "window"),
        (True, 8, 64, 1, 1, "window"),
        (True, None, 64, 2, 1, "window"),
        (True, None, 64, 1, 2, "single"),
    ],
)
def test_other_cache_layouts_preserve_upstream_resolver_contract(
    engine_hooks, prefix, hash_size, engine_block, dcp, pcp, shape
):
    config = _engine_config(
        prefix=prefix,
        hash_size=hash_size,
        engine_block=engine_block,
        dcp=dcp,
        pcp=pcp,
    )
    groups = [KVCacheGroupSpec(["mla"], _mla_spec())]
    if shape == "mamba":
        groups.append(KVCacheGroupSpec(["ssm"], _mamba_spec()))
    elif shape == "window":
        groups.append(
            KVCacheGroupSpec(
                ["window"],
                SlidingWindowSpec(
                    block_size=4,
                    num_kv_heads=1,
                    head_size=128,
                    dtype=torch.bfloat16,
                    sliding_window=4,
                ),
            )
        )
    cache = KVCacheConfig(num_blocks=64, kv_cache_tensors=[], kv_cache_groups=groups)
    template = config.cache_config
    try:
        expected = engine_hooks(cache, config)
    except ValueError as error:
        with pytest.raises(ValueError) as patched_error:
            kv_cache_utils.resolve_kv_cache_block_sizes(cache, config)
        assert str(patched_error.value) == str(error)
    else:
        assert kv_cache_utils.resolve_kv_cache_block_sizes(cache, config) == expected
    assert config.cache_config is template
    assert config.cache_config.block_size == engine_block


@pytest.mark.usefixtures("deterministic_hash_seed")
def test_imported_engine_alias_is_owned_and_partial_install_rolls_back():
    """Real modules, fresh process, one failure at the setattr install boundary."""
    code = textwrap.dedent(
        """
        import builtins
        import vllm.v1.engine.core as engine_core
        from vllm.v1.core import kv_cache_utils
        from vllm_fl import activation
        from vllm_fl.models.glm5 import cache as kpool

        assert kpool._EARLY_PATCHES is None, "bootstrap already ran in fresh process"
        pristine_resolver = kv_cache_utils.resolve_kv_cache_block_sizes
        pristine_alias = engine_core.resolve_kv_cache_block_sizes
        inventory_before = activation.patch_inventory()

        def fail_alias_install(owner, attr, value):
            if owner is engine_core and attr == "resolve_kv_cache_block_sizes":
                raise RuntimeError("injected engine alias install failure")
            builtins.setattr(owner, attr, value)

        activation.setattr = fail_alias_install
        try:
            try:
                kpool.install_glm5_next_kpool()
            except RuntimeError as error:
                assert str(error) == "injected engine alias install failure"
            else:
                raise AssertionError("engine alias was not staged in the transaction")
        finally:
            del activation.setattr
        assert kv_cache_utils.resolve_kv_cache_block_sizes is pristine_resolver
        assert engine_core.resolve_kv_cache_block_sizes is pristine_alias
        assert activation.patch_inventory() == inventory_before
        assert kpool._EARLY_PATCHES is None

        kpool.install_glm5_next_kpool()
        assert engine_core.resolve_kv_cache_block_sizes is kv_cache_utils.resolve_kv_cache_block_sizes
        targets = {
            record["target"]
            for record in activation.patch_inventory()
            if record["fingerprint"] == "glm5.kpool" and record["phase"] == "engine/config"
        }
        assert any(target.endswith("kv_cache_utils.resolve_kv_cache_block_sizes") for target in targets)
        assert any(target.endswith("engine.core.resolve_kv_cache_block_sizes") for target in targets)
        # Re-registration remains idempotent after a failed first transaction.
        resolver = kv_cache_utils.resolve_kv_cache_block_sizes
        kpool.install_glm5_next_kpool()
        assert kv_cache_utils.resolve_kv_cache_block_sizes is resolver
        assert engine_core.resolve_kv_cache_block_sizes is resolver
        """
    )
    env = dict(os.environ)
    # EngineCore needs the platform plugin on devices whose vLLM wheel lacks
    # NVIDIA extensions. Model activation still happens explicitly below.
    env["VLLM_PLUGINS"] = "fl"
    result = subprocess.run(
        [sys.executable, "-c", code],
        env=env,
        capture_output=True,
        text=True,
        timeout=180,
        check=False,
    )
    assert result.returncode == 0, result.stdout[-4000:] + result.stderr[-4000:]


KPOOL = 4


def _make_tail_block_table(own_blocks: list[int], width: int = 64) -> torch.Tensor:
    block_table = torch.zeros(len(own_blocks), width, dtype=torch.int32)
    block_table[:, 0] = torch.tensor(own_blocks, dtype=torch.int32)
    return block_table


def _legacy_tail_slots(
    block_table: torch.Tensor,
    query_start_loc: torch.Tensor,
    positions: torch.Tensor,
) -> torch.Tensor:
    slots = []
    for request in range(block_table.shape[0]):
        for token in range(query_start_loc[request], query_start_loc[request + 1]):
            position = int(positions[token])
            block = int(block_table[request, position // KPOOL])
            slots.append(block * KPOOL + position % KPOOL)
    return torch.tensor(slots, dtype=torch.int64)


def _make_batch(per_request_positions, padded_len=None):
    positions = torch.cat(
        [torch.tensor(values, dtype=torch.int64) for values in per_request_positions]
    )
    num_actual_tokens = positions.numel()
    num_reqs = len(per_request_positions)
    lengths = [len(values) for values in per_request_positions]
    query_start_loc = torch.zeros(num_reqs + 1, dtype=torch.int64)
    torch.cumsum(torch.tensor(lengths, dtype=torch.int64), 0, out=query_start_loc[1:])
    if padded_len is None:
        padded_len = num_actual_tokens
    slot_mapping = torch.full((padded_len,), -1, dtype=torch.int64)
    return (
        positions,
        query_start_loc,
        slot_mapping,
        num_actual_tokens,
        num_reqs,
    )


def _circular_tail_slots(
    slot_mapping,
    block_table,
    query_start_loc,
    positions,
    num_actual_tokens,
    num_reqs,
):
    return compute_kpool_tail_slot_mapping(
        slot_mapping,
        block_table,
        query_start_loc,
        positions,
        num_actual_tokens,
        num_reqs,
        KPOOL,
    )


def test_legacy_mapping_collapses_long_requests_onto_block_zero() -> None:
    own_blocks = [5, 9]
    per_request = [list(range(10)), list(range(12))]
    positions, query_start_loc, _, _, _ = _make_batch(per_request)
    legacy = _legacy_tail_slots(
        _make_tail_block_table(own_blocks), query_start_loc, positions
    )

    offset = 0
    request_slots = []
    for request, prompt in enumerate(per_request):
        slots = set()
        for position in range(len(prompt)):
            slot = int(legacy[offset + position])
            if position >= KPOOL:
                assert slot // KPOOL == 0
                assert slot // KPOOL != own_blocks[request]
            slots.add(slot)
        request_slots.append(slots)
        offset += len(prompt)
    assert request_slots[0] & request_slots[1]


def test_circular_mapping_isolates_concurrent_requests() -> None:
    own_blocks = [5, 9]
    per_request = [list(range(10)), list(range(12))]
    args = _make_batch(per_request)
    out = _circular_tail_slots(
        args[2],
        _make_tail_block_table(own_blocks),
        args[1],
        args[0],
        args[3],
        args[4],
    )

    offset = 0
    request_slots = []
    for request, prompt in enumerate(per_request):
        slots = set()
        for position in range(len(prompt)):
            slot = int(out[offset + position])
            assert slot // KPOOL == own_blocks[request]
            assert slot % KPOOL == position % KPOOL
            slots.add(slot)
        request_slots.append(slots)
        offset += len(prompt)
    assert not request_slots[0] & request_slots[1]


@pytest.mark.parametrize("prompt_len", [1, 2, 3, 4])
def test_circular_mapping_matches_generic_for_first_pool(prompt_len: int) -> None:
    per_request = [list(range(prompt_len))]
    positions, query_start_loc, slot_mapping, num_actual_tokens, num_reqs = _make_batch(
        per_request
    )
    block_table = _make_tail_block_table([7])
    legacy = _legacy_tail_slots(block_table, query_start_loc, positions)
    circular = _circular_tail_slots(
        slot_mapping,
        block_table,
        query_start_loc,
        positions,
        num_actual_tokens,
        num_reqs,
    )
    assert torch.equal(circular, legacy)


def test_circular_mapping_preserves_padding_and_empty_batch() -> None:
    per_request = [list(range(10)), list(range(12))]
    args = _make_batch(per_request, padded_len=30)
    block_table = _make_tail_block_table([5, 9])
    out = _circular_tail_slots(args[2], block_table, args[1], args[0], args[3], args[4])
    assert torch.equal(out[args[3] :], torch.full_like(out[args[3] :], -1))

    empty = _circular_tail_slots(args[2], block_table, args[1], args[0][:0], 0, args[4])
    assert torch.equal(empty, args[2])


def _make_common_metadata(
    per_request_positions, own_blocks, with_positions: bool = True
):
    positions, query_start_loc, slot_mapping, num_actual_tokens, num_reqs = _make_batch(
        per_request_positions,
        padded_len=sum(len(values) for values in per_request_positions) + 4,
    )
    seq_lens = torch.tensor(
        [max(values) + 1 if values else 1 for values in per_request_positions],
        dtype=torch.int64,
    )
    return CommonAttentionMetadata(
        query_start_loc=query_start_loc,
        query_start_loc_cpu=query_start_loc.clone(),
        seq_lens=seq_lens,
        num_reqs=num_reqs,
        num_actual_tokens=num_actual_tokens,
        max_query_len=max(map(len, per_request_positions), default=1),
        max_seq_len=int(seq_lens.max()) if num_reqs else 1,
        block_table_tensor=_make_tail_block_table(own_blocks),
        slot_mapping=slot_mapping,
        positions=positions if with_positions else None,
    )


def _make_tail_builder():
    builder = object.__new__(KpoolTailMetadataBuilder)
    builder.kv_cache_spec = SimpleNamespace(block_size=KPOOL)
    return builder


def test_builder_uses_circular_mapping() -> None:
    per_request = [list(range(10)), list(range(12))]
    own_blocks = [5, 9]
    metadata = _make_common_metadata(per_request, own_blocks)
    built = KpoolTailMetadataBuilder.build(_make_tail_builder(), 0, metadata)

    offset = 0
    for request, prompt in enumerate(per_request):
        for position in range(len(prompt)):
            slot = int(built.slot_mapping[offset + position])
            assert slot // KPOOL == own_blocks[request]
            assert slot % KPOOL == position % KPOOL
        offset += len(prompt)


def test_builder_keeps_generic_mapping_without_positions() -> None:
    metadata = _make_common_metadata([list(range(10))], [5], with_positions=False)
    built = KpoolTailMetadataBuilder.build(_make_tail_builder(), 0, metadata)
    assert built.slot_mapping is metadata.slot_mapping


def test_tail_backend_preserves_stride_capability_without_changing_deepseek():
    from vllm.v1.attention.backends.mla.indexer import DeepseekV32IndexerBackend

    from vllm_fl.models.glm5.kpool import Glm5NextTailCache

    # The runner replaces the spec's flag with the backend's declaration.
    # Tail pages share padded index-cache allocations, so that declaration
    # must survive removing the old global Deepseek backend patch.
    cache = object.__new__(Glm5NextTailCache)
    torch.nn.Module.__init__(cache)
    cache.index_kpool = 4
    cache.head_dim = 128
    spec = cache.get_kv_cache_spec(None)
    assert spec.indexes_kv_by_block_stride
    assert cache.get_attn_backend().indexes_kv_by_block_stride()
    assert not DeepseekV32IndexerBackend.indexes_kv_by_block_stride()


@pytest.mark.parametrize("preinitialized", [False, True])
def test_real_kv_registry_handles_lazy_and_late_plugin_registration(preinitialized):
    code = f"""
import torch
from vllm.v1 import kv_cache_spec_registry as registry
from vllm.v1.core import single_type_kv_cache_manager as managers
from vllm.v1.kv_cache_interface import FullAttentionSpec
from vllm_fl.models.glm5.kpool import KpoolTailSpec, KpoolTailManager
from vllm_fl.models.glm5.cache import install_glm5_next_kpool

registry._REGISTRY_KVCACHESPEC_LIST.clear()
if {preinitialized!r}:
    managers.register_all_kvcache_specs(None)
install_glm5_next_kpool()
tail = KpoolTailSpec(block_size=4, num_kv_heads=1, head_size=128,
                    dtype=torch.float32, sliding_window=4)
ordinary = FullAttentionSpec(block_size=16, num_kv_heads=1, head_size=128,
                             dtype=torch.float32)
registry.KVCacheSpecRegistry.check_kv_cache_spec_registry(
    {{"tail": tail, "ordinary": ordinary}})
assert registry.KVCacheSpecRegistry.get_manager_class(tail) is KpoolTailManager
assert registry.KVCacheSpecRegistry.get_uniform_type_base_spec(tail) is KpoolTailSpec
assert registry.KVCacheSpecRegistry.get_manager_class(ordinary) is managers.FullAttentionManager
assert registry.KVCacheSpecRegistry.get_uniform_type_base_spec(ordinary) is FullAttentionSpec
# Re-registration keeps the same native and custom manager contracts.
install_glm5_next_kpool()
managers.register_all_kvcache_specs(None)
assert registry.KVCacheSpecRegistry.get_manager_class(tail) is KpoolTailManager
"""
    result = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        timeout=120,
        env=os.environ.copy(),
    )
    assert result.returncode == 0, result.stdout + result.stderr


class OtherCompressedBackend:
    @staticmethod
    def get_kv_cache_shape(blocks, page, heads, width, **kwargs):
        return blocks, page, heads, width


def run_reshape(backend, spec, kernel_size):
    from vllm_fl.worker.model_runner import ModelRunnerFL

    group = SimpleNamespace(
        kv_cache_spec=spec,
        backend=backend,
        kv_cache_group_id=0,
        layer_names=["indexer"],
    )
    runner = SimpleNamespace(
        kv_cache_config=SimpleNamespace(kv_cache_tensors=[], kv_cache_groups=[group]),
        cache_config=SimpleNamespace(cache_dtype="auto"),
        runner_only_attn_layers=set(),
        use_ascend_hybrid_cache=False,
        _kv_cache_spec_attn_group_iterator=lambda: iter([group]),
    )
    raw = torch.zeros(spec.page_size_bytes * 3, dtype=torch.uint8)
    output = ModelRunnerFL._reshape_kv_cache_tensors(
        runner, {"indexer": raw}, [kernel_size]
    )
    return raw, output["indexer"]


@pytest.mark.parametrize("storage", [48, 64, 96])
def test_other_compressed_backend_keeps_own_page_size(storage):
    spec = MLAAttentionSpec(
        block_size=storage * 4,
        compress_ratio=4,
        num_kv_heads=1,
        head_size=132,
        dtype=torch.uint8,
    )
    assert get_physical_cache_layout(OtherCompressedBackend, spec) is None
    raw, cache = run_reshape(OtherCompressedBackend, spec, spec.block_size)
    assert cache.shape == (3, storage, 1, 132)
    assert cache.data_ptr() == raw.data_ptr()


@pytest.mark.parametrize("logical,padding", [(256, 0), (512, 0), (512, 256)])
@pytest.mark.parametrize("dtype,head_size", [(torch.uint8, 132), (torch.bfloat16, 128)])
def test_glm_padded_pages_are_views_with_shared_descriptor(
    logical, padding, dtype, head_size, monkeypatch
):
    from vllm.platforms import current_platform

    monkeypatch.setattr(
        current_platform, "get_device_capability", lambda: SimpleNamespace(major=9)
    )
    itemsize = torch.empty((), dtype=dtype).element_size()
    spec = MLAAttentionSpec(
        block_size=logical,
        compress_ratio=4,
        num_kv_heads=1,
        head_size=head_size,
        dtype=dtype,
        page_size_padded=logical // 4 * head_size * itemsize + padding,
        indexes_kv_by_block_stride=True,
    )
    layout = get_physical_cache_layout(Glm5NextIndexerAttentionBackend, spec)
    raw, cache = run_reshape(Glm5NextIndexerAttentionBackend, spec, 64)
    assert cache.shape[:2] == (3 * layout.pages_per_block, layout.kernel_block_size)
    assert cache.stride(0) * itemsize == spec.page_size_bytes // layout.pages_per_block
    assert cache.data_ptr() == raw.data_ptr()
    cache[-1, -1, ...] = 17
    assert raw.view(dtype).eq(17).any()


def test_metadata_builder_delegates_non_glm_and_uses_same_glm_layout(monkeypatch):
    from vllm.platforms import current_platform
    from vllm.v1.worker.utils import AttentionGroup

    from vllm_fl.models.glm5 import cache as hooks

    monkeypatch.setattr(
        current_platform, "get_device_capability", lambda: SimpleNamespace(major=9)
    )
    delegated = []
    monkeypatch.setitem(
        hooks._RUNTIME_BASELINES,
        hooks._kpool_target(AttentionGroup, "create_metadata_builders"),
        lambda *args: delegated.append(args[0]),
    )
    patched = hooks._create_metadata_builders_patch("test-layout").replacement
    spec = MLAAttentionSpec(
        block_size=512,
        compress_ratio=4,
        num_kv_heads=1,
        head_size=132,
        dtype=torch.uint8,
    )
    group = SimpleNamespace(
        backend=OtherCompressedBackend, kv_cache_spec=spec, layer_names=["indexer"]
    )
    patched(group, None, torch.device("cpu"), 64)
    assert delegated == [group]
    monkeypatch.setattr(
        Glm5NextIndexerAttentionBackend,
        "get_builder_cls",
        lambda: lambda spec, *args: SimpleNamespace(kv_cache_spec=spec),
    )
    group.backend = Glm5NextIndexerAttentionBackend
    patched(group, None, torch.device("cpu"), 64)
    builder = group.metadata_builders[0]
    assert builder.kv_cache_spec.storage_block_size == 64
    assert builder._glm5_physical_layout.pages_per_block == 2


def test_translated_block_table_reuses_storage_across_batch_sizes(monkeypatch):
    from vllm.v1.attention.backends.mla.indexer import DeepseekV32IndexerMetadataBuilder

    from vllm_fl.models.glm5 import cache as hooks

    monkeypatch.setitem(
        hooks._RUNTIME_BASELINES,
        hooks._kpool_target(DeepseekV32IndexerMetadataBuilder, "build"),
        lambda self, prefix, metadata, fast: metadata.block_table_tensor,
    )
    build = hooks._indexer_build_patch("stable-storage").replacement
    builder = SimpleNamespace(
        kv_cache_spec=SimpleNamespace(block_size=512),
        kernel_block_size=64,
        _glm5_physical_layout=object(),
        vllm_config=SimpleNamespace(
            scheduler_config=SimpleNamespace(max_num_batched_tokens=4)
        ),
    )
    outputs = []
    for rows in (1, 3, 2):
        table = torch.arange(rows * 16, dtype=torch.int32).reshape(rows, 16) + rows * 8
        metadata = SimpleNamespace(block_table_tensor=table, replace=SimpleNamespace)
        out = build(builder, 0, metadata)
        torch.testing.assert_close(out, table[:, ::8] // 8)
        outputs.append(out)
    assert len({out.data_ptr() for out in outputs}) == 1
    assert [out.shape for out in outputs] == [(1, 2), (3, 2), (2, 2)]


@pytest.mark.parametrize("padded", [False, True])
def test_decode_positions_and_tail_are_request_local(padded):
    lengths = torch.tensor([1, 0, 1] if padded else [1, 1, 1], dtype=torch.int32)
    positions = torch.tensor([30, 9, 555, 556] if padded else [30, 100, 9, 555])
    count = 2 if padded else 3
    metadata = SimpleNamespace(decode_lens=lengths, requires_padding=padded)
    uniform, group_lengths, width = _decode_write_layout(metadata, 3, count)
    assert uniform is not padded
    assert group_lengths is lengths
    assert width == 1
    seq_lens = _decode_topk_seq_lens(positions, lengths, count, 3, width, padded)
    assert seq_lens.tolist() == ([31, 0, 10] if padded else [31, 101, 10])
    pool_ids = torch.arange(4).expand(3, 4)
    expanded = expand_pools_to_tokens(
        pool_ids, torch.ones_like(pool_ids, dtype=torch.bool), 16, 4
    )
    out = append_tail_to_topk(expanded, seq_lens, seq_lens // 4, 4)
    assert out[:, 16:].tolist() == [
        [28, 29, 30],
        [-1, -1, -1] if padded else [100, -1, -1],
        [8, 9, -1],
    ]


def test_decode_writer_rejects_multi_token_verify():
    metadata = SimpleNamespace(decode_lens=torch.tensor([1, 3]), requires_padding=True)
    with pytest.raises(ValueError, match="one decode token"):
        _decode_write_layout(metadata, 2, 4)


def test_invalid_middle_block_is_compacted_out() -> None:
    request_ids = torch.tensor([0], dtype=torch.int32)
    block_table = torch.tensor([[10, -1, 12]], dtype=torch.int32)
    token_indices = torch.tensor([[0, 64, 128, 1]], dtype=torch.int32)

    physical, valid_lengths = _convert_request_to_physical_indices(
        request_ids, block_table, token_indices, block_size=64
    )

    assert physical.tolist() == [[640, 768, 641, -1]]
    assert valid_lengths.tolist() == [3]


def test_negative_and_out_of_range_tokens_stay_after_valid_prefix() -> None:
    request_ids = torch.tensor([0, 1], dtype=torch.int32)
    block_table = torch.tensor([[2, 3], [5, -1]], dtype=torch.int32)
    token_indices = torch.tensor([[-1, 65, 0, 128], [64, 1, -1, 0]], dtype=torch.int32)

    physical, valid_lengths = _convert_request_to_physical_indices(
        request_ids, block_table, token_indices, block_size=64
    )

    assert physical.tolist() == [[193, 128, -1, -1], [321, 320, -1, -1]]
    assert valid_lengths.tolist() == [2, 2]


def test_pool_expansion_and_tail_append_keep_request_local_indices() -> None:
    pools = torch.tensor([[2, 5], [0, 3]], dtype=torch.int32)
    valid = torch.tensor([[True, False], [True, True]])
    expanded = portable.expand_pools_to_tokens(pools, valid, topk=8, pool_size=4)
    expected = torch.tensor(
        [[8, 9, 10, 11, -1, -1, -1, -1], [0, 1, 2, 3, 12, 13, 14, 15]],
        dtype=torch.int32,
    )
    torch.testing.assert_close(expanded, expected)

    with_tail = portable.append_tail_to_topk(
        expanded,
        seq_lens=torch.tensor([14, 18]),
        pool_lens=torch.tensor([3, 4]),
        pool_size=4,
    )
    torch.testing.assert_close(
        with_tail[:, -3:],
        torch.tensor([[12, 13, -1], [16, 17, -1]], dtype=torch.int32),
    )
