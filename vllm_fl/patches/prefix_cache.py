"""Exact text prefix hashes and one-group FullAttention block lookup."""
from functools import lru_cache, wraps
from itertools import islice
import sys

from vllm.logger import init_logger

logger = init_logger("vllm.flagos.prefix_cache")
_CHUNK_BLOCKS = 32
_RETAINED_TOKENS = 2 ** 22


def _hash_factory(original, hash_block):
    @wraps(original)
    def factory(block_size, hash_fn):
        stock = original(block_size, hash_fn)
        chunk_size = block_size * _CHUNK_BLOCKS
        logged_hit = False

        def chain(parent, tokens, salt):
            result = []
            extra = (salt,) if salt else None
            for start in range(0, len(tokens), block_size):
                parent = hash_block(hash_fn, parent, tokens[start:start + block_size], extra)
                result.append(parent)
                extra = None
            return tuple(result)

        memo = lru_cache(maxsize=max(1, _RETAINED_TOKENS // chunk_size))(chain)

        def find(request):
            nonlocal logged_hit
            if (request.block_hashes or request.num_tokens < block_size
                    or request.prompt_embeds is not None or request.mm_features
                    or request.lora_request is not None):
                return stock(request)
            token_ids = request.all_token_ids
            end = request.num_tokens // block_size * block_size
            parent = None
            result = []
            hits_before = memo.cache_info().hits if not logged_hit else 0
            for start in range(0, end, chunk_size):
                tokens = tuple(token_ids[start:min(start + chunk_size, end)])
                salt = request.cache_salt if start == 0 else None
                blocks = memo(parent, tokens, salt) if len(tokens) == chunk_size else chain(parent, tokens, salt)
                result.extend(blocks)
                parent = blocks[-1]
            if not logged_hit and memo.cache_info().hits > hits_before:
                logged_hit = True
                logger.info("FL exact prefix hash memo hit")
            return result

        return find

    factory._fl_exact_prefix = True
    return factory


def _install_lookup():
    from vllm.v1.core.block_pool import BlockHashToBlockMap, BlockPool
    from vllm.v1.core.kv_cache_utils import KVCacheBlock, make_block_hash_with_group_id
    from vllm.v1.core.single_type_kv_cache_manager import FullAttentionManager
    from vllm.v1.kv_cache_interface import FullAttentionSpec

    original = FullAttentionManager.find_longest_cache_hit.__func__
    logged = False

    @wraps(original)
    def find(cls, block_hashes, max_length, kv_cache_group_ids, block_pool, kv_cache_spec,
             drop_eagle_block, alignment_tokens, dcp_world_size=1, pcp_world_size=1):
        nonlocal logged
        if (cls is not FullAttentionManager or type(kv_cache_spec) is not FullAttentionSpec
                or len(kv_cache_group_ids) != 1 or type(block_pool) is not BlockPool
                or dcp_world_size != 1 or pcp_world_size != 1
                or alignment_tokens != kv_cache_spec.block_size
                or type(block_pool.cached_block_hash_to_block) is not BlockHashToBlockMap):
            return original(cls, block_hashes, max_length, kv_cache_group_ids, block_pool,
                            kv_cache_spec, drop_eagle_block, alignment_tokens,
                            dcp_world_size, pcp_world_size)
        live = block_pool.cached_block_hash_to_block
        suffix = make_block_hash_with_group_id(b"", kv_cache_group_ids[0])
        get = live._cache.get
        hits = []
        for value in islice(block_hashes, max_length // kv_cache_spec.block_size):
            block = get(value + suffix)
            if block is None:
                break
            if isinstance(block, KVCacheBlock):
                hits.append(block)
            elif isinstance(block, dict):
                hits.append(next(iter(block.values())))
            else:
                live._unexpected_blocks_type(block)
        if drop_eagle_block and hits:
            hits.pop()
        if hits and not logged:
            logged = True
            logger.info("FL exact single-group prefix lookup hit")
        return (hits,)

    FullAttentionManager.find_longest_cache_hit = classmethod(find)


def install():
    from vllm.v1.core import kv_cache_utils

    original = kv_cache_utils.get_request_block_hasher
    if getattr(original, "_fl_exact_prefix", False):
        return
    _install_lookup()
    wrapped = _hash_factory(original, kv_cache_utils.hash_block_tokens)
    kv_cache_utils.get_request_block_hasher = wrapped
    engine = sys.modules.get("vllm.v1.engine.core")
    if engine is not None and getattr(engine, "get_request_block_hasher", None) is original:
        engine.get_request_block_hasher = wrapped
