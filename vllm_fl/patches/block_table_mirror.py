"""Track whether a mirror still contains the exact active CPU table."""
from functools import wraps


class Mirror:
    def __init__(self, table):
        self.table = table
        self.snapshot = None
        self.copies = 0
        self.reuses = 0
        for name in ("add_row", "clear_row", "move_row", "swap_row", "clear"):
            self.observe(name)
        self.observe("append_row", empty_append=True)

    def observe(self, name, empty_append=False):
        method = getattr(self.table, name)

        @wraps(method)
        def invalidate_on_write(*args, **kwargs):
            if not empty_append or (args[0] if args else kwargs["block_ids"]):
                self.invalidate()
            return method(*args, **kwargs)

        setattr(self.table, name, invalidate_on_write)

    def invalidate(self):
        self.snapshot = None

    def commit(self, active_rows, stream):
        key = active_rows, stream
        if self.snapshot == key:
            self.reuses += 1
            return
        self.table.commit_block_table(active_rows)
        self.copies += 1
        self.snapshot = key


"""Runner routing, appended to the measured E213 Mirror implementation."""
from functools import lru_cache
import logging
import mmap
import os

import torch


logger = logging.getLogger("vllm.flagos.block_table")
_PHASE = os.environ.get("E209_PHASE")
_ACTIVE_PHASES = (ord("1"), ord("2"), ord("3"), ord("4"))


@lru_cache(maxsize=1)
def phase_map():
    with open(_PHASE, "rb") as source:
        return mmap.mmap(source.fileno(), 1, access=mmap.ACCESS_READ)


def invalidate_runner(runner):
    mirror = getattr(runner, "_e214_mirror", None)
    if mirror is not None:
        mirror.invalidate()


def use_mirror(runner, table):
    config = runner.parallel_config
    return (
        type(runner.get_model()).__name__ == "LlamaForCausalLM"
        and not runner.model_config.is_encoder_decoder
        and not runner.is_pooling_model
        and not runner.enable_prompt_embeds
        and runner.speculative_config is None
        and runner.vllm_config.kv_transfer_config is None
        and config.tensor_parallel_size == config.pipeline_parallel_size == config.data_parallel_size == 1
        and table.pcp_world_size == table.dcp_world_size == 1
        and not table.use_hybrid_blocks
    )


def commit_runner(runner, scheduler, rows):
    multi = runner.input_batch.block_table
    phase = phase_map()[0] if _PHASE else ord("1")
    if (
        phase not in _ACTIVE_PHASES
        or len(runner.kv_cache_config.kv_cache_groups) != 1
        or rows != scheduler.total_num_scheduled_tokens
        or rows != len(scheduler.num_scheduled_tokens)
        or any(count != 1 for count in scheduler.num_scheduled_tokens.values())
    ):
        invalidate_runner(runner)
        return multi.commit_block_table(rows)

    stream = torch.cuda.current_stream(runner.device)
    if stream != torch.cuda.default_stream(runner.device):
        invalidate_runner(runner)
        return multi.commit_block_table(rows)

    table = multi[0]
    mirror = getattr(runner, "_e214_mirror", None)
    if mirror is None or mirror.table is not table:
        if not use_mirror(runner, table):
            invalidate_runner(runner)
            return multi.commit_block_table(rows)
        mirror = runner._e214_mirror = Mirror(table)
        mirror.phase = None

    if mirror.phase != phase:
        mirror.invalidate()
        mirror.phase = phase
        mirror.b64_steps = 0
        mirror.b64_reuses = 0
        mirror.short_verified = False
    before = mirror.reuses
    mirror.commit(rows, stream.cuda_stream)
    if phase == ord("4") and mirror.reuses > before and not mirror.short_verified:
        mirror.short_verified = True
        logger.info("E214 block table short reuse phase=4 rows=%d", rows)
    if rows == 64:
        mirror.b64_steps += 1
        mirror.b64_reuses += mirror.reuses - before
        if mirror.b64_steps == 1 or (_PHASE and mirror.b64_steps % 256 == 0):
            logger.info("E214 block table M64 phase=%s steps=%d reused=%d width=%d",
                        chr(phase), mirror.b64_steps, mirror.b64_reuses, table.max_num_blocks_per_req)
