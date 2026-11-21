"""Isolated full mixed backend binding; all actual attention uses one representation."""
import functools
import inspect
import json
import os
import re
from pathlib import Path

import torch
from vllm.v1.attention.backend import AttentionType
from .contract import C500MixedPageSpec, C500MixedStorageBackend, MixedAuxBudget, MixedCacheAllocation
from .ownership import RequestOwnership, reset_layer_slot_state
from .writer import request_bank_write
from .fullcover_decode_writer import request_bank_write as fullcover_decode_write
from .v846_readers import PreparedReader
from .request_phase_readers import RequestPhaseReader
from .quant import quantize_into
from .uniform_decode_metadata import prepare as uniform_decode_prepare
from vllm_fl.dispatch.backends.vendor.metax.impl.attention.flash_attn import MacaFlashAttentionBackend

INSTALLED = False
RUNTIME = None
EVIDENCE = Path(os.environ.get('C500_MIXED_EVIDENCE', '/arch/evidence/mixed_backend_smoke_r708'))


def record(stage, **values):
    if stage in ('execute_return', 'graph_replay', 'mixed_layer_call', 'prepare_real'):
        return
    EVIDENCE.mkdir(parents=True, exist_ok=True)
    with (EVIDENCE / f'runtime_{os.getpid()}.jsonl').open('a') as handle:
        handle.write(json.dumps({'stage': stage, **values})+'\n')


class Budget(MixedAuxBudget):
    def tensor_specs(self):
        specs = super().tensor_specs()
        specs.update({'cu': ((self.max_requests+1,), torch.int32),
                      'ends': ((self.max_requests,), torch.int32),
                      'reset_slots': ((self.max_requests,), torch.int32)})
        specs.update({name: ((self.max_requests,), torch.int32) for name in ['decode_owners', 'decode_bases', 'decode_counts', 'decode_visible']})
        return specs


class Runtime:
    def __init__(self, runner):
        global RUNTIME
        self.runner = runner
        self.budget = Budget(42, runner.scheduler_config.max_num_seqs, runner.max_num_tokens)
        self.owner = RequestOwnership(self.budget.max_requests)
        self.aux = None
        self.allocation = None
        self.real = False
        self.plan = None
        self.step = 0
        self.host_index = 0
        self.host_recorded = [False, False]
        self.host_events = None
        self.host = None
        from .writer import V_CACHE_LAYOUT
        if V_CACHE_LAYOUT != 846:
            raise ValueError('V846 readers require the matching V846 writer')
        self.reader = PreparedReader('decode', single_query=True)
        self.prefill_reader = RequestPhaseReader()
        RUNTIME = self

    @torch.inference_mode(False)
    def ensure_aux(self):
        if self.aux is not None:
            return
        before = torch.cuda.memory_allocated()
        self.aux = {name: torch.empty(shape, dtype=dtype, device=self.runner.device)
                    for name, (shape, dtype) in self.budget.tensor_specs().items()}
        self.prefill_reader.decode_metadata = tuple(self.aux[name] for name in ['decode_owners', 'decode_bases', 'decode_counts', 'decode_visible'])
        for name in ['stages', 'banks', 'request_generations', 'cu', 'ends']:
            self.aux[name].zero_()
        for name in ['request_slots', 'reset_slots', 'zero_ids_gpu']:
            self.aux[name].fill_(-1)
        self.host = [{name: torch.empty(self.budget.max_requests, dtype=dtype, pin_memory=True)
                      for name, dtype in [('slots', torch.int32), ('generation', torch.int64), ('reset', torch.int32)]} for _ in range(2)]
        self.host_events = [torch.cuda.Event() for _ in range(2)]
        record('aux_allocated_inside_profile', requested_bytes=self.budget.total_bytes,
               actual_allocated_delta=torch.cuda.memory_allocated()-before,
               max_requests=self.budget.max_requests, max_tokens=self.budget.max_query_tokens)

    def prepare_real(self, scheduler):
        self.ensure_aux()
        ids = list(self.runner.input_batch.req_ids)
        self.plan = self.owner.prepare(scheduler, self.runner.requests, ids, padded_rows=len(ids))
        index = self.host_index
        if self.host_recorded[index] and not self.host_events[index].query():
            self.host_events[index].synchronize()
        host = self.host[index]
        host['slots'].fill_(-1)
        host['generation'].zero_()
        host['reset'].fill_(-1)
        n = len(ids)
        if n:
            host['slots'][:n] = torch.tensor(self.plan.slots, dtype=torch.int32)
            host['generation'][:n] = torch.tensor(self.plan.generations, dtype=torch.int64)
        resets = len(self.plan.reset_slots)
        if resets:
            host['reset'][:resets] = torch.tensor(self.plan.reset_slots, dtype=torch.int32)
        for target, source in [('request_slots', 'slots'), ('request_generations', 'generation'), ('reset_slots', 'reset')]:
            self.aux[target].copy_(host[source], non_blocking=True)
        self.host_events[index].record()
        self.host_recorded[index] = True
        self.host_index = 1-index
        if resets:
            reset_layer_slot_state(self.aux['stages'], self.aux['banks'], self.aux['reset_slots'][:resets])
        record('prepare_real', step=self.step, ids=ids, slots=list(self.plan.slots),
               generations=list(self.plan.generations), starts=list(self.plan.starts),
               counts=list(self.plan.counts), reset_slots=list(self.plan.reset_slots),
               physical=list(self.plan.pending_commits),
               finished=sorted(scheduler.finished_req_ids))


def mixed_forward(self, layer, query, key, value, kv_cache, attn_metadata, output=None,
            output_scale=None, output_block_scale=None):
    if self.attn_type != AttentionType.DECODER:
        raise ValueError('experimental mixed runtime supports decoder attention only')
    if output is None or output_scale is not None or output_block_scale is not None:
        raise ValueError('mixed backend requires provided BF16 output')
    if attn_metadata is None:
        # The framework profiling contract has no actual attention inputs.
        return output.fill_(0)
    rt = RUNTIME
    if rt is None or rt.allocation is None:
        raise RuntimeError('mixed storage/ownership was not bound before compute')
    if self.num_heads != 16 or self.num_kv_heads != 2 or self.head_size != 128:
        raise ValueError('unexpected model attention dimensions')
    if self.sliding_window != (-1, -1) or self.alibi_slopes is not None or self.sinks is not None or self.logits_soft_cap:
        raise ValueError('unsupported attention semantic contract')
    if abs(self.scale-128**-.5) > 1e-12 or not attn_metadata.causal:
        raise ValueError('scale/causal contract mismatch')
    name = layer.layer_name
    index = rt.layer_index[name]
    views = rt.allocation.views[name]
    if kv_cache.data_ptr() != views.raw.data_ptr():
        raise RuntimeError('framework cache pointer differs from mixed owner')
    count = query.shape[0]
    requests = attn_metadata.seq_lens.shape[0]
    if count > rt.budget.max_query_tokens or requests > rt.budget.max_requests:
        raise ValueError('captured batch exceeds allocated scratch')
    if not query.is_contiguous() or not key.is_contiguous() or not output.is_contiguous():
        raise ValueError('native Q/K/output must satisfy measured dense input ABI')
    cu, ends = rt.aux['cu'][:requests+1], rt.aux['ends'][:requests]
    reader = rt.reader if attn_metadata.max_query_len == 1 else rt.prefill_reader
    tiles = min(count, (count+reader.tile-1)//reader.tile+requests)
    q8, qs = rt.aux['q8'][:count], rt.aux['qs'][:count]
    selected_write = fullcover_decode_write if attn_metadata.max_query_len == 1 else request_bank_write
    selected_write(key, value, rt.aux['stages'][index], rt.aux['banks'][index],
        rt.aux['request_slots'][:requests], views.k, views.scale, views.v,
        cu, ends, attn_metadata.block_table, attn_metadata.slot_mapping, max_query_tokens=(1 if attn_metadata.max_query_len == 1 else count))
    quantize_into(query, q8, qs)
    owners, bases, counts, visible = [rt.aux[x][:tiles] for x in ['owners', 'bases', 'counts', 'visible']]
    if attn_metadata.max_query_len == 1:
        uniform_decode_prepare(cu, ends, owners, bases, counts, visible)
    else:
        reader.prepare(cu, ends, owners, bases, counts, visible)
    reader.into(q8, views.k, views.v, qs, views.scale, cu, owners, bases, counts, visible,
        attn_metadata.block_table, rt.aux['partial'], rt.aux['maxima'], rt.aux['sums'], output,
        max_len=int(attn_metadata.max_seq_len), splits=(rt.budget.splits if attn_metadata.max_query_len == 1 else 1))
    if index in (0, 41):
        record('mixed_layer_call', layer=index, real=rt.real, step=rt.step,
               reader=type(reader).__name__, writer=selected_write.__module__,
               max_query_len=attn_metadata.max_query_len,
               capturing=torch.cuda.is_current_stream_capturing(), query_shape=list(query.shape),
               value_stride=list(value.stride()), requests=requests)
    return output

def install():
    global INSTALLED
    if INSTALLED:
        return
    from vllm.v1.kv_cache_spec_registry import KVCacheSpecRegistry
    from vllm.v1.core.single_type_kv_cache_manager import FullAttentionManager
    from vllm.model_executor.layers.attention import Attention
    from vllm_fl.worker.model_runner import ModelRunnerFL
    from vllm_fl.worker.worker import WorkerFL
    from vllm.v1.worker.utils import bind_kv_cache
    KVCacheSpecRegistry._ensure_registered()
    KVCacheSpecRegistry.register(C500MixedPageSpec, FullAttentionManager, uniform_type_base_spec=C500MixedPageSpec)
    original_spec = Attention.get_kv_cache_spec
    def spec(self, config):
        if self.attn_backend is not MacaFlashAttentionBackend:
            return original_spec(self, config)
        return C500MixedPageSpec(block_size=16, num_kv_heads=self.num_kv_heads, head_size=self.head_size, dtype=torch.int8)
    Attention.get_kv_cache_spec = spec
    original_init = ModelRunnerFL.__init__
    def init(self, *args, **kwargs):
        original_init(self, *args, **kwargs)
        if self.parallel_config.tensor_parallel_size != 1 or self.speculative_config is not None or self.parallel_config.use_ubatching:
            raise ValueError('initial mixed backend qualification requires TP1/no speculation/no microbatch reentry')
        self._c500 = Runtime(self)
    ModelRunnerFL.__init__ = init
    original_profile = ModelRunnerFL.profile_run
    def profile(self, *args, **kwargs):
        self._c500.ensure_aux()
        return original_profile(self, *args, **kwargs)
    ModelRunnerFL.profile_run = profile
    original_determine = WorkerFL.determine_available_memory
    def determine(self):
        if self.model_runner._c500.aux is not None:
            raise RuntimeError('first native profile must include the initial auxiliary allocation')
        available = original_determine(self)
        rt = self.model_runner._c500
        # Automatic native profile already counts allocation inside profile_run.
        # Explicit bytes bypass that profiling arithmetic and need a deduction.
        if self.cache_config.kv_cache_memory_bytes:
            available -= sum((x+511)//512*512 for x in rt.budget.byte_breakdown().values())
        if available <= 0:
            raise ValueError('no cache budget remains')
        rt.available_pages_bytes = available
        record('native_profile_return', cache_available=available, aux_requested=rt.budget.total_bytes,
               aux_in_native_profile=not bool(self.cache_config.kv_cache_memory_bytes))
        return available
    WorkerFL.determine_available_memory = determine
    def initialize_tensors(self, config, kernel_block_sizes):
        rt = self._c500
        rt.ensure_aux()
        raw = self._allocate_kv_cache_tensors(config)
        rt.allocation = MixedCacheAllocation(config, raw, device=self.device, aux_budget=rt.budget, existing_aux=rt.aux)
        rt.layer_index = {name:i for i,name in enumerate(rt.allocation.views)}
        containers = rt.allocation.bound_container_tensors()
        bind_kv_cache(containers, self.compilation_config.static_forward_context, self.kv_caches)
        rt.allocation.make_zero_plan()
        record('cache_bound', layers=len(raw), pages=config.num_blocks, raw_bytes=sum(t.numel() for t in raw.values()),
               aux_requested=rt.budget.total_bytes)
        return containers
    ModelRunnerFL.initialize_kv_cache_tensors = initialize_tensors
    def init_zero(self):
        self._c500.allocation.make_zero_plan()
    def zero(self, block_ids):
        rt = self._c500
        if len(block_ids)>rt.aux['zero_ids_gpu'].numel():
            raise ValueError('zero ID capacity exceeded')
        if block_ids:
            ids = rt.aux['zero_ids_gpu'][:len(block_ids)]
            ids.copy_(torch.tensor(block_ids, dtype=torch.int64), non_blocking=False)
            rt.allocation.zero_plan.zero(ids)
            record('zero_blocks', step=rt.step, ids=list(block_ids))
    ModelRunnerFL._init_kv_zero_meta = init_zero
    ModelRunnerFL._zero_block_ids = zero
    original_cleanup = ModelRunnerFL._cleanup_profiling_kv_cache
    def cleanup(self):
        self._c500.allocation = None
        return original_cleanup(self)
    ModelRunnerFL._cleanup_profiling_kv_cache = cleanup
    original_prepare = ModelRunnerFL._prepare_inputs
    def prepare(self, scheduler_output, *args, **kwargs):
        result = original_prepare(self, scheduler_output, *args, **kwargs)
        self._c500.prepare_real(scheduler_output)
        return result
    ModelRunnerFL._prepare_inputs = prepare
    original_metadata = ModelRunnerFL._build_attention_metadata
    signature = inspect.signature(original_metadata)
    def metadata(self, *args, **kwargs):
        result = original_metadata(self, *args, **kwargs)
        rt = self._c500
        rt.ensure_aux()
        bound = signature.bind(self, *args, **kwargs)
        bound.apply_defaults()
        n = bound.arguments['num_reqs_padded'] or bound.arguments['num_reqs']
        rt.aux['cu'][:n+1].copy_(self.query_start_loc.gpu[:n+1])
        rt.aux['ends'][:n].copy_(self.seq_lens[:n])
        if not rt.real:
            # Dummy rows have no user lease. The native slot_mapping=-1
            # contract skips cache writes and bank publication before lookup.
            rt.aux['request_slots'][:n].fill_(-1)
        return result
    ModelRunnerFL._build_attention_metadata = metadata
    original_execute = ModelRunnerFL.execute_model
    def execute(self, scheduler_output, *args, **kwargs):
        rt = self._c500
        rt.real = True
        rt.step += 1
        try:
            result = original_execute(self, scheduler_output, *args, **kwargs)
            if rt.plan is not None:
                rt.owner.commit(rt.plan)
                rt.plan = None
            elif not scheduler_output.total_num_scheduled_tokens:
                plan = rt.owner.prepare(scheduler_output, self.requests, [], padded_rows=0)
                rt.owner.commit(plan)
            record('execute_return', step=rt.step, live_leases=len(rt.owner.leases))
            return result
        except Exception as error:
            record('execute_failure', step=rt.step, error_type=type(error).__name__, error=str(error))
            raise
        finally:
            rt.real = False
    ModelRunnerFL.execute_model = execute
    INSTALLED = True
    record('installed', backend=MacaFlashAttentionBackend.get_name(), writer='V846_strided_slot_mapping_contract', prefill_reader='request_phase_r1317_M128S1_q1S16', decode_reader='unified_r1090', native_attention_called=False)
