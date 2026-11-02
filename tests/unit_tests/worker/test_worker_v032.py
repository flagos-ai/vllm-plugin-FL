"""Regression coverage for the vLLM 0.32 worker interface."""
import ast
import inspect


def test_worker_imports_against_target_vllm():
    from vllm_fl.worker.worker import WorkerFL

    assert callable(WorkerFL.execute_model)


def test_model_runner_imports_against_target_vllm():
    from vllm_fl.worker.model_runner import ModelRunnerFL

    assert callable(ModelRunnerFL.execute_model)
    assert callable(ModelRunnerFL.initialize_kv_cache)


def test_gcu_attention_imports_against_target_vllm():
    from vllm_fl.dispatch.backends.vendor.gcu.impl.attention import AttentionGCUBackend

    assert AttentionGCUBackend.get_name() == 'FLASH_ATTN'


def test_gcu_block_size_accepts_v032_spec():
    # customize_attention_spec calls this without a current-config context.
    from vllm_fl.dispatch.backends.vendor.gcu.impl import attention

    backend = attention.AttentionGCUBackend
    assert backend.get_supported_kernel_block_sizes(None) == backend.get_supported_kernel_block_sizes()


def test_pipeline_transport_uses_target_contract():
    from vllm_fl.worker import worker

    tree = ast.parse(inspect.getsource(worker))
    calls = [node for node in ast.walk(tree) if isinstance(node, ast.Call)
             and isinstance(node.func, ast.Attribute)
             and node.func.attr in {'irecv_tensor_dict', 'isend_tensor_dict'}]
    assert len(calls) == 2
    for call in calls:
        assert 'all_gather_tensors' not in {kw.arg for kw in call.keywords}
    assert not any(isinstance(node, ast.Name)
                   and node.id == 'is_residual_scattered_for_sp'
                   for node in ast.walk(tree))


def test_initialize_kv_cache_propagates_target_layout(monkeypatch):
    from contextlib import nullcontext
    from types import SimpleNamespace
    from unittest.mock import Mock
    from vllm_fl.worker import worker

    runner = Mock(spec=['initialize_kv_cache'])
    config = SimpleNamespace(num_blocks=7, kv_cache_layout=None,
                             hash_block_size=32, cache_hit_alignment_tokens=16,
                             needs_kv_cache_zeroing=False)
    cache = SimpleNamespace()
    fake = SimpleNamespace(cache_config=cache, vllm_config=object(),
                           model_runner=runner,
                           _maybe_get_memory_pool_context=lambda **kw: nullcontext())
    monkeypatch.setattr(worker, 'ensure_kv_transfer_initialized', Mock())
    monkeypatch.setattr(worker, 'has_kv_transfer_group', lambda: False)
    worker.WorkerFL.initialize_from_config(fake, config)
    assert cache.num_gpu_blocks == 7
    assert cache.hash_block_size == 32
    assert cache.cache_hit_alignment_tokens == 16
    args, kwargs = runner.initialize_kv_cache.call_args
    assert args == (config,)
    assert 'kv_cache_allocation_context' in kwargs


def test_gcu_does_not_query_cuda_allocator_settings(monkeypatch):
    from types import SimpleNamespace
    from vllm_fl.worker import worker

    monkeypatch.setattr(worker, 'current_platform', SimpleNamespace(device_type='gcu'))
    def forbidden():
        raise AssertionError('CUDA allocator queried on GCU')
    monkeypatch.setattr(worker.torch.cuda.memory, 'get_allocator_backend', forbidden)
    with worker.WorkerFL._scoped_allocator_max_split(object(), 20):
        pass


def test_gcu_forward_uses_standardized_kv_views(monkeypatch):
    from types import SimpleNamespace
    from unittest.mock import Mock
    import torch
    from vllm_fl.dispatch.backends.vendor.gcu.impl import attention

    impl = attention.AttentionGCUImpl.__new__(attention.AttentionGCUImpl)
    impl.attn_type = attention.AttentionType.DECODER
    impl.head_size = 8
    impl.num_kv_heads = 2
    impl.kv_sharing_target_layer_name = None
    impl.kv_cache_dtype = 'auto'
    impl.dcp_world_size = 1
    impl.scale = 1.0
    impl.alibi_slopes = None
    impl.sliding_window = (-1, -1)
    impl.logits_soft_cap = 0
    impl.vllm_flash_attn_version = 2
    impl.sinks = None
    meta = SimpleNamespace(num_actual_tokens=1, slot_mapping=torch.tensor([0]),
                           use_cascade=False, query_start_loc=torch.tensor([0, 1]),
                           seq_lens=torch.tensor([1]), max_query_len=1, max_seq_len=1,
                           block_table=torch.tensor([[0]]), scheduler_metadata=None,
                           causal=True, max_num_splits=0)
    cache = torch.arange(3 * 2 * 16 * 16).reshape(3, 2, 16, 16).float()
    write = Mock()
    flash = Mock()
    monkeypatch.setattr(attention, 'reshape_and_cache_flash', write)
    monkeypatch.setattr(attention, 'flash_attn_varlen_func', flash)
    scale = torch.ones(1)
    layer = SimpleNamespace(_k_scale=scale, _v_scale=scale, _q_scale=scale)
    q = torch.zeros(1, 2, 8)
    out = torch.empty_like(q)
    assert impl.forward(layer, q, q, q, cache, meta, out) is out
    k, v = write.call_args.args[2:4]
    assert k.shape == v.shape == (3, 16, 2, 8)
    torch.testing.assert_close(k, cache[..., :8].transpose(1, 2))
    torch.testing.assert_close(v, cache[..., 8:].transpose(1, 2))
    assert k.untyped_storage().data_ptr() == cache.untyped_storage().data_ptr()
    assert flash.call_args.kwargs['k'] is k
    assert flash.call_args.kwargs['v'] is v


def test_compiled_gcu_dispatch_warmup():
    """Execute the actual pre-trace guard without loading model weights."""
    from types import SimpleNamespace
    from unittest.mock import Mock
    from vllm_fl.worker import model_runner
    from textwrap import dedent
    tree = ast.parse(dedent(inspect.getsource(model_runner.ModelRunnerFL.load_model)))
    guards = [node for node in ast.walk(tree) if isinstance(node, ast.If)
              and any(isinstance(call, ast.Call) and isinstance(call.func, ast.Name)
                      and call.func.id == 'warmup_cached_ops' for call in ast.walk(node))]
    guard = min(guards, key=lambda node: len(list(ast.walk(node))))
    warmup = Mock()
    body = [node for node in guard.body if not isinstance(node, (ast.Import, ast.ImportFrom))]
    isolated = ast.fix_missing_locations(ast.Module(body=[ast.If(test=guard.test, body=body, orelse=[])], type_ignores=[]))
    for device, mode, expected in [('gcu', 3, 1), ('npu', 3, 1), ('gcu', 0, 0), ('cuda', 3, 0)]:
        warmup.reset_mock()
        exec(compile(isolated, '<warmup guard>', 'exec'), {
            'current_platform': SimpleNamespace(device_type=device),
            'self': SimpleNamespace(compilation_config=SimpleNamespace(mode=mode)),
            'CompilationMode': SimpleNamespace(NONE=0), 'warmup_cached_ops': warmup})
        assert warmup.call_count == expected
