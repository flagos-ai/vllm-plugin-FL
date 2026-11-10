"""Regression coverage for the vLLM 0.32 worker interface."""

import ast
import importlib.util
import inspect
import sys
from pathlib import Path
from types import ModuleType
from unittest.mock import Mock

import pytest


@pytest.fixture
def gcu_attention_module(monkeypatch):
    name = "flash_attn.vllm_flash_attn"
    try:
        available = importlib.util.find_spec(name) is not None
    except ModuleNotFoundError:
        available = False
    if not available:
        native = ModuleType(name)
        native.flash_attn_varlen_func = Mock(
            side_effect=RuntimeError("Enflame native kernel unavailable on this host")
        )
        monkeypatch.setitem(sys.modules, name, native)
    from vllm_fl.dispatch.backends.vendor.gcu.impl import attention

    yield attention
    if not available:
        sys.modules.pop(attention.__name__, None)
        package = sys.modules[attention.__package__]
        if getattr(package, "attention", None) is attention:
            delattr(package, "attention")


def test_worker_imports_against_target_vllm():
    from vllm_fl.worker.worker import WorkerFL

    assert callable(WorkerFL.execute_model)


def test_model_runner_imports_against_target_vllm():
    from vllm_fl.worker.model_runner import ModelRunnerFL

    assert callable(ModelRunnerFL.execute_model)
    assert callable(ModelRunnerFL.initialize_kv_cache)


def test_stream_signatures_defer_platform_lookup():
    from types import SimpleNamespace

    path = Path(__file__).resolve().parents[3] / "vllm_fl/worker/model_runner.py"
    tree = ast.parse(path.read_text())
    platform = SimpleNamespace(torch_device_fn=None)
    namespace = {"current_platform": platform}
    stream_type = type("Stream", (), {})
    signatures = []
    for node in ast.walk(tree):
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        arguments = [*node.args.posonlyargs, *node.args.args, *node.args.kwonlyargs]
        arguments += [arg for arg in (node.args.vararg, node.args.kwarg) if arg]
        annotations = [arg.annotation for arg in arguments] + [node.returns]
        for annotation in annotations:
            if annotation is None:
                continue
            if "current_platform.torch_device_fn.Stream" not in ast.unparse(annotation):
                continue
            platform.torch_device_fn = None
            deferred = eval(
                compile(ast.Expression(annotation), str(path), "eval"), namespace
            )
            assert isinstance(deferred, str), (node.name, annotation.lineno)
            platform.torch_device_fn = SimpleNamespace(Stream=stream_type)
            assert eval(deferred, namespace) is stream_type
            signatures.append(node.name)
    assert len(signatures) >= 3


def test_gcu_attention_imports_against_target_vllm(gcu_attention_module):
    assert gcu_attention_module.AttentionGCUBackend.get_name() == "FLASH_ATTN"


def test_gcu_block_size_accepts_v032_spec(gcu_attention_module):
    # customize_attention_spec calls this without a current-config context.
    backend = gcu_attention_module.AttentionGCUBackend
    assert (
        backend.get_supported_kernel_block_sizes(None)
        == backend.get_supported_kernel_block_sizes()
    )


def test_pipeline_transport_uses_target_contract():
    from vllm_fl.worker import worker

    tree = ast.parse(inspect.getsource(worker))
    calls = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr in {"irecv_tensor_dict", "isend_tensor_dict"}
    ]
    assert len(calls) == 2
    for call in calls:
        assert "all_gather_tensors" not in {kw.arg for kw in call.keywords}
    assert not any(
        isinstance(node, ast.Name) and node.id == "is_residual_scattered_for_sp"
        for node in ast.walk(tree)
    )


def test_initialize_kv_cache_propagates_target_layout(monkeypatch):
    from contextlib import nullcontext
    from types import SimpleNamespace
    from unittest.mock import Mock

    from vllm_fl.worker import worker

    runner = Mock(spec=["initialize_kv_cache"])
    config = SimpleNamespace(
        num_blocks=7,
        kv_cache_layout=None,
        hash_block_size=32,
        cache_hit_alignment_tokens=16,
        needs_kv_cache_zeroing=False,
    )
    cache = SimpleNamespace()
    fake = SimpleNamespace(
        cache_config=cache,
        vllm_config=object(),
        model_runner=runner,
        _maybe_get_memory_pool_context=lambda **kw: nullcontext(),
    )
    monkeypatch.setattr(worker, "ensure_kv_transfer_initialized", Mock())
    monkeypatch.setattr(worker, "has_kv_transfer_group", lambda: False)
    worker.WorkerFL.initialize_from_config(fake, config)
    assert cache.num_gpu_blocks == 7
    assert cache.hash_block_size == 32
    assert cache.cache_hit_alignment_tokens == 16
    args, kwargs = runner.initialize_kv_cache.call_args
    assert args == (config,)
    assert "kv_cache_allocation_context" in kwargs


def test_gcu_does_not_query_cuda_allocator_settings(monkeypatch):
    from types import SimpleNamespace

    from vllm_fl.worker import worker

    monkeypatch.setattr(worker, "current_platform", SimpleNamespace(device_type="gcu"))

    def forbidden():
        raise AssertionError("CUDA allocator queried on GCU")

    monkeypatch.setattr(worker.torch.cuda.memory, "get_allocator_backend", forbidden)
    with worker.WorkerFL._scoped_allocator_max_split(object(), 20):
        pass


def test_gcu_forward_uses_standardized_kv_views(monkeypatch, gcu_attention_module):
    from types import SimpleNamespace
    from unittest.mock import Mock

    import torch

    attention = gcu_attention_module

    impl = attention.AttentionGCUImpl.__new__(attention.AttentionGCUImpl)
    impl.attn_type = attention.AttentionType.DECODER
    impl.head_size = 8
    impl.num_kv_heads = 2
    impl.kv_sharing_target_layer_name = None
    impl.kv_cache_dtype = "auto"
    impl.dcp_world_size = 1
    impl.scale = 1.0
    impl.alibi_slopes = None
    impl.sliding_window = (-1, -1)
    impl.logits_soft_cap = 0
    impl.vllm_flash_attn_version = 2
    impl.sinks = None
    meta = SimpleNamespace(
        num_actual_tokens=1,
        slot_mapping=torch.tensor([0]),
        use_cascade=False,
        query_start_loc=torch.tensor([0, 1]),
        seq_lens=torch.tensor([1]),
        max_query_len=1,
        max_seq_len=1,
        block_table=torch.tensor([[0]]),
        scheduler_metadata=None,
        causal=True,
        max_num_splits=0,
    )
    cache = torch.arange(3 * 2 * 16 * 16).reshape(3, 2, 16, 16).float()
    write = Mock()
    flash = Mock()
    monkeypatch.setattr(attention, "reshape_and_cache_flash", write)
    monkeypatch.setattr(attention, "flash_attn_varlen_func", flash)
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
    assert flash.call_args.kwargs["k"] is k
    assert flash.call_args.kwargs["v"] is v


def test_compiled_gcu_dispatch_warmup():
    """Execute the actual pre-trace guard without loading model weights."""
    from textwrap import dedent
    from types import SimpleNamespace
    from unittest.mock import Mock

    from vllm_fl.worker import model_runner

    tree = ast.parse(dedent(inspect.getsource(model_runner.ModelRunnerFL.load_model)))
    guards = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.If)
        and any(
            isinstance(call, ast.Call)
            and isinstance(call.func, ast.Name)
            and call.func.id in {"warmup_cached_ops", "_warmup_cached_ops"}
            for call in ast.walk(node)
        )
    ]
    guard = min(guards, key=lambda node: len(list(ast.walk(node))))
    early_calls = [
        node
        for node in tree.body[0].body
        if isinstance(node, ast.Expr)
        and isinstance(node.value, ast.Call)
        and isinstance(node.value.func, ast.Name)
        and node.value.func.id == "warmup_cached_ops"
    ]
    assert len(early_calls) == 1
    assert early_calls[0].lineno < guard.lineno
    warmup = Mock()
    exec(
        compile(
            ast.Module(body=early_calls, type_ignores=[]), "<early warmup>", "exec"
        ),
        {"warmup_cached_ops": warmup},
    )
    warmup.assert_called_once_with()
    body = [
        node
        for node in guard.body
        if not isinstance(node, (ast.Import, ast.ImportFrom))
    ]
    isolated = ast.fix_missing_locations(
        ast.Module(
            body=[ast.If(test=guard.test, body=body, orelse=[])], type_ignores=[]
        )
    )
    for device, mode, expected in [
        ("gcu", 3, 1),
        ("npu", 3, 1),
        ("gcu", 0, 0),
        ("cuda", 3, 0),
    ]:
        warmup.reset_mock()
        exec(
            compile(isolated, "<warmup guard>", "exec"),
            {
                "current_platform": SimpleNamespace(device_type=device),
                "self": SimpleNamespace(compilation_config=SimpleNamespace(mode=mode)),
                "CompilationMode": SimpleNamespace(NONE=0),
                "warmup_cached_ops": warmup,
                "_warmup_cached_ops": warmup,
            },
        )
        assert warmup.call_count == expected
