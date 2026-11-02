"""Exercise the worker PP control flow without constructing a device worker."""
import ast
import contextlib
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest


def _execute_model():
    tree = ast.parse((Path(__file__).parents[2] / 'vllm_fl/worker/worker.py').read_text())
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == 'WorkerFL')
    method = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == 'execute_model')
    method.decorator_list = []
    module = ast.Module(body=[ast.ImportFrom(module='__future__', names=[ast.alias(name='annotations')], level=0), method], type_ignores=[])
    ast.fix_missing_locations(module)
    return compile(module, '<worker execute_model>', 'exec')


@pytest.mark.parametrize('first_rank', [True, False])
def test_pp_send_tracks_only_device_work(first_rank):
    class Intermediate:
        def __init__(self, tensors, **kwargs):
            self.tensors = tensors

    class Output:
        pass

    handles = [Mock(), Mock(), Mock()]
    pp = SimpleNamespace(is_first_rank=first_rank, is_last_rank=False,
                         irecv_tensor_dict=Mock(return_value=({'hidden': 1}, [], [])),
                         isend_tensor_dict=Mock(return_value=handles))
    tp = object()
    ns = dict(get_pp_group=lambda: pp, get_tp_group=lambda: tp,
              AsyncIntermediateTensors=Intermediate, IntermediateTensors=Intermediate,
              ModelRunnerOutput=Output, AsyncModelRunnerOutput=Output, NoneType=type(None))
    exec(_execute_model(), ns)
    previous = Mock()
    runner = Mock()
    runner.execute_model.return_value = Intermediate({'hidden': 2})
    worker = SimpleNamespace(_pp_send_work=[previous], model_runner=runner,
                             use_v2_model_runner=False,
                             annotate_profile=lambda _: contextlib.nullcontext(),
                             vllm_config=SimpleNamespace(parallel_config=SimpleNamespace(distributed_executor_backend='mp')))
    assert ns['execute_model'](worker, SimpleNamespace(total_num_scheduled_tokens=1)) is None
    previous.wait.assert_called_once()
    pp.isend_tensor_dict.assert_called_once_with({'hidden': 2}, all_gather_group=tp)
    assert worker._pp_send_work == handles[1:]
    if not first_rank:
        pp.irecv_tensor_dict.assert_called_once_with(all_gather_group=tp)
    else:
        pp.irecv_tensor_dict.assert_not_called()


@pytest.mark.parametrize('vendor,enabled', [('iluvatar', False), ('nvidia', False), ('nvidia', True)])
def test_kv_init_routed_expert_config_migration(vendor, enabled):
    tree = ast.parse((Path(__file__).parents[2] / 'vllm_fl/worker/worker.py').read_text())
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == 'WorkerFL')
    method = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == 'initialize_from_config')
    method.decorator_list = []
    module = ast.fix_missing_locations(ast.Module(body=[ast.ImportFrom(module='__future__', names=[ast.alias(name='annotations')], level=0), method], type_ignores=[]))
    namespace = {'current_platform': SimpleNamespace(vendor_name=vendor), 'ensure_kv_transfer_initialized': Mock()}
    exec(compile(module, '<worker KV init>', 'exec'), namespace)
    model_config = SimpleNamespace() if vendor == 'iluvatar' else SimpleNamespace(enable_return_routed_experts=enabled)
    worker = SimpleNamespace(cache_config=SimpleNamespace(), model_config=model_config,
                             vllm_config=object(), model_runner=Mock(),
                             _maybe_get_memory_pool_context=lambda **_: contextlib.nullcontext())
    kv = SimpleNamespace(num_blocks=17, needs_kv_cache_zeroing=False)
    namespace['initialize_from_config'](worker, kv)
    worker.model_runner.initialize_kv_cache.assert_called_once_with(kv)
    assert worker.cache_config.num_gpu_blocks == 17
    assert worker.model_runner.init_routed_experts_capturer.call_count == int(enabled)
