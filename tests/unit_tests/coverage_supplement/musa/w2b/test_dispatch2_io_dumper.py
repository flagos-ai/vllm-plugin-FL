"""Extra branch coverage for ``vllm_fl.dispatch.io_dumper`` internals.

Complements w1 (filtering / serialization) and the repo-level io_dumper
tests by exercising the async I/O helpers, hook lifecycle, summary
accumulation and the env/YAML initialisation merge.
"""
import concurrent.futures
import json
import os
import types
from pathlib import Path
import pytest
import torch
import torch.nn as nn
from vllm_fl.dispatch import io_common as c
from vllm_fl.dispatch import io_dumper as i
_IO_ENV_KEYS = ['VLLM_FL_IO_DUMP', 'VLLM_FL_IO_DUMP_OPS', 'VLLM_FL_IO_DUMP_MODULES', 'VLLM_FL_IO_DUMP_LAYERS', 'VLLM_FL_IO_DUMP_MAX_CALLS', 'VLLM_FL_IO_DUMP_STEP_RANGE', 'VLLM_FL_IO_DUMP_RANK', 'VLLM_FL_IO_DUMP_WITH_TORCH_FUNCS', 'VLLM_FL_IO_DUMP_WITH_VALUES', 'VLLM_FL_IO_DUMP_WITH_METAS', 'VLLM_FL_IO_DUMP_WITH_PRINT', 'VLLM_FL_IO_DUMP_BG_WORKERS', 'VLLM_FL_CONFIG']

def _clear_env(monkeypatch):
    for key in _IO_ENV_KEYS:
        monkeypatch.delenv(key, raising=False)

@pytest.fixture(autouse=True)
def _clean_state(monkeypatch):
    _clear_env(monkeypatch)
    c.reset_step()
    c.reset_exec_order()
    c.reset_rank()
    for tls in (i._dump_pairing, i._print_pairing):
        try:
            delattr(tls, 'stack')
        except AttributeError:
            pass
    yield
    i._enabled = False
    i._hooks_activated = False
    try:
        i._deactivate_hooks()
    except Exception:
        pass
    i._reset_state()
    i._clear_env_vars()
    c.set_io_active(False)
    c.reset_step()
    c.reset_exec_order()
    c.reset_rank()

class _FakeEvent:

    def __init__(self):
        self.synced = 0

    def synchronize(self):
        self.synced += 1

class TestExecutor:

    def test_get_executor_creates_and_reuses(self):
        i._io_executor = None
        ex1 = i._get_executor()
        try:
            assert isinstance(ex1, concurrent.futures.ThreadPoolExecutor)
            assert i._get_executor() is ex1
        finally:
            ex1.shutdown()

    def test_get_executor_recreates_after_shutdown(self):

        class _Dead:
            _shutdown = True
        i._io_executor = _Dead()
        ex = i._get_executor()
        try:
            assert isinstance(ex, concurrent.futures.ThreadPoolExecutor)
        finally:
            ex.shutdown()

    def test_submit_bg_disabled(self, monkeypatch):
        monkeypatch.setattr(i, '_enabled', False)
        submitted = []

        class _Mgr:

            def submit(self, *a, **k):
                submitted.append(1)
        monkeypatch.setattr(i, '_get_executor', lambda : _Mgr())
        i._submit_bg(lambda : None)
        assert submitted == []

    def test_submit_bg_enabled_tracks_future(self, monkeypatch):
        monkeypatch.setattr(i, '_enabled', True)

        class _Fut:
            pass

        class _Mgr:

            def submit(self, fn, *a, **k):
                fn()
                return _Fut()
        monkeypatch.setattr(i, '_get_executor', lambda : _Mgr())
        monkeypatch.setattr(i, '_pending_futures', [])
        i._submit_bg(lambda : None)
        assert len(i._pending_futures) == 1

class TestFileHandles:

    def test_open_reuses_and_close(self, tmp_path, monkeypatch):
        monkeypatch.setattr(i, '_open_file_handles', {})
        path = str(tmp_path / 'input.json')
        h1 = i._get_or_open_handle(path)
        h2 = i._get_or_open_handle(path)
        assert h1 is h2
        assert os.path.isfile(path)
        i._close_open_files()
        assert i._open_file_handles == {}
        assert h1.fh.closed is True

    def test_bg_compute_writes_in_call_order(self, tmp_path, monkeypatch):
        monkeypatch.setattr(i, '_open_file_handles', {})
        path = str(tmp_path / 'out.json')
        ev = _FakeEvent()
        refs = {'arg_0': torch.ones(2)}
        i._bg_compute_and_write(path, 2, {'op_name': 'mm'}, refs, ev)
        i._bg_compute_and_write(path, 1, {'op_name': 'mm'}, refs, ev)
        i._close_open_files()
        lines = Path(path).read_text(encoding='utf-8').splitlines()
        assert len(lines) == 2
        assert 'call_1' in lines[0]
        assert 'call_2' in lines[1]
        parsed = json.loads(lines[0])
        assert parsed['call_1']['tensors']['arg_0']['shape'] == [2]
        assert ev.synced == 2

    def test_bg_save_pt_input_and_output(self, tmp_path):
        ev = _FakeEvent()
        in_path = str(tmp_path / 'in.pt')
        i._bg_save_pt((torch.ones(2),), {'k': 3}, is_output=False, pt_path=in_path, device_event=ev)
        data = torch.load(in_path, weights_only=False)
        assert set(data) == {'arg_0', 'kwarg_k'}
        assert ev.synced == 1
        out_path = str(tmp_path / 'out.pt')
        i._bg_save_pt(((torch.zeros(1), torch.ones(1)),), {}, is_output=True, pt_path=out_path, device_event=None)
        out = torch.load(out_path, weights_only=False)
        assert set(out) == {'result_0', 'result_1'}

class TestWaitAndFlush:

    def test_failing_future_logged(self, monkeypatch):
        warnings = []
        monkeypatch.setattr(i, 'logger', types.SimpleNamespace(warning=lambda *a, **k: warnings.append(a)))
        i._open_file_handles.clear()
        ex = concurrent.futures.ThreadPoolExecutor(1)
        fut = ex.submit(lambda : 1 / 0)
        i._pending_futures.append(fut)
        try:
            i._wait_and_flush()
        finally:
            ex.shutdown()
        assert warnings
        assert i._pending_futures == []

    def test_close_oserror_swallowed(self, monkeypatch):

        class _BadFH:

            def flush(self):
                raise OSError('flush')

            def close(self):
                raise OSError('close')
        i._open_file_handles.clear()
        i._open_file_handles['x'] = types.SimpleNamespace(fh=_BadFH())
        i._pending_futures.clear()
        i._wait_and_flush()
        assert i._open_file_handles == {}

class TestResetState:

    def test_shutdown_and_close(self, monkeypatch):

        class _Ex:
            _shutdown = False

            def __init__(self):
                self.called = False

            def shutdown(self, wait):
                self.called = True

        class _BadFH:

            def close(self):
                raise OSError('close')
        ex = _Ex()
        monkeypatch.setattr(i, '_io_executor', ex)
        monkeypatch.setattr(i, '_pending_futures', [object()])
        i._open_file_handles.clear()
        i._open_file_handles['x'] = types.SimpleNamespace(fh=_BadFH())
        i._reset_state()
        assert ex.called is True
        assert i._io_executor is None
        assert i._pending_futures == []
        assert i._open_file_handles == {}

    def test_reset_clears_config(self):
        i._enabled = True
        i._match_all = True
        i._op_filter = {'a'}
        i._call_counters['x'] = 1
        i._op_summary['y'] = {'dispatch_keys': '', 'call_count': 1}
        i._reset_state()
        assert i._enabled is False
        assert i._match_all is False
        assert i._op_filter == set()
        assert i._call_counters == {}
        assert i._op_summary == {}

class TestDumpHelpers:

    def test_dump_input_exception_swallowed(self, monkeypatch):
        warnings = []
        monkeypatch.setattr(i, 'logger', types.SimpleNamespace(warning=lambda *a, **k: warnings.append(a)))
        monkeypatch.setattr(i, '_next_call_num', lambda name: (_ for _ in ()).throw(RuntimeError('counter')))
        i._dump_input('op', (1,), {}, exec_order=0)
        assert warnings

    def test_dump_output_without_pairing(self, monkeypatch):
        warnings = []
        monkeypatch.setattr(i, 'logger', types.SimpleNamespace(warning=lambda *a, **k: warnings.append(a)))
        try:
            delattr(i._dump_pairing, 'stack')
        except AttributeError:
            pass
        i._dump_output('op', 1)
        assert warnings

class _FakeMgr:

    def __init__(self, entered=False):
        self.entered = entered
        self.requests = []
        self.enters = []

    def is_entered(self, name):
        return self.entered

    def request_exit(self, name):
        self.requests.append(name)

    def enter(self, name, instance):
        self.enters.append(name)

class TestModeLifecycle:

    def test_enter_modes(self, monkeypatch):
        dmgr = _FakeMgr()
        fmgr = _FakeMgr()
        monkeypatch.setattr(i, 'dispatch_mode_mgr', dmgr)
        monkeypatch.setattr(i, 'func_mode_mgr', fmgr)
        monkeypatch.setattr(i, 'HAS_TORCH_DISPATCH_MODE', True)
        monkeypatch.setattr(i, 'HAS_TORCH_FUNC_MODE', True)
        monkeypatch.setattr(i, '_DumpDispatchMode', lambda : 'd', raising=False)
        monkeypatch.setattr(i, '_DumpTorchFuncMode', lambda : 'f', raising=False)
        monkeypatch.setattr(i, '_torch_funcs_enabled', True)
        i._enter_dispatch_modes()
        assert dmgr.enters == ['dump']
        assert fmgr.enters == ['dump']

    def test_enter_modes_without_torch_funcs(self, monkeypatch):
        dmgr = _FakeMgr()
        fmgr = _FakeMgr()
        monkeypatch.setattr(i, 'dispatch_mode_mgr', dmgr)
        monkeypatch.setattr(i, 'func_mode_mgr', fmgr)
        monkeypatch.setattr(i, 'HAS_TORCH_DISPATCH_MODE', True)
        monkeypatch.setattr(i, 'HAS_TORCH_FUNC_MODE', True)
        monkeypatch.setattr(i, '_DumpDispatchMode', lambda : 'd', raising=False)
        monkeypatch.setattr(i, '_torch_funcs_enabled', False)
        i._enter_dispatch_modes()
        assert dmgr.enters == ['dump']
        assert fmgr.enters == []

    def test_enter_modes_unavailable(self, monkeypatch):
        dmgr = _FakeMgr()
        fmgr = _FakeMgr()
        monkeypatch.setattr(i, 'dispatch_mode_mgr', dmgr)
        monkeypatch.setattr(i, 'func_mode_mgr', fmgr)
        monkeypatch.setattr(i, 'HAS_TORCH_DISPATCH_MODE', False)
        monkeypatch.setattr(i, 'HAS_TORCH_FUNC_MODE', False)
        i._enter_dispatch_modes()
        assert dmgr.enters == [] and fmgr.enters == []

    def test_exit_modes(self, monkeypatch):
        dmgr = _FakeMgr()
        fmgr = _FakeMgr()
        monkeypatch.setattr(i, 'dispatch_mode_mgr', dmgr)
        monkeypatch.setattr(i, 'func_mode_mgr', fmgr)
        i._exit_dispatch_modes()
        assert dmgr.requests == ['dump']
        assert fmgr.requests == ['dump']

    def test_pause_entered(self, monkeypatch):
        dmgr = _FakeMgr(entered=True)
        fmgr = _FakeMgr(entered=True)
        monkeypatch.setattr(i, 'dispatch_mode_mgr', dmgr)
        monkeypatch.setattr(i, 'func_mode_mgr', fmgr)
        i.pause_dispatch_modes()
        assert dmgr.requests == ['dump']
        assert fmgr.requests == ['dump']

    def test_pause_not_entered(self, monkeypatch):
        dmgr = _FakeMgr(entered=False)
        fmgr = _FakeMgr(entered=False)
        monkeypatch.setattr(i, 'dispatch_mode_mgr', dmgr)
        monkeypatch.setattr(i, 'func_mode_mgr', fmgr)
        i.pause_dispatch_modes()
        assert dmgr.requests == [] and fmgr.requests == []

    def test_resume_reenenters(self, monkeypatch):
        calls = []
        monkeypatch.setattr(i, '_enabled', True)
        monkeypatch.setattr(i, '_hooks_activated', True)
        monkeypatch.setattr(i, 'dispatch_mode_mgr', _FakeMgr(entered=False))
        monkeypatch.setattr(i, '_enter_dispatch_modes', lambda : calls.append(1))
        i.resume_dispatch_modes()
        assert calls == [1]

    def test_resume_noop(self, monkeypatch):
        calls = []
        monkeypatch.setattr(i, '_enabled', False)
        monkeypatch.setattr(i, '_hooks_activated', True)
        monkeypatch.setattr(i, 'dispatch_mode_mgr', _FakeMgr(entered=False))
        monkeypatch.setattr(i, '_enter_dispatch_modes', lambda : calls.append(1))
        i.resume_dispatch_modes()
        assert calls == []

class TestActivateDeactivate:

    def test_activate_defers_when_step_starts_later(self, monkeypatch):
        entered = []
        monkeypatch.setattr(i, '_step_range', (2, 5))
        monkeypatch.setattr(i, 'register_step_callback', lambda cb: None)
        monkeypatch.setattr(i, '_enter_dispatch_modes', lambda : entered.append(1))
        i._activate_hooks()
        assert entered == []
        assert i._hooks_activated is True

    def test_activate_immediately(self, monkeypatch):
        entered = []
        monkeypatch.setattr(i, '_step_range', None)
        monkeypatch.setattr(i, 'register_step_callback', lambda cb: None)
        monkeypatch.setattr(i, '_enter_dispatch_modes', lambda : entered.append(1))
        i._activate_hooks()
        assert entered == [1]

    def test_deactivate(self, monkeypatch):
        calls = []
        monkeypatch.setattr(i, '_hooks_activated', True)
        monkeypatch.setattr(i, 'unregister_step_callback', lambda cb: calls.append('u'))
        monkeypatch.setattr(i, '_remove_module_hooks', lambda : calls.append('r'))
        monkeypatch.setattr(i, '_exit_dispatch_modes', lambda : calls.append('e'))
        i._deactivate_hooks()
        assert calls == ['u', 'r', 'e']
        assert i._hooks_activated is False

class TestModuleHooks:

    def test_register_run_and_remove(self, monkeypatch):
        monkeypatch.setattr(i, '_enabled', True)
        monkeypatch.setattr(i, '_module_hook_handles', [])
        model = nn.Sequential(nn.Linear(2, 2))
        try:
            i.register_io_module_hooks(model)
            assert len(i._module_hook_handles) == 2 * len(list(model.modules()))
            model(torch.zeros(1, 2))
            assert c.get_current_module() is None
            i.register_io_module_hooks(model)
            assert len(i._module_hook_handles) == 2 * len(list(model.modules()))
        finally:
            i._remove_module_hooks()
        assert i._module_hook_handles == []

    def test_noop_when_disabled(self, monkeypatch):
        monkeypatch.setattr(i, '_enabled', False)
        monkeypatch.setattr(i, '_module_hook_handles', [])
        i.register_io_module_hooks(nn.Linear(2, 2))
        assert i._module_hook_handles == []

class TestOnStepAdvance:

    def test_lazy_activation(self, monkeypatch):
        entered = []
        monkeypatch.setattr(i, '_step_range', (1, 3))
        monkeypatch.setattr(i, 'dispatch_mode_mgr', _FakeMgr(entered=False))
        monkeypatch.setattr(i, '_enter_dispatch_modes', lambda : entered.append(1))
        monkeypatch.setattr(i, '_dump_dir', '')
        i._on_step_advance(0, set(), set())
        assert entered == [1]

    def test_lazy_activation_skipped_when_entered(self, monkeypatch):
        entered = []
        monkeypatch.setattr(i, '_step_range', (1, 3))
        monkeypatch.setattr(i, 'dispatch_mode_mgr', _FakeMgr(entered=True))
        monkeypatch.setattr(i, '_enter_dispatch_modes', lambda : entered.append(1))
        monkeypatch.setattr(i, '_dump_dir', '')
        i._on_step_advance(0, set(), set())
        assert entered == []

    def test_clears_counters_and_logs(self, monkeypatch):
        log_calls = []
        print_calls = []
        monkeypatch.setattr(i, '_step_range', None)
        monkeypatch.setattr(i, '_dump_dir', '')
        monkeypatch.setattr(i, '_print_enabled', True)
        monkeypatch.setattr(i, '_call_counters', {'op': 3})
        monkeypatch.setattr(i, 'logger', types.SimpleNamespace(info=lambda *a, **k: log_calls.append(a)))
        monkeypatch.setattr(i, '_print_logger', types.SimpleNamespace(info=lambda *a, **k: print_calls.append(a)))
        i._on_step_advance(2, {'Linear'}, {'mm'})
        assert i._call_counters == {}
        assert len(log_calls) == 1
        assert len(print_calls) == 1

    def test_no_logging_when_empty(self, monkeypatch):
        log_calls = []
        print_calls = []
        monkeypatch.setattr(i, '_step_range', None)
        monkeypatch.setattr(i, '_dump_dir', '')
        monkeypatch.setattr(i, 'logger', types.SimpleNamespace(info=lambda *a, **k: log_calls.append(a)))
        monkeypatch.setattr(i, '_print_logger', types.SimpleNamespace(info=lambda *a, **k: print_calls.append(a)))
        i._on_step_advance(1, set(), set())
        assert log_calls == [] and print_calls == []

    def test_dump_dir_flushes_and_writes_summary(self, monkeypatch):
        calls = []
        monkeypatch.setattr(i, '_step_range', None)
        monkeypatch.setattr(i, '_dump_dir', '/tmp/x')
        monkeypatch.setattr(i, '_wait_and_flush', lambda : calls.append('flush'))
        monkeypatch.setattr(i, '_write_summary', lambda : calls.append('summary'))
        i._on_step_advance(0, set(), set())
        assert calls == ['flush', 'summary']

class TestSummary:

    def test_record_op_summary_new_and_existing(self):
        i._op_summary.clear()
        i._record_op_summary('mm', [('CUDA', 'FlagGems', False)])
        assert i._op_summary['mm']['dispatch_keys'] == '[(CUDA, FlagGems, False)]'
        assert i._op_summary['mm']['call_count'] == 1
        i._record_op_summary('mm', None)
        assert i._op_summary['mm']['dispatch_keys'] == '[(CUDA, FlagGems, False)]'
        assert i._op_summary['mm']['call_count'] == 2

    def test_record_op_summary_without_keys(self):
        i._op_summary.clear()
        i._record_op_summary('op', None)
        assert i._op_summary['op']['dispatch_keys'] == ''

    def test_is_flaggems_from_string(self, monkeypatch):
        import vllm_fl.dispatch.manager as mgr_mod
        assert i._is_flaggems_op('mm', '... (CUDA, FlagGems, False) ...') is True
        monkeypatch.setattr(mgr_mod, 'get_default_manager', lambda : (_ for _ in ()).throw(RuntimeError('skip real manager')))
        assert i._is_flaggems_op('unused', '') is False

    def test_is_flaggems_from_manager(self, monkeypatch):
        import vllm_fl.dispatch.manager as mgr_mod
        from vllm_fl.dispatch.types import BackendImplKind
        impl = types.SimpleNamespace(impl_id='default.flagos', kind=BackendImplKind.DEFAULT)

        class _Snap:
            impls_by_op = {'rms_norm': [impl]}

        class _Registry:

            def snapshot(self):
                return _Snap()

        class _Mgr:
            _called_ops = {'rms_norm': 'default.flagos'}
            _registry = _Registry()
        monkeypatch.setattr(mgr_mod, 'get_default_manager', lambda : _Mgr())
        assert i._is_flaggems_op('rms_norm', '') is True

    def test_is_flaggems_from_manager_reference(self, monkeypatch):
        import vllm_fl.dispatch.manager as mgr_mod
        from vllm_fl.dispatch.types import BackendImplKind
        impl = types.SimpleNamespace(impl_id='reference.torch', kind=BackendImplKind.REFERENCE)

        class _Snap:
            impls_by_op = {'rms_norm': [impl]}

        class _Registry:

            def snapshot(self):
                return _Snap()

        class _Mgr:
            _called_ops = {'rms_norm': 'reference.torch'}
            _registry = _Registry()
        monkeypatch.setattr(mgr_mod, 'get_default_manager', lambda : _Mgr())
        assert i._is_flaggems_op('rms_norm', '') is False

    def test_is_flaggems_manager_exception(self, monkeypatch):
        import vllm_fl.dispatch.manager as mgr_mod
        monkeypatch.setattr(mgr_mod, 'get_default_manager', lambda : (_ for _ in ()).throw(RuntimeError('no manager')))
        assert i._is_flaggems_op('x', '') is False

    def test_write_summary_skips(self, monkeypatch):
        monkeypatch.setattr(i, '_dump_dir', '')
        i._write_summary()
        monkeypatch.setattr(i, '_dump_dir', '/tmp/x')
        monkeypatch.setattr(i, '_op_summary', {})
        i._write_summary()

    def test_write_summary_success(self, tmp_path, monkeypatch):
        monkeypatch.setattr(i, '_dump_dir', str(tmp_path))
        monkeypatch.setattr(i, '_op_summary', {'fg': {'dispatch_keys': 'x', 'call_count': 1}, 'plain': {'dispatch_keys': '', 'call_count': 1}})
        monkeypatch.setattr(i, '_is_flaggems_op', lambda op, dk: op == 'fg')
        i._write_summary()
        summary = json.loads((tmp_path / 'rank_0000' / 'summary.json').read_text(encoding='utf-8'))
        assert summary['flaggems_ops'] == ['fg']
        assert summary['non_flaggems_ops'] == ['plain']

    def test_write_summary_oserror(self, tmp_path, monkeypatch):
        warnings = []
        monkeypatch.setattr(i, '_dump_dir', str(tmp_path))
        monkeypatch.setattr(i, '_op_summary', {'op': {'dispatch_keys': '', 'call_count': 1}})
        monkeypatch.setattr(i.os, 'replace', lambda *a, **k: (_ for _ in ()).throw(OSError('disk full')))
        monkeypatch.setattr(i, 'logger', types.SimpleNamespace(warning=lambda *a, **k: warnings.append(a)))
        i._write_summary()
        assert warnings

class TestInitFromEnv:

    def test_already_enabled_returns(self, monkeypatch):
        monkeypatch.setattr(i, '_enabled', True)
        monkeypatch.setattr(i, '_dump_dir', '/keep')
        i._init_from_env()
        assert i._dump_dir == '/keep'

    def test_explicit_zero_disables(self, monkeypatch):
        monkeypatch.setenv('VLLM_FL_IO_DUMP', '0')
        i._init_from_env()
        assert i._enabled is False

    def test_no_config_noop(self):
        i._init_from_env()
        assert i._enabled is False

    def test_filters_auto_enable(self, tmp_path, monkeypatch):
        monkeypatch.setenv('VLLM_FL_IO_DUMP_STEP_RANGE', '0-1')
        monkeypatch.setattr(i.os, 'getcwd', lambda : str(tmp_path))
        i._init_from_env()
        assert i._enabled is True
        assert i._dump_dir == str(tmp_path / 'io_dump')
        assert i._step_range == (0, 2)

    def test_print_only(self, monkeypatch):
        monkeypatch.setenv('VLLM_FL_IO_DUMP_WITH_PRINT', '1')
        i._init_from_env()
        assert i._enabled is True
        assert i._dump_dir == ''
        assert i._print_enabled is True

    def test_yaml_config(self, tmp_path, monkeypatch):
        cfg = tmp_path / 'cfg.yaml'
        cfg.write_text(f"io_dump:\n  dir: {tmp_path / 'dump'}\n  ops: [rms_norm]\n  max_calls: 3\n  step_range: '1-2'\n  with_metas: true\n  with_values: true\n  ranks: [0]\n", encoding='utf-8')
        monkeypatch.setenv('VLLM_FL_CONFIG', str(cfg))
        i._init_from_env()
        assert i._enabled is True
        assert i._dump_dir == str(tmp_path / 'dump')
        assert i._op_filter == {'rms_norm'}
        assert i._max_calls == 3
        assert i._step_range == (1, 3)
        assert i._with_metas is True
        assert i._with_values is True
        assert i._rank_filter == {0}

    def test_makedirs_failure_disables(self, tmp_path, monkeypatch):
        monkeypatch.setenv('VLLM_FL_IO_DUMP', str(tmp_path / 'd'))
        monkeypatch.setattr(i.os, 'makedirs', lambda *a, **k: (_ for _ in ()).throw(OSError('no perms')))
        monkeypatch.setattr(i, 'logger', types.SimpleNamespace(warning=lambda *a, **k: None))
        i._init_from_env()
        assert i._enabled is False

class TestEnableIoDump:

    @pytest.fixture(autouse=True)
    def _patch_hooks(self, monkeypatch):
        monkeypatch.setattr(i, '_activate_hooks', lambda : None)

    def test_print_only_skips_file_io(self):
        i.enable_io_dump(with_print=True)
        assert i._print_enabled is True
        assert i._dump_dir == ''

    def test_env_vars_parsed(self, tmp_path, monkeypatch):
        monkeypatch.setenv('VLLM_FL_IO_DUMP', str(tmp_path / 'dump'))
        monkeypatch.setenv('VLLM_FL_IO_DUMP_OPS', 'a, b')
        monkeypatch.setenv('VLLM_FL_IO_DUMP_MODULES', 'Linear')
        monkeypatch.setenv('VLLM_FL_IO_DUMP_LAYERS', '1')
        monkeypatch.setenv('VLLM_FL_IO_DUMP_MAX_CALLS', 'not-int')
        monkeypatch.setenv('VLLM_FL_IO_DUMP_STEP_RANGE', '2-3')
        monkeypatch.setenv('VLLM_FL_IO_DUMP_WITH_TORCH_FUNCS', 'mm,add')
        monkeypatch.setenv('VLLM_FL_IO_DUMP_RANK', '0,2')
        monkeypatch.setenv('VLLM_FL_IO_DUMP_WITH_VALUES', 'true')
        monkeypatch.setenv('VLLM_FL_IO_DUMP_WITH_METAS', '1')
        monkeypatch.setenv('VLLM_FL_IO_DUMP_WITH_PRINT', '0')
        monkeypatch.setenv('VLLM_FL_IO_DUMP_BG_WORKERS', 'oops')
        i.enable_io_dump()
        assert i._op_filter == {'a', 'b'}
        assert i._module_filter == {'Linear'}
        assert 'model.layers.1' in i._layer_filter
        assert i._max_calls == 0
        assert i._step_range == (2, 4)
        assert i._torch_funcs_enabled is True
        assert i._torch_func_filter == {'mm', 'add'}
        assert i._rank_filter == {0, 2}
        assert i._with_values is True
        assert i._with_metas is True
        assert i._print_enabled is False
        assert i._bg_workers == 8

    def test_api_overrides(self, tmp_path):
        i.enable_io_dump(dump_dir=str(tmp_path / 'api'), ops={'rms_norm'}, layers='0', max_calls=4, step_range='1-2', with_values=True, with_metas=True, ranks={1}, bg_workers=0)
        assert i._dump_dir == str(tmp_path / 'api')
        assert i._op_filter == {'rms_norm'}
        assert i._match_all is False
        assert 'model.layers.0' in i._layer_filter
        assert i._max_calls == 4
        assert i._step_range == (1, 3)
        assert i._with_values is True
        assert i._with_metas is True
        assert i._rank_filter == {1}
        assert i._bg_workers == 1

    def test_falsy_dump_dir_uses_cwd(self, tmp_path, monkeypatch):
        monkeypatch.setattr(i.os, 'getcwd', lambda : str(tmp_path))
        i.enable_io_dump(dump_dir='')
        assert i._dump_dir == str(tmp_path / 'io_dump')

    def test_yaml_config_applied(self, tmp_path, monkeypatch):
        cfg = tmp_path / 'cfg.yaml'
        cfg.write_text(f"io_dump:\n  dir: {tmp_path / 'yaml_dump'}\n  modules: Linear\n  max_calls: 2\n  with_torch_funcs: [mm]\n  with_print: false\n  with_metas: true\n", encoding='utf-8')
        monkeypatch.setenv('VLLM_FL_CONFIG', str(cfg))
        i.enable_io_dump()
        assert i._dump_dir == str(tmp_path / 'yaml_dump')
        assert i._module_filter == {'Linear'}
        assert i._max_calls == 2
        assert i._torch_funcs_enabled is True
        assert i._torch_func_filter == {'mm'}
        assert i._print_enabled is False
        assert i._with_metas is True

class TestEnvVarPropagation:

    def test_set_and_clear(self, monkeypatch):
        i._set_env_vars('/d', {'b', 'a'}, {'M'}, {'model.layers.1'}, 3, (1, 3), True, {2, 0}, True, True, True)
        assert os.environ['VLLM_FL_IO_DUMP'] == '/d'
        assert os.environ['VLLM_FL_IO_DUMP_OPS'] == 'a,b'
        assert os.environ['VLLM_FL_IO_DUMP_MODULES'] == 'M'
        assert os.environ['VLLM_FL_IO_DUMP_LAYERS'] == 'model.layers.1'
        assert os.environ['VLLM_FL_IO_DUMP_MAX_CALLS'] == '3'
        assert os.environ['VLLM_FL_IO_DUMP_STEP_RANGE'] == '1-2'
        assert os.environ['VLLM_FL_IO_DUMP_WITH_TORCH_FUNCS'] == '1'
        assert os.environ['VLLM_FL_IO_DUMP_RANK'] == '0,2'
        assert os.environ['VLLM_FL_IO_DUMP_WITH_VALUES'] == '1'
        assert os.environ['VLLM_FL_IO_DUMP_WITH_METAS'] == '1'
        assert os.environ['VLLM_FL_IO_DUMP_WITH_PRINT'] == '1'
        i._set_env_vars('', None, None, set(), 0, None, False, None, False, False, False)
        assert os.environ['VLLM_FL_IO_DUMP'] == ''
        for key in ('VLLM_FL_IO_DUMP_OPS', 'VLLM_FL_IO_DUMP_MODULES', 'VLLM_FL_IO_DUMP_LAYERS', 'VLLM_FL_IO_DUMP_MAX_CALLS', 'VLLM_FL_IO_DUMP_STEP_RANGE', 'VLLM_FL_IO_DUMP_WITH_TORCH_FUNCS', 'VLLM_FL_IO_DUMP_RANK', 'VLLM_FL_IO_DUMP_WITH_VALUES', 'VLLM_FL_IO_DUMP_WITH_METAS', 'VLLM_FL_IO_DUMP_WITH_PRINT'):
            assert key not in os.environ