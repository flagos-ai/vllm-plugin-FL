"""Real dry-run CLI chain must preserve an explicit sync-scheduling request."""
import json
from pathlib import Path
import subprocess
import sys

import pytest

PLUGIN = Path(__file__).resolve().parents[2]
DEPLOY = PLUGIN / "deployment/glm52-hygon"


@pytest.mark.parametrize("sync", [False, True])
def test_both_nodes_receive_sync_scheduling_without_disabling_graph(tmp_path, sync):
    command = [sys.executable, str(DEPLOY / "profile_dual_node.py"),
               "--project", str(tmp_path), "--model", "diagnostic-model",
               "--run-root", str(tmp_path / "not-created"),
               "--benchmark-only", "--dry-run"]
    if sync:
        command.append("--no-async-scheduling")
    result = subprocess.run(command, text=True, capture_output=True, timeout=15)
    assert result.returncode == 0, result.stderr
    plan = json.loads(result.stdout)
    for rank, server in enumerate(plan["servers"]):
        runner_args = server[server.index(str(DEPLOY / "dual_node_smoke.py")) + 1:]
        child = subprocess.run([sys.executable, str(DEPLOY / "dual_node_smoke.py"),
                                *runner_args, "--dry-run"],
                               text=True, capture_output=True, timeout=15)
        assert child.returncode == 0, child.stderr
        launch = json.loads(child.stdout)
        assert launch["config"]["node_rank"] == rank
        assert launch["config"]["enforce_eager"] is False
        assert launch["config"]["compilation_config"]["cudagraph_capture_sizes"] == [1,2,4,8,16,32,64]
        if sync:
            assert launch["config"]["async_scheduling"] is False
            assert "--no-async-scheduling" in launch["cli"]
        else:
            assert "async_scheduling" not in launch["config"]
            assert "--no-async-scheduling" not in launch["cli"]
    assert not (tmp_path / "not-created").exists()


@pytest.mark.parametrize('split', [False, True])
def test_split_communicator_reaches_both_nodes_without_disabling_async(tmp_path, split):
    command = [sys.executable, str(DEPLOY / 'profile_dual_node.py'),
               '--project', str(tmp_path), '--model', 'diagnostic-model',
               '--run-root', str(tmp_path / 'not-created'),
               '--benchmark-only', '--dry-run']
    if split:
        command.append('--split-graph-comm')
    result = subprocess.run(command, text=True, capture_output=True, timeout=15)
    assert result.returncode == 0, result.stderr
    for server in json.loads(result.stdout)['servers']:
        child = subprocess.run([*server, '--dry-run'], text=True,
                               capture_output=True, timeout=15)
        assert child.returncode == 0, child.stderr
        launch = json.loads(child.stdout)
        assert launch['env']['VLLM_FL_HYGON_SPLIT_GRAPH_COMM'] == ('1' if split else '0')
        assert 'async_scheduling' not in launch['config']
        assert '--no-async-scheduling' not in launch['cli']
        assert launch['config']['enforce_eager'] is False
    assert not (tmp_path / 'not-created').exists()


def test_window_profile_keeps_workload_and_uses_offline_export(tmp_path):
    result = subprocess.run([
        sys.executable, str(DEPLOY / 'profile_dual_node.py'),
        '--project', str(tmp_path), '--model', 'diagnostic-model',
        '--run-root', str(tmp_path / 'not-created'), '--dry-run',
        '--profile-seconds', '6', '--profile-devices', '2',
        '--kv-cache-memory-bytes', '1248563903',
        '--num-prompts', '64', '--output-len', '1024',
    ], text=True, capture_output=True, timeout=15)
    assert result.returncode == 0, result.stderr
    plan = json.loads(result.stdout)
    assert plan['profile_window_seconds'] == 6
    assert plan['profile_window_policy'] == 'natural_exit_then_offline_index_range'
    assert plan['profile_devices_each'] == [2]
    for server in plan['servers']:
        assert server[server.index('--devices') + 1] == '2'
        child = subprocess.run(server[server.index('env'):] + ['--dry-run'],
                               text=True, capture_output=True, timeout=15)
        assert child.returncode == 0, child.stderr
        assert json.loads(child.stdout)['config']['kv_cache_memory_bytes'] == 1248563903
    for bench in plan['benches'].values():
        assert bench[bench.index('--output-len') + 1] == '1024'
        assert bench[bench.index('--num-prompts') + 1] == '64'
    assert plan['export'][plan['export'].index('--devices') + 1] == '2'
    assert not (tmp_path / 'not-created').exists()


@pytest.mark.parametrize('flags', [
    ['--profile-seconds', '0'],
    ['--profile-seconds', '-1'],
    ['--profile-seconds', '6', '--benchmark-only'],
    ['--profile-devices', '8'],
    ['--profile-devices', '2,2'],
    ['--profile-seconds', '6', '--profile-devices', '1,2'],
])
def test_invalid_profile_window_refused_before_side_effects(tmp_path, flags):
    result = subprocess.run([
        sys.executable, str(DEPLOY / 'profile_dual_node.py'),
        '--project', str(tmp_path), '--model', 'diagnostic-model',
        '--run-root', str(tmp_path / 'not-created'), '--dry-run', *flags,
    ], text=True, capture_output=True, timeout=15)
    assert result.returncode == 2
    assert not (tmp_path / 'not-created').exists()


def test_scheduler_metrics_preserve_histograms_and_ignore_unrelated():
    import runpy
    module = runpy.run_path(str(DEPLOY / 'profile_dual_node.py'))
    assert 'scheduler_metrics' in module
    metrics = module['scheduler_metrics'](
        '# HELP ignored\n'
        'vllm:num_requests_running{model_name="glm",engine="0"} 23.0\n'
        'vllm:num_requests_waiting{model_name="glm",engine="0"} 41.0\n'
        'vllm:kv_cache_usage_perc{model_name="glm",engine="0"} 0.99\n'
        'vllm:iteration_tokens_total_bucket{le="32",engine="0"} 120.0\n'
        'process_cpu_seconds_total 99.0\n')
    assert metrics['running'] == 23
    assert metrics['waiting'] == 41
    assert metrics['kv_usage'] == .99
    assert len(metrics['samples']) == 4
    assert metrics['samples'][-1].endswith('120.0')


def test_window_indices_preserve_entire_graph_launch_and_source(tmp_path):
    import hashlib
    import runpy
    import sqlite3
    module = runpy.run_path(str(DEPLOY / 'profile_dual_node.py'))
    assert 'profile_index_window' in module
    source = tmp_path / 'trace.db'
    with sqlite3.connect(source) as c:
        c.execute('CREATE TABLE HIPOPS_example (BeginNs INTEGER, EndNs INTEGER, '
                  '_Index INTEGER, DurationNs INTEGER, dev_id INTEGER, pid INTEGER)')
        c.executemany('INSERT INTO HIPOPS_example VALUES (?,?,?,?,?,?)', [
            (1000000000, 1100000000, 10, 100000000, 2, 123),
            (6900000000, 7000000000, 20, 100000000, 2, 123),
            # Same graph launch index: preserve it across the nominal 6s edge.
            (7010000000, 7050000000, 20, 40000000, 2, 123),
            (7100000000, 7200000000, 30, 100000000, 2, 123),
        ])
    before = hashlib.sha256(source.read_bytes()).hexdigest()
    window = module['profile_index_window'](source, 6, 2)
    assert window['index_range'] == '10:20'
    assert window['pid'] == 123
    assert window['kernel_calls'] == 3
    assert window['kernel_total_ns'] == 240000000
    assert window['actual_span_ns'] == 6050000000
    assert hashlib.sha256(source.read_bytes()).hexdigest() == before
    with pytest.raises(ValueError, match='exactly one'):
        module['profile_index_window'](source, 6, 0)


def test_shutdown_grace_allows_owned_process_to_finish(tmp_path):
    import runpy
    module = runpy.run_path(str(DEPLOY / 'profile_dual_node.py'))
    assert 'shutdown_group_command' in module
    child = subprocess.Popen([
        sys.executable, '-u', '-c',
        'import signal,time,sys; '
        'signal.signal(signal.SIGINT, lambda *a: '
        '(time.sleep(.2), print("FLUSH_FINISHED",flush=True), sys.exit(0))); '
        'print("READY",flush=True); time.sleep(30)',
        'dual_node_smoke',
    ], stdout=subprocess.PIPE, text=True, start_new_session=True)
    try:
        assert child.stdout.readline().strip() == 'READY'
        command, timeout = module['shutdown_group_command'](child.pid, 2)
        result = subprocess.run(command, text=True, capture_output=True,
                                timeout=timeout)
        assert result.returncode == 0, result.stderr
        assert 'forced cleanup' not in result.stdout
        assert 'FLUSH_FINISHED' in child.communicate(timeout=5)[0]
        assert child.returncode == 0
    finally:
        if child.poll() is None:
            child.kill()
            child.wait(timeout=5)


def test_profile_shutdown_budget_is_explicit_and_bounded(tmp_path):
    command = [sys.executable, str(DEPLOY / 'profile_dual_node.py'),
               '--project', str(tmp_path), '--model', 'diagnostic-model',
               '--run-root', str(tmp_path / 'not-created'), '--dry-run',
               '--shutdown-seconds']
    result = subprocess.run(command + ['900'], text=True, capture_output=True,
                            timeout=15)
    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout)['args']['shutdown_seconds'] == 900
    for invalid in ['0', '-1', '1801']:
        result = subprocess.run(command + [invalid], text=True,
                                capture_output=True, timeout=15)
        assert result.returncode == 2
    assert not (tmp_path / 'not-created').exists()


@pytest.mark.parametrize("route", ["vendor", "triton_grouped"])
def test_fp8_route_and_explicit_dependency_paths_reach_both_nodes(tmp_path, route):
    gems, compressor = tmp_path / "gems", tmp_path / "compressor"
    command = [sys.executable, str(DEPLOY / "profile_dual_node.py"),
               "--project", str(tmp_path), "--model", "diagnostic-model",
               "--run-root", str(tmp_path / "not-created"), "--benchmark-only", "--dry-run",
               "--flaggems-root", str(gems), "--compressor-root", str(compressor),
               "--kv-cache-dtype", "fp8_ds_mla", "--fp8-mla-impl", route,
               "--fp8-reuse-metadata", "--split-graph-comm"]
    result = subprocess.run(command, text=True, capture_output=True, timeout=15)
    assert result.returncode == 0, result.stderr
    for server in json.loads(result.stdout)["servers"]:
        child = subprocess.run([*server, "--dry-run"], text=True, capture_output=True, timeout=15)
        assert child.returncode == 0, child.stderr
        plan = json.loads(child.stdout)
        assert plan["config"]["kv_cache_dtype"] == "fp8_ds_mla"
        assert plan["cli"][plan["cli"].index("--kv-cache-dtype") + 1] == "fp8_ds_mla"
        assert plan["env"]["VLLM_FL_GLM_FP8_MLA_IMPL"] == route
        assert plan["env"]["VLLM_FL_GLM_FP8_REUSE_METADATA"] == "1"
        assert plan["env"]["GLM52_FLAGGEMS_ROOT"] == str(gems)
        assert plan["env"]["GLM52_COMPRESSOR_ROOT"] == str(compressor)
    assert not (tmp_path / "not-created").exists()
