import concurrent.futures as cf
import datetime
import hashlib
import json
import os
from pathlib import Path
import signal
import socket
import subprocess
import sys
import threading

ACTIVE = set()
ACTIVE_LOCK = threading.Lock()


def cancel(signum, frame):
    with ACTIVE_LOCK:
        processes = tuple(ACTIVE)
    for process in processes:
        try:
            os.killpg(process.pid, signal.SIGTERM)
        except ProcessLookupError:
            pass
    raise SystemExit(128 + signum)

REPO = Path('/workspace/pr675-remove-added-tests')
SRC = REPO / 'tools/adaptation-gate-cases'
sys.path.insert(0, str(REPO))
from tests.utils.model_config import ModelConfig

ROOT = Path('/workspace/adaptation-gate-results') / ('cyber-review-' + datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%SZ'))
CASES = os.environ.get('REVIEW_CASES', '27b_tp2_eager,27b_tp2_graph,35b_a3b_tp2_eager,35b_a3b_tp2_graph').split(',')
DEVICE_OFFSET = int(os.environ.get('DEVICE_OFFSET', '0'))


def prepare(case, index):
    cfg = ModelConfig.load('qwen3_6', case, platform='enflame', device='s60')
    model = '/models/' + ('Qwen3.6-27B' if case.startswith('27b') else 'Qwen3.6-35B-A3B')
    assert Path(model, 'config.json').is_file()
    port = int(os.environ.get('REVIEW_PORT_BASE', '18400')) + index
    with socket.socket() as sock:
        assert sock.connect_ex(('127.0.0.1', port)) != 0, port
    command = [sys.executable, '-m', 'vllm.entrypoints.cli.main', 'serve', model,
               '--tensor-parallel-size', '2', '--served-model-name', 'qwen',
               '--host', '127.0.0.1', '--port', str(port),
               '--allowed-local-media-path', str(SRC / 'images')] + cfg.serve_args()
    env = os.environ.copy()
    env.update(TOPS_VISIBLE_DEVICES=f'{DEVICE_OFFSET+index*2},{DEVICE_OFFSET+index*2+1}', VLLM_PLUGINS='fl',
               PYTHONPATH=str(REPO), PYTHONUNBUFFERED='1', MODEL_PATH=model,
               SERVED_MODEL_NAME='qwen', PORT=str(port), SERVICE_TIMEOUT='3000',
               REQUEST_TIMEOUT='600', NO_PROXY='localhost,127.0.0.1',
               no_proxy='localhost,127.0.0.1', TORCH_GCU_ENABLE_INT64_AND_UINT64='1',
               ENABLE_I64_CHECK='0', TORCHGCU_INDUCTOR_ENABLE='1')
    for key in ('BASE_URL', 'VLLM_USE_V2_MODEL_RUNNER', 'GRAPH_COMPILATION_CONFIG'):
        env.pop(key, None)
    return command, env


def stop(process):
    if process.poll() is None:
        os.killpg(process.pid, signal.SIGINT)
        try:
            process.wait(timeout=45)
        except subprocess.TimeoutExpired:
            os.killpg(process.pid, signal.SIGKILL)
            process.wait()


def run_case(item):
    index, case = item
    command, env = prepare(case, index)
    out = ROOT / case
    out.mkdir(parents=True)
    env['RESULTS_DIR'] = str(out / 'results')
    (out / 'command.json').write_text(json.dumps(command, indent=2))
    server = None
    client = None
    result = {'case': case, 'devices': env['TOPS_VISIBLE_DEVICES']}
    try:
        with (out / 'server.log').open('w') as slog, (out / 'client.log').open('w') as clog:
            server = subprocess.Popen(command, cwd=REPO, env=env, stdout=slog, stderr=subprocess.STDOUT, start_new_session=True)
            client = subprocess.Popen(['bash', str(SRC / 'run_test.sh')], cwd=SRC, env=env, stdout=clog, stderr=subprocess.STDOUT, start_new_session=True)
            with ACTIVE_LOCK:
                ACTIVE.update((server, client))
            result['client_exit'] = client.wait(timeout=5400)
            result['server_exit_before_cleanup'] = server.poll()
    except Exception as exc:
        result['error'] = repr(exc)
    finally:
        with ACTIVE_LOCK:
            ACTIVE.discard(server)
            ACTIVE.discard(client)
        for proc in (client, server):
            if proc is not None:
                stop(proc)
        (out / 'status.json').write_text(json.dumps(result, indent=2))
    print(json.dumps(result), flush=True)
    return result


def main():
    signal.signal(signal.SIGTERM, cancel)
    signal.signal(signal.SIGINT, cancel)
    for index, case in enumerate(CASES):
        prepare(case, index)
    if '--check' in sys.argv:
        print('PASS four real ModelConfig commands, model files and available ports')
        return
    ROOT.mkdir(parents=True)
    diff = subprocess.check_output(['git', 'diff', '--binary', 'HEAD'], cwd=REPO)
    (ROOT / 'worktree.patch').write_bytes(diff)
    hashes = {}
    paths = subprocess.check_output(['git', 'ls-files', '--modified', '--others', '--exclude-standard'], cwd=REPO, text=True).splitlines()
    for path in paths:
        hashes[path] = hashlib.sha256((REPO / path).read_bytes()).hexdigest()
    (ROOT / 'manifest.json').write_text(json.dumps({'head': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=REPO, text=True).strip(), 'files': hashes}, indent=2))
    print('OUTPUT_ROOT=' + str(ROOT), flush=True)
    with cf.ThreadPoolExecutor(max_workers=4) as pool:
        results = list(pool.map(run_case, enumerate(CASES)))
    (ROOT / 'status.json').write_text(json.dumps(results, indent=2))
    if any(r.get('client_exit') != 0 for r in results):
        raise SystemExit(1)


if __name__ == '__main__':
    main()
