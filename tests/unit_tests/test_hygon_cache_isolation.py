# SPDX-License-Identifier: Apache-2.0
"""Exercise the real runner and Python startup across independent workers."""

import json
import os
import sys
import textwrap
from pathlib import Path

import pytest

from tests.run import TestCase as Case, TestRunner as Runner


def test_hygon_e2e_workers_use_distinct_autotuning_databases(monkeypatch, tmp_path):
    original_cache = str(tmp_path / "persistent")
    monkeypatch.setenv("FLAGGEMS_CACHE_DIR", original_cache)
    startup = tmp_path / "startup"
    startup.mkdir()
    (startup / "sitecustomize.py").write_text(
        "import os\n"
        "marker = str(os.getpid())\n"
        "from sitecustomize import marker as current\n"
        "assert current == marker\n"
        "os.environ['FL_CACHE_TEST_STARTUP_PID'] = marker\n"
    )
    monkeypatch.setenv("PYTHONPATH", str(startup))
    runner = Runner("hygon", device="bw1000", scope="e2e")
    probe = textwrap.dedent(
        """
        import json
        import multiprocessing
        import os
        import sqlite3
        import sys
        import time
        import traceback
        from pathlib import Path

        def worker(parent_cache, results):
            try:
                pid = os.getpid()
                assert os.environ['FL_CACHE_TEST_STARTUP_PID'] == str(pid)
                cache = Path(os.environ['FLAGGEMS_CACHE_DIR'])
                cache.mkdir(parents=True, exist_ok=True)
                # The parent holds an exclusive transaction on this same
                # filename. A shared directory must fail with a real lock.
                with sqlite3.connect(cache / 'tuning.db', timeout=0.25) as db:
                    db.execute('CREATE TABLE benchmark (pid INTEGER)')
                    db.execute('INSERT INTO benchmark VALUES (?)', (pid,))
                    assert db.execute('SELECT pid FROM benchmark').fetchall() == [(pid,)]
                assert cache.name == str(pid)
                assert cache.parent == Path(parent_cache)
                results.put(str(cache))
            except BaseException:
                # The runner captures stdout; preserve the actual child error
                # there as well as the process's nonzero exit status.
                traceback.print_exc(file=sys.stdout)
                raise

        def main():
            pid = os.getpid()
            assert os.environ['FL_CACHE_TEST_STARTUP_PID'] == str(pid)
            cache = Path(os.environ['FLAGGEMS_CACHE_DIR'])
            cache.mkdir(parents=True, exist_ok=True)
            context = multiprocessing.get_context('spawn')
            results = context.Queue()
            workers = []
            try:
                with sqlite3.connect(cache / 'tuning.db') as db:
                    db.execute('CREATE TABLE benchmark (pid INTEGER)')
                    db.execute('BEGIN EXCLUSIVE')
                    db.execute('INSERT INTO benchmark VALUES (?)', (pid,))
                    deadline = time.monotonic() + 20
                    for _ in range(4):
                        child = context.Process(target=worker, args=(str(cache), results))
                        child.start()
                        workers.append(child)
                    for child in workers:
                        child.join(timeout=max(0, deadline - time.monotonic()))
                        assert not child.is_alive(), 'spawn worker timed out'
                        assert child.exitcode == 0, f'worker {child.pid}: {child.exitcode}'
                    paths = [results.get(timeout=2) for _ in workers]
                    assert db.execute('SELECT pid FROM benchmark').fetchall() == [(pid,)]
                assert cache.name == str(pid)
                assert len(set(paths)) == 4
                print(json.dumps([str(cache), *paths]))
            finally:
                for child in workers:
                    if child.is_alive():
                        child.terminate()
                    child.join(timeout=2)
                    if child.is_alive():
                        child.kill()
                        child.join(timeout=2)
                results.close()
                results.join_thread()

        if __name__ == '__main__':
            main()
        """
    )
    script = tmp_path / "cache_probe.py"
    script.write_text(probe)
    monkeypatch.setattr(
        runner, "_build_pytest_cmd", lambda _: [sys.executable, str(script)]
    )
    result = runner._run_single(
        Case("cache-probe", "unused", task="inference"), _quiet=True
    )
    assert result.passed, result.stdout
    paths = json.loads(result.stdout)
    assert len(set(paths)) == 5
    parent = Path(paths[0])
    assert all(Path(path).parent == parent for path in paths[1:])
    assert not parent.parent.exists(), "case-local tuning caches must be cleaned up"
    assert os.environ["FLAGGEMS_CACHE_DIR"] == original_cache


@pytest.mark.parametrize(
    "platform,device,task",
    [("hygon", "bw1000", "unit"), ("nvidia", "a100", "inference")],
)
def test_other_scopes_preserve_cache_environment(
    monkeypatch, tmp_path, platform, device, task
):
    cache = str(tmp_path / "persistent")
    monkeypatch.setenv("FLAGGEMS_CACHE_DIR", cache)
    runner = Runner(platform, device=device, scope=task)
    probe = "import os; print(os.environ['FLAGGEMS_CACHE_DIR'])"
    monkeypatch.setattr(
        runner, "_build_pytest_cmd", lambda _: [sys.executable, "-c", probe]
    )
    result = runner._run_single(Case("cache-probe", "unused", task=task), _quiet=True)
    assert result.passed, result.stdout
    assert result.stdout.strip() == cache
