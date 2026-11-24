# Copyright 2026 FlagOS Contributors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""G9 connector margin tests (814b ``flagcx_connector``).

Adds coverage for the sender-worker status/error paths and the stats/metrics
builders that the G5 file does not reach.  The module-level ``get_kv_cache_layout``
fallback mirrors the G5 fix so this file is independent.
"""

from __future__ import annotations

import types
from types import SimpleNamespace

import pytest


def _ensure_kv_layout_symbol():
    try:
        import vllm.v1.attention.backends.utils as utils
    except Exception:  # pragma: no cover
        return
    if not hasattr(utils, "get_kv_cache_layout"):
        utils.get_kv_cache_layout = lambda *a, **k: "NHD"
    if not hasattr(utils, "NULL_BLOCK_ID"):
        utils.NULL_BLOCK_ID = -1


_ensure_kv_layout_symbol()

try:  # pragma: no cover - environment specific
    import vllm_fl.distributed.kv_transfer.flagcx_connector as mod
except Exception as exc:  # pragma: no cover
    pytest.skip(f"flagcx_connector unavailable: {exc}", allow_module_level=True)


class _Pusher:
    def __init__(self, error=None):
        self.sent = []
        self.closed = False
        self.error = error

    def send_multipart(self, frames):
        if self.error is not None:
            raise self.error
        self.sent.append(frames)

    def close(self):
        self.closed = True


def _worker_with_sender(pusher, *, decode_error=None):
    worker = object.__new__(mod.FlagCXConnectorWorker)
    worker.zmq_ctx = object()

    def _decode(raw):
        if decode_error is not None:
            raise decode_error
        return "META"

    worker._decoder = SimpleNamespace(decode=_decode)
    sent = []
    worker._send_kv_to_decode = lambda meta: sent.append(meta)
    return worker, sent, pusher


def test_sender_worker_marks_transfer_done(monkeypatch):
    pusher = _Pusher()
    monkeypatch.setattr(
        mod, "make_zmq_socket", lambda ctx, path, kind: pusher, raising=False
    )
    fake_zmq = SimpleNamespace(PUSH="push", ZMQError=type("ZMQError", (Exception,), {}))
    monkeypatch.setattr(mod, "zmq", fake_zmq, raising=False)

    worker, sent, _ = _worker_with_sender(pusher)
    mod.FlagCXConnectorWorker._sender_worker(worker, b"id", b"raw", "/path")

    assert sent == ["META"]
    assert pusher.sent == [(b"id", mod.TRANS_DONE)]
    assert pusher.closed is True


def test_sender_worker_reports_error_and_swallows_send_failure(monkeypatch):
    zmq_error = type("ZMQError", (Exception,), {})
    pusher = _Pusher(error=zmq_error("down"))
    monkeypatch.setattr(
        mod, "make_zmq_socket", lambda ctx, path, kind: pusher, raising=False
    )
    fake_zmq = SimpleNamespace(PUSH="push", ZMQError=zmq_error)
    monkeypatch.setattr(mod, "zmq", fake_zmq, raising=False)

    worker, sent, _ = _worker_with_sender(
        pusher, decode_error=RuntimeError("bad payload")
    )
    mod.FlagCXConnectorWorker._sender_worker(worker, b"id", b"raw", "/path")

    assert sent == []
    # The status was computed but the send raised and was swallowed.
    assert pusher.sent == []
    assert pusher.closed is True


def test_build_connector_stats_uses_stats_class(monkeypatch):
    created = []

    class _Stats:
        def __init__(self, data):
            created.append(data)

    monkeypatch.setattr(mod, "FlagCXKVConnectorStats", _Stats, raising=False)
    mod.FlagCXConnector.build_kv_connector_stats(None)
    mod.FlagCXConnector.build_kv_connector_stats({"x": 1})
    assert created == [{}, {"x": 1}]


def test_build_prom_metrics_uses_prom_class(monkeypatch):
    created = []

    class _Prom:
        def __init__(self, *args):
            created.append(args)

    monkeypatch.setattr(mod, "FlagCXPromMetrics", _Prom, raising=False)
    result = mod.FlagCXConnector.build_prom_metrics("cfg", {}, ["a"], {0: ["x"]})
    assert isinstance(result, _Prom)
    assert created == [("cfg", {}, ["a"], {0: ["x"]})]


def test_facade_asserts_without_scheduler():
    facade = object.__new__(mod.FlagCXConnector)
    facade.connector_scheduler = None
    with pytest.raises(AssertionError):
        facade.get_num_new_matched_tokens(object(), 0)


def test_producer_cache_is_replicated_reads_topology():
    worker = object.__new__(mod.FlagCXConnectorWorker)
    worker.kv_topo = SimpleNamespace(local_replicates_kv_cache=True)
    assert mod.FlagCXConnectorWorker._producer_cache_is_replicated(worker) is True
    worker.kv_topo.local_replicates_kv_cache = False
    assert mod.FlagCXConnectorWorker._producer_cache_is_replicated(worker) is False
