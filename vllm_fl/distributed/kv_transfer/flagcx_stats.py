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
# SPDX-License-Identifier: Apache-2.0
"""Stats and Prometheus metrics for the FlagCX connector."""

import threading
from dataclasses import dataclass
from typing import Any

import numpy as np

from vllm.config import VllmConfig
from vllm.distributed.kv_transfer.kv_connector.v1.metrics import (
    KVConnectorPromMetrics,
    KVConnectorStats,
    PromMetric,
    PromMetricT,
)
from vllm.v1.metrics.utils import create_metric_per_engine


@dataclass
class FlagCXKVConnectorStats(KVConnectorStats):
    """Container for FlagCX KV transfer performance metrics.

    `_lock` serializes record_* against clone_and_reset so each row's
    appends are atomic and column lengths stay aligned. Writers run on
    the sender pool / receiver loop; the reader runs on the main worker
    thread.
    """

    def __post_init__(self):
        self._lock = threading.Lock()
        if not self.data:
            self.reset()

    # threading.Lock is not picklable; strip it from the wire form and
    # rebuild a fresh per-process lock on the receiver side.
    def __getstate__(self) -> dict[str, Any]:
        state = self.__dict__.copy()
        state.pop("_lock", None)
        return state

    def __setstate__(self, state: dict[str, Any]) -> None:
        self.__dict__.update(state)
        self._lock = threading.Lock()

    def reset(self):
        self.data: dict[str, list[float | int]] = {
            "transfer_duration": [],
            "bytes_transferred": [],
            "num_descriptors": [],
            "num_failed_transfers": [],
            "num_failed_recvs": [],
            "num_kv_expired_reqs": [],
        }

    def record_transfer(self, duration_s: float, total_bytes: int, num_descs: int):
        with self._lock:
            self.data["transfer_duration"].append(duration_s)
            self.data["bytes_transferred"].append(total_bytes)
            self.data["num_descriptors"].append(num_descs)

    # Failure counters store a list of 1s so a future Prom counter can iterate
    # with .inc(list_item), mirroring NIXL's NixlPromMetrics.observe.
    def record_failed_transfer(self):
        with self._lock:
            self.data["num_failed_transfers"].append(1)

    def record_failed_recv(self):
        with self._lock:
            self.data["num_failed_recvs"].append(1)

    def record_kv_expired_req(self):
        with self._lock:
            self.data["num_kv_expired_reqs"].append(1)

    def clone_and_reset(self) -> "FlagCXKVConnectorStats":
        # Copy lists under the lock for length alignment; return a fresh
        # instance so the snapshot has its own _lock.
        with self._lock:
            snapshot_data: dict[str, list[float | int]] = {
                k: list(v) for k, v in self.data.items()
            }
            self.reset()
        return FlagCXKVConnectorStats(data=snapshot_data)

    def is_empty(self) -> bool:
        return (
            self.num_successful_transfers == 0
            and len(self.data["num_failed_transfers"]) == 0
            and len(self.data["num_failed_recvs"]) == 0
            and len(self.data["num_kv_expired_reqs"]) == 0
        )

    def aggregate(self, other: KVConnectorStats) -> KVConnectorStats:
        if not other.is_empty():
            for k, v in other.data.items():
                accumulator = self.data[k]
                assert isinstance(accumulator, list)
                accumulator.extend(v)
        return self

    def reduce(self) -> dict[str, int | float]:
        num_failed_transfers = len(self.data["num_failed_transfers"])
        num_failed_recvs = len(self.data["num_failed_recvs"])
        num_kv_expired_reqs = len(self.data["num_kv_expired_reqs"])

        if self.num_successful_transfers == 0:
            return {
                "Num successful transfers": 0,
                "Avg xfer time (ms)": 0,
                "P90 xfer time (ms)": 0,
                "Avg MB per transfer": 0,
                "Throughput (MB/s)": 0,
                "Avg number of descriptors": 0,
                "Num failed transfers": num_failed_transfers,
                "Num failed recvs": num_failed_recvs,
                "Num KV expired reqs": num_kv_expired_reqs,
            }

        xfer_time = np.asarray(self.data["transfer_duration"])
        mb = np.asarray(self.data["bytes_transferred"]) / 2**20
        descs = np.asarray(self.data["num_descriptors"], dtype=np.uint32)
        n = len(descs)
        assert n == self.num_successful_transfers

        total_mb = mb.sum()
        avg_mb = total_mb / n
        total_time_seconds = xfer_time.sum()
        throughput_mb_s = (
            total_mb / total_time_seconds if total_time_seconds > 0 else 0.0
        )

        return {
            "Num successful transfers": n,
            "Avg xfer time (ms)": round(xfer_time.mean() * 1e3, 3),
            "P90 xfer time (ms)": round(np.percentile(xfer_time, 90).item() * 1e3, 3),
            "Avg MB per transfer": round(avg_mb, 3),
            "Throughput (MB/s)": round(throughput_mb_s, 3),
            "Avg number of descriptors": round(descs.mean(), 1),
            "Num failed transfers": num_failed_transfers,
            "Num failed recvs": num_failed_recvs,
            "Num KV expired reqs": num_kv_expired_reqs,
        }

    @property
    def num_successful_transfers(self) -> int:
        return len(self.data["transfer_duration"])


class FlagCXPromMetrics(KVConnectorPromMetrics):
    """Prometheus metrics for FlagCX KV transfers.

    Counters (monotonic, so ``rate()``/``increase()`` work in PromQL) carry
    the volume series; histograms carry the per-transfer distributions:

      vllm:flagcx_total_bytes_transferred -> rate() = KV transfer bytes/s
      vllm:flagcx_total_xfer_time_seconds -> ratio with the above gives
                                             effective bytes/s while the link
                                             was actually busy
      vllm:flagcx_transfers               -> rate() = transfers/s
      vllm:flagcx_total_descriptors       -> descriptors/s
      vllm:flagcx_num_failed_transfers    -> P-side write failures
      vllm:flagcx_num_failed_recvs        -> D-side side-channel failures
      vllm:flagcx_num_kv_expired_reqs     -> P-side unsent, expired requests
      vllm:flagcx_xfer_time_seconds       -> per-transfer latency histogram
      vllm:flagcx_bytes_transferred       -> per-transfer size histogram
      vllm:flagcx_num_descriptors         -> per-transfer descriptor count

    Counter names deliberately avoid a ``_total`` suffix: prometheus_client
    strips it and appends its own, so ``vllm:flagcx_xfer_time_seconds_total``
    would register as ``vllm:flagcx_xfer_time_seconds`` and collide with the
    histogram of the same name. The scraped counters still end in ``_total``.

    Same P/D asymmetry as the stats container: FlagCX is P-push, so the
    success series only advance on the P engine, while
    ``num_failed_recvs`` only advances on D.
    """

    def __init__(
        self,
        vllm_config: VllmConfig,
        metric_types: dict[type[PromMetric], type[PromMetricT]],
        labelnames: list[str],
        per_engine_labelvalues: dict[int, list[object]],
    ):
        super().__init__(vllm_config, metric_types, labelnames, per_engine_labelvalues)

        # ---- Counters: the series to use with rate()/increase(). ----
        # (metric name, documentation, stats data key)
        counter_specs: list[tuple[str, str, str]] = [
            (
                "vllm:flagcx_total_bytes_transferred",
                "Total bytes transferred by FlagCX KV Cache transfers.",
                "bytes_transferred",
            ),
            (
                "vllm:flagcx_total_xfer_time_seconds",
                "Total time spent inside FlagCX KV Cache transfers. Divide "
                "vllm:flagcx_total_bytes_transferred_total by this to get "
                "effective link throughput.",
                "transfer_duration",
            ),
            (
                "vllm:flagcx_transfers",
                "Number of successful FlagCX KV Cache transfers.",
                "num_transfers",
            ),
            (
                "vllm:flagcx_total_descriptors",
                "Total number of descriptors written by FlagCX KV Cache transfers.",
                "num_descriptors",
            ),
            (
                "vllm:flagcx_num_failed_transfers",
                "Number of failed FlagCX KV Cache transfers. "
                "NOTE: This metric is tracked on the P instance.",
                "num_failed_transfers",
            ),
            (
                "vllm:flagcx_num_failed_recvs",
                "Number of failed FlagCX KV Cache receives (side-channel or "
                "transfer errors reported to D). "
                "NOTE: This metric is tracked on the D instance.",
                "num_failed_recvs",
            ),
            (
                "vllm:flagcx_num_kv_expired_reqs",
                "Number of requests that had their KV expire before being sent. "
                "NOTE: This metric is tracked on the P instance.",
                "num_kv_expired_reqs",
            ),
        ]
        self._counters: list[tuple[dict[int, PromMetric], str]] = []
        for name, documentation, data_key in counter_specs:
            counter = self._counter_cls(
                name=name,
                documentation=documentation,
                labelnames=labelnames,
            )
            self._counters.append(
                (
                    create_metric_per_engine(counter, self.per_engine_labelvalues),
                    data_key,
                )
            )

        # ---- Histograms: per-transfer distributions. ----
        histogram_xfer_time = self._histogram_cls(
            name="vllm:flagcx_xfer_time_seconds",
            documentation="Histogram of transfer duration for FlagCX KV Cache "
            "transfers.",
            buckets=[
                0.005,
                0.01,
                0.025,
                0.05,
                0.075,
                0.1,
                0.2,
                0.3,
                0.5,
                0.75,
                1.0,
                5.0,
            ],
            labelnames=labelnames,
        )
        # Uniform 2KB to 16GB range, matching NixlPromMetrics.
        histogram_bytes_transferred = self._histogram_cls(
            name="vllm:flagcx_bytes_transferred",
            documentation="Histogram of bytes transferred per FlagCX KV Cache "
            "transfer.",
            buckets=[2 ** (10 + i) for i in range(1, 25, 2)],
            labelnames=labelnames,
        )
        histogram_num_descriptors = self._histogram_cls(
            name="vllm:flagcx_num_descriptors",
            documentation="Histogram of number of descriptors per FlagCX KV Cache "
            "transfer.",
            buckets=[
                10,
                20,
                30,
                50,
                75,
                100,
                200,
                400,
                1000,
                2000,
                4000,
                10000,
                20000,
                50000,
            ],
            labelnames=labelnames,
        )
        self._histograms: list[tuple[dict[int, PromMetric], str]] = [
            (
                create_metric_per_engine(
                    histogram_xfer_time, self.per_engine_labelvalues
                ),
                "transfer_duration",
            ),
            (
                create_metric_per_engine(
                    histogram_bytes_transferred, self.per_engine_labelvalues
                ),
                "bytes_transferred",
            ),
            (
                create_metric_per_engine(
                    histogram_num_descriptors, self.per_engine_labelvalues
                ),
                "num_descriptors",
            ),
        ]

    def observe(self, transfer_stats_data: dict[str, Any], engine_idx: int = 0):
        # `num_transfers` is not stored in the stats container: one successful
        # transfer is one `transfer_duration` observation, so the counter is
        # advanced by that column's length.
        num_transfers = len(transfer_stats_data.get("transfer_duration", ()))
        for counter_obj, data_key in self._counters:
            counter = counter_obj.get(engine_idx)
            if counter is None:
                continue
            if data_key == "num_transfers":
                counter.inc(num_transfers)
                continue
            for value in transfer_stats_data.get(data_key, ()):
                counter.inc(value)
        for histogram_obj, data_key in self._histograms:
            histogram = histogram_obj.get(engine_idx)
            if histogram is None:
                continue
            for value in transfer_stats_data.get(data_key, ()):
                histogram.observe(value)
