# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Discover a Prefill TP group's side-channel addresses through its first rank.

This module deliberately has no vLLM, torch, or FlagCX runtime dependencies.
Only small control messages use this protocol; KV transfers still use FlagCX.
"""

from __future__ import annotations

import asyncio
import time
from collections.abc import Iterable

import msgspec
import zmq
import zmq.asyncio

GET_TP_ADDRESSES = b"flagcx_get_tp_addresses_v1"
LEGACY_TRANSFER_ERROR = b"trans_error"


class RankAddress(msgspec.Struct, frozen=True):
    host: str
    port: int


class TPAddresses(msgspec.Struct, frozen=True):
    engine_id: str
    ranks: dict[int, RankAddress]


def tcp_address(host: str, port: int) -> str:
    if ":" in host and not host.startswith("["):
        host = f"[{host}]"
    return f"tcp://{host}:{port}"


def validate_addresses(metadata: TPAddresses) -> TPAddresses:
    if not metadata.engine_id or not metadata.ranks:
        raise ValueError("FlagCX discovery requires an engine ID and TP ranks")
    if set(metadata.ranks) != set(range(len(metadata.ranks))):
        raise ValueError("FlagCX discovery requires contiguous TP ranks from zero")
    for rank, address in metadata.ranks.items():
        if not address.host or address.host in ("0.0.0.0", "::", "[::]"):
            raise ValueError(f"TP rank {rank} advertised an unspecified host")
        if not 1 <= address.port <= 65535:
            raise ValueError(f"TP rank {rank} advertised an invalid port")
    return metadata


def build_addresses(
    engine_id: str, registrations: Iterable[tuple[int, str, int]]
) -> TPAddresses:
    ranks: dict[int, RankAddress] = {}
    for rank, host, port in registrations:
        if rank in ranks:
            raise ValueError(f"Duplicate FlagCX TP rank {rank}")
        ranks[rank] = RankAddress(host, port)
    return validate_addresses(TPAddresses(engine_id, ranks))


class PeerDiscovery:
    """Resolve and briefly cache a P group's actual rank addresses.

    A legacy producer explicitly responds with ``trans_error`` to the unknown
    query. Only that response allows the caller's legacy host/port arithmetic;
    timeouts and malformed replies are errors, not evidence of a legacy peer.
    Concurrent requests for a single peer share one discovery operation.
    """

    def __init__(
        self,
        context: zmq.asyncio.Context,
        timeout_ms: int = 5000,
        cache_ttl: float = 30.0,
    ):
        self.context = context
        self.timeout_ms = timeout_ms
        self.cache_ttl = cache_ttl
        self._cache: dict[tuple[str, int], tuple[float, TPAddresses | None]] = {}
        self._pending: dict[tuple[str, int], asyncio.Task[TPAddresses | None]] = {}

    async def get(self, host: str, port: int) -> TPAddresses | None:
        key = (host, port)
        cached = self._cache.get(key)
        if cached is not None and time.monotonic() < cached[0]:
            return cached[1]
        task = self._pending.get(key)
        if task is None:
            task = asyncio.create_task(self._query_and_cache(key))
            self._pending[key] = task
            # A cancelled request must not cancel discovery for other callers.
            # Retrieve exceptions even if all callers have been cancelled.
            task.add_done_callback(self._consume_exception)
        return await asyncio.shield(task)

    @staticmethod
    def _consume_exception(task: asyncio.Task) -> None:
        if not task.cancelled():
            task.exception()

    async def _query_and_cache(self, key: tuple[str, int]) -> TPAddresses | None:
        try:
            metadata = await self._query(*key)
            if len(self._cache) >= 128 and key not in self._cache:
                self._cache.pop(next(iter(self._cache)))
            self._cache[key] = (time.monotonic() + self.cache_ttl, metadata)
            return metadata
        finally:
            del self._pending[key]

    async def _query(self, host: str, port: int) -> TPAddresses | None:
        address = tcp_address(host, port)
        sock = self.context.socket(zmq.REQ)
        sock.setsockopt(zmq.LINGER, 0)
        sock.setsockopt(zmq.SNDTIMEO, self.timeout_ms)
        sock.setsockopt(zmq.RCVTIMEO, self.timeout_ms)
        try:
            if ":" in host:
                sock.setsockopt(zmq.IPV6, 1)
            sock.connect(address)
            await sock.send(GET_TP_ADDRESSES)
            reply = await sock.recv()
            if reply == LEGACY_TRANSFER_ERROR:
                return None
            return validate_addresses(msgspec.msgpack.decode(reply, type=TPAddresses))
        except (zmq.ZMQError, msgspec.DecodeError, ValueError) as exc:
            raise RuntimeError(
                f"FlagCX TP address discovery failed at {address}"
            ) from exc
        finally:
            sock.close()
