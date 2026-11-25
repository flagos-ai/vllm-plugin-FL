# Copyright (c) 2026 BAAI. All rights reserved.

"""Tests for FlagCX Prefill TP address discovery (no GPU or FlagCX needed)."""

import asyncio

import pytest

msgspec = pytest.importorskip("msgspec")
zmq = pytest.importorskip("zmq")
pytest.importorskip("zmq.asyncio")

from vllm_fl.distributed.kv_transfer.flagcx_discovery import (  # noqa: E402
    GET_TP_ADDRESSES,
    LEGACY_TRANSFER_ERROR,
    PeerDiscovery,
    build_addresses,
    tcp_address,
)


def test_build_addresses_across_nodes():
    meta = build_addresses(
        "prefill",
        [(1, "10.0.0.2", 9001), (0, "10.0.0.1", 9000)],
    )
    assert meta.engine_id == "prefill"
    assert meta.ranks[0].host == "10.0.0.1"
    assert meta.ranks[1].port == 9001


@pytest.mark.parametrize(
    "registrations, message",
    [
        ([(0, "h", 1), (0, "h", 2)], "Duplicate"),
        ([(0, "h", 1), (2, "h", 3)], "contiguous"),
        ([(0, "0.0.0.0", 1)], "unspecified"),
        ([(0, "h", 0)], "invalid port"),
        ([], "requires"),
    ],
)
def test_build_addresses_rejects_bad_input(registrations, message):
    with pytest.raises(ValueError, match=message):
        build_addresses("prefill", registrations)


def test_tcp_address_brackets_ipv6():
    assert tcp_address("10.0.0.1", 5) == "tcp://10.0.0.1:5"
    assert tcp_address("fe80::1", 5) == "tcp://[fe80::1]:5"


def _serve(reply, delay=0.0):
    """Start a ROUTER that answers discovery queries; returns (port, task, hits)."""
    ctx = zmq.asyncio.Context.instance()
    sock = ctx.socket(zmq.ROUTER)
    port = sock.bind_to_random_port("tcp://127.0.0.1")
    hits = []

    async def loop():
        try:
            while True:
                identity, _, body = await sock.recv_multipart()
                hits.append(body)
                await asyncio.sleep(delay)
                await sock.send_multipart((identity, b"", reply))
        finally:
            sock.close(linger=0)

    return port, asyncio.ensure_future(loop()), hits


def _run(coro):
    return asyncio.run(coro)


def test_discovery_returns_addresses_and_caches():
    meta = build_addresses("prefill", [(0, "10.0.0.1", 9000), (1, "10.0.0.2", 9001)])

    async def main():
        port, server, hits = _serve(msgspec.msgpack.encode(meta), delay=0.05)
        disc = PeerDiscovery(zmq.asyncio.Context.instance(), timeout_ms=2000)
        try:
            results = await asyncio.gather(
                *(disc.get("127.0.0.1", port) for _ in range(5))
            )
            again = await disc.get("127.0.0.1", port)
        finally:
            server.cancel()
        return results, again, hits

    results, again, hits = _run(main())
    assert all(r == meta for r in results)
    assert again == meta
    # Concurrent callers share one query; the follow-up hits the cache.
    assert hits == [GET_TP_ADDRESSES]


def test_discovery_legacy_peer_returns_none():
    async def main():
        port, server, _ = _serve(LEGACY_TRANSFER_ERROR)
        try:
            return await PeerDiscovery(zmq.asyncio.Context.instance()).get(
                "127.0.0.1", port
            )
        finally:
            server.cancel()

    assert _run(main()) is None


def test_discovery_malformed_reply_raises():
    async def main():
        port, server, _ = _serve(b"not msgpack")
        try:
            await PeerDiscovery(zmq.asyncio.Context.instance()).get("127.0.0.1", port)
        finally:
            server.cancel()

    with pytest.raises(RuntimeError, match="discovery failed"):
        _run(main())


def test_discovery_timeout_raises():
    async def main():
        ctx = zmq.asyncio.Context.instance()
        silent = ctx.socket(zmq.ROUTER)
        port = silent.bind_to_random_port("tcp://127.0.0.1")
        try:
            await PeerDiscovery(ctx, timeout_ms=200).get("127.0.0.1", port)
        finally:
            silent.close(linger=0)

    with pytest.raises(RuntimeError, match="discovery failed"):
        _run(main())
