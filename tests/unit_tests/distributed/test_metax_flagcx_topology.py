# Copyright (c) 2026 BAAI. All rights reserved.
"""CPU contracts for the real FlagCX probe's physical-host validation."""

import ast
import hashlib
import ipaddress
import os
import socket
import uuid
from pathlib import Path
from unittest.mock import Mock

import pytest


@pytest.fixture
def topology():
    # Execute the probe's real helpers without importing Torch, vLLM or GPU
    # fixtures. These contracts must also run on hosts without an accelerator.
    path = (
        Path(__file__).resolve().parents[3]
        / "tests/functional_tests/distributed/test_metax_flagcx_collectives.py"
    )
    names = {"_expected_nodes", "_host_identity", "_validate_host_topology"}
    nodes = [
        node
        for node in ast.parse(path.read_text(encoding="utf-8")).body
        if isinstance(node, ast.FunctionDef) and node.name in names
    ]
    namespace = {
        "hashlib": hashlib,
        "ipaddress": ipaddress,
        "os": os,
        "socket": socket,
        "uuid": uuid,
        "Path": Path,
    }
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(path), "exec"), namespace)
    return namespace


def _record(rank, *, host="a", local_rank=0, local_size=1, devices=1, world=2):
    return {
        "host_boot_id_sha256": host * 64,
        # Identical container hostnames are valid on different physical hosts.
        "hostname": "vllm-plugin-fl",
        "rank": rank,
        "local_rank": local_rank,
        "world_size": world,
        "local_world_size": local_size,
        "device_count": devices,
        "ips": {},
    }


def test_two_hosts_with_identical_container_names_are_accepted(topology):
    topology["_validate_host_topology"]([_record(0), _record(1, host="b")], 2)


def test_one_host_cannot_masquerade_as_two_by_renaming_containers(topology):
    records = [
        _record(0, local_rank=0, local_size=2, devices=2),
        _record(1, local_rank=1, local_size=2, devices=2),
    ]
    records[1]["hostname"] = "a-different-container"
    with pytest.raises(AssertionError, match="Expected 2 physical hosts, observed 1"):
        topology["_validate_host_topology"](records, 2)
    topology["_validate_host_topology"](records, 1)


def test_four_ranks_can_share_two_physical_hosts(topology):
    records = [
        _record(
            rank,
            host="a" if rank < 2 else "b",
            local_rank=rank % 2,
            local_size=2,
            devices=2,
            world=4,
        )
        for rank in range(4)
    ]
    topology["_validate_host_topology"](records, 2)


@pytest.mark.parametrize(
    "field,value,error",
    [
        ("host_boot_id_sha256", None, "boot ID SHA256"),
        ("host_boot_id_sha256", "not-a-boot-hash", "boot ID SHA256"),
        ("rank", 0, "every global rank exactly once"),
        ("world_size", 4, "WORLD_SIZE"),
        ("local_rank", 1, "distinct, contiguous local ranks"),
        ("local_world_size", 2, "LOCAL_WORLD_SIZE"),
        ("device_count", 0, "enough devices"),
        ("device_count", True, "must be an integer"),
    ],
)
def test_inconsistent_rank_or_host_metadata_is_rejected(topology, field, value, error):
    records = [_record(0), _record(1, host="b")]
    records[1][field] = value
    with pytest.raises(AssertionError, match=error):
        topology["_validate_host_topology"](records, 2)


@pytest.mark.parametrize("local_rank,devices", [(0, 2), (1, 1), (1, 3)])
def test_shared_host_requires_unique_ranks_and_consistent_device_visibility(
    topology, local_rank, devices
):
    records = [
        _record(0, local_rank=0, local_size=2, devices=2),
        _record(1, local_rank=local_rank, local_size=2, devices=devices),
    ]
    with pytest.raises(AssertionError):
        topology["_validate_host_topology"](records, 1)


@pytest.mark.parametrize("value", ["", "two", "0", "-1", "3"])
def test_invalid_expected_node_count_is_an_error(topology, value):
    with pytest.raises(ValueError, match="FLAGCX_EXPECT_NODES"):
        topology["_expected_nodes"](value, 2)


def test_host_guard_is_optional_and_rejects_a_single_process_two_host_claim(topology):
    assert topology["_expected_nodes"](None, 1) is None
    assert topology["_expected_nodes"]("2", 2) == 2
    with pytest.raises(ValueError, match="WORLD_SIZE"):
        topology["_expected_nodes"]("2", 1)


def test_identity_hashes_the_kernel_boot_id_and_only_records_allowlisted_ips(
    topology, monkeypatch
):
    boot_id = "f5c5c651-9cfe-4c4c-a33d-6fa01979deea"
    read_text = Mock(return_value=boot_id + "\n")
    path = Mock(return_value=Mock(read_text=read_text))
    topology["Path"] = path
    monkeypatch.setenv("LOCAL_WORLD_SIZE", "1")
    monkeypatch.setenv("VLLM_HOST_IP", "192.0.2.10")
    monkeypatch.setenv("MASTER_ADDR", "192.0.2.11")
    monkeypatch.setenv("UNRELATED_SECRET", "never-record-this")
    record = topology["_host_identity"](1, 0, 2, 1)
    path.assert_called_once_with("/proc/sys/kernel/random/boot_id")
    read_text.assert_called_once_with(encoding="ascii")
    assert record["host_boot_id_sha256"] == hashlib.sha256(boot_id.encode()).hexdigest()
    assert record["ips"] == {"VLLM_HOST_IP": "192.0.2.10", "MASTER_ADDR": "192.0.2.11"}
    assert set(record) == {
        "host_boot_id_sha256",
        "hostname",
        "rank",
        "local_rank",
        "world_size",
        "local_world_size",
        "device_count",
        "ips",
    }
    assert boot_id not in repr(record)
    assert "never-record-this" not in repr(record)


@pytest.mark.parametrize("boot_id", [OSError("unreadable boot ID"), "", "invalid"])
def test_unreadable_boot_identity_cannot_produce_a_valid_host_claim(
    topology, monkeypatch, boot_id
):
    read_text = Mock()
    if isinstance(boot_id, Exception):
        read_text.side_effect = boot_id
    else:
        read_text.return_value = boot_id
    topology["Path"] = Mock(return_value=Mock(read_text=read_text))
    monkeypatch.setenv("LOCAL_WORLD_SIZE", "1")
    record = topology["_host_identity"](0, 0, 2, 1)
    assert record["host_boot_id_sha256"] is None
    with pytest.raises(AssertionError, match="boot ID SHA256"):
        topology["_validate_host_topology"]([record, _record(1, host="b")], 2)


@pytest.mark.parametrize("value", ["master.example.test", "", "not-an-ip"])
def test_identity_does_not_log_non_ip_environment_values(topology, monkeypatch, value):
    topology["Path"] = Mock(
        return_value=Mock(
            read_text=Mock(return_value="f5c5c651-9cfe-4c4c-a33d-6fa01979deea")
        )
    )
    monkeypatch.setenv("LOCAL_WORLD_SIZE", "1")
    monkeypatch.setenv("VLLM_HOST_IP", value)
    monkeypatch.setenv("MASTER_ADDR", value)
    assert topology["_host_identity"](0, 0, 2, 1)["ips"] == {}
