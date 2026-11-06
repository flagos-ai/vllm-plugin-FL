# SPDX-License-Identifier: Apache-2.0
"""Startup-only model contracts. No source patching or per-token observer."""

import functools
import hashlib
import json
import os
from pathlib import Path
import torch

_expected = None
_audit_dir = None


def configure(expected_tables, audit_dir):
    global _expected, _audit_dir
    _expected = tuple(expected_tables)
    _audit_dir = Path(audit_dir)


def validate_hash_tables_after_load(fn):
    @functools.wraps(fn)
    def load_and_validate(self, *args, **kwargs):
        loaded = fn(self, *args, **kwargs)
        if _expected is None:
            return loaded
        from vllm_fl.ops.dsv4_embedding import DSV4VocabParallelEmbedding

        assert isinstance(self.model.embed_tokens, DSV4VocabParallelEmbedding), (
            "Embedding adapter was bypassed"
        )
        rows = []
        for spec in _expected:
            ffn = self.model.layers[spec["layer"]].ffn
            table, router = ffn.gate.tid2eid, ffn.experts.router
            assert table is not None and router._hash_indices_table is table, (
                "Hash ownership lost"
            )
            assert (
                router._compute_routing.__func__.__module__
                == "vllm_fl.ops.fused_moe.router"
            ), "Native router bypassed FL"
            data = table.detach().to(device="cpu", dtype=torch.int32).contiguous()
            digest = hashlib.sha256(data.numpy().tobytes()).hexdigest()
            assert list(table.shape) == spec["shape"]
            assert digest == spec["canonical_int32_sha256"], (
                "Checkpoint hash table mismatch"
            )
            rows.append(
                {
                    "layer": spec["layer"],
                    "table_sha256": digest,
                    "alias_preserved": True,
                    "router_type": type(router).__name__,
                }
            )
        _audit_dir.mkdir(parents=True, exist_ok=True)
        (_audit_dir / f"model_{os.getpid()}.json").write_text(
            json.dumps(rows, indent=2)
        )
        return loaded

    return load_and_validate
