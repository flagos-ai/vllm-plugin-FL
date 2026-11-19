"""Untimed PD audits on identical real layer inputs; no second GDN state update."""

import ast
from pathlib import Path

import torch
from pd_worker import PDWorker


def difference(actual, expected):
    delta = (actual.float() - expected.float()).abs()
    return {
        "equal": torch.equal(actual, expected),
        "shape": list(actual.shape),
        "finite": bool(torch.isfinite(actual).all() and torch.isfinite(expected).all()),
        "max_abs": float(delta.max()) if delta.numel() else 0.0,
        "relative_rms": float(
            delta.square().mean().sqrt()
            / expected.float().square().mean().sqrt().clamp_min(1e-12)
        )
        if delta.numel()
        else 0.0,
    }


class PDAuditWorker(PDWorker):
    def begin_pd_layout_audit(self, reference_source):
        from vllm.model_executor.layers.attention.kv_transfer_utils import (
            maybe_transfer_kv_layer,
        )

        from vllm_fl.cp import fused_gdn, prstyle

        assert not hasattr(self, "_pd_layout_audit"), "Audit already installed"
        self._pd_layout_audit = []
        self._pd_layout_restore = []
        tree = ast.parse((Path(reference_source) / "vllm_fl/cp/prstyle.py").read_text())
        definitions = list(tree.body)
        attention = next(
            n
            for n in tree.body
            if isinstance(n, ast.ClassDef) and n.name == "QwenPRStyleAttention"
        )
        definitions.extend(attention.body)

        def reference(name):
            node = next(
                n
                for n in definitions
                if isinstance(n, ast.FunctionDef) and n.name == name
            )
            node.decorator_list = []
            namespace = dict(vars(prstyle))
            exec(
                compile(
                    ast.fix_missing_locations(ast.Module(body=[node], type_ignores=[])),
                    "pd_reference",
                    "exec",
                ),
                namespace,
            )
            return namespace[name]

        def install(module, name, value):
            self._pd_layout_restore.append((module, name, getattr(module, name)))
            setattr(module, name, value)

        def record(kind, actual, expected, **details):
            row = dict(kind=kind, **difference(actual, expected), **details)
            self._pd_layout_audit.append(row)
            assert row["finite"] and row["relative_rms"] < 0.003, row

        old_moe = reference("_pr_moe")
        new_moe = prstyle._pr_moe

        def moe(mlp, local_input, plan, plans, group):
            actual = new_moe(mlp, local_input, plan, plans, group)
            expected = old_moe(mlp, local_input, plan, plans, group)
            record(
                "moe",
                actual,
                expected,
                tokens=plan.num_tokens,
                uniform=plan.uniform_rows,
                local_rows=len(plan.rows),
            )
            return actual

        install(prstyle, "_pr_moe", moe)
        old_attention = reference("_pr_attention")
        new_attention = prstyle.QwenPRStyleAttention._pr_attention.__wrapped__

        def attention(layer, q, k, v, gate, plan, plans, metadata, layer_name):
            actual = new_attention(
                layer, q, k, v, gate, plan, plans, metadata, layer_name
            )
            expected = old_attention(
                layer, q, k, v, gate, plan, plans, metadata, layer_name
            )
            record(
                "attention",
                actual,
                expected,
                tokens=plan.num_tokens,
                full_prefill=bool(plan.full_prefill_lengths),
            )
            return actual

        install(
            prstyle.QwenPRStyleAttention,
            "_pr_attention",
            maybe_transfer_kv_layer(attention),
        )
        norm = fused_gdn.strided_gated_norm

        def gated_norm(module, x, z, output_rows=None):
            actual = norm(module, x, z, output_rows)
            expected = norm(module, x, z)
            if output_rows is not None:
                expected = expected.index_select(0, output_rows.clamp_min(0))
                expected[output_rows < 0] = 0
            record("gdn_output_norm", actual, expected, tokens=x.shape[0])
            return actual

        install(fused_gdn, "strided_gated_norm", gated_norm)
        return {"installed": True}

    def end_pd_layout_audit(self):
        for module, name, original in reversed(self._pd_layout_restore):
            setattr(module, name, original)
        self._pd_layout_restore.clear()
        return self._pd_layout_audit
