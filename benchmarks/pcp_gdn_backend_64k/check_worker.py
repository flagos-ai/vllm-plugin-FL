"""Paired actual-input backend errors and same-history full-logit diagnostics."""

import torch

from vllm_fl.cp.gdn_backend_probe import GDNBackendProbe
from vllm_fl.cp.inspection import QwenCPInspectionWorker


def compare(a, b):
    a, b = a.float(), b.float()
    delta = a - b
    ref_rms = a.square().mean().sqrt().item()
    rms = delta.square().mean().sqrt().item()
    return {
        "shape": list(a.shape),
        "finite": bool(torch.isfinite(a).all() and torch.isfinite(b).all()),
        "equal": torch.equal(a, b),
        "max_abs": delta.abs().max().item(),
        "rms": rms,
        "reference_rms": ref_rms,
        "relative_rms": rms / ref_rms if ref_rms else None,
        "different_elements": int((a != b).sum().item()),
        "elements": a.numel(),
    }


class CheckWorker(QwenCPInspectionWorker):
    def configure(self, backend, phase):
        if not hasattr(self, "_probe"):
            self._probe = GDNBackendProbe(self.model_runner.model)
            self._original_logits = self.model_runner.model.compute_logits
            self.model_runner.model.compute_logits = self._capture_logits
        identity = self._probe.configure(backend)
        self._phase = phase
        self._core_checks, self._logit_checks = [], []
        self._index = 0
        if phase == "capture":
            self._reference_logits = []
        if phase == "core_audit":
            for name, layer in self._probe.layers:
                self._install_core_audit(name, layer)
        return {"backend": backend, "phase": phase, "layers": identity}

    def _install_core_audit(self, name, layer):
        op = layer.chunk_gated_delta_rule

        def paired(*args, **kwargs):
            assert not args
            initial = kwargs["initial_state"]
            before = initial.clone()
            expected = op.forward_cuda(**kwargs)
            actual = op.forward_native(**kwargs)
            row = {
                "layer": name,
                "initial_unchanged": torch.equal(before, initial),
                "tokens": kwargs["q"].shape[1],
                "chunk_metadata_present": kwargs["chunk_indices"] is not None
                and kwargs["chunk_offsets"] is not None,
                "output": compare(expected[0], actual[0]),
                "state": compare(expected[1], actual[1]),
            }
            self._core_checks.append(row)
            assert (
                row["initial_unchanged"]
                and row["output"]["finite"]
                and row["state"]["finite"]
            ), row
            return expected

        op._forward_method = paired

    def _capture_logits(self, *args, **kwargs):
        logits = self._original_logits(*args, **kwargs)
        if logits is None:
            return logits
        if self._phase == "capture":
            self._reference_logits.append(logits.detach().cpu())
        elif self._phase in {"repeat", "teacher"}:
            expected = self._reference_logits[self._index].to(logits.device)
            row = compare(expected, logits)
            log_p = torch.log_softmax(expected.float(), -1)
            log_q = torch.log_softmax(logits.float(), -1)
            top = expected.float().topk(2, -1).values
            row.update(
                step=self._index,
                argmax_equal=torch.equal(expected.argmax(-1), logits.argmax(-1)),
                baseline_argmax=expected.argmax(-1).cpu().tolist(),
                candidate_argmax=logits.argmax(-1).cpu().tolist(),
                kl_reference_to_candidate=(log_p.exp() * (log_p - log_q))
                .sum(-1)
                .mean()
                .item(),
                baseline_top1_margin=(top[..., 0] - top[..., 1]).cpu().tolist(),
            )
            self._logit_checks.append(row)
            assert row["finite"], row
            # Same baseline logits force exactly the same autoregressive inputs
            # in BOTH the repeat noise control and candidate comparison. No
            # throughput conclusions are drawn from these instrumented runs.
            logits = expected
        self._index += 1
        return logits

    def results(self):
        from vllm.distributed import get_tp_group

        if self._phase in {"repeat", "teacher"}:
            assert self._index == len(self._reference_logits)
        return {
            "rank": get_tp_group().rank_in_group,
            "phase": self._phase,
            "core_checks": self._core_checks,
            "logit_checks": self._logit_checks,
            "logit_calls": self._index,
        }
