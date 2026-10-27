"""Numerical CPU check of real four-stream feedback without a GPU model load."""

import ast
import types
import unittest
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "vllm_fl/models/qwen3_8_flash_next/gpu/mtp.py"


def production_forward():
    tree = ast.parse(SOURCE.read_text())
    cls = next(
        n
        for n in tree.body
        if isinstance(n, ast.ClassDef)
        and n.name == "Qwen3_8FlashNextMultiTokenPredictor"
    )
    fn = next(
        n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == "forward"
    )
    namespace = {"torch": torch}
    code = compile(ast.Module(body=[fn], type_ignores=[]), str(SOURCE), "exec")
    exec(code, namespace)
    return namespace["forward"]


class MTPForwardTest(unittest.TestCase):
    def test_full_stream_norm_embedding_injection_and_recurrent_feedback(self):
        forward = production_forward()

        def norm(x):
            variance = x.float().square().mean(-1, keepdim=True)
            return x * torch.rsqrt(variance + 1e-6)

        seen = []
        hidden_matrix = torch.tensor([[1.0, 2.0], [-0.5, 0.25]])
        embed_matrix = torch.tensor([[0.5, -1.0], [2.0, 1.0]])

        def decoder(*, hidden_states, residual, positions):
            self.assertIsNone(residual)
            seen.append(hidden_states.clone())
            return hidden_states + torch.arange(8).view(1, 8) * 0.1, None, None

        model = types.SimpleNamespace(
            hc_count=4,
            hidden_size=2,
            num_mtp_layers=1,
            pre_fc_norm_embedding=norm,
            pre_fc_norm_hidden=norm,
            fc_embedding=lambda x: x @ embed_matrix.T,
            fc_hidden=lambda x: x @ hidden_matrix.T,
            layers=[decoder],
            hyper_connection_mixer=types.SimpleNamespace(
                mix=lambda x: (x.view(-1, 4, 2).mean(1), x)
            ),
        )
        hidden = torch.tensor(
            [
                [1.0, 2.0, 3.0, 5.0, 7.0, 11.0, 13.0, 17.0],
                [2.0, -1.0, 4.0, -3.0, 6.0, -5.0, 8.0, -7.0],
            ]
        )
        embedding = torch.tensor([[2.0, -3.0], [-4.0, 1.0]])
        positions = torch.tensor([4, 7])
        for step in range(3):
            logits_hidden, feedback = forward(
                model,
                None,
                positions + step,
                hidden,
                inputs_embeds=embedding,
                spec_step_idx=step,
            )
            ref_hidden = norm(hidden).view(2, 4, 2) @ hidden_matrix.T
            ref_embedding = norm(embedding) @ embed_matrix.T
            expected_input = (ref_hidden + ref_embedding[:, None, :]).flatten(1)
            expected_feedback = expected_input + torch.arange(8).view(1, 8) * 0.1
            torch.testing.assert_close(seen[-1], expected_input)
            torch.testing.assert_close(feedback, expected_feedback)
            torch.testing.assert_close(
                logits_hidden, expected_feedback.view(2, 4, 2).mean(1)
            )
            self.assertEqual(feedback.shape, (2, 8))
            self.assertEqual(logits_hidden.shape, (2, 2))
            self.assertFalse(torch.equal(feedback[:, :2], feedback[:, 2:4]))
            hidden = feedback
            embedding = embedding.flip(0)
        self.assertEqual(len(seen), 3)
        with self.assertRaisesRegex(ValueError, "real 4H"):
            forward(model, None, positions, hidden[:, :2], inputs_embeds=embedding)


if __name__ == "__main__":
    unittest.main()
