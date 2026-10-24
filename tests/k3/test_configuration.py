"""CPU policy contracts; no model construction or GPU arithmetic."""

import unittest
from types import SimpleNamespace as NS

from vllm_fl.kimi_k3_config import K3ExecutionConfig


def config(model="kimi_k3", tp=8, pp=4, dp=1, ep=False, options=None, family=None):
    return NS(
        model_config=NS(hf_config=NS(model_type=model)),
        quant_config=NS(model_family=family),
        parallel_config=NS(
            tensor_parallel_size=tp,
            pipeline_parallel_size=pp,
            data_parallel_size=dp,
            enable_expert_parallel=ep,
        ),
        additional_config={"k3": options or {}},
    )


class ConfigurationTests(unittest.TestCase):
    def test_verified_topology_selects_tp8_and_feedback(self):
        policy = K3ExecutionConfig.from_vllm(config(), vendor="metax")
        self.assertTrue(
            policy.enabled and policy.tp8_compute and policy.host_pp_feedback
        )

    def test_other_vendor_does_not_activate_k3(self):
        self.assertEqual(
            K3ExecutionConfig.from_vllm(config(), vendor="nvidia"), K3ExecutionConfig()
        )

    def test_unrelated_model_does_not_activate_k3(self):
        self.assertEqual(
            K3ExecutionConfig.from_vllm(config(model="other"), vendor="metax"),
            K3ExecutionConfig(),
        )

    def test_language_submodel_requires_family_marker(self):
        self.assertFalse(
            K3ExecutionConfig.from_vllm(
                config(model="kimi_linear"), vendor="metax"
            ).enabled
        )
        self.assertTrue(
            K3ExecutionConfig.from_vllm(
                config(model="kimi_linear", family="kimi_k3"), vendor="metax"
            ).enabled
        )

    def test_unknown_algorithm_switch_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "Unknown K3 execution"):
            K3ExecutionConfig.from_vllm(
                config(options={"enable_fast": True}), vendor="metax"
            )

    def test_removed_transport_option_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "Unknown K3 execution"):
            K3ExecutionConfig.from_vllm(
                config(options={"expert_transport": "other"}), vendor="metax"
            )

    def test_ep_transport_cannot_silently_change_tp8pp4(self):
        for transport in ("tp8_dp8_allreduce", "feature_striped"):
            with self.subTest(transport=transport), self.assertRaises(ValueError):
                K3ExecutionConfig.from_vllm(
                    config(options={"expert_transport": transport}), vendor="metax"
                )

    def test_removed_topologies_are_rejected(self):
        for topology in (
            dict(tp=32, pp=1),
            dict(tp=8, pp=1, dp=8, ep=True),
            dict(tp=8, pp=4, ep=True),
            dict(tp=8, pp=4, dp=2),
        ):
            with (
                self.subTest(topology=topology),
                self.assertRaisesRegex(ValueError, "TP8/DP1 with EP disabled"),
            ):
                K3ExecutionConfig.from_vllm(config(**topology), vendor="metax")

    def test_pipeline_depth_preserves_tp8_policy(self):
        for pp in (2, 4, 8):
            with self.subTest(pp=pp):
                policy = K3ExecutionConfig.from_vllm(config(pp=pp), vendor="metax")
                self.assertTrue(policy.enabled and policy.tp8_compute)
                self.assertTrue(policy.host_pp_feedback)

    def test_single_pipeline_stage_has_no_feedback(self):
        self.assertFalse(
            K3ExecutionConfig.from_vllm(config(pp=1), vendor="metax").host_pp_feedback
        )


if __name__ == "__main__":
    unittest.main()
