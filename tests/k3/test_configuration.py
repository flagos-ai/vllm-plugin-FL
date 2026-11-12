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
            disable_custom_all_reduce=True,
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


def test_common_registration_keeps_non_k3_platform_hooks(monkeypatch):
    from unittest.mock import Mock

    import vllm.platforms

    import vllm_fl as plugin
    from vllm_fl.utils import DeviceInfo

    connector, gdn = Mock(), Mock()
    monkeypatch.setattr(plugin, "_register_flagcx_connector", connector)
    monkeypatch.setattr(plugin, "_register_gdn_packed_decode_patch", gdn)
    monkeypatch.setattr(plugin, "register_quant_linear", Mock())
    monkeypatch.setattr(plugin, "register_router", Mock())
    monkeypatch.setattr(DeviceInfo, "vendor_name", property(lambda _: "metax"))
    for vendor, profile, expected in (
        ("metax", True, 0),
        ("metax", False, 1),
        ("nvidia", True, 1),
    ):
        monkeypatch.setattr(
            vllm.platforms,
            "current_platform",
            NS(device_type="cuda", vendor_name=vendor),
        )
        monkeypatch.setattr(
            plugin, "is_kimi_k3_profile", lambda profile=profile: profile
        )
        connector.reset_mock()
        gdn.reset_mock()
        plugin.register_model()
        assert connector.call_count == gdn.call_count == expected


def test_model_registration_requires_both_vendor_and_profile(monkeypatch):
    from unittest.mock import Mock

    from vllm_fl import model_registration as entry

    target = Mock()
    common = Mock()
    monkeypatch.setattr(entry, "version", lambda _: "0.28.0+empty")
    monkeypatch.setattr(entry, "resolve_obj_by_qualname", lambda _: target)
    monkeypatch.setattr(entry, "register_common_model_extensions", common)
    for vendor, profile, expected in (
        ("metax", True, 1),
        ("metax", False, 0),
        ("nvidia", True, 0),
    ):
        monkeypatch.setattr(entry, "current_platform", NS(vendor_name=vendor))
        monkeypatch.setattr(
            entry, "is_kimi_k3_profile", lambda profile=profile: profile
        )
        target.reset_mock()
        common.reset_mock()
        entry.register_model()
        assert target.call_count == expected
        common.assert_called_once_with()


def test_operator_records_are_disabled_only_for_k3(monkeypatch):
    from unittest.mock import Mock

    import flag_gems

    from vllm_fl.ops import custom_ops
    from vllm_fl.worker import worker

    monkeypatch.setattr(worker, "patch_mm_encoder_attention", Mock())
    monkeypatch.setattr(custom_ops, "register_oot_ops", Mock())
    monkeypatch.setattr(worker.fl_envs, "USE_FLAGGEMS", True)
    monkeypatch.setattr(
        worker, "get_flag_gems_whitelist_blacklist", lambda: (["linear"], [])
    )
    enabled = Mock()
    monkeypatch.setattr(flag_gems, "only_enable", enabled)
    for k3, rank, expected in ((True, 0, False), (False, 0, True), (False, 1, False)):
        monkeypatch.setattr(worker, "execution_config", lambda _, k3=k3: NS(enabled=k3))
        worker._initialize_fl_runtime(rank, object())
        assert enabled.call_args.kwargs["record"] is expected
        assert ("registrar" in enabled.call_args.kwargs) is k3


def test_k3_collective_constraint_is_model_scoped():
    cfg = config()
    cfg.parallel_config.disable_custom_all_reduce = False
    with unittest.TestCase().assertRaisesRegex(NotImplementedError, "MCCL"):
        K3ExecutionConfig.from_vllm(cfg, vendor="metax")
    assert not K3ExecutionConfig.from_vllm(cfg, vendor="nvidia").enabled
    cfg.model_config.hf_config.model_type = "other"
    assert not K3ExecutionConfig.from_vllm(cfg, vendor="metax").enabled


class DependencyProvenanceTests(unittest.TestCase):
    def test_dependency_provenance_rejects_wrong_revision_and_import_shadow(self):
        import json
        from pathlib import Path
        from types import SimpleNamespace
        from unittest.mock import patch

        import vllm_fl.kimi_k3_config as config

        lock = json.loads(
            (
                Path(config.__file__).with_name("configs") / "kimi_k3_dependencies.json"
            ).read_text()
        )
        items = {x["distribution"]: x for x in lock["dependencies"].values()}

        class Installed:
            def __init__(self, item, wrong=False):
                self.item, self.wrong = item, wrong

            def read_text(self, name):
                return json.dumps(
                    {
                        "vcs_info": {
                            "vcs": "git",
                            "commit_id": "0" * 40
                            if self.wrong
                            else self.item["commit"],
                        }
                    }
                )

            def locate_file(self, path):
                return Path("/installed") / path

        for wrong, shadow in ((False, False), (True, False), (False, True)):
            with (
                self.subTest(wrong=wrong, shadow=shadow),
                patch.object(
                    config,
                    "distribution",
                    side_effect=lambda name, wrong=wrong: Installed(items[name], wrong),
                ),
                patch.object(
                    config,
                    "find_spec",
                    side_effect=lambda name, shadow=shadow: SimpleNamespace(
                        origin=str(
                            Path("/shadow" if shadow else "/installed")
                            / name
                            / "__init__.py"
                        )
                    ),
                ),
            ):
                if wrong or shadow:
                    with self.assertRaises(RuntimeError):
                        config.verify_dependencies()
                else:
                    config.verify_dependencies()


def test_k3_admission_checks_sources_before_device_imports(monkeypatch):
    from unittest.mock import Mock

    import vllm_fl as plugin

    monkeypatch.setattr(plugin, "is_kimi_k3_profile", lambda: True)
    verify = Mock(side_effect=RuntimeError("wrong source revision"))
    device = Mock()
    monkeypatch.setattr(plugin, "verify_dependencies", verify)
    monkeypatch.setattr(plugin, "_init_vendor_device", device)
    with unittest.TestCase().assertRaisesRegex(RuntimeError, "wrong source"):
        plugin.register()
    device.assert_not_called()


def test_non_k3_admission_does_not_apply_k3_source_pins(monkeypatch):
    from unittest.mock import Mock

    import vllm_fl as plugin

    verify = Mock()
    monkeypatch.setattr(plugin, "is_kimi_k3_profile", lambda: False)
    monkeypatch.setattr(plugin, "verify_dependencies", verify)
    monkeypatch.setattr(plugin, "_init_vendor_device", Mock())
    monkeypatch.setattr(plugin, "_arm_cpu_platform", lambda: "native_cpu")
    assert plugin.register() == "native_cpu"
    verify.assert_not_called()
