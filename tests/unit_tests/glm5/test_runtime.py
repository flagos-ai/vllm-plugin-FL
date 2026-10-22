# SPDX-License-Identifier: Apache-2.0
"""Runtime contracts and regression coverage."""

import os
from types import SimpleNamespace

import pytest

from vllm_fl import activation, flaggems_runtime
from vllm_fl.activation import ActivationConflict, activate, reset_activation_for_tests
from vllm_fl.dispatch.binding import OperatorBinding
from vllm_fl.dispatch.manager import OpManager
from vllm_fl.dispatch.policy import PolicyManager, SelectionPolicy
from vllm_fl.dispatch.types import BackendImplKind, OpImpl
from vllm_fl.models.glm5 import provider, runtime as glm_patch, worker as worker_patch
from vllm_fl.utils import use_flaggems_op


@pytest.fixture(autouse=False)
def _reset():
    reset_activation_for_tests()
    provider._has_nvidia_reference_kernels.cache_clear()
    yield
    reset_activation_for_tests()
    provider._has_nvidia_reference_kernels.cache_clear()


@pytest.mark.usefixtures("_reset")
def test_registration_does_not_mutate_env_or_shared_classes(monkeypatch):
    monkeypatch.setattr(provider, "_has_vllm_native_extension", lambda: False)
    monkeypatch.setenv("VLLM_FL_FLAGOS_WHITELIST", "grouped_topk,moe_sum")
    monkeypatch.setenv(
        "VLLM_FL_PER_OP", "moe_align_block_size=vendor.cuda;fused_moe=reference"
    )
    provider._has_nvidia_reference_kernels.cache_clear()

    from vllm.v1.attention.backends.mla.indexer import DeepseekV32IndexerBackend

    dispatch_keys = ("VLLM_FL_FLAGOS_WHITELIST", "VLLM_FL_PER_OP")
    env_before = {k: os.environ.get(k) for k in dispatch_keys}
    indexer_before = DeepseekV32IndexerBackend.indexes_kv_by_block_stride.__func__(
        DeepseekV32IndexerBackend
    )
    from vllm.model_executor.layers.mhc import MHCPreOp

    mhc_before = MHCPreOp.forward_oot

    assert glm_patch.register_glm5_next_support() is True

    assert {k: os.environ.get(k) for k in dispatch_keys} == env_before
    assert (
        DeepseekV32IndexerBackend.indexes_kv_by_block_stride.__func__(
            DeepseekV32IndexerBackend
        )
        is indexer_before
    )
    assert MHCPreOp.forward_oot is mhc_before


@pytest.mark.usefixtures("_reset")
def test_is_glm5_model_matcher():
    glm_text = SimpleNamespace(
        model_config=SimpleNamespace(
            hf_config=SimpleNamespace(model_type="glm5_next_text"),
            hf_text_config=SimpleNamespace(model_type="glm5_next_text"),
        )
    )
    glm_arch = SimpleNamespace(
        model_config=SimpleNamespace(
            hf_config=SimpleNamespace(architectures=["Glm5NextForCausalLM"]),
            hf_text_config=None,
            architectures=None,
        )
    )
    other = SimpleNamespace(
        model_config=SimpleNamespace(
            hf_config=SimpleNamespace(
                model_type="qwen3_8_flash_next", architectures=["Qwen3Model"]
            ),
            hf_text_config=SimpleNamespace(model_type="qwen3_8_flash_next"),
            architectures=["Qwen3Model"],
        )
    )
    assert glm_patch._is_glm5_model(glm_text)
    assert glm_patch._is_glm5_model(glm_arch)
    assert not glm_patch._is_glm5_model(other)
    assert not glm_patch._is_glm5_model(SimpleNamespace())


@pytest.mark.usefixtures("_reset")
def test_plan_provider_returns_none_for_non_glm():
    other = SimpleNamespace(
        model_config=SimpleNamespace(
            hf_config=SimpleNamespace(
                model_type="qwen3_8_flash_next", architectures=["Qwen3Model"]
            ),
            hf_text_config=SimpleNamespace(model_type="qwen3_8_flash_next"),
        )
    )
    assert glm_patch._glm5_plan_provider(other) is None


@pytest.mark.usefixtures("_reset")
def test_platform_ignores_glm_provider_without_active_plan(monkeypatch):
    """A GLM capability check must preserve a non-GLM MLA model's path.

    The generic dispatch result must be reached verbatim when no plan is active.
    """
    monkeypatch.setattr(provider, "_has_vllm_native_extension", lambda: False)
    provider._has_nvidia_reference_kernels.cache_clear()
    reset_activation_for_tests()

    import vllm_fl.dispatch as dispatch

    monkeypatch.setattr(
        dispatch, "call_op", lambda op, **kwargs: "GENERIC_DISPATCH_SENTINEL"
    )
    from vllm_fl.platform import PlatformFL

    selector = SimpleNamespace(use_mla=True, use_sparse=False)
    assert (
        PlatformFL.get_attn_backend_cls(None, selector) == "GENERIC_DISPATCH_SENTINEL"
    )


@pytest.mark.usefixtures("_reset")
def test_portable_moe_defaults_expose_required_impls(monkeypatch):
    monkeypatch.setattr(provider, "_has_vllm_native_extension", lambda: False)
    provider._has_nvidia_reference_kernels.cache_clear()

    defaults = glm_patch.glm5_portable_moe_defaults()
    assert "moe_align_block_size" in defaults.whitelist_ops
    assert "invoke_fused_moe_triton_kernel" in defaults.whitelist_ops
    assert defaults.required_for("moe_align_block_size") == ("flagos", "reference")


def _glm_config():
    return SimpleNamespace(
        model_config=SimpleNamespace(
            hf_config=SimpleNamespace(
                model_type="glm5_next_text", architectures=["Glm5NextForCausalLM"]
            ),
            hf_text_config=SimpleNamespace(model_type="glm5_next_text"),
            architectures=["Glm5NextForCausalLM"],
        )
    )


def _protect_patched_state(monkeypatch):
    """Restore any class/module attributes the real activation patches."""
    # Follow the real lifecycle even when xdist runs this test before every
    # registration test in this worker process.
    assert glm_patch.register_glm5_next_support() is True
    from vllm.v1.attention.backends.mla.indexer import DeepseekV32IndexerBackend

    monkeypatch.setattr(
        DeepseekV32IndexerBackend,
        "indexes_kv_by_block_stride",
        DeepseekV32IndexerBackend.indexes_kv_by_block_stride,
    )
    from vllm.model_executor.layers.mhc import MHCFusedPostPreOp, MHCPostOp, MHCPreOp

    for mhc_cls in (MHCPreOp, MHCPostOp, MHCFusedPostPreOp):
        monkeypatch.setattr(mhc_cls, "forward_oot", mhc_cls.forward_oot)
    from vllm.model_executor.layers.activation import SiluAndMulWithClamp

    monkeypatch.setattr(
        SiluAndMulWithClamp, "forward_oot", SiluAndMulWithClamp.forward_oot
    )
    from vllm import _custom_ops

    for name in ("concat_mla_q", "concat_and_cache_mla"):
        if hasattr(_custom_ops, name):
            monkeypatch.setattr(_custom_ops, name, getattr(_custom_ops, name))
    from vllm.v1.attention.backends.mla import indexer as indexer_backend
    from vllm.v1.worker import utils as worker_utils

    monkeypatch.setattr(
        worker_utils.AttentionGroup,
        "create_metadata_builders",
        worker_utils.AttentionGroup.create_metadata_builders,
    )
    monkeypatch.setattr(
        indexer_backend.DeepseekV32IndexerMetadataBuilder,
        "build",
        indexer_backend.DeepseekV32IndexerMetadataBuilder.build,
    )
    monkeypatch.setattr(
        worker_utils.KVBlockZeroer, "__init__", worker_utils.KVBlockZeroer.__init__
    )


@pytest.mark.usefixtures("_reset")
def test_real_glm_activation_applies_before_model_construction(monkeypatch):
    _protect_patched_state(monkeypatch)
    provider._has_nvidia_reference_kernels.cache_clear()

    from vllm.v1.attention.backends.mla.indexer import DeepseekV32IndexerBackend

    plan = glm_patch._glm5_plan_provider(_glm_config())
    assert plan is not None

    assert activate(plan) is True
    from vllm_fl.models.glm5.kpool import Glm5NextIndexerAttentionBackend

    assert Glm5NextIndexerAttentionBackend.indexes_kv_by_block_stride() is True
    assert DeepseekV32IndexerBackend.indexes_kv_by_block_stride() is False
    assert activate(plan) is False

    reset_activation_for_tests()


@pytest.mark.usefixtures("_reset")
def test_registration_leaves_worker_kpool_paths_pristine():
    """Item 1: the runner-side kpool hooks are plan-bound, not import-bound."""
    assert glm_patch.register_glm5_next_support() is True

    from vllm_fl.models.glm5 import cache as kpool

    patches = kpool.glm5_next_kpool_runtime_patches("test@1")
    assert {patch.attr for patch in patches} == {
        "create_metadata_builders",
        "build",
        "__init__",
    }
    for patch in patches:
        assert patch.current() is patch.pristine


@pytest.mark.usefixtures("_reset")
def test_real_glm_activation_binds_kpool_runtime_patches(monkeypatch):
    _protect_patched_state(monkeypatch)
    provider._has_nvidia_reference_kernels.cache_clear()

    from vllm_fl.models.glm5 import cache as kpool

    plan = glm_patch._glm5_plan_provider(_glm_config())
    assert plan is not None
    assert activate(plan) is True
    for patch in kpool.glm5_next_kpool_runtime_patches(plan.fingerprint):
        assert patch.current() is not patch.pristine
    reset_activation_for_tests()


@pytest.mark.usefixtures("_reset")
def test_foreign_prepatch_conflicts_without_partial_side_effect(monkeypatch):
    _protect_patched_state(monkeypatch)
    provider._has_nvidia_reference_kernels.cache_clear()

    from vllm.model_executor.layers.mhc import MHCPreOp
    from vllm.v1.attention.backends.mla.indexer import DeepseekV32IndexerBackend

    # Simulate another plugin having patched mHC before GLM activation.
    monkeypatch.setattr(MHCPreOp, "forward_oot", lambda self, *a, **k: None)

    plan = glm_patch._glm5_plan_provider(_glm_config())
    indexer_before = DeepseekV32IndexerBackend.indexes_kv_by_block_stride.__func__(
        DeepseekV32IndexerBackend
    )
    with pytest.raises(ActivationConflict):
        activate(plan)

    # Preflight aborted before any target was written.
    assert (
        DeepseekV32IndexerBackend.indexes_kv_by_block_stride.__func__(
            DeepseekV32IndexerBackend
        )
        is indexer_before
    )


@pytest.fixture(autouse=False)
def _reset_policy(monkeypatch):
    activation.reset_activation_for_tests()
    monkeypatch.setattr(PolicyManager, "_instance", PolicyManager())
    provider._has_nvidia_reference_kernels.cache_clear()
    yield
    activation.reset_activation_for_tests()
    provider._has_nvidia_reference_kernels.cache_clear()


def _other_config():
    return SimpleNamespace(
        model_config=SimpleNamespace(
            hf_config=SimpleNamespace(
                model_type="llama", architectures=["LlamaForCausalLM"]
            ),
            hf_text_config=SimpleNamespace(model_type="llama"),
            architectures=["LlamaForCausalLM"],
        )
    )


def _portable_registered_plan(monkeypatch):
    monkeypatch.setattr(glm_patch, "use_nvidia_reference", lambda: False)
    assert glm_patch.register_glm5_next_support() is True
    plan = activation.resolve_model_plan(_glm_config())
    assert plan is not None
    # Actual registration/defaults, with only the class-patching callback held
    # at its boundary. Real patch ownership is covered by activation tests.
    monkeypatch.setattr(glm_patch, "_apply_glm5_activation", lambda: None)
    return activation.resolve_model_plan(_glm_config())


@pytest.mark.usefixtures("_reset_policy")
def test_glm_activation_adds_only_required_moe_defaults(monkeypatch):
    _portable_registered_plan(monkeypatch)
    manager = PolicyManager.get_instance()
    user = SelectionPolicy.from_dict(prefer="vendor", deny_vendors={"cuda"})
    manager.set_global_policy(user)
    assert activation.activate_for_model(_glm_config()) is not None
    policy = manager.get_policy()
    for op in ("moe_align_block_size", "invoke_fused_moe_triton_kernel"):
        assert policy.get_per_op_order(op) == ["flagos", "reference"]
    assert policy.prefer == "vendor"
    assert policy.deny_vendors == user.deny_vendors
    # Indexer defaults are read through common dispatch, rather than frozen
    # into an additional explicit per-op policy at startup.
    assert policy.get_per_op_order("fp8_fp4_mqa_logits") is None
    assert policy.get_default_order()[0] == "vendor"


@pytest.mark.usefixtures("_reset_policy")
def test_active_glm_preserves_explicit_policy_and_later_preference(monkeypatch):
    _portable_registered_plan(monkeypatch)
    manager = PolicyManager.get_instance()
    manager.set_global_policy(
        SelectionPolicy.from_dict(
            prefer="vendor", per_op_order={"moe_align_block_size": ["flagos"]}
        )
    )
    activation.activate_for_model(_glm_config())
    assert manager.get_policy().get_per_op_order("moe_align_block_size") == ["flagos"]
    manager.set_global_policy(
        SelectionPolicy.from_dict(
            prefer="reference", per_op_order={"moe_align_block_size": ["flagos"]}
        )
    )
    policy = manager.get_policy()
    assert policy.get_per_op_order("moe_align_block_size") == ["flagos"]
    assert policy.get_per_op_order("fp8_fp4_mqa_logits") is None
    assert policy.get_default_order()[0] == "reference"


@pytest.mark.usefixtures("_reset_policy")
def test_glm_provider_leaves_other_model_policy_unchanged(monkeypatch):
    _portable_registered_plan(monkeypatch)
    manager = PolicyManager.get_instance()
    user = SelectionPolicy.from_dict(prefer="reference")
    manager.set_global_policy(user)
    assert activation.activate_for_model(_other_config()) is None
    assert manager.get_policy() is user
    assert manager.get_policy().get_per_op_order("moe_align_block_size") is None


@pytest.mark.usefixtures("_reset_policy")
@pytest.mark.parametrize("sparse", [False, True])
def test_attention_override_tracks_public_policy_and_native_capability(
    monkeypatch, sparse
):
    from vllm_fl.dispatch.backends.flaggems.flaggems import FlagGemsBackend
    from vllm_fl.dispatch.policy import policy_context

    available_checks = []
    native_capability = [True]
    monkeypatch.setattr(glm_patch, "use_nvidia_reference", lambda: native_capability[0])
    monkeypatch.setattr(
        FlagGemsBackend, "is_available", lambda self: available_checks.append(1) or True
    )
    op = "flash_mla_sparse_fwd" if sparse else "attention"
    vendor = SelectionPolicy.from_dict(prefer="vendor")
    flagos = SelectionPolicy.from_dict(prefer="vendor", per_op_order={op: ["flagos"]})
    denied = SelectionPolicy.from_dict(prefer="vendor", deny_vendors={"cuda"})
    expected = "FlagGemsSparseMLABackend" if sparse else "MLAFLBackend"
    # Begin with a valid native selection, then change only one public condition.
    for policy, native, uses_library in (
        (vendor, True, False),
        (flagos, True, True),
        (vendor, True, False),
        (denied, True, True),
        (vendor, False, True),
    ):
        native_capability[0] = native
        before = len(available_checks)
        with policy_context(policy):
            result = glm_patch._glm5_attention_override(True, sparse)
        if uses_library:
            assert result.endswith(expected)
            assert len(available_checks) == before + 1
        else:
            assert result is None
            assert len(available_checks) == before
    # Non-MLA must bypass model-specific capability/dependency work entirely.
    before = len(available_checks)
    with policy_context(flagos):
        assert glm_patch._glm5_attention_override(False, sparse) is None
    assert len(available_checks) == before


_REQUIRED_OPS = ("moe_align_block_size", "invoke_fused_moe_triton_kernel")


def _config(glm=True, arch="Glm5NextForCausalLM"):
    model_type = "glm5_next_text" if glm else "llama"
    architecture = arch if glm else "LlamaForCausalLM"
    return SimpleNamespace(
        model_config=SimpleNamespace(
            hf_config=SimpleNamespace(
                model_type=model_type, architectures=[architecture]
            ),
            hf_text_config=SimpleNamespace(model_type=model_type),
            architectures=[architecture],
            quantization=None,
        ),
        parallel_config=SimpleNamespace(pipeline_parallel_size=1, enable_eplb=False),
        speculative_config=None,
        quant_config=None,
    )


@pytest.fixture
def startup(monkeypatch, tmp_path):
    import flag_gems

    activation.reset_activation_for_tests()
    monkeypatch.setattr(flaggems_runtime, "_STATE", None)
    monkeypatch.setattr(flaggems_runtime, "_FAILED", False)
    monkeypatch.setattr(worker_patch.fl_envs, "USE_FLAGGEMS", True)
    oplist_path = str(tmp_path / "oplist.txt")
    monkeypatch.setattr(
        worker_patch.fl_envs, "FLAGGEMS_ENABLE_OPLIST_PATH", oplist_path
    )
    monkeypatch.setenv("VLLM_FL_PREFER_ENABLED", "1")
    monkeypatch.setenv("VLLM_FL_PREFER", "flagos")
    monkeypatch.setenv("USE_FLAGGEMS", "1")
    for name in ("VLLM_FL_FLAGOS_WHITELIST", "VLLM_FL_FLAGOS_BLACKLIST_APPEND"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("VLLM_FL_FLAGOS_BLACKLIST", "mm,linear")
    monkeypatch.setattr(glm_patch, "use_nvidia_reference", lambda: False)

    events = []
    enable_calls = []
    numeric_calls = []
    applied = []
    available = {op: True for op in _REQUIRED_OPS}
    manager = OpManager(register_builtins=False)
    policy_manager = PolicyManager()
    monkeypatch.setattr(PolicyManager, "_instance", policy_manager)
    policy_manager.set_global_policy(SelectionPolicy.from_dict(prefer="vendor"))
    monkeypatch.setattr(worker_patch, "get_default_manager", lambda: manager)
    monkeypatch.setattr(
        worker_patch, "register_oot_ops", lambda: events.append("register")
    )

    def apply():
        events.append("activate")
        applied.append(True)

    monkeypatch.setattr(glm_patch, "_apply_glm5_activation", apply)
    activation.register_plan_provider(glm_patch._glm5_plan_provider)

    def implementation(name, kind, vendor=None):
        def run(*args, **kwargs):
            numeric_calls.append((name, kind))
            return kind.value

        if kind == BackendImplKind.DEFAULT:

            def is_available():
                events.append(f"check:{name}")
                return available[name] and use_flaggems_op(name)

            run._is_available = is_available
        return OpImpl(name, f"{name}.{kind.value}", kind, run, vendor=vendor)

    manager.registry.register_many(
        [
            implementation(op, kind, vendor)
            for op in _REQUIRED_OPS
            for kind, vendor in (
                (BackendImplKind.DEFAULT, None),
                (BackendImplKind.VENDOR, "cuda"),
            )
        ]
    )

    def enable(kind, **kwargs):
        events.append("enable")
        enable_calls.append((kind, kwargs))

    monkeypatch.setattr(flag_gems, "enable", lambda **kwargs: enable("all", **kwargs))
    monkeypatch.setattr(
        flag_gems, "only_enable", lambda **kwargs: enable("only", **kwargs)
    )
    yield SimpleNamespace(
        manager=manager,
        policy_manager=policy_manager,
        events=events,
        enable_calls=enable_calls,
        numeric_calls=numeric_calls,
        applied=applied,
        available=available,
        flag_gems=flag_gems,
        oplist_path=oplist_path,
    )
    activation.reset_activation_for_tests()


def test_whitelist_conflict_precedes_registration_and_activation(startup, monkeypatch):
    monkeypatch.delenv("VLLM_FL_FLAGOS_BLACKLIST")
    monkeypatch.setenv("VLLM_FL_FLAGOS_WHITELIST", "grouped_topk")
    with pytest.raises(activation.ActivationConflict, match="excludes operators"):
        worker_patch.initialize_glm5_worker(_config(), rank=0)
    assert startup.events == []
    assert startup.applied == []
    assert startup.enable_calls == []
    assert activation.get_active_plan() is None


def test_invalid_model_mode_precedes_registration_and_activation(startup):
    config = _config()
    config.parallel_config.pipeline_parallel_size = 2
    with pytest.raises(glm_patch.Glm5ConfigError, match="pipeline_parallel_size"):
        worker_patch.initialize_glm5_worker(config, rank=0)
    assert startup.events == []
    assert startup.applied == []
    assert startup.enable_calls == []
    assert activation.get_active_plan() is None


@pytest.mark.parametrize("failure", ["unavailable", "blacklist", "per_op"])
def test_actual_resolver_rejects_policy_before_patches_or_enable(
    startup, monkeypatch, failure
):
    if failure == "unavailable":
        startup.available[_REQUIRED_OPS[0]] = False
    elif failure == "blacklist":
        monkeypatch.setenv("VLLM_FL_FLAGOS_BLACKLIST", _REQUIRED_OPS[0])
    else:
        startup.policy_manager.set_global_policy(
            SelectionPolicy.from_dict(per_op_order={_REQUIRED_OPS[0]: ["vendor:cuda"]})
        )
    with pytest.raises(activation.ActivationConflict):
        worker_patch.initialize_glm5_worker(_config(), rank=0)
    assert startup.events[0] == "register"
    assert startup.applied == []
    assert startup.enable_calls == []
    assert startup.numeric_calls == []
    assert activation.get_active_plan() is None


@pytest.mark.parametrize(
    "arch", ["Glm5NextForCausalLM", "Glm5NextForConditionalGeneration"]
)
@pytest.mark.parametrize("rank", [0, 1])
def test_text_and_vision_startup_preserve_phase_order_and_native_aten_blacklist(
    startup, arch, rank
):
    assert worker_patch.initialize_glm5_worker(_config(arch=arch), rank=rank) is True
    assert startup.events[0] == "register"
    assert startup.events[-2:] == ["activate", "enable"]
    for op in _REQUIRED_OPS:
        assert startup.events.index(f"check:{op}") < startup.events.index("activate")
    assert startup.numeric_calls == []
    assert startup.enable_calls == [
        (
            "all",
            dict(
                unused=["mm", "linear"],
                record=rank == 0,
                once=True,
                path=startup.oplist_path,
            ),
        )
    ]
    assert activation.get_active_plan().name == "glm5_next"
    policy = startup.policy_manager.get_policy()
    assert policy.get_per_op_order(_REQUIRED_OPS[0]) == ["flagos", "reference"]
    assert policy.get_per_op_order("fp8_fp4_mqa_logits") is None


def test_explicit_flaggems_whitelist_is_forwarded_without_extension(
    startup, monkeypatch
):
    whitelist = [*_REQUIRED_OPS, "grouped_topk"]
    monkeypatch.delenv("VLLM_FL_FLAGOS_BLACKLIST")
    monkeypatch.setenv("VLLM_FL_FLAGOS_WHITELIST", ",".join(whitelist))
    assert worker_patch.initialize_glm5_worker(_config(), rank=0) is True
    assert startup.enable_calls[0] == (
        "only",
        dict(include=whitelist, record=True, once=True, path=startup.oplist_path),
    )


def test_disabled_generic_aten_does_not_enable_flaggems(startup, monkeypatch):
    monkeypatch.setattr(worker_patch.fl_envs, "USE_FLAGGEMS", False)
    assert worker_patch.initialize_glm5_worker(_config(), rank=0) is True
    assert startup.applied == [True]
    assert startup.enable_calls == []
    assert startup.numeric_calls == []


def test_repeated_initialization_preserves_process_owned_patches_and_aten(startup):
    assert worker_patch.initialize_glm5_worker(_config(), rank=0) is True
    assert worker_patch.initialize_glm5_worker(_config(), rank=0) is True
    assert startup.applied == [True]
    assert len(startup.enable_calls) == 1


@pytest.mark.parametrize("glm_first", [False, True])
def test_empty_plan_rejects_cross_model_process_reuse(startup, glm_first):
    first = _config(glm=glm_first)
    assert worker_patch.initialize_glm5_worker(first, rank=0) is glm_first
    assert startup.applied == ([True] if glm_first else [])
    enable_count = len(startup.enable_calls)
    with pytest.raises(activation.ActivationConflict):
        worker_patch.initialize_glm5_worker(_config(glm=not glm_first), rank=0)
    assert startup.applied == ([True] if glm_first else [])
    assert len(startup.enable_calls) == enable_count


def test_ordinary_model_skips_glm_validation_registry_and_flaggems(startup):
    config = _config(glm=False)
    config.parallel_config.pipeline_parallel_size = 2
    config.parallel_config.enable_eplb = True
    config.speculative_config = object()
    config.quant_config = object()
    assert worker_patch.initialize_glm5_worker(config, rank=0) is False
    assert startup.events == []
    assert startup.enable_calls == []
    assert activation.get_active_plan().name == "<no-plan>"


def test_startup_does_not_freeze_indexer_default_selection(startup):
    name = "fp8_fp4_mqa_logits"
    startup.manager.registry.register_many(
        [
            OpImpl(name, "flag", BackendImplKind.DEFAULT, lambda: "flag"),
            OpImpl(name, "cuda", BackendImplKind.VENDOR, lambda: "cuda", vendor="cuda"),
        ]
    )
    assert worker_patch.initialize_glm5_worker(_config(), rank=0) is True
    binding = OperatorBinding(startup.manager, name)
    assert binding() == "cuda"
    startup.policy_manager.set_global_policy(SelectionPolicy.from_dict(prefer="flagos"))
    assert binding() == "flag"
    assert startup.policy_manager.get_policy().get_per_op_order(name) is None


def test_flaggems_enable_failure_is_not_retried_in_the_process(startup, monkeypatch):
    attempts = []

    def fail(**kwargs):
        attempts.append(kwargs)
        raise RuntimeError("ATen registration partially failed")

    monkeypatch.setattr(startup.flag_gems, "enable", fail)
    with pytest.raises(RuntimeError, match="partially failed"):
        worker_patch.initialize_glm5_worker(_config(), rank=0)
    with pytest.raises(RuntimeError, match="restart the process"):
        worker_patch.initialize_glm5_worker(_config(), rank=0)
    assert len(attempts) == 1
    assert startup.applied == [True]
    assert startup.numeric_calls == []


@pytest.fixture(autouse=False)
def clear_provider_cache():
    provider._has_nvidia_reference_kernels.cache_clear()
    yield
    provider._has_nvidia_reference_kernels.cache_clear()


@pytest.mark.usefixtures("clear_provider_cache")
@pytest.mark.parametrize(
    "cuda,capability,abi,deep,expected",
    [
        (True, 90, True, True, True),
        (True, 80, True, True, False),
        (True, 89, True, True, False),
        (False, 90, True, True, False),
        (True, 90, False, True, False),
        (True, 90, True, False, False),
    ],
)
def test_native_requires_all_capabilities(
    monkeypatch, cuda, capability, abi, deep, expected
):
    monkeypatch.setattr(
        provider.current_platform,
        "has_device_capability",
        lambda minimum: capability >= minimum,
    )
    monkeypatch.setattr(provider.current_platform, "is_cuda", lambda: cuda)
    monkeypatch.setattr(provider, "_has_vllm_native_extension", lambda: abi)
    monkeypatch.setattr(provider, "_has_deep_gemm", lambda: deep)
    assert provider.use_nvidia_reference() is expected


def test_kpool_custom_op_is_a_piecewise_split() -> None:
    from vllm.config.compilation import CompilationConfig

    from vllm_fl.activation import patch_inventory
    from vllm_fl.models.glm5.runtime import register_glm5_next_support

    register_glm5_next_support()
    assert "vllm::sparse_attn_indexer_kpool" in CompilationConfig._attention_ops
    assert any(
        p["target"].endswith("CompilationConfig._attention_ops")
        and p["phase"] == "engine/config"
        for p in patch_inventory()
    )


def test_graph_config_disables_glm5_allreduce_fusion(monkeypatch) -> None:
    """Graph mode must not inherit the generic H100 O2 allreduce fusion."""
    from vllm.model_executor.models.config import HybridAttentionMambaModelConfig

    from vllm_fl.models.glm5.runtime import Glm5NextForCausalLMConfig

    # Keep this a focused config test; the common hybrid checks are covered by
    # vLLM and are unrelated to the GLM5 graph safety override.
    monkeypatch.setattr(
        HybridAttentionMambaModelConfig,
        "verify_and_update_config",
        classmethod(lambda cls, _config: None),
    )

    def make_config(*, enforce_eager: bool, fuse_allreduce_rms):
        return SimpleNamespace(
            model_config=SimpleNamespace(
                enforce_eager=enforce_eager,
                hf_text_config=SimpleNamespace(index_kpool_compress=False),
            ),
            compilation_config=SimpleNamespace(
                pass_config=SimpleNamespace(
                    fuse_allreduce_rms=fuse_allreduce_rms,
                )
            ),
            cache_config=SimpleNamespace(cache_dtype="auto", block_size=16),
        )

    graph_config = make_config(enforce_eager=False, fuse_allreduce_rms=True)
    Glm5NextForCausalLMConfig.verify_and_update_config(graph_config)
    assert graph_config.compilation_config.pass_config.fuse_allreduce_rms is False

    eager_config = make_config(enforce_eager=True, fuse_allreduce_rms=True)
    Glm5NextForCausalLMConfig.verify_and_update_config(eager_config)
    assert eager_config.compilation_config.pass_config.fuse_allreduce_rms is True
