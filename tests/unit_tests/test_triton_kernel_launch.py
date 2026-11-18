# SPDX-License-Identifier: Apache-2.0
"""Launch metadata remains separate from kernel numerical implementation."""

import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest

_spec = importlib.util.spec_from_file_location(
    "launch_meta",
    Path(__file__).resolve().parents[2] / "vllm_fl/patches/triton_kernel.py",
)
_module = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_module)


class Kernel:
    arg_names = ["output", "BLOCK_SIZE"]

    def __init__(self):
        self.calls = []

    def __getitem__(self, grid):
        def launch(*args, **kwargs):
            self.calls.append((grid, args, kwargs))
            return "launched"

        return launch


@pytest.mark.parametrize("positional", [False, True])
def test_kernel_argument_override(positional):
    kernel = Kernel()
    module = SimpleNamespace(kernel=kernel)
    _module.patch_kernel_launch_meta(module, "kernel", {"BLOCK_SIZE": 1024})
    proxy = module.kernel
    _module.patch_kernel_launch_meta(module, "kernel", {"BLOCK_SIZE": 1024})
    assert module.kernel is proxy
    assert proxy.arg_names == kernel.arg_names
    args, kwargs = (
        (("out", 8192), {}) if positional else (("out",), {"BLOCK_SIZE": 8192})
    )
    assert proxy[(3,)](*args, **kwargs, num_warps=4) == "launched"
    expected = (
        (("out", 1024), {"num_warps": 4})
        if positional
        else (("out",), {"BLOCK_SIZE": 1024, "num_warps": 4})
    )
    assert kernel.calls == [((3,), *expected)]
    with pytest.raises(RuntimeError):
        _module.patch_kernel_launch_meta(module, "kernel", {"BLOCK_SIZE": 512})
    assert module.kernel is proxy


def test_launch_options_preserve_arguments_and_errors():
    kernel = Kernel()
    proxy = _module.KernelLaunchMetaProxy(kernel, {"num_warps": 4})
    proxy[(2,)]("out", 16, num_warps=8)
    assert kernel.calls == [((2,), ("out", 16), {"num_warps": 4})]

    class Broken(Kernel):
        def __getitem__(self, grid):
            def launch(*args, **kwargs):
                self.calls.append(grid)
                raise RuntimeError("launch failed")

            return launch

    broken = Broken()
    with pytest.raises(RuntimeError, match="launch failed"):
        _module.KernelLaunchMetaProxy(broken, {"num_warps": 4})[(1,)]("out", 16)
    assert broken.calls == [(1,)]


@pytest.mark.parametrize("positional", [False, True])
@pytest.mark.parametrize("dtype", ["fp64", "fp32", None])
def test_npu_uniform_adapter_preserves_inputs_and_launch(
    monkeypatch, positional, dtype
):
    import sys

    spec = importlib.util.spec_from_file_location(
        "npu_launch_adapters",
        Path(__file__).resolve().parents[2]
        / "vllm_fl/dispatch/backends/vendor/ascend/patch.py",
    )
    vendor = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(vendor)
    monkeypatch.setitem(
        sys.modules, "torch", SimpleNamespace(float64="fp64", float32="fp32")
    )
    converted = SimpleNamespace(dtype="fp32")
    casts = []

    class Probabilities:
        def __init__(self, dtype):
            self.dtype = dtype

        def to(self, dtype):
            casts.append(dtype)
            return converted

    class ProbabilityKernel(Kernel):
        arg_names = ["output", "uniform_probs_ptr"]

    original = Probabilities(dtype) if dtype else None
    expected = converted if dtype == "fp64" else original
    kernel = ProbabilityKernel()
    adapter = vendor._NpuUniformProbabilityKernel(kernel)
    if positional:
        adapter[(4,)]("out", original, SYNTHETIC_MODE=False)
        arguments = (("out", expected), {"SYNTHETIC_MODE": False})
    else:
        adapter[(4,)]("out", uniform_probs_ptr=original, SYNTHETIC_MODE=False)
        arguments = (("out",), {"uniform_probs_ptr": expected, "SYNTHETIC_MODE": False})
    assert kernel.calls == [((4,), *arguments)]
    assert casts == (["fp32"] if dtype == "fp64" else [])
    assert original is None or original.dtype == dtype


@pytest.fixture
def mtp_transaction(monkeypatch):
    import sys
    from types import ModuleType

    def package(name):
        if name in sys.modules:
            return sys.modules[name]
        module = ModuleType(name)
        module.__path__ = []
        monkeypatch.setitem(sys.modules, name, module)
        if "." in name:
            parent, attr = name.rsplit(".", 1)
            monkeypatch.setattr(package(parent), attr, module, raising=False)
        return module

    paths = {
        "conv": "vllm.model_executor.layers.mamba.ops.causal_conv1d",
        "gdn": "vllm.model_executor.layers.mamba.gdn.qwen_gdn_linear_attn",
        "registry": "vllm.model_executor.custom_op",
        "utils": "vllm.v1.spec_decode.utils",
        "proposer": "vllm.v1.spec_decode.llm_base_proposer",
        "sample": "vllm.v1.sample.rejection_sampler",
        "filter": "vllm.v1.sample.ops.topk_topp_sampler",
    }
    modules = {key: package(path) for key, path in paths.items()}
    package("vllm.v1.attention.backends.utils").NULL_BLOCK_ID = 0
    metadata = Kernel()
    metadata.arg_names = ["sampled", "counts"]
    package("vllm_fl.worker.spec_decode").prepare_next_token_padded_kernel = metadata
    candidate_name = "transaction_candidate.patch"
    package("transaction_candidate")
    impl = package("transaction_candidate.impl.gdn")

    class GDN:
        pass

    impl.AscendQwenGatedDeltaNetAttention = GDN
    modules["registry"].op_registry_oot = {}
    for key in ("utils", "proposer"):
        modules[key].eagle_prepare_next_token_padded_kernel = metadata

    def update(
        x,
        conv_state,
        weight,
        bias=None,
        activation=None,
        conv_state_indices=None,
        num_accepted_tokens=None,
        query_start_loc=None,
        max_query_len=-1,
        null_block_id=0,
        block_idx_last_scheduled_token=None,
        initial_state_idx=None,
        validate_data=False,
    ):
        pass

    for key in ("conv", "gdn"):
        modules[key].causal_conv1d_fn = lambda: None
        modules[key].causal_conv1d_update = update
    for name, names in {
        "sample_recovered_tokens_kernel": [
            "BLOCK_SIZE",
            "NO_DRAFT_PROBS",
            "USE_FP64_GUMBEL",
        ],
        "rejection_greedy_sample_kernel": ["uniform_probs_ptr", "SYNTHETIC_MODE"],
        "rejection_random_sample_kernel": ["uniform_probs_ptr", "SYNTHETIC_MODE"],
    }.items():
        kernel = Kernel()
        kernel.arg_names = names
        setattr(modules["sample"], name, kernel)
    for key in ("sample", "filter"):
        modules[key].apply_top_k_top_p = lambda: None
    spec = importlib.util.spec_from_file_location(
        candidate_name,
        Path(__file__).resolve().parents[2]
        / "vllm_fl/dispatch/backends/vendor/ascend/patch.py",
    )
    vendor = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(vendor)
    return vendor, modules, GDN


@pytest.mark.parametrize(
    "fault", ["missing_alias", "metadata_order", "conv_contract", "conflict"]
)
def test_mtp_prevalidation_rejects_before_writes(mtp_transaction, fault):
    vendor, modules, gdn = mtp_transaction
    if fault == "missing_alias":
        del modules["proposer"].eagle_prepare_next_token_padded_kernel
    elif fault == "metadata_order":
        changed = Kernel()
        changed.arg_names = ["counts", "sampled"]
        modules["proposer"].eagle_prepare_next_token_padded_kernel = changed
    elif fault == "conv_contract":
        modules["conv"].causal_conv1d_update = lambda x: None
    else:
        modules["registry"].op_registry_oot["QwenGatedDeltaNetAttention"] = object()
    snapshots = {key: vars(module).copy() for key, module in modules.items()}
    with pytest.raises((AttributeError, RuntimeError)), vendor._mtp_patch_transaction():
        pytest.fail("Mutation phase reached with incompatible API")
    for key, module in modules.items():
        assert vars(module) == snapshots[key]
    assert "name" not in gdn.__dict__


@pytest.mark.parametrize("existing", [False, True])
@pytest.mark.parametrize("fault", ["setattr", "after_registration"])
def test_mtp_transaction_restores_objects_and_allows_retry(
    monkeypatch, mtp_transaction, existing, fault
):
    vendor, modules, gdn = mtp_transaction
    registry = modules["registry"].op_registry_oot
    if existing:
        registry["QwenGatedDeltaNetAttention"] = gdn
        gdn.name = "prior-own-name"
    # An existing wrapper must be restored by identity, not reconstructed.
    original = modules["sample"].sample_recovered_tokens_kernel
    wrapper = _module.KernelLaunchMetaProxy(original, {"BLOCK_SIZE": 1024})
    modules["sample"].sample_recovered_tokens_kernel = wrapper
    snapshots = {key: vars(module).copy() for key, module in modules.items()}
    import builtins

    real_setattr = builtins.setattr
    calls = []

    def fail_once(obj, name, value):
        calls.append((obj, name))
        if len(calls) == 2:
            raise RuntimeError("injected assignment failure")
        real_setattr(obj, name, value)

    with pytest.raises(RuntimeError, match="injected"), vendor._mtp_patch_transaction():
        if fault == "setattr":
            with monkeypatch.context() as patch:
                patch.setattr(builtins, "setattr", fail_once)
                for key, name in (
                    ("sample", "sample_recovered_tokens_kernel"),
                    ("utils", "eagle_prepare_next_token_padded_kernel"),
                ):
                    setattr(modules[key], name, Kernel())
        else:
            modules["sample"].sample_recovered_tokens_kernel = Kernel()
            modules["utils"].eagle_prepare_next_token_padded_kernel = Kernel()
            registry["QwenGatedDeltaNetAttention"] = gdn
            gdn.name = "QwenGatedDeltaNetAttention"
            raise RuntimeError("injected post-registration failure")
    for key, module in modules.items():
        assert vars(module) == snapshots[key]
    assert modules["sample"].sample_recovered_tokens_kernel is wrapper
    assert registry.get("QwenGatedDeltaNetAttention") is (gdn if existing else None)
    assert gdn.__dict__.get("name") == ("prior-own-name" if existing else None)
    with vendor._mtp_patch_transaction():
        pass


def test_mtp_apply_flags_reset_after_failure_and_retry(monkeypatch, mtp_transaction):
    import sys
    from types import ModuleType

    vendor, modules, _ = mtp_transaction
    names = (
        "patch_rejection_sampling",
        "patch_speculative_metadata",
        "patch_causal_conv1d",
        "patch_fla_ops",
        "patch_op_cls",
        "patch_fused_moe",
        "patch_gdn",
        "patch_vision_position_interpolation",
        "patch_sampler_filter",
    )
    for name in names:
        monkeypatch.setattr(vendor, name, lambda: None)
    mamba = ModuleType("transaction_candidate.patches.patch_mamba_utils")
    mamba.apply_mamba_copy_patch = lambda: None
    monkeypatch.setitem(sys.modules, mamba.__name__, mamba)
    original = modules["sample"].sample_recovered_tokens_kernel

    def fail():
        modules["sample"].sample_recovered_tokens_kernel = Kernel()
        raise RuntimeError("later patch failure")

    monkeypatch.setattr(vendor, "patch_gdn", fail)
    with pytest.raises(RuntimeError, match="later patch failure"):
        vendor.apply_ascend_patches()
    assert not vendor._patch_state.applied
    assert not vendor._patch_state.applying
    assert modules["sample"].sample_recovered_tokens_kernel is original
    monkeypatch.setattr(vendor, "patch_gdn", lambda: None)
    vendor.apply_ascend_patches()
    assert vendor._patch_state.applied
    assert not vendor._patch_state.applying
    vendor.apply_ascend_patches()
