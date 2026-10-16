import pytest


@pytest.fixture(autouse=True)
def hy4_shared_registry_isolation(request, monkeypatch):
    if request.node.fspath.basename not in (
        "test_hy_v4_selected_paths.py",
        "test_hy_v4_review_contracts.py",
    ):
        yield
        return
    from vllm_fl.dispatch import builtin_ops
    from vllm_fl.dispatch.backends.flaggems.impl import mla_ops
    from vllm_fl.dispatch.backends.vendor.cuda.impl import mla
    from vllm_fl.dispatch.manager import reset_default_manager
    from vllm_fl.models.hy4 import runtime

    monkeypatch.setattr(
        builtin_ops, "_get_current_vendor_backend_dirs", lambda *_: "cuda"
    )
    monkeypatch.setattr(
        mla, "has_device_kernel", lambda *a: runtime.has_device_kernel(*a)
    )
    monkeypatch.setattr(
        mla_ops, "use_flaggems_op", lambda name: runtime.use_flaggems_op(name)
    )
    reset_default_manager()
    yield
    reset_default_manager()


@pytest.fixture
def prefer_native():
    from vllm_fl.dispatch.policy import SelectionPolicy, policy_context

    with policy_context(SelectionPolicy.from_dict(prefer="vendor")):
        yield
