from types import SimpleNamespace

import pytest
import torch

from vllm.distributed.device_communicators.base_device_communicator import (
    DeviceCommunicatorBase,
)

from vllm_fl.dispatch.backends.vendor.iluvatar.communicator import (
    IluvatarCommunicatorFL,
)
from vllm_fl.distributed import communicator as module


@pytest.mark.parametrize("disabled", [False, True])
def test_iluvatar_flagcx_fail_closed(monkeypatch, disabled):
    def base_init(self, *args, **kwargs):
        self.world_size = 2
        self.cpu_group = object()
        self.device = torch.device("cuda:0")
        self.use_all2all = False

    monkeypatch.setattr(DeviceCommunicatorBase, "__init__", base_init)
    assert IluvatarCommunicatorFL is not module.CommunicatorFL
    error = OSError("missing test library")
    comm = SimpleNamespace(disabled=disabled, init_error=error)
    monkeypatch.setattr(module, "PyFlagcxCommunicator", lambda **kwargs: comm)
    if disabled:
        with pytest.raises(RuntimeError, match="FlagCX requested") as exc:
            IluvatarCommunicatorFL(object())
        assert exc.value.__cause__ is error
    else:
        assert IluvatarCommunicatorFL(object()).pyflagcx_comm is comm
