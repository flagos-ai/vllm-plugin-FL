import gc
import weakref
from types import SimpleNamespace

import pytest

from vllm_fl.dispatch.backends.vendor.iluvatar.worker import (
    _release_benchmark_tensors,
)


@pytest.mark.parametrize("fail", [False, True])
def test_release_on_success_and_exception(fail):
    class Tensor:
        pass

    obj = SimpleNamespace(cache={"shape": "compiled"})
    x = Tensor()
    ref = weakref.ref(x)

    def run(self, value, *, flag):
        self._last_benchmark_args = (value,)
        self._last_benchmark_meta = {"x": value}
        self.nargs = {"x": value}
        assert flag == 3
        if fail:
            raise ValueError("failed tuning")
        return "result"

    wrapped = _release_benchmark_tensors(run)
    assert _release_benchmark_tensors(wrapped) is wrapped
    if fail:
        with pytest.raises(ValueError, match="failed tuning"):
            wrapped(obj, x, flag=3)
    else:
        assert wrapped(obj, x, flag=3) == "result"
    del x
    gc.collect()
    assert ref() is None
    assert obj._last_benchmark_args == ()
    assert obj._last_benchmark_meta == {}
    assert obj.nargs is None
    assert obj.cache == {"shape": "compiled"}
