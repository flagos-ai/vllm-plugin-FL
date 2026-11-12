"""Choose one Linear provider from the model policy before ATen registration."""

from functools import partial, update_wrapper
from flag_gems.runtime.op_registrar import GeneralOpRegistrar
from vllm_fl.dispatch import get_default_manager


class K3OpRegistrar(GeneralOpRegistrar):
    def __init__(self, *args, execution_config, **kwargs):
        # Bind once. No environment/config lookup occurs in a model step.
        linear = get_default_manager().resolve("kimi_k3_linear")
        self._linear = update_wrapper(
            partial(linear, tp8_compute=execution_config.tp8_compute), linear
        )
        super().__init__(*args, **kwargs)

    def _resolve_live_override(self, key, fn):
        if key == "linear":
            return self._linear
        return super()._resolve_live_override(key, fn)
