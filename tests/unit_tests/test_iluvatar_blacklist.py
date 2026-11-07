from types import SimpleNamespace
from unittest.mock import patch

from vllm_fl.dispatch.config import load_platform_config
from vllm_fl.utils import get_flag_gems_whitelist_blacklist, use_flaggems_op


def test_iluvatar_yaml_excludes_incompatible_ops_without_environment(monkeypatch):
    for key in (
        "VLLM_FL_CONFIG",
        "VLLM_FL_FLAGOS_WHITELIST",
        "VLLM_FL_FLAGOS_BLACKLIST",
        "VLLM_FL_FLAGOS_BLACKLIST_APPEND",
    ):
        monkeypatch.delenv(key, raising=False)
    monkeypatch.setenv("VLLM_FL_PREFER_ENABLED", "True")
    monkeypatch.setenv("VLLM_FL_PREFER", "flagos")
    monkeypatch.setenv("USE_FLAGGEMS", "1")

    with patch(
        "vllm.platforms.current_platform", SimpleNamespace(vendor_name="iluvatar")
    ):
        config = load_platform_config("iluvatar")
        whitelist, blacklist = get_flag_gems_whitelist_blacklist()
        assert config["flagos_blacklist"] == [
            "zeros",
            "zero_",
            "linear",
            "native_layer_norm",
            "addmm",
            "mm",
            "slice",
            "mm_out",
            "_scaled_dot_product_flash_attention",
            "_flash_attention_forward",
        ]
        assert whitelist is None
        assert blacklist == config["flagos_blacklist"]
        for op_name in ("mm", "slice", "mm_out"):
            assert not use_flaggems_op(op_name)
        assert use_flaggems_op("add")
