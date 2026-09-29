# SPDX-License-Identifier: Apache-2.0

from unittest.mock import patch

from vllm_fl.platform import PlatformFL


def test_sunrise_supports_static_graph_mode():
    with patch.object(PlatformFL, "vendor_name", "sunrise"):
        assert PlatformFL.support_static_graph_mode()
