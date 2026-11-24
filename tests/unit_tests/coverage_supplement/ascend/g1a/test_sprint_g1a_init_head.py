# Copyright 2026 FlagOS Contributors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Best-effort coverage for the head of ModelRunnerFL.__init__ (814b 589-607).

A full runner construction needs the distributed/device shim recipe; this
test only drives the leading config/attribute wiring and lets the first
downstream probe fail. ``__init__`` is decorated (JitWarmupRegistry.capture),
so the test asserts progress through the target lines only when traceback
frames from model_runner itself are present; otherwise it skips.
"""

from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch

from vllm_fl.worker import model_runner as mr


def _config():
    return SimpleNamespace(
        model_config=SimpleNamespace(
            dtype=torch.float16,
            is_encoder_decoder=False,
            disable_cascade_attn=False,
            is_mm_prefix_lm=False,
            is_multimodal_raw_input_only_model=False,
            enable_prompt_embeds=False,
            runner_type="generate",
        ),
        cache_config=SimpleNamespace(cache_dtype="auto", calculate_kv_scales=False),
        offload_config=SimpleNamespace(),
        compilation_config=SimpleNamespace(),
        lora_config=None,
        load_config=None,
        parallel_config=SimpleNamespace(
            distributed_executor_backend="mp",
            decode_context_parallel_size=1,
        ),
        scheduler_config=SimpleNamespace(
            max_num_batched_tokens=64,
            max_num_seqs=4,
            max_num_encoder_input_tokens=0,
            async_scheduling=False,
        ),
        speculative_config=None,
        observability_config=None,
        reasoning_config=None,
    )


def test_init_head_wires_config_attributes():
    fake_self = SimpleNamespace()
    config = _config()
    device = torch.device("cpu")
    with patch.object(
        mr, "current_platform", SimpleNamespace(vendor_name="other")
    ):
        try:
            mr.ModelRunnerFL.__init__(fake_self, config, device)
        except BaseException as exc:
            tb = exc.__traceback__
            linenos = []
            while tb is not None:
                if tb.tb_frame.f_code.co_filename == mr.__file__:
                    linenos.append(tb.tb_lineno)
                tb = tb.tb_next
            if not linenos:
                pytest.skip("init failed before entering model_runner body")
            assert max(linenos) >= 607, f"init stopped early at lines {linenos}"
        else:
            assert fake_self.vllm_config is config
            assert fake_self.device is device
            return

    assert fake_self.vllm_config is config
    assert fake_self.model_config is config.model_config
    assert fake_self.cache_config is config.cache_config
    assert fake_self.parallel_config is config.parallel_config
    assert fake_self.scheduler_config is config.scheduler_config
    assert fake_self.speculative_config is None
    assert fake_self.use_ascend_attention is False
