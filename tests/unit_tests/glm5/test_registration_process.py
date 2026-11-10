# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 BAAI. All rights reserved.

import subprocess
import sys
import textwrap


def test_general_plugin_keeps_glm_adapter_registration():
    code = """
        import vllm_fl
        from vllm.model_executor.models.registry import ModelRegistry
        from vllm.model_executor.models.config import MODELS_CONFIG_MAP

        expected = {
            'Glm5NextForCausalLM': 'vllm_fl.models.glm5.model',
            'Glm5NextForConditionalGeneration': 'vllm_fl.models.glm5.multimodal',
        }
        for _ in range(2):
            vllm_fl.register_model()
            for architecture, module in expected.items():
                entry = ModelRegistry.models[architecture]
                assert entry.module_name == module, (architecture, entry)
                assert entry.class_name == architecture
                assert MODELS_CONFIG_MAP[architecture].__module__ == 'vllm_fl.models.glm5.runtime'
    """
    subprocess.run([sys.executable, "-c", textwrap.dedent(code)], check=True)
