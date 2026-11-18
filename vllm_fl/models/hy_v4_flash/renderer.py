# SPDX-License-Identifier: Apache-2.0
from dataclasses import replace

from vllm.renderers.hf import HfRenderer


class HYV4FlashRenderer(HfRenderer):
    """Preserve the vendor default string-content template on vLLM 0.32."""

    @staticmethod
    def _flash_params(params):
        if (
            params.chat_template is None
            and params.chat_template_content_format == "auto"
        ):
            return replace(params, chat_template_content_format="string")
        return params

    def render_messages(self, messages, params):
        return super().render_messages(messages, self._flash_params(params))

    async def render_messages_async(self, messages, params):
        return await super().render_messages_async(messages, self._flash_params(params))
