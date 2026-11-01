"""SiliconFlow configuration for the shared Chat Completions transport."""
import os

from core.chat_client import ChatCompletionsClient, ChatContent, chat_messages


class SiliconFlowClient(ChatCompletionsClient):
    default_base_url = 'https://api.siliconflow.cn/v1'
    config_prefix = 'SILICONFLOW'
    provider_label = '硅基流动'

    def __init__(self, api_key, base_url=None, *, transport=None):
        setting = os.getenv('SILICONFLOW_ENABLE_THINKING', 'false').strip().lower()
        if setting not in {'true', 'false', '1', '0', 'yes', 'no', 'on', 'off'}:
            raise ValueError('SILICONFLOW_ENABLE_THINKING 必须为 true/false')
        self.enable_thinking = setting in {'true', '1', 'yes', 'on'}
        super().__init__(api_key, base_url, transport=transport)

    def generation_parameters(self, model, max_tokens, temperature):
        return {**super().generation_parameters(model, max_tokens, temperature),
                'enable_thinking': self.enable_thinking}
