"""Provider factory; OpenAI-compatible services use Chat Completions."""
import os
import re

from anthropic import AsyncAnthropic
from core.chat_client import ChatCompletionsClient


DEFAULT_OPENAI_MODEL = 'gpt-4.1'


def provider():
    value = os.getenv('LLM_PROVIDER', 'anthropic').strip().lower()
    if value not in {'anthropic', 'openai', 'siliconflow'}:
        raise ValueError('LLM_PROVIDER 必须是 siliconflow、openai 或 anthropic')
    return value


def llm_config():
    selected = provider()
    if selected == 'siliconflow':
        return {'api_key': os.getenv('SILICONFLOW_API_KEY', '').strip(),
                'model': os.getenv('SILICONFLOW_MODEL', 'deepseek-ai/DeepSeek-V4-Flash').strip() or 'deepseek-ai/DeepSeek-V4-Flash',
                'base_url': os.getenv('SILICONFLOW_BASE_URL', '').strip() or 'https://api.siliconflow.cn/v1'}
    if selected == 'openai':
        return {'api_key': os.getenv('OPENAI_API_KEY', '').strip(),
                'model': os.getenv('OPENAI_MODEL', DEFAULT_OPENAI_MODEL).strip() or DEFAULT_OPENAI_MODEL,
                'base_url': os.getenv('OPENAI_BASE_URL', '').strip() or 'https://api.openai.com/v1'}
    key = os.getenv('ANTHROPIC_API_KEY', '').strip()
    if not key:
        raise RuntimeError('未设置 ANTHROPIC_API_KEY')
    config = {'api_key': key, 'model': os.getenv('ANTHROPIC_MODEL', 'claude-3-5-sonnet-20241022').strip()}
    if os.getenv('ANTHROPIC_BASE_URL', '').strip():
        config['base_url'] = os.environ['ANTHROPIC_BASE_URL'].strip()
    return config


def create_client(api_key, base_url=None, **kwargs):
    if provider() == 'siliconflow':
        from core.siliconflow_client import SiliconFlowClient
        return SiliconFlowClient(api_key, base_url)
    if provider() == 'openai':
        return OpenAIChatClient(api_key, base_url)
    return AsyncAnthropic(api_key=api_key, **({'base_url': base_url} if base_url else {}), **kwargs)


class OpenAIChatClient(ChatCompletionsClient):
    provider_label = 'OpenAI 兼容服务'

    def generation_parameters(self, model, max_tokens, temperature):
        if re.match(r'^(?:gpt-[56](?:[.-]|$)|o[134](?:-|$))', model):
            return {'max_completion_tokens': max(max_tokens, 2048),
                    'reasoning_effort': os.getenv('OPENAI_REASONING_EFFORT', 'low').strip() or 'low'}
        return super().generation_parameters(model, max_tokens, temperature)
