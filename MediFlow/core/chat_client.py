"""Shared OpenAI-compatible Chat Completions transport and tool history."""
import copy
import json
from types import SimpleNamespace
from urllib.parse import urlsplit

import httpx


class ChatContent(list):
    """Keep the original assistant tool calls and optional reasoning per request."""
    def __init__(self, blocks, assistant):
        super().__init__(blocks)
        self.assistant_message = copy.deepcopy(assistant)


def chat_messages(messages):
    result = []
    for message in messages:
        role, content = message['role'], message['content']
        if role == 'assistant' and isinstance(content, ChatContent):
            result.append(copy.deepcopy(content.assistant_message))
        elif isinstance(content, str):
            result.append({'role':role, 'content':content})
        elif role == 'assistant':
            assistant = {'role':'assistant', 'content':''}
            calls = []
            for block in content:
                if block['type'] == 'text':
                    assistant['content'] += block['text']
                elif block['type'] == 'tool_use':
                    calls.append({'id':block['id'], 'type':'function', 'function':{
                        'name':block['name'], 'arguments':json.dumps(block['input'],ensure_ascii=False)}})
                else:
                    raise ValueError('不支持的助手内容类型')
            if calls:
                assistant['tool_calls'] = calls
                assistant['content'] = assistant['content'] or None
            result.append(assistant)
        else:
            for block in content:
                if block['type'] == 'tool_result':
                    output = block['content']
                    result.append({'role':'tool', 'tool_call_id':block['tool_use_id'],
                        'content':output if isinstance(output,str) else json.dumps(output,ensure_ascii=False)})
                elif block['type'] == 'text':
                    result.append({'role':role, 'content':block['text']})
                else:
                    raise ValueError('不支持的模型输入内容类型')
    return result


class ChatCompletionsClient:
    default_base_url = 'https://api.openai.com/v1'
    config_prefix = 'OPENAI'
    provider_label = '模型服务'

    def __init__(self, api_key, base_url=None, *, transport=None):
        self.api_key = api_key.strip()
        self.base_url = (base_url or self.default_base_url).rstrip('/')
        parsed = urlsplit(self.base_url)
        if parsed.scheme not in {'https','http'} or not parsed.hostname or parsed.username or parsed.password or parsed.query or parsed.fragment:
            raise ValueError(f'{self.config_prefix}_BASE_URL 必须是无凭据或查询参数的 HTTP(S) 接口基址')
        self._http = httpx.AsyncClient(transport=transport, timeout=45, follow_redirects=False)
        self.messages = SimpleNamespace(create=self.create)

    def generation_parameters(self, model, max_tokens, temperature):
        params = {'max_tokens': max_tokens}
        if temperature is not None:
            params['temperature'] = temperature
        return params

    async def close(self):
        await self._http.aclose()

    async def create(self, *, model, messages, max_tokens, system=None, tools=None, temperature=None, **kwargs):
        if not self.api_key or self.api_key in {'xxx','your-api-key','your-siliconflow-api-key','your-openai-api-key'}:
            raise RuntimeError(f'尚未配置 {self.config_prefix}_API_KEY，请在后端 .env 填写 API Key')
        conversation = chat_messages(messages)
        if system:
            conversation.insert(0, {'role':'system','content':system})
        body = {'model':model, 'messages':conversation, 'stream':False}
        body.update(self.generation_parameters(model, max_tokens, temperature))
        # Shared instrumentation can supply provider-specific thinking controls.
        extra_body = kwargs.get('extra_body') or {}
        if 'thinking' in extra_body:
            body['thinking'] = extra_body['thinking']
        if tools:
            body['tools'] = [{'type':'function', 'function':{'name':tool['name'],
                'description':tool.get('description',''), 'parameters':tool['input_schema']}} for tool in tools]
        response = await self._http.post(self.base_url + '/chat/completions',
            headers={'Authorization':'Bearer '+self.api_key}, json=body)
        if not response.is_success:
            raise RuntimeError(f'{self.provider_label}请求失败（HTTP {response.status_code}）')
        data = response.json()
        choices = data.get('choices')
        if not isinstance(choices,list) or not choices:
            raise RuntimeError(f'{self.provider_label}未返回有效回答，未执行工具')
        choice = choices[0]
        finish = choice.get('finish_reason')
        if finish not in {'stop','eos','tool_calls','length'}:
            raise RuntimeError(f'{self.provider_label}回答未正常结束，未执行工具')
        message = choice.get('message')
        if not isinstance(message,dict):
            raise RuntimeError(f'{self.provider_label}消息格式不正确，未执行工具')
        text = message.get('content') or message.get('refusal')
        if text is not None and not isinstance(text,str):
            raise RuntimeError(f'{self.provider_label}回答正文格式不正确，未执行工具')
        blocks = [{'type':'text','text':text}] if text else []
        calls = message.get('tool_calls') or []
        if not isinstance(calls,list):
            raise RuntimeError(f'{self.provider_label}工具调用格式不正确，未执行工具')
        if finish != 'length':
            if bool(calls) != (finish == 'tool_calls'):
                raise RuntimeError(f'{self.provider_label}工具调用与结束状态不一致，未执行工具')
            # Validate all calls before returning any of them for execution.
            seen_ids = set()
            for call in calls:
                if not isinstance(call,dict) or call.get('type') != 'function' or not isinstance(call.get('function'),dict):
                    raise RuntimeError(f'{self.provider_label}工具调用格式不正确，未执行工具')
                function = call['function']
                call_id, name = call.get('id'), function.get('name')
                if not isinstance(call_id,str) or not call_id or call_id in seen_ids or not isinstance(name,str) or not name:
                    raise RuntimeError(f'{self.provider_label}工具标识不正确，未执行工具')
                seen_ids.add(call_id)
                try:
                    arguments = json.loads(function['arguments'])
                except (KeyError,TypeError,ValueError):
                    raise RuntimeError(f'{self.provider_label}工具参数不是完整 JSON，未执行工具') from None
                if not isinstance(arguments,dict):
                    raise RuntimeError(f'{self.provider_label}工具参数必须是 JSON 对象，未执行工具')
                blocks.append({'type':'tool_use','id':call_id,'name':name,'input':arguments})
        assistant = {'role':'assistant','content':text}
        if calls:
            assistant['tool_calls'] = calls
        if isinstance(message.get('reasoning_content'),str):
            assistant['reasoning_content'] = message['reasoning_content']
        usage = data.get('usage') or {}
        details = usage.get('prompt_tokens_details') or {}
        return SimpleNamespace(content=ChatContent(blocks,assistant),
            stop_reason='max_tokens' if finish=='length' else 'tool_use' if calls else 'end_turn',
            usage={'input_tokens':usage.get('prompt_tokens'), 'output_tokens':usage.get('completion_tokens'),
                'cache_read_input_tokens':details.get('cached_tokens',usage.get('prompt_cache_hit_tokens')),
                'cache_creation_input_tokens':None})
