"""
adapter.py — 模型适配器

将不同的 LLM API 统一为神经元可调用的接口。
支持: OpenAI / Anthropic / Gemini / Ollama
全部通过 yunwu.ai 中转，统一 base_url。
"""

from typing import Optional
import os
import json


def load_env():
    """加载 .env 文件"""
    env_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), ".env")
    if os.path.exists(env_path):
        with open(env_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#") and "=" in line:
                    key, value = line.split("=", 1)
                    os.environ.setdefault(key.strip(), value.strip())


# 启动时加载环境变量
load_env()


def adaptive_timeout(prompt_length: int, base: float = 300.0) -> float:
    """
    自适应超时算法

    基础超时 300 秒 (5分钟)，给模型充分的深度思考时间。
    根据 prompt 长度进一步调整。
    """
    if prompt_length < 500:
        return base
    elif prompt_length < 2000:
        return base * 1.2
    elif prompt_length < 5000:
        return base * 1.5
    else:
        return base * 2


class BaseAdapter:
    """适配器基类"""

    max_retries: int = 3
    _timeout_strikes: dict = {}  # 类级别: {adapter_id: 超时次数}

    async def call(self, prompt: str, context: dict) -> str:
        raise NotImplementedError

    async def call_with_retry(self, prompt: str, context: dict) -> str:
        """带重试的调用 — 超时3次后永久跳过该模型"""
        import httpx

        # 检查是否已被禁用
        adapter_id = id(self)
        strikes = BaseAdapter._timeout_strikes.get(adapter_id, 0)
        if strikes >= 3:
            raise RuntimeError(f"模型已被跳过 (连续超时{strikes}次)")

        last_error = None
        for attempt in range(self.max_retries):
            try:
                result = await self.call(prompt, context)
                # 成功调用，重置超时计数
                BaseAdapter._timeout_strikes[adapter_id] = 0
                return result
            except (httpx.ReadTimeout, httpx.ConnectTimeout, httpx.ConnectError) as e:
                last_error = e
                wait = (attempt + 1) * 5
                print(f"    [重试] 第{attempt+1}次超时，等待{wait}秒后重试...")
                import asyncio
                await asyncio.sleep(wait)

        # 3次重试都失败，记录一次 strike
        BaseAdapter._timeout_strikes[adapter_id] = strikes + 1
        new_strikes = BaseAdapter._timeout_strikes[adapter_id]
        if new_strikes >= 3:
            print(f"    [禁用] 该模型连续超时{new_strikes}次，后续轮次将跳过")
        raise last_error


class OpenAIAdapter(BaseAdapter):
    """
    OpenAI 兼容接口适配器
    通过 yunwu.ai 中转访问 GPT-5.4
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: Optional[str] = None,
        base_url: Optional[str] = None,
    ):
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY", "")
        self.model = model or os.environ.get("OPENAI_MODEL", "gpt-5.4-2026-03-05")
        yunwu = os.environ.get("YUNWU_BASE_URL", "https://yunwu.ai")
        self.base_url = base_url or f"{yunwu}/v1"

    async def call(self, prompt: str, context: dict) -> str:
        import httpx

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

        # 自适应超时: 根据 prompt 长度动态调整
        prompt_len = len(prompt)
        timeout = adaptive_timeout(prompt_len)

        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": f"你的角色: {context.get('role', 'assistant')}"},
                {"role": "user", "content": prompt},
            ],
            "temperature": 0.3,
            "max_tokens": 2000,
        }

        async with httpx.AsyncClient() as client:
            response = await client.post(
                f"{self.base_url}/chat/completions",
                headers=headers,
                json=payload,
                timeout=timeout,
            )
            response.raise_for_status()
            data = response.json()
            return data["choices"][0]["message"]["content"]


class GeminiAdapter(BaseAdapter):
    """
    Gemini 适配器
    通过 yunwu.ai 中转访问 Gemini 3.1 Pro
    使用 OpenAI 兼容格式
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: Optional[str] = None,
        base_url: Optional[str] = None,
    ):
        self.api_key = api_key or os.environ.get("GEMINI_API_KEY", "")
        self.model = model or os.environ.get("GEMINI_MODEL", "gemini-3.1-pro-preview")
        yunwu = os.environ.get("YUNWU_BASE_URL", "https://yunwu.ai")
        self.base_url = base_url or f"{yunwu}/v1"

    async def call(self, prompt: str, context: dict) -> str:
        import httpx

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

        prompt_len = len(prompt)
        timeout = adaptive_timeout(prompt_len)

        payload = {
            "model": self.model,
            "messages": [
                {"role": "user", "content": prompt},
            ],
            "temperature": 0.3,
            "max_tokens": 2000,
        }

        async with httpx.AsyncClient() as client:
            response = await client.post(
                f"{self.base_url}/chat/completions",
                headers=headers,
                json=payload,
                timeout=timeout,
            )
            response.raise_for_status()
            data = response.json()
            return data["choices"][0]["message"]["content"]


class AnthropicAdapter(BaseAdapter):
    """
    Anthropic Claude 适配器
    通过 yunwu.ai 中转访问 Claude Opus 4.6
    使用 Anthropic Messages 格式
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: Optional[str] = None,
        base_url: Optional[str] = None,
    ):
        self.api_key = api_key or os.environ.get("ANTHROPIC_API_KEY", "")
        self.model = model or os.environ.get("ANTHROPIC_MODEL", "claude-opus-4-6")
        yunwu = os.environ.get("YUNWU_BASE_URL", "https://yunwu.ai")
        self.base_url = base_url or f"{yunwu}/v1"

    async def call(self, prompt: str, context: dict) -> str:
        import httpx

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

        prompt_len = len(prompt)
        timeout = adaptive_timeout(prompt_len)

        payload = {
            "model": self.model,
            "messages": [
                {"role": "user", "content": prompt},
            ],
            "temperature": 0.2,
            "max_tokens": 4000,
        }

        async with httpx.AsyncClient() as client:
            response = await client.post(
                f"{self.base_url}/chat/completions",
                headers=headers,
                json=payload,
                timeout=timeout,
            )
            response.raise_for_status()
            data = response.json()
            return data["choices"][0]["message"]["content"]


class QwenAdapter(BaseAdapter):
    """
    阿里云百炼平台适配器
    访问 Qwen 系列模型 (qwen-max, qwen-plus, qwen-turbo 等)
    以及百炼平台上的其他模型 (DeepSeek, Kimi, GLM, MiniMax)
    OpenAI 兼容格式
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: Optional[str] = None,
        base_url: Optional[str] = None,
    ):
        self.api_key = api_key or os.environ.get("QWEN_API_KEY", "")
        self.model = model or os.environ.get("QWEN_MODEL", "qwen-max")
        self.base_url = base_url or os.environ.get(
            "QWEN_BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1"
        )

    async def call(self, prompt: str, context: dict) -> str:
        import httpx

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

        prompt_len = len(prompt)
        timeout = adaptive_timeout(prompt_len)

        payload = {
            "model": self.model,
            "messages": [
                {"role": "user", "content": prompt},
            ],
            "temperature": 0.3,
        }

        async with httpx.AsyncClient() as client:
            response = await client.post(
                f"{self.base_url}/chat/completions",
                headers=headers,
                json=payload,
                timeout=timeout,
            )
            response.raise_for_status()
            data = response.json()
            return data["choices"][0]["message"]["content"]


class DeepSeekAdapter(BaseAdapter):
    """
    DeepSeek 官方 API 适配器
    访问 DeepSeek V4 Pro — 专职查证，不参与辩论
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: Optional[str] = None,
        base_url: Optional[str] = None,
    ):
        self.api_key = api_key or os.environ.get("DEEPSEEK_API_KEY", "")
        self.model = model or os.environ.get("DEEPSEEK_MODEL", "deepseek-chat")
        self.base_url = base_url or os.environ.get(
            "DEEPSEEK_BASE_URL", "https://api.deepseek.com"
        )

    async def call(self, prompt: str, context: dict) -> str:
        import httpx

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

        prompt_len = len(prompt)
        timeout = adaptive_timeout(prompt_len)

        payload = {
            "model": self.model,
            "messages": [
                {"role": "user", "content": prompt},
            ],
            "temperature": 0.3,
            "stream": False,
        }

        async with httpx.AsyncClient() as client:
            response = await client.post(
                f"{self.base_url}/chat/completions",
                headers=headers,
                json=payload,
                timeout=timeout,
            )
            response.raise_for_status()
            data = response.json()
            return data["choices"][0]["message"]["content"]


class DoubaoAdapter(BaseAdapter):
    """
    字节跳动豆包适配器
    通过火山引擎 Ark Responses API 访问 doubao-seed-2-0-pro
    API Key 通过环境变量 DOUBAO_API_KEY 设置，不可硬编码
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: Optional[str] = None,
        base_url: Optional[str] = None,
    ):
        self.api_key = api_key or os.environ.get("DOUBAO_API_KEY", "")
        self.model = model or os.environ.get("DOUBAO_MODEL", "doubao-seed-2-0-pro-260215")
        self.base_url = base_url or os.environ.get(
            "DOUBAO_BASE_URL", "https://ark.cn-beijing.volces.com/api/v3"
        )

    async def call(self, prompt: str, context: dict) -> str:
        import httpx

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json; charset=utf-8",
        }

        prompt_len = len(prompt)
        timeout = adaptive_timeout(prompt_len)

        # Ark Responses API 格式
        payload = {
            "model": self.model,
            "input": [
                {
                    "role": "user",
                    "content": [
                        {"type": "input_text", "text": prompt},
                    ],
                },
            ],
        }

        async with httpx.AsyncClient() as client:
            response = await client.post(
                f"{self.base_url}/responses",
                headers=headers,
                json=payload,
                timeout=timeout,
            )
            response.raise_for_status()
            data = response.json()

        # 从 Responses API 的 output 中提取 assistant message
        for item in data.get("output", []):
            if item.get("type") == "message" and item.get("role") == "assistant":
                for block in item.get("content", []):
                    if block.get("type") == "output_text":
                        return block["text"]

        return ""


class OllamaAdapter(BaseAdapter):
    """Ollama 本地模型适配器 (免费)"""

    def __init__(self, model: str = "llama3.1:8b",
                 base_url: str = "http://localhost:11434"):
        self.model = model
        self.base_url = base_url

    async def call(self, prompt: str, context: dict) -> str:
        import httpx

        payload = {
            "model": self.model,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": 0.3,
                "num_predict": 2000,
            },
        }

        async with httpx.AsyncClient() as client:
            response = await client.post(
                f"{self.base_url}/api/generate",
                json=payload,
                timeout=60.0,
            )
            response.raise_for_status()
            data = response.json()
            return data["response"]


class MiMoAdapter(BaseAdapter):
    """
    小米 MiMo 适配器
    访问 MiMo V2.5 Pro — OpenAI 兼容格式
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: Optional[str] = None,
        base_url: Optional[str] = None,
    ):
        self.api_key = api_key or os.environ.get("MIMO_API_KEY", "")
        self.model = model or os.environ.get("MIMO_MODEL", "mimo-v2.5-pro")
        self.base_url = base_url or os.environ.get(
            "MIMO_BASE_URL", "https://api.xiaomimimo.com/v1"
        )

    async def call(self, prompt: str, context: dict) -> str:
        import httpx

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

        prompt_len = len(prompt)
        timeout = adaptive_timeout(prompt_len)

        payload = {
            "model": self.model,
            "messages": [
                {"role": "user", "content": prompt},
            ],
            "max_completion_tokens": 4000,
            "temperature": 0.3,
            "top_p": 0.95,
            "stream": False,
        }

        async with httpx.AsyncClient() as client:
            response = await client.post(
                f"{self.base_url}/chat/completions",
                headers=headers,
                json=payload,
                timeout=timeout,
            )
            response.raise_for_status()
            data = response.json()
            return data["choices"][0]["message"]["content"]


class KimiK2Adapter(BaseAdapter):
    """
    Kimi K2 Thinking 适配器
    通过阿里云百炼平台访问 Kimi-K2-Thinking
    OpenAI 兼容格式 (非流式)
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: Optional[str] = None,
        base_url: Optional[str] = None,
    ):
        self.api_key = api_key or os.environ.get("QWEN_API_KEY", "")
        self.model = model or os.environ.get("KIMI_K2_MODEL", "kimi-k2-thinking")
        self.base_url = base_url or os.environ.get(
            "QWEN_BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1"
        )

    async def call(self, prompt: str, context: dict) -> str:
        import httpx

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

        prompt_len = len(prompt)
        timeout = adaptive_timeout(prompt_len)

        payload = {
            "model": self.model,
            "messages": [
                {"role": "user", "content": prompt},
            ],
            "temperature": 0.3,
            "stream": False,
        }

        async with httpx.AsyncClient() as client:
            response = await client.post(
                f"{self.base_url}/chat/completions",
                headers=headers,
                json=payload,
                timeout=timeout,
            )
            response.raise_for_status()
            data = response.json()
            return data["choices"][0]["message"]["content"]


class MockAdapter(BaseAdapter):
    """模拟适配器 — 用于测试"""

    def __init__(self, responses: Optional[list[str]] = None):
        self.responses = responses or ["INTENT: confirm\nCERTAINTY: strong\nCONTENT: 测试通过"]
        self.call_count = 0

    async def call(self, prompt: str, context: dict) -> str:
        idx = self.call_count % len(self.responses)
        self.call_count += 1
        return self.responses[idx]
