"""
tools.py — NCP 外部工具注册与调用

将搜索、数据库查询等外部能力注册为协议层的"真值锚点"。
工具返回的结果不是某个模型的观点，而是可审计的外部事实。

设计原则:
- 工具结果标记为 grounded_fact，不可被普通 CHALLENGE 攻击
- 要推翻 grounded_fact，必须提供另一个外部工具的反向证据
- 工具调用有完整的审计日志（谁调用的、什么时候、返回了什么）

兼容 MCP 的工具定义格式（name + description + inputSchema）
"""

import asyncio
import time
import hashlib
from dataclasses import dataclass, field
from typing import Callable, Optional, Any
from datetime import datetime, timezone


# ═══════════════════════════════════════════════════════════════
# 工具定义
# ═══════════════════════════════════════════════════════════════

@dataclass
class ToolDefinition:
    """工具定义 — 兼容 MCP 格式"""
    name: str                           # 唯一标识
    description: str                    # 功能描述
    input_schema: dict = field(default_factory=dict)  # JSON Schema
    output_type: str = "text"           # text / json / url
    trust_level: str = "external"       # external (外部真值) / internal (辅助)
    rate_limit: int = 10                # 每 session 最多调用次数
    timeout: float = 60.0               # 超时秒数


@dataclass
class ToolResult:
    """工具调用结果 — 带审计元数据"""
    tool_name: str
    query: str
    content: str                        # 返回内容
    source_urls: list[str] = field(default_factory=list)  # 来源 URL
    timestamp: str = ""
    elapsed_ms: float = 0.0
    success: bool = True
    error: str = ""

    # 审计字段
    call_id: str = ""                   # 唯一调用 ID
    triggered_by: str = ""              # 谁触发的 (agent_id 或 "system")
    session_id: str = ""


@dataclass
class GroundedFact:
    """
    外部锚定事实 — 不可被普通攻击推翻

    只有提供另一个外部工具的反向证据才能挑战 grounded_fact。
    这是解决"优化共识不是优化真理"的关键机制。
    """
    fact_id: str
    content: str                        # 事实内容
    source_tool: str                    # 来源工具
    source_urls: list[str] = field(default_factory=list)
    verified_at: str = ""
    confidence: float = 0.9             # 外部事实默认高置信度
    challenge_requires: str = "external_evidence"  # 挑战条件


# ═══════════════════════════════════════════════════════════════
# 工具注册表
# ═══════════════════════════════════════════════════════════════

class ToolRegistry:
    """
    NCP 工具注册表

    注册外部工具，管理调用配额，记录审计日志。
    工具结果自动转化为 GroundedFact 注入证据池。
    """

    def __init__(self):
        self.tools: dict[str, ToolDefinition] = {}
        self.handlers: dict[str, Callable] = {}  # name → async handler
        self.call_log: list[ToolResult] = []
        self.grounded_facts: list[GroundedFact] = []
        self._call_counts: dict[str, int] = {}  # name → 本 session 调用次数

    def register(self, definition: ToolDefinition, handler: Callable):
        """
        注册一个外部工具

        handler 签名: async def handler(query: str, **kwargs) -> ToolResult
        """
        self.tools[definition.name] = definition
        self.handlers[definition.name] = handler
        self._call_counts[definition.name] = 0

    def list_tools(self) -> list[dict]:
        """列出所有已注册工具 (MCP 兼容格式)"""
        return [
            {
                "name": t.name,
                "description": t.description,
                "inputSchema": t.input_schema,
                "trust_level": t.trust_level,
            }
            for t in self.tools.values()
        ]

    async def call(self, tool_name: str, query: str,
                    triggered_by: str = "system",
                    session_id: str = "") -> Optional[ToolResult]:
        """
        调用工具并记录审计日志

        返回 ToolResult，同时自动生成 GroundedFact
        """
        if tool_name not in self.tools:
            return ToolResult(
                tool_name=tool_name, query=query,
                content="", success=False, error=f"Tool '{tool_name}' not registered"
            )

        definition = self.tools[tool_name]
        handler = self.handlers[tool_name]

        # 配额检查
        if self._call_counts[tool_name] >= definition.rate_limit:
            return ToolResult(
                tool_name=tool_name, query=query,
                content="", success=False, error="Rate limit exceeded"
            )

        # 调用
        call_id = hashlib.sha256(
            f"{tool_name}_{query}_{time.time()}".encode()
        ).hexdigest()[:10]

        start = time.time()
        try:
            result = await asyncio.wait_for(
                handler(query),
                timeout=definition.timeout
            )
            result.elapsed_ms = (time.time() - start) * 1000
            result.call_id = call_id
            result.triggered_by = triggered_by
            result.session_id = session_id
            result.timestamp = datetime.now(timezone.utc).isoformat()

        except asyncio.TimeoutError:
            result = ToolResult(
                tool_name=tool_name, query=query,
                content="", success=False, error="Timeout",
                call_id=call_id, triggered_by=triggered_by,
                timestamp=datetime.now(timezone.utc).isoformat(),
            )
        except Exception as e:
            result = ToolResult(
                tool_name=tool_name, query=query,
                content="", success=False, error=str(e)[:200],
                call_id=call_id, triggered_by=triggered_by,
                timestamp=datetime.now(timezone.utc).isoformat(),
            )

        # 记录
        self._call_counts[tool_name] += 1
        self.call_log.append(result)

        # 成功的外部工具结果 → GroundedFact
        if result.success and result.content and definition.trust_level == "external":
            fact = GroundedFact(
                fact_id=f"gf_{call_id}",
                content=result.content[:500],
                source_tool=tool_name,
                source_urls=result.source_urls,
                verified_at=result.timestamp,
            )
            self.grounded_facts.append(fact)

        return result

    def get_grounded_facts(self) -> list[GroundedFact]:
        """获取所有已锚定的外部事实"""
        return self.grounded_facts

    def format_grounded_facts(self) -> str:
        """格式化外部事实，用于注入到链展示中"""
        if not self.grounded_facts:
            return ""

        lines = ["[外部已验证事实 — 不可被普通攻击推翻，需提供反向外部证据]"]
        for fact in self.grounded_facts:
            urls = f" ({', '.join(fact.source_urls[:2])})" if fact.source_urls else ""
            lines.append(f"  ★ {fact.content[:150]}{urls}")
        return "\n".join(lines)

    def get_audit_log(self) -> list[dict]:
        """获取完整审计日志"""
        return [
            {
                "call_id": r.call_id,
                "tool": r.tool_name,
                "query": r.query[:100],
                "success": r.success,
                "triggered_by": r.triggered_by,
                "timestamp": r.timestamp,
                "elapsed_ms": r.elapsed_ms,
                "content_length": len(r.content),
            }
            for r in self.call_log
        ]


# ═══════════════════════════════════════════════════════════════
# 预置搜索工具
# ═══════════════════════════════════════════════════════════════



# ═══════════════════════════════════════════════════════════════
# 预置工具: DeepSeek V4 Pro + Firecrawl 联合查证
# ═══════════════════════════════════════════════════════════════

def create_search_tool() -> tuple[ToolDefinition, Callable]:
    """
    DeepSeek V4 Pro + Firecrawl 联合查证工具

    流程: Firecrawl 搜索获取原始网页 → DeepSeek 提取可验证事实
    """
    import os

    ds_key = os.environ.get("DEEPSEEK_API_KEY", "")
    ds_url = os.environ.get("DEEPSEEK_BASE_URL", "https://api.deepseek.com")
    fc_key = os.environ.get("FIRECRAWL_API_KEY", "")

    definition = ToolDefinition(
        name="web_search",
        description="DeepSeek + Firecrawl 联合查证。搜索互联网获取可验证事实，返回带 URL 来源的外部证据。",
        input_schema={
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "搜索查询"}
            },
            "required": ["query"]
        },
        output_type="text",
        trust_level="external",
        rate_limit=20,
        timeout=120.0,
    )

    async def handler(query: str) -> ToolResult:
        import httpx

        # Step 1: Firecrawl 搜索
        fc_headers = {
            "Authorization": f"Bearer {fc_key}",
            "Content-Type": "application/json",
        }

        async with httpx.AsyncClient() as client:
            fc_resp = await client.post(
                "https://api.firecrawl.dev/v2/search",
                headers=fc_headers,
                json={"query": query, "limit": 5},
                timeout=60.0,
            )
            fc_resp.raise_for_status()
            fc_data = fc_resp.json()

        # 解析 Firecrawl 结果
        raw = fc_data.get("data", {})
        web_results = raw.get("web", []) if isinstance(raw, dict) else (raw if isinstance(raw, list) else [])

        source_urls = []
        evidence_lines = []
        for r in web_results[:5]:
            title = r.get("title", "")
            url = r.get("url", "")
            desc = r.get("description", "")
            source_urls.append(url)
            evidence_lines.append(f"[{title}]({url}): {desc}")

        if not evidence_lines:
            return ToolResult(tool_name="web_search", query=query, content="未找到相关结果", success=True)

        # Step 2: DeepSeek 总结事实
        ds_headers = {
            "Authorization": f"Bearer {ds_key}",
            "Content-Type": "application/json",
        }
        ds_payload = {
            "model": "deepseek-chat",
            "messages": [{"role": "user", "content": (
                f"从以下搜索结果中提取可验证的事实。只输出事实，不要观点。标注来源URL。\n\n"
                f"查询: {query}\n\n结果:\n" + "\n".join(evidence_lines)
            )}],
            "temperature": 0.1,
            "stream": False,
        }

        async with httpx.AsyncClient() as client:
            ds_resp = await client.post(
                f"{ds_url}/chat/completions",
                headers=ds_headers,
                json=ds_payload,
                timeout=60.0,
            )
            ds_resp.raise_for_status()
            summary = ds_resp.json()["choices"][0]["message"]["content"]

        return ToolResult(
            tool_name="web_search", query=query,
            content=summary, source_urls=source_urls, success=True,
        )

    return definition, handler


def create_firecrawl_scrape_tool() -> tuple[ToolDefinition, Callable]:
    """Firecrawl Scrape — 抓取指定 URL 的完整页面内容"""
    import os
    fc_key = os.environ.get("FIRECRAWL_API_KEY", "")

    definition = ToolDefinition(
        name="firecrawl_scrape",
        description="抓取指定 URL 的完整页面内容，用于验证具体网页/论文的内容。",
        input_schema={
            "type": "object",
            "properties": {"query": {"type": "string", "description": "要抓取的 URL"}},
            "required": ["query"]
        },
        output_type="text",
        trust_level="external",
        rate_limit=10,
        timeout=90.0,
    )

    async def handler(query: str) -> ToolResult:
        import httpx
        headers = {"Authorization": f"Bearer {fc_key}", "Content-Type": "application/json"}

        async with httpx.AsyncClient() as client:
            resp = await client.post(
                "https://api.firecrawl.dev/v2/scrape",
                headers=headers,
                json={"url": query, "formats": ["markdown"]},
                timeout=90.0,
            )
            resp.raise_for_status()
            data = resp.json()

        markdown = data.get("data", {}).get("markdown", "")
        title = data.get("data", {}).get("metadata", {}).get("title", "")
        content = f"# {title}\n\n{markdown[:2000]}" if markdown else "无法抓取"

        return ToolResult(
            tool_name="firecrawl_scrape", query=query,
            content=content, source_urls=[query], success=True,
        )

    return definition, handler


def create_paper_verify_tool() -> tuple[ToolDefinition, Callable]:
    """验证学术论文是否真实存在 — Firecrawl 搜索 PubMed/Nature"""
    import os
    fc_key = os.environ.get("FIRECRAWL_API_KEY", "")

    definition = ToolDefinition(
        name="paper_verify",
        description="验证学术论文是否真实存在。搜索 PubMed/Nature/Science 验证。",
        input_schema={
            "type": "object",
            "properties": {"query": {"type": "string", "description": "论文信息"}},
            "required": ["query"]
        },
        output_type="text",
        trust_level="external",
        rate_limit=10,
        timeout=90.0,
    )

    async def handler(query: str) -> ToolResult:
        import httpx
        headers = {"Authorization": f"Bearer {fc_key}", "Content-Type": "application/json"}

        try:
            async with httpx.AsyncClient() as client:
                resp = await client.post(
                    "https://api.firecrawl.dev/v2/search",
                    headers=headers,
                    json={"query": f"site:pubmed.ncbi.nlm.nih.gov OR site:nature.com {query}", "limit": 3},
                    timeout=60.0,
                )
                resp.raise_for_status()
                data = resp.json()

            raw = data.get("data", {})
            results = raw.get("web", []) if isinstance(raw, dict) else (raw if isinstance(raw, list) else [])

            if not results:
                return ToolResult(
                    tool_name="paper_verify", query=query,
                    content=f"未找到论文: {query}。该论文可能不存在。", success=True,
                )

            lines = []
            urls = []
            for r in results[:3]:
                urls.append(r.get("url", ""))
                lines.append(f"[{r.get('title','')}]({r.get('url','')}): {r.get('description','')[:150]}")

            return ToolResult(
                tool_name="paper_verify", query=query,
                content=f"找到 {len(results)} 条结果:\n\n" + "\n\n".join(lines),
                source_urls=urls, success=True,
            )
        except Exception as e:
            return ToolResult(
                tool_name="paper_verify", query=query,
                content=f"验证失败: {e}", success=False, error=str(e)[:200],
            )

    return definition, handler


# ═══════════════════════════════════════════════════════════════
# 快捷初始化
# ═══════════════════════════════════════════════════════════════

def create_default_registry() -> ToolRegistry:
    """创建带预置工具的注册表"""
    from .adapter import load_env
    load_env()

    registry = ToolRegistry()
    registry.register(*create_search_tool())
    registry.register(*create_firecrawl_scrape_tool())
    registry.register(*create_paper_verify_tool())
    return registry
