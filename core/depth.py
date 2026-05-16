"""
depth.py — 深度推敲算法 v2

核心思想: 不让模型停在第一个答案。
通过递归质疑，强制模型对自己的每一步推理进行深度审视。

v2 优化:
- 断言提取更鲁棒（多种解析策略 fallback）
- 质疑判定更精确（量化评分而非关键词匹配）
- 支持并行质疑（同层多个断言并发处理）
- 进度回调（实时输出推敲过程）
- 深度递进策略（每层质疑角度不同）
"""

from dataclasses import dataclass, field
from typing import Callable, Optional
import json
import asyncio
import re
import time


@dataclass
class Claim:
    """一个断言 — 逻辑链条中的最小单元"""
    id: int
    content: str                    # 断言内容
    basis: str                      # 依据
    depth: int                      # 在推敲树中的深度
    parent_id: Optional[int] = None # 由哪个断言推出
    challenged: bool = False        # 是否被质疑过
    survived: bool = False          # 是否经受住了质疑
    challenge_detail: str = ""      # 质疑的具体内容
    confidence: float = 0.0         # 经受质疑后的置信度 (0-1)


@dataclass
class DepthResult:
    """深度推敲的结果"""
    original_answer: str
    final_answer: str
    claims: list[Claim]
    max_depth_reached: int
    uncertainties: list[str]
    revisions: int
    total_challenges: int
    elapsed_seconds: float = 0.0


# === 提示词模板 ===

EXTRACT_CLAIMS_PROMPT = """你是一个逻辑分析器。从以下文本中提取关键断言。

要求:
- 每个断言必须是一个可以被独立验证或反驳的具体陈述
- 不要提取修饰语或过渡句
- 提取 3-5 个最核心的断言
- 严格按 JSON 格式输出

文本:
---
{answer}
---

输出格式（严格 JSON，不要 markdown 代码块）:
[{{"claim":"断言1","basis":"依据1"}},{{"claim":"断言2","basis":"依据2"}}]"""


CHALLENGE_PROMPT_BY_DEPTH = [
    # 深度0: 质疑事实准确性
    """你是事实核查员。质疑以下断言的事实准确性。

断言: {claim}
依据: {basis}

问题:
1. 这个断言在事实层面是否准确？有没有过时或错误的信息？
2. 依据是否真实可靠？

如果断言事实准确，输出: VERDICT: SOLID | 理由
如果有事实错误，输出: VERDICT: FLAWED | 具体错误是什么""",

    # 深度1: 质疑逻辑严密性
    """你是逻辑审查员。质疑以下断言的逻辑严密性。

断言: {claim}
依据: {basis}
上下文: {context}

问题:
1. 从依据到断言的推理是否有逻辑跳跃？
2. 是否存在隐含假设未被说明？
3. 是否存在反例？

如果逻辑严密，输出: VERDICT: SOLID | 理由
如果有逻辑漏洞，输出: VERDICT: FLAWED | 具体漏洞是什么""",

    # 深度2: 质疑完备性
    """你是完备性审查员。质疑以下断言是否遗漏了重要因素。

断言: {claim}
依据: {basis}
上下文: {context}

问题:
1. 是否遗漏了重要的考虑因素？
2. 在什么条件下这个断言会不成立？
3. 是否过度简化了问题？

如果考虑充分，输出: VERDICT: SOLID | 理由
如果有重要遗漏，输出: VERDICT: FLAWED | 遗漏了什么""",
]


REVISE_PROMPT = """你之前的分析中有一个断言被质疑了。

原始断言: {claim}
质疑结果: {challenge_result}
原始问题: {question}

请针对这个质疑，修正或补充你的分析。
只输出修正后的内容，不要重复原始回答的全部内容。
聚焦在被质疑的点上。"""


SYNTHESIZE_PROMPT = """你是一个严谨的决策分析师。基于以下推敲过程，给出最终结论。

原始问题: {question}

经过 {total_challenges} 次质疑，{revisions} 次修正。

经受住质疑的核心断言（高置信度）:
{survived_claims}

被推翻或存在不确定性的断言:
{uncertain_claims}

请给出:
1. 最终结论（基于经受住质疑的断言）
2. 置信度说明（哪些部分高度确定，哪些有保留）
3. 关键前提（结论成立的条件）"""


class DepthEngine:
    """
    深度推敲引擎 v2

    优化点:
    1. 断言提取: 多策略 fallback，确保总能提取出断言
    2. 质疑深度: 每层用不同角度质疑（事实→逻辑→完备性）
    3. 判定精度: 用 VERDICT 标记而非模糊关键词匹配
    4. 并行处理: 同层多个断言并发质疑
    5. 实时输出: 每一步都打印进度
    6. 知识源: 第三模型作为公共查证工具，不做决策
    """

    def __init__(
        self,
        think_fn: Callable,
        challenge_fn: Callable,
        oracle_fn: Callable = None,   # 知识源（查证工具，不做决策）
        max_depth: int = 3,
        max_claims_per_level: int = 3,
        parallel: bool = True,
    ):
        self.think_fn = think_fn
        self.challenge_fn = challenge_fn
        self.oracle_fn = oracle_fn
        self.max_depth = max_depth
        self.max_claims_per_level = max_claims_per_level
        self.parallel = parallel

    async def deep_reason(self, question: str) -> DepthResult:
        """深度推敲主流程"""
        start_time = time.time()

        # Step 1: 初始回答
        print("  [1/4] 生成初始回答...")
        original_answer = await self.think_fn(question, {"role": "analyst"})
        print(f"  [1/4] 完成 ({len(original_answer)} 字)")

        all_claims: list[Claim] = []
        uncertainties: list[str] = []
        revisions = 0
        total_challenges = 0
        current_answer = original_answer
        max_depth_reached = 0

        # Step 2-5: 递归质疑循环
        for depth in range(self.max_depth):
            max_depth_reached = depth + 1
            depth_label = ["事实核查", "逻辑审查", "完备性审查"][min(depth, 2)]
            print(f"\n  [2/4] 深度 {depth} — {depth_label}")

            # 提取断言
            print(f"    提取断言...")
            claims = await self._extract_claims(current_answer, depth)
            if not claims:
                print(f"    未提取到断言，推敲结束")
                break

            print(f"    提取到 {len(claims)} 个断言:")
            for c in claims[:self.max_claims_per_level]:
                print(f"      • {c.content[:60]}")

            all_claims.extend(claims)
            claims_to_check = claims[:self.max_claims_per_level]

            # 质疑断言
            print(f"    开始质疑...")
            if self.parallel and len(claims_to_check) > 1:
                results = await self._challenge_parallel(
                    claims_to_check, current_answer, question, depth
                )
            else:
                results = []
                for claim in claims_to_check:
                    r = await self._challenge_single(
                        claim, current_answer, question, depth
                    )
                    results.append(r)

            total_challenges += len(results)

            # 处理质疑结果
            needs_revision = False
            failed_with_evidence = []
            for claim, (is_solid, detail) in zip(claims_to_check, results):
                claim.challenged = True
                claim.challenge_detail = detail

                if is_solid:
                    claim.survived = True
                    claim.confidence = 0.85
                    print(f"    ✓ 通过: {claim.content[:50]}")
                else:
                    # 质疑发现漏洞 → 调用知识源查证（如果可用）
                    oracle_evidence = None
                    if self.oracle_fn:
                        print(f"    ? 分歧: {claim.content[:50]}")
                        print(f"      → 查询知识源...")
                        oracle_evidence = await self._consult_oracle(
                            claim, detail, question
                        )
                        claim.challenge_detail = (
                            f"质疑: {detail[:300]}\n"
                            f"知识源补充: {oracle_evidence[:300]}"
                        )
                        print(f"      ← 知识源已回复 ({len(oracle_evidence)} 字)")

                    claim.survived = False
                    claim.confidence = 0.3
                    print(f"    ✗ 有漏洞: {claim.content[:50]}")
                    print(f"      原因: {detail[:80]}")
                    uncertainties.append(
                        f"[{depth_label}] {claim.content[:80]} → {detail[:150]}"
                    )
                    failed_with_evidence.append((claim, oracle_evidence))
                    needs_revision = True

            # 如果有断言被推翻，修正回答（带上知识源的补充信息）
            if needs_revision:
                print(f"    修正回答 (基于 {len(failed_with_evidence)} 个质疑)...")
                current_answer = await self._revise_with_evidence(
                    failed_with_evidence, current_answer, question
                )
                revisions += 1
                print(f"    修正完成")

        # Step 6: 综合最终回答
        print(f"\n  [3/4] 综合最终回答...")
        final_answer = await self._synthesize(
            question, original_answer, all_claims, uncertainties,
            total_challenges, revisions
        )
        print(f"  [4/4] 完成")

        elapsed = time.time() - start_time

        return DepthResult(
            original_answer=original_answer,
            final_answer=final_answer,
            claims=all_claims,
            max_depth_reached=max_depth_reached,
            uncertainties=uncertainties,
            revisions=revisions,
            total_challenges=total_challenges,
            elapsed_seconds=elapsed,
        )

    async def _extract_claims(self, answer: str, depth: int) -> list[Claim]:
        """从回答中提取断言 — 多策略 fallback"""
        # 如果回答太长，截取核心部分
        text = answer[:3000] if len(answer) > 3000 else answer

        prompt = EXTRACT_CLAIMS_PROMPT.format(answer=text)
        raw = await self.think_fn(prompt, {"role": "claim_extractor"})

        claims = self._parse_claims_json(raw, depth)

        # Fallback 1: 如果 JSON 解析失败，尝试用正则提取
        if not claims:
            claims = self._parse_claims_regex(raw, depth)

        # Fallback 2: 如果还是失败，按句子拆分
        if not claims:
            claims = self._parse_claims_sentences(text, depth)

        return claims

    def _parse_claims_json(self, raw: str, depth: int) -> list[Claim]:
        """策略1: 标准 JSON 解析"""
        claims = []
        try:
            # 去除 markdown 代码块标记
            cleaned = raw.replace("```json", "").replace("```", "").strip()

            # 找到 JSON 数组
            start = cleaned.find("[")
            end = cleaned.rfind("]") + 1
            if start >= 0 and end > start:
                json_str = cleaned[start:end]
                # 修复常见的 JSON 问题
                json_str = json_str.replace("\n", " ")
                json_str = re.sub(r',\s*]', ']', json_str)  # 去除尾逗号
                json_str = re.sub(r',\s*}', '}', json_str)

                data = json.loads(json_str)
                for i, item in enumerate(data):
                    if isinstance(item, dict):
                        content = item.get("claim", item.get("断言", ""))
                        basis = item.get("basis", item.get("依据", ""))
                        if content:
                            claims.append(Claim(
                                id=depth * 100 + i,
                                content=content,
                                basis=basis,
                                depth=depth,
                            ))
        except (json.JSONDecodeError, ValueError, TypeError):
            pass
        return claims

    def _parse_claims_regex(self, raw: str, depth: int) -> list[Claim]:
        """策略2: 正则提取 — 匹配 claim/断言 相关的模式"""
        claims = []
        # 匹配 "claim": "..." 或 断言: ...
        patterns = [
            r'"claim"\s*:\s*"([^"]+)"',
            r'断言[：:]\s*(.+?)(?:\n|$)',
            r'\d+\.\s*(.+?)(?:\n|$)',  # 编号列表
            r'[-•]\s*(.+?)(?:\n|$)',   # 无序列表
        ]
        for pattern in patterns:
            matches = re.findall(pattern, raw)
            if matches and len(matches) >= 2:
                for i, match in enumerate(matches[:5]):
                    content = match.strip()
                    if len(content) > 10:  # 过滤太短的
                        claims.append(Claim(
                            id=depth * 100 + i,
                            content=content,
                            basis="从回答中提取",
                            depth=depth,
                        ))
                break
        return claims

    def _parse_claims_sentences(self, text: str, depth: int) -> list[Claim]:
        """策略3: 按句子拆分 — 最后的 fallback"""
        # 取前几个有实质内容的句子作为断言
        sentences = re.split(r'[。.！!？?\n]', text)
        claims = []
        for i, s in enumerate(sentences):
            s = s.strip()
            if len(s) > 20 and len(claims) < 3:
                claims.append(Claim(
                    id=depth * 100 + i,
                    content=s[:200],
                    basis="原文陈述",
                    depth=depth,
                ))
        return claims

    async def _challenge_single(self, claim: Claim, context: str,
                                question: str, depth: int) -> tuple[bool, str]:
        """质疑单个断言"""
        prompt_template = CHALLENGE_PROMPT_BY_DEPTH[min(depth, 2)]
        prompt = prompt_template.format(
            claim=claim.content,
            basis=claim.basis,
            context=f"问题: {question}\n回答摘要: {context[:800]}",
        )
        result = await self.challenge_fn(prompt, {"role": "challenger"})
        is_solid = self._parse_verdict(result)
        return (is_solid, result)

    async def _challenge_parallel(self, claims: list[Claim], context: str,
                                  question: str, depth: int) -> list[tuple[bool, str]]:
        """并行质疑多个断言"""
        tasks = [
            self._challenge_single(claim, context, question, depth)
            for claim in claims
        ]
        return await asyncio.gather(*tasks)

    def _parse_verdict(self, result: str) -> bool:
        """解析质疑结果的判定"""
        result_upper = result.upper()

        # 优先看 VERDICT 标记
        if "VERDICT: SOLID" in result_upper or "VERDICT:SOLID" in result_upper:
            return True
        if "VERDICT: FLAWED" in result_upper or "VERDICT:FLAWED" in result_upper:
            return False

        # Fallback: 关键词判断
        solid_keywords = ["SOLID", "成立", "经得起", "没有漏洞", "逻辑正确",
                          "事实准确", "推理严密", "考虑充分"]
        flaw_keywords = ["FLAWED", "漏洞", "错误", "不成立", "过度简化",
                         "遗漏", "不准确", "逻辑跳跃", "隐含假设"]

        solid_count = sum(1 for kw in solid_keywords if kw in result)
        flaw_count = sum(1 for kw in flaw_keywords if kw in result)

        if flaw_count > solid_count:
            return False
        return True  # 默认通过（宁可漏判也不误杀）

    async def _revise_batch(self, failed_claims: list[Claim],
                            current_answer: str, question: str) -> str:
        """批量修正 — 将所有失败的质疑合并为一次修正请求"""
        challenges_summary = "\n".join(
            f"- 断言「{c.content[:60]}」的问题: {c.challenge_detail[:100]}"
            for c in failed_claims
        )

        prompt = f"""以下断言在质疑中暴露了问题:

{challenges_summary}

原始问题: {question}
当前回答摘要: {current_answer[:1500]}

请针对以上问题修正你的分析。只输出修正和补充的部分。"""

        return await self.think_fn(prompt, {"role": "reviser"})

    async def _consult_oracle(self, claim: Claim, challenge_detail: str,
                              question: str) -> str:
        """
        查询知识源 — Claude 作为公共图书馆

        不做决策，只提供:
        - 相关事实
        - 逻辑关系验证
        - 补充信息

        思考者和质疑者各自用这些信息重新判断。
        """
        prompt = f"""你是一个知识源。不做判断，只提供事实和信息。

当前有一个分歧:
- 断言: {claim.content}
- 依据: {claim.basis}
- 质疑: {challenge_detail[:500]}

请提供:
1. 与这个断言相关的客观事实（如有）
2. 这个领域的主流认知是什么
3. 是否存在被忽略的重要背景信息

不要说"我认为断言对/错"，只提供信息让双方自行判断。"""

        return await self.oracle_fn(prompt, {"role": "knowledge_source"})

    async def _revise_with_evidence(self, failed_with_evidence: list,
                                    current_answer: str, question: str) -> str:
        """带知识源证据的修正"""
        parts = []
        for claim, evidence in failed_with_evidence:
            part = f"- 断言「{claim.content[:60]}」的问题: {claim.challenge_detail[:150]}"
            if evidence:
                part += f"\n  知识源补充: {evidence[:200]}"
            parts.append(part)

        challenges_summary = "\n".join(parts)

        prompt = f"""以下断言在质疑中暴露了问题，知识源也提供了补充信息:

{challenges_summary}

原始问题: {question}
当前回答摘要: {current_answer[:1500]}

请综合质疑意见和知识源提供的信息，修正你的分析。
只输出修正和补充的部分。"""

        return await self.think_fn(prompt, {"role": "reviser"})

    async def _synthesize(self, question: str, original_answer: str,
                          claims: list[Claim], uncertainties: list[str],
                          total_challenges: int, revisions: int) -> str:
        """综合推敲结果"""
        survived = [c for c in claims if c.survived]
        failed = [c for c in claims if c.challenged and not c.survived]

        survived_text = "\n".join(
            f"  ✓ {c.content} (依据: {c.basis})" for c in survived
        ) or "  (无)"

        uncertain_text = "\n".join(
            f"  ✗ {c.content} → {c.challenge_detail[:100]}" for c in failed
        ) or "  (无)"

        prompt = SYNTHESIZE_PROMPT.format(
            question=question,
            total_challenges=total_challenges,
            revisions=revisions,
            survived_claims=survived_text,
            uncertain_claims=uncertain_text,
        )
        return await self.think_fn(prompt, {"role": "synthesizer"})
