"""
argument_graph.py — 论证图引擎 (进阶版深度推敲)

核心区别:
- 旧版: 线性断言列表 → 逐个质疑 → 修正
- 新版: 构建因果论证图 → 追溯到源头 → 找崩塌条件 → 传播影响

一个论证不是一句话，而是一棵树:
  结论 ← 中间推理 ← 前提 ← 事实/假设

推敲不是问"对不对"，而是:
1. 追溯: 这个结论依赖哪些前提？前提又依赖什么？一直追到源头。
2. 攻击: 每个节点在什么条件下会崩塌？
3. 传播: 如果底层节点崩塌，哪些上层结论会跟着倒？
4. 加固: 崩塌条件在现实中是否可能发生？如果可能，结论需要修正。
"""

from dataclasses import dataclass, field
from typing import Callable, Optional
from enum import Enum
import json
import asyncio
import time

from .entropy import EntropyVector, EntropyCalculator
from .state_machine import StateMachine, ThinkingState
from . import prompts


class NodeType(Enum):
    FACT = "fact"               # 可验证的事实 (触底节点)
    ASSUMPTION = "assumption"   # 假设 (未验证，可能是漏洞)
    INFERENCE = "inference"     # 推理 (从前提推出)
    CONCLUSION = "conclusion"   # 结论 (最终输出)


class NodeStatus(Enum):
    STANDING = "standing"       # 成立
    CHALLENGED = "challenged"   # 被质疑，待验证
    COLLAPSED = "collapsed"     # 已崩塌
    FORTIFIED = "fortified"     # 经受住攻击，加固


@dataclass
class ArgumentNode:
    """论证图中的一个节点"""
    id: str
    content: str                        # 这个节点的陈述
    node_type: NodeType                 # 类型
    status: NodeStatus = NodeStatus.STANDING
    supports: list[str] = field(default_factory=list)    # 支撑它的下游节点 id
    supported_by: list[str] = field(default_factory=list) # 它支撑的上游节点 id
    collapse_conditions: list[str] = field(default_factory=list)  # 在什么条件下崩塌
    evidence: str = ""                  # 支撑证据
    attack_result: str = ""             # 攻击结果


@dataclass
class GraphResult:
    """论证图推敲结果"""
    nodes: dict[str, ArgumentNode]
    root_conclusion: str
    collapsed_nodes: list[str]
    fortified_nodes: list[str]
    critical_assumptions: list[str]
    revised_conclusion: str
    total_nodes: int
    attack_count: int
    elapsed_seconds: float
    entropy: EntropyVector = field(default_factory=EntropyVector)  # 最终熵状态


# === 提示词 ===

BUILD_GRAPH_PROMPT = None  # Moved to prompts.py
FIND_COLLAPSE_PROMPT = None
VERIFY_CONDITION_PROMPT = None
PROPAGATE_PROMPT = None


class ArgumentGraphEngine:
    """
    论证图推敲引擎

    流程:
    1. 构建: 将回答分解为论证图 (节点 + 因果关系)
    2. 追溯: 找到所有叶子节点 (事实和假设)
    3. 攻击: 对每个假设节点找崩塌条件
    4. 验证: 用知识源验证崩塌条件是否成立
    5. 传播: 如果底层崩塌，计算上层影响
    6. 修正: 基于崩塌传播结果修正最终结论
    """

    def __init__(
        self,
        think_fn: Callable,       # 思考者 (构建论证、修正)
        attack_fn: Callable,      # 攻击者 (找崩塌条件)
        verify_fn: Callable,      # 验证者 (查证事实)
    ):
        self.think_fn = think_fn
        self.attack_fn = attack_fn
        self.verify_fn = verify_fn
        self.state = StateMachine()
        self.state.register("thinker", "构建/修正")
        self.state.register("attacker", "对抗攻击")
        self.state.register("verifier", "事实验证")

    async def reason(self, question: str) -> GraphResult:
        """完整推敲流程"""
        start_time = time.time()
        self.state.start_session()

        # Step 1: 生成初始回答
        print("  [1/6] 生成初始论述...")
        self.state.transition("thinker", ThinkingState.THINKING, task="生成初始论述")
        answer = await self.think_fn(
            prompts.INITIAL_ANALYSIS.format(question=question), {}
        )
        self.state.transition("thinker", ThinkingState.DONE, reason=f"完成 ({len(answer)} 字)")
        print(f"  [1/6] 完成 ({len(answer)} 字)")

        # Step 2: 构建论证图
        print("  [2/6] 构建论证图...")
        self.state.transition("thinker", ThinkingState.THINKING, task="分解论证结构")
        nodes, root_id = await self._build_graph(answer)
        self.state.transition("thinker", ThinkingState.DONE, reason=f"{len(nodes)} 个节点")
        print(f"  [2/6] 完成 ({len(nodes)} 个节点)")
        for nid, node in nodes.items():
            print(f"    [{node.node_type.value}] {nid}: {node.content[:60]}")

        # Step 3: 找到可攻击的节点 (假设和推理)
        attackable = [
            n for n in nodes.values()
            if n.node_type in (NodeType.ASSUMPTION, NodeType.INFERENCE)
        ]
        print(f"\n  [3/6] 寻找崩塌条件 ({len(attackable)} 个可攻击节点)...")

        # 计算初始熵，决定攻击策略
        initial_entropy = EntropyCalculator.calculate(nodes, [])
        print(f"    初始熵: {initial_entropy}")

        attack_count = 0
        attack_results = {}
        for node in attackable:
            # 熵驱动: 依赖熵高的节点优先攻击
            supported = [
                nodes[sid].content[:50] for sid in node.supported_by
                if sid in nodes
            ]
            self.state.transition("attacker", ThinkingState.THINKING, task=f"攻击 {node.id}")
            conditions = await self._find_collapse_conditions(node, supported)
            self.state.transition("attacker", ThinkingState.DONE, reason=f"{len(conditions)} 个条件")
            node.collapse_conditions = conditions
            attack_count += 1
            attack_results[node.id] = (True, "")  # 暂时标记为通过
            print(f"    攻击 {node.id}: 找到 {len(conditions)} 个崩塌条件")
            for c in conditions:
                print(f"      • {c[:70]}")

        # Step 4: 验证崩塌条件 (熵驱动: 传播熵高的优先验证)
        print(f"\n  [4/6] 验证崩塌条件...")

        # 按传播影响排序: 支撑更多上游节点的优先验证
        attackable_sorted = sorted(
            attackable,
            key=lambda n: len(n.supports),
            reverse=True
        )

        collapsed_nodes = []
        for node in attackable_sorted:
            node_collapsed = False
            for condition in node.collapse_conditions:
                # 所有崩塌条件都验证（不再靠关键词过滤）
                self.state.transition("verifier", ThinkingState.THINKING, task=f"验证 {node.id}")
                is_real = await self._verify_condition(node, condition)
                self.state.transition("verifier", ThinkingState.DONE,
                                      reason="崩塌确认" if is_real else "条件不成立")
                if is_real:
                        node.status = NodeStatus.COLLAPSED
                        node.attack_result = condition
                        collapsed_nodes.append(node.id)
                        attack_results[node.id] = (False, condition)
                        print(f"    ✗ {node.id} 崩塌: {condition[:60]}")
                        node_collapsed = True
                        break
            if not node_collapsed:
                node.status = NodeStatus.FORTIFIED
                attack_results[node.id] = (True, "加固")
                print(f"    ✓ {node.id} 加固: 崩塌条件不成立")

            # 动态熵检查: 每次验证后重新计算，决定是否继续
            current_entropy = EntropyCalculator.calculate(nodes, collapsed_nodes, attack_results)
            if current_entropy.can_converge:
                print(f"    [熵收敛] 不确定性已足够低，跳过剩余节点")
                break

        # Step 5: 传播崩塌影响
        revised_conclusion = ""
        if collapsed_nodes:
            print(f"\n  [5/6] 传播崩塌影响 ({len(collapsed_nodes)} 个节点崩塌)...")
            revised_conclusion = await self._propagate_collapse(
                nodes, collapsed_nodes, root_id
            )
            print(f"  [5/6] 完成")
        else:
            print(f"\n  [5/6] 无节点崩塌，结论成立")
            root_node = nodes.get(root_id)
            revised_conclusion = root_node.content if root_node else answer

        # Step 6: 输出
        print(f"  [6/6] 推敲完成")

        elapsed = time.time() - start_time
        fortified = [n.id for n in nodes.values() if n.status == NodeStatus.FORTIFIED]
        critical = [
            n.id for n in nodes.values()
            if n.node_type == NodeType.ASSUMPTION and n.status != NodeStatus.FORTIFIED
        ]

        # 最终熵计算
        final_entropy = EntropyCalculator.calculate(nodes, collapsed_nodes, attack_results)
        print(f"    最终熵: {final_entropy}")

        return GraphResult(
            nodes=nodes,
            root_conclusion=root_id,
            collapsed_nodes=collapsed_nodes,
            fortified_nodes=fortified,
            critical_assumptions=critical,
            revised_conclusion=revised_conclusion,
            total_nodes=len(nodes),
            attack_count=attack_count,
            elapsed_seconds=elapsed,
            entropy=final_entropy,
        )

    async def _build_graph(self, answer: str) -> tuple[dict[str, ArgumentNode], str]:
        """将回答分解为论证图"""
        prompt = prompts.DECOMPOSE_ARGUMENT.format(answer=answer[:3000])
        raw = await self.think_fn(prompt, {})

        nodes = {}
        root_id = "N1"

        try:
            # 清理并解析 JSON
            cleaned = raw.replace("```json", "").replace("```", "").strip()
            start = cleaned.find("{")
            end = cleaned.rfind("}") + 1
            if start >= 0 and end > start:
                data = json.loads(cleaned[start:end])
                root_id = data.get("root", "N1")

                for item in data.get("nodes", []):
                    nid = item["id"]
                    node_type = NodeType(item.get("type", "inference"))
                    supports = item.get("supports", [])

                    node = ArgumentNode(
                        id=nid,
                        content=item.get("content", ""),
                        node_type=node_type,
                        supports=supports,
                    )
                    nodes[nid] = node

                # 建立反向引用
                for nid, node in nodes.items():
                    for supported_id in node.supports:
                        if supported_id in nodes:
                            nodes[supported_id].supported_by.append(nid)

        except (json.JSONDecodeError, KeyError, ValueError) as e:
            # Fallback: 把整个回答作为单节点
            nodes["N1"] = ArgumentNode(
                id="N1",
                content=answer[:500],
                node_type=NodeType.CONCLUSION,
            )
            root_id = "N1"

        return nodes, root_id

    async def _find_collapse_conditions(
        self, node: ArgumentNode, supported_conclusions: list[str]
    ) -> list[str]:
        """找出节点的崩塌条件"""
        prompt = prompts.FIND_COLLAPSE.format(
            content=node.content,
            node_type=node.node_type.value,
            supported_conclusions="\n".join(f"  - {c}" for c in supported_conclusions) or "(无)",
        )
        raw = await self.attack_fn(prompt, {})

        # 解析条件
        conditions = []
        for line in raw.split("\n"):
            line = line.strip()
            if line.startswith("CONDITION") and ":" in line:
                # CONDITION_1: xxx | LIKELIHOOD: 高
                parts = line.split(":", 1)[1].strip()
                conditions.append(parts)
            elif "|" in line and ("高" in line or "中" in line or "低" in line):
                conditions.append(line.strip())

        # Fallback: 如果没解析到，取前3行非空内容
        if not conditions:
            lines = [l.strip() for l in raw.split("\n") if l.strip() and len(l.strip()) > 15]
            conditions = lines[:3]

        return conditions[:3]

    async def _verify_condition(self, node: ArgumentNode, condition: str) -> bool:
        """验证崩塌条件是否在现实中成立"""
        prompt = prompts.VERIFY_CONDITION.format(
            node_content=node.content,
            condition=condition,
        )
        raw = await self.verify_fn(prompt, {})

        raw_upper = raw.upper()
        if "VERIFIED: YES" in raw_upper or "VERIFIED:YES" in raw_upper:
            return True
        if "VERIFIED: NO" in raw_upper or "VERIFIED:NO" in raw_upper:
            return False
        # UNCERTAIN 视为不崩塌（保守策略）
        return False

    async def _propagate_collapse(
        self, nodes: dict[str, ArgumentNode],
        collapsed_ids: list[str], root_id: str
    ) -> str:
        """传播崩塌影响并修正结论"""
        collapsed_info = []
        for cid in collapsed_ids:
            node = nodes[cid]
            collapsed_info.append(f"{cid}: {node.content} (原因: {node.attack_result[:100]})")

        dependent_info = []
        for cid in collapsed_ids:
            node = nodes[cid]
            for sup_id in node.supports:
                if sup_id in nodes:
                    dependent_info.append(f"{sup_id}: {nodes[sup_id].content[:80]}")

        prompt = prompts.PROPAGATE_IMPACT.format(
            collapsed_node="\n".join(collapsed_info),
            collapse_reason="见上",
            dependent_nodes="\n".join(dependent_info) or "(无直接依赖)",
        )
        raw = await self.think_fn(prompt, {})

        # 提取修正后的结论
        for line in raw.split("\n"):
            if line.strip().startswith("REVISED_CONCLUSION:"):
                return line.split(":", 1)[1].strip()

        # Fallback: 返回完整分析
        return raw
