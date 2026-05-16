"""
entropy.py — 动态熵引擎 v3

10 个维度的不确定性度量，驱动系统的资源分配和收敛决策。

每个熵值 0-1:
  0 = 完全确定
  1 = 完全混沌

v3 变化:
- 新增影响熵 (impact): 结论落地后对现实世界的影响不确定性
- 优化计算逻辑: 更精确的量化方法
- 熵之间的耦合关系: 某些熵高会放大其他熵的权重
"""

from dataclasses import dataclass
import math


@dataclass
class EntropyVector:
    """10 维动态熵向量"""

    semantic: float = 0.0       # 语义熵: 模型间理解是否一致
    causal: float = 0.0         # 因果熵: 推理路径是否唯一确定
    boundary: float = 0.0       # 边界熵: 结论适用范围是否清晰
    temporal: float = 0.0       # 时序熵: 结论对时间的敏感度
    dependency: float = 0.0     # 依赖熵: 依赖多少未验证假设
    divergence: float = 0.0     # 分歧熵: 模型间分歧程度
    information: float = 0.0    # 信息熵: 缺少多少关键信息
    propagation: float = 0.0    # 传播熵: 逻辑错误的扩散范围
    evidence: float = 0.0       # 证据熵: 已有证据的可靠性
    impact: float = 0.0         # 影响熵: 被攻击后的自证难度和连锁震动

    @property
    def total(self) -> float:
        """
        加权总熵 — 不是简单均值

        影响熵高时，其他所有熵的权重都被放大。
        因为: 如果一个决策影响巨大，那么任何不确定性都更危险。
        """
        values = [
            self.semantic, self.causal, self.boundary, self.temporal,
            self.dependency, self.divergence, self.information,
            self.propagation, self.evidence, self.impact
        ]

        # 基础均值
        base = sum(values) / len(values)

        # 影响放大因子: impact 越高，总熵越被放大
        # impact=0 → 放大1.0x; impact=1 → 放大1.5x
        amplifier = 1.0 + self.impact * 0.5

        return min(1.0, base * amplifier)

    @property
    def max_entropy(self) -> tuple[str, float]:
        """最高熵的维度和值"""
        dims = self._all_dims()
        name = max(dims, key=dims.get)
        return name, dims[name]

    @property
    def top3_entropy(self) -> list[tuple[str, float]]:
        """前3高的熵维度"""
        dims = self._all_dims()
        sorted_dims = sorted(dims.items(), key=lambda x: x[1], reverse=True)
        return sorted_dims[:3]

    @property
    def should_deepen(self) -> bool:
        """是否需要更深的推敲"""
        return self.total > 0.5 or self.max_entropy[1] > 0.7

    @property
    def can_converge(self) -> bool:
        """是否可以收敛"""
        values = list(self._all_dims().values())
        # 所有维度 < 0.4 且总熵 < 0.3
        return all(v < 0.4 for v in values) and self.total < 0.3

    @property
    def needs_oracle(self) -> bool:
        """是否需要调用知识源"""
        return self.information > 0.6 or self.temporal > 0.7 or self.evidence > 0.6

    @property
    def needs_stronger_model(self) -> bool:
        """是否需要调用更强的模型"""
        return self.causal > 0.7 or self.dependency > 0.7

    @property
    def needs_external_evidence(self) -> bool:
        """是否需要搜索外部证据"""
        return self.evidence > 0.5

    @property
    def is_high_stakes(self) -> bool:
        """是否是高风险决策（影响熵高）"""
        return self.impact > 0.6

    @property
    def risk_level(self) -> str:
        """风险等级"""
        if self.impact < 0.3:
            return "low"
        elif self.impact < 0.6:
            return "medium"
        else:
            return "high"

    def _all_dims(self) -> dict[str, float]:
        return {
            "semantic": self.semantic,
            "causal": self.causal,
            "boundary": self.boundary,
            "temporal": self.temporal,
            "dependency": self.dependency,
            "divergence": self.divergence,
            "information": self.information,
            "propagation": self.propagation,
            "evidence": self.evidence,
            "impact": self.impact,
        }

    def to_dict(self) -> dict:
        d = {k: round(v, 3) for k, v in self._all_dims().items()}
        d["total"] = round(self.total, 3)
        d["risk_level"] = self.risk_level
        return d

    def delta(self, other: "EntropyVector") -> "EntropyVector":
        """计算两个熵向量的差值（用于检测变化趋势）"""
        return EntropyVector(
            semantic=self.semantic - other.semantic,
            causal=self.causal - other.causal,
            boundary=self.boundary - other.boundary,
            temporal=self.temporal - other.temporal,
            dependency=self.dependency - other.dependency,
            divergence=self.divergence - other.divergence,
            information=self.information - other.information,
            propagation=self.propagation - other.propagation,
            evidence=self.evidence - other.evidence,
            impact=self.impact - other.impact,
        )

    def __repr__(self):
        max_name, max_val = self.max_entropy
        return (
            f"Entropy(total={self.total:.2f}, max={max_name}:{max_val:.2f}, "
            f"risk={self.risk_level}, converge={'✓' if self.can_converge else '✗'})"
        )


class EntropyCalculator:
    """
    熵计算器 — 从论证图或对话历史中计算 10 维熵

    不依赖模型，纯算法计算。
    """

    @staticmethod
    def from_graph(nodes: dict, collapsed_ids: list,
                   attack_results: dict = None) -> EntropyVector:
        """从论证图状态计算熵"""
        if not nodes:
            return EntropyVector()

        attack_results = attack_results or {}
        total_nodes = len(nodes)

        semantic = EntropyCalculator._calc_semantic(nodes)
        causal = EntropyCalculator._calc_causal(nodes)
        boundary = EntropyCalculator._calc_boundary(nodes)
        temporal = EntropyCalculator._calc_temporal(nodes)
        dependency = EntropyCalculator._calc_dependency(nodes, collapsed_ids)
        divergence = EntropyCalculator._calc_divergence(attack_results)
        information = EntropyCalculator._calc_information(nodes)
        propagation = EntropyCalculator._calc_propagation(nodes, collapsed_ids)
        evidence = EntropyCalculator._calc_evidence(nodes)
        impact = EntropyCalculator._calc_impact(nodes)

        return EntropyVector(
            semantic=semantic,
            causal=causal,
            boundary=boundary,
            temporal=temporal,
            dependency=dependency,
            divergence=divergence,
            information=information,
            propagation=propagation,
            evidence=evidence,
            impact=impact,
        )

    @staticmethod
    def from_conversation(utterances: list, round_num: int = 0) -> EntropyVector:
        """从对话历史计算熵（用于 Arena 模式）"""
        if not utterances:
            return EntropyVector()

        contents = [u.content for u in utterances]
        all_text = " ".join(contents)

        # 分歧熵
        last_round = max(u.round for u in utterances)
        last_utterances = [u for u in utterances if u.round == last_round]
        divergence = EntropyCalculator._calc_conversation_divergence(last_utterances)

        # 时序熵 (观点稳定性)
        temporal = EntropyCalculator._calc_conversation_temporal(utterances, last_round)

        # 信息熵
        information = EntropyCalculator._calc_text_information(all_text)

        # 证据熵
        evidence = EntropyCalculator._calc_text_evidence(all_text)

        # 边界熵
        boundary = EntropyCalculator._calc_text_boundary(all_text)

        # 影响熵 (对话中: 攻击频率和强度)
        impact = EntropyCalculator._calc_conversation_impact(utterances)

        # 语义熵 (和分歧相关)
        semantic = divergence * 0.6

        return EntropyVector(
            semantic=semantic,
            causal=0.3,  # 自由对话中因果较难量化
            boundary=boundary,
            temporal=temporal,
            dependency=0.2,
            divergence=divergence,
            information=information,
            propagation=0.0,  # 无图结构
            evidence=evidence,
            impact=impact,
        )

    # === 论证图模式的计算方法 ===

    @staticmethod
    def _calc_semantic(nodes: dict) -> float:
        """语义熵: 节点间内容重叠度"""
        contents = [n.content.lower() for n in nodes.values()]
        if len(contents) < 2:
            return 0.0

        overlap_scores = []
        for i in range(len(contents)):
            for j in range(i + 1, len(contents)):
                words_i = set(contents[i])
                words_j = set(contents[j])
                if words_i and words_j:
                    overlap = len(words_i & words_j) / len(words_i | words_j)
                    overlap_scores.append(overlap)

        if not overlap_scores:
            return 0.0
        avg_overlap = sum(overlap_scores) / len(overlap_scores)
        return min(1.0, avg_overlap * 1.5)

    @staticmethod
    def _calc_causal(nodes: dict) -> float:
        """因果熵: 结论的支撑路径数"""
        conclusion_nodes = [n for n in nodes.values() if n.node_type.value == "conclusion"]
        if not conclusion_nodes:
            return 0.5

        total_support = sum(len(cn.supported_by) for cn in conclusion_nodes)
        avg_support = total_support / len(conclusion_nodes)

        if avg_support >= 4:
            return 0.1
        elif avg_support >= 2:
            return 0.3
        elif avg_support >= 1:
            return 0.6
        else:
            return 0.9

    @staticmethod
    def _calc_boundary(nodes: dict) -> float:
        """边界熵: 限定词 vs 绝对词"""
        all_content = " ".join(n.content for n in nodes.values())
        return EntropyCalculator._calc_text_boundary(all_content)

    @staticmethod
    def _calc_temporal(nodes: dict) -> float:
        """时序熵: 时间敏感词密度"""
        all_content = " ".join(n.content for n in nodes.values())
        time_words = ["目前", "当前", "现在", "2024", "2025", "2026",
                      "最近", "近期", "未来", "即将", "趋势", "预计"]
        time_count = sum(1 for w in time_words if w in all_content)
        return min(1.0, time_count * 0.1)

    @staticmethod
    def _calc_dependency(nodes: dict, collapsed_ids: list) -> float:
        """依赖熵: 假设占比 + 崩塌占比"""
        total = len(nodes)
        if total == 0:
            return 0.0

        assumption_count = sum(1 for n in nodes.values() if n.node_type.value == "assumption")
        collapsed_count = len(collapsed_ids)

        assumption_ratio = assumption_count / total
        collapsed_ratio = collapsed_count / total

        return min(1.0, assumption_ratio * 0.6 + collapsed_ratio * 0.8)

    @staticmethod
    def _calc_divergence(attack_results: dict) -> float:
        """分歧熵: 二元熵"""
        if not attack_results:
            return 0.5

        flawed_count = sum(1 for is_solid, _ in attack_results.values() if not is_solid)
        total = len(attack_results)

        ratio = flawed_count / total
        if ratio == 0 or ratio == 1:
            return 0.1
        entropy = -(ratio * math.log2(ratio) + (1 - ratio) * math.log2(1 - ratio))
        return entropy

    @staticmethod
    def _calc_information(nodes: dict) -> float:
        """信息熵: 不确定性标记密度"""
        all_content = " ".join(n.content for n in nodes.values())
        return EntropyCalculator._calc_text_information(all_content)

    @staticmethod
    def _calc_propagation(nodes: dict, collapsed_ids: list) -> float:
        """传播熵: 崩塌影响范围"""
        if not collapsed_ids or not nodes:
            return 0.0

        affected = set()
        queue = list(collapsed_ids)
        while queue:
            nid = queue.pop(0)
            if nid in nodes:
                for sup_id in nodes[nid].supports:
                    if sup_id not in affected:
                        affected.add(sup_id)
                        queue.append(sup_id)

        total = len(nodes)
        return len(affected) / total if total > 0 else 0.0

    @staticmethod
    def _calc_evidence(nodes: dict) -> float:
        """证据熵: 证据可靠性"""
        all_content = " ".join(n.content for n in nodes.values())
        base = EntropyCalculator._calc_text_evidence(all_content)

        # fact 节点占比越高，证据熵越低
        fact_ratio = sum(1 for n in nodes.values() if n.node_type.value == "fact") / len(nodes)
        return base * (1.0 - fact_ratio * 0.4)

    @staticmethod
    def _calc_impact(nodes: dict) -> float:
        """
        影响熵: 被攻击后的自证难度和连锁震动

        衡量: 如果某个节点被攻击，整个论证链受到多大震动，
        以及恢复（自证）需要多高的证据成本。

        计算因素:
        1. 关键节点集中度: 是否存在"单点故障"节点（被很多结论依赖）
        2. 替代路径: 被攻击节点有没有其他支撑可以兜底
        3. 假设脆弱性: 假设节点被攻击时，自证难度天然更高
        """
        if not nodes:
            return 0.0

        total = len(nodes)

        # 1. 单点故障检测: 找到被最多上游依赖的节点
        max_supports = 0
        for node in nodes.values():
            support_count = len(node.supports)  # 它支撑了多少上游
            max_supports = max(max_supports, support_count)

        # 如果有节点支撑了 3+ 个上游结论，它一旦被攻击影响巨大
        single_point_risk = min(1.0, max_supports * 0.25)

        # 2. 替代路径: 结论节点是否有多条独立支撑
        conclusion_nodes = [n for n in nodes.values() if n.node_type.value == "conclusion"]
        redundancy = 0.0
        if conclusion_nodes:
            avg_support = sum(len(cn.supported_by) for cn in conclusion_nodes) / len(conclusion_nodes)
            # 支撑越多 → 替代路径越多 → 影响熵越低
            if avg_support >= 4:
                redundancy = 0.1
            elif avg_support >= 2:
                redundancy = 0.3
            elif avg_support >= 1:
                redundancy = 0.6
            else:
                redundancy = 0.9

        # 3. 假设脆弱性: 假设节点占比越高，被攻击时自证越难
        assumption_count = sum(1 for n in nodes.values() if n.node_type.value == "assumption")
        assumption_ratio = assumption_count / total
        # 假设多 → 自证需要更多外部证据 → 影响熵高
        assumption_fragility = assumption_ratio * 0.8

        # 4. 已崩塌节点的震动: 如果已经有节点崩塌了，说明攻击已经生效
        collapsed_count = sum(1 for n in nodes.values()
                             if hasattr(n, 'status') and
                             hasattr(n.status, 'value') and
                             n.status.value == "collapsed")
        collapse_shock = min(1.0, collapsed_count * 0.3)

        # 综合: 取各因素的加权
        impact = (
            single_point_risk * 0.3 +
            redundancy * 0.25 +
            assumption_fragility * 0.25 +
            collapse_shock * 0.2
        )

        return min(1.0, impact)

    # === 通用文本分析方法 (论证图和对话共用) ===

    @staticmethod
    def _calc_text_boundary(text: str) -> float:
        """从文本计算边界熵"""
        boundary_words = ["如果", "当", "在...条件下", "仅限", "部分", "某些",
                          "通常", "大多数", "一般", "特定", "有限"]
        absolute_words = ["所有", "一定", "必然", "绝对", "完全", "永远", "从不"]

        b_count = sum(1 for w in boundary_words if w in text)
        a_count = sum(1 for w in absolute_words if w in text)

        if b_count > a_count:
            return max(0.1, 0.5 - b_count * 0.04)
        elif a_count > 0:
            return min(0.9, 0.5 + a_count * 0.12)
        return 0.5

    @staticmethod
    def _calc_text_information(text: str) -> float:
        """从文本计算信息熵"""
        uncertain_words = ["不确定", "未知", "缺乏数据", "难以判断", "有待验证",
                           "可能", "也许", "或许", "尚不清楚", "证据不足"]
        count = sum(1 for w in uncertain_words if w in text)
        return min(1.0, count * 0.08)

    @staticmethod
    def _calc_text_evidence(text: str) -> float:
        """从文本计算证据熵"""
        strong = ["数据显示", "研究表明", "根据", "报告指出", "实验证明",
                  "统计", "%", "亿", "万", "论文", "调查"]
        weak = ["普遍认为", "大家都说", "一般来说", "通常认为",
                "据说", "有人认为", "理论上"]

        strong_count = sum(1 for w in strong if w in text)
        weak_count = sum(1 for w in weak if w in text)

        if strong_count > weak_count:
            return max(0.1, 0.4 - strong_count * 0.05)
        elif weak_count > strong_count:
            return min(0.8, 0.4 + weak_count * 0.1)
        return 0.4

    @staticmethod
    def _calc_text_impact(text: str) -> float:
        """
        从文本计算影响熵

        高影响指标: 涉及大规模、不可逆、长期、高成本
        低影响指标: 个人、可逆、短期、低成本
        """
        high_impact = [
            "国家", "全球", "人类", "社会", "经济", "政策",
            "万亿", "数十亿", "百万", "千万",
            "不可逆", "永久", "根本性", "颠覆",
            "战略", "安全", "生死", "存亡",
            "未来几十年", "长期", "代际",
            "基础设施", "能源", "教育", "医疗",
        ]
        low_impact = [
            "个人", "自己", "小范围", "局部",
            "可以撤回", "可逆", "临时", "短期",
            "尝试", "实验", "测试",
        ]

        high_count = sum(1 for w in high_impact if w in text)
        low_count = sum(1 for w in low_impact if w in text)

        if high_count > low_count + 3:
            return min(0.9, 0.3 + high_count * 0.06)
        elif low_count > high_count:
            return max(0.1, 0.3 - low_count * 0.05)
        else:
            return 0.3 + (high_count - low_count) * 0.04

    # === 对话模式专用 ===

    @staticmethod
    def _calc_conversation_divergence(last_utterances: list) -> float:
        """对话中的分歧熵"""
        if len(last_utterances) < 2:
            return 0.5

        texts = [u.content.lower() for u in last_utterances]
        overlaps = []
        for i in range(len(texts)):
            for j in range(i + 1, len(texts)):
                chars_i = set(texts[i])
                chars_j = set(texts[j])
                if chars_i and chars_j:
                    overlap = len(chars_i & chars_j) / len(chars_i | chars_j)
                    overlaps.append(overlap)

        if not overlaps:
            return 0.5
        avg_overlap = sum(overlaps) / len(overlaps)
        return 1.0 - avg_overlap

    @staticmethod
    def _calc_conversation_temporal(utterances: list, last_round: int) -> float:
        """对话中的时序熵 (观点稳定性)"""
        if last_round < 2:
            return 0.5

        prev = [u for u in utterances if u.round == last_round - 1]
        curr = [u for u in utterances if u.round == last_round]

        if not prev or not curr:
            return 0.5

        prev_text = " ".join(u.content[:200] for u in prev)
        curr_text = " ".join(u.content[:200] for u in curr)
        prev_chars = set(prev_text.lower())
        curr_chars = set(curr_text.lower())

        if prev_chars and curr_chars:
            change = 1.0 - len(prev_chars & curr_chars) / len(prev_chars | curr_chars)
            return change
        return 0.5

    @staticmethod
    def _calc_conversation_impact(utterances: list) -> float:
        """
        对话中的影响熵: 攻击的震动程度

        衡量: 模型之间的攻击有多"致命"——
        如果一个模型在后续轮次大幅修改了自己的观点，说明它被攻击后震动大。
        如果所有模型都坚持原来的观点，说明攻击没有产生影响。
        """
        if len(utterances) < 4:
            return 0.3

        # 检测每个模型在不同轮次之间的观点变化幅度
        sources = set(u.source for u in utterances)
        max_round = max(u.round for u in utterances)

        if max_round < 1:
            return 0.3

        shifts = []
        for source in sources:
            my_utterances = sorted(
                [u for u in utterances if u.source == source],
                key=lambda u: u.round
            )
            if len(my_utterances) >= 2:
                # 比较第一次和最后一次发言的差异
                first = my_utterances[0].content[:300].lower()
                last = my_utterances[-1].content[:300].lower()
                first_chars = set(first)
                last_chars = set(last)
                if first_chars and last_chars:
                    shift = 1.0 - len(first_chars & last_chars) / len(first_chars | last_chars)
                    shifts.append(shift)

        if not shifts:
            return 0.3

        # 平均观点漂移量 = 影响熵
        # 漂移大 → 攻击有效 → 影响熵高（还在被震动，未稳定）
        # 漂移小 → 攻击无效或已消化 → 影响熵低
        avg_shift = sum(shifts) / len(shifts)
        return min(1.0, avg_shift * 1.5)


# 向后兼容: 保留旧的 calculate 方法名
EntropyCalculator.calculate = EntropyCalculator.from_graph
