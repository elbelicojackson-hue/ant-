"""
strategy.py — 数学驱动的攻击策略与信誉引擎

解决5个结构性问题:
1. 攻击目标分配: UCB (Upper Confidence Bound) 算法，平衡探索与利用
2. 证据池注入: 查证结果结构化存储，注入攻击 prompt
3. 防守记忆: 已反驳的攻击角度不再重复
4. 动态信誉: ELO 变体，基于攻击/防守胜率计算权重
5. 攻击覆盖率: 信息增益最大化，优先攻击未验证步骤

核心数学:
- UCB1: score(step) = exploitation + C * sqrt(ln(N) / n_i)
- ELO: R_new = R_old + K * (actual - expected)
- 信息增益: IG(step) = H(before) - H(after|attack)
"""

import math
from dataclasses import dataclass, field
from typing import Optional


# ═══════════════════════════════════════════════════════════════
# 1. UCB 攻击目标选择
# ═══════════════════════════════════════════════════════════════

@dataclass
class StepStats:
    """每个步骤的攻击统计"""
    agent_id: str
    step_index: int
    attack_count: int = 0           # 被攻击次数
    collapse_count: int = 0         # 崩塌次数
    defend_count: int = 0           # 防守成功次数
    last_attack_round: int = -1     # 上次被攻击的轮次
    defense_reasons: list[str] = field(default_factory=list)  # 已有的防守理由


class AttackScheduler:
    """
    UCB1 攻击调度器

    把"选择攻击哪个步骤"建模为多臂老虎机问题:
    - 每个步骤是一个臂 (arm)
    - 攻击成功 = reward 1, 失败 = reward 0
    - UCB1 平衡: 攻击成功率高的步骤 (exploitation) vs 从未被攻击的步骤 (exploration)

    公式: UCB(i) = mean_reward(i) + C * sqrt(ln(N) / n_i)
    - mean_reward(i) = 该步骤的历史崩塌率
    - N = 总攻击次数
    - n_i = 该步骤被攻击次数
    - C = 探索系数 (默认 sqrt(2))

    效果: 从未被攻击的步骤 UCB → ∞，一定会被优先选中
    """

    C = math.sqrt(2)  # 探索系数

    def __init__(self):
        self.stats: dict[str, StepStats] = {}  # key: "agent_id.step_index"
        self.total_attacks: int = 0

    def register_step(self, agent_id: str, step_index: int):
        """注册一个可攻击的步骤"""
        key = f"{agent_id}.{step_index}"
        if key not in self.stats:
            self.stats[key] = StepStats(agent_id=agent_id, step_index=step_index)

    def register_chain(self, agent_id: str, step_count: int):
        """注册一条链的所有步骤"""
        for i in range(1, step_count + 1):
            self.register_step(agent_id, i)

    def record_attack(self, agent_id: str, step_index: int,
                       success: bool, round_num: int, defense_reason: str = ""):
        """记录一次攻击结果"""
        key = f"{agent_id}.{step_index}"
        if key not in self.stats:
            self.register_step(agent_id, step_index)

        stat = self.stats[key]
        stat.attack_count += 1
        stat.last_attack_round = round_num
        self.total_attacks += 1

        if success:
            stat.collapse_count += 1
        else:
            stat.defend_count += 1
            if defense_reason and defense_reason not in stat.defense_reasons:
                stat.defense_reasons.append(defense_reason[:100])

    def select_targets(self, attacker_id: str, top_k: int = 3,
                        current_round: int = 0) -> list[tuple[str, int, float]]:
        """
        为攻击者选择最优攻击目标

        返回: [(agent_id, step_index, ucb_score), ...] 按 UCB 分数降序

        排除:
        - 攻击者自己的步骤
        - 已经崩塌的步骤 (collapse_count > 0 且 defend_count == 0)
        - 已加固的步骤 (defend_count >= 2 且 collapse_count == 0)
        """
        candidates = []

        for key, stat in self.stats.items():
            # 排除自己
            if stat.agent_id == attacker_id:
                continue

            # 排除已确定崩塌的
            if stat.collapse_count > 0 and stat.defend_count == 0:
                continue

            # 排除已充分加固的 (被攻击2次以上且从未崩塌)
            if stat.defend_count >= 3 and stat.collapse_count == 0:
                continue

            ucb = self._compute_ucb(stat, current_round)
            candidates.append((stat.agent_id, stat.step_index, ucb))

        # 按 UCB 降序
        candidates.sort(key=lambda x: x[2], reverse=True)
        return candidates[:top_k]

    def _compute_ucb(self, stat: StepStats, current_round: int) -> float:
        """
        计算 UCB1 分数

        UCB(i) = exploitation + exploration + recency_bonus

        exploitation: 历史崩塌率 (高 = 这个步骤容易被攻破)
        exploration: sqrt(ln(N) / n_i) (高 = 这个步骤很少被攻击)
        recency_bonus: 长时间未被攻击的步骤额外加分
        """
        n_i = stat.attack_count

        # 从未被攻击 → UCB = ∞ (最高优先级)
        if n_i == 0:
            return float('inf')

        N = max(1, self.total_attacks)

        # Exploitation: 崩塌率
        mean_reward = stat.collapse_count / n_i

        # Exploration: UCB1 探索项
        exploration = self.C * math.sqrt(math.log(N) / n_i)

        # Recency bonus: 距离上次攻击越久，越应该重新验证
        rounds_since = current_round - stat.last_attack_round
        recency = 0.1 * math.log(1 + rounds_since)

        return mean_reward + exploration + recency

    def get_coverage(self) -> float:
        """攻击覆盖率: 被攻击过的步骤占比"""
        if not self.stats:
            return 0.0
        attacked = sum(1 for s in self.stats.values() if s.attack_count > 0)
        return attacked / len(self.stats)

    def get_defense_reasons(self, agent_id: str, step_index: int) -> list[str]:
        """获取某步骤已有的防守理由 (避免重复攻击角度)"""
        key = f"{agent_id}.{step_index}"
        stat = self.stats.get(key)
        return stat.defense_reasons if stat else []

    def format_attack_guidance(self, attacker_id: str, current_round: int) -> str:
        """
        生成攻击指导文本，注入到攻击 prompt 中

        告诉模型:
        1. 推荐攻击哪些目标 (UCB 最高的)
        2. 哪些攻击角度已经被反驳过 (避免重复)
        3. 当前攻击覆盖率
        """
        targets = self.select_targets(attacker_id, top_k=3, current_round=current_round)
        if not targets:
            return ""

        lines = ["[攻击策略建议]"]
        lines.append(f"当前攻击覆盖率: {self.get_coverage():.0%}")
        lines.append("推荐攻击目标 (按优先级):")

        for agent_id, step_idx, score in targets:
            stat = self.stats.get(f"{agent_id}.{step_idx}")
            if not stat:
                continue

            if stat.attack_count == 0:
                reason = "从未被验证"
            else:
                reason = f"崩塌率{stat.collapse_count}/{stat.attack_count}"

            lines.append(f"  1. {agent_id}.Step{step_idx} ({reason})")

            # 已有的防守理由
            if stat.defense_reasons:
                lines.append(f"     已被反驳的角度: {'; '.join(stat.defense_reasons[:2])}")
                lines.append(f"     请从新角度攻击。")

        return "\n".join(lines)


# ═══════════════════════════════════════════════════════════════
# 2. 证据池
# ═══════════════════════════════════════════════════════════════

@dataclass
class Evidence:
    """一条查证到的证据"""
    content: str                    # 证据内容
    source: str                     # 来源 (哪个模型查证的)
    round: int                      # 哪一轮查证的
    relevance_keywords: list[str] = field(default_factory=list)  # 相关关键词


class EvidencePool:
    """
    证据池: 存储查证结果，按相关性注入攻击 prompt

    解决: 查证结果只是一条 utterance，其他模型看不到
    方案: 结构化存储，按关键词匹配注入
    """

    def __init__(self):
        self.evidences: list[Evidence] = []

    def add(self, content: str, source: str, round_num: int):
        """添加一条证据"""
        # 提取关键词
        import re
        keywords = re.findall(r'[\u4e00-\u9fff]{2,4}|[a-zA-Z]{4,}', content)
        # 去重取前20个
        keywords = list(dict.fromkeys(keywords))[:20]

        self.evidences.append(Evidence(
            content=content[:500],
            source=source,
            round=round_num,
            relevance_keywords=keywords,
        ))

    def query_relevant(self, step_content: str, limit: int = 2) -> list[Evidence]:
        """查找与某个步骤相关的证据"""
        if not self.evidences:
            return []

        # 简单关键词匹配
        scores = []
        for ev in self.evidences:
            score = sum(1 for kw in ev.relevance_keywords if kw in step_content)
            scores.append((score, ev))

        scores.sort(key=lambda x: x[0], reverse=True)
        return [ev for score, ev in scores[:limit] if score > 0]

    def format_for_attack(self, target_step_content: str) -> str:
        """格式化相关证据，注入攻击 prompt"""
        relevant = self.query_relevant(target_step_content)
        if not relevant:
            return ""

        lines = ["[已查证的相关事实]"]
        for ev in relevant:
            lines.append(f"  • {ev.content[:150]} (来源: {ev.source}, R{ev.round})")
        return "\n".join(lines)

    def format_all(self) -> str:
        """格式化所有证据"""
        if not self.evidences:
            return ""
        lines = ["[证据池 — 已查证的事实]"]
        for ev in self.evidences:
            lines.append(f"  • {ev.content[:200]}")
        return "\n".join(lines)


# ═══════════════════════════════════════════════════════════════
# 3. 动态信誉 (ELO 变体)
# ═══════════════════════════════════════════════════════════════

class ReputationEngine:
    """
    动态信誉系统 — ELO 变体

    每个 agent 有一个信誉分 (初始 1000)。
    攻击成功: 攻击方 +K, 防守方 -K (K 根据双方信誉差调整)
    防守成功: 防守方 +K/2, 攻击方 -K/2
    链从未被攻破: 每轮 +bonus

    信誉影响:
    - 投票权重: 信誉高的模型投票权重更大
    - α 计算: 信誉高的模型同意某 claim，贡献更多 α
    - 攻击目标: 信誉高的模型的链更难被攻破 (需要更强证据)

    数学:
    expected(A vs B) = 1 / (1 + 10^((R_B - R_A) / 400))
    R_new = R_old + K * (actual - expected)
    """

    K_BASE = 32            # 基础 K 值
    INITIAL_RATING = 1000  # 初始信誉
    SURVIVE_BONUS = 5      # 每轮链存活奖励

    def __init__(self):
        self.ratings: dict[str, float] = {}  # agent_id → rating
        self.history: list[dict] = []         # 信誉变化历史

    def register(self, agent_id: str):
        """注册 agent"""
        if agent_id not in self.ratings:
            self.ratings[agent_id] = self.INITIAL_RATING

    def record_attack_result(self, attacker_id: str, defender_id: str, success: bool):
        """
        记录攻击结果，更新双方信誉

        攻击成功: 攻击方赢，防守方输
        攻击失败: 防守方赢，攻击方输
        """
        self.register(attacker_id)
        self.register(defender_id)

        r_a = self.ratings[attacker_id]
        r_d = self.ratings[defender_id]

        # 期望胜率
        expected_a = 1.0 / (1.0 + 10 ** ((r_d - r_a) / 400))
        expected_d = 1.0 - expected_a

        # 实际结果
        if success:
            actual_a, actual_d = 1.0, 0.0
        else:
            actual_a, actual_d = 0.0, 1.0

        # 更新
        k = self.K_BASE
        self.ratings[attacker_id] = r_a + k * (actual_a - expected_a)
        self.ratings[defender_id] = r_d + k * (actual_d - expected_d)

        self.history.append({
            "attacker": attacker_id,
            "defender": defender_id,
            "success": success,
            "attacker_delta": k * (actual_a - expected_a),
            "defender_delta": k * (actual_d - expected_d),
        })

    def record_survive(self, agent_id: str):
        """链存活一轮，小幅加分"""
        self.register(agent_id)
        self.ratings[agent_id] += self.SURVIVE_BONUS

    def get_vote_weight(self, agent_id: str) -> float:
        """
        获取投票权重 (0.5 - 2.0)

        信誉 1000 → 权重 1.0
        信誉 1200 → 权重 1.5
        信誉 800  → 权重 0.7
        """
        self.register(agent_id)
        rating = self.ratings[agent_id]
        # sigmoid-like mapping: 800→0.5, 1000→1.0, 1200→1.5, 1400→2.0
        weight = 0.5 + (rating - 800) / 400
        return max(0.5, min(2.0, weight))

    def get_alpha_weight(self, agent_id: str) -> float:
        """
        获取 α 计算权重

        信誉高的模型同意某 claim，贡献更多 α
        """
        return self.get_vote_weight(agent_id)

    def get_ranking(self) -> list[tuple[str, float]]:
        """获取信誉排名"""
        return sorted(self.ratings.items(), key=lambda x: x[1], reverse=True)

    def format_status(self) -> str:
        """格式化信誉状态"""
        ranking = self.get_ranking()
        lines = ["[信誉排名]"]
        for agent_id, rating in ranking:
            weight = self.get_vote_weight(agent_id)
            delta = rating - self.INITIAL_RATING
            sign = "+" if delta >= 0 else ""
            lines.append(f"  {agent_id}: {rating:.0f} ({sign}{delta:.0f}) 权重={weight:.2f}")
        return "\n".join(lines)


# ═══════════════════════════════════════════════════════════════
# 4. 信息增益计算
# ═══════════════════════════════════════════════════════════════

def information_gain(step_attack_count: int, step_defend_count: int,
                     step_collapse_count: int, chain_length: int) -> float:
    """
    计算攻击某个步骤的期望信息增益

    IG = H(current) - E[H(after)]

    H(current): 当前对该步骤的不确定性
    E[H(after)]: 攻击后的期望不确定性

    未被攻击的步骤: H = 1.0 (完全不确定)
    被攻击且结果一致的步骤: H → 0 (确定)
    被攻击且结果矛盾的步骤: H 仍然高 (需要更多攻击)

    额外因素: 步骤在链中的位置 (靠前的步骤崩塌影响更大)
    """
    total = step_attack_count
    if total == 0:
        # 从未被攻击 → 信息增益最大
        return 1.0

    # 当前熵: 基于崩塌率的二元熵
    p = step_collapse_count / total
    if p == 0 or p == 1:
        current_h = 0.0
    else:
        current_h = -(p * math.log2(p) + (1 - p) * math.log2(1 - p))

    # 位置权重: 链前面的步骤崩塌影响更大
    # position_weight = 1 + (chain_length - step_index) / chain_length
    # 简化: 用 chain_length 作为代理
    position_factor = 1.0 + 0.1 * chain_length

    # 信息增益 = 当前不确定性 * 位置权重
    return current_h * position_factor
