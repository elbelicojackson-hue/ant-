"""
arena.py — Chain-Centric 对话场

每个 agent 维护一条结构化推理链 (Step1→Step2→...)。
攻击精确到某条链的某个步骤。
链有版本管理: 被攻击后可修正(版本+1)或崩塌。

算法负责:
1. 维护链注册表: 每个 agent 的当前链和历史版本
2. 路由攻击: 把攻击结果反馈给被攻击方
3. 熵计算: 基于链状态而非文本相似度
4. 收敛判断: 所有链稳定(无新攻击成功)时收敛
"""

from dataclasses import dataclass, field
from typing import Callable
from datetime import datetime, timezone
import asyncio
import time
import hashlib
import json
import re

from .entropy import EntropyVector, EntropyCalculator
from .state_machine import StateMachine, ThinkingState
from .consensus_store import ConsensusStore, VerifiedClaim, format_prior_consensus
from .strategy import AttackScheduler, EvidencePool, ReputationEngine
from .tools import ToolRegistry, create_default_registry, GroundedFact


# === 数据结构 ===

@dataclass
class ChainStep:
    """推理链中的一个步骤"""
    index: int                      # 步骤序号 (1-based)
    content: str                    # 步骤内容
    step_type: str = "claim"        # fact / assumption / inference / conclusion
    attacked_count: int = 0         # 被攻击次数
    status: str = "active"          # active / collapsed / fortified


@dataclass
class ReasoningChain:
    """一个 agent 的推理链"""
    chain_id: str                   # 唯一标识
    agent_id: str                   # 所属 agent
    version: int = 1                # 版本号
    steps: list[ChainStep] = field(default_factory=list)
    status: str = "active"          # active / broken / archived
    created_round: int = 0
    last_modified_round: int = 0


@dataclass
class Attack:
    """一次攻击记录"""
    attacker_id: str                # 攻击者
    target_chain_id: str            # 被攻击的链
    target_step_index: int          # 被攻击的步骤序号
    reason: str                     # 攻击理由
    round: int                      # 发生在第几轮
    success: bool = False           # 攻击是否导致崩塌 (由被攻击方回应决定)


@dataclass
class Utterance:
    """一次发言 — 对话场中的最小单元"""
    id: str
    round: int
    source: str
    content: str
    timestamp: str
    elapsed_ms: float
    reply_to: list[str] = field(default_factory=list)
    stance: str = ""
    claims: list[str] = field(default_factory=list)
    targets: list[str] = field(default_factory=list)


@dataclass
class Relation:
    """两个发言之间的关系"""
    source_id: str
    target_id: str
    relation_type: str      # supports / opposes / refines / introduces
    strength: float = 0.5


@dataclass
class ArenaResult:
    """对话场最终结果"""
    question: str
    utterances: list[Utterance]
    relations: list[Relation]
    total_rounds: int
    convergence_reason: str
    entropy_history: list[dict]
    final_entropy: EntropyVector
    consensus: list[str]
    disputes: list[str]
    elapsed_seconds: float
    # Chain-centric 新增
    chains: list[ReasoningChain] = field(default_factory=list)
    attacks: list[Attack] = field(default_factory=list)


class Arena:
    """
    Chain-Centric 对话场

    工作方式:
    1. 第0轮: 所有模型独立构建推理链 (Step1→Step2→...)
    2. 第1轮+: 每个模型看到所有链，攻击最薄弱的步骤
    3. 被攻击方必须: 修正链(版本+1) 或 承认崩塌
    4. 收敛: 连续N轮无成功攻击 或 所有链稳定

    链是一等公民，不是事后从文本中提取的。
    """

    def __init__(
        self,
        models: dict[str, Callable],
        max_rounds: int = 50,
        convergence_threshold: float = 0.25,
        min_rounds: int = 5,
        min_attacks: int = 3,
        verifier_id: str = "",
    ):
        self.models = models
        self.max_rounds = max_rounds
        self.convergence_threshold = convergence_threshold
        self.min_rounds = min_rounds
        self.min_attacks = min_attacks
        self.verifier_id = verifier_id
        self.state = StateMachine()
        for mid in models:
            self.state.register(mid, "participant")

        # Chain-centric 状态
        self.chains: dict[str, ReasoningChain] = {}
        self.attacks: list[Attack] = []

        # 共识存储 (可组合性)
        self.consensus_store = ConsensusStore()
        self.prior_consensus: list[VerifiedClaim] = []
        self._last_claim_map: dict = {}

        # 策略引擎 (数学优化)
        self.scheduler = AttackScheduler()      # UCB 攻击调度
        self.evidence_pool = EvidencePool()      # 证据池
        self.reputation = ReputationEngine()     # 动态信誉
        for mid in models:
            self.reputation.register(mid)

        # 外部工具注册表 (真值锚点)
        self.tool_registry = create_default_registry()

    async def debate(self, question: str) -> ArenaResult:
        """启动 chain-centric 对话"""
        start_time = time.time()
        self.state.start_session()

        utterances: list[Utterance] = []
        relations: list[Relation] = []
        entropy_history: list[dict] = []
        convergence_reason = "max_rounds"

        print(f"  [Chain-Centric 对话场启动] {len(self.models)} 个模型参与")
        print(f"  [最大轮数] {self.max_rounds}")
        print()

        # === Round 0: 所有模型独立构建推理链 ===
        print(f"  ── 第 0 轮: 构建推理链 ──")
        round_0 = await self._round_build_chains(question, 0, start_time)
        utterances.extend(round_0)

        self._print_chains_status()

        # 注册所有链到攻击调度器
        for chain in self.chains.values():
            if chain.status == "active":
                self.scheduler.register_chain(chain.agent_id, len(chain.steps))

        entropy = self._compute_entropy(utterances)
        entropy_history.append({"round": 0, **entropy.to_dict()})
        print(f"  [熵] {entropy}")
        print()

        # === Round 1+: 攻击循环 ===
        no_successful_attack_rounds = 0

        for round_num in range(1, self.max_rounds):
            print(f"  ── 第 {round_num} 轮: 攻击 ──")

            round_utterances, round_attacks = await self._round_attack(
                question, round_num, start_time
            )
            utterances.extend(round_utterances)
            self.attacks.extend(round_attacks)
            successful_this_round = sum(1 for a in round_attacks if a.success)

            new_relations = self._analyze_relations(round_utterances, utterances)
            relations.extend(new_relations)

            print(f"    本轮攻击: {len(round_attacks)} 次, 成功: {successful_this_round} 次")

            # 如果有成功攻击，让被攻击方修正链
            if successful_this_round > 0:
                no_successful_attack_rounds = 0
                print(f"  ── 第 {round_num} 轮: 修正 ──")
                repair_utterances = await self._round_repair(
                    question, round_attacks, round_num, start_time
                )
                utterances.extend(repair_utterances)
                self._print_chains_status()
            else:
                no_successful_attack_rounds += 1

            entropy = self._compute_entropy(utterances)
            entropy_history.append({"round": round_num, **entropy.to_dict()})
            print(f"  [熵] {entropy}")

            # === Claim 提取 + 跨链共识度 (α) ===
            # 每2轮做一次 claim 提取（平衡效率和精度）
            if round_num % 2 == 0 or no_successful_attack_rounds >= 1:
                consensus_alpha, claim_map = await self._extract_and_match_claims(
                    question, round_num, start_time
                )
                self._last_claim_map = claim_map  # 保存供 session 结束时持久化
                if consensus_alpha >= 0.65 and round_num >= self.min_rounds:
                    convergence_reason = (
                        f"true_consensus (round {round_num}, α={consensus_alpha:.2f}, "
                        f"attacks={len(self.attacks)})"
                    )
                    print(f"  [真共识] α={consensus_alpha:.2f}，多数链收敛到相同断言")
                    break

            # 收敛检测
            total_attacks = len(self.attacks)
            if (no_successful_attack_rounds >= 2
                and round_num >= self.min_rounds
                and total_attacks >= self.min_attacks):
                successful_attacks = sum(1 for a in self.attacks if a.success)
                # 检查脆弱共识: 如果共识是脆弱的，不允许收敛
                active_chains_now = [c for c in self.chains.values() if c.status == "active"]
                _, fragile = self._compute_consensus_entropy(active_chains_now)
                if fragile < 0.5:
                    convergence_reason = (
                        f"chains_stable (round {round_num}, "
                        f"attacks={total_attacks}, successful={successful_attacks}, "
                        f"fragile={fragile:.2f})"
                    )
                    print(f"  [收敛] 连续{no_successful_attack_rounds}轮无成功攻击，共识坚固 (脆弱度={fragile:.2f})")
                    break
                else:
                    print(f"  [未收敛] 攻击停滞但共识脆弱 (脆弱度={fragile:.2f})，继续")

            if (round_num >= self.min_rounds
                and total_attacks >= self.min_attacks
                and entropy.can_converge):
                convergence_reason = f"entropy_converged (round {round_num})"
                print(f"  [收敛] 熵已足够低且攻击充分")
                break

            # 查证触发: 第1轮强制触发 + 后续按需触发
            if self.verifier_id and self.verifier_id in self.models:
                should_verify = False

                # 条件1: 第1轮强制查证（问题刚展开，需要事实基础）
                if round_num == 1:
                    should_verify = True

                # 条件2: 证据熵或信息熵超阈值
                if entropy.evidence > 0.35 or entropy.information > 0.4:
                    should_verify = True

                # 条件3: 有成功攻击且涉及事实性断言
                if successful_this_round > 0 and round_num % 3 == 0:
                    should_verify = True

                if should_verify:
                    print(f"  [查证触发] R{round_num}")
                    try:
                        verify_utterance = await asyncio.wait_for(
                            self._trigger_verification(
                                question, utterances, round_num, start_time
                            ),
                            timeout=120.0  # 查证最多等2分钟
                        )
                    except asyncio.TimeoutError:
                        verify_utterance = None
                        print(f"    [查证超时]")
                    if verify_utterance:
                        utterances.append(verify_utterance)
                        # 添加到证据池
                        self.evidence_pool.add(
                            verify_utterance.content,
                            verify_utterance.source,
                            round_num
                        )

            # 停滞检测
            if len(entropy_history) >= self.min_rounds + 4:
                recent_3 = entropy_history[-3:]
                delta_1 = abs(recent_3[-1]["total"] - recent_3[-2]["total"])
                delta_2 = abs(recent_3[-2]["total"] - recent_3[-3]["total"])
                if delta_1 < 0.02 and delta_2 < 0.02:
                    convergence_reason = f"stagnation (round {round_num})"
                    print(f"  [停滞] 连续两轮熵不再变化")
                    break

            print()

        # === 最终分析 ===
        consensus, disputes = self._extract_consensus_from_chains()
        elapsed = time.time() - start_time

        # === 持久化共识 (可组合性) ===
        if hasattr(self, '_last_claim_map') and self._last_claim_map:
            stored = self.consensus_store.store_session_consensus(
                session_id=hashlib.sha256(f"{question}{start_time}".encode()).hexdigest()[:8],
                question=question,
                claims=[
                    {"content": claim, "alpha": count / len([c for c in self.chains.values() if c.status == "active"]),
                     "agree_count": count}
                    for claim, count in self._last_claim_map.items()
                ],
                participants=list(self.models.keys()),
                total_rounds=len(entropy_history),
                total_attacks=len(self.attacks),
            )
            if stored:
                print(f"  [共识持久化] 存储了 {len(stored)} 条新共识")
                stats = self.consensus_store.get_stats()
                print(f"    总共识库: {stats['active']} 条活跃, {stats['total_references']} 次引用")

        return ArenaResult(
            question=question,
            utterances=utterances,
            relations=relations,
            total_rounds=len(entropy_history),
            convergence_reason=convergence_reason,
            entropy_history=entropy_history,
            final_entropy=entropy,
            consensus=consensus,
            disputes=disputes,
            elapsed_seconds=elapsed,
            chains=list(self.chains.values()),
            attacks=self.attacks,
        )

    async def _round_build_chains(self, question: str, round_num: int,
                                   start_time: float) -> list[Utterance]:
        """第0轮: 所有模型独立构建结构化推理链"""

        # 加载与问题相关的先验共识
        self.prior_consensus = self.consensus_store.query_relevant(question)
        prior_text = format_prior_consensus(self.prior_consensus)
        if self.prior_consensus:
            print(f"  [先验共识] 加载了 {len(self.prior_consensus)} 条已验证共识")
            for pc in self.prior_consensus:
                print(f"    α={pc.alpha:.2f} | {pc.content[:60]}")
                self.consensus_store.reference_claim(pc.claim_id)

        prior_section = ""
        if prior_text:
            prior_section = f"\n{prior_text}\n\n"

        prompt = f"""分析以下问题，给出你的推理链。

问题: {question}
{prior_section}要求: 将你的分析拆分为 3-7 个步骤，每步一个独立断言。
用以下格式输出:

Step1: [你的第一个断言或前提]
Step2: [基于Step1的推理或新前提]
Step3: [进一步推理]
...
StepN: [最终结论]

每个 Step 必须是一个可以被独立攻击的断言。
不要写成散文，必须是 Step1/Step2/... 格式。"""

        tasks = []
        for mid, call_fn in self.models.items():
            tasks.append(self._call_model(mid, call_fn, prompt, round_num, start_time))

        results = await asyncio.gather(*tasks, return_exceptions=True)

        utterances = []
        for result in results:
            if isinstance(result, Utterance):
                utterances.append(result)
                chain = self._parse_chain(result.source, result.content, round_num)
                if chain:
                    self.chains[chain.chain_id] = chain
                    print(f"    {result.source}: {len(chain.steps)} 步推理链")
                    for step in chain.steps:
                        print(f"      Step{step.index}: {step.content[:60]}")
                else:
                    print(f"    {result.source}: (未能解析为链，作为单步处理)")
                    fallback_chain = ReasoningChain(
                        chain_id=f"c_{hashlib.sha256(f'{result.source}_{time.time()}'.encode()).hexdigest()[:10]}",
                        agent_id=result.source,
                        steps=[ChainStep(index=1, content=result.content[:200])],
                        created_round=round_num,
                        last_modified_round=round_num,
                    )
                    self.chains[fallback_chain.chain_id] = fallback_chain
            elif isinstance(result, Exception):
                print(f"    [错误] {result}")

        return utterances

    async def _round_attack(self, question: str, round_num: int,
                             start_time: float) -> tuple[list[Utterance], list[Attack]]:
        """
        攻击轮: UCB 策略通过信息路由实现，不注入 prompt

        算法控制的是"每个模型看到哪些链"，而不是"告诉模型攻击谁"。
        UCB 分数高的步骤所在的链会被优先展示给攻击者。
        模型依然自由选择攻击什么，但选择空间被路由限制了。
        """
        tasks = []

        for mid, call_fn in self.models.items():
            # UCB 路由: 选择展示给这个模型的链 (优先展示包含高 UCB 步骤的链)
            visible_chains = self._route_chains_for_attacker(mid, round_num)
            my_chain = self._get_agent_active_chain(mid)
            my_chain_display = self._format_single_chain(my_chain) if my_chain else "(你还没有链)"

            # 证据池作为"已知事实"展示 (不是指令，是信息)
            evidence_text = self.evidence_pool.format_all() if self.evidence_pool.evidences else ""

            # 外部锚定事实 (不可被普通攻击推翻)
            grounded_text = self.tool_registry.format_grounded_facts()

            prompt = f"""问题: {question}

以下是其他参与者当前的推理链:
{visible_chains}

{evidence_text}

{grounded_text}

你的当前推理链:
{my_chain_display}

任务: 从其他参与者的推理链中，选择一个最薄弱的步骤进行攻击。

要求:
1. 明确指出攻击目标: 哪个参与者的哪个 Step
2. 给出攻击理由: 为什么这个步骤不成立（反例、逻辑漏洞、缺乏证据）
3. 说明崩塌影响: 如果这个步骤崩塌，后续哪些步骤会跟着倒

输出格式 (严格遵守):
ATTACK_TARGET: [参与者名称].Step[N]
ATTACK_REASON: [具体的攻击理由]
COLLAPSE_IMPACT: [如果成功，哪些后续步骤崩塌]
CONFIDENCE: [高/中/低]

如果你认为所有链都无懈可击，输出:
NO_ATTACK: [说明为什么找不到漏洞]"""

            tasks.append(self._call_model(mid, call_fn, prompt, round_num, start_time))

        results = await asyncio.gather(*tasks, return_exceptions=True)

        utterances = []
        attacks = []
        for result in results:
            if isinstance(result, Utterance):
                utterances.append(result)
                attack = self._parse_attack(result.source, result.content, round_num)
                if attack:
                    attacks.append(attack)
                    target_chain = self.chains.get(attack.target_chain_id)
                    target_name = target_chain.agent_id if target_chain else "?"
                    print(f"    {result.source} → 攻击 {target_name}.Step{attack.target_step_index}: {attack.reason[:50]}")
                else:
                    print(f"    {result.source}: (未发起攻击或无法解析)")
            elif isinstance(result, Exception):
                print(f"    [错误] {result}")

        # 判定攻击是否成功
        if attacks:
            attacks = await self._judge_attacks(question, attacks, round_num, start_time)

        # 更新策略引擎状态
        for attack in attacks:
            target_chain = self.chains.get(attack.target_chain_id)
            if target_chain:
                self.scheduler.record_attack(
                    target_chain.agent_id, attack.target_step_index,
                    attack.success, round_num
                )
                self.reputation.record_attack_result(
                    attack.attacker_id, target_chain.agent_id, attack.success
                )

        # 存活的链获得信誉奖励
        for chain in self.chains.values():
            if chain.status == "active":
                self.reputation.record_survive(chain.agent_id)

        # 打印覆盖率和信誉
        coverage = self.scheduler.get_coverage()
        print(f"    [策略] 攻击覆盖率: {coverage:.0%}")
        if round_num % 3 == 0:
            print(f"    {self.reputation.format_status()}")

        return utterances, attacks

    def _route_chains_for_attacker(self, attacker_id: str, round_num: int) -> str:
        """
        UCB 信息路由: 决定展示给攻击者哪些链

        策略:
        - 所有活跃链都展示 (保证公平性)
        - 但包含高 UCB 步骤的链排在前面 (注意力偏向)
        - 已加固的步骤标注 ✓ (暗示"这个不容易攻破")
        - 从未被攻击的步骤标注 ⚠ (暗示"这个未经验证")

        这不是指令，而是信息结构的差异——
        模型天然会更关注排在前面的、带 ⚠ 标记的步骤。
        """
        active_chains = [
            c for c in self.chains.values()
            if c.status == "active" and c.agent_id != attacker_id
        ]

        if not active_chains:
            return "(无其他参与者的链)"

        # 计算每条链的"攻击价值" (包含高 UCB 步骤的链价值更高)
        chain_scores = []
        for chain in active_chains:
            max_ucb = 0.0
            for step in chain.steps:
                targets = self.scheduler.select_targets(attacker_id, top_k=100, current_round=round_num)
                for t_agent, t_step, t_score in targets:
                    if t_agent == chain.agent_id and t_step == step.index:
                        max_ucb = max(max_ucb, t_score)
                        break
            chain_scores.append((chain, max_ucb))

        # 按 UCB 价值排序 (高价值链排前面)
        chain_scores.sort(key=lambda x: x[1], reverse=True)

        # 格式化，带状态标记
        parts = []
        for chain, score in chain_scores:
            header = f"[{chain.agent_id}] v{chain.version}"
            steps_text = []
            for step in chain.steps:
                stat = self.scheduler.stats.get(f"{chain.agent_id}.{step.index}")
                if step.status == "fortified":
                    icon = "✓"
                elif stat and stat.attack_count == 0:
                    icon = "⚠"  # 未验证
                elif step.status == "collapsed":
                    icon = "✗"
                else:
                    icon = " "
                steps_text.append(f"  {icon} Step{step.index}: {step.content}")
            parts.append(header + "\n" + "\n".join(steps_text))

        return "\n\n".join(parts)

    async def _judge_attacks(self, question: str, attacks: list[Attack],
                              round_num: int, start_time: float) -> list[Attack]:
        """
        联合判定: 攻击是否成立不由被攻击方单独决定

        判定流程:
        1. 被攻击方回应（辩护或承认）
        2. 攻击方和一个第三方各投一票
        3. 多数决: 3票中2票以上认为崩塌 → 崩塌

        这避免了"自我辩护偏差"——模型很难承认自己错了。
        """
        tasks = []
        attack_indices = []

        for i, attack in enumerate(attacks):
            target_chain = self.chains.get(attack.target_chain_id)
            if not target_chain:
                continue
            target_step = None
            for step in target_chain.steps:
                if step.index == attack.target_step_index:
                    target_step = step
                    break
            if not target_step:
                continue
            defender_id = target_chain.agent_id
            if defender_id not in self.models:
                continue

            prompt = f"""你的推理链中的一个步骤被攻击了。

问题: {question}

你的完整推理链:
{self._format_single_chain(target_chain)}

被攻击的步骤: Step{target_step.index}: {target_step.content}

攻击理由: {attack.reason}

请判断这个攻击是否成立。
- 如果攻击确实指出了你的逻辑漏洞或事实错误，承认崩塌
- 如果攻击不成立（你能反驳），给出反驳理由

输出格式:
VERDICT: [COLLAPSED/DEFENDED]
REASON: [为什么崩塌或为什么攻击不成立]"""

            call_fn = self.models[defender_id]
            tasks.append(self._call_model(defender_id, call_fn, prompt, round_num, start_time))
            attack_indices.append(i)

        if not tasks:
            return attacks

        results = await asyncio.gather(*tasks, return_exceptions=True)

        # 第一轮: 收集被攻击方的回应
        defender_verdicts = {}
        for idx, result in zip(attack_indices, results):
            if isinstance(result, Utterance):
                content_upper = result.content.upper()
                collapsed = "COLLAPSED" in content_upper or "崩塌" in result.content
                defender_verdicts[idx] = (collapsed, result.content)

        # 第二轮: 联合判定 — 找一个第三方模型投票
        # 选择一个既不是攻击方也不是防守方的模型
        for idx, (defender_collapsed, defender_response) in defender_verdicts.items():
            attack = attacks[idx]
            target_chain = self.chains.get(attack.target_chain_id)
            if not target_chain:
                continue

            defender_id = target_chain.agent_id
            attacker_id = attack.attacker_id

            # 找第三方裁判
            judge_id = None
            for mid in self.models:
                if mid != defender_id and mid != attacker_id:
                    judge_id = mid
                    break

            # 投票计数: 攻击方天然投"崩塌"(1票), 防守方的回应(1票), 第三方(1票)
            votes_collapse = 1  # 攻击方的票
            votes_defend = 0

            if defender_collapsed:
                votes_collapse += 1  # 防守方也承认崩塌
            else:
                votes_defend += 1    # 防守方辩护

            # 第三方裁判投票 (如果有的话)
            if judge_id and judge_id in self.models:
                target_step = None
                for step in target_chain.steps:
                    if step.index == attack.target_step_index:
                        target_step = step
                        break

                judge_prompt = f"""作为第三方，判断以下攻击是否成立。

被攻击的断言: {target_step.content if target_step else ''}

攻击理由: {attack.reason}

防守方回应: {defender_response[:300]}

这个攻击是否确实指出了逻辑漏洞或事实错误？
输出: VERDICT: [COLLAPSED/DEFENDED]"""

                try:
                    judge_fn = self.models[judge_id]
                    judge_result = await self._call_model(
                        judge_id, judge_fn, judge_prompt, round_num, start_time
                    )
                    judge_upper = judge_result.content.upper()
                    if "COLLAPSED" in judge_upper or "崩塌" in judge_result.content:
                        votes_collapse += 1
                    else:
                        votes_defend += 1
                except Exception:
                    pass  # 裁判失败不影响结果

            # 多数决
            final_collapsed = votes_collapse >= 2

            if final_collapsed:
                attacks[idx].success = True
                if target_chain:
                    for step in target_chain.steps:
                        if step.index == attacks[idx].target_step_index:
                            step.status = "collapsed"
                            step.attacked_count += 1
                            break
                    collapsed_steps = [s for s in target_chain.steps if s.status == "collapsed"]
                    last_step = target_chain.steps[-1] if target_chain.steps else None
                    if last_step and last_step.status == "collapsed":
                        target_chain.status = "broken"
                    elif len(collapsed_steps) >= len(target_chain.steps) // 2:
                        target_chain.status = "broken"
                print(f"      ✗ 攻击成功 ({votes_collapse}:{votes_defend}): Step{attacks[idx].target_step_index} 崩塌")
            else:
                attacks[idx].success = False
                if target_chain:
                    for step in target_chain.steps:
                        if step.index == attacks[idx].target_step_index:
                            step.attacked_count += 1
                            if step.attacked_count >= 2:
                                step.status = "fortified"
                            break
                print(f"      ✓ 攻击失败 ({votes_collapse}:{votes_defend}): Step{attacks[idx].target_step_index} 防守成功")

        return attacks

    async def _round_repair(self, question: str, attacks: list[Attack],
                             round_num: int, start_time: float) -> list[Utterance]:
        """被攻击方修正自己的链"""
        repair_agents = set()
        for attack in attacks:
            if not attack.success:
                continue
            target_chain = self.chains.get(attack.target_chain_id)
            if target_chain:
                repair_agents.add(target_chain.agent_id)

        utterances = []
        for agent_id in repair_agents:
            chain = self._get_agent_active_chain(agent_id)
            if not chain or agent_id not in self.models:
                for c in self.chains.values():
                    if c.agent_id == agent_id and c.status == "broken":
                        chain = c
                        break
            if not chain:
                continue

            collapsed_steps = [s for s in chain.steps if s.status == "collapsed"]
            if not collapsed_steps:
                continue

            collapsed_info = "\n".join(f"  Step{s.index}: {s.content}" for s in collapsed_steps)

            prompt = f"""你的推理链中有步骤被成功攻击并崩塌了。

问题: {question}

你的当前推理链:
{self._format_single_chain(chain)}

崩塌的步骤:
{collapsed_info}

请修正你的推理链:
1. 删除崩塌的步骤，用新的步骤替代
2. 修改后续依赖崩塌步骤的推理
3. 如果整条链无法修复，构建全新的链

输出修正后的完整推理链:
Step1: ...
Step2: ...
...

必须是 Step1/Step2/... 格式。"""

            call_fn = self.models[agent_id]
            task_result = await self._call_model(agent_id, call_fn, prompt, round_num, start_time)
            utterances.append(task_result)

            chain.status = "archived"
            new_chain = self._parse_chain(agent_id, task_result.content, round_num)
            if new_chain:
                new_chain.version = chain.version + 1
                self.chains[new_chain.chain_id] = new_chain
                print(f"    {agent_id}: 链修正 v{chain.version} → v{new_chain.version} ({len(new_chain.steps)} 步)")
            else:
                print(f"    {agent_id}: 修正失败")

        return utterances

    async def _call_model(self, model_id: str, call_fn: Callable,
                           prompt: str, round_num: int,
                           start_time: float) -> Utterance:
        """调用单个模型并包装为 Utterance"""
        self.state.transition(model_id, ThinkingState.THINKING, task=f"round {round_num}")
        try:
            content = await call_fn(prompt, {})
        except RuntimeError as e:
            if "已被跳过" in str(e):
                self.state.transition(model_id, ThinkingState.FAILED, reason="超时禁用")
                raise  # 让上层 gather 捕获为 Exception
            raise
        elapsed_ms = (time.time() - start_time) * 1000
        self.state.transition(model_id, ThinkingState.DONE, reason=f"{len(content)} 字")

        uid = hashlib.sha256(f"{model_id}{round_num}{time.time()}".encode()).hexdigest()[:12]
        return Utterance(
            id=uid,
            round=round_num,
            source=model_id,
            content=content,
            timestamp=datetime.now(timezone.utc).isoformat(timespec="milliseconds"),
            elapsed_ms=elapsed_ms,
        )


    async def _extract_and_match_claims(self, question: str, round_num: int,
                                          start_time: float) -> tuple[float, dict]:
        """
        Claim 提取 + 跨链共识度 (α)

        用一个模型从所有活跃链中提取原子断言，然后计算跨链匹配度。
        当多个 agent 的链里包含相同的 claim → α 上升。

        返回: (consensus_alpha, claim_map)
        - consensus_alpha: 0-1，越高表示共识越强
        - claim_map: {claim_text: [agent_ids]}
        """
        active_chains = [c for c in self.chains.values() if c.status == "active"]
        if len(active_chains) < 2:
            return 0.0, {}

        # 构建所有链的摘要
        chains_text = ""
        for chain in active_chains:
            steps_str = " | ".join(f"Step{s.index}: {s.content[:80]}" for s in chain.steps)
            chains_text += f"[{chain.agent_id}]: {steps_str}\n"

        # 用一个模型提取跨链共同断言
        # 选择一个快速模型（Qwen-Max 通常最快）
        extractor_id = None
        for mid in ["Qwen-Max", "Qwen3.6"]:
            if mid in self.models:
                extractor_id = mid
                break
        if not extractor_id:
            extractor_id = list(self.models.keys())[0]

        prompt = f"""以下是多个参与者对同一问题的推理链:

{chains_text}

任务: 提取所有参与者之间的共同断言（即多数参与者都同意的结论）。

对每个共同断言，标注有多少参与者同意。

输出格式 (每行一个):
CLAIM: [断言内容] | AGREE: [同意的参与者数量]/[总参与者数量] | AGENTS: [同意的参与者名称列表]

只输出被 3 个以上参与者共同持有的断言。如果没有共同断言，输出:
NO_CONSENSUS: [原因]"""

        try:
            call_fn = self.models[extractor_id]
            result = await asyncio.wait_for(
                self._call_model(extractor_id, call_fn, prompt, round_num, start_time),
                timeout=60.0  # claim 提取最多等60秒
            )

            # 解析 claim
            claim_map = {}
            total_agents = len(active_chains)
            lines = result.content.split("\n")

            for line in lines:
                line = line.strip()
                if line.startswith("CLAIM:") and "AGREE:" in line:
                    # 解析 CLAIM: xxx | AGREE: 5/7 | AGENTS: ...
                    parts = line.split("|")
                    claim_text = parts[0].replace("CLAIM:", "").strip()[:100]

                    agree_part = ""
                    for p in parts:
                        if "AGREE:" in p:
                            agree_part = p.strip()
                            break

                    # 提取 agree 数字
                    agree_count = 0
                    if "/" in agree_part:
                        try:
                            num_str = agree_part.split(":")[1].strip().split("/")[0].strip()
                            agree_count = int(num_str)
                        except (ValueError, IndexError):
                            agree_count = 0

                    if agree_count >= 3 and claim_text:
                        claim_map[claim_text] = agree_count

            if not claim_map:
                if "NO_CONSENSUS" in result.content:
                    print(f"    [α] 无跨链共识")
                    return 0.0, {}
                return 0.3, {}

            # 计算 α: 加权平均共识度
            # α = Σ(agree_i / total) / num_claims
            alphas = [count / total_agents for count in claim_map.values()]
            consensus_alpha = sum(alphas) / len(alphas) if alphas else 0.0

            print(f"    [α] 共识度={consensus_alpha:.2f}, {len(claim_map)} 个共同断言")
            for claim, count in list(claim_map.items())[:3]:
                print(f"      α={count}/{total_agents}: {claim[:60]}")

            return consensus_alpha, claim_map

        except (Exception, asyncio.TimeoutError) as e:
            print(f"    [α] 提取失败或超时: {e}")
            return 0.0, {}

    async def _trigger_verification(self, question: str, history: list[Utterance],
                                     round_num: int, start_time: float) -> Utterance:
        """
        触发查证 — 通过工具注册表调用外部搜索

        结果作为 GroundedFact 存入，不可被普通攻击推翻。
        """
        # 构建搜索查询
        recent = [u for u in history if u.round >= max(0, round_num - 1)]
        context = " ".join(u.content[:100] for u in recent[-3:])
        search_query = f"{question[:100]} {context[:100]}"

        # 调用搜索工具
        result = await self.tool_registry.call(
            "web_search",
            query=search_query,
            triggered_by="system",
            session_id="",
        )

        if not result or not result.success:
            return None

        # 添加到证据池
        self.evidence_pool.add(result.content, f"web_search[{result.call_id}]", round_num)

        # 包装为 Utterance
        uid = hashlib.sha256(f"verify_{round_num}_{time.time()}".encode()).hexdigest()[:12]
        elapsed_ms = (time.time() - start_time) * 1000

        return Utterance(
            id=uid,
            round=round_num,
            source="[外部查证]",
            content=result.content,
            timestamp=datetime.now(timezone.utc).isoformat(timespec="milliseconds"),
            elapsed_ms=elapsed_ms,
        )

    def _parse_chain(self, agent_id: str, content: str, round_num: int) -> ReasoningChain | None:
        """从模型输出中解析推理链"""
        steps = []
        pattern = r'Step\s*(\d+)\s*[:：]\s*(.+?)(?=Step\s*\d+\s*[:：]|$)'
        matches = re.findall(pattern, content, re.DOTALL)

        if not matches:
            pattern2 = r'(\d+)\s*[.、)]\s*(.+?)(?=\d+\s*[.、)]|$)'
            matches = re.findall(pattern2, content, re.DOTALL)

        if not matches:
            return None

        for idx_str, step_content in matches:
            idx = int(idx_str)
            clean = step_content.strip().split("\n")[0].strip()
            if len(clean) < 5:
                continue
            steps.append(ChainStep(index=idx, content=clean[:200]))

        if not steps:
            return None

        chain_id = hashlib.sha256(f"{agent_id}_{round_num}_{time.time()}".encode()).hexdigest()[:10]
        return ReasoningChain(
            chain_id=f"c_{chain_id}",
            agent_id=agent_id,
            steps=steps,
            created_round=round_num,
            last_modified_round=round_num,
        )

    def _parse_attack(self, attacker_id: str, content: str, round_num: int) -> Attack | None:
        """从模型输出中解析攻击"""
        if "NO_ATTACK" in content:
            return None

        target_match = re.search(
            r'ATTACK_TARGET\s*[:：]\s*\[?(.+?)\]?\.?\s*Step\s*(\d+)',
            content, re.IGNORECASE
        )
        if not target_match:
            target_match = re.search(
                r'(?:攻击目标)\s*[:：]\s*\[?(.+?)\]?\s*(?:的|\.)\s*Step\s*(\d+)',
                content, re.IGNORECASE
            )
        if not target_match:
            return None

        target_agent = target_match.group(1).strip().strip("[]【】")
        target_step = int(target_match.group(2))

        target_chain_id = None
        for chain in self.chains.values():
            if chain.status == "active" and chain.agent_id == target_agent:
                target_chain_id = chain.chain_id
                break
        if not target_chain_id:
            for chain in self.chains.values():
                if chain.status == "active" and target_agent in chain.agent_id:
                    target_chain_id = chain.chain_id
                    break
        if not target_chain_id:
            return None

        reason_match = re.search(
            r'(?:ATTACK_REASON|攻击理由)\s*[:：]\s*(.+?)(?=COLLAPSE|CONFIDENCE|$)',
            content, re.DOTALL | re.IGNORECASE
        )
        reason = reason_match.group(1).strip()[:200] if reason_match else content[:100]

        return Attack(
            attacker_id=attacker_id,
            target_chain_id=target_chain_id,
            target_step_index=target_step,
            reason=reason,
            round=round_num,
        )

    def _get_agent_active_chain(self, agent_id: str) -> ReasoningChain | None:
        """获取某个 agent 当前活跃的链"""
        for chain in self.chains.values():
            if chain.agent_id == agent_id and chain.status == "active":
                return chain
        return None

    def _format_chains_except(self, exclude_agent: str) -> str:
        """格式化除某个 agent 外的所有活跃链"""
        parts = []
        for chain in self.chains.values():
            if chain.status != "active" or chain.agent_id == exclude_agent:
                continue
            parts.append(self._format_single_chain(chain))
        return "\n\n".join(parts) if parts else "(无其他参与者的链)"

    def _format_single_chain(self, chain: ReasoningChain) -> str:
        """格式化单条链"""
        header = f"[{chain.agent_id}] v{chain.version} ({chain.status})"
        steps_text = []
        for step in chain.steps:
            icon = {"active": " ", "collapsed": "✗", "fortified": "✓"}.get(step.status, " ")
            attacked = f" (被攻击{step.attacked_count}次)" if step.attacked_count > 0 else ""
            steps_text.append(f"  {icon} Step{step.index}: {step.content}{attacked}")
        return header + "\n" + "\n".join(steps_text)

    def _print_chains_status(self):
        """打印当前所有链的状态"""
        active = [c for c in self.chains.values() if c.status == "active"]
        broken = [c for c in self.chains.values() if c.status == "broken"]
        print(f"  [链状态] 活跃: {len(active)}, 崩塌: {len(broken)}")
        for chain in self.chains.values():
            if chain.status == "archived":
                continue
            active_s = sum(1 for s in chain.steps if s.status == "active")
            collapsed_s = sum(1 for s in chain.steps if s.status == "collapsed")
            fortified_s = sum(1 for s in chain.steps if s.status == "fortified")
            print(f"    {chain.chain_id} ({chain.agent_id}) v{chain.version} "
                  f"[{chain.status}] {active_s}活/{collapsed_s}崩/{fortified_s}固")

    def _analyze_relations(self, new_utterances: list[Utterance],
                           all_utterances: list[Utterance]) -> list[Relation]:
        """分析关系"""
        relations = []
        oppose_words = ["攻击", "ATTACK", "不成立", "漏洞", "崩塌", "COLLAPSED"]
        for new_u in new_utterances:
            if any(w in new_u.content for w in oppose_words):
                prev_others = [u for u in all_utterances
                              if u.source != new_u.source and u.round < new_u.round]
                if prev_others:
                    relations.append(Relation(
                        source_id=new_u.id,
                        target_id=prev_others[-1].id,
                        relation_type="opposes",
                        strength=0.8,
                    ))
        return relations

    def _compute_entropy(self, utterances: list[Utterance]) -> EntropyVector:
        """基于链状态计算熵"""
        if not self.chains:
            return EntropyVector()

        active_chains = [c for c in self.chains.values() if c.status == "active"]
        all_chains = list(self.chains.values())

        # 分歧熵: 基于立场一致性而非文本相似度
        # 从结论中提取对核心断言的立场，比较立场是否一致
        divergence = self._compute_stance_divergence(active_chains)

        # 依赖熵: 崩塌步骤占比
        total_steps = sum(len(c.steps) for c in all_chains if c.status != "archived")
        collapsed_steps = sum(
            sum(1 for s in c.steps if s.status == "collapsed")
            for c in all_chains if c.status != "archived"
        )
        dependency = collapsed_steps / total_steps if total_steps > 0 else 0.0

        # 传播熵: 崩塌链占比
        non_archived = [c for c in all_chains if c.status != "archived"]
        broken_ratio = sum(1 for c in non_archived if c.status == "broken") / len(non_archived) if non_archived else 0.0

        # 时序熵: 链修正次数
        recent_repairs = sum(1 for c in all_chains if c.version > 1)
        temporal = min(1.0, recent_repairs * 0.15)

        # 证据熵
        all_text = " ".join(u.content for u in utterances[-10:])
        strong_evidence = ["数据", "研究", "统计", "%", "报告", "实验", "论文"]
        evidence_count = sum(1 for w in strong_evidence if w in all_text)
        evidence = max(0.0, 0.6 - evidence_count * 0.08)

        # 信息熵
        uncertain_words = ["不确定", "可能", "也许", "或许", "有待", "未知"]
        info = min(1.0, sum(1 for w in uncertain_words if w in all_text) * 0.08)

        # 影响熵: 成功攻击率
        total_attacks_count = len(self.attacks)
        successful = sum(1 for a in self.attacks if a.success)
        impact = successful / total_attacks_count if total_attacks_count > 0 else 0.3

        # === 共识熵 + 脆弱共识熵 ===
        consensus_entropy, fragile_consensus = self._compute_consensus_entropy(active_chains)

        return EntropyVector(
            semantic=divergence * 0.5,
            causal=max(0.1, 0.3 - len(active_chains) * 0.03),
            boundary=0.4,
            temporal=temporal,
            dependency=dependency,
            divergence=divergence,
            information=info,
            propagation=broken_ratio,
            evidence=evidence,
            impact=max(impact, fragile_consensus),  # 脆弱共识放大影响熵
        )

    @staticmethod
    def _ngram_similarity(text_a: str, text_b: str, n: int = 3) -> float:
        """n-gram 文本相似度"""
        if not text_a or not text_b:
            return 0.0
        ngrams_a = set(text_a[i:i+n] for i in range(len(text_a) - n + 1))
        ngrams_b = set(text_b[i:i+n] for i in range(len(text_b) - n + 1))
        if not ngrams_a or not ngrams_b:
            return 0.0
        return len(ngrams_a & ngrams_b) / len(ngrams_a | ngrams_b)

    def _compute_consensus_entropy(self, active_chains: list[ReasoningChain]) -> tuple[float, float]:
        """
        计算共识熵和脆弱共识熵

        共识熵: 衡量当前共识的强度
          0 = 完全共识 (所有链结论一致且经过攻击验证)
          1 = 无共识 (各说各话)

        脆弱共识熵: 衡量共识的脆弱程度
          0 = 坚固共识 (经过多轮攻击仍然成立)
          1 = 脆弱共识 (表面一致但未经验证，一攻就碎)

        区分:
        - 共识熵低 + 脆弱共识低 = 真正的共识 (可以收敛)
        - 共识熵低 + 脆弱共识高 = 虚假共识 (不应收敛，需要更多攻击)
        - 共识熵高 = 还没达成共识
        """
        if len(active_chains) < 2:
            return 0.5, 0.5

        # === 共识熵: 立场一致性 ===
        # 已经由 _compute_stance_divergence 计算了分歧度
        # 共识熵 = 分歧度 (分歧低 = 共识高)
        divergence = self._compute_stance_divergence(active_chains)
        consensus_entropy = divergence  # 直接等于分歧度

        # === 脆弱共识熵 ===
        # 衡量: 当前共识是否经过了充分的攻击验证

        # 因素1: 加固步骤占比 — 被攻击过且存活的步骤越多，共识越坚固
        total_steps = sum(len(c.steps) for c in active_chains)
        fortified_steps = sum(
            sum(1 for s in c.steps if s.status == "fortified")
            for c in active_chains
        )
        fortified_ratio = fortified_steps / total_steps if total_steps > 0 else 0.0

        # 因素2: 被攻击过的步骤占比 — 从未被攻击的步骤是"未验证"的
        attacked_steps = sum(
            sum(1 for s in c.steps if s.attacked_count > 0)
            for c in active_chains
        )
        attacked_ratio = attacked_steps / total_steps if total_steps > 0 else 0.0

        # 因素3: 链版本 — 修正过的链比从未修正的更可靠 (经过了攻击-修正循环)
        avg_version = sum(c.version for c in active_chains) / len(active_chains)
        version_maturity = min(1.0, (avg_version - 1) * 0.3)  # v1=0, v2=0.3, v4+=1.0

        # 因素4: 总攻击轮数 — 攻击越多，存活的共识越坚固
        total_attacks = len(self.attacks)
        attack_pressure = min(1.0, total_attacks * 0.05)  # 20次攻击 = 满压力

        # 脆弱度 = 1 - 验证充分度
        verification_score = (
            fortified_ratio * 0.3 +      # 加固步骤多 → 不脆弱
            attacked_ratio * 0.25 +       # 被攻击过 → 不脆弱
            version_maturity * 0.2 +      # 修正过 → 不脆弱
            attack_pressure * 0.25        # 经历了足够攻击 → 不脆弱
        )
        fragile_consensus = 1.0 - verification_score

        return consensus_entropy, fragile_consensus

    def _compute_stance_divergence(self, active_chains: list[ReasoningChain]) -> float:
        """
        基于立场一致性计算分歧熵

        原理: 不比较文本措辞，而是提取每条链对核心问题的"立场信号"。
        立场信号 = 一组二元判断 (支持/反对/未提及)

        例如对于量子计算问题:
        - 链A结论: "RSA尚未被破解" → 立场: [反对原命题]
        - 链B结论: "Willow不具备破解能力" → 立场: [反对原命题]
        → 虽然措辞不同，但立场一致 → 分歧熵低

        实现: 用关键词/短语匹配提取立场向量，计算向量间的一致性。
        """
        if len(active_chains) < 2:
            return 0.5

        # 从所有链的所有步骤中收集文本
        chain_texts = []
        for chain in active_chains:
            if chain.steps:
                # 用整条链的内容（不只是结论），因为立场可能分布在多个步骤中
                full_text = " ".join(s.content for s in chain.steps)
                chain_texts.append(full_text)

        if len(chain_texts) < 2:
            return 0.5

        # 立场信号词典: 每组是一对对立的立场
        # 格式: (信号名, 正面词列表, 反面词列表)
        stance_signals = [
            ("affirm_threat",
             ["确实", "已经", "能够破解", "不安全", "已被破解", "威胁已实现", "紧迫"],
             ["尚未", "不能", "无法破解", "仍然安全", "未被破解", "夸大", "误导", "错误"]),
            ("urgency",
             ["立即", "紧急", "马上", "刻不容缓", "必须立刻"],
             ["不紧急", "无需立即", "循序渐进", "规划", "中长期", "逐步", "不必恐慌"]),
            ("scope",
             ["所有", "全部", "全面", "一切都"],
             ["部分", "有限", "特定场景", "并非所有", "不是所有"]),
            ("factual_basis",
             ["已证实", "已实现", "事实", "已经做到"],
             ["未证实", "尚未实现", "误解", "混淆", "偷换", "前提错误", "事实错误"]),
            ("harvest_now",
             ["追溯解密", "历史数据", "先存后解"],
             ["前向保密", "临时密钥", "不可追溯"]),
        ]

        # 为每条链计算立场向量
        # 值: +1=正面立场, -1=反面立场, 0=未提及
        stance_vectors = []
        for text in chain_texts:
            vector = []
            for signal_name, positive_words, negative_words in stance_signals:
                pos_count = sum(1 for w in positive_words if w in text)
                neg_count = sum(1 for w in negative_words if w in text)
                if pos_count > neg_count:
                    vector.append(1)
                elif neg_count > pos_count:
                    vector.append(-1)
                else:
                    vector.append(0)
            stance_vectors.append(vector)

        # 计算立场一致性: 所有链对之间的立场向量相似度
        agreements = []
        for i in range(len(stance_vectors)):
            for j in range(i + 1, len(stance_vectors)):
                # 计算两个立场向量的一致性
                agree_count = 0
                total_count = 0
                for vi, vj in zip(stance_vectors[i], stance_vectors[j]):
                    if vi == 0 and vj == 0:
                        continue  # 都未提及，不计入
                    total_count += 1
                    if vi == vj:
                        agree_count += 1
                    elif vi == 0 or vj == 0:
                        agree_count += 0.5  # 一方未提及，算半同意

                agreement = agree_count / total_count if total_count > 0 else 0.5
                agreements.append(agreement)

        if not agreements:
            return 0.5

        avg_agreement = sum(agreements) / len(agreements)
        # 高一致性 → 低分歧; 低一致性 → 高分歧
        return 1.0 - avg_agreement

    def _extract_consensus_from_chains(self) -> tuple[list[str], list[str]]:
        """从链状态中提取共识和分歧 — 基于立场一致性"""
        consensus = []
        disputes = []

        active_chains = [c for c in self.chains.values() if c.status == "active"]
        if not active_chains:
            return [], []

        # 用立场向量判断共识
        conclusions = [(c.agent_id, " ".join(s.content for s in c.steps)) for c in active_chains if c.steps]

        if len(conclusions) >= 2:
            # 简化: 看最终结论的方向性关键词
            for agent_id, full_text in conclusions:
                last_step = next((c.steps[-1].content for c in active_chains
                                  if c.agent_id == agent_id and c.steps), "")

                # 判断这条链是否反对原始命题
                oppose_original = any(w in full_text for w in
                    ["尚未", "不能", "无法破解", "错误", "误导", "夸大", "未被破解",
                     "不紧急", "无需立即", "前提错误"])
                support_original = any(w in full_text for w in
                    ["已经破解", "确实不安全", "必须立即", "已被破解"])

                if oppose_original and not support_original:
                    consensus.append(f"{agent_id}: {last_step[:80]}")
                elif support_original and not oppose_original:
                    disputes.append(f"{agent_id} [支持原命题]: {last_step[:80]}")
                else:
                    disputes.append(f"{agent_id} [立场不明]: {last_step[:80]}")

        # 崩塌的链
        for chain in self.chains.values():
            if chain.status == "broken" and chain.steps:
                disputes.append(f"{chain.agent_id} v{chain.version} [崩塌]: {chain.steps[-1].content[:60]}")

        return consensus[:5], disputes[:5]
