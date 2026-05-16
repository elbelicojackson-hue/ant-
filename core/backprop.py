"""
backprop.py — 合成梯度引擎 (Synthetic Gradient Engine)

黑盒神经回路中的信用分配 — 这是整个 NeuralComm 协议最核心的工程难题。

问题: 你有一条 LLM 链路 P→R→I→M→E，最终输出质量高/低，
      但你无法对 LLM 求导，怎么知道每个神经元贡献了多少？

解: 四个代理梯度信号 (Proxy Gradient Signals):

  1. Delta Quality     — 信号通过神经元后，变好了还是变坏了？
  2. Entropy Gradient  — 这个神经元降低还是增加了系统熵？
  3. Claim Survival    — 这个神经元的断言有多少在下游存活？
  4. Consensus Alignment — 这个神经元的输出和最终共识有多对齐？

这四个信号合成一个 "伪梯度"，沿回路反向传播，更新突触权重。

关键创新:
  - 跨运行学习: 权重在多次执行间积累，神经元自然分化
  - 信用不确定性: 当信用分配不确定时，更新幅度自动降低
  - 元学习: 信用组件的权重比例随问题域自适应调整
"""

from dataclasses import dataclass, field
from typing import Optional
import math
import time
import hashlib
import json

from .signal import NeuralSignal, Intent, Certainty, TraceStep
from .neuron import Neuron, NeuronRole
from .entropy import EntropyVector


# ═══════════════════════════════════════════════════════════════════════════════
# 数据模型
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class ContributionTrace:
    """单个神经元在一次执行中的贡献快照"""
    neuron_id: str
    neuron_role: str
    input_signal_id: str
    output_signal_id: str

    # 信号变换度量
    signal_delta: float = 0.0           # 信号经过该神经元后的变化幅度 (0-1)
    content_novelty: float = 0.0        # 引入了多少新内容 (vs 复述输入)

    # 熵变化
    entropy_before: Optional[EntropyVector] = None
    entropy_after: Optional[EntropyVector] = None
    entropy_reduction: float = 0.0      # 正值 = 降低了熵 (好事)

    # 断言追踪
    claims_made: list[str] = field(default_factory=list)
    claims_survived_count: int = 0
    claims_challenged_count: int = 0
    claims_refuted_count: int = 0

    # 下游反馈
    was_inhibited: bool = False         # 被抑制层拦截了吗
    was_challenged: bool = False        # 被下游质疑了吗
    challenge_outcome: str = "none"     # "upheld" | "overruled" | "none"

    # 时序
    processing_time_ms: float = 0.0
    position_in_chain: int = 0          # 在链路中的位置 (0=first)


@dataclass
class CreditAssignment:
    """一次信用分配的结果"""
    neuron_id: str
    credit: float                       # -1.0 ~ 1.0, 正值=做得好, 负值=拖后腿
    confidence: float                   # 0.0 ~ 1.0, 这次分配有多确定
    components: dict[str, float] = field(default_factory=dict)  # 各代理梯度分量
    explanation: str = ""


@dataclass
class ExecutionTrace:
    """一次完整回路执行的可追溯记录"""
    trace_id: str
    input_signal_id: str
    final_signal_id: str
    contributions: list[ContributionTrace] = field(default_factory=list)
    credits: list[CreditAssignment] = field(default_factory=list)
    quality_score: float = 0.0          # 最终输出质量 (0-1)
    converged: bool = False
    iterations: int = 0
    total_time_ms: float = 0.0
    problem_domain: str = "general"     # 问题域标签


@dataclass
class SynapseStats:
    """突触级别的累积统计 — 跨运行学习的基础"""
    times_used: int = 0
    avg_credit: float = 0.0             # 历史平均信用
    credit_variance: float = 0.0        # 信用方差 (高方差 = 不稳定)
    last_credit: float = 0.0
    momentum: float = 0.0               # 当前动量
    specialization_score: dict[str, float] = field(default_factory=dict)  # 问题域→平均信用


# ═══════════════════════════════════════════════════════════════════════════════
# 合成梯度引擎
# ═══════════════════════════════════════════════════════════════════════════════

class SynapticOptimizer:
    """
    突触优化器 — 带动量、学习率衰减、信用置信度调制的梯度更新。

    模仿 SGD with Momentum，但"梯度"来自代理信号而非自动微分。
    """

    def __init__(self, learning_rate: float = 0.05, momentum: float = 0.9,
                 weight_decay: float = 0.001, min_weight: float = 0.05,
                 max_weight: float = 0.95):
        self.lr = learning_rate
        self.momentum = momentum
        self.weight_decay = weight_decay
        self.min_weight = min_weight
        self.max_weight = max_weight
        self.synapse_stats: dict[str, SynapseStats] = {}  # key = "source→target"
        self.total_updates: int = 0

    def _synapse_key(self, source: str, target: str) -> str:
        return f"{source}→{target}"

    def get_stats(self, source: str, target: str) -> SynapseStats:
        key = self._synapse_key(source, target)
        if key not in self.synapse_stats:
            self.synapse_stats[key] = SynapseStats()
        return self.synapse_stats[key]

    def apply(self, circuit: 'NeuralCircuit', credits: list[CreditAssignment],
              problem_domain: str = "general"):
        """
        将信用分配转化为突触权重更新。

        更新公式:
          momentum_t = β * momentum_{t-1} + (1-β) * credit * confidence * lr
          weight_t = clip(weight_{t-1} + momentum_t - weight_decay * weight_{t-1})
        """
        credit_map = {c.neuron_id: c for c in credits}

        for synapse in circuit.synapses:
            source_credit = credit_map.get(synapse.source)
            target_credit = credit_map.get(synapse.target)

            if source_credit is None and target_credit is None:
                continue

            # 合成梯度: 源和目标的信用加权平均
            # 前馈突触主要看源的信用 (源的输出质量决定信号值不值得传递)
            # 反馈突触主要看目标的信用 (目标发起的反馈是否有用)
            if synapse.signal_type == "feedback":
                credit = target_credit.credit if target_credit else source_credit.credit
                confidence = target_credit.confidence if target_credit else source_credit.confidence
            else:
                credit = source_credit.credit if source_credit else 0.0
                # forward 连接: source 输出质量 + target 是否有效利用
                if target_credit:
                    credit = 0.7 * credit + 0.3 * target_credit.credit
                    confidence = 0.7 * source_credit.confidence + 0.3 * target_credit.confidence
                else:
                    confidence = source_credit.confidence if source_credit else 0.5

            stats = self.get_stats(synapse.source, synapse.target)
            stats.times_used += 1

            # 指数移动平均更新信用统计
            alpha = 0.1
            stats.avg_credit = (1 - alpha) * stats.avg_credit + alpha * credit
            stats.credit_variance = (1 - alpha) * stats.credit_variance + alpha * (credit - stats.avg_credit) ** 2
            stats.last_credit = credit

            # 更新问题域专业化分数
            if problem_domain not in stats.specialization_score:
                stats.specialization_score[problem_domain] = credit
            else:
                stats.specialization_score[problem_domain] = (
                    0.9 * stats.specialization_score[problem_domain] + 0.1 * credit
                )

            # 动量更新
            effective_lr = self.lr * confidence  # 不确定时减小学习率
            stats.momentum = (
                self.momentum * stats.momentum
                + (1 - self.momentum) * credit * effective_lr
            )

            # 权重更新 + 衰减
            old_weight = synapse.weight
            synapse.weight += stats.momentum - self.weight_decay * synapse.weight
            synapse.weight = max(self.min_weight, min(self.max_weight, synapse.weight))

            self.total_updates += 1

    def get_domain_weights(self, problem_domain: str) -> dict[str, float]:
        """获取指定问题域的突触权重快照"""
        weights = {}
        for key, stats in self.synapse_stats.items():
            weights[key] = stats.specialization_score.get(problem_domain, stats.avg_credit)
        return weights

    def reset_momentum(self):
        """重置所有动量 (问题域切换时调用)"""
        for stats in self.synapse_stats.values():
            stats.momentum = 0.0


class CreditAnalyzer:
    """
    信用分析器 — 计算四个代理梯度信号并合成最终信用。
    """

    def __init__(self):
        # 代理梯度权重 (可通过元学习调整)
        self.component_weights = {
            "delta_quality": 0.30,       # 信号变换质量
            "entropy_reduction": 0.30,   # 熵降低贡献
            "claim_survival": 0.25,      # 断言存活率
            "consensus_alignment": 0.15, # 共识对齐
        }
        self.min_confidence = 0.3

    def analyze(self, contributions: list[ContributionTrace],
                quality_score: float, converged: bool) -> list[CreditAssignment]:
        """
        分析整条链路的每个神经元的贡献，分配信用。

        关键算法:
          从链路末端反向遍历，累积下游反馈信息，
          为每个神经元计算四个代理梯度的加权得分。
        """
        if not contributions:
            return []

        credits = []
        n = len(contributions)

        for i in range(n - 1, -1, -1):  # 反向遍历
            contrib = contributions[i]
            is_last = (i == n - 1)

            # ── 代理梯度 1: Delta Quality ──
            # 信号通过神经元后的变化质量
            # 高位神经元 (抑制剂、元认知) 的变换质量权重大
            role_multiplier = self._role_quality_multiplier(contrib.neuron_role)
            delta_q = contrib.signal_delta * role_multiplier
            if contrib.content_novelty > 0.3:
                delta_q *= (1.0 + contrib.content_novelty * 0.5)  # 有价值的新内容加分

            # ── 代理梯度 2: Entropy Gradient ──
            # 降低熵 = 正面贡献
            entropy_grad = contrib.entropy_reduction
            # 熵在危险区的降低更可贵 (从0.6降到0.4 比 从0.2降到0.1 更值钱)
            if contrib.entropy_before and contrib.entropy_before.total > 0.4:
                entropy_grad *= 1.5

            # ── 代理梯度 3: Claim Survival ──
            # 断言在下游存活的比例
            total_claims = len(contrib.claims_made)
            if total_claims > 0:
                survival_rate = contrib.claims_survived_count / total_claims
                # 被质疑但存活 = 经过考验的断言，更可信
                if contrib.claims_challenged_count > 0 and contrib.claims_refuted_count == 0:
                    survival_rate = min(1.0, survival_rate * 1.2)
                claim_score = survival_rate
            else:
                claim_score = 0.5  # 没有断言 → 中性

            # ── 代理梯度 4: Consensus Alignment ──
            # 最后位置的神经元天然对齐共识
            if is_last and converged:
                consensus_score = 1.0
            elif contrib.was_inhibited:
                consensus_score = -0.5  # 被抑制 = 偏离共识
            elif contrib.was_challenged and contrib.challenge_outcome == "overruled":
                consensus_score = -0.3
            else:
                # 中间神经元: 基于下游反馈估计
                consensus_score = 0.5 if not contrib.was_challenged else 0.2

            # ── 合成最终信用 ──
            w = self.component_weights
            credit = (
                w["delta_quality"] * delta_q
                + w["entropy_reduction"] * entropy_grad
                + w["claim_survival"] * claim_score
                + w["consensus_alignment"] * consensus_score
            )
            credit = max(-1.0, min(1.0, credit))

            # ── 信用置信度 ──
            # 可用的信号越多，置信度越高
            available_signals = 0
            if contrib.entropy_before is not None:
                available_signals += 1
            if total_claims > 0:
                available_signals += 1
            if contrib.was_challenged or contrib.was_inhibited:
                available_signals += 1
            confidence = max(self.min_confidence, available_signals / 3.0)

            # ── 生成解释 ──
            explanation_parts = []
            if delta_q > 0.2:
                explanation_parts.append(f"信号变换正向 (+{delta_q:.2f})")
            elif delta_q < -0.1:
                explanation_parts.append(f"信号变换负向 ({delta_q:.2f})")
            if entropy_grad > 0.1:
                explanation_parts.append(f"降低系统熵 (+{entropy_grad:.2f})")
            elif entropy_grad < -0.1:
                explanation_parts.append(f"增加系统熵 ({entropy_grad:.2f})")
            if contrib.claims_survived_count > 0:
                explanation_parts.append(f"断言存活 {contrib.claims_survived_count}/{total_claims}")
            if contrib.was_inhibited:
                explanation_parts.append("被抑制层拦截")
            elif contrib.was_challenged:
                explanation_parts.append(f"被质疑-{contrib.challenge_outcome}")
            if not explanation_parts:
                explanation_parts.append("中性贡献")

            credits.append(CreditAssignment(
                neuron_id=contrib.neuron_id,
                credit=credit,
                confidence=confidence,
                components={
                    "delta_quality": delta_q,
                    "entropy_reduction": entropy_grad,
                    "claim_survival": claim_score,
                    "consensus_alignment": consensus_score,
                },
                explanation="; ".join(explanation_parts),
            ))

        # 还原为正向顺序
        credits.reverse()
        return credits

    def _role_quality_multiplier(self, role: str) -> float:
        """不同角色的输出质量权重不同"""
        multipliers = {
            "perception": 1.0,     # 输入端，变换期望低
            "reasoning": 1.2,      # 核心推理，变换期望高
            "inhibition": 1.5,     # 质量控制，正确抑制很值钱
            "metacognition": 1.3,  # 元认知审视
            "execution": 1.0,      # 执行端
        }
        return multipliers.get(role, 1.0)

    def update_component_weights(self, domain: str, performance_history: list[dict]):
        """
        元学习: 根据历史表现调整代理梯度权重。

        如果某个问题域中 claim_survival 信号和最终质量相关性强，
        就增大其权重；如果 entropy_reduction 信号噪声大，就减小。
        """
        if len(performance_history) < 5:
            return  # 数据不够，保持默认

        # 计算每个组件与 quality_score 的相关性
        correlations = {}
        for comp in self.component_weights:
            comp_scores = []
            quality_scores = []
            for record in performance_history:
                if comp in record.get("components", {}):
                    comp_scores.append(record["components"][comp])
                    quality_scores.append(record.get("quality_score", 0.5))

            if len(comp_scores) >= 5:
                correlations[comp] = self._pearson_correlation(comp_scores, quality_scores)
            else:
                correlations[comp] = 0.0

        # 用 softmax 重新分配权重
        # 负相关的组件权重降到接近0
        shifted = {k: max(0.05, v + 0.5) for k, v in correlations.items()}
        total = sum(shifted.values())
        if total > 0:
            self.component_weights = {k: v / total for k, v in shifted.items()}

    @staticmethod
    def _pearson_correlation(x: list[float], y: list[float]) -> float:
        n = len(x)
        if n < 2:
            return 0.0
        mean_x = sum(x) / n
        mean_y = sum(y) / n
        cov = sum((x[i] - mean_x) * (y[i] - mean_y) for i in range(n))
        std_x = math.sqrt(sum((v - mean_x) ** 2 for v in x))
        std_y = math.sqrt(sum((v - mean_y) ** 2 for v in y))
        if std_x == 0 or std_y == 0:
            return 0.0
        return cov / (std_x * std_y)


# ═══════════════════════════════════════════════════════════════════════════════
# 主引擎
# ═══════════════════════════════════════════════════════════════════════════════

class NeuralBackprop:
    """
    神经反向传播引擎 — 黑盒回路中的合成梯度。

    用法:
        bp = NeuralBackprop(learning_rate=0.05, momentum=0.9)
        circuit = NeuralCircuit()
        # ... 添加神经元、自动布线 ...

        # 多次执行积累学习
        for problem in problems:
            signal = NeuralSignal(content=problem, ...)
            trace, credits = await bp.execute(circuit, signal)
            print(f"Quality: {trace.quality_score:.2f}")

        # 查看学到的突触专业化
        print(bp.optimizer.synapse_stats)
    """

    def __init__(self, learning_rate: float = 0.05, momentum: float = 0.9,
                 weight_decay: float = 0.001, enable_meta_learning: bool = True):
        self.optimizer = SynapticOptimizer(
            learning_rate=learning_rate,
            momentum=momentum,
            weight_decay=weight_decay,
        )
        self.analyzer = CreditAnalyzer()
        self.enable_meta_learning = enable_meta_learning
        self.execution_db: list[ExecutionTrace] = []
        self.problem_domain_history: list[dict] = []  # 用于元学习

    def _generate_trace_id(self) -> str:
        raw = f"{time.time()}{len(self.execution_db)}"
        return hashlib.sha256(raw.encode()).hexdigest()[:12]

    def _classify_domain(self, content: str) -> str:
        """简单的问题域分类 (可扩展为 LLM 分类)"""
        content_lower = content.lower()
        domains = {
            "math": ["计算", "数学", "公式", "方程", "calculate", "math", "solve",
                     "sum", "product", "integral"],
            "logic": ["推理", "逻辑", "如果", "悖论", "logic", "reason", "argue",
                      "premise", "conclusion"],
            "code": ["代码", "编程", "bug", "函数", "code", "function", "algorithm",
                     "implement"],
            "creative": ["创意", "设计", "故事", "诗歌", "creative", "design", "story",
                         "write", "imagine"],
            "factual": ["什么", "谁", "哪里", "何时", "定义", "what", "who", "where",
                        "when", "define", "fact"],
        }
        for domain, keywords in domains.items():
            if any(kw in content_lower for kw in keywords):
                return domain
        return "general"

    async def execute(self, circuit: 'NeuralCircuit',
                      input_signal: NeuralSignal) -> tuple[ExecutionTrace, list[CreditAssignment]]:
        """
        执行回路 + 反向信用分配 + 突触更新。

        这是核心 API — 替换 circuit.propagate() 的直接调用。
        """
        trace_id = self._generate_trace_id()
        domain = self._classify_domain(input_signal.content)

        # ── Phase 1: 前向传播 (带贡献追踪) ──
        contributions = await self._forward_with_tracing(circuit, input_signal)
        circuit_result = circuit.execution_history[-1] if circuit.execution_history else None

        # ── Phase 2: 质量评估 ──
        quality_score = self._evaluate_quality(contributions, circuit_result)

        # ── Phase 3: 反向信用分配 ──
        credits = self.analyzer.analyze(
            contributions=contributions,
            quality_score=quality_score,
            converged=circuit_result.converged if circuit_result else False,
        )

        # ── Phase 4: 突触更新 ──
        self.optimizer.apply(circuit, credits, problem_domain=domain)

        # ── 记录 ──
        trace = ExecutionTrace(
            trace_id=trace_id,
            input_signal_id=input_signal.signal_id,
            final_signal_id=circuit_result.final_signal.signal_id if circuit_result and circuit_result.final_signal else "",
            contributions=contributions,
            credits=credits,
            quality_score=quality_score,
            converged=circuit_result.converged if circuit_result else False,
            iterations=circuit_result.iterations if circuit_result else 0,
            total_time_ms=circuit_result.total_time_ms if circuit_result else 0.0,
            problem_domain=domain,
        )
        self.execution_db.append(trace)

        # ── 元学习: 更新代理梯度权重 ──
        if self.enable_meta_learning and len(self.execution_db) % 10 == 0:
            history = []
            for et in self.execution_db[-20:]:  # 最近20条
                for c in et.credits:
                    history.append({
                        "components": c.components,
                        "quality_score": et.quality_score,
                        "domain": et.problem_domain,
                    })
            self.analyzer.update_component_weights(domain, history)

        return trace, credits

    async def _forward_with_tracing(self, circuit: 'NeuralCircuit',
                                    input_signal: NeuralSignal) -> list[ContributionTrace]:
        """
        在执行回路的同时，记录每个神经元的贡献。

        通过猴子补丁 Neuron.process 来拦截输入/输出信号。
        更优雅的方式是修改 circuit.propagate() 来原生支持 tracing。
        """
        contributions = []

        # 保存原有的 process 方法
        original_processes = {}
        for nid, neuron in circuit.neurons.items():
            original_processes[nid] = neuron.process

        async def traced_process(neuron, signal, _original):
            """包装 process 方法，记录输入输出"""
            start = time.time()
            input_copy = NeuralSignal.from_dict(signal.to_dict())

            # 调用原始 process
            result = await _original(signal)

            elapsed = (time.time() - start) * 1000

            if result is not None:
                output_copy = NeuralSignal.from_dict(result.to_dict())

                # 基本信号变换度量
                signal_delta = self._compute_signal_delta(input_copy, output_copy)
                content_novelty = self._compute_content_novelty(input_copy, output_copy)

                # 熵变化 (如果有熵计算器)
                try:
                    from .entropy import EntropyCalculator
                    entropy_before = EntropyCalculator.from_signal(input_copy)
                    entropy_after = EntropyCalculator.from_signal(output_copy)
                    entropy_reduction = entropy_before.total - entropy_after.total
                except Exception:
                    entropy_before = None
                    entropy_after = None
                    entropy_reduction = 0.0

                # 断言提取 (简单规则)
                claims = self._extract_claims(output_copy)

                contrib = ContributionTrace(
                    neuron_id=neuron.model_id,
                    neuron_role=neuron.role.value,
                    input_signal_id=input_copy.signal_id,
                    output_signal_id=output_copy.signal_id,
                    signal_delta=signal_delta,
                    content_novelty=content_novelty,
                    entropy_before=entropy_before,
                    entropy_after=entropy_after,
                    entropy_reduction=entropy_reduction,
                    claims_made=claims,
                    processing_time_ms=elapsed,
                )
                contributions.append(contrib)

            return result

        # 注入 traced process
        for nid, neuron in circuit.neurons.items():
            _original = original_processes[nid]
            neuron.process = lambda sig, n=neuron, o=_original: traced_process(n, sig, o)

        try:
            # 执行回路 (这会触发 traced process)
            await circuit.propagate(input_signal)
        finally:
            # 恢复原始 process
            for nid, neuron in circuit.neurons.items():
                neuron.process = original_processes[nid]

        # 后处理: 填充下游反馈信息
        self._fill_downstream_feedback(contributions, circuit)

        return contributions

    def _fill_downstream_feedback(self, contributions: list[ContributionTrace],
                                   circuit: 'NeuralCircuit'):
        """填充每个贡献的下游反馈信息"""
        if len(contributions) < 2:
            return

        n = len(contributions)
        for i in range(n):
            contrib = contributions[i]
            contrib.position_in_chain = i

            # 检查下游神经元是否质疑/抑制了该神经元的输出
            for j in range(i + 1, n):
                downstream = contributions[j]

                # 抑制层拦截
                if downstream.neuron_role == "inhibition":
                    contrib.was_inhibited = True

                # 下游质疑: 检查下游信号是否包含质疑关键字
                # (实际应解析下游的 trace/uncertainties，这里用启发式)
                if any(kw in str(downstream.claims_made).lower()
                       for kw in ["wrong", "错误", "invalid", "无效", "disagree", "不同意"]):
                    contrib.was_challenged = True
                    contrib.claims_challenged_count += 1

            # 估算断言存活
            if contrib.claims_made:
                surviving = 0
                refuted = 0
                for claim in contrib.claims_made:
                    claim_words = set(claim.lower().split())
                    # 检查下游是否复现了该断言的要素
                    downstream_mentions = 0
                    for j in range(i + 1, n):
                        ds = contributions[j]
                        for ds_claim in ds.claims_made:
                            ds_words = set(ds_claim.lower().split())
                            overlap = len(claim_words & ds_words) / max(1, len(claim_words))
                            if overlap > 0.4:
                                downstream_mentions += 1

                    if downstream_mentions > 0:
                        surviving += 1
                    else:
                        refuted += 1

                contrib.claims_survived_count = surviving
                contrib.claims_refuted_count = refuted

    def _compute_signal_delta(self, before: NeuralSignal,
                               after: NeuralSignal) -> float:
        """计算信号通过神经元后的变化幅度"""
        delta = 0.0

        # 置信度变化
        cert_order = {Certainty.GUESS: 0, Certainty.WEAK: 1, Certainty.MODERATE: 2,
                      Certainty.STRONG: 3, Certainty.VERIFIED: 4}
        cert_delta = abs(cert_order.get(after.certainty, 2) - cert_order.get(before.certainty, 2))
        delta += cert_delta * 0.2

        # 意图变化
        if before.intent != after.intent:
            delta += 0.15

        # 约束条件变化 (新增约束 = 细化)
        new_constraints = set(after.constraints) - set(before.constraints)
        if new_constraints:
            delta += min(0.3, len(new_constraints) * 0.1)

        # trace 步数变化
        trace_delta = abs(len(after.trace) - len(before.trace))
        delta += min(0.2, trace_delta * 0.05)

        # 不确定性变化
        unc_delta = abs(len(after.uncertainties) - len(before.uncertainties))
        delta += min(0.15, unc_delta * 0.05)

        return min(1.0, delta)

    def _compute_content_novelty(self, before: NeuralSignal,
                                   after: NeuralSignal) -> float:
        """估算输出内容相对于输入的新颖度"""
        before_words = set(before.content.lower().split())
        after_words = set(after.content.lower().split())
        if not after_words:
            return 0.0
        new_words = after_words - before_words
        return len(new_words) / len(after_words)

    def _extract_claims(self, signal: NeuralSignal) -> list[str]:
        """从信号中提取断言"""
        claims = []

        # 从 trace 提取
        for step in signal.trace:
            if step.claim and step.claim.strip():
                claims.append(step.claim)

        # trace 为空时，从 content 中简单拆句
        if not claims and signal.content:
            for sentence in signal.content.replace("。", ".").replace("！", ".").replace("？", ".").split("."):
                s = sentence.strip()
                if len(s) > 5:
                    claims.append(s)

        return claims

    def _evaluate_quality(self, contributions: list[ContributionTrace],
                          circuit_result) -> float:
        """
        评估最终输出质量。

        质量属性:
        1. 收敛与否 (收敛 > 未收敛)
        2. 熵变化 (总熵降低 > 熵不变 > 熵升高)
        3. 链路效率 (少迭代 > 多迭代)
        4. 输出确定性 (置信度高 > 低)
        """
        if not contributions:
            return 0.0

        score = 0.5  # 中性起点

        # 收敛加成
        if circuit_result and circuit_result.converged:
            score += 0.2
            # 快速收敛更优
            if circuit_result.iterations <= 2:
                score += 0.1

        # 熵降低
        total_entropy_reduction = sum(c.entropy_reduction for c in contributions)
        score += max(-0.2, min(0.3, total_entropy_reduction * 0.5))

        # 最终信号置信度
        if circuit_result and circuit_result.final_signal:
            cert_score = {
                Certainty.VERIFIED: 0.15, Certainty.STRONG: 0.1,
                Certainty.MODERATE: 0.0, Certainty.WEAK: -0.1,
                Certainty.GUESS: -0.2,
            }.get(circuit_result.final_signal.certainty, 0.0)
            score += cert_score

        # 断言存活率
        all_claims = sum(len(c.claims_made) for c in contributions)
        all_survived = sum(c.claims_survived_count for c in contributions)
        if all_claims > 0:
            survival_rate = all_survived / all_claims
            score += (survival_rate - 0.5) * 0.2  # center at 0.5

        return max(0.0, min(1.0, score))

    def get_learning_summary(self) -> dict:
        """获取跨运行学习摘要"""
        if not self.execution_db:
            return {"status": "no_data"}

        qualities = [e.quality_score for e in self.execution_db]
        domains = {}
        for e in self.execution_db:
            if e.problem_domain not in domains:
                domains[e.problem_domain] = []
            domains[e.problem_domain].append(e.quality_score)

        return {
            "total_executions": len(self.execution_db),
            "total_updates": self.optimizer.total_updates,
            "avg_quality": sum(qualities) / len(qualities),
            "quality_trend": qualities[-5:] if len(qualities) >= 5 else qualities,
            "best_domain": max(domains, key=lambda d: sum(domains[d]) / len(domains[d])) if domains else "n/a",
            "synapse_count": len(self.optimizer.synapse_stats),
            "component_weights": self.analyzer.component_weights,
            "top_synapses": sorted(
                [(k, s.avg_credit) for k, s in self.optimizer.synapse_stats.items()],
                key=lambda x: x[1], reverse=True
            )[:5],
        }

    def export_learning_data(self, filepath: str):
        """导出学习数据供分析"""
        data = {
            "executions": [
                {
                    "trace_id": e.trace_id,
                    "quality_score": e.quality_score,
                    "domain": e.problem_domain,
                    "credits": [
                        {"neuron": c.neuron_id, "credit": c.credit,
                         "confidence": c.confidence, "explanation": c.explanation}
                        for c in e.credits
                    ],
                }
                for e in self.execution_db
            ],
            "synapse_stats": {
                k: {
                    "avg_credit": s.avg_credit,
                    "variance": s.credit_variance,
                    "momentum": s.momentum,
                    "specialization": s.specialization_score,
                }
                for k, s in self.optimizer.synapse_stats.items()
            },
            "component_weights": self.analyzer.component_weights,
        }
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)


# ═══════════════════════════════════════════════════════════════════════════════
# 对抗验证器 — 元认知层的一个具体实现
# ═══════════════════════════════════════════════════════════════════════════════

class AdversarialValidator:
    """
    对抗验证: 元认知神经元不仅要确认输出，还要主动寻找输出中的漏洞。

    工作方式:
    1. 接收执行层输出
    2. 切换到攻击者视角: "如果这是错的，可能错在哪？"
    3. 生成对抗测试用例
    4. 如果输出通过测试 → CONFIRM; 否则 → CHALLENGE 回退
    """

    def __init__(self, adapter):
        self.adapter = adapter

    async def validate(self, signal: NeuralSignal, original_question: str) -> NeuralSignal:
        """
        对抗验证一个神经信号。

        返回: CONFIRM 信号 (通过) 或 CHALLENGE 信号 (发现漏洞)
        """
        prompt = self._build_validation_prompt(signal, original_question)

        response = await self.adapter.call(prompt, {})

        # 解析结果
        if "PASS" in response.upper() and "FAIL" not in response.upper():
            return NeuralSignal(
                source_model="adversarial-validator",
                content=response,
                intent=Intent.CONFIRM,
                certainty=Certainty.STRONG,
            )
        else:
            return NeuralSignal(
                source_model="adversarial-validator",
                content=response,
                intent=Intent.CHALLENGE,
                certainty=Certainty.STRONG,
                expectation=Expectation(
                    action="refine",
                    focus=response,
                    urgency="immediate",
                ),
            )

    def _build_validation_prompt(self, signal: NeuralSignal, question: str) -> str:
        return f"""你是一个对抗验证器。你的任务是找出以下分析中的漏洞。

原始问题: {question}

分析结论: {signal.content}

断言链:
{chr(10).join(f'  {i+1}. {c.claim}' for i, c in enumerate(signal.trace))}

不确定性标注:
{chr(10).join(f'  - {u.about} (影响: {u.impact})' for u in signal.uncertainties)}

请逐一攻击每个断言:
1. 假设这个断言是错误的，找到可能的反例
2. 检查推理中是否有未声明的假设
3. 验证逻辑推导的每一步

如果所有断言成立且推理严密，回复 PASS。
如果发现任何漏洞，回复 FAIL 并详细说明漏洞位置和原因。"""
