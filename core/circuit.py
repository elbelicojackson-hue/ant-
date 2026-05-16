"""
circuit.py — 神经回路

将多个神经元连接成一个有反馈回路的链路。
这是整个协议的核心 — 实现了:
1. 前馈传播 (感知 → 推理 → 执行)
2. 反馈回退 (抑制层发现问题 → 回退到推理层)
3. 稀疏激活 (只激活相关神经元)
4. 可塑性 (成功路径被强化，失败路径被弱化)
"""

from dataclasses import dataclass, field
from typing import Optional
import asyncio
import time

from .signal import NeuralSignal, Intent, Certainty, Expectation
from .neuron import Neuron, NeuronRole


@dataclass
class Synapse:
    """突触 — 两个神经元之间的连接"""
    source: str         # 源神经元 model_id
    target: str         # 目标神经元 model_id
    weight: float       # 连接权重 (0.0-1.0)，影响信号传递强度
    signal_type: str    # 这条突触传递什么类型的信号 (forward/feedback/inhibit)

    def strengthen(self, amount: float = 0.05):
        """强化突触 (成功后)"""
        self.weight = min(1.0, self.weight + amount)

    def weaken(self, amount: float = 0.05):
        """弱化突触 (失败后)"""
        self.weight = max(0.0, self.weight - amount)


@dataclass
class CircuitResult:
    """回路执行结果"""
    final_signal: Optional[NeuralSignal]    # 最终输出信号
    signal_chain: list[NeuralSignal]        # 完整信号链路
    iterations: int                          # 经过了几轮迭代
    converged: bool                          # 是否收敛(达成共识)
    total_time_ms: float                     # 总耗时
    neurons_activated: list[str]             # 哪些神经元被激活了


class NeuralCircuit:
    """
    神经回路 — 多神经元协作的核心引擎

    工作流程:
    1. 输入信号进入感知层
    2. 感知层输出传递给推理层
    3. 推理层输出传递给抑制层检查
    4. 如果抑制层发现问题 → 回退信号传回推理层 → 重新推理
    5. 如果通过 → 传递给元认知层做最终审视
    6. 元认知通过 → 执行层输出最终结果
    7. 整个过程有最大迭代次数限制，防止死循环
    """

    def __init__(self, max_iterations: int = 5):
        self.neurons: dict[str, Neuron] = {}
        self.synapses: list[Synapse] = []
        self.max_iterations = max_iterations
        self.execution_history: list[CircuitResult] = []
        self._trace_on_before: callable = None
        self._trace_on_after: callable = None

    def add_neuron(self, neuron: Neuron):
        """添加神经元到回路"""
        self.neurons[neuron.model_id] = neuron

    def connect(self, source_id: str, target_id: str,
                weight: float = 0.8, signal_type: str = "forward"):
        """建立突触连接"""
        self.synapses.append(Synapse(
            source=source_id,
            target=target_id,
            weight=weight,
            signal_type=signal_type,
        ))

    def auto_wire(self):
        """
        自动布线 — 根据神经元角色自动建立标准连接

        标准链路:
        感知 → 推理 → 抑制 → 元认知 → 执行
                ↑        │
                └────────┘ (反馈回路)
        """
        by_role: dict[NeuronRole, list[Neuron]] = {}
        for neuron in self.neurons.values():
            by_role.setdefault(neuron.role, []).append(neuron)

        # 前馈连接
        role_order = [
            NeuronRole.PERCEPTION,
            NeuronRole.REASONING,
            NeuronRole.INHIBITION,
            NeuronRole.METACOGNITION,
            NeuronRole.EXECUTION,
        ]

        for i in range(len(role_order) - 1):
            sources = by_role.get(role_order[i], [])
            targets = by_role.get(role_order[i + 1], [])
            for s in sources:
                for t in targets:
                    self.connect(s.model_id, t.model_id, weight=0.8, signal_type="forward")

        # 反馈连接: 抑制层 → 推理层
        for inhibitor in by_role.get(NeuronRole.INHIBITION, []):
            for reasoner in by_role.get(NeuronRole.REASONING, []):
                self.connect(
                    inhibitor.model_id, reasoner.model_id,
                    weight=0.9, signal_type="feedback"
                )

        # 反馈连接: 元认知层 → 推理层
        for meta in by_role.get(NeuronRole.METACOGNITION, []):
            for reasoner in by_role.get(NeuronRole.REASONING, []):
                self.connect(
                    meta.model_id, reasoner.model_id,
                    weight=0.7, signal_type="feedback"
                )

    def set_tracer(self, on_before: callable = None, on_after: callable = None):
        """
        设置信号追踪回调 — 供 NeuralBackprop 使用。

        on_before(neuron_id, input_signal) → None
        on_after(neuron_id, input_signal, output_signal, elapsed_ms) → None
        """
        self._trace_on_before = on_before
        self._trace_on_after = on_after

    async def propagate(self, input_signal: NeuralSignal) -> CircuitResult:
        """
        信号传播 — 核心算法

        实现带反馈回路的信号传播:
        - 前馈: 信号沿链路向前传递
        - 反馈: 抑制信号触发回退
        - 收敛: 当信号通过所有检查或达到最大迭代次数时停止
        """
        start_time = time.time()
        signal_chain: list[NeuralSignal] = [input_signal]
        neurons_activated: list[str] = []
        current_signal = input_signal
        iterations = 0
        converged = False

        while iterations < self.max_iterations:
            iterations += 1

            # 找到当前信号应该传递给谁
            next_neurons = self._get_next_neurons(current_signal)

            if not next_neurons:
                # 没有下一个神经元了，链路结束
                converged = True
                break

            # 依次激活下游神经元
            response_signal = None
            for neuron in next_neurons:
                if neuron.should_activate(current_signal):
                    # Trace: before hook
                    if self._trace_on_before:
                        self._trace_on_before(neuron.model_id, current_signal)

                    t0 = time.time()
                    response_signal = await neuron.process(current_signal)
                    elapsed = (time.time() - t0) * 1000

                    # Trace: after hook
                    if self._trace_on_after and response_signal:
                        self._trace_on_after(neuron.model_id, current_signal,
                                            response_signal, elapsed)

                    if response_signal:
                        neurons_activated.append(neuron.model_id)
                        signal_chain.append(response_signal)
                        break  # 一次只激活一个（稀疏激活）

            if not response_signal:
                # 没有神经元响应，链路结束
                converged = True
                break

            # 检查是否收到抑制信号
            if response_signal.intent == Intent.INHIBIT:
                # 回退: 找到反馈连接，将信号传回上游
                feedback_target = self._get_feedback_target(response_signal)
                if feedback_target:
                    # 构造回退信号
                    rollback_signal = NeuralSignal(
                        source_model=response_signal.source_model,
                        target_model=feedback_target.model_id,
                        reply_to=response_signal.signal_id,
                        content=f"回退原因: {response_signal.content}",
                        intent=Intent.CHALLENGE,
                        certainty=response_signal.certainty,
                        expectation=Expectation(
                            action="refine",
                            focus=response_signal.content,
                            urgency="immediate",
                        ),
                    )
                    signal_chain.append(rollback_signal)
                    current_signal = rollback_signal
                    continue
                else:
                    # 没有反馈目标，链路终止
                    break

            # 检查是否是确认信号（只有抑制层/元认知层的确认才算收敛）
            if response_signal.intent == Intent.CONFIRM:
                responding_neuron = self.neurons.get(response_signal.source_model)
                if responding_neuron and responding_neuron.role in (
                    NeuronRole.METACOGNITION, NeuronRole.EXECUTION
                ):
                    # 最终层确认，链路收敛
                    converged = True
                    current_signal = response_signal
                    break
                else:
                    # 非最终层的确认，继续前馈传播
                    current_signal = response_signal
                    continue

            # 继续前馈传播
            current_signal = response_signal

        total_time = (time.time() - start_time) * 1000

        result = CircuitResult(
            final_signal=current_signal,
            signal_chain=signal_chain,
            iterations=iterations,
            converged=converged,
            total_time_ms=total_time,
            neurons_activated=neurons_activated,
        )

        # 可塑性: 根据结果调整突触权重
        self._apply_plasticity(result)

        self.execution_history.append(result)
        return result

    def _get_next_neurons(self, signal: NeuralSignal) -> list[Neuron]:
        """根据当前信号和突触连接，找到下游神经元"""
        next_ids = []

        # 如果信号有明确目标
        if signal.target_model and signal.target_model != "*" and signal.target_model != "":
            if signal.target_model in self.neurons:
                return [self.neurons[signal.target_model]]

        # 如果信号来自外部(不在回路中)，路由到感知层
        if signal.source_model not in self.neurons:
            for neuron in self.neurons.values():
                if neuron.role == NeuronRole.PERCEPTION:
                    next_ids.append(neuron.model_id)
            if next_ids:
                return [self.neurons[nid] for nid in next_ids]

        # 根据突触连接找下游
        for synapse in self.synapses:
            if synapse.source == signal.source_model and synapse.signal_type == "forward":
                if synapse.weight > 0.3:  # 权重太低的连接忽略
                    next_ids.append(synapse.target)

        # 如果是反馈信号，走反馈连接
        if signal.intent in (Intent.INHIBIT, Intent.CHALLENGE):
            for synapse in self.synapses:
                if synapse.source == signal.source_model and synapse.signal_type == "feedback":
                    if synapse.weight > 0.3:
                        next_ids.append(synapse.target)

        # 去重并按权重排序
        seen = set()
        neurons = []
        for nid in next_ids:
            if nid not in seen and nid in self.neurons:
                seen.add(nid)
                neurons.append(self.neurons[nid])

        return neurons

    def _get_feedback_target(self, inhibit_signal: NeuralSignal) -> Optional[Neuron]:
        """找到抑制信号应该回退到的目标神经元"""
        for synapse in self.synapses:
            if (synapse.source == inhibit_signal.source_model
                    and synapse.signal_type == "feedback"):
                if synapse.target in self.neurons:
                    return self.neurons[synapse.target]
        return None

    def _apply_plasticity(self, result: CircuitResult):
        """
        可塑性 — 根据执行结果调整突触权重

        - 收敛且迭代少 → 强化路径 (这条路走得通)
        - 不收敛或迭代多 → 弱化路径 (这条路有问题)
        """
        if result.converged and result.iterations <= 2:
            # 快速收敛 — 强化所有参与的突触
            for synapse in self.synapses:
                if (synapse.source in result.neurons_activated
                        or synapse.target in result.neurons_activated):
                    synapse.strengthen(0.03)

        elif not result.converged or result.iterations >= self.max_iterations:
            # 未收敛或迭代过多 — 弱化
            for synapse in self.synapses:
                if (synapse.source in result.neurons_activated
                        or synapse.target in result.neurons_activated):
                    synapse.weaken(0.02)

    def get_topology(self) -> dict:
        """获取回路拓扑结构"""
        return {
            "neurons": {
                nid: {
                    "role": n.role.value,
                    "model": n.config.model_id,
                    "stats": n.get_stats(),
                }
                for nid, n in self.neurons.items()
            },
            "synapses": [
                {
                    "source": s.source,
                    "target": s.target,
                    "weight": round(s.weight, 3),
                    "type": s.signal_type,
                }
                for s in self.synapses
            ],
            "total_executions": len(self.execution_history),
        }
