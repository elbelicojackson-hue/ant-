"""
neuron.py — 神经元节点

每个模型被包装为一个"神经元"。
神经元有输入突触(接收信号)和输出突触(发送信号)。
神经元内部是一个 LLM，但对外暴露的是统一的信号接口。
"""

from dataclasses import dataclass, field
from typing import Callable, Optional
from enum import Enum
import time

from .signal import NeuralSignal, Intent, Certainty, Expectation


class NeuronRole(Enum):
    """神经元在链路中的角色"""
    PERCEPTION = "perception"       # 感知层 — 快速反应，初步判断
    REASONING = "reasoning"         # 推理层 — 深度思考，逻辑推演
    INHIBITION = "inhibition"       # 抑制层 — 检测错误，发出停止信号
    METACOGNITION = "metacognition" # 元认知层 — 审视推理过程本身
    EXECUTION = "execution"         # 执行层 — 将决策转化为行动


@dataclass
class NeuronConfig:
    """神经元配置"""
    model_id: str                       # 模型标识 (如 "gpt-4o-mini", "claude-haiku")
    role: NeuronRole                    # 在链路中的角色
    activation_threshold: float = 0.3   # 激活阈值 — 信号强度低于此值不响应
    max_response_time_ms: int = 5000    # 最大响应时间
    cost_per_call: float = 0.0          # 每次调用成本 (用于路由优化)
    capabilities: list[str] = field(default_factory=list)  # 擅长什么


class Neuron:
    """
    神经元 — 链路中的基本计算单元

    包装一个 LLM，使其能通过 NeuralSignal 与其他神经元通信。
    """

    def __init__(self, config: NeuronConfig, llm_call: Callable):
        """
        Args:
            config: 神经元配置
            llm_call: 实际调用 LLM 的函数
                      签名: async (prompt: str, context: dict) -> str
        """
        self.config = config
        self.llm_call = llm_call
        self.history: list[NeuralSignal] = []       # 处理过的信号历史
        self.activation_count: int = 0
        self.success_count: int = 0
        self.failure_count: int = 0
        self._inhibited: bool = False

    @property
    def model_id(self) -> str:
        return self.config.model_id

    @property
    def role(self) -> NeuronRole:
        return self.config.role

    def should_activate(self, signal: NeuralSignal) -> bool:
        """
        判断是否应该响应这个信号

        核心原则: 如果信号通过突触连接到达了这个神经元，默认应该激活。
        只有在以下情况才不激活:
        - 被抑制状态
        - 信号强度极低且不紧急
        """
        if self._inhibited:
            return False

        # 置信度映射为数值
        certainty_values = {
            Certainty.GUESS: 0.15,
            Certainty.WEAK: 0.4,
            Certainty.MODERATE: 0.6,
            Certainty.STRONG: 0.8,
            Certainty.VERIFIED: 0.95,
        }

        signal_strength = certainty_values.get(signal.certainty, 0.5)

        # INHIBIT 信号: 所有神经元必须响应
        if signal.intent == Intent.INHIBIT:
            return True

        # 信号强度检查: 低于阈值不激活
        if signal_strength < self.config.activation_threshold:
            return False

        # 通过了阈值检查就激活
        # 信号能到达这个神经元说明突触连接存在，应该处理
        return True

    async def process(self, signal: NeuralSignal) -> Optional[NeuralSignal]:
        """
        处理输入信号，产生输出信号

        这是神经元的核心方法:
        1. 将 NeuralSignal 转化为 LLM 可理解的 prompt
        2. 调用 LLM
        3. 将 LLM 输出解析为新的 NeuralSignal
        """
        if not self.should_activate(signal):
            return None

        self.activation_count += 1

        # 构建 prompt — 将信号转化为模型能理解的指令
        prompt = self._build_prompt(signal)

        # 调用 LLM
        try:
            raw_output = await self.llm_call(prompt, self._build_context(signal))
        except Exception as e:
            self.failure_count += 1
            return self._create_error_signal(signal, str(e))

        # 解析输出为新信号
        response_signal = self._parse_output(raw_output, signal)

        self.history.append(signal)
        self.success_count += 1

        return response_signal

    def inhibit(self):
        """抑制此神经元 — 暂时停止响应"""
        self._inhibited = True

    def activate(self):
        """解除抑制"""
        self._inhibited = False

    def _build_prompt(self, signal: NeuralSignal) -> str:
        """将信号转化为 LLM prompt"""

        role_instructions = {
            NeuronRole.PERCEPTION: (
                "你是感知层。快速分析输入，给出初步判断。"
                "不需要深度推理，重点是速度和模式识别。"
            ),
            NeuronRole.REASONING: (
                "你是推理层。基于输入进行深度逻辑推演。"
                "每一步推理都要有明确依据。标注你不确定的部分。"
            ),
            NeuronRole.INHIBITION: (
                "你是抑制层。你的职责是找出推理中的错误和漏洞。"
                "如果发现逻辑问题，明确指出哪一步有问题以及为什么。"
                "如果没有问题，回复 PASS。"
            ),
            NeuronRole.METACOGNITION: (
                "你是元认知层。审视整个推理过程的质量。"
                "检查: 前提是否成立？逻辑链是否完整？有无跳跃？"
                "结论是否过度自信？是否遗漏了重要因素？"
            ),
            NeuronRole.EXECUTION: (
                "你是执行层。将已确认的决策转化为具体可执行的操作。"
                "输出精确的命令或步骤，不做额外推理。"
            ),
        }

        parts = []
        parts.append(f"[系统角色] {role_instructions.get(self.role, '')}")
        parts.append(f"\n[输入信号]")
        parts.append(f"来源: {signal.source_model}")
        parts.append(f"意图: {signal.intent.value}")
        parts.append(f"置信度: {signal.certainty.value}")
        parts.append(f"内容: {signal.content}")

        if signal.trace:
            parts.append(f"\n[推理轨迹]")
            for step in signal.trace:
                parts.append(
                    f"  步骤{step.step_id}: {step.claim} "
                    f"(依据: {step.basis}, 确定度: {step.certainty.value})"
                )

        if signal.uncertainties:
            parts.append(f"\n[已知不确定性]")
            for u in signal.uncertainties:
                parts.append(f"  - {u.about} (影响: {u.impact})")

        if signal.expectation:
            parts.append(f"\n[期望你做]")
            parts.append(f"  动作: {signal.expectation.action}")
            if signal.expectation.focus:
                parts.append(f"  聚焦: {signal.expectation.focus}")

        if signal.constraints:
            parts.append(f"\n[前提约束]")
            for c in signal.constraints:
                parts.append(f"  - {c}")

        parts.append(f"\n[输出要求]")
        parts.append("请用以下格式回复:")
        parts.append("INTENT: propose/validate/challenge/refine/inhibit/confirm")
        parts.append("CERTAINTY: guess/weak/moderate/strong/verified")
        parts.append("CONTENT: 你的回复内容")
        parts.append("TRACE: 步骤1: 断言 | 依据 | 确定度")
        parts.append("UNCERTAINTIES: 不确定点1 | 影响程度")

        return "\n".join(parts)

    def _build_context(self, signal: NeuralSignal) -> dict:
        """构建调用上下文"""
        return {
            "role": self.role.value,
            "signal_id": signal.signal_id,
            "history_length": len(self.history),
            "recent_signals": [s.content[:100] for s in self.history[-3:]],
        }

    def _parse_output(self, raw_output: str, input_signal: NeuralSignal) -> NeuralSignal:
        """将 LLM 原始输出解析为 NeuralSignal"""

        # 默认值: 根据角色设定合理的默认意图
        role_default_intent = {
            NeuronRole.PERCEPTION: Intent.PROPOSE,
            NeuronRole.REASONING: Intent.PROPOSE,
            NeuronRole.INHIBITION: Intent.CONFIRM,  # 抑制层默认通过
            NeuronRole.METACOGNITION: Intent.CONFIRM,
            NeuronRole.EXECUTION: Intent.CONFIRM,
        }

        intent = role_default_intent.get(self.role, Intent.PROPOSE)
        certainty = Certainty.MODERATE
        content = raw_output

        lines = raw_output.strip().split("\n")
        parsed_content_parts = []
        found_structured = False

        for line in lines:
            line_stripped = line.strip()
            if line_stripped.startswith("INTENT:"):
                intent_str = line_stripped[7:].strip().lower()
                try:
                    intent = Intent(intent_str)
                    found_structured = True
                except ValueError:
                    pass
            elif line_stripped.startswith("CERTAINTY:"):
                cert_str = line_stripped[10:].strip().lower()
                try:
                    certainty = Certainty(cert_str)
                    found_structured = True
                except ValueError:
                    pass
            elif line_stripped.startswith("CONTENT:"):
                parsed_content_parts.append(line_stripped[8:].strip())
                found_structured = True
            elif found_structured and parsed_content_parts:
                parsed_content_parts.append(line_stripped)

        if parsed_content_parts:
            content = "\n".join(parsed_content_parts)

        # 如果模型没有按格式输出，根据角色强制设定意图
        if not found_structured:
            intent = role_default_intent.get(self.role, Intent.PROPOSE)
            certainty = Certainty.MODERATE

        # 抑制层特殊逻辑: 如果输出中包含否定/问题关键词，设为 INHIBIT
        if self.role == NeuronRole.INHIBITION:
            inhibit_keywords = ["问题", "错误", "不对", "漏洞", "风险", "矛盾",
                                "error", "wrong", "flaw", "risk", "issue"]
            pass_keywords = ["PASS", "通过", "没有问题", "逻辑正确", "合理"]

            output_lower = raw_output.lower()
            has_inhibit = any(kw in raw_output for kw in inhibit_keywords)
            has_pass = any(kw in raw_output for kw in pass_keywords)

            if has_inhibit and not has_pass:
                intent = Intent.INHIBIT
            else:
                intent = Intent.CONFIRM

        return NeuralSignal(
            source_model=self.config.model_id,
            target_model="",  # 不指定目标，由回路根据突触决定
            reply_to=input_signal.signal_id,
            content=content,
            intent=intent,
            certainty=certainty,
        )

    def _create_error_signal(self, input_signal: NeuralSignal, error: str) -> NeuralSignal:
        """创建错误信号"""
        return NeuralSignal(
            source_model=self.config.model_id,
            target_model=input_signal.source_model,
            reply_to=input_signal.signal_id,
            content=f"处理失败: {error}",
            intent=Intent.REPORT,
            certainty=Certainty.VERIFIED,
        )

    def get_stats(self) -> dict:
        """获取神经元统计信息"""
        return {
            "model_id": self.config.model_id,
            "role": self.role.value,
            "activation_count": self.activation_count,
            "success_rate": (
                self.success_count / self.activation_count
                if self.activation_count > 0 else 0.0
            ),
            "inhibited": self._inhibited,
        }
