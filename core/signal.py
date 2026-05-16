"""
signal.py — 神经信号定义

模型之间传递的不是文本，而是结构化的神经信号。
一个信号包含：内容、意图、推理轨迹、不确定性、期望动作。
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional
import time
import hashlib
import json


class Intent(Enum):
    """信号意图 — 告诉接收方"我为什么发这个信号" """
    PROPOSE = "propose"          # 提出方案，等待反馈
    VALIDATE = "validate"        # 请求验证我的推理
    CHALLENGE = "challenge"      # 质疑对方的结论
    REFINE = "refine"            # 在对方基础上改进
    DELEGATE = "delegate"        # 超出能力，委托给你
    INHIBIT = "inhibit"          # 停止，这个方向有问题
    CONFIRM = "confirm"          # 同意，可以执行
    QUERY = "query"              # 纯粹的信息查询
    REPORT = "report"            # 汇报执行结果


class Certainty(Enum):
    """归一化置信度等级 — 跨模型可比较"""
    GUESS = "guess"              # 0.0-0.3 纯猜测
    WEAK = "weak"                # 0.3-0.5 有一点依据
    MODERATE = "moderate"        # 0.5-0.7 有依据但未验证
    STRONG = "strong"            # 0.7-0.9 有较强依据
    VERIFIED = "verified"        # 0.9-1.0 已验证的事实


@dataclass
class TraceStep:
    """推理轨迹中的一步"""
    step_id: int
    claim: str                          # 这一步的断言
    basis: str                          # 依据是什么
    certainty: Certainty                # 这一步的确定程度
    dependencies: list[int] = field(default_factory=list)  # 依赖哪些前置步骤


@dataclass
class Uncertainty:
    """显式标注的不确定性"""
    about: str                          # 不确定什么
    impact: str                         # 如果判断错了影响多大 (low/medium/high/critical)
    resolvable_by: Optional[str] = None # 谁/什么能解决这个不确定性


@dataclass
class Expectation:
    """对接收方的期望"""
    action: str             # validate / refine / execute / answer
    focus: Optional[str] = None    # 聚焦在哪个部分
    urgency: str = "normal"        # immediate / normal / background


@dataclass
class NeuralSignal:
    """
    神经信号 — 模型间通信的基本单元

    不是一段文本，而是一个完整的认知状态快照。
    """
    # 元信息
    signal_id: str = ""
    source_model: str = ""              # 发送方模型标识
    target_model: str = ""              # 接收方 ("*" = 广播)
    timestamp: float = 0.0
    reply_to: Optional[str] = None      # 回复哪个信号

    # 核心内容
    content: str = ""                   # 主要内容（人类可读）
    intent: Intent = Intent.PROPOSE     # 意图
    certainty: Certainty = Certainty.MODERATE  # 整体置信度

    # 推理状态
    trace: list[TraceStep] = field(default_factory=list)
    uncertainties: list[Uncertainty] = field(default_factory=list)
    expectation: Optional[Expectation] = None

    # 上下文引用
    context_refs: list[str] = field(default_factory=list)  # 依赖的外部信息源
    constraints: list[str] = field(default_factory=list)   # 结论成立的前提条件

    def __post_init__(self):
        if not self.signal_id:
            raw = f"{self.source_model}{self.content}{time.time()}"
            self.signal_id = hashlib.sha256(raw.encode()).hexdigest()[:16]
        if not self.timestamp:
            self.timestamp = time.time()

    def to_dict(self) -> dict:
        """序列化为可传输的字典"""
        return {
            "signal_id": self.signal_id,
            "source_model": self.source_model,
            "target_model": self.target_model,
            "timestamp": self.timestamp,
            "reply_to": self.reply_to,
            "content": self.content,
            "intent": self.intent.value,
            "certainty": self.certainty.value,
            "trace": [
                {
                    "step_id": s.step_id,
                    "claim": s.claim,
                    "basis": s.basis,
                    "certainty": s.certainty.value,
                    "dependencies": s.dependencies,
                }
                for s in self.trace
            ],
            "uncertainties": [
                {
                    "about": u.about,
                    "impact": u.impact,
                    "resolvable_by": u.resolvable_by,
                }
                for u in self.uncertainties
            ],
            "expectation": {
                "action": self.expectation.action,
                "focus": self.expectation.focus,
                "urgency": self.expectation.urgency,
            } if self.expectation else None,
            "context_refs": self.context_refs,
            "constraints": self.constraints,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "NeuralSignal":
        """从字典反序列化"""
        signal = cls(
            signal_id=data["signal_id"],
            source_model=data["source_model"],
            target_model=data["target_model"],
            timestamp=data["timestamp"],
            reply_to=data.get("reply_to"),
            content=data["content"],
            intent=Intent(data["intent"]),
            certainty=Certainty(data["certainty"]),
            context_refs=data.get("context_refs", []),
            constraints=data.get("constraints", []),
        )
        signal.trace = [
            TraceStep(
                step_id=s["step_id"],
                claim=s["claim"],
                basis=s["basis"],
                certainty=Certainty(s["certainty"]),
                dependencies=s.get("dependencies", []),
            )
            for s in data.get("trace", [])
        ]
        signal.uncertainties = [
            Uncertainty(
                about=u["about"],
                impact=u["impact"],
                resolvable_by=u.get("resolvable_by"),
            )
            for u in data.get("uncertainties", [])
        ]
        if data.get("expectation"):
            signal.expectation = Expectation(
                action=data["expectation"]["action"],
                focus=data["expectation"].get("focus"),
                urgency=data["expectation"].get("urgency", "normal"),
            )
        return signal

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False, indent=2)

    @classmethod
    def from_json(cls, json_str: str) -> "NeuralSignal":
        return cls.from_dict(json.loads(json_str))
