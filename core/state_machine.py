"""
state_machine.py — 思考状态机

追踪每个模型节点的实时状态，带精确时间戳。
让系统（和用户）随时知道：谁在想什么，想了多久，卡在哪里。

状态流转:
  IDLE → THINKING → RESPONDING → DONE
                  → TIMEOUT → RETRYING → RESPONDING → DONE
                  → FAILED
"""

from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime, timezone
import time


class ThinkingState(Enum):
    IDLE = "idle"               # 空闲，等待信号
    THINKING = "thinking"       # 正在推理（已发送请求，等待响应）
    RESPONDING = "responding"   # 收到响应，正在解析
    RETRYING = "retrying"       # 超时后重试中
    DONE = "done"               # 完成
    FAILED = "failed"           # 失败
    INHIBITED = "inhibited"     # 被抑制，跳过


@dataclass
class StateTransition:
    """一次状态转换记录"""
    from_state: ThinkingState
    to_state: ThinkingState
    timestamp: str              # ISO 格式时间戳
    epoch_ms: float             # 毫秒级精度
    reason: str = ""            # 转换原因
    duration_ms: float = 0.0    # 在前一个状态停留了多久


@dataclass
class NodeState:
    """单个模型节点的状态"""
    model_id: str
    role: str
    current_state: ThinkingState = ThinkingState.IDLE
    task: str = ""                                      # 当前在做什么
    state_entered_at: float = 0.0                       # 进入当前状态的时间
    total_thinking_ms: float = 0.0                      # 累计思考时间
    call_count: int = 0                                 # 调用次数
    retry_count: int = 0                                # 重试次数
    transitions: list[StateTransition] = field(default_factory=list)

    @property
    def time_in_current_state_ms(self) -> float:
        """在当前状态已经待了多久"""
        if self.state_entered_at == 0:
            return 0.0
        return (time.time() - self.state_entered_at) * 1000

    @property
    def is_active(self) -> bool:
        return self.current_state in (ThinkingState.THINKING, ThinkingState.RETRYING)


class StateMachine:
    """
    思考状态机 — 管理所有模型节点的状态

    功能:
    1. 追踪每个模型的实时状态
    2. 记录精确时间戳（毫秒级）
    3. 记录状态转换历史
    4. 提供实时状态面板输出
    """

    def __init__(self):
        self.nodes: dict[str, NodeState] = {}
        self.global_start: float = 0.0
        self.events: list[dict] = []  # 全局事件日志

    def register(self, model_id: str, role: str):
        """注册一个模型节点"""
        self.nodes[model_id] = NodeState(model_id=model_id, role=role)

    def start_session(self):
        """开始一次推敲会话"""
        self.global_start = time.time()
        self._log_event("session_start", "推敲会话开始")

    def transition(self, model_id: str, new_state: ThinkingState,
                   reason: str = "", task: str = ""):
        """触发状态转换"""
        if model_id not in self.nodes:
            return

        node = self.nodes[model_id]
        old_state = node.current_state
        now = time.time()

        # 计算在前一个状态停留的时间
        duration_ms = 0.0
        if node.state_entered_at > 0:
            duration_ms = (now - node.state_entered_at) * 1000

        # 如果从 THINKING 转出，累加思考时间
        if old_state == ThinkingState.THINKING:
            node.total_thinking_ms += duration_ms

        # 记录转换
        transition = StateTransition(
            from_state=old_state,
            to_state=new_state,
            timestamp=self._now_iso(),
            epoch_ms=now * 1000,
            reason=reason,
            duration_ms=duration_ms,
        )
        node.transitions.append(transition)

        # 更新状态
        node.current_state = new_state
        node.state_entered_at = now
        if task:
            node.task = task

        # 计数
        if new_state == ThinkingState.THINKING:
            node.call_count += 1
        elif new_state == ThinkingState.RETRYING:
            node.retry_count += 1

        # 打印状态变化
        elapsed = self._elapsed_str()
        state_str = self._state_icon(new_state)
        print(f"    [{elapsed}] {state_str} {model_id}: {reason or task or new_state.value}")

        # 记录全局事件
        self._log_event("transition", f"{model_id}: {old_state.value} → {new_state.value}", {
            "model_id": model_id,
            "from": old_state.value,
            "to": new_state.value,
            "reason": reason,
            "duration_ms": round(duration_ms, 1),
        })

    def get_status(self) -> str:
        """获取当前所有节点的状态面板"""
        lines = []
        lines.append(f"  ┌{'─' * 56}┐")
        lines.append(f"  │ {'模型':<20} {'状态':<10} {'耗时':>8} {'任务':<16} │")
        lines.append(f"  ├{'─' * 56}┤")

        for node in self.nodes.values():
            icon = self._state_icon(node.current_state)
            state_name = node.current_state.value
            time_str = f"{node.time_in_current_state_ms/1000:.1f}s"
            task_str = node.task[:14] if node.task else "-"
            lines.append(
                f"  │ {icon} {node.model_id:<18} {state_name:<10} {time_str:>6} {task_str:<14} │"
            )

        lines.append(f"  └{'─' * 56}┘")
        return "\n".join(lines)

    def get_summary(self) -> dict:
        """获取会话统计摘要"""
        total_elapsed = (time.time() - self.global_start) * 1000 if self.global_start else 0

        return {
            "total_elapsed_ms": round(total_elapsed, 1),
            "nodes": {
                nid: {
                    "role": node.role,
                    "state": node.current_state.value,
                    "total_thinking_ms": round(node.total_thinking_ms, 1),
                    "call_count": node.call_count,
                    "retry_count": node.retry_count,
                }
                for nid, node in self.nodes.items()
            },
        }

    def _elapsed_str(self) -> str:
        """从会话开始到现在的时间"""
        if not self.global_start:
            return "00:00"
        elapsed = time.time() - self.global_start
        minutes = int(elapsed // 60)
        seconds = int(elapsed % 60)
        return f"{minutes:02d}:{seconds:02d}"

    def _now_iso(self) -> str:
        """当前时间 ISO 格式"""
        return datetime.now(timezone.utc).isoformat(timespec="milliseconds")

    def _state_icon(self, state: ThinkingState) -> str:
        """状态图标"""
        icons = {
            ThinkingState.IDLE: "○",
            ThinkingState.THINKING: "◉",
            ThinkingState.RESPONDING: "◈",
            ThinkingState.RETRYING: "↻",
            ThinkingState.DONE: "✓",
            ThinkingState.FAILED: "✗",
            ThinkingState.INHIBITED: "⊘",
        }
        return icons.get(state, "?")

    def _log_event(self, event_type: str, message: str, data: dict = None):
        """记录全局事件"""
        self.events.append({
            "type": event_type,
            "timestamp": self._now_iso(),
            "elapsed_ms": round((time.time() - self.global_start) * 1000, 1) if self.global_start else 0,
            "message": message,
            "data": data or {},
        })
