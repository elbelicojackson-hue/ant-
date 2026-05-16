"""
server.py — FastAPI 后端，连接 Arena chain-centric 引擎到 Web UI

通过 SSE 实时推送 Arena 运行事件到前端。
不修改 core/arena.py，而是通过包装器拦截状态变化。
"""

import asyncio
import sys
import os
import time
import uuid
import traceback
from datetime import datetime
from pathlib import Path
from dataclasses import dataclass, field
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel
from sse_starlette.sse import EventSourceResponse

# 确保项目根目录在 sys.path 中
PROJECT_ROOT = str(Path(__file__).resolve().parent.parent)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from core.arena import Arena, ReasoningChain, ChainStep, Attack
from core.adapter import (
    OpenAIAdapter, DeepSeekAdapter, QwenAdapter,
    DoubaoAdapter, MiMoAdapter,
)


# ═══════════════════════════════════════════════════════════════
# App & Config
# ═══════════════════════════════════════════════════════════════

app = FastAPI(title="NeuralComm Lab API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

STATIC_DIR = Path(__file__).parent / "static"
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


# ═══════════════════════════════════════════════════════════════
# Session Management
# ═══════════════════════════════════════════════════════════════

@dataclass
class Session:
    session_id: str
    question: str
    status: str = "running"          # running / finished / stopped / error
    queue: asyncio.Queue = field(default_factory=asyncio.Queue)
    task: asyncio.Task | None = None
    created_at: str = ""
    finished_at: str = ""
    result_summary: dict = field(default_factory=dict)


sessions: dict[str, Session] = {}


# ═══════════════════════════════════════════════════════════════
# Request / Response Models
# ═══════════════════════════════════════════════════════════════

class RunRequest(BaseModel):
    question: str
    max_rounds: int = 50
    convergence_threshold: float = 0.25
    verifier: str = "auto"  # auto / enabled / disabled


class SessionInfo(BaseModel):
    session_id: str
    question: str
    status: str
    created_at: str
    finished_at: str


# ═══════════════════════════════════════════════════════════════
# Model Factory (same config as run_arena.py)
# ═══════════════════════════════════════════════════════════════

MODEL_NAMES = [
    "MiMo-V2.5", "Kimi-K2.6", "Doubao",
]


def create_models() -> dict:
    """3模型辩论 + DeepSeek 专职查证"""
    return {
        "MiMo-V2.5": MiMoAdapter().call_with_retry,
        "Kimi-K2.6": QwenAdapter(model="kimi-k2.6").call_with_retry,
        "Doubao": DoubaoAdapter().call_with_retry,
    }


# ═══════════════════════════════════════════════════════════════
# Arena Wrapper — 拦截状态变化，推送 SSE 事件
# ═══════════════════════════════════════════════════════════════

class ArenaEventEmitter:
    """
    包装 Arena 运行过程，通过轮询 Arena 内部状态来发射事件。
    不修改 arena.py，而是在外部观察状态变化。
    """

    def __init__(self, session: Session, arena: Arena, question: str):
        self.session = session
        self.arena = arena
        self.question = question
        self.queue = session.queue
        self._prev_chains: dict[str, dict] = {}
        self._prev_attacks_count: int = 0

    async def emit(self, event_type: str, data: dict):
        """推送一个 SSE 事件到队列"""
        await self.queue.put({"event": event_type, "data": data})

    async def run(self):
        """运行 Arena 并发射事件"""
        start_time = time.time()

        await self.emit("session_started", {
            "session_id": self.session.session_id,
            "question": self.question,
            "models": MODEL_NAMES,
            "mode": "arena",
        })

        try:
            result = await self._run_with_events(start_time)

            # session_finished
            elapsed = time.time() - start_time
            chains_summary = []
            for chain in self.arena.chains.values():
                if chain.status == "archived":
                    continue
                chains_summary.append({
                    "agent_id": chain.agent_id,
                    "chain_id": chain.chain_id,
                    "version": chain.version,
                    "status": chain.status,
                    "steps_count": len(chain.steps),
                })

            attacks_summary = {
                "total": len(self.arena.attacks),
                "successful": sum(1 for a in self.arena.attacks if a.success),
                "failed": sum(1 for a in self.arena.attacks if not a.success),
            }

            finish_data = {
                "convergence_reason": result.convergence_reason,
                "total_rounds": result.total_rounds,
                "elapsed_seconds": round(elapsed, 1),
                "consensus": result.consensus,
                "disputes": result.disputes,
                "chains_summary": chains_summary,
                "attacks_summary": attacks_summary,
            }
            await self.emit("session_finished", finish_data)
            self.session.status = "finished"
            self.session.finished_at = datetime.now().isoformat()
            self.session.result_summary = finish_data

        except asyncio.CancelledError:
            await self.emit("session_finished", {
                "convergence_reason": "user_stopped",
                "total_rounds": 0,
                "elapsed_seconds": round(time.time() - start_time, 1),
                "consensus": [],
                "disputes": [],
                "chains_summary": [],
                "attacks_summary": {},
            })
            self.session.status = "stopped"
            self.session.finished_at = datetime.now().isoformat()

        except Exception as e:
            tb = traceback.format_exc()
            await self.emit("session_error", {
                "error": str(e),
                "traceback": tb,
            })
            self.session.status = "error"
            self.session.finished_at = datetime.now().isoformat()

    async def _run_with_events(self, start_time: float):
        """
        运行 Arena debate，在关键节点拦截状态并发射事件。

        策略: 不修改 arena.py，而是重写 debate 的逻辑在外部，
        调用 Arena 的内部方法并在每步之间发射事件。
        """
        arena = self.arena
        arena.state.start_session()

        utterances = []
        relations = []
        entropy_history = []
        convergence_reason = "max_rounds"

        # === Round 0: 构建推理链 (逐个模型返回时立即推送) ===
        await self.emit("round_started", {"round": 0, "phase": "build_chains"})

        prompt = f"""分析以下问题，给出你的推理链。

问题: {self.question}

要求: 将你的分析拆分为 3-7 个步骤，每步一个独立断言。
用以下格式输出:

Step1: [你的第一个断言或前提]
Step2: [基于Step1的推理或新前提]
Step3: [进一步推理]
...
StepN: [最终结论]

每个 Step 必须是一个可以被独立攻击的断言。
不要写成散文，必须是 Step1/Step2/... 格式。"""

        # 并行调用所有模型，但用 as_completed 逐个处理
        import hashlib as _hashlib
        tasks_map = {}
        for mid, call_fn in arena.models.items():
            task = asyncio.create_task(
                arena._call_model(mid, call_fn, prompt, 0, start_time)
            )
            tasks_map[task] = mid

        for completed_task in asyncio.as_completed(tasks_map.keys()):
            try:
                result = await completed_task
                utterances.append(result)

                # 立即通知前端: 这个模型返回了
                await self.emit("model_responded", {
                    "agent_id": result.source,
                    "round": 0,
                    "content_length": len(result.content),
                })

                # 解析推理链
                chain = arena._parse_chain(result.source, result.content, 0)
                if chain:
                    arena.chains[chain.chain_id] = chain
                    await self.emit("chain_built", {
                        "agent_id": chain.agent_id,
                        "chain_id": chain.chain_id,
                        "version": chain.version,
                        "steps": [
                            {"index": s.index, "content": s.content, "status": s.status}
                            for s in chain.steps
                        ],
                    })
                else:
                    # Fallback
                    fallback_chain = ReasoningChain(
                        chain_id=f"c_{_hashlib.sha256(f'{result.source}_{time.time()}'.encode()).hexdigest()[:10]}",
                        agent_id=result.source,
                        steps=[ChainStep(index=1, content=result.content[:200])],
                        created_round=0,
                        last_modified_round=0,
                    )
                    arena.chains[fallback_chain.chain_id] = fallback_chain
                    await self.emit("chain_built", {
                        "agent_id": result.source,
                        "chain_id": fallback_chain.chain_id,
                        "version": 1,
                        "steps": [{"index": 1, "content": result.content[:200], "status": "active"}],
                    })
            except Exception as e:
                await self.emit("model_error", {
                    "agent_id": tasks_map.get(completed_task, "unknown"),
                    "round": 0,
                    "error": str(e)[:200],
                })

        entropy = arena._compute_entropy(utterances)
        entropy_history.append({"round": 0, **entropy.to_dict()})
        await self._emit_entropy(0, entropy)

        # Round summary for round 0
        await self.emit("round_summary", {
            "round": 0,
            "attacks_count": 0,
            "successful_count": 0,
            "active_chains": len([c for c in arena.chains.values() if c.status == "active"]),
            "broken_chains": len([c for c in arena.chains.values() if c.status == "broken"]),
        })

        # === Round 1+: 攻击循环 ===
        no_successful_attack_rounds = 0

        for round_num in range(1, arena.max_rounds):
            # 检查是否被取消
            if self.session.status == "stopped":
                convergence_reason = "user_stopped"
                break

            await self.emit("round_started", {"round": round_num, "phase": "attack"})

            # === 攻击轮: 逐个模型实时推送 ===
            # 构建 prompt (和 arena._round_attack 一样的逻辑)
            attack_tasks = {}
            for mid, call_fn in arena.models.items():
                visible_chains = arena._route_chains_for_attacker(mid, round_num)
                my_chain = arena._get_agent_active_chain(mid)
                my_chain_display = arena._format_single_chain(my_chain) if my_chain else "(无链)"
                evidence_text = arena.evidence_pool.format_all() if arena.evidence_pool.evidences else ""
                grounded_text = arena.tool_registry.format_grounded_facts()

                prompt = f"""问题: {self.question}

以下是其他参与者当前的推理链:
{visible_chains}

{evidence_text}

{grounded_text}

你的当前推理链:
{my_chain_display}

任务: 从其他参与者的推理链中，选择一个最薄弱的步骤进行攻击。

要求:
1. 明确指出攻击目标: 哪个参与者的哪个 Step
2. 给出攻击理由: 为什么这个步骤不成立
3. 说明崩塌影响: 如果成功，哪些后续步骤崩塌

输出格式:
ATTACK_TARGET: [参与者名称].Step[N]
ATTACK_REASON: [具体的攻击理由]
COLLAPSE_IMPACT: [崩塌影响]
CONFIDENCE: [高/中/低]

如果找不到漏洞: NO_ATTACK: [原因]"""

                task = asyncio.create_task(
                    arena._call_model(mid, call_fn, prompt, round_num, start_time)
                )
                attack_tasks[task] = mid

            # 逐个模型返回时立即推送
            round_attacks = []
            round_utterances = []
            for completed in asyncio.as_completed(attack_tasks.keys()):
                try:
                    result = await completed
                    round_utterances.append(result)

                    await self.emit("model_responded", {
                        "agent_id": result.source,
                        "round": round_num,
                        "content_length": len(result.content),
                        "phase": "attack",
                    })

                    # 解析攻击
                    attack = arena._parse_attack(result.source, result.content, round_num)
                    if attack:
                        target_chain = arena.chains.get(attack.target_chain_id)
                        target_agent = target_chain.agent_id if target_chain else "?"

                        await self.emit("attack_started", {
                            "round": round_num,
                            "attacker_id": attack.attacker_id,
                            "target_agent": target_agent,
                            "target_step": attack.target_step_index,
                            "reason": attack.reason[:100],
                        })
                        round_attacks.append(attack)
                except Exception as e:
                    await self.emit("model_error", {
                        "agent_id": attack_tasks.get(completed, "?"),
                        "round": round_num,
                        "error": str(e)[:100],
                    })

            utterances.extend(round_utterances)

            # === 去中心化判定: 所有非攻击方并行投票 ===
            if round_attacks:
                judged_attacks = await self._judge_attacks_realtime(
                    arena, round_num, start_time, round_attacks
                )
                arena.attacks.extend(judged_attacks)
                round_attacks = judged_attacks

            successful_this_round = sum(1 for a in round_attacks if a.success)

    async def _judge_attacks_realtime(self, arena, round_num, start_time, attacks):
        """
        去中心化判定: 所有非攻击方节点并行投票，多数决
        没有裁判，没有特权节点。每个节点平等投票。
        所有投票并行发起 (asyncio.gather) → 速度接近单次调用。
        """
        from core.arena import Attack

        for i, attack in enumerate(attacks):
            target_chain = arena.chains.get(attack.target_chain_id)
            if not target_chain:
                continue
            target_step = None
            for step in target_chain.steps:
                if step.index == attack.target_step_index:
                    target_step = step
                    break
            if not target_step:
                continue

            # 所有非攻击方节点并行投票
            voters = [mid for mid in arena.models if mid != attack.attacker_id]
            if not voters:
                continue

            vote_prompt = f"""一个推理步骤被攻击了。判断攻击是否成立。
被攻击的断言: {target_step.content}
攻击理由: {attack.reason[:200]}
这个攻击是否指出了真正的逻辑漏洞或事实错误？
只输出: COLLAPSED 或 DEFENDED"""

            # 并行投票 (一次 gather)
            vote_tasks = []
            for voter_id in voters:
                call_fn = arena.models[voter_id]
                vote_tasks.append(arena._call_model(voter_id, call_fn, vote_prompt, round_num, start_time))

            await self.emit("vote_started", {
                "round": round_num,
                "attack_index": i,
                "voters": voters,
                "target": f"{target_chain.agent_id}.Step{attack.target_step_index}",
            })

            vote_results = await asyncio.gather(*vote_tasks, return_exceptions=True)

            votes_collapse = 0
            votes_defend = 0
            for voter_id, result in zip(voters, vote_results):
                if isinstance(result, Exception):
                    continue
                content_upper = result.content.upper()
                if "COLLAPSED" in content_upper:
                    votes_collapse += 1
                    verdict = "collapsed"
                else:
                    votes_defend += 1
                    verdict = "defended"
                await self.emit("vote_cast", {
                    "round": round_num,
                    "voter": voter_id,
                    "verdict": verdict,
                })

            # 多数决 (无特权节点)
            total_votes = votes_collapse + votes_defend
            final_collapsed = votes_collapse > votes_defend and total_votes > 0

            if final_collapsed:
                attacks[i].success = True
                for step in target_chain.steps:
                    if step.index == attack.target_step_index:
                        step.status = "collapsed"
                        step.attacked_count += 1
                        break
                collapsed_steps = [s for s in target_chain.steps if s.status == "collapsed"]
                if target_chain.steps and target_chain.steps[-1].status == "collapsed":
                    target_chain.status = "broken"
                elif len(collapsed_steps) >= len(target_chain.steps) // 2:
                    target_chain.status = "broken"
            else:
                attacks[i].success = False
                for step in target_chain.steps:
                    if step.index == attack.target_step_index:
                        step.attacked_count += 1
                        if step.attacked_count >= 2:
                            step.status = "fortified"
                        break

            await self.emit("attack_result", {
                "round": round_num,
                "attacker_id": attack.attacker_id,
                "target_agent": target_chain.agent_id,
                "target_step": attack.target_step_index,
                "success": final_collapsed,
                "reason": attack.reason[:150],
                "votes": f"{votes_collapse}:{votes_defend}",
            })

        return attacks

    async def _emit_entropy(self, round_num: int, entropy):
        """发射熵更新事件"""
        await self.emit("entropy_update", {
            "round": round_num,
            "entropy": {
                "semantic": round(entropy.semantic, 3),
                "causal": round(entropy.causal, 3),
                "boundary": round(entropy.boundary, 3),
                "temporal": round(entropy.temporal, 3),
                "dependency": round(entropy.dependency, 3),
                "divergence": round(entropy.divergence, 3),
                "information": round(entropy.information, 3),
                "propagation": round(entropy.propagation, 3),
                "evidence": round(entropy.evidence, 3),
                "impact": round(entropy.impact, 3),
                "total": round(entropy.total, 3),
            },
        })


# ═══════════════════════════════════════════════════════════════
# API Routes
# ═══════════════════════════════════════════════════════════════

@app.get("/")
async def index():
    """Serve the main HTML page"""
    return FileResponse(str(STATIC_DIR / "index.html"))


@app.post("/api/run")
async def run_session(req: RunRequest):
    """启动一个新的 Arena 会话"""
    session_id = str(uuid.uuid4())[:8]

    # 创建模型和 Arena 实例
    models = create_models()

    verifier_id = ""
    if req.verifier == "enabled" or req.verifier == "auto":
        verifier_id = ""  # DeepSeek 不在 models 里，查证通过工具注册表

    arena = Arena(
        models=models,
        max_rounds=req.max_rounds,
        convergence_threshold=req.convergence_threshold,
        min_rounds=5,
        min_attacks=3,
        verifier_id=verifier_id,
    )

    session = Session(
        session_id=session_id,
        question=req.question,
        created_at=datetime.now().isoformat(),
    )
    sessions[session_id] = session

    # 启动后台任务
    emitter = ArenaEventEmitter(session, arena, req.question)
    task = asyncio.create_task(emitter.run())
    session.task = task

    return {"session_id": session_id, "status": "started"}


@app.get("/api/stream/{session_id}")
async def stream_session(session_id: str):
    """SSE 端点 — 实时推送 Arena 事件"""
    session = sessions.get(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")

    async def event_generator():
        import json
        while True:
            try:
                # 等待事件，超时 30 秒发送心跳
                event = await asyncio.wait_for(session.queue.get(), timeout=30.0)
                yield {
                    "event": event["event"],
                    "data": json.dumps(event["data"], ensure_ascii=False),
                }
                # 如果是终止事件，结束流
                if event["event"] in ("session_finished", "session_error"):
                    break
            except asyncio.TimeoutError:
                # 心跳保活
                yield {"event": "heartbeat", "data": "{}"}
            except Exception:
                break

    return EventSourceResponse(event_generator())


@app.post("/api/stop/{session_id}")
async def stop_session(session_id: str):
    """停止一个运行中的会话"""
    session = sessions.get(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")

    if session.status != "running":
        return {"session_id": session_id, "status": session.status, "message": "Session not running"}

    session.status = "stopped"
    if session.task and not session.task.done():
        session.task.cancel()

    return {"session_id": session_id, "status": "stopped"}


@app.get("/api/sessions")
async def list_sessions():
    """列出所有会话"""
    result = []
    for sid, session in sessions.items():
        result.append(SessionInfo(
            session_id=sid,
            question=session.question[:100],
            status=session.status,
            created_at=session.created_at,
            finished_at=session.finished_at,
        ))
    return result


# ═══════════════════════════════════════════════════════════════
# Entry Point
# ═══════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "server:app",
        host="0.0.0.0",
        port=8000,
        reload=False,
    )
