"""
run_arena.py — 自由对话场模式

无预设流程。三个模型自由交互、互相攻击/支持/补充。
算法只负责路由、记录和判断收敛。

模型不知道自己是谁，不知道该做什么。
它们只看到: 问题 + 其他人说了什么 + 自己之前说了什么。
涌现出来的协作结构是自发的。
"""

import asyncio
import sys
import json
import os
from datetime import datetime

from core.arena import Arena
from core.adapter import OpenAIAdapter, DeepSeekAdapter, QwenAdapter, DoubaoAdapter, MiMoAdapter, KimiK2Adapter


async def main():
    if len(sys.argv) > 1:
        question = " ".join(sys.argv[1:])
    else:
        question = input("输入问题 > ")

    print()
    print("=" * 60)
    print("NeuralComm — 自由对话场")
    print("=" * 60)
    print(f"\n[问题] {question}")
    print()
    print("[模型]")
    print("  A: GPT-5.4 (yunwu.ai)")
    print("  B: MiMo V2.5 Pro (xiaomi)")
    print("  C: DeepSeek V4 Pro (bocha.cn) — 具备搜索查证能力")
    print("  D: Qwen-Max (阿里云百炼)")
    print("  E: Qwen3.6-Plus (阿里云百炼)")
    print("  F: Kimi-K2.6 (阿里云百炼)")
    print("  G: 豆包 (火山引擎)")
    print("  无身份定义，无预设流程，无轮数上限")
    print("  收敛条件: 熵 < 0.25 或 连续两轮停滞")
    print()

    # 3个模型辩论 + DeepSeek 专职查证
    models = {
        "MiMo-V2.5": MiMoAdapter().call_with_retry,
        "Kimi-K2.6": QwenAdapter(model="kimi-k2.6").call_with_retry,
        "Doubao": DoubaoAdapter().call_with_retry,
    }

    arena = Arena(
        models=models,
        max_rounds=50,              # 不设实际上限
        convergence_threshold=0.25, # 靠熵收敛停止
        min_rounds=5,               # 至少跑5轮（给攻击留时间）
        min_attacks=3,              # 至少产生3次有效攻击才允许收敛
        verifier_id="",             # DeepSeek 不在 models 里，查证通过工具注册表
    )

    result = await arena.debate(question)

    # 输出
    print()
    print("=" * 60)
    print("[对话场结果]")
    print("=" * 60)
    print(f"  总轮数: {result.total_rounds}")
    print(f"  总发言: {len(result.utterances)}")
    print(f"  关系数: {len(result.relations)}")
    print(f"  收敛原因: {result.convergence_reason}")
    print(f"  耗时: {result.elapsed_seconds:.1f} 秒")

    # 熵变化
    print()
    print("[熵变化轨迹]")
    for eh in result.entropy_history:
        r = eh["round"]
        t = eh["total"]
        d = eh["divergence"]
        bars = "█" * int(t * 20) + "░" * (20 - int(t * 20))
        print(f"  第{r}轮: {bars} total={t:.2f} divergence={d:.2f}")

    # 9维熵
    print()
    print("[最终熵]")
    e = result.final_entropy
    bars_fn = lambda v: "█" * int(v * 10) + "░" * (10 - int(v * 10))
    print(f"  语义熵   {bars_fn(e.semantic)} {e.semantic:.2f}")
    print(f"  因果熵   {bars_fn(e.causal)} {e.causal:.2f}")
    print(f"  边界熵   {bars_fn(e.boundary)} {e.boundary:.2f}")
    print(f"  时序熵   {bars_fn(e.temporal)} {e.temporal:.2f}")
    print(f"  依赖熵   {bars_fn(e.dependency)} {e.dependency:.2f}")
    print(f"  分歧熵   {bars_fn(e.divergence)} {e.divergence:.2f}")
    print(f"  信息熵   {bars_fn(e.information)} {e.information:.2f}")
    print(f"  传播熵   {bars_fn(e.propagation)} {e.propagation:.2f}")
    print(f"  证据熵   {bars_fn(e.evidence)} {e.evidence:.2f}")
    print(f"  总熵     {bars_fn(e.total)} {e.total:.2f}  {'收敛 ✓' if e.can_converge else '未收敛 ✗'}")

    # 共识与分歧
    if result.consensus:
        print()
        print("[涌现的共识]")
        for c in result.consensus:
            print(f"  ✓ {c}")

    if result.disputes:
        print()
        print("[仍有分歧]")
        for d in result.disputes:
            print(f"  ? {d}")

    # 关系图
    if result.relations:
        print()
        print("[交互关系]")
        for r in result.relations[:10]:
            src = next((u for u in result.utterances if u.id == r.source_id), None)
            tgt = next((u for u in result.utterances if u.id == r.target_id), None)
            if src and tgt:
                icon = "⚔" if r.relation_type == "opposes" else "🤝"
                print(f"  {icon} {src.source}(R{src.round}) {r.relation_type} {tgt.source}(R{tgt.round})")

    # 模型状态
    print()
    print("[模型状态]")
    summary = arena.state.get_summary()
    for mid, info in summary["nodes"].items():
        think_sec = info["total_thinking_ms"] / 1000
        print(f"  {mid}: {info['call_count']}次发言, 思考{think_sec:.1f}s")

    # === 保存日志和数据 ===
    save_logs(question, result, arena.state.events)


def save_logs(question, result, events):
    """保存完整日志: JSON (机器可读) + Markdown (人类可读)"""
    logs_dir = os.path.join(os.path.dirname(__file__), "logs")
    os.makedirs(logs_dir, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base_name = f"arena_{timestamp}"

    # JSON 日志: 完整结构化数据
    json_path = os.path.join(logs_dir, f"{base_name}.json")
    json_data = {
        "timestamp": datetime.now().isoformat(),
        "question": question,
        "summary": {
            "total_rounds": result.total_rounds,
            "total_utterances": len(result.utterances),
            "total_relations": len(result.relations),
            "convergence_reason": result.convergence_reason,
            "elapsed_seconds": result.elapsed_seconds,
        },
        "entropy_history": result.entropy_history,
        "final_entropy": result.final_entropy.to_dict(),
        "utterances": [
            {
                "id": u.id,
                "round": u.round,
                "source": u.source,
                "timestamp": u.timestamp,
                "elapsed_ms": u.elapsed_ms,
                "content": u.content,
            }
            for u in result.utterances
        ],
        "relations": [
            {
                "source_id": r.source_id,
                "target_id": r.target_id,
                "type": r.relation_type,
                "strength": r.strength,
            }
            for r in result.relations
        ],
        "consensus": result.consensus,
        "disputes": result.disputes,
        "events": events,
    }

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(json_data, f, ensure_ascii=False, indent=2)

    # Markdown 日志: 人类可读
    md_path = os.path.join(logs_dir, f"{base_name}.md")
    with open(md_path, "w", encoding="utf-8") as f:
        f.write(f"# Arena 推敲日志\n\n")
        f.write(f"**时间**: {datetime.now().isoformat()}\n\n")
        f.write(f"## 问题\n\n{question}\n\n")
        f.write(f"## 结果摘要\n\n")
        f.write(f"- 总轮数: {result.total_rounds}\n")
        f.write(f"- 总发言: {len(result.utterances)}\n")
        f.write(f"- 关系数: {len(result.relations)}\n")
        f.write(f"- 收敛原因: {result.convergence_reason}\n")
        f.write(f"- 耗时: {result.elapsed_seconds:.1f} 秒\n\n")

        f.write(f"## 最终熵\n\n")
        e = result.final_entropy
        f.write(f"| 维度 | 值 | 说明 |\n|---|---|---|\n")
        f.write(f"| 语义熵 | {e.semantic:.2f} | 模型间理解一致性 |\n")
        f.write(f"| 因果熵 | {e.causal:.2f} | 推理路径唯一性 |\n")
        f.write(f"| 边界熵 | {e.boundary:.2f} | 适用范围清晰度 |\n")
        f.write(f"| 时序熵 | {e.temporal:.2f} | 时间敏感度 |\n")
        f.write(f"| 依赖熵 | {e.dependency:.2f} | 未验证假设占比 |\n")
        f.write(f"| 分歧熵 | {e.divergence:.2f} | 模型间分歧度 |\n")
        f.write(f"| 信息熵 | {e.information:.2f} | 关键信息缺失度 |\n")
        f.write(f"| 传播熵 | {e.propagation:.2f} | 错误扩散范围 |\n")
        f.write(f"| 证据熵 | {e.evidence:.2f} | 证据可靠性 |\n")
        f.write(f"| 影响熵 | {e.impact:.2f} | 自证难度 |\n")
        f.write(f"| **总熵** | **{e.total:.2f}** | {'可收敛 ✓' if e.can_converge else '需继续 ✗'} |\n\n")

        f.write(f"## 熵变化轨迹\n\n")
        for eh in result.entropy_history:
            f.write(f"- 第{eh['round']}轮: total={eh['total']:.2f} divergence={eh['divergence']:.2f}\n")
        f.write("\n")

        f.write(f"## 完整对话\n\n")
        for u in result.utterances:
            f.write(f"### 第 {u.round} 轮 — {u.source}\n\n")
            f.write(f"*时间: {u.timestamp}, 用时: {u.elapsed_ms/1000:.1f}s*\n\n")
            f.write(f"{u.content}\n\n")
            f.write(f"---\n\n")

        if result.relations:
            f.write(f"## 交互关系\n\n")
            for r in result.relations:
                src = next((u for u in result.utterances if u.id == r.source_id), None)
                tgt = next((u for u in result.utterances if u.id == r.target_id), None)
                if src and tgt:
                    rt = r.relation_type
                    icon = "⚔" if rt == "opposes" else "🤝"
                    f.write(f"- {icon} {src.source}(R{src.round}) **{rt}** {tgt.source}(R{tgt.round})\n")
            f.write("\n")

        if result.consensus:
            f.write(f"## 涌现的共识\n\n")
            for c in result.consensus:
                f.write(f"- {c}\n")
            f.write("\n")

        if result.disputes:
            f.write(f"## 仍有分歧\n\n")
            for d in result.disputes:
                f.write(f"- {d}\n")
            f.write("\n")

    print()
    print(f"[日志已保存]")
    print(f"  JSON: {json_path}")
    print(f"  Markdown: {md_path}")


if __name__ == "__main__":
    asyncio.run(main())
