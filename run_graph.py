"""
run_graph.py — 论证图推敲模式

进阶版: 不是线性质疑断言，而是:
1. 构建因果论证图
2. 追溯到源头 (事实/假设)
3. 对每个假设找崩塌条件
4. 用知识源验证崩塌条件是否成立
5. 如果成立 → 传播影响 → 修正结论

三模型角色:
- GPT-5.4: 构建论证图、修正结论
- Claude Opus 4.6: 对抗性攻击 (找崩塌条件)
- DeepSeek V4 Pro: 验证崩塌条件是否在现实中成立
"""

import asyncio
import sys

from core.argument_graph import ArgumentGraphEngine, NodeStatus, NodeType
from core.adapter import OpenAIAdapter, AnthropicAdapter, BochaAdapter


async def main():
    if len(sys.argv) > 1:
        question = " ".join(sys.argv[1:])
    else:
        question = input("输入问题 > ")

    print()
    print("=" * 60)
    print("NeuralComm — 论证图推敲模式 (进阶)")
    print("=" * 60)
    print(f"\n[问题] {question}")
    print()
    print("[配置]")
    print("  构建者: GPT-5.4 — 构建论证图、分析影响、修正结论")
    print("  攻击者: Claude Opus 4.6 — 对抗性思考，找崩塌条件")
    print("  验证者: DeepSeek V4 Pro — 查证崩塌条件是否在现实中成立")
    print()

    # 三模型
    builder = OpenAIAdapter()
    attacker = AnthropicAdapter()
    verifier = BochaAdapter()

    engine = ArgumentGraphEngine(
        think_fn=builder.call_with_retry,
        attack_fn=attacker.call_with_retry,
        verify_fn=verifier.call_with_retry,
    )

    result = await engine.reason(question)

    # 输出
    print()
    print("=" * 60)
    print("[推敲结果]")
    print("=" * 60)
    print(f"  总节点: {result.total_nodes}")
    print(f"  攻击次数: {result.attack_count}")
    print(f"  崩塌节点: {len(result.collapsed_nodes)}")
    print(f"  加固节点: {len(result.fortified_nodes)}")
    print(f"  关键假设(未验证): {len(result.critical_assumptions)}")
    print(f"  耗时: {result.elapsed_seconds:.1f} 秒")

    print()
    print("[动态熵 — 10维不确定性]")
    e = result.entropy
    bars = lambda v: "█" * int(v * 10) + "░" * (10 - int(v * 10))
    print(f"  语义熵   {bars(e.semantic)} {e.semantic:.2f}  模型间理解一致性")
    print(f"  因果熵   {bars(e.causal)} {e.causal:.2f}  推理路径唯一性")
    print(f"  边界熵   {bars(e.boundary)} {e.boundary:.2f}  结论适用范围清晰度")
    print(f"  时序熵   {bars(e.temporal)} {e.temporal:.2f}  时间敏感度")
    print(f"  依赖熵   {bars(e.dependency)} {e.dependency:.2f}  未验证假设占比")
    print(f"  分歧熵   {bars(e.divergence)} {e.divergence:.2f}  模型间分歧度")
    print(f"  信息熵   {bars(e.information)} {e.information:.2f}  关键信息缺失度")
    print(f"  传播熵   {bars(e.propagation)} {e.propagation:.2f}  错误扩散范围")
    print(f"  证据熵   {bars(e.evidence)} {e.evidence:.2f}  已有证据可靠性")
    print(f"  影响熵   {bars(e.impact)} {e.impact:.2f}  被攻击后的自证难度")
    print(f"  ─────────────────────────────")
    print(f"  总熵     {bars(e.total)} {e.total:.2f}  {'可收敛 ✓' if e.can_converge else '需继续推敲 ✗'}")
    print(f"  风险等级: {e.risk_level}")
    top3 = e.top3_entropy
    print(f"  薄弱点: {top3[0][0]}({top3[0][1]:.2f}) > {top3[1][0]}({top3[1][1]:.2f}) > {top3[2][0]}({top3[2][1]:.2f})")
    if e.needs_external_evidence:
        print(f"  ⚠ 建议: 证据不足，需搜索外部数据源验证")
    if e.is_high_stakes:
        print(f"  ⚠ 高震动: 论证链存在单点故障，被攻击后自证成本高")

    print()
    print("[论证图状态]")
    for nid, node in result.nodes.items():
        status_icon = {
            NodeStatus.STANDING: "○",
            NodeStatus.FORTIFIED: "✓",
            NodeStatus.COLLAPSED: "✗",
            NodeStatus.CHALLENGED: "?",
        }[node.status]
        type_tag = node.node_type.value[:4]
        print(f"  {status_icon} [{type_tag}] {nid}: {node.content[:70]}")
        if node.collapse_conditions:
            for c in node.collapse_conditions[:2]:
                print(f"      崩塌条件: {c[:60]}")
        if node.status == NodeStatus.COLLAPSED:
            print(f"      *** 已崩塌: {node.attack_result[:60]}")

    if result.collapsed_nodes:
        print()
        print("[崩塌传播]")
        for cid in result.collapsed_nodes:
            node = result.nodes[cid]
            affected = [sid for sid in node.supports if sid in result.nodes]
            if affected:
                print(f"  {cid} 崩塌 → 影响: {', '.join(affected)}")

    print()
    print("=" * 60)
    print("[最终结论 — 经过论证图推敲]")
    print("=" * 60)
    print()
    print(result.revised_conclusion)

    # 状态机摘要
    print()
    print("[模型状态摘要]")
    summary = engine.state.get_summary()
    for mid, info in summary["nodes"].items():
        think_sec = info["total_thinking_ms"] / 1000
        print(f"  {mid} ({info['role']}): 调用{info['call_count']}次, "
              f"思考{think_sec:.1f}s, 重试{info['retry_count']}次")


if __name__ == "__main__":
    asyncio.run(main())
