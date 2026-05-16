"""
run_deep.py — 深度推敲模式运行

GPT-5.4 负责思考和修正
Gemini 3.1 Pro 负责质疑（独立视角找漏洞）

演示: 对一个问题进行递归质疑，直到每个断言都经受住审视。
"""

import asyncio
import sys

from core.depth import DepthEngine
from core.adapter import OpenAIAdapter, AnthropicAdapter, BochaAdapter


async def main():
    if len(sys.argv) > 1:
        question = " ".join(sys.argv[1:])
    else:
        question = input("输入问题 > ")

    print()
    print("=" * 60)
    print("NeuralComm — 深度推敲模式")
    print("=" * 60)
    print(f"\n[问题] {question}")
    print()
    print("[配置]")
    print("  思考者: GPT-5.4 (yunwu.ai)")
    print("  质疑者: Claude Opus 4.6 (yunwu.ai)")
    print("  知识源: DeepSeek V4 Pro (bocha.cn) — 查证事实，不做决策")
    print("  最大深度: 3 层 (事实→逻辑→完备性)")
    print("  并行质疑: 开启")
    print()

    # 思考者: GPT-5.4
    thinker = OpenAIAdapter()
    # 质疑者: Claude Opus 4.6 (最强逻辑审查能力)
    challenger = AnthropicAdapter()
    # 知识源: DeepSeek V4 Pro (查证事实、搜索数据，不做决策)
    oracle = BochaAdapter()

    engine = DepthEngine(
        think_fn=thinker.call_with_retry,
        challenge_fn=challenger.call_with_retry,
        oracle_fn=oracle.call_with_retry,
        max_depth=3,
        max_claims_per_level=3,
        parallel=True,
    )

    result = await engine.deep_reason(question)

    # 输出结果
    print()
    print("-" * 60)
    print(f"[推敲统计]")
    print(f"  最大深度: {result.max_depth_reached} 层")
    print(f"  总质疑次数: {result.total_challenges}")
    print(f"  修正次数: {result.revisions}")
    print(f"  发现盲点: {len(result.uncertainties)} 个")
    print(f"  总耗时: {result.elapsed_seconds:.1f} 秒")
    print()

    if result.uncertainties:
        print("[发现的盲点/不确定性]")
        for u in result.uncertainties:
            print(f"  {u}")
        print()

    print("[断言审查结果]")
    for claim in result.claims:
        status = "✓ 通过" if claim.survived else ("✗ 有漏洞" if claim.challenged else "- 未审查")
        print(f"  [{status}] (深度{claim.depth}) {claim.content[:80]}")
    print()

    print("=" * 60)
    print("[最终回答 — 经过深度推敲]")
    print("=" * 60)
    print()
    print(result.final_answer)


if __name__ == "__main__":
    asyncio.run(main())
