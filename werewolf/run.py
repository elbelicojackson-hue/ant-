"""
run.py — 启动 AI 狼人杀
"""

import asyncio
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.adapter import MiMoAdapter, QwenAdapter, DoubaoAdapter, load_env
from werewolf.game import WerewolfGame

load_env()


async def main():
    print("=" * 60)
    print("NeuralComm — AI 狼人杀")
    print("=" * 60)
    print()

    # 6个模型玩家
    players = {
        "MiMo": MiMoAdapter().call_with_retry,
        "Kimi": QwenAdapter(model="kimi-k2.6").call_with_retry,
        "Doubao": DoubaoAdapter().call_with_retry,
        "Qwen": QwenAdapter().call_with_retry,
        "Qwen3.6": QwenAdapter(model="qwen3.6-plus").call_with_retry,
        "GPT": __import__('core.adapter', fromlist=['OpenAIAdapter']).OpenAIAdapter().call_with_retry,
    }

    game = WerewolfGame(players)
    result = await game.play()

    print()
    print("=" * 60)
    print("[游戏结果]")
    print(f"  胜利方: {result.winner}")
    print(f"  轮数: {result.rounds}")
    print(f"  耗时: {result.elapsed_seconds:.1f}s")
    print()
    print("[玩家状态]")
    for p in result.players:
        status = "存活" if p.alive else "死亡"
        print(f"  {p.id}: {p.role.value} — {status}")


if __name__ == "__main__":
    asyncio.run(main())
