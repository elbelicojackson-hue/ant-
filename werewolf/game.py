"""
werewolf/game.py — AI 狼人杀

基于 NCP 协议的多模型社会推理博弈。
每个模型扮演一个玩家，拥有隐藏身份。
通过发言、指控、投票来找出狼人。

核心映射:
  NCP 攻击 → 狼人杀指控
  NCP 投票 → 放逐投票
  NCP 推理链 → 玩家辩护逻辑
  NCP 信息路由 → 身份信息不对称 (狼人互知，村民不知)

规则:
  - 6-8 个模型玩家
  - 2 狼人 + 1 预言家 + 1 女巫 + 村民
  - 白天: 发言 → 指控 → 投票放逐
  - 夜晚: 狼人杀人 / 预言家查验 / 女巫救人或毒人
  - 胜利: 狼人全死 (好人胜) 或 狼人数 ≥ 好人数 (狼人胜)
"""

import asyncio
import random
import time
import hashlib
from dataclasses import dataclass, field
from typing import Callable
from datetime import datetime, timezone
from enum import Enum

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from core.adapter import load_env
load_env()


# ═══════════════════════════════════════════════════════════════
# 数据结构
# ═══════════════════════════════════════════════════════════════

class Role(Enum):
    WEREWOLF = "狼人"
    VILLAGER = "村民"
    SEER = "预言家"
    WITCH = "女巫"
    HUNTER = "猎人"


class Phase(Enum):
    NIGHT = "夜晚"
    DAY_SPEECH = "白天发言"
    DAY_VOTE = "白天投票"
    GAME_OVER = "游戏结束"


@dataclass
class Player:
    """一个玩家"""
    id: str                         # 模型名
    role: Role                      # 身份
    alive: bool = True
    call_fn: Callable = None        # 模型调用函数
    # 记录
    speeches: list[str] = field(default_factory=list)
    votes: list[str] = field(default_factory=list)
    accusations: list[str] = field(default_factory=list)
    # 女巫专用
    has_antidote: bool = True       # 解药 (只能用1次)
    has_poison: bool = True         # 毒药 (只能用1次)


@dataclass
class GameEvent:
    """游戏事件"""
    round: int
    phase: str
    actor: str
    action: str
    target: str = ""
    content: str = ""
    timestamp: str = ""


@dataclass
class GameResult:
    """游戏结果"""
    winner: str                     # "好人" / "狼人"
    rounds: int
    events: list[GameEvent]
    players: list[Player]
    elapsed_seconds: float


# ═══════════════════════════════════════════════════════════════
# 狼人杀引擎
# ═══════════════════════════════════════════════════════════════

class WerewolfGame:
    """
    AI 狼人杀引擎

    信息路由原则 (和 NCP 一致):
    - 算法不告诉模型"做什么"，只控制模型"看到什么"
    - 狼人看到: 谁是同伴狼人
    - 预言家看到: 查验结果
    - 村民看到: 只有公开发言和投票结果
    - 所有人看到: 谁被放逐了、谁死了
    """

    def __init__(self, players: dict[str, Callable]):
        """
        players: {model_name: call_fn}
        至少需要 5 个模型
        """
        self.players: dict[str, Player] = {}
        self.events: list[GameEvent] = []
        self.round: int = 0
        self.phase: Phase = Phase.NIGHT
        self.public_log: list[str] = []  # 所有人可见的公开信息

        # 分配身份
        names = list(players.keys())
        random.shuffle(names)

        roles = self._assign_roles(len(names))

        for i, name in enumerate(names):
            self.players[name] = Player(
                id=name,
                role=roles[i],
                call_fn=players[name],
            )

        # 打印身份 (上帝视角)
        print("  [身份分配]")
        for name, player in self.players.items():
            print(f"    {name}: {player.role.value}")

    def _assign_roles(self, n: int) -> list[Role]:
        """根据人数分配身份"""
        if n <= 4:
            roles = [Role.WEREWOLF, Role.SEER, Role.VILLAGER, Role.VILLAGER]
        elif n <= 6:
            roles = [Role.WEREWOLF, Role.WEREWOLF, Role.SEER, Role.WITCH,
                     Role.VILLAGER, Role.VILLAGER]
        else:
            roles = [Role.WEREWOLF, Role.WEREWOLF, Role.SEER, Role.WITCH,
                     Role.HUNTER, Role.VILLAGER, Role.VILLAGER, Role.VILLAGER]

        random.shuffle(roles)
        return roles[:n]

    async def play(self) -> GameResult:
        """运行完整游戏"""
        start_time = time.time()
        max_rounds = 10

        for self.round in range(1, max_rounds + 1):
            print(f"\n  ══ 第 {self.round} 轮 ══")

            # 夜晚
            await self._night_phase()
            if self._check_game_over():
                break

            # 白天发言
            await self._day_speech_phase()

            # 白天投票
            await self._day_vote_phase()
            if self._check_game_over():
                break

        # 结果
        winner = self._get_winner()
        elapsed = time.time() - start_time

        print(f"\n  ══ 游戏结束 ══")
        print(f"  胜利方: {winner}")
        print(f"  轮数: {self.round}")
        print(f"  耗时: {elapsed:.1f}s")

        return GameResult(
            winner=winner,
            rounds=self.round,
            events=self.events,
            players=list(self.players.values()),
            elapsed_seconds=elapsed,
        )

    # === 夜晚阶段 ===

    async def _night_phase(self):
        """夜晚: 狼人杀人 / 预言家查验 / 女巫行动"""
        print(f"  ── 夜晚 ──")
        self.phase = Phase.NIGHT

        # 狼人行动
        wolves = [p for p in self.players.values() if p.role == Role.WEREWOLF and p.alive]
        if wolves:
            target = await self._wolf_kill(wolves)
            if target:
                self.events.append(GameEvent(
                    round=self.round, phase="night", actor="狼人",
                    action="杀", target=target,
                    timestamp=datetime.now(timezone.utc).isoformat(),
                ))
                print(f"    🐺 狼人选择杀: {target}")

                # 女巫救人
                witch = next((p for p in self.players.values()
                             if p.role == Role.WITCH and p.alive), None)
                saved = False
                if witch:
                    saved = await self._witch_save(witch, target)
                    if saved:
                        print(f"    🧪 女巫救了 {target}")

                if not saved:
                    self.players[target].alive = False
                    self.public_log.append(f"第{self.round}夜，{target} 被杀害。")
                    print(f"    💀 {target} 死亡")
                else:
                    self.public_log.append(f"第{self.round}夜，平安夜。")

        # 女巫毒人 (独立于救人)
        witch = next((p for p in self.players.values()
                     if p.role == Role.WITCH and p.alive and p.has_poison), None)
        if witch:
            poison_target = await self._witch_poison(witch)
            if poison_target:
                self.players[poison_target].alive = False
                self.public_log.append(f"第{self.round}夜，{poison_target} 死亡（毒杀）。")
                print(f"    ☠️ 女巫毒杀 {poison_target}")

        # 预言家查验
        seer = next((p for p in self.players.values()
                    if p.role == Role.SEER and p.alive), None)
        if seer:
            await self._seer_check(seer)

    async def _wolf_kill(self, wolves: list[Player]) -> str:
        """狼人选择杀谁"""
        alive_others = [p.id for p in self.players.values()
                       if p.alive and p.role != Role.WEREWOLF]
        if not alive_others:
            return ""

        # 狼人看到: 谁是同伴 + 谁还活着
        wolf_names = [w.id for w in wolves]
        wolf = wolves[0]  # 第一个狼人决策

        prompt = f"""你是狼人杀游戏中的狼人。

你的同伴狼人: {', '.join(wolf_names)}
当前存活的非狼人玩家: {', '.join(alive_others)}

公开信息:
{chr(10).join(self.public_log[-5:]) if self.public_log else '(游戏刚开始)'}

选择今晚要杀的目标。只输出一个玩家名字:"""

        try:
            content = await wolf.call_fn(prompt, {})
            # 从回复中提取目标名
            for name in alive_others:
                if name in content:
                    return name
            return random.choice(alive_others)
        except Exception:
            return random.choice(alive_others)

    async def _witch_save(self, witch: Player, target: str) -> bool:
        """女巫决定是否救人 (解药只有1瓶)"""
        if not witch.has_antidote:
            return False

        prompt = f"""你是狼人杀游戏中的女巫。
今晚 {target} 被狼人杀害了。你有一瓶解药（用完就没了）。
要救 {target} 吗？只回答 YES 或 NO:"""

        try:
            content = await witch.call_fn(prompt, {})
            if "YES" in content.upper():
                witch.has_antidote = False  # 用掉解药
                return True
            return False
        except Exception:
            return False

    async def _witch_poison(self, witch: Player) -> str:
        """女巫决定是否毒人 (毒药只有1瓶)"""
        alive_others = [p.id for p in self.players.values()
                       if p.alive and p.id != witch.id]
        if not alive_others:
            return ""

        prompt = f"""你是狼人杀游戏中的女巫，你有一瓶毒药（用完就没了）。
存活玩家: {', '.join(alive_others)}

公开信息:
{chr(10).join(self.public_log[-5:]) if self.public_log else '(无)'}

要毒杀某个玩家吗？如果要，输出玩家名字。如果不用，输出 NO:"""

        try:
            content = await witch.call_fn(prompt, {})
            if "NO" in content.upper():
                return ""
            for name in alive_others:
                if name in content:
                    witch.has_poison = False
                    return name
            return ""
        except Exception:
            return ""

    async def _seer_check(self, seer: Player):
        """预言家查验"""
        alive_others = [p.id for p in self.players.values()
                       if p.alive and p.id != seer.id]
        if not alive_others:
            return

        prompt = f"""你是狼人杀游戏中的预言家。
存活玩家: {', '.join(alive_others)}
选择今晚要查验的玩家。只输出一个名字:"""

        try:
            content = await seer.call_fn(prompt, {})
            target = None
            for name in alive_others:
                if name in content:
                    target = name
                    break
            if not target:
                target = random.choice(alive_others)

            # 告知结果
            is_wolf = self.players[target].role == Role.WEREWOLF
            result = "狼人" if is_wolf else "好人"
            print(f"    🔮 预言家查验 {target}: {result}")

            # 预言家记住这个信息 (下次发言可以用)
            seer.speeches.append(f"[私密] 查验{target}={result}")
        except Exception:
            pass

    # === 白天发言 ===

    async def _day_speech_phase(self):
        """白天发言: 每个存活玩家发言"""
        print(f"  ── 白天发言 ──")
        self.phase = Phase.DAY_SPEECH

        alive_players = [p for p in self.players.values() if p.alive]

        for player in alive_players:
            speech = await self._get_speech(player)
            player.speeches.append(speech)
            self.public_log.append(f"[R{self.round}] {player.id}: {speech[:100]}")
            self.events.append(GameEvent(
                round=self.round, phase="day_speech", actor=player.id,
                action="发言", content=speech[:200],
                timestamp=datetime.now(timezone.utc).isoformat(),
            ))
            print(f"    💬 {player.id}: {speech[:80]}")

    async def _get_speech(self, player: Player) -> str:
        """获取玩家发言"""
        # 信息路由: 每个玩家只看到公开信息 + 自己的私密信息
        private_info = ""
        if player.role == Role.WEREWOLF:
            wolves = [p.id for p in self.players.values()
                     if p.role == Role.WEREWOLF and p.alive and p.id != player.id]
            private_info = f"\n[你是狼人。同伴: {', '.join(wolves) if wolves else '无'}。不要暴露身份。]"
        elif player.role == Role.SEER:
            checks = [s for s in player.speeches if s.startswith("[私密]")]
            private_info = f"\n[你是预言家。查验记录: {'; '.join(checks) if checks else '无'}]"
        elif player.role == Role.WITCH:
            private_info = "\n[你是女巫。]"

        prompt = f"""狼人杀游戏，第{self.round}轮白天。你是 {player.id}。
{private_info}

公开信息:
{chr(10).join(self.public_log[-8:]) if self.public_log else '(无)'}

存活玩家: {', '.join(p.id for p in self.players.values() if p.alive)}

请发言（分析局势、指控可疑玩家或为自己辩护）。简短，不超过3句话:"""

        try:
            content = await player.call_fn(prompt, {})
            return content.strip()[:200]
        except Exception:
            return "我没什么要说的。"

    # === 白天投票 ===

    async def _day_vote_phase(self):
        """白天投票: 放逐一个玩家"""
        print(f"  ── 白天投票 ──")
        self.phase = Phase.DAY_VOTE

        alive_players = [p for p in self.players.values() if p.alive]
        votes = {}

        for player in alive_players:
            target = await self._get_vote(player, alive_players)
            votes[player.id] = target
            player.votes.append(target)
            print(f"    🗳 {player.id} → {target}")

        # 计票
        vote_counts = {}
        for target in votes.values():
            vote_counts[target] = vote_counts.get(target, 0) + 1

        # 最高票放逐
        if vote_counts:
            max_votes = max(vote_counts.values())
            candidates = [k for k, v in vote_counts.items() if v == max_votes]
            exiled = random.choice(candidates)

            self.players[exiled].alive = False
            role_reveal = self.players[exiled].role.value
            self.public_log.append(
                f"第{self.round}天，{exiled} 被放逐 ({max_votes}票)。身份: {role_reveal}。"
            )
            self.events.append(GameEvent(
                round=self.round, phase="day_vote", actor="全体",
                action="放逐", target=exiled, content=f"{max_votes}票, 身份={role_reveal}",
                timestamp=datetime.now(timezone.utc).isoformat(),
            ))
            print(f"    ⚰️ {exiled} 被放逐 ({max_votes}票) — 身份: {role_reveal}")

    async def _get_vote(self, player: Player, alive_players: list[Player]) -> str:
        """获取玩家投票"""
        others = [p.id for p in alive_players if p.id != player.id]

        private_info = ""
        if player.role == Role.WEREWOLF:
            wolves = [p.id for p in self.players.values()
                     if p.role == Role.WEREWOLF and p.alive and p.id != player.id]
            private_info = f"[你是狼人，不要投同伴: {', '.join(wolves)}]"
        elif player.role == Role.SEER:
            checks = [s for s in player.speeches if s.startswith("[私密]")]
            if checks:
                private_info = f"[你是预言家，查验记录: {'; '.join(checks)}]"

        # 包含本轮所有发言
        round_speeches = [log for log in self.public_log if f"[R{self.round}]" in log]
        speeches_text = chr(10).join(round_speeches) if round_speeches else "(无发言记录)"

        prompt = f"""狼人杀投票。你是 {player.id}。{private_info}

本轮所有人的发言:
{speeches_text}

历史信息:
{chr(10).join(log for log in self.public_log if '[R' not in log)[-5:] if self.public_log else '(无)'}

可投票对象: {', '.join(others)}

根据发言内容分析谁最可疑，选择要放逐的玩家。只输出一个名字:"""

        try:
            content = await player.call_fn(prompt, {})
            for name in others:
                if name in content:
                    return name
            return random.choice(others)
        except Exception:
            return random.choice(others)

    # === 游戏状态 ===

    def _check_game_over(self) -> bool:
        """检查游戏是否结束"""
        wolves_alive = sum(1 for p in self.players.values()
                          if p.role == Role.WEREWOLF and p.alive)
        good_alive = sum(1 for p in self.players.values()
                        if p.role != Role.WEREWOLF and p.alive)

        if wolves_alive == 0:
            return True  # 好人胜
        if wolves_alive >= good_alive:
            return True  # 狼人胜
        return False

    def _get_winner(self) -> str:
        """获取胜利方"""
        wolves_alive = sum(1 for p in self.players.values()
                          if p.role == Role.WEREWOLF and p.alive)
        return "狼人" if wolves_alive > 0 else "好人"
