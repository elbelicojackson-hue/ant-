"""
consensus_store.py — 共识持久化与跨 Session 可组合性

核心能力:
1. 持久化: 高 α 的 claim 存储为"已验证共识"，带完整验证元数据
2. 引用: 新 session 可加载已有共识作为预设前提，攻击门槛更高
3. 衰减: 旧共识随时间衰减，可被新证据重新挑战
4. 链接: 共识之间可以形成依赖图（A 依赖 B，B 崩塌则 A 降级）

存储格式: JSON 文件，每个共识一条记录
位置: .neuralcomm-state/consensus.json
"""

import json
import os
import time
import hashlib
from dataclasses import dataclass, field, asdict
from typing import Optional
from datetime import datetime, timezone


# ═══════════════════════════════════════════════════════════════
# 数据结构
# ═══════════════════════════════════════════════════════════════

@dataclass
class VerifiedClaim:
    """一条已验证的共识断言"""
    claim_id: str                       # 唯一标识 (内容 hash)
    content: str                        # 断言内容
    alpha: float                        # 共识度 (0-1)
    verified_at: str                    # ISO 时间戳
    session_id: str                     # 来源 session
    question_context: str               # 原始问题 (前100字)

    # 验证元数据
    participants: list[str] = field(default_factory=list)   # 参与验证的 agent
    agree_count: int = 0                # 同意的 agent 数
    total_count: int = 0                # 总 agent 数
    attacks_survived: int = 0           # 经受住的攻击次数
    rounds_tested: int = 0             # 经历了几轮测试

    # 状态
    status: str = "active"              # active / decayed / challenged / revoked
    confidence: float = 1.0             # 当前置信度 (会衰减)
    last_referenced: str = ""           # 上次被引用的时间
    reference_count: int = 0            # 被引用次数

    # 依赖
    depends_on: list[str] = field(default_factory=list)     # 依赖的其他 claim_id
    depended_by: list[str] = field(default_factory=list)    # 被哪些 claim 依赖

    # 挑战记录
    challenge_history: list[dict] = field(default_factory=list)  # 被重新挑战的记录


@dataclass
class ConsensusQuery:
    """查询已有共识的条件"""
    keywords: list[str] = field(default_factory=list)   # 关键词匹配
    min_alpha: float = 0.6                              # 最低 α 值
    min_confidence: float = 0.5                         # 最低置信度
    max_age_days: int = 90                              # 最大年龄 (天)
    limit: int = 10                                     # 最多返回几条


# ═══════════════════════════════════════════════════════════════
# 共识存储引擎
# ═══════════════════════════════════════════════════════════════

class ConsensusStore:
    """
    共识持久化存储

    设计原则:
    - 纯文件存储，无外部依赖
    - 每次写入都是原子的 (写临时文件 → rename)
    - 支持并发读，单写
    """

    # 衰减参数
    DECAY_RATE = 0.005          # 每天衰减 0.5%
    DECAY_FLOOR = 0.3           # 最低不低于 0.3 (除非被 revoke)
    REFERENCE_BOOST = 0.02      # 每次被引用恢复 2%
    CHALLENGE_PENALTY = 0.15    # 被挑战一次降低 15%

    def __init__(self, store_dir: Optional[str] = None):
        if store_dir is None:
            # 默认存储在项目根目录的 .neuralcomm-state/
            project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            store_dir = os.path.join(project_root, ".neuralcomm-state")

        self.store_dir = store_dir
        self.store_path = os.path.join(store_dir, "consensus.json")
        os.makedirs(store_dir, exist_ok=True)

        # 加载或初始化
        self._claims: dict[str, VerifiedClaim] = {}
        self._load()

    # === 核心操作 ===

    def store_claim(self, claim: VerifiedClaim) -> str:
        """存储一条新的已验证共识"""
        # 生成 ID (基于内容 hash，相同内容不重复存储)
        if not claim.claim_id:
            claim.claim_id = self._hash_content(claim.content)

        # 如果已存在相同内容的 claim，合并（取更高的 α）
        existing = self._claims.get(claim.claim_id)
        if existing:
            if claim.alpha > existing.alpha:
                existing.alpha = claim.alpha
                existing.agree_count = max(existing.agree_count, claim.agree_count)
                existing.attacks_survived += claim.attacks_survived
                existing.rounds_tested += claim.rounds_tested
                existing.confidence = min(1.0, existing.confidence + 0.1)
                # 合并参与者
                for p in claim.participants:
                    if p not in existing.participants:
                        existing.participants.append(p)
            return existing.claim_id

        self._claims[claim.claim_id] = claim
        self._save()
        return claim.claim_id

    def store_session_consensus(self, session_id: str, question: str,
                                 claims: list[dict], participants: list[str],
                                 total_rounds: int, total_attacks: int) -> list[str]:
        """
        从一次 session 的结果中批量存储共识

        claims 格式: [{"content": "...", "alpha": 0.7, "agree_count": 5}, ...]
        """
        stored_ids = []
        for claim_data in claims:
            alpha = claim_data.get("alpha", 0.0)
            if alpha < 0.6:  # 只存储 α ≥ 0.6 的
                continue

            claim = VerifiedClaim(
                claim_id=self._hash_content(claim_data["content"]),
                content=claim_data["content"],
                alpha=alpha,
                verified_at=datetime.now(timezone.utc).isoformat(),
                session_id=session_id,
                question_context=question[:100],
                participants=participants,
                agree_count=claim_data.get("agree_count", 0),
                total_count=len(participants),
                attacks_survived=total_attacks,
                rounds_tested=total_rounds,
            )
            cid = self.store_claim(claim)
            stored_ids.append(cid)

        if stored_ids:
            self._save()
        return stored_ids

    def query(self, q: ConsensusQuery) -> list[VerifiedClaim]:
        """查询匹配条件的已有共识"""
        self._apply_decay()  # 每次查询时应用衰减

        results = []
        now = time.time()

        for claim in self._claims.values():
            # 状态过滤
            if claim.status == "revoked":
                continue

            # α 过滤
            if claim.alpha < q.min_alpha:
                continue

            # 置信度过滤
            if claim.confidence < q.min_confidence:
                continue

            # 年龄过滤
            try:
                created = datetime.fromisoformat(claim.verified_at).timestamp()
                age_days = (now - created) / 86400
                if age_days > q.max_age_days:
                    continue
            except (ValueError, TypeError):
                pass

            # 关键词匹配
            if q.keywords:
                content_lower = claim.content.lower()
                if not any(kw.lower() in content_lower for kw in q.keywords):
                    continue

            results.append(claim)

        # 按 α * confidence 排序
        results.sort(key=lambda c: c.alpha * c.confidence, reverse=True)
        return results[:q.limit]

    def query_relevant(self, question: str, min_alpha: float = 0.6,
                        limit: int = 5) -> list[VerifiedClaim]:
        """
        根据问题内容查找相关的已有共识

        简单实现: 从问题中提取关键词，匹配已有 claim
        """
        # 提取问题中的关键词 (简单分词: 取长度 > 2 的连续中文/英文)
        import re
        words = re.findall(r'[\u4e00-\u9fff]{2,6}|[a-zA-Z]{3,}', question)
        # 去掉太常见的词
        stop_words = {"问题", "分析", "认为", "可能", "因为", "所以", "如果",
                      "什么", "怎么", "为什么", "是否", "以下", "关于"}
        keywords = [w for w in words if w not in stop_words][:10]

        if not keywords:
            return []

        return self.query(ConsensusQuery(
            keywords=keywords,
            min_alpha=min_alpha,
            limit=limit,
        ))

    def reference_claim(self, claim_id: str) -> Optional[VerifiedClaim]:
        """引用一条共识 (增加引用计数，恢复置信度)"""
        claim = self._claims.get(claim_id)
        if not claim or claim.status == "revoked":
            return None

        claim.reference_count += 1
        claim.last_referenced = datetime.now(timezone.utc).isoformat()
        claim.confidence = min(1.0, claim.confidence + self.REFERENCE_BOOST)
        self._save()
        return claim

    def challenge_claim(self, claim_id: str, challenger: str,
                         reason: str, success: bool) -> Optional[VerifiedClaim]:
        """
        挑战一条已有共识

        如果挑战成功: 降低置信度，可能 revoke
        如果挑战失败: 增加 attacks_survived，加固
        """
        claim = self._claims.get(claim_id)
        if not claim:
            return None

        challenge_record = {
            "challenger": challenger,
            "reason": reason[:200],
            "success": success,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
        claim.challenge_history.append(challenge_record)

        if success:
            claim.confidence -= self.CHALLENGE_PENALTY
            if claim.confidence <= 0.1:
                claim.status = "revoked"
                # 传播: 依赖这条 claim 的其他 claim 也要降级
                self._propagate_revocation(claim_id)
        else:
            claim.attacks_survived += 1
            claim.confidence = min(1.0, claim.confidence + 0.05)

        self._save()
        return claim

    def add_dependency(self, claim_id: str, depends_on_id: str):
        """声明 claim A 依赖 claim B"""
        claim = self._claims.get(claim_id)
        dep = self._claims.get(depends_on_id)
        if claim and dep:
            if depends_on_id not in claim.depends_on:
                claim.depends_on.append(depends_on_id)
            if claim_id not in dep.depended_by:
                dep.depended_by.append(claim_id)
            self._save()

    def get_stats(self) -> dict:
        """获取存储统计"""
        active = [c for c in self._claims.values() if c.status == "active"]
        decayed = [c for c in self._claims.values() if c.status == "decayed"]
        revoked = [c for c in self._claims.values() if c.status == "revoked"]
        return {
            "total": len(self._claims),
            "active": len(active),
            "decayed": len(decayed),
            "revoked": len(revoked),
            "avg_alpha": sum(c.alpha for c in active) / len(active) if active else 0,
            "avg_confidence": sum(c.confidence for c in active) / len(active) if active else 0,
            "total_references": sum(c.reference_count for c in self._claims.values()),
        }

    # === 内部方法 ===

    def _apply_decay(self):
        """应用时间衰减"""
        now = time.time()
        changed = False

        for claim in self._claims.values():
            if claim.status in ("revoked",):
                continue

            try:
                created = datetime.fromisoformat(claim.verified_at).timestamp()
                age_days = (now - created) / 86400
            except (ValueError, TypeError):
                continue

            # 衰减公式: confidence = max(floor, original - rate * days)
            new_confidence = max(
                self.DECAY_FLOOR,
                1.0 - self.DECAY_RATE * age_days
            )

            # 引用可以抵消衰减
            reference_boost = claim.reference_count * self.REFERENCE_BOOST
            new_confidence = min(1.0, new_confidence + reference_boost)

            if abs(new_confidence - claim.confidence) > 0.01:
                claim.confidence = new_confidence
                if claim.confidence <= self.DECAY_FLOOR and claim.status == "active":
                    claim.status = "decayed"
                changed = True

        if changed:
            self._save()

    def _propagate_revocation(self, revoked_id: str):
        """传播撤销: 依赖被撤销 claim 的其他 claim 降级"""
        claim = self._claims.get(revoked_id)
        if not claim:
            return

        for dep_id in claim.depended_by:
            dep_claim = self._claims.get(dep_id)
            if dep_claim and dep_claim.status == "active":
                dep_claim.confidence -= 0.2
                if dep_claim.confidence <= 0.2:
                    dep_claim.status = "challenged"

    def _hash_content(self, content: str) -> str:
        """生成内容 hash 作为 ID"""
        # 标准化: 去空格、小写
        normalized = content.strip().lower().replace(" ", "").replace("\n", "")
        return hashlib.sha256(normalized.encode()).hexdigest()[:12]

    def _load(self):
        """从文件加载"""
        if not os.path.exists(self.store_path):
            self._claims = {}
            return

        try:
            with open(self.store_path, "r", encoding="utf-8") as f:
                data = json.load(f)

            self._claims = {}
            for item in data.get("claims", []):
                claim = VerifiedClaim(**item)
                self._claims[claim.claim_id] = claim
        except (json.JSONDecodeError, TypeError, KeyError):
            self._claims = {}

    def _save(self):
        """原子写入文件"""
        data = {
            "version": "1.0",
            "updated_at": datetime.now(timezone.utc).isoformat(),
            "stats": self.get_stats(),
            "claims": [asdict(c) for c in self._claims.values()],
        }

        # 原子写入: 先写临时文件，再 rename
        tmp_path = self.store_path + ".tmp"
        with open(tmp_path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

        # Windows 上 rename 需要先删除目标
        if os.path.exists(self.store_path):
            os.remove(self.store_path)
        os.rename(tmp_path, self.store_path)


# ═══════════════════════════════════════════════════════════════
# Arena 集成接口
# ═══════════════════════════════════════════════════════════════

def format_prior_consensus(claims: list[VerifiedClaim]) -> str:
    """
    将已有共识格式化为可注入 Arena prompt 的文本

    这些共识会作为"已验证前提"出现在模型的输入中。
    攻击已验证前提需要更强的证据。
    """
    if not claims:
        return ""

    lines = ["以下是之前已经过多模型验证的共识 (攻击这些需要提供新的强证据):"]
    for i, claim in enumerate(claims, 1):
        confidence_bar = "█" * int(claim.confidence * 5) + "░" * (5 - int(claim.confidence * 5))
        lines.append(
            f"  [{confidence_bar}] α={claim.alpha:.2f} | "
            f"{claim.content[:100]} "
            f"(验证: {claim.agree_count}/{claim.total_count}人, "
            f"经受{claim.attacks_survived}次攻击)"
        )

    return "\n".join(lines)
