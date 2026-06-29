"""
SeedCandidateManager — 候选种子管理器

处理未匹配关键词的候选种子创建、计数累加、升级为正式种子、
领域推断、初始业力边构建、过期与清理机制。

Phase 3 自生长核心组件 (#18)。
"""

from __future__ import annotations

import json
import logging
import sqlite3
import threading
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from enum import Enum

from consciousness_sea.infrastructure.config import (
    CANDIDATE_SEED_AUTO_CREATE,
    CANDIDATE_SEED_EXPIRE_DAYS,
    CANDIDATE_SEED_MIN_COUNT,
    CANDIDATE_SEED_PROMOTE_COUNT,
    CANDIDATE_SEED_PURGE_DAYS,
)
from consciousness_sea.domain.graph_db import GraphDB

log = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════
#  数据类与枚举
# ═══════════════════════════════════════════════════════════


class CandidateStatus(str, Enum):
    """候选种子状态枚举"""

    CANDIDATE = "candidate"
    PROMOTED = "promoted"
    EXPIRED = "expired"


@dataclass
class CandidateSeed:
    """候选种子数据类"""

    label: str
    status: CandidateStatus
    count: int
    domain: str | None
    co_occur_seeds: list[str]
    candidate_since: str
    last_seen_at: str
    promoted_at: str | None
    promoted_seed_id: str | None


@dataclass
class PromotionResult:
    """升级操作结果"""

    label: str
    domain: str
    initial_edges: int
    success: bool
    error: str | None = None


# ═══════════════════════════════════════════════════════════
#  SeedCandidateManager
# ═══════════════════════════════════════════════════════════


class SeedCandidateManager:
    """候选种子管理器

    职责：
    1. 处理未匹配关键词（创建候选种子或累加计数）
    2. 候选种子升级为正式种子（原子操作）
    3. 领域推断
    4. 初始业力边构建
    5. 过期和清理机制
    """

    def __init__(self, graph: GraphDB) -> None:
        self._graph = graph
        self._counters_lock = threading.Lock()
        self._pending_counters: dict[str, int] = {}  # 内存预计数器

    # ───────────────────────────────────────────────────────
    #  公开接口
    # ───────────────────────────────────────────────────────

    def process_unmatched_keywords(
        self,
        keywords: list[str],
        co_occur_seeds: list[str] | None = None,
    ) -> int:
        """处理未匹配关键词列表

        对每个关键词：
        - 若 CANDIDATE_SEED_AUTO_CREATE 为 False，仅记录统计日志
        - 否则调用 _increment_or_create() 累加计数或创建候选种子

        Args:
            keywords: 未匹配关键词列表
            co_occur_seeds: 与这些关键词共现的正式种子列表

        Returns:
            实际处理（写入或计数）的关键词数量
        """
        if not keywords:
            return 0

        if not CANDIDATE_SEED_AUTO_CREATE:
            log.info(
                "unmatched keywords skipped (auto_create disabled): %d keywords, sample=%s",
                len(keywords),
                keywords[:5],
            )
            return 0

        processed = 0
        for keyword in keywords:
            if not keyword or not keyword.strip():
                continue
            keyword = keyword.strip()
            self._increment_or_create(keyword, co_occur_seeds)
            processed += 1

        log.debug(
            "processed unmatched keywords: %d/%d, co_occur=%s",
            processed,
            len(keywords),
            co_occur_seeds,
        )
        return processed

    def promote_candidate(self, label: str) -> PromotionResult:
        """将候选种子升级为正式种子（原子操作）

        流程：
        1. 查询候选种子记录，验证状态为 candidate 且 count >= PROMOTE_COUNT
        2. 开启 IMMEDIATE 事务
        3. 推断领域
        4. 插入正式种子（INSERT OR IGNORE）
        5. 构建初始业力边
        6. 更新候选种子状态为 promoted
        7. 提交事务

        Args:
            label: 候选种子标签

        Returns:
            PromotionResult 包含升级结果
        """
        # 查询候选种子记录
        row = self._graph.conn.execute(
            "SELECT * FROM candidate_seeds WHERE label = ? AND status = ?",
            (label, CandidateStatus.CANDIDATE),
        ).fetchone()

        if not row:
            return PromotionResult(
                label=label,
                domain="",
                initial_edges=0,
                success=False,
                error=f"candidate seed not found or not in candidate status: '{label}'",
            )

        candidate = self._row_to_candidate(row)

        if candidate.count < CANDIDATE_SEED_PROMOTE_COUNT:
            return PromotionResult(
                label=label,
                domain="",
                initial_edges=0,
                success=False,
                error=(
                    f"candidate seed count ({candidate.count}) "
                    f"below promote threshold ({CANDIDATE_SEED_PROMOTE_COUNT})"
                ),
            )

        co_occur_seeds = candidate.co_occur_seeds
        now = datetime.now(timezone.utc).isoformat()

        try:
            self._graph.conn.execute("BEGIN IMMEDIATE")

            # 推断领域
            domain = self._infer_domain(label, co_occur_seeds)

            # 插入正式种子（INSERT OR IGNORE 防止重复）
            self._graph.conn.execute(
                "INSERT OR IGNORE INTO seeds (label, domain, definition, aliases, activation_bias) "
                "VALUES (?, ?, '', '[]', 0.05)",
                (label, domain),
            )

            # 构建初始业力边
            edge_count = self._build_initial_karma_edges(label, co_occur_seeds)

            # 更新候选种子状态
            self._graph.conn.execute(
                "UPDATE candidate_seeds SET status='promoted', promoted_at=?, promoted_seed_id=? "
                "WHERE label=?",
                (now, label, label),
            )

            self._graph.conn.commit()

            log.info(
                "candidate seed promoted: '%s', domain='%s', initial_edges=%d",
                label,
                domain,
                edge_count,
            )

            return PromotionResult(
                label=label,
                domain=domain,
                initial_edges=edge_count,
                success=True,
            )

        except Exception as exc:
            self._graph.conn.rollback()
            log.error("candidate promotion failed: '%s', error=%s", label, exc)
            return PromotionResult(
                label=label,
                domain="",
                initial_edges=0,
                success=False,
                error=str(exc),
            )

    def expire_candidates(self) -> int:
        """标记过期候选种子

        将 last_seen_at 距今超过 CANDIDATE_SEED_EXPIRE_DAYS 天的
        status='candidate' 记录标记为 'expired'。

        Returns:
            被标记为过期的记录数
        """
        cutoff = (
            datetime.now(timezone.utc) - timedelta(days=CANDIDATE_SEED_EXPIRE_DAYS)
        ).isoformat()

        cursor = self._graph.conn.execute(
            "UPDATE candidate_seeds SET status = ? "
            "WHERE status = ? AND last_seen_at < ?",
            (CandidateStatus.EXPIRED, CandidateStatus.CANDIDATE, cutoff),
        )
        self._graph.conn.commit()

        expired_count = cursor.rowcount
        if expired_count > 0:
            log.info("expired %d candidate seeds (cutoff=%s)", expired_count, cutoff)

        return expired_count

    def purge_expired_candidates(self) -> int:
        """清理长期过期的候选种子

        删除 status='expired' 且 promoted_at 距今超过
        CANDIDATE_SEED_PURGE_DAYS 天的记录。
        若 promoted_at 为 NULL，则使用 candidate_since 判断。

        Returns:
            被删除的记录数
        """
        cutoff = (
            datetime.now(timezone.utc) - timedelta(days=CANDIDATE_SEED_PURGE_DAYS)
        ).isoformat()

        # 对于 expired 状态的记录，promoted_at 可能为 NULL
        # 使用 COALESCE 回退到 candidate_since
        cursor = self._graph.conn.execute(
            "DELETE FROM candidate_seeds "
            "WHERE status = ? AND COALESCE(promoted_at, candidate_since) < ?",
            (CandidateStatus.EXPIRED, cutoff),
        )
        self._graph.conn.commit()

        purged_count = cursor.rowcount
        if purged_count > 0:
            log.info("purged %d expired candidate seeds (cutoff=%s)", purged_count, cutoff)

        return purged_count

    def get_status(self) -> dict:
        """查询候选种子状态

        Returns:
            包含总数、各状态计数、最近升级记录的字典
        """
        total = self._graph.conn.execute(
            "SELECT COUNT(*) FROM candidate_seeds"
        ).fetchone()[0]

        candidate_count = self._graph.conn.execute(
            "SELECT COUNT(*) FROM candidate_seeds WHERE status = ?",
            (CandidateStatus.CANDIDATE,),
        ).fetchone()[0]

        promoted_count = self._graph.conn.execute(
            "SELECT COUNT(*) FROM candidate_seeds WHERE status = ?",
            (CandidateStatus.PROMOTED,),
        ).fetchone()[0]

        expired_count = self._graph.conn.execute(
            "SELECT COUNT(*) FROM candidate_seeds WHERE status = ?",
            (CandidateStatus.EXPIRED,),
        ).fetchone()[0]

        # 最近升级记录（最多 10 条）
        recent_rows = self._graph.conn.execute(
            "SELECT label, domain, promoted_at, count "
            "FROM candidate_seeds "
            "WHERE status = ? AND promoted_at IS NOT NULL "
            "ORDER BY promoted_at DESC LIMIT 10",
            (CandidateStatus.PROMOTED,),
        ).fetchall()

        recent_promotions = [
            {
                "label": r["label"],
                "domain": r["domain"],
                "promoted_at": r["promoted_at"],
                "count": r["count"],
            }
            for r in recent_rows
        ]

        return {
            "total_candidates": total,
            "candidate_count": candidate_count,
            "promoted_count": promoted_count,
            "expired_count": expired_count,
            "recent_promotions": recent_promotions,
        }

    # ───────────────────────────────────────────────────────
    #  内部方法
    # ───────────────────────────────────────────────────────

    def _should_create_candidate(self, keyword: str) -> bool:
        """判断是否应创建候选种子

        排除以下情况：
        1. 已是正式种子的关键词（seeds 表中存在 label=keyword）
        2. 已通过别名关联的关键词（alias_index 中存在）
        3. 已有候选种子记录的关键词（candidate_seeds 表中存在 status='candidate'）

        Args:
            keyword: 待检查的关键词

        Returns:
            True 表示可以创建候选种子
        """
        # 排除 1: 已是正式种子
        seed_row = self._graph.conn.execute(
            "SELECT 1 FROM seeds WHERE label = ?", (keyword,)
        ).fetchone()
        if seed_row:
            return False

        # 排除 2: 已通过别名关联
        self._graph._ensure_alias_index()
        if keyword in self._graph._alias_index:
            return False

        # 排除 3: 已有候选种子记录（status='candidate'）
        candidate_row = self._graph.conn.execute(
            "SELECT 1 FROM candidate_seeds WHERE label = ? AND status = ?",
            (keyword, CandidateStatus.CANDIDATE),
        ).fetchone()
        if candidate_row:
            return False

        return True

    def _increment_or_create(
        self, keyword: str, co_occur_seeds: list[str] | None
    ) -> None:
        """累加计数或创建候选种子

        使用内存预计数器 _pending_counters：
        - 首次出现的关键词在内存中计数
        - 达到 MIN_COUNT 阈值时才写入数据库
        - 已在数据库中的候选种子直接累加 count 并更新 last_seen_at

        Args:
            keyword: 关键词
            co_occur_seeds: 共现种子列表
        """
        now = datetime.now(timezone.utc).isoformat()

        # 检查数据库中是否已有 candidate 状态的记录
        existing = self._graph.conn.execute(
            "SELECT count FROM candidate_seeds WHERE label = ? AND status = ?",
            (keyword, CandidateStatus.CANDIDATE),
        ).fetchone()

        if existing:
            # 已在数据库中，直接累加（使用 count = count + 1 避免 read-modify-write 竞态）
            self._graph.conn.execute(
                "UPDATE candidate_seeds SET count = count + 1, last_seen_at = ? WHERE label = ?",
                (now, keyword),
            )
            self._graph.conn.commit()
            return

        # 不在数据库中，使用内存预计数器
        with self._counters_lock:
            self._pending_counters[keyword] = self._pending_counters.get(keyword, 0) + 1
            current_count = self._pending_counters[keyword]

        # 未达到阈值，仅内存计数
        if current_count < CANDIDATE_SEED_MIN_COUNT:
            return

        # 达到阈值，检查是否应创建候选种子
        if not self._should_create_candidate(keyword):
            # 不应创建，清理内存计数器
            with self._counters_lock:
                self._pending_counters.pop(keyword, None)
            return

        # 创建候选种子记录
        co_occur_json = json.dumps(co_occur_seeds or [], ensure_ascii=False)

        try:
            self._graph.conn.execute(
                "INSERT OR IGNORE INTO candidate_seeds "
                "(label, status, count, domain, co_occur_seeds, candidate_since, last_seen_at) "
                "VALUES (?, ?, ?, NULL, ?, ?, ?)",
                (keyword, CandidateStatus.CANDIDATE, current_count, co_occur_json, now, now),
            )
            self._graph.conn.commit()

            # 创建成功后清理内存计数器
            with self._counters_lock:
                self._pending_counters.pop(keyword, None)

            log.info(
                "candidate seed created: '%s', count=%d, co_occur=%s",
                keyword,
                current_count,
                co_occur_seeds,
            )

        except Exception as exc:
            log.error("failed to create candidate seed '%s': %s", keyword, exc)
            # 创建失败不清理计数器，下次可重试

    def _infer_domain(self, label: str, co_occur_seeds: list[str]) -> str:
        """根据共现种子推断领域

        批量查询共现种子的 domain，按 domain 分组计数，
        选择出现次数最多的 domain。无共现种子或无领域信息时返回 "未分类"。

        Args:
            label: 候选种子标签（用于日志）
            co_occur_seeds: 共现种子列表

        Returns:
            推断出的领域名称
        """
        if not co_occur_seeds:
            log.debug("no co-occur seeds for '%s', domain fallback to '未分类'", label)
            return "未分类"

        # 批量查询共现种子的 domain
        seed_info_map = self._graph.batch_get_seeds(co_occur_seeds)

        # 按 domain 分组计数
        domain_counts: dict[str, int] = {}
        for seed_label in co_occur_seeds:
            info = seed_info_map.get(seed_label)
            if info and info.get("domain"):
                domain = info["domain"]
                domain_counts[domain] = domain_counts.get(domain, 0) + 1

        if not domain_counts:
            log.debug(
                "no domain info from co-occur seeds for '%s', domain fallback to '未分类'",
                label,
            )
            return "未分类"

        # 选择出现次数最多的 domain
        best_domain = max(domain_counts, key=domain_counts.get)  # type: ignore[arg-type]
        log.debug(
            "inferred domain for '%s': '%s' (counts=%s)",
            label,
            best_domain,
            domain_counts,
        )
        return best_domain

    def _build_initial_karma_edges(
        self, new_seed_label: str, co_occur_seeds: list[str]
    ) -> int:
        """创建初始业力边（双向，weight=0.05，source_tag='candidate_promotion'）

        对每个共现种子创建双向 RELATED 边，使用 ON CONFLICT DO NOTHING
        避免覆盖已有边。

        Args:
            new_seed_label: 新升级的正式种子标签
            co_occur_seeds: 共现种子列表

        Returns:
            实际创建的边数
        """
        if not co_occur_seeds:
            return 0

        edge_count = 0


        for co_seed in co_occur_seeds:
            if co_seed == new_seed_label:
                continue

            # 正向边: new_seed → co_seed
            cursor_forward = self._graph.conn.execute(
                "INSERT INTO karma_edges (source, target, relation, weight, source_tag) "
                "VALUES (?, ?, 'RELATED', 0.05, 'candidate_promotion') "
                "ON CONFLICT (source, target, relation) DO NOTHING",
                (new_seed_label, co_seed),
            )
            if cursor_forward.rowcount > 0:
                edge_count += 1

            # 反向边: co_seed → new_seed
            cursor_backward = self._graph.conn.execute(
                "INSERT INTO karma_edges (source, target, relation, weight, source_tag) "
                "VALUES (?, ?, 'RELATED', 0.05, 'candidate_promotion') "
                "ON CONFLICT (source, target, relation) DO NOTHING",
                (co_seed, new_seed_label),
            )
            if cursor_backward.rowcount > 0:
                edge_count += 1

        # 对已存在的边更新 source_tag 标记
        for co_seed in co_occur_seeds:
            if co_seed == new_seed_label:
                continue
            self._graph.conn.execute(
                "UPDATE karma_edges SET source_tag = 'candidate_promotion' "
                "WHERE source = ? AND target = ? AND relation = 'RELATED' "
                "AND source_tag != 'candidate_promotion'",
                (new_seed_label, co_seed),
            )
            self._graph.conn.execute(
                "UPDATE karma_edges SET source_tag = 'candidate_promotion' "
                "WHERE source = ? AND target = ? AND relation = 'RELATED' "
                "AND source_tag != 'candidate_promotion'",
                (co_seed, new_seed_label),
            )

        log.debug(
            "built %d initial karma edges for '%s', co_occur=%s",
            edge_count,
            new_seed_label,
            co_occur_seeds,
        )

        return edge_count

    # ───────────────────────────────────────────────────────
    #  辅助方法
    # ───────────────────────────────────────────────────────

    @staticmethod
    def _row_to_candidate(row: sqlite3.Row) -> CandidateSeed:
        """将数据库行转换为 CandidateSeed 数据类

        Args:
            row: sqlite3.Row 对象

        Returns:
            CandidateSeed 实例
        """

        co_occur_raw = row["co_occur_seeds"]
        if isinstance(co_occur_raw, str):
            co_occur_seeds = json.loads(co_occur_raw)
        else:
            co_occur_seeds = co_occur_raw or []

        return CandidateSeed(
            label=row["label"],
            status=CandidateStatus(row["status"]),
            count=row["count"],
            domain=row["domain"],
            co_occur_seeds=co_occur_seeds,
            candidate_since=row["candidate_since"],
            last_seen_at=row["last_seen_at"],
            promoted_at=row["promoted_at"],
            promoted_seed_id=row["promoted_seed_id"],
        )