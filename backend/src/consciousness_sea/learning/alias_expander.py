"""
AliasExpander — 别名自动扩展器

当未匹配的查询词反复指向某个种子时，自动将其追加为该种子的别名。
核心流程：记录回指事件 → 统计回指率 → 阈值判定 → 冲突检测 → 别名追加。
"""

from __future__ import annotations

import json
import logging
import threading
from dataclasses import dataclass
from datetime import datetime, timezone
from enum import Enum
from typing import Optional

from consciousness_sea.infrastructure.config import (
    ALIAS_AUTO_EXTEND,
    ALIAS_BACK_REF_THRESHOLD,
    ALIAS_CONFLICT_MARGIN,
    ALIAS_MIN_COUNT,
)
from consciousness_sea.domain.graph_db import GraphDB

log = logging.getLogger(__name__)


# ── 数据类 ──────────────────────────────────────────────────


class BackrefStatus(str, Enum):
    """回指事件状态"""

    TRACKING = "tracking"
    ALIASED = "aliased"
    CONFLICTED = "conflicted"


@dataclass
class BackrefEvent:
    """回指事件：未匹配关键词 → 专家答案中的种子"""

    source_keyword: str
    target_seed: str


@dataclass
class BackrefStats:
    """回指统计"""

    source_keyword: str
    target_seed: str
    ref_count: int
    total_count: int
    back_ref_rate: float
    status: BackrefStatus


@dataclass
class AliasExpansionResult:
    """别名扩展结果"""

    keyword: str
    seed_label: str
    back_ref_rate: float
    total_count: int
    action: str  # "aliased" / "conflicted" / "threshold_not_met" / "disabled" / "already_aliased"


# ── AliasExpander ───────────────────────────────────────────


class AliasExpander:
    """别名自动扩展器

    记录回指事件（未匹配查询词 → 专家答案中的种子），
    统计回指率并判定阈值，自动追加别名到种子，检测别名冲突。
    """

    def __init__(self, graph: GraphDB) -> None:
        self._graph = graph
        self._stats_lock = threading.Lock()

    # ── 公开接口 ────────────────────────────────────────────

    def record_backref_events(
        self,
        events: list[BackrefEvent],
        unmatched_keywords: list[str] | None = None,
    ) -> list[AliasExpansionResult]:
        """记录回指事件并执行阈值判定

        Args:
            events: 回指事件列表，每个事件包含 source_keyword 和 target_seed
            unmatched_keywords: 未匹配关键词列表（无 target_seed）

        Returns:
            每个有 target_seed 的事件经过阈值判定后的结果列表
        """
        results: list[AliasExpansionResult] = []
        now = datetime.now(timezone.utc).isoformat()

        with self._stats_lock:
            # 1. 处理有 target_seed 的回指事件
            for event in events:
                stats = self._update_backref_stats(
                    event.source_keyword, event.target_seed, now=now
                )
                if stats is not None:
                    result = self._check_threshold_and_expand(stats)
                    results.append(result)

            # 2. 处理无 target_seed 的未匹配关键词
            if unmatched_keywords:
                for keyword in unmatched_keywords:
                    self._update_backref_stats(keyword, None, now=now)

            self._graph.conn.commit()

        return results

    # ── 内部方法 ────────────────────────────────────────────

    def _update_backref_stats(
        self,
        source_keyword: str,
        target_seed: str | None,
        now: str | None = None,
    ) -> BackrefStats | None:
        """更新回指统计（UPSERT）

        有 target_seed 时: ref_count+1, total_count+1
        无 target_seed 时: 对该 keyword 的所有现有记录 total_count+1；
            无现有记录则创建 (source_keyword, "__none__") 记录

        Args:
            source_keyword: 源关键词
            target_seed: 目标种子，None 表示未匹配
            now: 当前时间 ISO 格式

        Returns:
            有 target_seed 时返回更新后的统计；无 target_seed 时返回 None
        """
        if now is None:
            now = datetime.now(timezone.utc).isoformat()

        conn = self._graph.conn

        if target_seed is not None:
            # 有 target_seed: UPSERT (source_keyword, target_seed)
            conn.execute(
                "INSERT INTO alias_backref_events "
                "(source_keyword, target_seed, ref_count, total_count, back_ref_rate, status, created_at, updated_at) "
                "VALUES (?, ?, 1, 1, 1.0, 'tracking', ?, ?) "
                "ON CONFLICT (source_keyword, target_seed) DO UPDATE "
                "SET ref_count = ref_count + 1, "
                "    total_count = total_count + 1, "
                "    back_ref_rate = CAST(ref_count + 1 AS REAL) / CAST(total_count + 1 AS REAL), "
                "    updated_at = ?",
                (source_keyword, target_seed, now, now, now),
            )

            # 读取更新后的记录
            row = conn.execute(
                "SELECT source_keyword, target_seed, ref_count, total_count, "
                "       back_ref_rate, status "
                "FROM alias_backref_events "
                "WHERE source_keyword = ? AND target_seed = ?",
                (source_keyword, target_seed),
            ).fetchone()

            if row is None:
                return None

            return BackrefStats(
                source_keyword=row["source_keyword"],
                target_seed=row["target_seed"],
                ref_count=row["ref_count"],
                total_count=row["total_count"],
                back_ref_rate=row["back_ref_rate"],
                status=BackrefStatus(row["status"]),
            )

        else:
            # 无 target_seed: 对该 keyword 的所有现有记录 total_count+1
            existing_rows = conn.execute(
                "SELECT source_keyword, target_seed FROM alias_backref_events "
                "WHERE source_keyword = ?",
                (source_keyword,),
            ).fetchall()

            if existing_rows:
                # 有现有记录：全部 total_count+1，并重算 back_ref_rate
                conn.execute(
                    "UPDATE alias_backref_events "
                    "SET total_count = total_count + 1, "
                    "    back_ref_rate = CAST(ref_count AS REAL) / CAST(total_count + 1 AS REAL), "
                    "    updated_at = ? "
                    "WHERE source_keyword = ?",
                    (now, source_keyword),
                )
            else:
                # 无现有记录：创建 (source_keyword, "__none__") 记录
                conn.execute(
                    "INSERT INTO alias_backref_events "
                    "(source_keyword, target_seed, ref_count, total_count, back_ref_rate, status, created_at, updated_at) "
                    "VALUES (?, '__none__', 0, 1, 0.0, 'tracking', ?, ?)",
                    (source_keyword, now, now),
                )

            return None

    def _check_threshold_and_expand(self, stats: BackrefStats) -> AliasExpansionResult:
        """阈值判定 + 冲突检测 + 别名追加

        判定顺序:
          1. ALIAS_AUTO_EXTEND=False → "disabled"
          2. total_count < ALIAS_MIN_COUNT → "threshold_not_met"
          3. back_ref_rate < ALIAS_BACK_REF_THRESHOLD → "threshold_not_met"
          4. status == "aliased" → "already_aliased"
          5. 冲突检测 → "conflicted" 或 "aliased"

        Args:
            stats: 回指统计

        Returns:
            别名扩展结果
        """
        keyword = stats.source_keyword
        seed_label = stats.target_seed

        # 1. 功能开关检查
        if not ALIAS_AUTO_EXTEND:
            return AliasExpansionResult(
                keyword=keyword,
                seed_label=seed_label,
                back_ref_rate=stats.back_ref_rate,
                total_count=stats.total_count,
                action="disabled",
            )

        # 2. 最小次数检查
        if stats.total_count < ALIAS_MIN_COUNT:
            return AliasExpansionResult(
                keyword=keyword,
                seed_label=seed_label,
                back_ref_rate=stats.back_ref_rate,
                total_count=stats.total_count,
                action="threshold_not_met",
            )

        # 3. 回指率阈值检查
        if stats.back_ref_rate < ALIAS_BACK_REF_THRESHOLD:
            return AliasExpansionResult(
                keyword=keyword,
                seed_label=seed_label,
                back_ref_rate=stats.back_ref_rate,
                total_count=stats.total_count,
                action="threshold_not_met",
            )

        # 4. 已是别名状态
        if stats.status == BackrefStatus.ALIASED:
            return AliasExpansionResult(
                keyword=keyword,
                seed_label=seed_label,
                back_ref_rate=stats.back_ref_rate,
                total_count=stats.total_count,
                action="already_aliased",
            )

        # 5. 冲突检测
        if self._detect_conflict(keyword, seed_label, stats.back_ref_rate):
            # 标记冲突状态
            now = datetime.now(timezone.utc).isoformat()
            self._graph.conn.execute(
                "UPDATE alias_backref_events SET status = 'conflicted', updated_at = ? "
                "WHERE source_keyword = ? AND target_seed = ?",
                (now, keyword, seed_label),
            )
            return AliasExpansionResult(
                keyword=keyword,
                seed_label=seed_label,
                back_ref_rate=stats.back_ref_rate,
                total_count=stats.total_count,
                action="conflicted",
            )

        # 6. 追加别名
        success = self._append_alias_to_seed(keyword, seed_label)
        if success:
            log.info(
                "alias auto-extended: '%s' → seed '%s', back_ref_rate=%.2f, count=%d",
                keyword,
                seed_label,
                stats.back_ref_rate,
                stats.total_count,
            )
            return AliasExpansionResult(
                keyword=keyword,
                seed_label=seed_label,
                back_ref_rate=stats.back_ref_rate,
                total_count=stats.total_count,
                action="aliased",
            )
        else:
            # 种子不存在或其他异常，视为阈值未满足
            return AliasExpansionResult(
                keyword=keyword,
                seed_label=seed_label,
                back_ref_rate=stats.back_ref_rate,
                total_count=stats.total_count,
                action="threshold_not_met",
            )

    def _detect_conflict(
        self, source_keyword: str, best_target: str, best_rate: float
    ) -> bool:
        """检测别名冲突

        查询同一 source_keyword 的其他 target_seed，
        如果次高回指率与最高之差 < ALIAS_CONFLICT_MARGIN → 冲突

        Args:
            source_keyword: 源关键词
            best_target: 最高回指率的目标种子
            best_rate: 最高回指率

        Returns:
            True 表示存在冲突
        """
        conn = self._graph.conn

        # 查找同一 keyword 下其他 target_seed 的最高回指率（排除 __none__ 和当前最佳）
        row = conn.execute(
            "SELECT target_seed, back_ref_rate "
            "FROM alias_backref_events "
            "WHERE source_keyword = ? "
            "  AND target_seed != ? "
            "  AND target_seed != '__none__' "
            "  AND status != 'aliased' "
            "ORDER BY back_ref_rate DESC "
            "LIMIT 1",
            (source_keyword, best_target),
        ).fetchone()

        if row is None:
            return False

        second_target = row["target_seed"]
        second_rate = row["back_ref_rate"]

        # 次高与最高之差 < ALIAS_CONFLICT_MARGIN → 冲突
        margin = best_rate - second_rate
        if margin < ALIAS_CONFLICT_MARGIN:
            log.warning(
                "别名冲突待审核: %s → %s(%.2f) vs %s(%.2f)",
                source_keyword,
                best_target,
                best_rate,
                second_target,
                second_rate,
            )
            return True

        return False

    def _append_alias_to_seed(self, keyword: str, seed_label: str) -> bool:
        """向种子 aliases 字段追加别名

        读取当前 aliases（NULL 视为 '[]'），解析为列表，去重追加。
        更新 seeds 表，调用 invalidate_cache()，
        更新 alias_backref_events.status = "aliased"。

        Args:
            keyword: 要追加的别名
            seed_label: 目标种子 label

        Returns:
            True 表示成功追加
        """
        conn = self._graph.conn
        now = datetime.now(timezone.utc).isoformat()

        # 1. 读取当前 aliases
        row = conn.execute(
            "SELECT aliases FROM seeds WHERE label = ?",
            (seed_label,),
        ).fetchone()

        if row is None:
            log.warning(
                "alias append failed: seed '%s' not found", seed_label
            )
            return False

        aliases_raw = row["aliases"]
        if aliases_raw is None or aliases_raw == "":
            aliases_raw = "[]"

        try:
            aliases: list[str] = json.loads(aliases_raw)
        except (json.JSONDecodeError, TypeError):
            aliases = []

        # 2. 去重追加
        if keyword in aliases:
            # 已存在，仅更新状态
            conn.execute(
                "UPDATE alias_backref_events SET status = 'aliased', updated_at = ? "
                "WHERE source_keyword = ? AND target_seed = ?",
                (now, keyword, seed_label),
            )
            return True

        aliases.append(keyword)

        # 3. 更新 seeds 表
        conn.execute(
            "UPDATE seeds SET aliases = ? WHERE label = ?",
            (json.dumps(aliases, ensure_ascii=False), seed_label),
        )

        # 4. 更新 alias_backref_events 状态
        conn.execute(
            "UPDATE alias_backref_events SET status = 'aliased', updated_at = ? "
            "WHERE source_keyword = ? AND target_seed = ?",
            (now, keyword, seed_label),
        )

        # 5. 刷新缓存
        self._graph.invalidate_cache()

        return True

    # ── 查询接口 ────────────────────────────────────────────

    def get_alias_stats(self) -> dict:
        """查询别名扩展统计

        Returns:
            包含 total_tracked, total_aliased, total_conflicted,
            recent_aliases, conflicted_items 的字典
        """
        conn = self._graph.conn

        rows = conn.execute(
            "SELECT status, COUNT(*) as cnt FROM alias_backref_events GROUP BY status"
        ).fetchall()
        counts = {r['status']: r['cnt'] for r in rows}
        total_tracked = counts.get('tracking', 0)
        total_aliased = counts.get('aliased', 0)
        total_conflicted = counts.get('conflicted', 0)
        # 最近10条已别名化的记录
        recent_rows = conn.execute(
            "SELECT source_keyword, target_seed, back_ref_rate, total_count, updated_at "
            "FROM alias_backref_events "
            "WHERE status = 'aliased' "
            "ORDER BY updated_at DESC "
            "LIMIT 10"
        ).fetchall()

        recent_aliases = [
            {
                "keyword": r["source_keyword"],
                "seed_label": r["target_seed"],
                "back_ref_rate": r["back_ref_rate"],
                "total_count": r["total_count"],
                "updated_at": r["updated_at"],
            }
            for r in recent_rows
        ]

        # 当前冲突项
        conflicted_rows = conn.execute(
            "SELECT source_keyword, target_seed, back_ref_rate, total_count, updated_at "
            "FROM alias_backref_events "
            "WHERE status = 'conflicted' "
            "ORDER BY updated_at DESC"
        ).fetchall()

        conflicted_items = [
            {
                "keyword": r["source_keyword"],
                "seed_label": r["target_seed"],
                "back_ref_rate": r["back_ref_rate"],
                "total_count": r["total_count"],
                "updated_at": r["updated_at"],
            }
            for r in conflicted_rows
        ]

        return {
            "total_tracked": total_tracked,
            "total_aliased": total_aliased,
            "total_conflicted": total_conflicted,
            "recent_aliases": recent_aliases,
            "conflicted_items": conflicted_items,
        }