"""
ColdStartManager — 新用户冷启动管理器

新用户在冷启动期内，博识偏向权重较低（cold_factor < 1.0），
随着查询次数增加逐步衰减至 1.0，使系统从保守策略平滑过渡到正常策略。

核心公式:
    cold_factor = min(query_count / COLD_START_QUERIES, 1.0)

异常恢复:
    当 user_cold_start 表无记录时，从 karma_edges_personal 表
    估算查询次数，避免数据丢失后新用户被误判为冷启动。
"""

from __future__ import annotations

import threading
import logging
from dataclasses import dataclass
from datetime import datetime, timezone

from .config import COLD_START_ENABLED, COLD_START_QUERIES, KARMA_MAX_PAIRS
from .graph_db import GraphDB

log = logging.getLogger(__name__)


@dataclass
class ColdStartState:
    """用户冷启动状态快照"""

    user_label: str
    query_count: int
    is_cold_start: bool
    cold_factor: float  # [0.0, 1.0]


class ColdStartManager:
    """新用户冷启动管理器

    管理用户冷启动期判定、cold_factor 计算与查询计数持久化。
    线程安全：内部使用 double-checked locking 保护缓存读写。
    """

    def __init__(self, graph: GraphDB) -> None:
        self._graph = graph
        self._cache: dict[str, ColdStartState] = {}
        self._cache_lock = threading.Lock()

    # ── 公开接口 ─────────────────────────────────────────────

    def get_cold_factor(self, user_label: str | None) -> float:
        """获取用户的冷启动衰减系数

        返回:
          - 无用户标识 → 1.0
          - COLD_START_ENABLED=False → 1.0
          - 冷启动期内 → query_count / COLD_START_QUERIES
          - 冷启动期后 → 1.0
        """
        if not user_label:
            return 1.0
        if not COLD_START_ENABLED:
            return 1.0
        state = self._get_or_load_state(user_label)
        return state.cold_factor

    def increment_query_count(self, user_label: str | None) -> int:
        """递增用户查询计数（UPSERT + 更新缓存）

        使用 INSERT ... ON CONFLICT DO UPDATE 实现原子递增，
        避免并发场景下计数丢失。

        Args:
            user_label: 用户标识，为 None 时直接返回 0

        Returns:
            递增后的查询计数，user_label 为 None 时返回 0
        """
        if not user_label:
            return 0

        now = datetime.now(timezone.utc).isoformat()

        # UPSERT: 无记录则插入 (query_count=1)，有记录则递增
        self._graph.conn.execute(
            "INSERT INTO user_cold_start (user_label, query_count, updated_at) "
            "VALUES (?, 1, ?) "
            "ON CONFLICT (user_label) DO UPDATE SET "
            "query_count = query_count + 1, updated_at = ?",
            (user_label, now, now),
        )
        self._graph.conn.commit()

        # 读取递增后的值
        row = self._graph.conn.execute(
            "SELECT query_count FROM user_cold_start WHERE user_label = ?",
            (user_label,),
        ).fetchone()
        new_count: int = row[0] if row else 1

        # 同步更新内存缓存
        cold_factor = min(new_count / COLD_START_QUERIES, 1.0)
        is_cold_start = new_count < COLD_START_QUERIES
        updated_state = ColdStartState(
            user_label=user_label,
            query_count=new_count,
            is_cold_start=is_cold_start,
            cold_factor=cold_factor,
        )
        with self._cache_lock:
            self._cache[user_label] = updated_state

        log.debug(
            "cold_start increment: user=%s, count=%d, factor=%.4f",
            user_label,
            new_count,
            cold_factor,
        )
        return new_count

    def get_state(self, user_label: str) -> ColdStartState:
        """查询用户冷启动状态

        优先从内存缓存读取，缓存未命中则从数据库加载。

        Args:
            user_label: 用户标识

        Returns:
            ColdStartState 数据类实例
        """
        return self._get_or_load_state(user_label)

    def invalidate_cache(self, user_label: str | None = None) -> None:
        """清除缓存

        Args:
            user_label: 指定用户则只清除该用户缓存，为 None 则清除全部
        """
        with self._cache_lock:
            if user_label is None:
                self._cache.clear()
                log.debug("cold_start cache fully invalidated")
            else:
                self._cache.pop(user_label, None)
                log.debug("cold_start cache invalidated for user=%s", user_label)

    # ── 内部方法 ─────────────────────────────────────────────

    def _get_or_load_state(self, user_label: str) -> ColdStartState:
        """获取或加载用户冷启动状态（double-checked locking）

        快速路径：缓存命中直接返回（无锁）。
        慢路径：加锁后再次检查缓存，仍未命中则从数据库加载。

        Args:
            user_label: 用户标识

        Returns:
            ColdStartState 数据类实例
        """
        # 快速路径：缓存命中
        if user_label in self._cache:
            return self._cache[user_label]

        # 慢路径：加锁加载
        with self._cache_lock:
            if user_label in self._cache:
                return self._cache[user_label]
            state = self._load_state(user_label)
            self._cache[user_label] = state
            return state

    def _load_state(self, user_label: str) -> ColdStartState:
        """从数据库加载用户冷启动状态

        异常恢复策略:
          1. user_cold_start 表有记录 → 直接使用
          2. 无记录时从 karma_edges_personal 表估算查询次数:
             estimated_count = personal_edge_count / max(KARMA_MAX_PAIRS, 1)
          3. 无法估算时视为冷启动期（query_count=0）

        Args:
            user_label: 用户标识

        Returns:
            ColdStartState 数据类实例
        """
        # 尝试从 user_cold_start 表读取
        row = self._graph.conn.execute(
            "SELECT query_count FROM user_cold_start WHERE user_label = ?",
            (user_label,),
        ).fetchone()

        if row:
            query_count: int = row[0]
        else:
            # 异常恢复：从个人业力边估算查询次数
            query_count = self._estimate_count_from_karma(user_label)
            log.info(
                "cold_start recovery: user=%s, estimated_count=%d (from karma_edges_personal)",
                user_label,
                query_count,
            )

        cold_factor = min(query_count / COLD_START_QUERIES, 1.0)
        is_cold_start = query_count < COLD_START_QUERIES

        return ColdStartState(
            user_label=user_label,
            query_count=query_count,
            is_cold_start=is_cold_start,
            cold_factor=cold_factor,
        )

    def _estimate_count_from_karma(self, user_label: str) -> int:
        """从个人业力边估算查询次数

        每次查询最多产生 KARMA_MAX_PAIRS 条个人业力边，
        因此估算公式为: personal_edge_count // max(KARMA_MAX_PAIRS, 1)

        Args:
            user_label: 用户标识

        Returns:
            估算的查询次数，最低为 0
        """
        try:
            row = self._graph.conn.execute(
                "SELECT COUNT(*) FROM karma_edges_personal WHERE user_label = ?",
                (user_label,),
            ).fetchone()
            personal_edge_count: int = row[0] if row else 0
            estimated_count = personal_edge_count // max(KARMA_MAX_PAIRS, 1)
            return estimated_count
        except Exception:
            log.warning(
                "cold_start estimate failed for user=%s, falling back to 0",
                user_label,
                exc_info=True,
            )
            return 0