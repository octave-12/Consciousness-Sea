"""
ExpertReliabilityStore — 专家可靠性分数持久化存储

职责:
  - 从 SQLite expert_reliability 表读取可靠性分数
  - 写入/更新可靠性分数
  - 内存缓存 + 持久化双保险
  - 线程安全的缓存访问
"""

from __future__ import annotations

import logging
import sqlite3
import threading
from datetime import datetime, timezone
from typing import Optional

from .config import EXPERT_RELIABILITY

log = logging.getLogger(__name__)

# 默认可靠性分数
DEFAULT_RELIABILITY = 0.7


class ExpertReliabilityStore:
    """专家可靠性分数持久化存储

    职责:
      - 从 SQLite expert_reliability 表读取可靠性分数
      - 写入/更新可靠性分数
      - 内存缓存 + 持久化双保险

    表结构:
      CREATE TABLE IF NOT EXISTS expert_reliability (
          domain  TEXT PRIMARY KEY,
          score   REAL NOT NULL CHECK(score >= 0.0 AND score <= 1.0),
          updated_at TEXT NOT NULL
      )

    Args:
        initial_scores: 初始可靠性分数（来自 config.py）
    """

    def __init__(
        self,
        initial_scores: dict[str, float] | None = None,
    ) -> None:
        self._initial_scores = initial_scores if initial_scores is not None else dict(EXPERT_RELIABILITY)
        self._cache: dict[str, float] = {}
        self._cache_lock = threading.Lock()
        self._conn: Optional[sqlite3.Connection] = None

    def initialize_table(self, conn: sqlite3.Connection) -> None:
        """创建 expert_reliability 表（幂等）

        首次调用时将 config.py 中的初始分数写入表。

        Args:
            conn: SQLite 连接
        """
        self._conn = conn

        # 幂等创建表
        conn.execute("""
            CREATE TABLE IF NOT EXISTS expert_reliability (
                domain     TEXT PRIMARY KEY,
                score      REAL NOT NULL CHECK(score >= 0.0 AND score <= 1.0),
                updated_at TEXT NOT NULL
            )
        """)
        conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_expert_reliability_domain
                ON expert_reliability (domain)
        """)

        # 写入初始分数（仅当表为空时）
        existing = conn.execute(
            "SELECT COUNT(*) FROM expert_reliability"
        ).fetchone()[0]

        if existing == 0 and self._initial_scores:
            now = datetime.now(timezone.utc).isoformat()
            for domain, score in self._initial_scores.items():
                clamped = max(0.0, min(1.0, score))
                if clamped != score:
                    log.warning(
                        "初始可靠性分数超出 [0.0, 1.0] 范围: domain=%s, score=%.4f, 截断为 %.4f",
                        domain, score, clamped,
                    )
                conn.execute(
                    "INSERT INTO expert_reliability (domain, score, updated_at) VALUES (?, ?, ?)",
                    (domain, clamped, now),
                )
                # 同时更新内存缓存
                with self._cache_lock:
                    self._cache[domain] = clamped
            conn.commit()
            log.info("初始可靠性分数已写入: %d 个领域", len(self._initial_scores))
        else:
            # 表已有数据，加载到内存缓存
            rows = conn.execute(
                "SELECT domain, score FROM expert_reliability"
            ).fetchall()
            with self._cache_lock:
                for row in rows:
                    self._cache[row[0]] = row[1]

    def get_reliability(self, domain: str) -> float:
        """获取指定领域的可靠性分数

        查找顺序:
          1. 内存缓存
          2. 数据库表
          3. 默认值 0.7

        Args:
            domain: 领域名

        Returns:
            可靠性分数 [0.0, 1.0]
        """
        # 1. 内存缓存
        with self._cache_lock:
            if domain in self._cache:
                return self._cache[domain]

        # 2. 数据库表
        if self._conn is not None:
            try:
                row = self._conn.execute(
                    "SELECT score FROM expert_reliability WHERE domain = ?",
                    (domain,),
                ).fetchone()
                if row is not None:
                    score = row[0]
                    with self._cache_lock:
                        self._cache[domain] = score
                    return score
            except Exception as e:
                log.warning("从数据库读取可靠性分数失败: domain=%s, error=%s", domain, e)

        # 3. 默认值
        return DEFAULT_RELIABILITY

    def update_reliability(self, domain: str, score: float) -> None:
        """更新可靠性分数

        写入数据库 + 更新内存缓存。
        score 超出 [0.0, 1.0] 时截断并记录 WARNING。

        Args:
            domain: 领域名
            score: 新的可靠性分数
        """
        # 截断到 [0.0, 1.0]
        clamped = max(0.0, min(1.0, score))
        if clamped != score:
            log.warning(
                "可靠性分数超出 [0.0, 1.0] 范围: domain=%s, score=%.4f, 截断为 %.4f",
                domain, score, clamped,
            )

        now = datetime.now(timezone.utc).isoformat()

        # 写入数据库
        if self._conn is not None:
            try:
                self._conn.execute(
                    "INSERT INTO expert_reliability (domain, score, updated_at) "
                    "VALUES (?, ?, ?) "
                    "ON CONFLICT (domain) DO UPDATE SET score = ?, updated_at = ?",
                    (domain, clamped, now, clamped, now),
                )
                self._conn.commit()
            except Exception as e:
                log.error("更新可靠性分数到数据库失败: domain=%s, error=%s", domain, e)
                # 即使数据库写入失败，也更新内存缓存

        # 更新内存缓存
        with self._cache_lock:
            self._cache[domain] = clamped

    def get_all_scores(self) -> dict[str, float]:
        """获取所有领域的可靠性分数

        Returns:
            {domain: score} 映射字典
        """
        # 优先从数据库读取完整数据
        if self._conn is not None:
            try:
                rows = self._conn.execute(
                    "SELECT domain, score FROM expert_reliability"
                ).fetchall()
                result = {row[0]: row[1] for row in rows}
                # 同步更新内存缓存
                with self._cache_lock:
                    self._cache.update(result)
                return result
            except Exception as e:
                log.warning("从数据库读取所有可靠性分数失败: %s", e)

        # 降级到内存缓存
        with self._cache_lock:
            return dict(self._cache)