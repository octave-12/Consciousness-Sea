"""
查询历史模块 (Query History)

记录每次查询的详细信息，支持分页检索。
覆盖 REQ-P1-018 / REQ-P1-019 / REQ-P1-020 / REQ-P1-021。
"""

from __future__ import annotations

import json
import logging
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .config import (
    DEFAULT_DB_PATH,
    HISTORY_DEFAULT_LIMIT,
    HISTORY_MAX_LIMIT,
)

log = logging.getLogger(__name__)

# query_text 最大长度
MAX_QUERY_TEXT_LENGTH = 1000


def ensure_history_table(conn: sqlite3.Connection) -> None:
    """
    确保 query_history 表存在（首次调用自动建表）。

    表结构:
      - query_id: INTEGER PRIMARY KEY AUTOINCREMENT
      - query_text: TEXT NOT NULL
      - matched_seeds_count: INTEGER
      - selected_domains: TEXT (JSON 数组)
      - confidence: REAL
      - karma_direction: INTEGER
      - created_at: TEXT (ISO 8601 格式)

    同时创建 idx_history_created_at 索引。

    Args:
        conn: SQLite 数据库连接
    """
    conn.execute("""
        CREATE TABLE IF NOT EXISTS query_history (
            query_id INTEGER PRIMARY KEY AUTOINCREMENT,
            query_text TEXT NOT NULL,
            matched_seeds_count INTEGER DEFAULT 0,
            selected_domains TEXT DEFAULT '[]',
            confidence REAL DEFAULT 0.0,
            karma_direction INTEGER DEFAULT 0,
            created_at TEXT NOT NULL
        )
    """)
    conn.execute("""
        CREATE INDEX IF NOT EXISTS idx_history_created_at
        ON query_history (created_at DESC)
    """)
    conn.commit()


def record_query(
    conn: sqlite3.Connection,
    query_text: str,
    matched_seeds_count: int = 0,
    selected_domains: list[str] | None = None,
    confidence: float = 0.0,
    karma_direction: int = 0,
) -> bool:
    """
    记录一次查询到历史表。

    同步写入，异常不阻塞主流程，返回 True/False 表示写入成功/失败。
    调用方可用 try/except 包裹，失败只记日志。

    Args:
        conn: SQLite 数据库连接
        query_text: 查询文本（截断为最大 1000 字符）
        matched_seeds_count: 匹配到的种子数量
        selected_domains: 选定的领域列表
        confidence: 校验置信度
        karma_direction: 熏习方向 (+1 / -1 / 0)

    Returns:
        True 表示写入成功，False 表示写入失败
    """
    try:
        # 确保表存在
        ensure_history_table(conn)

        # 截断 query_text
        truncated_query = query_text[:MAX_QUERY_TEXT_LENGTH] if query_text else ""

        # 序列化 selected_domains
        domains_json = json.dumps(selected_domains or [], ensure_ascii=False)

        # ISO 8601 时间戳
        created_at = datetime.now(timezone.utc).isoformat()

        conn.execute(
            """
            INSERT INTO query_history
                (query_text, matched_seeds_count, selected_domains, confidence, karma_direction, created_at)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (
                truncated_query,
                matched_seeds_count,
                domains_json,
                confidence,
                karma_direction,
                created_at,
            ),
        )
        conn.commit()
        return True

    except Exception as e:
        log.warning("查询历史写入失败: %s", e)
        return False


def get_history(
    conn: sqlite3.Connection,
    limit: int = HISTORY_DEFAULT_LIMIT,
    offset: int = 0,
) -> dict[str, Any]:
    """
    获取查询历史记录（分页，按时间倒序）。

    Args:
        conn: SQLite 数据库连接
        limit: 返回条数（默认 20，最大 100）
        offset: 偏移量（默认 0）

    Returns:
        {
            "records": [...],
            "total": int,
            "limit": int,
            "offset": int,
        }
    """
    try:
        # 确保表存在
        ensure_history_table(conn)

        # 参数校验
        limit = max(1, min(limit, HISTORY_MAX_LIMIT))
        offset = max(0, offset)

        # 总数
        total = conn.execute("SELECT COUNT(*) FROM query_history").fetchone()[0]

        # 分页查询
        rows = conn.execute(
            """
            SELECT query_id, query_text, matched_seeds_count,
                   selected_domains, confidence, karma_direction, created_at
            FROM query_history
            ORDER BY created_at DESC
            LIMIT ? OFFSET ?
            """,
            (limit, offset),
        ).fetchall()

        records = []
        for row in rows:
            # 解析 selected_domains JSON
            try:
                domains = json.loads(row[3]) if row[3] else []
            except (json.JSONDecodeError, TypeError):
                domains = []

            records.append({
                'query_id': row[0],
                'query_text': row[1],
                'matched_seeds_count': row[2],
                'selected_domains': domains,
                'confidence': row[4],
                'karma_direction': row[5],
                'created_at': row[6],
            })

        return {
            'records': records,
            'total': total,
            'limit': limit,
            'offset': offset,
        }

    except Exception as e:
        log.warning("查询历史读取失败: %s", e)
        return {
            'records': [],
            'total': 0,
            'limit': limit,
            'offset': offset,
        }