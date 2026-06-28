"""
参数统计收集 — 每次查询后记录参数快照

记录字段：query_text、decay_factor、domain_threshold、confidence_high、
ripple_depth、activated_count、selected_domains、confidence、karma_direction、created_at

统计记录失败不影响查询结果返回（由调用方 try/except 包裹）。
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .graph_db import GraphDB
    from .router import RippleResult

log = logging.getLogger(__name__)


# ── param_stats 表建表 SQL ──────────────────────────────────
_CREATE_PARAM_STATS_TABLE = """
CREATE TABLE IF NOT EXISTS param_stats (
    stat_id          INTEGER PRIMARY KEY AUTOINCREMENT,
    query_text       TEXT    NOT NULL,
    decay_factor     REAL    NOT NULL,
    domain_threshold REAL    NOT NULL,
    confidence_high  REAL    NOT NULL,
    ripple_depth     INTEGER NOT NULL,
    activated_count  INTEGER NOT NULL,
    selected_domains TEXT    NOT NULL,
    confidence       REAL    NOT NULL,
    karma_direction  INTEGER NOT NULL,
    created_at       TEXT    NOT NULL
)
"""

_CREATE_PARAM_STATS_INDEX = """
CREATE INDEX IF NOT EXISTS idx_param_stats_created
    ON param_stats (created_at)
"""


def ensure_param_stats_table(graph: GraphDB) -> None:
    """确保 param_stats 表存在

    优先使用 graph.ensure_phase2_tables()（如果可用），
    否则自行创建 param_stats 表及索引。

    Args:
        graph: 知识图谱数据库连接
    """
    # 优先使用 graph.ensure_phase2_tables()（如果另一个开发者已实现）
    if hasattr(graph, 'ensure_phase2_tables'):
        try:
            graph.ensure_phase2_tables()
            return
        except Exception:
            log.debug("ensure_phase2_tables() 调用失败，回退到自行创建 param_stats 表")

    # 自行创建表和索引
    graph.conn.execute(_CREATE_PARAM_STATS_TABLE)
    graph.conn.execute(_CREATE_PARAM_STATS_INDEX)


def record_param_stats(
    graph: GraphDB,
    query_text: str,
    result: RippleResult,
    verdict: dict,
) -> None:
    """记录参数统计到 param_stats 表

    每次查询完成后调用，记录本次查询使用的参数和结果。
    统计记录失败不影响查询结果返回（由调用方 try/except 包裹）。

    Args:
        graph: 知识图谱数据库连接
        query_text: 查询文本
        result: 涟漪传播结果
        verdict: 校验结果字典，包含 confidence 和 karma_direction
    """
    from .config import RIPPLE_DECAY, DOMAIN_THRESHOLD, CONFIDENCE_HIGH

    # 确保表存在
    ensure_param_stats_table(graph)

    # 计算最大传播深度
    max_depth = max(
        (n.depth for n in result.activated.values()),
        default=0,
    )

    # 序列化选定领域为 JSON 数组
    selected_domains_json = json.dumps(
        result.selected_domains, ensure_ascii=False
    )

    # 插入统计记录
    graph.conn.execute(
        "INSERT INTO param_stats "
        "(query_text, decay_factor, domain_threshold, confidence_high, "
        " ripple_depth, activated_count, selected_domains, confidence, "
        " karma_direction, created_at) "
        "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
        (
            query_text,
            RIPPLE_DECAY,
            DOMAIN_THRESHOLD,
            CONFIDENCE_HIGH,
            max_depth,
            len(result.activated),
            selected_domains_json,
            verdict['confidence'],
            verdict['karma_direction'],
            datetime.now(timezone.utc).isoformat(),
        ),
    )
    graph.conn.commit()

    log.debug(
        "参数统计已记录: query='%s', decay=%.2f, confidence=%.3f, karma_direction=%+d",
        query_text[:30],
        RIPPLE_DECAY,
        verdict['confidence'],
        verdict['karma_direction'],
    )


# 向后兼容别名（任务描述中使用 _record_param_stats 名称）
_record_param_stats = record_param_stats