"""
低权业力边定期清理器

扫描并删除所有 weight < KARMA_MIN 的业力边，
保护 source_tag='loong_cg_import' 的初始导入边（除非权重确实 < KARMA_MIN），
检测孤立节点数量。

REQ-P2-009、REQ-P2-010、REQ-P2-011
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from typing import Any

from .connection_pool import ConnectionPool
from .graph_db import GraphDB

log = logging.getLogger(__name__)


class KarmaCleaner:
    """低权业力边定期清理器

    接收 ConnectionPool 实例，扫描并删除所有 weight < KARMA_MIN 的业力边。

    Args:
        pool: 连接池实例
    """

    def __init__(self, pool: ConnectionPool) -> None:
        self._pool = pool

    def cleanup_low_weight_edges(self) -> dict[str, int]:
        """扫描并删除所有 weight < KARMA_MIN 的业力边

        保护 source_tag='loong_cg_import' 的初始导入边（除非权重确实 < KARMA_MIN）。
        删除后检测孤立节点数量。

        Returns:
            {'deleted': int, 'protected': int, 'orphaned_nodes': int}
        """
        from .config import KARMA_MIN

        graph = self._pool.acquire()
        try:
            # 显式开启事务，确保所有删除操作原子性
            graph.conn.execute("BEGIN IMMEDIATE")
            
            # 查找低权边
            rows = graph.conn.execute(
                "SELECT source, target, relation, weight, source_tag "
                "FROM karma_edges WHERE weight < ?",
                (KARMA_MIN,)
            ).fetchall()

            deleted = 0
            protected = 0

            for r in rows:
                # 注：loong_cg_import 边若 weight < KARMA_MIN 也会被删除
                # 保护条件 r['weight'] >= KARMA_MIN 在此查询结果中恒为 False（已过滤 weight < KARMA_MIN）
                # 若需保护所有 loong_cg_import 边，应在查询 WHERE 中添加 AND source_tag != 'loong_cg_import'

                graph.conn.execute(
                    "DELETE FROM karma_edges WHERE source=? AND target=? AND relation=?",
                    (r['source'], r['target'], r['relation'])
                )
                log.info(
                    "karma edge deleted: %s → %s (%s), final_weight=%.4f",
                    r['source'], r['target'], r['relation'], r['weight']
                )
                deleted += 1

                # 全局业力边删除后，提炼池候选退回（REQ-P2-029）
                self._handle_distillation_cooldown(
                    graph, r['source'], r['target'], r['relation']
                )

            graph.conn.commit()

            # 检测孤立节点
            orphaned = self._count_orphaned_nodes(graph)

            return {
                'deleted': deleted,
                'protected': protected,
                'orphaned_nodes': orphaned,
            }
        finally:
            self._pool.release(graph)

    def _count_orphaned_nodes(self, graph: GraphDB) -> int:
        """统计既无出边也无入边的种子节点数"""
        row = graph.conn.execute("""
            SELECT COUNT(*) FROM seeds s
            WHERE s.label NOT IN (SELECT source FROM karma_edges)
              AND s.label NOT IN (SELECT target FROM karma_edges)
              AND s.type != 'USER'
        """).fetchone()
        return row[0] if row else 0

    def _handle_distillation_cooldown(
        self, graph: GraphDB, source: str, target: str, relation: str,
    ) -> None:
        """全局业力边删除后，提炼池候选退回

        当全局业力边因权重降至 KARMA_MIN 以下被删除时，
        将提炼池中对应候选的 count 减 1，若 count < DISTILLATION_THRESHOLD
        则状态改为 cooled。

        Args:
            graph: 知识图谱连接（已持有，不额外获取）
            source: 源节点 label
            target: 目标节点 label
            relation: 关系类型
        """
        from .config import DISTILLATION_THRESHOLD

        try:
            row = graph.conn.execute(
                "SELECT candidate_id, count FROM distillation_pool "
                "WHERE canonical_source=? AND canonical_target=? AND canonical_relation=? "
                "AND status='upgraded'",
                (source, target, relation)
            ).fetchone()

            if row:
                new_count = max(0, row['count'] - 1)
                new_status = 'cooled' if new_count < DISTILLATION_THRESHOLD else 'upgraded'
                graph.conn.execute(
                    "UPDATE distillation_pool SET count=?, status=?, updated_at=? "
                    "WHERE candidate_id=?",
                    (new_count, new_status,
                     datetime.now(timezone.utc).isoformat(),
                     row['candidate_id'])
                )
        except Exception as e:
            log.warning("提炼池冷却退回失败: %s", e)