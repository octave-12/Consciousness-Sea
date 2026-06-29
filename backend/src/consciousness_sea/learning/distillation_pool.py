"""
提炼池管理器 — 候选提交、等价判定、升级逻辑

个人业力 → 提炼池 → 全局业力的升级链路。
三重等价判定：精确匹配 → 关系等价映射 → 涟漪验证。

REQ-P2-023 ~ REQ-P2-030
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from typing import Optional

from consciousness_sea.domain.graph_db import GraphDB

log = logging.getLogger(__name__)


class DistillationPool:
    """提炼池管理器 — 候选提交、等价判定、升级逻辑

    Args:
        graph: GraphDB 实例
    """

    def __init__(self, graph: GraphDB) -> None:
        self._graph = graph

    def submit_candidate(
        self,
        user_label: str,
        source: str,
        target: str,
        relation: str,
    ) -> int:
        """提交候选到提炼池

        流程:
          1. 归一化 source/target/relation（别名匹配 + 关系等价映射）
          2. 查找提炼池中是否已有等价候选
          3. 有 → 合并（count +1, 添加 contributor）
          4. 无 → 新建候选（count=1）
          5. 若 count ≥ DISTILLATION_THRESHOLD → 自动升级为全局业力

        Args:
            user_label: 用户标识
            source: 源节点 label
            target: 目标节点 label
            relation: 关系类型

        Returns:
            候选的 candidate_id
        """
        from consciousness_sea.infrastructure.config import DISTILLATION_THRESHOLD

        # 1. 归一化
        canonical_source, canonical_target, canonical_relation = self._canonicalize(
            source, target, relation
        )

        now = datetime.now(timezone.utc).isoformat()

        # 2. 查找等价候选
        existing_id = self._find_equivalent_candidate(
            canonical_source, canonical_target, canonical_relation
        )

        if existing_id is not None:
            # 3. 合并：count +1 (原子), 添加 contributor
            now_str = datetime.now(timezone.utc).isoformat()
            
            # 先获取当前 contributors 以判断是否需要添加
            row = self._graph.conn.execute(
                "SELECT count, contributor_users FROM distillation_pool "
                "WHERE candidate_id=?",
                (existing_id,)
            ).fetchone()

            if row:
                contributors: list[str] = json.loads(row['contributor_users'])
                if user_label not in contributors:
                    contributors.append(user_label)

                # 原子更新: count = count + 1
                self._graph.conn.execute(
                    "UPDATE distillation_pool SET count = count + 1, contributor_users=?, "
                    "representative_label=?, updated_at=? "
                    "WHERE candidate_id=?",
                    (json.dumps(contributors, ensure_ascii=False),
                     f"{canonical_source}→{canonical_target}",
                     now_str, existing_id)
                )

                # 读取更新后的 count 用于升级检查
                updated_row = self._graph.conn.execute(
                    "SELECT count FROM distillation_pool WHERE candidate_id=?",
                    (existing_id,)
                ).fetchone()
                new_count = updated_row['count'] if updated_row else 0

                # 5. 自动升级检查
                if new_count >= DISTILLATION_THRESHOLD and self._get_status_by_id(existing_id) == 'pending':
                    self._upgrade_to_global(
                        existing_id, canonical_source, canonical_target, canonical_relation
                    )

                return existing_id

        # 4. 新建候选
        cursor = self._graph.conn.execute(
            "INSERT INTO distillation_pool "
            "(canonical_source, canonical_target, canonical_relation, "
            " representative_label, count, contributor_users, status, created_at, updated_at) "
            "VALUES (?, ?, ?, ?, 1, ?, 'pending', ?, ?)",
            (canonical_source, canonical_target, canonical_relation,
             f"{canonical_source}→{canonical_target}",
             json.dumps([user_label], ensure_ascii=False),
             now, now)
        )
        candidate_id = cursor.lastrowid

        # 5. 自动升级检查（count=1 不会触发，但防御性检查）
        if DISTILLATION_THRESHOLD <= 1:
            self._upgrade_to_global(
                candidate_id, canonical_source, canonical_target, canonical_relation
            )

        return candidate_id

    def _canonicalize(self, source: str, target: str, relation: str) -> tuple[str, str, str]:
        """归一化候选三元组

        1. 别名匹配: 查 alias_index 将 source/target 映射到主 label
        2. 关系等价映射: 查 RELATION_EQUIVALENCE_MAP 将 relation 映射到标准关系

        Args:
            source: 原始源节点 label
            target: 原始目标节点 label
            relation: 原始关系类型

        Returns:
            (canonical_source, canonical_target, canonical_relation)
        """
        from consciousness_sea.infrastructure.config import RELATION_EQUIVALENCE_MAP

        # 1. 别名匹配
        canonical_source = source
        canonical_target = target

        self._graph._ensure_alias_index()
        if self._graph._alias_index is not None:
            if source in self._graph._alias_index:
                canonical_source = self._graph._alias_index[source]
            if target in self._graph._alias_index:
                canonical_target = self._graph._alias_index[target]

        # 2. 关系等价映射
        canonical_relation = RELATION_EQUIVALENCE_MAP.get(relation, relation)

        return canonical_source, canonical_target, canonical_relation

    def _find_equivalent_candidate(
        self, canonical_source: str, canonical_target: str, canonical_relation: str,
    ) -> int | None:
        """查找提炼池中的等价候选

        三重机制（依次尝试）:
          1. 精确匹配: canonical_source + canonical_target + canonical_relation
          2. 关系等价匹配: source/target 相同，relation 为等价关系
          3. 涟漪验证: source/target 1-hop 邻居重叠度 > OVERLAP_THRESHOLD

        Args:
            canonical_source: 归一化后的源节点
            canonical_target: 归一化后的目标节点
            canonical_relation: 归一化后的关系

        Returns:
            匹配的 candidate_id，未找到返回 None
        """
        from consciousness_sea.infrastructure.config import RELATION_EQUIVALENCE_MAP, NEIGHBOR_OVERLAP_THRESHOLD

        # 1. 精确匹配
        row = self._graph.conn.execute(
            "SELECT candidate_id FROM distillation_pool "
            "WHERE canonical_source=? AND canonical_target=? AND canonical_relation=? "
            "AND status != 'upgraded'",
            (canonical_source, canonical_target, canonical_relation)
        ).fetchone()
        if row:
            return row['candidate_id']

        # 2. 关系等价匹配
        # 收集所有与 canonical_relation 等价的关系
        equivalent_relations = {canonical_relation}
        for rel, canonical_rel in RELATION_EQUIVALENCE_MAP.items():
            if canonical_rel == canonical_relation:
                equivalent_relations.add(rel)

        if len(equivalent_relations) > 1:
            placeholders = ','.join('?' * len(equivalent_relations))
            row = self._graph.conn.execute(
                f"SELECT candidate_id FROM distillation_pool "
                f"WHERE canonical_source=? AND canonical_target=? "
                f"AND canonical_relation IN ({placeholders}) "
                f"AND status != 'upgraded'",
                [canonical_source, canonical_target] + list(equivalent_relations)
            ).fetchone()
            if row:
                return row['candidate_id']

        # 3. 涟漪验证：source/target 1-hop 邻居重叠度
        source_neighbors = self._get_1hop_neighbors(canonical_source)
        target_neighbors = self._get_1hop_neighbors(canonical_target)

        # 查找所有 pending/cooled 候选
        candidates = self._graph.conn.execute(
            "SELECT candidate_id, canonical_source, canonical_target "
            "FROM distillation_pool WHERE status != 'upgraded'"
        ).fetchall()

        for c in candidates:
            c_source_neighbors = self._get_1hop_neighbors(c['canonical_source'])
            c_target_neighbors = self._get_1hop_neighbors(c['canonical_target'])

            # 计算 source 和 target 的邻居重叠度
            overlap = self._compute_neighbor_overlap(
                source_neighbors, c_source_neighbors
            ) * self._compute_neighbor_overlap(
                target_neighbors, c_target_neighbors
            )

            if overlap > NEIGHBOR_OVERLAP_THRESHOLD:
                return c['candidate_id']

        return None

    def _get_1hop_neighbors(self, label: str) -> set[str]:
        """获取节点的 1-hop 邻居集合

        Args:
            label: 节点 label

        Returns:
            邻居 label 集合
        """
        rows = self._graph.conn.execute(
            "SELECT target FROM karma_edges WHERE source=? "
            "UNION "
            "SELECT source FROM karma_edges WHERE target=?",
            (label, label)
        ).fetchall()
        return {r[0] for r in rows}

    def _compute_neighbor_overlap(self, neighbors_a: set[str], neighbors_b: set[str]) -> float:
        """计算两个邻居集合的 Jaccard 相似度

        Args:
            neighbors_a: 集合 A
            neighbors_b: 集合 B

        Returns:
            Jaccard 相似度 [0, 1]
        """
        if not neighbors_a and not neighbors_b:
            return 0.0
        intersection = neighbors_a & neighbors_b
        union = neighbors_a | neighbors_b
        return len(intersection) / len(union) if union else 0.0

    def _get_status_by_id(self, candidate_id: int) -> str | None:
        """根据 candidate_id 获取候选状态

        Args:
            candidate_id: 候选 ID

        Returns:
            状态字符串，不存在返回 None
        """
        row = self._graph.conn.execute(
            "SELECT status FROM distillation_pool WHERE candidate_id=?",
            (candidate_id,)
        ).fetchone()
        return row['status'] if row else None

    def _upgrade_to_global(
        self, candidate_id: int, canonical_source: str, canonical_target: str,
        canonical_relation: str,
    ) -> None:
        """将候选升级为全局业力

        1. UPSERT 全局业力边（初始权重 DISTILLATION_INITIAL_WEIGHT）
        2. 若全局业力边已存在，取 max(现有权重, DISTILLATION_INITIAL_WEIGHT) 不降低
        3. 更新候选状态为 upgraded，记录 upgraded_at 时间
        4. 原子操作：升级成功或完全回滚

        Args:
            candidate_id: 候选 ID
            canonical_source: 归一化后的源节点
            canonical_target: 归一化后的目标节点
            canonical_relation: 归一化后的关系
        """
        from consciousness_sea.infrastructure.config import DISTILLATION_INITIAL_WEIGHT

        now = datetime.now(timezone.utc).isoformat()

        try:
            # 1. UPSERT 全局业力边
            # 若已存在，取 max(现有权重, 初始权重)，不降低
            self._graph.conn.execute(
                "INSERT INTO karma_edges (source, target, relation, weight, source_tag) "
                "VALUES (?, ?, ?, ?, 'distillation_upgrade') "
                "ON CONFLICT (source, target, relation) DO UPDATE "
                "SET weight = MAX(weight, ?)",
                (canonical_source, canonical_target, canonical_relation,
                 DISTILLATION_INITIAL_WEIGHT, DISTILLATION_INITIAL_WEIGHT)
            )

            # 2. 更新候选状态
            self._graph.conn.execute(
                "UPDATE distillation_pool SET status='upgraded', upgraded_at=?, updated_at=? "
                "WHERE candidate_id=?",
                (now, now, candidate_id)
            )

            log.info(
                "提炼池候选升级为全局业力: %s → %s (%s), candidate_id=%d",
                canonical_source, canonical_target, canonical_relation, candidate_id
            )

        except Exception as e:
            log.warning("提炼池升级失败: %s", e)
            # 不自行 rollback，由调用方管理事务边界
            raise

    def get_status(self) -> dict[str, int]:
        """查询提炼池状态

        Returns:
            {
                'total_candidates': int,
                'upgraded_count': int,
                'pending_count': int,
                'cooled_count': int,
            }
        """
        total = self._graph.conn.execute(
            "SELECT COUNT(*) FROM distillation_pool"
        ).fetchone()[0]

        upgraded = self._graph.conn.execute(
            "SELECT COUNT(*) FROM distillation_pool WHERE status='upgraded'"
        ).fetchone()[0]

        pending = self._graph.conn.execute(
            "SELECT COUNT(*) FROM distillation_pool WHERE status='pending'"
        ).fetchone()[0]

        cooled = self._graph.conn.execute(
            "SELECT COUNT(*) FROM distillation_pool WHERE status='cooled'"
        ).fetchone()[0]

        return {
            'total_candidates': total,
            'upgraded_count': upgraded,
            'pending_count': pending,
            'cooled_count': cooled,
        }

    def rebuild_from_personal_karma(self) -> int:
        """从个人业力表重建提炼池候选

        扫描 karma_edges_personal 表，为每条个人业力边提交候选。
        用于系统重启后重建提炼池。

        Returns:
            重建的候选数量
        """
        rows = self._graph.conn.execute(
            "SELECT DISTINCT user_label, source, target, relation "
            "FROM karma_edges_personal"
        ).fetchall()

        count = 0
        for r in rows:
            try:
                self.submit_candidate(
                    user_label=r['user_label'],
                    source=r['source'],
                    target=r['target'],
                    relation=r['relation'],
                )
                count += 1
            except Exception as e:
                log.warning("重建提炼池候选失败: %s", e)

        return count

    # ═══════════════════════════════════════════════════════════
    #  Phase 5: 好奇心引擎集成方法
    # ═══════════════════════════════════════════════════════════

    def try_upgrade_by_domain(self, domain: str) -> int:
        """尝试升级指定领域相关的提炼池候选

        Phase 5: 好奇心引擎候选升级策略使用。
        查找与指定领域相关的 pending 候选，尝试触发升级。

        Args:
            domain: 知识领域

        Returns:
            成功升级的候选数量
        """
        upgraded = 0
        try:
            # 通过 seeds 表关联查找（精确匹配 domain）
            rows = self._graph.conn.execute(
                "SELECT dp.candidate_id, dp.canonical_source, dp.canonical_target, dp.canonical_relation "
                "FROM distillation_pool dp "
                "LEFT JOIN seeds s1 ON dp.canonical_source = s1.label "
                "LEFT JOIN seeds s2 ON dp.canonical_target = s2.label "
                "WHERE dp.status = 'pending' AND "
                "(s1.domain = ? OR s2.domain = ?)",
                (domain, domain),
            ).fetchall()

            # 如果精确匹配无结果，回退到 LIKE 匹配（转义通配符）
            if not rows:
                escaped = domain.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")
                rows = self._graph.conn.execute(
                    "SELECT candidate_id, canonical_source, canonical_target, canonical_relation "
                    "FROM distillation_pool "
                    "WHERE status = 'pending' AND "
                    "(canonical_source LIKE ? ESCAPE '\\' OR canonical_target LIKE ? ESCAPE '\\')",
                    (f"%{escaped}%", f"%{escaped}%"),
                ).fetchall()

            for r in rows:
                try:
                    self.upgrade_candidate(
                        r["candidate_id"],
                        r["canonical_source"],
                        r["canonical_target"],
                        r["canonical_relation"],
                    )
                    upgraded += 1
                except Exception as e:
                    log.debug("候选升级跳过: %s, candidate_id=%s", e, r["candidate_id"])
        except Exception as e:
            log.warning("按领域升级候选失败: %s, domain=%s", e, domain)

        return upgraded

    def submit_external_candidate(
        self,
        label: str,
        domain: str,
        summary: str = "",
    ) -> int:
        """提交外部查询结果到提炼池

        Phase 5: 外部知识源查询结果写入提炼池。
        如果已有相同 label 的候选，则递增 count。

        Args:
            label: 条目标题
            domain: 所属领域
            summary: 摘要

        Returns:
            candidate_id（新建或已有）
        """
        from consciousness_sea.infrastructure.config import DISTILLATION_THRESHOLD
        now = datetime.now(timezone.utc).isoformat()

        try:
            # 检查是否已有相同 label 的候选
            existing = self._graph.conn.execute(
                "SELECT candidate_id, count FROM distillation_pool "
                "WHERE canonical_source = ? AND canonical_target = ? AND canonical_relation = 'DEFINED_AS'",
                (label, domain),
            ).fetchone()

            if existing:
                # 递增 count
                new_count = existing["count"] + 1
                self._graph.conn.execute(
                    "UPDATE distillation_pool SET count = ?, updated_at = ? "
                    "WHERE candidate_id = ?",
                    (new_count, now, existing["candidate_id"]),
                )
                return existing["candidate_id"]

            # 新建候选
            representative_label = f"{label} → {domain}"
            self._graph.conn.execute(
                "INSERT INTO distillation_pool "
                "(canonical_source, canonical_target, canonical_relation, "
                " representative_label, count, contributor_users, status, created_at, updated_at) "
                "VALUES (?, ?, 'DEFINED_AS', ?, 1, '[\"system:curiosity\"]', 'pending', ?, ?)",
                (label, domain, representative_label, now, now),
            )

            # 检查是否达到升级阈值
            if 1 >= DISTILLATION_THRESHOLD:
                candidate_id = self._graph.conn.execute(
                    "SELECT last_insert_rowid()"
                ).fetchone()[0]
                try:
                    self.upgrade_candidate(candidate_id, label, domain, "DEFINED_AS")
                except Exception:
                    pass

            return self._graph.conn.execute("SELECT last_insert_rowid()").fetchone()[0]

        except Exception as e:
            log.warning("外部候选提交失败: %s, label=%s", e, label)
            return -1