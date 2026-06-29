"""
MetaSeedManager — 元种子管理器

元种子是识海"看见自己"的基础设施。它们不参与常规涟漪传播，
而是持续监控全局状态，并通过同一套熏习机制自然建立元认知关联。

职责:
  - 从现有知识图谱自动生成元种子（领域/关系/系统/自边界/未知领域）
  - 查询元种子列表和详情
  - 更新元种子指标
  - 根据指标变化触发元业力边创建
  - 元种子休眠/退役判定

线程安全:
  - _metrics_lock 保护指标更新
  - 数据库操作由 SQLite WAL 模式保证并发安全
"""

from __future__ import annotations

import json
import logging
import threading
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Optional

from consciousness_sea.domain.graph_db import GraphDB
from consciousness_sea.infrastructure.config import (
    META_SEED_ENABLED,
    META_KARMA_INITIAL_WEIGHT,
    META_KARMA_DELTA_THRESHOLD,
    META_SEED_DORMANT_CYCLES,
    META_EXPLORE_WINDOW,
    META_EXPLORE_LOW_CONF_THRESHOLD,
    GUARDIAN_METRICS_WINDOW,
    META_ALERT_CONFLICT_THRESHOLD,
    CONFIDENCE_LOW,
)

log = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════
#  枚举与数据类
# ═══════════════════════════════════════════════════════════


class MetaSeedCategory(str, Enum):
    """元种子类别枚举"""
    DOMAIN_MONITOR = "domain_monitor"            # 领域监控
    SYSTEM_MONITOR = "system_monitor"            # 全局监控
    SELF_BOUNDARY = "self_boundary"              # 自我边界
    PERFORMANCE_MONITOR = "performance_monitor"  # 性能监控（未知领域探测）
    RELATION_QUALITY = "relation_quality"        # 关系质量


class MetaSeedStatus(str, Enum):
    """元种子状态枚举"""
    ACTIVE = "active"      # 活跃监控
    DORMANT = "dormant"    # 休眠（指标长期无变化）
    RETIRED = "retired"    # 退役（不再更新）


@dataclass
class MetaSeedData:
    """元种子数据类"""
    label: str
    category: MetaSeedCategory
    metrics: dict = field(default_factory=dict)
    status: MetaSeedStatus = MetaSeedStatus.ACTIVE
    source_domain: str | None = None
    dormant_since: str | None = None
    created_at: str = ""
    updated_at: str = ""

    # 指标变化追踪（守护循环内部使用，不持久化）
    _previous_metrics: dict = field(default_factory=dict, repr=False)
    _unchanged_cycles: int = field(default=0, repr=False)


# ═══════════════════════════════════════════════════════════
#  各类别默认指标
# ═══════════════════════════════════════════════════════════

DOMAIN_MONITOR_DEFAULT_METRICS: dict = {
    "avg_karma_density": 0.0,
    "ripple_success_rate": 0.0,
    "conflict_frequency": 0,
}

RELATION_QUALITY_DEFAULT_METRICS: dict = {
    "avg_weight": 0.0,
    "verification_count": 0,
}

SYSTEM_MONITOR_DEFAULT_METRICS: dict = {
    "value": 0,
    "delta_24h": 0,
}

SELF_BOUNDARY_DEFAULT_METRICS: dict = {
    "unmatched_keywords": [],
    "unmatched_count": 0,
    "top_unmatched": [],
}

PERFORMANCE_MONITOR_DEFAULT_METRICS: dict = {
    "low_confidence_rate": 0.0,
    "query_count": 0,
}

# ═══════════════════════════════════════════════════════════
#  系统级元种子固定列表
# ═══════════════════════════════════════════════════════════

SYSTEM_META_SEEDS: list[tuple[str, str]] = [
    ("meta:system_total_nodes", "总节点数"),
    ("meta:system_total_edges", "总边数"),
    ("meta:system_avg_confidence", "平均置信度"),
    ("meta:system_distillation_rate", "熏习速率"),
    ("meta:system_seed_growth_rate", "种子增长率"),
]


# ═══════════════════════════════════════════════════════════
#  MetaSeedManager
# ═══════════════════════════════════════════════════════════


class MetaSeedManager:
    """元种子管理器 — 生成/查询/指标更新/元业力边创建

    职责:
      - 从现有知识图谱自动生成元种子（领域/关系/系统/自边界/未知领域）
      - 查询元种子列表和详情
      - 更新元种子指标
      - 根据指标变化触发元业力边创建
      - 元种子休眠/退役判定

    线程安全:
      - _metrics_lock 保护指标更新
      - 数据库操作由 SQLite WAL 模式保证并发安全

    Args:
        graph: 知识图谱连接
    """

    def __init__(self, graph: GraphDB) -> None:
        self._graph = graph
        self._metrics_lock = threading.Lock()

    # ── 元种子生成 ──────────────────────────────────────

    def generate_domain_monitors(self) -> int:
        """从现有领域标签自动生成领域监控元种子

        流程:
          1. 查询 seeds 表中不同 domain 值（排除 domain='元认知'）
          2. 对每个领域标签，检查 meta_seeds 表是否已存在
          3. 不存在则创建: seeds 表 + meta_seeds 表

        Returns:
            新创建的元种子数量
        """
        if not META_SEED_ENABLED:
            return 0

        try:
            rows = self._graph.conn.execute(
                "SELECT DISTINCT domain FROM seeds WHERE domain IS NOT NULL AND domain != '元认知'"
            ).fetchall()
        except Exception as e:
            log.warning("领域标签查询失败: %s", e)
            return 0

        created = 0
        for r in rows:
            domain = r["domain"]
            if not domain:
                continue
            label = f"meta:{domain}"
            if self._create_meta_seed_record(
                label, MetaSeedCategory.DOMAIN_MONITOR,
                dict(DOMAIN_MONITOR_DEFAULT_METRICS),
                source_domain=domain,
            ):
                created += 1

        if created > 0:
            log.info("领域监控元种子生成: %d 个新创建", created)
        return created

    def generate_relation_monitors(self) -> int:
        """从现有关系类型自动生成关系质量元种子

        流程:
          1. 查询 karma_edges 表中不同 relation 值
          2. 对每个关系类型，检查 meta_seeds 表是否已存在
          3. 不存在则创建: seeds 表 + meta_seeds 表

        Returns:
            新创建的元种子数量
        """
        if not META_SEED_ENABLED:
            return 0

        try:
            rows = self._graph.conn.execute(
                "SELECT DISTINCT relation FROM karma_edges"
            ).fetchall()
        except Exception as e:
            log.warning("关系类型查询失败: %s", e)
            return 0

        created = 0
        for r in rows:
            relation = r["relation"]
            if not relation:
                continue
            label = f"meta:{relation}"
            if self._create_meta_seed_record(
                label, MetaSeedCategory.RELATION_QUALITY,
                dict(RELATION_QUALITY_DEFAULT_METRICS),
                source_domain=relation,
            ):
                created += 1

        if created > 0:
            log.info("关系质量元种子生成: %d 个新创建", created)
        return created

    def generate_system_monitors(self) -> int:
        """生成固定 5 个系统级元种子

        Returns:
            新创建的元种子数量
        """
        if not META_SEED_ENABLED:
            return 0

        created = 0
        for label, desc in SYSTEM_META_SEEDS:
            if self._create_meta_seed_record(
                label, MetaSeedCategory.SYSTEM_MONITOR,
                dict(SYSTEM_MONITOR_DEFAULT_METRICS),
                source_domain=label.replace("meta:", ""),
            ):
                # 设置 definition
                try:
                    self._graph.conn.execute(
                        "UPDATE seeds SET definition = ? WHERE label = ?",
                        (desc, label),
                    )
                except Exception:
                    pass
                created += 1

        if created > 0:
            log.info("系统级元种子生成: %d 个新创建", created)
        return created

    def update_self_boundary(self) -> int:
        """更新自边界元种子（meta:unknown）的指标

        从 candidate_seeds 表中提取状态为 candidate 的记录，
        更新 metrics.unmatched_keywords、unmatched_count、top_unmatched。

        Returns:
            更新的元种子数量（0 或 1）
        """
        if not META_SEED_ENABLED:
            return 0

        try:
            rows = self._graph.conn.execute(
                "SELECT label, count FROM candidate_seeds WHERE status = 'candidate'"
            ).fetchall()
        except Exception as e:
            log.warning("候选种子查询失败: %s", e)
            return 0

        unmatched_keywords = [r["label"] for r in rows]
        unmatched_count = len(rows)
        # 按 count 降序取前 10 个
        sorted_rows = sorted(rows, key=lambda r: r["count"], reverse=True)
        top_unmatched = [r["label"] for r in sorted_rows[:10]]

        metrics = {
            "unmatched_keywords": unmatched_keywords,
            "unmatched_count": unmatched_count,
            "top_unmatched": top_unmatched,
        }

        if self.update_metrics("meta:unknown", metrics):
            return 1
        return 0

    def detect_unknown_domains(self) -> int:
        """探测低置信度高频区域，自动生成未知领域探测元种子

        从 param_stats 表中计算各领域最近 META_EXPLORE_WINDOW 次查询中
        低置信度频率，超过阈值时创建或更新探测元种子。

        Returns:
            新创建或更新的元种子数量
        """
        if not META_SEED_ENABLED:
            return 0

        try:
            rows = self._graph.conn.execute(
                "SELECT selected_domains, confidence FROM param_stats "
                "ORDER BY created_at DESC LIMIT ?",
                (META_EXPLORE_WINDOW,),
            ).fetchall()
        except Exception as e:
            log.warning("param_stats 查询失败: %s", e)
            return 0

        if not rows:
            return 0

        domain_low_conf: dict[str, int] = {}
        domain_total: dict[str, int] = {}

        for r in rows:
            try:
                domains = json.loads(r["selected_domains"]) if r["selected_domains"] else []
            except (json.JSONDecodeError, TypeError):
                continue
            confidence = r["confidence"]
            for domain in domains:
                domain_total[domain] = domain_total.get(domain, 0) + 1
                if confidence < CONFIDENCE_LOW:
                    domain_low_conf[domain] = domain_low_conf.get(domain, 0) + 1

        updated_or_created = 0
        for domain, total in domain_total.items():
            low_conf_count = domain_low_conf.get(domain, 0)
            rate = low_conf_count / total if total > 0 else 0.0
            if rate > META_EXPLORE_LOW_CONF_THRESHOLD:
                label = f"meta:explore_{domain}"
                metrics = {
                    "low_confidence_rate": round(rate, 4),
                    "query_count": total,
                }
                # 检查是否已存在
                existing = self.get_meta_seed(label)
                if existing is not None:
                    # 已存在，仅更新指标
                    if self.update_metrics(label, metrics):
                        updated_or_created += 1
                else:
                    # 不存在，创建
                    if self._create_meta_seed_record(
                        label, MetaSeedCategory.PERFORMANCE_MONITOR,
                        metrics,
                        source_domain=domain,
                    ):
                        updated_or_created += 1

        if updated_or_created > 0:
            log.info("未知领域探测: %d 个元种子创建或更新", updated_or_created)
        return updated_or_created

    # ── 元种子查询 ──────────────────────────────────────

    def get_meta_seed(self, label: str) -> MetaSeedData | None:
        """查询单个元种子

        Args:
            label: 元种子 label（如 "meta:物理"）

        Returns:
            MetaSeedData 或 None
        """
        if not META_SEED_ENABLED:
            return None

        try:
            row = self._graph.conn.execute(
                "SELECT * FROM meta_seeds WHERE label = ?", (label,)
            ).fetchone()
        except Exception as e:
            log.warning("元种子查询失败: %s", e)
            return None

        if not row:
            return None

        try:
            metrics = json.loads(row["metrics_json"])
        except (json.JSONDecodeError, TypeError):
            log.warning("元种子指标 JSON 格式异常: %s, 重置为空对象", label)
            metrics = {}

        # 读取持久化的追踪字段
        previous_metrics: dict = {}
        try:
            prev_json = row["previous_metrics_json"] if "previous_metrics_json" in row.keys() else "{}"
            if prev_json:
                previous_metrics = json.loads(prev_json)
        except (json.JSONDecodeError, TypeError):
            previous_metrics = {}

        unchanged_cycles = row["unchanged_cycles"] if "unchanged_cycles" in row.keys() else 0

        return MetaSeedData(
            label=row["label"],
            category=MetaSeedCategory(row["category"]),
            metrics=metrics,
            status=MetaSeedStatus(row["status"]),
            source_domain=row["source_domain"],
            dormant_since=row["dormant_since"],
            created_at=row["created_at"],
            updated_at=row["updated_at"],
            _previous_metrics=previous_metrics,
            _unchanged_cycles=unchanged_cycles,
        )

    def list_meta_seeds(
        self,
        category: MetaSeedCategory | None = None,
        status: MetaSeedStatus | None = None,
    ) -> list[MetaSeedData]:
        """查询元种子列表

        Args:
            category: 按类别过滤（可选）
            status: 按状态过滤（可选）

        Returns:
            MetaSeedData 列表
        """
        if not META_SEED_ENABLED:
            return []

        conditions: list[str] = []
        params: list = []

        if category is not None:
            conditions.append("category = ?")
            params.append(category.value)
        if status is not None:
            conditions.append("status = ?")
            params.append(status.value)

        where_clause = ""
        if conditions:
            where_clause = "WHERE " + " AND ".join(conditions)

        try:
            rows = self._graph.conn.execute(
                f"SELECT * FROM meta_seeds {where_clause} ORDER BY label",
                params,
            ).fetchall()
        except Exception as e:
            log.warning("元种子列表查询失败: %s", e)
            return []

        result: list[MetaSeedData] = []
        for row in rows:
            try:
                metrics = json.loads(row["metrics_json"])
            except (json.JSONDecodeError, TypeError):
                metrics = {}

            # 读取持久化的追踪字段
            previous_metrics: dict = {}
            try:
                prev_json = row["previous_metrics_json"] if "previous_metrics_json" in row.keys() else "{}"
                if prev_json:
                    previous_metrics = json.loads(prev_json)
            except (json.JSONDecodeError, TypeError):
                previous_metrics = {}

            unchanged_cycles = row["unchanged_cycles"] if "unchanged_cycles" in row.keys() else 0

            result.append(MetaSeedData(
                label=row["label"],
                category=MetaSeedCategory(row["category"]),
                metrics=metrics,
                status=MetaSeedStatus(row["status"]),
                source_domain=row["source_domain"],
                dormant_since=row["dormant_since"],
                created_at=row["created_at"],
                updated_at=row["updated_at"],
                _previous_metrics=previous_metrics,
                _unchanged_cycles=unchanged_cycles,
            ))

        return result

    # ── 元种子指标更新 ──────────────────────────────────

    def update_metrics(self, label: str, metrics: dict) -> bool:
        """更新元种子指标

        原子操作：指标要么全部更新成功，要么全部不更新。
        更新前保存 _previous_metrics 用于元业力熏习判定。

        Args:
            label: 元种子 label
            metrics: 新的指标字典

        Returns:
            True 更新成功, False 失败
        """
        if not META_SEED_ENABLED:
            return False

        with self._metrics_lock:
            # 读取当前指标作为 previous_metrics
            existing = self.get_meta_seed(label)
            if existing is None:
                return False

            now = datetime.now(timezone.utc).isoformat()
            metrics_json = json.dumps(metrics, ensure_ascii=False)

            try:
                self._graph.conn.execute(
                    "UPDATE meta_seeds SET metrics_json = ?, updated_at = ? WHERE label = ?",
                    (metrics_json, now, label),
                )
            except Exception as e:
                log.warning("元种子指标更新失败: %s, label=%s", e, label)
                return False

            return True

    def increment_metric(self, label: str, key: str, delta: int | float = 1) -> bool:
        """递增元种子的某个指标

        用于校验器集成时快速递增 conflict_frequency 等指标。

        Args:
            label: 元种子 label
            key: 指标键名
            delta: 递增量

        Returns:
            True 成功, False 失败
        """
        if not META_SEED_ENABLED:
            return False

        with self._metrics_lock:
            try:
                # 读取当前指标
                row = self._graph.conn.execute(
                    "SELECT metrics_json FROM meta_seeds WHERE label = ?", (label,)
                ).fetchone()
                if not row:
                    return False

                try:
                    metrics = json.loads(row["metrics_json"])
                except (json.JSONDecodeError, TypeError):
                    metrics = {}

                # 递增指定指标
                current = metrics.get(key, 0)
                if isinstance(current, (int, float)) and isinstance(delta, (int, float)):
                    metrics[key] = current + delta
                else:
                    metrics[key] = delta

                now = datetime.now(timezone.utc).isoformat()
                metrics_json = json.dumps(metrics, ensure_ascii=False)

                self._graph.conn.execute(
                    "UPDATE meta_seeds SET metrics_json = ?, updated_at = ? WHERE label = ?",
                    (metrics_json, now, label),
                )
                return True

            except Exception as e:
                log.warning("元种子指标递增失败: %s, label=%s, key=%s", e, label, key)
                return False

    # ── 元业力边 ──────────────────────────────────────

    def check_and_create_meta_karma(self) -> int:
        """检查所有元种子指标变化，触发元业力熏习

        对每个 active 状态的元种子:
          1. 比较 current_metrics 和 _previous_metrics
          2. 若指标变化量 > META_KARMA_DELTA_THRESHOLD:
             查找相关元种子（同领域、同类型）
             调用 graph.adjust_karma_atomic() 创建/增强元业力边
          3. 正向变化（指标恶化）→ delta=+META_KARMA_INITIAL_WEIGHT
             负向变化（指标改善）→ delta=-META_KARMA_INITIAL_WEIGHT

        Returns:
            创建或更新的元业力边数量
        """
        if not META_SEED_ENABLED:
            return 0

        meta_seeds = self.list_meta_seeds(status=MetaSeedStatus.ACTIVE)
        if not meta_seeds:
            return 0

        edges_created = 0

        for ms in meta_seeds:
            # 读取当前指标和前一次指标
            current = self.get_meta_seed(ms.label)
            if current is None:
                continue

            current_metrics = current.metrics
            # previous_metrics 从数据库读取时为空，需要从内存追踪
            # 在守护循环中，update_metrics 之前会先读取 previous
            previous_metrics = ms._previous_metrics if ms._previous_metrics else {}

            if not previous_metrics:
                # 首次运行，保存当前指标作为基线
                ms._previous_metrics = dict(current_metrics)
                self._persist_tracking_fields(ms)
                continue

            has_significant_change = False
            for key, current_value in current_metrics.items():
                if not isinstance(current_value, (int, float)):
                    continue  # 跳过非数值指标

                previous_value = previous_metrics.get(key, 0)
                if not isinstance(previous_value, (int, float)):
                    previous_value = 0

                delta = abs(current_value - previous_value)
                if delta < META_KARMA_DELTA_THRESHOLD:
                    continue

                has_significant_change = True

                # 指标变化显著 → 触发元业力熏习
                direction = +1 if current_value > previous_value else -1
                karma_delta = META_KARMA_INITIAL_WEIGHT * direction

                # 查找相关元种子（同类别）
                related_seeds = [
                    s for s in meta_seeds
                    if s.label != ms.label and s.category == ms.category
                ]

                for related in related_seeds:
                    try:
                        self._graph.conn.execute(
                            "INSERT INTO karma_edges (source, target, relation, weight, source_tag) "
                            "VALUES (?, ?, ?, ?, 'meta_karma') "
                            "ON CONFLICT (source, target, relation) DO UPDATE "
                            "SET weight = MAX(?, MIN(?, weight + ?))",
                            (ms.label, related.label, "META_CORRELATED",
                             max(0.01, min(2.0, 0.5 + karma_delta)),
                             0.01, 2.0, karma_delta),
                        )
                        edges_created += 1
                    except Exception as e:
                        log.warning("元业力边创建失败: %s → %s: %s", ms.label, related.label, e)

            # 更新 previous_metrics 和 unchanged_cycles
            if has_significant_change:
                ms._previous_metrics = dict(current_metrics)
                ms._unchanged_cycles = 0
            else:
                ms._unchanged_cycles += 1
            self._persist_tracking_fields(ms)

        if edges_created > 0:
            log.info("元业力边创建: %d 条", edges_created)
        return edges_created

    # ── 休眠/退役判定 ──────────────────────────────────

    def check_dormant_status(self) -> int:
        """检查元种子休眠/退役状态

        连续 META_SEED_DORMANT_CYCLES 个周期指标无变化 → dormant
        dormant 状态超过 3 × DORMANT_CYCLES → retired

        Returns:
            状态变更的元种子数量
        """
        if not META_SEED_ENABLED:
            return 0

        meta_seeds = self.list_meta_seeds()
        if not meta_seeds:
            return 0

        changed = 0
        now = datetime.now(timezone.utc).isoformat()

        for ms in meta_seeds:
            if ms.status == MetaSeedStatus.RETIRED:
                continue  # 退役不再变更

            # 检查指标是否有变化
            current = self.get_meta_seed(ms.label)
            if current is None:
                continue

            current_metrics = current.metrics
            previous = ms._previous_metrics if ms._previous_metrics else {}

            metrics_changed = current_metrics != previous

            if ms.status == MetaSeedStatus.ACTIVE:
                if not metrics_changed:
                    ms._unchanged_cycles += 1
                else:
                    ms._unchanged_cycles = 0
                    ms._previous_metrics = dict(current_metrics)

                if ms._unchanged_cycles >= META_SEED_DORMANT_CYCLES:
                    # active → dormant
                    try:
                        self._graph.conn.execute(
                            "UPDATE meta_seeds SET status = ?, dormant_since = ?, updated_at = ? WHERE label = ?",
                            (MetaSeedStatus.DORMANT.value, now, now, ms.label),
                        )
                        ms.status = MetaSeedStatus.DORMANT
                        ms.dormant_since = now
                        changed += 1
                        log.info("元种子休眠: %s (连续 %d 周期无变化)", ms.label, ms._unchanged_cycles)
                    except Exception as e:
                        log.warning("元种子状态更新失败: %s", e)

                self._persist_tracking_fields(ms)

            elif ms.status == MetaSeedStatus.DORMANT:
                if metrics_changed:
                    # dormant → active（指标变化恢复活跃）
                    try:
                        self._graph.conn.execute(
                            "UPDATE meta_seeds SET status = ?, dormant_since = NULL, updated_at = ? WHERE label = ?",
                            (MetaSeedStatus.ACTIVE.value, now, ms.label),
                        )
                        ms.status = MetaSeedStatus.ACTIVE
                        ms.dormant_since = None
                        ms._unchanged_cycles = 0
                        ms._previous_metrics = dict(current_metrics)
                        changed += 1
                        log.info("元种子恢复活跃: %s", ms.label)
                    except Exception as e:
                        log.warning("元种子状态更新失败: %s", e)
                else:
                    ms._unchanged_cycles += 1
                    # dormant 超过 3 × DORMANT_CYCLES → retired
                    if ms._unchanged_cycles >= META_SEED_DORMANT_CYCLES * 3:
                        try:
                            self._graph.conn.execute(
                                "UPDATE meta_seeds SET status = ?, updated_at = ? WHERE label = ?",
                                (MetaSeedStatus.RETIRED.value, now, ms.label),
                            )
                            ms.status = MetaSeedStatus.RETIRED
                            changed += 1
                            log.info("元种子退役: %s (连续 %d 周期无变化)", ms.label, ms._unchanged_cycles)
                        except Exception as e:
                            log.warning("元种子状态更新失败: %s", e)

                self._persist_tracking_fields(ms)

        return changed

    # ── 内部方法 ──────────────────────────────────────

    def _persist_tracking_fields(self, ms: MetaSeedData) -> None:
        """将 _previous_metrics 和 _unchanged_cycles 持久化到数据库

        Args:
            ms: 元种子数据对象
        """
        try:
            prev_json = json.dumps(ms._previous_metrics, ensure_ascii=False)
            self._graph.conn.execute(
                "UPDATE meta_seeds SET unchanged_cycles = ?, previous_metrics_json = ?, updated_at = ? WHERE label = ?",
                (ms._unchanged_cycles, prev_json, datetime.now(timezone.utc).isoformat(), ms.label),
            )
        except Exception as e:
            log.warning("元种子追踪字段持久化失败: %s, label=%s", e, ms.label)

    def _create_meta_seed_record(
        self,
        label: str,
        category: MetaSeedCategory,
        metrics: dict,
        source_domain: str | None = None,
    ) -> bool:
        """创建元种子记录（seeds 表 + meta_seeds 表，原子操作）

        流程:
          1. 检查 label 前缀（不以 "meta:" 开头则自动添加）
          2. 检查 category 合法性
          3. 检查 meta_seeds 表是否已存在
          4. INSERT INTO seeds (label, type='META', domain='元认知', activation=0.0, aliases='[]')
          5. INSERT INTO meta_seeds (label, category, metrics_json, status='active', ...)
          6. 若步骤 5 失败 → 回滚步骤 4（DELETE FROM seeds）

        Args:
            label: 元种子 label
            category: 类别
            metrics: 初始指标
            source_domain: 来源领域/关系/系统标识

        Returns:
            True 创建成功, False 已存在或创建失败
        """
        # 确保 label 前缀
        label = self._ensure_meta_prefix(label)

        # 检查 category 合法性
        if not isinstance(category, MetaSeedCategory):
            log.warning("元种子类别不合法: %s", category)
            return False

        # 检查是否已存在
        try:
            existing = self._graph.conn.execute(
                "SELECT 1 FROM meta_seeds WHERE label = ?", (label,)
            ).fetchone()
            if existing:
                log.debug("元种子已存在，跳过: %s", label)
                return False
        except Exception as e:
            log.warning("元种子存在性检查失败: %s", e)
            return False

        now = datetime.now(timezone.utc).isoformat()
        metrics_json = json.dumps(metrics, ensure_ascii=False)

        # 写入 seeds 表
        try:
            self._graph.conn.execute(
                "INSERT OR IGNORE INTO seeds (label, type, domain, activation, aliases) "
                "VALUES (?, 'META', '元认知', 0.0, '[]')",
                (label,),
            )
        except Exception as e:
            log.error("元种子 seeds 表写入失败: %s, label=%s", e, label)
            return False

        # 写入 meta_seeds 表
        try:
            self._graph.conn.execute(
                "INSERT INTO meta_seeds (label, category, metrics_json, status, source_domain, "
                "unchanged_cycles, previous_metrics_json, created_at, updated_at) "
                "VALUES (?, ?, ?, 'active', ?, 0, '{}', ?, ?)",
                (label, category.value, metrics_json, source_domain, now, now),
            )
        except Exception as e:
            log.error("元种子 meta_seeds 表写入失败: %s, label=%s, 回滚 seeds 表记录", e, label)
            # 回滚 seeds 表记录
            try:
                self._graph.conn.execute(
                    "DELETE FROM seeds WHERE label = ? AND type = 'META'", (label,)
                )
            except Exception:
                pass
            return False

        log.debug("元种子创建成功: %s (category=%s)", label, category.value)
        return True

    def _ensure_meta_prefix(self, label: str) -> str:
        """确保 label 以 "meta:" 前缀开头"""
        if not label.startswith("meta:"):
            return f"meta:{label}"
        return label

    def _is_meta_seed_domain(self, domain: str) -> bool:
        """判断是否为元种子的领域（避免元种子监控元种子）"""
        return domain == "元认知"