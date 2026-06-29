"""
CognitiveGoalManager — 认知目标管理器

基于元种子指标自动生成认知目标，驱动好奇心引擎探索薄弱区域。

职责:
  - 认知目标生成（四种触发条件）
  - 目标去重（同 domain + goal_type）
  - 优先级权重计算（四因子加权）
  - 目标冷却（权重衰减/过期/归档/池淘汰）
  - 认知目标查询和统计

线程安全:
  - _goal_lock 保护目标创建和更新
  - 数据库操作由 SQLite WAL 模式保证并发安全
"""

from __future__ import annotations

import hashlib
import json
import logging
import threading
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Optional

from consciousness_sea.domain.graph_db import GraphDB
from .meta_seed import MetaSeedManager, MetaSeedCategory, MetaSeedData
from consciousness_sea.infrastructure.config import (
    COGNITIVE_GOAL_ENABLED,
    GOAL_LOW_CONF_THRESHOLD,
    GOAL_LOW_CONF_WINDOW,
    GOAL_LOW_DENSITY_RATIO,
    GOAL_HIGH_CONFLICT_THRESHOLD,
    GOAL_NEW_TERM_THRESHOLD,
    GOAL_NEW_TERM_WINDOW,
    GOAL_AUTO_EXPLORE_THRESHOLD,
    GOAL_MAX_EXPLORE_PER_CYCLE,
    GOAL_DECAY_CYCLES,
    GOAL_DECAY_FACTOR,
    GOAL_EXPIRE_THRESHOLD,
    GOAL_USER_ABSENCE_CYCLES,
    GOAL_POOL_MAX_SIZE,
    GOAL_DECOMPOSABILITY_NORM,
    GOAL_WEIGHT_USER_RELEVANCE,
    GOAL_WEIGHT_SYSTEM_CORENESS,
    GOAL_WEIGHT_UNCERTAINTY,
    GOAL_WEIGHT_DECOMPOSABILITY,
    GUARDIAN_LOOP_INTERVAL,
)

log = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════
#  枚举与数据类
# ═══════════════════════════════════════════════════════════


class GoalType(str, Enum):
    """认知目标类型枚举"""
    LOW_CONFIDENCE = "low_confidence"    # 低置信度频率过高
    LOW_DENSITY = "low_density"          # 业力密度过低
    HIGH_CONFLICT = "high_conflict"      # 冲突频率过高
    NEW_TERM = "new_term"                # 用户高频触及新词


class GoalStatus(str, Enum):
    """认知目标状态枚举"""
    PENDING = "pending"                          # 待调度
    EXPLORING = "exploring"                      # 内部探索中
    QUERYING_EXTERNAL = "querying_external"      # 外部查询中
    COMPLETED = "completed"                      # 已完成
    ARCHIVED = "archived"                        # 已归档
    EXPIRED = "expired"                          # 已过期


@dataclass
class CognitiveGoalData:
    """认知目标数据类"""
    goal_id: str
    goal_type: GoalType
    trigger_condition: str
    domain: str
    priority_weight: float = 0.0
    status: GoalStatus = GoalStatus.PENDING
    sub_goals: list[str] = field(default_factory=list)
    execution_log: list[dict] = field(default_factory=list)
    associated_user: str | None = None
    decay_cycles_count: int = 0
    last_touched_at: str = ""
    created_at: str = ""
    updated_at: str = ""


# ═══════════════════════════════════════════════════════════
#  CognitiveGoalManager
# ═══════════════════════════════════════════════════════════


class CognitiveGoalManager:
    """认知目标管理器 — 生成/去重/优先级计算/冷却/查询/统计

    职责:
      - 基于元种子指标自动生成认知目标
      - 目标去重（同 domain + goal_type）
      - 优先级权重计算（四因子加权）
      - 目标冷却（权重衰减/过期/归档/池淘汰）
      - 认知目标查询和统计

    线程安全:
      - _goal_lock 保护目标创建和更新
      - 数据库操作由 SQLite WAL 模式保证并发安全

    Args:
        graph: 知识图谱连接
    """

    def __init__(self, graph: GraphDB) -> None:
        self._graph = graph
        self._meta_seed_mgr = MetaSeedManager(graph)
        self._goal_lock = threading.Lock()

    # ── 目标生成 ──────────────────────────────────────

    def generate_goals(self) -> int:
        """基于元种子指标自动生成认知目标

        流程:
          1. 读取所有领域监控元种子指标
          2. 检查四种触发条件
          3. 读取自边界元种子指标（新词触发）
          4. 去重与优先级计算

        Returns:
            新创建或更新的目标数量
        """
        if not COGNITIVE_GOAL_ENABLED:
            return 0

        created_or_updated = 0

        # A. 读取领域监控元种子
        try:
            domain_seeds = self._meta_seed_mgr.list_meta_seeds(
                category=MetaSeedCategory.DOMAIN_MONITOR
            )
        except Exception as e:
            log.warning("领域监控元种子查询失败: %s", e)
            return 0

        # 计算全局平均业力密度
        global_avg_density = self._compute_global_avg_density()

        for ms in domain_seeds:
            if ms.status.value != "active":
                continue

            metrics = ms.metrics
            domain = ms.source_domain or ms.label.replace("meta:", "")

            # B1. 低置信度频率触发
            conflict_freq = metrics.get("conflict_frequency", 0)
            if isinstance(conflict_freq, (int, float)) and conflict_freq > GOAL_LOW_CONF_THRESHOLD:
                if self._create_or_update_goal(
                    goal_type=GoalType.LOW_CONFIDENCE,
                    domain=domain,
                    trigger_condition=f"meta:{domain}.conflict_frequency={conflict_freq}",
                ):
                    created_or_updated += 1

            # B2. 业力密度过低触发
            avg_density = metrics.get("avg_karma_density", 0.0)
            if (isinstance(avg_density, (int, float))
                    and global_avg_density > 0
                    and avg_density < global_avg_density * GOAL_LOW_DENSITY_RATIO):
                if self._create_or_update_goal(
                    goal_type=GoalType.LOW_DENSITY,
                    domain=domain,
                    trigger_condition=(
                        f"meta:{domain}.avg_karma_density={avg_density}"
                        f"<global_avg×{GOAL_LOW_DENSITY_RATIO}"
                    ),
                ):
                    created_or_updated += 1

            # B3. 冲突频率过高触发
            if isinstance(conflict_freq, (int, float)) and conflict_freq > GOAL_HIGH_CONFLICT_THRESHOLD:
                if self._create_or_update_goal(
                    goal_type=GoalType.HIGH_CONFLICT,
                    domain=domain,
                    trigger_condition=f"meta:{domain}.conflict_frequency={conflict_freq}",
                ):
                    created_or_updated += 1

        # C. 新词触发
        try:
            boundary_seed = self._meta_seed_mgr.get_meta_seed("meta:unknown")
            if boundary_seed and boundary_seed.metrics.get("top_unmatched"):
                for keyword in boundary_seed.metrics["top_unmatched"]:
                    count = self._count_keyword_in_window(keyword)
                    if count > GOAL_NEW_TERM_THRESHOLD:
                        domain = self._infer_domain_for_keyword(keyword)
                        if self._create_or_update_goal(
                            goal_type=GoalType.NEW_TERM,
                            domain=domain,
                            trigger_condition=f"meta:unknown.top_unmatched.{keyword}.count={count}",
                            associated_user=self._find_active_user_for_keyword(keyword),
                        ):
                            created_or_updated += 1
        except Exception as e:
            log.warning("新词触发检查失败: %s", e)

        if created_or_updated > 0:
            log.info("认知目标生成: %d 个新创建或更新", created_or_updated)
        return created_or_updated

    # ── 目标去重与创建 ──────────────────────────────

    def _create_or_update_goal(
        self,
        goal_type: GoalType,
        domain: str,
        trigger_condition: str,
        associated_user: str | None = None,
    ) -> bool:
        """创建或更新认知目标（去重逻辑）

        去重策略:
          - 同 domain + goal_type 且 status 为 pending/exploring → 更新
          - 不同 goal_type → 允许共存

        Args:
            goal_type: 目标类型
            domain: 关联领域
            trigger_condition: 触发条件描述
            associated_user: 关联用户（可选）

        Returns:
            True 创建或更新成功, False 失败
        """
        with self._goal_lock:
            try:
                # 去重检查
                existing = self._graph.conn.execute(
                    "SELECT goal_id, priority_weight FROM cognitive_goals "
                    "WHERE domain = ? AND goal_type = ? AND status IN ('pending', 'exploring')",
                    (domain, goal_type.value),
                ).fetchone()

                now = datetime.now(timezone.utc).isoformat()

                if existing:
                    # 更新已有目标
                    priority = self._compute_priority_weight(domain, goal_type)
                    self._graph.conn.execute(
                        "UPDATE cognitive_goals SET priority_weight = ?, trigger_condition = ?, "
                        "updated_at = ?, last_touched_at = ? "
                        "WHERE goal_id = ?",
                        (priority, trigger_condition, now, now, existing["goal_id"]),
                    )
                    log.debug(
                        "认知目标更新: goal_id=%s, type=%s, domain=%s, priority=%.2f",
                        existing["goal_id"], goal_type.value, domain, priority,
                    )
                    return True

                # 检查池大小
                active_count = self._get_active_goal_count()
                if active_count >= GOAL_POOL_MAX_SIZE:
                    log.warning(
                        "认知目标池已满: active=%d, max=%d, 跳过创建 domain=%s type=%s",
                        active_count, GOAL_POOL_MAX_SIZE, domain, goal_type.value,
                    )
                    return False

                # 创建新目标
                goal_id = self._generate_goal_id(domain, goal_type)
                priority = self._compute_priority_weight(domain, goal_type)

                self._graph.conn.execute(
                    "INSERT INTO cognitive_goals "
                    "(goal_id, goal_type, trigger_condition, domain, priority_weight, "
                    " status, sub_goals, execution_log, associated_user, "
                    " decay_cycles_count, last_touched_at, created_at, updated_at) "
                    "VALUES (?, ?, ?, ?, ?, 'pending', '[]', '[]', ?, 0, ?, ?, ?)",
                    (goal_id, goal_type.value, trigger_condition, domain, priority,
                     associated_user, now, now, now),
                )

                log.info(
                    "认知目标生成: goal_id=%s, type=%s, domain=%s, priority=%.2f",
                    goal_id, goal_type.value, domain, priority,
                )
                return True

            except Exception as e:
                log.warning("认知目标创建/更新失败: %s, domain=%s type=%s", e, domain, goal_type.value)
                return False

    # ── 优先级权重计算 ──────────────────────────────

    def _compute_priority_weight(self, domain: str, goal_type: GoalType) -> float:
        """计算认知目标的优先级权重

        公式: 用户相关性 × W1 + 系统核心度 × W2 + 不确定性 × W3 + 可分解性 × W4

        Returns:
            优先级权重 [0, 1]，保留两位小数
        """
        user_relevance = self._compute_user_relevance(domain)
        system_coreness = self._compute_system_coreness(domain)
        uncertainty = self._compute_uncertainty(domain)
        decomposability = self._compute_decomposability(domain)

        weight = (
            user_relevance * GOAL_WEIGHT_USER_RELEVANCE
            + system_coreness * GOAL_WEIGHT_SYSTEM_CORENESS
            + uncertainty * GOAL_WEIGHT_UNCERTAINTY
            + decomposability * GOAL_WEIGHT_DECOMPOSABILITY
        )
        return round(max(0.0, min(1.0, weight)), 2)

    def _compute_user_relevance(self, domain: str) -> float:
        """计算用户相关性因子

        公式: 目标领域中与活跃用户有业力关联的种子数 / 领域总种子数
        缺失时默认 0.5
        """
        try:
            # 领域总种子数
            total_row = self._graph.conn.execute(
                "SELECT COUNT(*) as cnt FROM seeds WHERE domain = ? AND type != 'META'",
                (domain,),
            ).fetchone()
            total = total_row["cnt"] if total_row else 0

            if total == 0:
                return 0.5  # 默认值

            # 有个人业力关联的种子数
            user_row = self._graph.conn.execute(
                "SELECT COUNT(DISTINCT ke.source) as cnt "
                "FROM karma_edges_personal ke "
                "JOIN seeds s ON ke.source = s.label "
                "WHERE s.domain = ? AND s.type != 'META' "
                "AND ke.updated_at > datetime('now', '-7 days')",
                (domain,),
            ).fetchone()
            user_related = user_row["cnt"] if user_row else 0

            return round(min(1.0, user_related / total), 4)
        except Exception as e:
            log.warning("用户相关性计算失败: %s, domain=%s", e, domain)
            return 0.5

    def _compute_system_coreness(self, domain: str) -> float:
        """计算系统核心度因子

        公式: 1 - (目标领域到核心领域的最短路径长度 / 最大路径长度)
        核心领域 = 种子数和业力边数最多的领域
        缺失时默认 0.5
        """
        try:
            # 找到核心领域（种子数最多的领域）
            core_row = self._graph.conn.execute(
                "SELECT domain, COUNT(*) as cnt FROM seeds "
                "WHERE type != 'META' AND domain IS NOT NULL "
                "GROUP BY domain ORDER BY cnt DESC LIMIT 1"
            ).fetchone()
            if not core_row:
                return 0.5

            core_domain = core_row["domain"]
            if core_domain == domain:
                return 1.0

            # 通过业力边计算最短路径长度（BFS）
            path_length = self._bfs_domain_distance(domain, core_domain)
            if path_length is None:
                return 0.5  # 不可达

            # 最大路径长度：取领域图的直径近似值
            max_path = 5  # 经验值
            return round(max(0.0, 1.0 - path_length / max_path), 4)
        except Exception as e:
            log.warning("系统核心度计算失败: %s, domain=%s", e, domain)
            return 0.5

    def _compute_uncertainty(self, domain: str) -> float:
        """计算不确定性因子

        公式: min(1.0, conflict_frequency / GOAL_HIGH_CONFLICT_THRESHOLD)
        """
        try:
            meta_label = f"meta:{domain}"
            meta_seed = self._meta_seed_mgr.get_meta_seed(meta_label)
            if meta_seed is None:
                return 0.5

            conflict_freq = meta_seed.metrics.get("conflict_frequency", 0)
            if not isinstance(conflict_freq, (int, float)):
                return 0.5

            return round(min(1.0, conflict_freq / GOAL_HIGH_CONFLICT_THRESHOLD), 4)
        except Exception as e:
            log.warning("不确定性计算失败: %s, domain=%s", e, domain)
            return 0.5

    def _compute_decomposability(self, domain: str) -> float:
        """计算可分解性因子

        公式: min(1.0, 目标领域子领域数 / GOAL_DECOMPOSABILITY_NORM)
        子领域 = 该领域下有独立业力边的种子聚类
        """
        try:
            # 统计该领域下有出边的种子数作为子领域近似
            row = self._graph.conn.execute(
                "SELECT COUNT(DISTINCT ke.source) as cnt "
                "FROM karma_edges ke "
                "JOIN seeds s ON ke.source = s.label "
                "WHERE s.domain = ? AND s.type != 'META'",
                (domain,),
            ).fetchone()
            sub_domain_count = row["cnt"] if row else 0

            return round(min(1.0, sub_domain_count / GOAL_DECOMPOSABILITY_NORM), 4)
        except Exception as e:
            log.warning("可分解性计算失败: %s, domain=%s", e, domain)
            return 0.5

    # ── 目标冷却 ──────────────────────────────────────

    def cool_goals(self) -> int:
        """执行目标冷却检查

        Returns:
            状态变更的目标数量
        """
        if not COGNITIVE_GOAL_ENABLED:
            return 0

        cooled = 0
        now = datetime.now(timezone.utc)

        # A. 权重衰减
        try:
            pending_goals = self._graph.conn.execute(
                "SELECT goal_id, priority_weight, decay_cycles_count, "
                "last_touched_at, associated_user FROM cognitive_goals "
                "WHERE status = 'pending'"
            ).fetchall()

            for goal in pending_goals:
                # 检查是否达到衰减周期
                last_touched = goal["last_touched_at"]
                if last_touched:
                    try:
                        touched_dt = datetime.fromisoformat(last_touched)
                        cycles_since_touch = int(
                            (now - touched_dt).total_seconds() / GUARDIAN_LOOP_INTERVAL
                        )
                    except (ValueError, TypeError):
                        cycles_since_touch = 0
                else:
                    cycles_since_touch = 0

                if cycles_since_touch >= GOAL_DECAY_CYCLES:
                    old_weight = goal["priority_weight"]
                    new_weight = round(old_weight * GOAL_DECAY_FACTOR, 4)

                    if new_weight < GOAL_EXPIRE_THRESHOLD:
                        self._record_goal_history(
                            goal["goal_id"], "pending", "expired",
                            old_weight, new_weight, "decay_expire",
                        )
                        self._graph.conn.execute(
                            "UPDATE cognitive_goals SET status = 'expired', "
                            "priority_weight = ?, updated_at = ? WHERE goal_id = ?",
                            (new_weight, now.isoformat(), goal["goal_id"]),
                        )
                        log.info(
                            "认知目标过期: goal_id=%s, weight=%.4f < threshold=%.4f",
                            goal["goal_id"], new_weight, GOAL_EXPIRE_THRESHOLD,
                        )
                    else:
                        self._record_goal_history(
                            goal["goal_id"], "pending", "pending",
                            old_weight, new_weight, "decay",
                        )
                        self._graph.conn.execute(
                            "UPDATE cognitive_goals SET priority_weight = ?, "
                            "decay_cycles_count = decay_cycles_count + 1, "
                            "updated_at = ? WHERE goal_id = ?",
                            (new_weight, now.isoformat(), goal["goal_id"]),
                        )
                        log.info(
                            "认知目标衰减: goal_id=%s, weight %.4f → %.4f",
                            goal["goal_id"], old_weight, new_weight,
                        )
                    cooled += 1
        except Exception as e:
            log.warning("目标权重衰减失败: %s", e)

        # B. 用户缺席归档
        try:
            goals_with_user = self._graph.conn.execute(
                "SELECT goal_id, associated_user, priority_weight FROM cognitive_goals "
                "WHERE status = 'pending' AND associated_user IS NOT NULL"
            ).fetchall()

            for goal in goals_with_user:
                if self._is_user_absent(goal["associated_user"]):
                    self._record_goal_history(
                        goal["goal_id"], "pending", "archived",
                        goal["priority_weight"], goal["priority_weight"], "user_absent",
                    )
                    self._graph.conn.execute(
                        "UPDATE cognitive_goals SET status = 'archived', "
                        "updated_at = ? WHERE goal_id = ?",
                        (now.isoformat(), goal["goal_id"]),
                    )
                    log.info(
                        "认知目标归档（用户离开）: goal_id=%s, user=%s",
                        goal["goal_id"], goal["associated_user"],
                    )
                    cooled += 1
        except Exception as e:
            log.warning("用户缺席归档失败: %s", e)

        # C. 池大小检查
        try:
            active_count = self._get_active_goal_count()
            if active_count > GOAL_POOL_MAX_SIZE:
                evict_count = active_count - GOAL_POOL_MAX_SIZE
                lowest = self._graph.conn.execute(
                    "SELECT goal_id, priority_weight FROM cognitive_goals "
                    "WHERE status = 'pending' "
                    "ORDER BY priority_weight ASC, created_at ASC "
                    "LIMIT ?",
                    (evict_count,),
                ).fetchall()
                for row in lowest:
                    self._record_goal_history(
                        row["goal_id"], "pending", "expired",
                        row["priority_weight"], row["priority_weight"], "pool_evict",
                    )
                    self._graph.conn.execute(
                        "UPDATE cognitive_goals SET status = 'expired', "
                        "updated_at = ? WHERE goal_id = ?",
                        (now.isoformat(), row["goal_id"]),
                    )
                    cooled += 1
                log.warning(
                    "认知目标池超限淘汰: evicted=%d, active=%d, max=%d",
                    evict_count, active_count, GOAL_POOL_MAX_SIZE,
                )
        except Exception as e:
            log.warning("池大小检查失败: %s", e)

        return cooled

    # ── 目标查询 ──────────────────────────────────────

    def get_goal(self, goal_id: str) -> CognitiveGoalData | None:
        """查询单个认知目标"""
        if not COGNITIVE_GOAL_ENABLED:
            return None
        try:
            row = self._graph.conn.execute(
                "SELECT * FROM cognitive_goals WHERE goal_id = ?",
                (goal_id,),
            ).fetchone()
            if not row:
                return None
            return self._row_to_goal_data(row)
        except Exception as e:
            log.warning("查询认知目标失败: %s, goal_id=%s", e, goal_id)
            return None

    def list_goals(
        self,
        status: GoalStatus | None = None,
        goal_type: GoalType | None = None,
    ) -> list[CognitiveGoalData]:
        """查询认知目标列表"""
        if not COGNITIVE_GOAL_ENABLED:
            return []
        try:
            query = "SELECT * FROM cognitive_goals WHERE 1=1"
            params: list = []

            if status is not None:
                query += " AND status = ?"
                params.append(status.value)
            if goal_type is not None:
                query += " AND goal_type = ?"
                params.append(goal_type.value)

            query += " ORDER BY priority_weight DESC, created_at DESC"

            rows = self._graph.conn.execute(query, params).fetchall()
            return [self._row_to_goal_data(r) for r in rows]
        except Exception as e:
            log.warning("查询认知目标列表失败: %s", e)
            return []

    def get_goal_stats(self) -> dict:
        """查询认知目标统计信息"""
        if not COGNITIVE_GOAL_ENABLED:
            return {"by_status": {}, "by_type": {}, "avg_priority_weight": 0.0,
                    "pool_usage": {"active": 0, "max": GOAL_POOL_MAX_SIZE, "usage_percent": 0.0}}
        try:
            # 按状态统计
            by_status: dict[str, int] = {}
            status_rows = self._graph.conn.execute(
                "SELECT status, COUNT(*) as cnt FROM cognitive_goals GROUP BY status"
            ).fetchall()
            for r in status_rows:
                by_status[r["status"]] = r["cnt"]

            # 按类型统计
            by_type: dict[str, int] = {}
            type_rows = self._graph.conn.execute(
                "SELECT goal_type, COUNT(*) as cnt FROM cognitive_goals GROUP BY goal_type"
            ).fetchall()
            for r in type_rows:
                by_type[r["goal_type"]] = r["cnt"]

            # 平均优先级权重
            avg_row = self._graph.conn.execute(
                "SELECT AVG(priority_weight) as avg_w FROM cognitive_goals"
            ).fetchone()
            avg_priority = round(avg_row["avg_w"], 4) if avg_row and avg_row["avg_w"] is not None else 0.0

            # 池使用率
            active_count = self._get_active_goal_count()
            usage_percent = round(active_count / GOAL_POOL_MAX_SIZE * 100, 2) if GOAL_POOL_MAX_SIZE > 0 else 0.0

            return {
                "by_status": by_status,
                "by_type": by_type,
                "avg_priority_weight": avg_priority,
                "pool_usage": {
                    "active": active_count,
                    "max": GOAL_POOL_MAX_SIZE,
                    "usage_percent": usage_percent,
                },
            }
        except Exception as e:
            log.warning("查询认知目标统计失败: %s", e)
            return {"by_status": {}, "by_type": {}, "avg_priority_weight": 0.0,
                    "pool_usage": {"active": 0, "max": GOAL_POOL_MAX_SIZE, "usage_percent": 0.0}}

    def touch_goal_domain(self, domain: str) -> None:
        """更新目标领域的最近触及时间（由校验器调用）"""
        if not COGNITIVE_GOAL_ENABLED:
            return
        try:
            now = datetime.now(timezone.utc).isoformat()
            self._graph.conn.execute(
                "UPDATE cognitive_goals SET last_touched_at = ?, updated_at = ? "
                "WHERE domain = ? AND status = 'pending'",
                (now, now, domain),
            )
        except Exception as e:
            log.debug("目标触及更新失败: %s, domain=%s", e, domain)

    # ── 内部方法 ──────────────────────────────────────

    def _generate_goal_id(self, domain: str, goal_type: GoalType) -> str:
        """生成目标 ID: goal_{timestamp}_{domain}_{hash}"""
        now = datetime.now(timezone.utc)
        timestamp = now.strftime("%Y%m%dT%H%M%S")
        hash_input = f"{domain}:{goal_type.value}:{now.isoformat()}"
        hash_suffix = hashlib.md5(hash_input.encode()).hexdigest()[:6]
        return f"goal_{timestamp}_{domain}_{hash_suffix}"

    def _compute_global_avg_density(self) -> float:
        """计算全局平均业力密度"""
        try:
            row = self._graph.conn.execute(
                "SELECT AVG(ke.weight) as avg_w "
                "FROM karma_edges ke "
                "JOIN seeds s ON ke.source = s.label "
                "WHERE s.type != 'META' AND ke.source NOT LIKE 'meta:%'"
            ).fetchone()
            return round(row["avg_w"], 4) if row and row["avg_w"] is not None else 0.0
        except Exception:
            return 0.0

    def _count_keyword_in_window(self, keyword: str) -> int:
        """统计关键词在最近 N 小时内的出现次数"""
        try:
            row = self._graph.conn.execute(
                "SELECT count FROM unmatched_queries WHERE query_text = ?",
                (keyword,),
            ).fetchone()
            return row["count"] if row else 0
        except Exception:
            return 0

    def _infer_domain_for_keyword(self, keyword: str) -> str:
        """推断关键词所属领域"""
        # 从候选种子表中查找
        try:
            row = self._graph.conn.execute(
                "SELECT domain FROM candidate_seeds WHERE label = ?",
                (keyword,),
            ).fetchone()
            if row and row["domain"]:
                return row["domain"]
        except Exception:
            pass
        return "未分类"

    def _find_active_user_for_keyword(self, keyword: str) -> str | None:
        """查找最近触及该关键词的活跃用户"""
        try:
            # 转义 LIKE 通配符，防止意外匹配
            escaped = keyword.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")
            row = self._graph.conn.execute(
                "SELECT user_id FROM query_history "
                "WHERE query_text LIKE ? ESCAPE '\\' "
                "ORDER BY created_at DESC LIMIT 1",
                (f"%{escaped}%",),
            ).fetchone()
            return row["user_id"] if row else None
        except Exception:
            return None

    def _get_active_goal_count(self) -> int:
        """获取活跃目标数"""
        try:
            row = self._graph.conn.execute(
                "SELECT COUNT(*) as cnt FROM cognitive_goals "
                "WHERE status IN ('pending', 'exploring', 'querying_external')"
            ).fetchone()
            return row["cnt"] if row else 0
        except Exception:
            return 0

    def _bfs_domain_distance(self, from_domain: str, to_domain: str) -> int | None:
        """BFS 计算两个领域之间的最短路径长度

        通过业力边连接的种子跨领域关系计算。
        """
        try:
            # 获取 from_domain 的所有种子
            from_seeds = self._graph.conn.execute(
                "SELECT label FROM seeds WHERE domain = ? AND type != 'META'",
                (from_domain,),
            ).fetchall()
            from_labels = {r["label"] for r in from_seeds}

            # 获取 to_domain 的所有种子
            to_seeds = self._graph.conn.execute(
                "SELECT label FROM seeds WHERE domain = ? AND type != 'META'",
                (to_domain,),
            ).fetchall()
            to_labels = {r["label"] for r in to_seeds}

            if not from_labels or not to_labels:
                return None

            # BFS: 从 from_labels 出发，通过业力边到达 to_labels
            visited: set[str] = set(from_labels)
            queue: deque[tuple[str, int]] = deque((label, 0) for label in from_labels)

            while queue:
                current, depth = queue.popleft()
                if depth > 5:  # 限制搜索深度
                    return None

                # 获取当前种子的所有邻居
                edges = self._graph.conn.execute(
                    "SELECT target FROM karma_edges WHERE source = ?",
                    (current,),
                ).fetchall()
                neighbors = {r["target"] for r in edges}

                for neighbor in neighbors:
                    if neighbor in to_labels:
                        return depth + 1
                    if neighbor not in visited:
                        visited.add(neighbor)
                        queue.append((neighbor, depth + 1))

            return None  # 不可达
        except Exception as e:
            log.warning("BFS 领域距离计算失败: %s", e)
            return None

    def _is_user_absent(self, user_label: str) -> bool:
        """检查用户是否缺席"""
        try:
            row = self._graph.conn.execute(
                "SELECT updated_at FROM user_cold_start WHERE user_label = ?",
                (user_label,),
            ).fetchone()
            if not row or not row["updated_at"]:
                return True
            last_active = datetime.fromisoformat(row["updated_at"])
            # 确保时区一致：如果 last_active 没有时区信息，假定为 UTC
            if last_active.tzinfo is None:
                last_active = last_active.replace(tzinfo=timezone.utc)
            absence_seconds = (datetime.now(timezone.utc) - last_active).total_seconds()
            absence_cycles = int(absence_seconds / GUARDIAN_LOOP_INTERVAL)
            return absence_cycles >= GOAL_USER_ABSENCE_CYCLES
        except Exception:
            return False

    def _record_goal_history(
        self,
        goal_id: str,
        old_status: str,
        new_status: str,
        old_weight: float,
        new_weight: float,
        reason: str,
    ) -> None:
        """记录目标状态变更历史"""
        try:
            now = datetime.now(timezone.utc).isoformat()
            self._graph.conn.execute(
                "INSERT INTO goal_history "
                "(goal_id, old_status, new_status, old_weight, new_weight, reason, created_at) "
                "VALUES (?, ?, ?, ?, ?, ?, ?)",
                (goal_id, old_status, new_status, old_weight, new_weight, reason, now),
            )
        except Exception as e:
            log.warning("目标历史记录失败: %s, goal_id=%s", e, goal_id)

    def _row_to_goal_data(self, row) -> CognitiveGoalData:
        """将数据库行转换为 CognitiveGoalData"""
        # 解析 sub_goals
        try:
            sub_goals = json.loads(row["sub_goals"]) if row["sub_goals"] else []
        except (json.JSONDecodeError, TypeError):
            sub_goals = []

        # 解析 execution_log
        try:
            execution_log = json.loads(row["execution_log"]) if row["execution_log"] else []
        except (json.JSONDecodeError, TypeError):
            execution_log = []

        return CognitiveGoalData(
            goal_id=row["goal_id"],
            goal_type=GoalType(row["goal_type"]),
            trigger_condition=row["trigger_condition"],
            domain=row["domain"],
            priority_weight=row["priority_weight"],
            status=GoalStatus(row["status"]),
            sub_goals=sub_goals,
            execution_log=execution_log,
            associated_user=row["associated_user"],
            decay_cycles_count=row["decay_cycles_count"],
            last_touched_at=row["last_touched_at"] or "",
            created_at=row["created_at"] or "",
            updated_at=row["updated_at"] or "",
        )