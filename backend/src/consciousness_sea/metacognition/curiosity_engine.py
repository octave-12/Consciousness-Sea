"""
CuriosityEngine — 好奇心引擎

基于认知目标池自动执行内部探索和外部查询。

职责:
  - 判断探索策略（内部探索/候选升级/外部查询）
  - 构造虚拟查询执行内部涟漪探索
  - 执行外部知识源查询
  - 探索结果写入提炼池
  - 运行状态管理

线程安全:
  - _explore_lock 保护并发控制
  - 数据库操作由 SQLite WAL 模式保证并发安全
"""

from __future__ import annotations

import json
import logging
import sqlite3
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

from consciousness_sea.domain.graph_db import GraphDB
from consciousness_sea.infrastructure.config import (
    CURIOSITY_ACTIVATION_THRESHOLD,
    CURIOSITY_ENGINE_ENABLED,
    CURIOSITY_MAX_CONCURRENT,
    CURIOSITY_MAX_DEPTH,
    DEFAULT_DATA_DIR,
    EXTERNAL_QUERY_ENABLED,
    EXTERNAL_QUERY_MAX_RETRIES,
    EXTERNAL_SOURCE_TYPE,
    ZHWIKI_FILENAME,
)

from .cognitive_goal import CognitiveGoalData, CognitiveGoalManager

log = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════
#  数据类
# ═══════════════════════════════════════════════════════════


@dataclass
class ExternalQueryResult:
    """外部查询结果"""
    title: str
    summary: str
    related_terms: list[str]
    categories: list[str]


@dataclass
class ExplorationResult:
    """探索结果"""
    goal_id: str
    strategy: str  # "internal" / "candidate_upgrade" / "external" / "none" / "disabled" / "error"
    explored_seeds: list[str] = field(default_factory=list)
    new_associations: int = 0
    distillation_candidates: int = 0
    duration_ms: int = 0
    error: str | None = None


@dataclass
class CuriosityEngineStatus:
    """好奇心引擎运行状态"""
    total_explorations: int = 0
    total_new_associations: int = 0
    total_external_queries: int = 0
    last_exploration_time: str | None = None
    last_exploration_result: str | None = None
    is_exploring: bool = False


# ═══════════════════════════════════════════════════════════
#  CuriosityEngine
# ═══════════════════════════════════════════════════════════


class CuriosityEngine:
    """好奇心引擎 — 基于认知目标池自动执行内部探索和外部查询

    职责:
      - 判断探索策略（内部探索/候选升级/外部查询）
      - 构造虚拟查询执行内部涟漪探索
      - 执行外部知识源查询
      - 探索结果写入提炼池
      - 运行状态管理

    线程安全:
      - _explore_lock 保护并发控制
      - 数据库操作由 SQLite WAL 模式保证并发安全

    Args:
        graph: 知识图谱连接
        goal_mgr: 认知目标管理器
    """

    def __init__(self, graph: GraphDB, goal_mgr: CognitiveGoalManager) -> None:
        self._graph = graph
        self._goal_mgr = goal_mgr
        self._explore_lock = threading.Lock()
        self._current_explorations = 0

        # 运行状态
        self._total_explorations = 0
        self._total_new_associations = 0
        self._total_external_queries = 0
        self._last_exploration_time: str | None = None
        self._last_exploration_result: str | None = None

    # ── 探索入口 ──────────────────────────────────────

    def explore(self, goal: CognitiveGoalData) -> ExplorationResult:
        """对指定认知目标执行探索

        流程:
          1. 判断探索策略
          2. 执行探索
          3. 写入提炼池
          4. 更新目标状态

        Args:
            goal: 认知目标数据

        Returns:
            ExplorationResult
        """
        if not CURIOSITY_ENGINE_ENABLED:
            return ExplorationResult(
                goal_id=goal.goal_id, strategy="disabled",
                error="好奇心引擎已禁用",
            )

        # 并发控制
        with self._explore_lock:
            if self._current_explorations >= CURIOSITY_MAX_CONCURRENT:
                return ExplorationResult(
                    goal_id=goal.goal_id, strategy="none",
                    error="好奇心引擎并发已满",
                )
            self._current_explorations += 1

        start_time = time.monotonic()

        try:
            # 判断探索策略
            strategy = self._determine_strategy(goal.domain)

            result = ExplorationResult(goal_id=goal.goal_id, strategy=strategy)

            if strategy == "internal":
                result = self._explore_internal(goal)
            elif strategy == "candidate_upgrade":
                result = self._explore_candidate_upgrade(goal)
            elif strategy == "external":
                result = self._explore_external(goal)
            else:
                result = ExplorationResult(
                    goal_id=goal.goal_id, strategy="none",
                    error="无法确定探索策略",
                )

            # 更新目标状态
            now = datetime.now(timezone.utc).isoformat()
            if result.error:
                # 探索失败 → 恢复 pending，记录失败日志
                self._append_execution_log(goal.goal_id, {
                    "timestamp": now,
                    "action": f"{strategy}_exploration",
                    "result": f"failed: {result.error}",
                })
                self._graph.conn.execute(
                    "UPDATE cognitive_goals SET status = 'pending', updated_at = ? "
                    "WHERE goal_id = ?",
                    (now, goal.goal_id),
                )
            else:
                # 探索成功 → 标记 completed
                self._append_execution_log(goal.goal_id, {
                    "timestamp": now,
                    "action": f"{strategy}_exploration",
                    "result": f"found {result.new_associations} new associations",
                    "seeds": result.explored_seeds[:10],
                })
                self._graph.conn.execute(
                    "UPDATE cognitive_goals SET status = 'completed', "
                    "priority_weight = 0.1, updated_at = ? WHERE goal_id = ?",
                    (now, goal.goal_id),
                )

            # 更新运行状态
            self._total_explorations += 1
            self._total_new_associations += result.new_associations
            if strategy == "external":
                self._total_external_queries += 1
            self._last_exploration_time = now
            self._last_exploration_result = "success" if not result.error else "failed"

            result.duration_ms = int((time.monotonic() - start_time) * 1000)
            return result

        except Exception as e:
            log.error("好奇心引擎探索异常: %s, goal_id=%s", e, goal.goal_id)
            return ExplorationResult(
                goal_id=goal.goal_id, strategy="error", error=str(e),
            )

        finally:
            with self._explore_lock:
                self._current_explorations -= 1

    # ── 策略判断 ──────────────────────────────────────

    def _determine_strategy(self, domain: str) -> str:
        """判断探索策略

        分界逻辑:
          1. 内部已有种子且存在业力边 → "internal"
          2. 内部无种子但提炼池有候选 → "candidate_upgrade"
          3. 内部完全空白 → "external"（若启用）或 "none"
        """
        # 检查内部种子
        try:
            seed_row = self._graph.conn.execute(
                "SELECT COUNT(*) as cnt FROM seeds "
                "WHERE domain = ? AND type != 'META'",
                (domain,),
            ).fetchone()
            seed_count = seed_row["cnt"] if seed_row else 0

            if seed_count > 0:
                # 检查是否有业力边
                edge_row = self._graph.conn.execute(
                    "SELECT COUNT(*) as cnt FROM karma_edges ke "
                    "JOIN seeds s ON ke.source = s.label "
                    "WHERE s.domain = ? AND s.type != 'META'",
                    (domain,),
                ).fetchone()
                edge_count = edge_row["cnt"] if edge_row else 0

                if edge_count > 0:
                    return "internal"
        except Exception as e:
            log.warning("内部种子检查失败: %s, domain=%s", e, domain)

        # 检查提炼池候选（通过 seeds 表关联或 LIKE 匹配）
        try:
            # 优先通过 seeds 表精确匹配 domain
            pool_row = self._graph.conn.execute(
                "SELECT COUNT(*) as cnt FROM distillation_pool dp "
                "INNER JOIN seeds s1 ON dp.canonical_source = s1.label "
                "WHERE dp.status = 'pending' AND s1.domain = ?",
                (domain,),
            ).fetchone()
            pool_count = pool_row["cnt"] if pool_row else 0

            if pool_count == 0:
                # 回退：通过 seeds 表匹配 target
                pool_row = self._graph.conn.execute(
                    "SELECT COUNT(*) as cnt FROM distillation_pool dp "
                    "INNER JOIN seeds s2 ON dp.canonical_target = s2.label "
                    "WHERE dp.status = 'pending' AND s2.domain = ?",
                    (domain,),
                ).fetchone()
                pool_count = pool_row["cnt"] if pool_row else 0

            if pool_count == 0:
                # 最终回退：LIKE 匹配（转义通配符）
                escaped = domain.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")
                pool_row = self._graph.conn.execute(
                    "SELECT COUNT(*) as cnt FROM distillation_pool "
                    "WHERE status = 'pending' AND "
                    "(canonical_source LIKE ? ESCAPE '\\' OR canonical_target LIKE ? ESCAPE '\\')",
                    (f"%{escaped}%", f"%{escaped}%"),
                ).fetchone()
                pool_count = pool_row["cnt"] if pool_row else 0

            if pool_count > 0:
                return "candidate_upgrade"
        except Exception as e:
            log.warning("提炼池检查失败: %s, domain=%s", e, domain)

        # 完全空白 → 外部查询
        if EXTERNAL_QUERY_ENABLED:
            return "external"

        return "none"

    # ── 内部探索 ──────────────────────────────────────

    def _explore_internal(self, goal: CognitiveGoalData) -> ExplorationResult:
        """内部探索：构造虚拟查询执行涟漪传播

        虚拟查询构造策略:
          - 获取目标领域的核心种子（出边数最多的前 5 个）
          - 组合核心种子标签为虚拟查询文本
          - 调用 router.route(query, skip_verification=True)
          - 从涟漪结果中提取新关联（无业力边的种子对）
          - 写入提炼池 (source_tag="auto_curiosity")
        """
        from consciousness_sea.domain.router import route

        # 更新目标状态为 exploring
        now = datetime.now(timezone.utc).isoformat()
        self._graph.conn.execute(
            "UPDATE cognitive_goals SET status = 'exploring', updated_at = ? "
            "WHERE goal_id = ?",
            (now, goal.goal_id),
        )

        # 获取核心种子
        try:
            rows = self._graph.conn.execute(
                "SELECT s.label, COUNT(ke.source) as edge_count "
                "FROM seeds s LEFT JOIN karma_edges ke ON s.label = ke.source "
                "WHERE s.domain = ? AND s.type != 'META' "
                "GROUP BY s.label ORDER BY edge_count DESC LIMIT 5",
                (goal.domain,),
            ).fetchall()
            core_seeds = [r["label"] for r in rows]
        except Exception as e:
            return ExplorationResult(
                goal_id=goal.goal_id, strategy="internal",
                error=f"核心种子查询失败: {e}",
            )

        if not core_seeds:
            # 无种子 → 切换到外部查询
            if EXTERNAL_QUERY_ENABLED:
                self._graph.conn.execute(
                    "UPDATE cognitive_goals SET status = 'querying_external', "
                    "updated_at = ? WHERE goal_id = ?",
                    (now, goal.goal_id),
                )
                return self._explore_external(goal)
            return ExplorationResult(
                goal_id=goal.goal_id, strategy="internal",
                error="无核心种子，外部查询已禁用",
            )

        # 构造虚拟查询
        virtual_query = f"{goal.domain} {' '.join(core_seeds)}"

        # 执行涟漪传播（跳过校验和熏习）
        try:
            result = route(
                virtual_query, self._graph,
                skip_verification=True,
                max_depth=CURIOSITY_MAX_DEPTH,
            )
        except Exception as e:
            return ExplorationResult(
                goal_id=goal.goal_id, strategy="internal",
                error=f"虚拟查询执行失败: {e}",
            )

        # 提取新关联
        new_associations = 0
        explored_seeds = list(result.activated.keys())[:20]
        distillation_candidates = 0

        # 过滤低激活值种子
        active_labels = {
            label for label, node in result.activated.items()
            if node.activation >= CURIOSITY_ACTIVATION_THRESHOLD
        }

        # 检查种子对之间是否缺少业力边
        # 限制最大检查对数，避免 O(n²) 性能问题
        max_pairs_to_check = 50
        pairs_checked = 0
        active_list = list(active_labels)
        for i in range(len(active_list)):
            if pairs_checked >= max_pairs_to_check:
                break
            for j in range(i + 1, min(i + 5, len(active_list))):
                if pairs_checked >= max_pairs_to_check:
                    break
                pairs_checked += 1
                src, tgt = active_list[i], active_list[j]
                # 检查是否已有业力边
                try:
                    existing = self._graph.conn.execute(
                        "SELECT 1 FROM karma_edges "
                        "WHERE (source=? AND target=?) OR (source=? AND target=?)",
                        (src, tgt, tgt, src),
                    ).fetchone()
                    if not existing:
                        # 写入提炼池
                        try:
                            from consciousness_sea.learning.distillation_pool import (
                                DistillationPool,
                            )
                            distill = DistillationPool(self._graph)
                            distill.submit_candidate(
                                user_label="system:curiosity",
                                source=src,
                                target=tgt,
                                relation="COOCCURS_WITH",
                            )
                            new_associations += 1
                            distillation_candidates += 1
                        except Exception as e:
                            log.warning("提炼池写入失败: %s", e)
                except Exception as e:
                    log.warning("业力边检查失败: %s", e)

        return ExplorationResult(
            goal_id=goal.goal_id,
            strategy="internal",
            explored_seeds=explored_seeds,
            new_associations=new_associations,
            distillation_candidates=distillation_candidates,
        )

    # ── 候选升级 ──────────────────────────────────────

    def _explore_candidate_upgrade(self, goal: CognitiveGoalData) -> ExplorationResult:
        """候选升级：触发提炼池中相关候选的升级流程"""
        try:
            from consciousness_sea.learning.distillation_pool import DistillationPool
            distill = DistillationPool(self._graph)
            # 查找相关候选并尝试升级
            upgraded = distill.try_upgrade_by_domain(goal.domain)
            return ExplorationResult(
                goal_id=goal.goal_id,
                strategy="candidate_upgrade",
                new_associations=upgraded,
                distillation_candidates=0,
            )
        except Exception as e:
            return ExplorationResult(
                goal_id=goal.goal_id, strategy="candidate_upgrade",
                error=f"候选升级失败: {e}",
            )

    # ── 外部查询 ──────────────────────────────────────

    def _explore_external(self, goal: CognitiveGoalData) -> ExplorationResult:
        """外部查询：向外部知识源发送查询

        流程:
          1. 更新目标状态为 querying_external
          2. 根据配置选择知识源
          3. 执行查询（带重试和超时）
          4. 转换为统一格式
          5. 写入提炼池 (source_tag="external_seed")
        """
        if not EXTERNAL_QUERY_ENABLED:
            return ExplorationResult(
                goal_id=goal.goal_id, strategy="external",
                error="外部查询已禁用",
            )

        # 更新目标状态
        now = datetime.now(timezone.utc).isoformat()
        self._graph.conn.execute(
            "UPDATE cognitive_goals SET status = 'querying_external', "
            "updated_at = ? WHERE goal_id = ?",
            (now, goal.goal_id),
        )

        # 执行外部查询（带重试）
        query_result: ExternalQueryResult | None = None
        for attempt in range(1, EXTERNAL_QUERY_MAX_RETRIES + 1):
            try:
                query_result = self._query_external_source(goal.domain)
                break
            except Exception as e:
                log.warning(
                    "外部查询失败 (第 %d 次): %s, domain=%s",
                    attempt, e, goal.domain,
                )
                if attempt == EXTERNAL_QUERY_MAX_RETRIES:
                    return ExplorationResult(
                        goal_id=goal.goal_id, strategy="external",
                        error=f"外部查询重试 {EXTERNAL_QUERY_MAX_RETRIES} 次后仍失败: {e}",
                    )

        if query_result is None:
            return ExplorationResult(
                goal_id=goal.goal_id, strategy="external",
                error="外部查询返回空结果",
            )

        # 写入提炼池
        new_associations = 0
        distillation_candidates = 0

        try:
            from consciousness_sea.learning.distillation_pool import DistillationPool
            distill = DistillationPool(self._graph)

            # 写入主条目
            distill.submit_external_candidate(
                label=query_result.title,
                domain=query_result.categories[0] if query_result.categories else goal.domain,
                summary=query_result.summary,
            )
            distillation_candidates += 1

            # 写入关联词
            for related in query_result.related_terms:
                distill.submit_candidate(
                    user_label="system:curiosity",
                    source=query_result.title,
                    target=related,
                    relation="RELATED",
                )
                new_associations += 1
                distillation_candidates += 1

        except Exception as e:
            log.warning("外部查询结果写入提炼池失败: %s", e)

        return ExplorationResult(
            goal_id=goal.goal_id,
            strategy="external",
            explored_seeds=[query_result.title],
            new_associations=new_associations,
            distillation_candidates=distillation_candidates,
        )

    def _query_external_source(self, domain: str) -> ExternalQueryResult | None:
        """执行外部知识源查询

        支持:
          - wikipedia_dump: 本地 Wikipedia 数据库
          - web_search: 网络搜索（可选，暂不实现）
        """
        if EXTERNAL_SOURCE_TYPE == "wikipedia_dump":
            return self._query_wikipedia_dump(domain)
        else:
            log.warning("不支持的外部知识源类型: %s", EXTERNAL_SOURCE_TYPE)
            return None

    def _query_wikipedia_dump(self, query: str) -> ExternalQueryResult | None:
        """从本地 Wikipedia 数据库查询"""
        db_path = Path(DEFAULT_DATA_DIR) / ZHWIKI_FILENAME
        if not db_path.exists():
            log.warning("Wikipedia 数据库不存在: %s", db_path)
            return None

        try:
            conn = sqlite3.connect(str(db_path))
            conn.row_factory = sqlite3.Row
            try:
                # 转义 LIKE 通配符，防止意外匹配
                escaped = query.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")
                row = conn.execute(
                    "SELECT title, summary, related_terms, categories FROM articles "
                    "WHERE title LIKE ? ESCAPE '\\' LIMIT 1",
                    (f"%{escaped}%",),
                ).fetchone()

                if not row:
                    return None

                related = json.loads(row["related_terms"]) if row["related_terms"] else []
                categories = json.loads(row["categories"]) if row["categories"] else []

                return ExternalQueryResult(
                    title=row["title"],
                    summary=row["summary"] or "",
                    related_terms=related,
                    categories=categories,
                )
            finally:
                conn.close()
        except Exception as e:
            log.warning("Wikipedia 查询失败: %s, query=%s", e, query)
            return None

    # ── 状态查询 ──────────────────────────────────────

    def get_status(self) -> CuriosityEngineStatus:
        """查询好奇心引擎运行状态"""
        return CuriosityEngineStatus(
            total_explorations=self._total_explorations,
            total_new_associations=self._total_new_associations,
            total_external_queries=self._total_external_queries,
            last_exploration_time=self._last_exploration_time,
            last_exploration_result=self._last_exploration_result,
            is_exploring=self._current_explorations > 0,
        )

    # ── 内部方法 ──────────────────────────────────────

    def _append_execution_log(self, goal_id: str, entry: dict) -> None:
        """追加执行日志（限制最大 20 条）"""
        try:
            row = self._graph.conn.execute(
                "SELECT execution_log FROM cognitive_goals WHERE goal_id = ?",
                (goal_id,),
            ).fetchone()
            if not row:
                return

            log_list = json.loads(row["execution_log"]) if row["execution_log"] else []
            log_list.append(entry)
            # 仅保留最近 20 条
            if len(log_list) > 20:
                log_list = log_list[-20:]

            self._graph.conn.execute(
                "UPDATE cognitive_goals SET execution_log = ? WHERE goal_id = ?",
                (json.dumps(log_list, ensure_ascii=False), goal_id),
            )
        except Exception as e:
            log.warning("执行日志追加失败: %s, goal_id=%s", e, goal_id)
