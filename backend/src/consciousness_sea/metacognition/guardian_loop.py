"""
GuardianLoop — 守护循环

后台定时任务，驱动元种子体系的指标更新、元业力熏习、新增领域/关系探测。

守护线程模式参考 checkpoint.py 的 start_daemon / stop_daemon。

职责:
  - 定期检查新增领域/关系类型 → 生成缺失的元种子
  - 更新所有元种子的监控指标
  - 根据指标变化触发元业力熏习
  - 提交所有数据库变更
  - 记录执行日志

线程安全:
  - _loop_lock 保护单次执行的串行化
  - 数据库操作由 SQLite WAL 模式保证并发安全
  - 与查询并发安全由 WAL 快照读保证
"""

from __future__ import annotations

import json
import logging
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Optional

from consciousness_sea.domain.graph_db import GraphDB
from .meta_seed import MetaSeedManager, MetaSeedCategory, MetaSeedData
from consciousness_sea.infrastructure.config import (
    META_SEED_ENABLED,
    GUARDIAN_LOOP_INTERVAL,
    GUARDIAN_LOOP_TIMEOUT,
    GUARDIAN_LOOP_INITIAL_DELAY,
    GUARDIAN_METRICS_WINDOW,
    META_SEED_DORMANT_CYCLES,
    META_EXPLORE_WINDOW,
    META_EXPLORE_LOW_CONF_THRESHOLD,
    META_KARMA_DELTA_THRESHOLD,
    META_KARMA_INITIAL_WEIGHT,
    META_ALERT_CONFLICT_THRESHOLD,
    CONFIDENCE_LOW,
    COGNITIVE_GOAL_ENABLED,
    CURIOSITY_ENGINE_ENABLED,
    GOAL_AUTO_EXPLORE_THRESHOLD,
    GOAL_MAX_EXPLORE_PER_CYCLE,
    PERCEPTION_ENABLED,
    MULTIMODAL_ALIGNMENT_ENABLED,
    MULTIMODAL_ALIGNMENT_INTERVAL,
)

log = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════
#  数据类
# ═══════════════════════════════════════════════════════════


@dataclass
class GuardianLoopResult:
    """守护循环单次执行结果"""
    success: bool
    meta_seeds_updated: int = 0
    meta_karma_edges_created: int = 0
    duration_ms: int = 0
    error: str | None = None


@dataclass
class GuardianLoopStatus:
    """守护循环运行状态"""
    is_running: bool
    last_execution_time: str | None
    last_execution_result: str | None  # "success" / "failed"
    last_execution_duration_ms: int | None
    total_meta_seeds: int
    total_meta_karma_edges: int
    interval_seconds: int
    consecutive_failures: int = 0


# ═══════════════════════════════════════════════════════════
#  GuardianLoop
# ═══════════════════════════════════════════════════════════


class GuardianLoop:
    """守护循环 — 后台定时任务，驱动元种子体系

    职责:
      - 定期检查新增领域/关系类型 → 生成缺失的元种子
      - 更新所有元种子的监控指标
      - 根据指标变化触发元业力熏习
      - 提交所有数据库变更
      - 记录执行日志

    线程安全:
      - _loop_lock 保护单次执行串行化
      - 数据库操作由 SQLite WAL 模式保证并发安全
      - 与查询并发安全由 WAL 快照读保证

    守护线程模式参考 checkpoint.py 的 start_daemon / stop_daemon。

    Args:
        graph: 知识图谱连接
    """

    def __init__(self, graph: GraphDB) -> None:
        self._graph = graph
        self._meta_seed_mgr = MetaSeedManager(graph)
        self._loop_lock = threading.Lock()
        self._shutdown_event = threading.Event()
        self._daemon_thread: threading.Thread | None = None
        self._is_executing = False

        # 运行状态
        self._last_execution_time: str | None = None
        self._last_execution_result: str | None = None
        self._last_execution_duration_ms: int | None = None
        self._consecutive_failures: int = 0
        self._total_meta_seeds: int = 0
        self._total_meta_karma_edges: int = 0

        # Phase 5: 认知目标与好奇心引擎
        self._goal_mgr = None
        self._curiosity_engine = None
        if COGNITIVE_GOAL_ENABLED:
            try:
                from .cognitive_goal import CognitiveGoalManager
                self._goal_mgr = CognitiveGoalManager(graph)
            except Exception as e:
                log.warning("CognitiveGoalManager 初始化失败: %s", e)
        if CURIOSITY_ENGINE_ENABLED and self._goal_mgr is not None:
            try:
                from .curiosity_engine import CuriosityEngine
                self._curiosity_engine = CuriosityEngine(graph, self._goal_mgr)
            except Exception as e:
                log.warning("CuriosityEngine 初始化失败: %s", e)

    # ── 守护线程控制 ──────────────────────────────────

    def start(self) -> None:
        """启动守护线程

        在 GUARDIAN_LOOP_INITIAL_DELAY（默认 10）秒后执行首次循环，
        之后每 GUARDIAN_LOOP_INTERVAL（默认 60）秒执行一次。
        """
        if self._daemon_thread is not None and self._daemon_thread.is_alive():
            log.warning("guardian loop already running")
            return

        self._shutdown_event.clear()
        self._daemon_thread = threading.Thread(
            target=self._daemon_loop,
            name="guardian-loop",
            daemon=True,
        )
        self._daemon_thread.start()
        log.info(
            "guardian loop started, initial_delay=%ds, interval=%ds",
            GUARDIAN_LOOP_INITIAL_DELAY,
            GUARDIAN_LOOP_INTERVAL,
        )

    def stop(self) -> None:
        """优雅停止守护线程

        等待当前执行完成（最多 GUARDIAN_LOOP_TIMEOUT 秒），然后停止。
        """
        if self._daemon_thread is None or not self._daemon_thread.is_alive():
            return

        self._shutdown_event.set()
        self._daemon_thread.join(timeout=GUARDIAN_LOOP_TIMEOUT + 5.0)

        if self._daemon_thread.is_alive():
            log.warning("guardian loop did not stop within timeout")
        else:
            log.info("guardian loop stopped")

        self._daemon_thread = None

    def _daemon_loop(self) -> None:
        """守护线程主循环"""
        # 首次延迟
        self._shutdown_event.wait(timeout=GUARDIAN_LOOP_INITIAL_DELAY)

        while not self._shutdown_event.is_set():
            try:
                result = self.execute_once()
                if result.success:
                    self._consecutive_failures = 0
                else:
                    self._consecutive_failures += 1
            except Exception as e:
                log.error("guardian loop failed: %s", e)
                self._consecutive_failures += 1

            # 等待下一个间隔或收到关闭信号
            self._shutdown_event.wait(timeout=GUARDIAN_LOOP_INTERVAL)

    # ── 单次执行 ──────────────────────────────────────

    def execute_once(self) -> GuardianLoopResult:
        """执行一次守护循环

        Phase 5 扩展流程 (7 步):
          1. 检查新增领域/关系类型 → 生成缺失的元种子
          2. 更新所有元种子的监控指标
          3. 生成认知目标 [Phase 5 新增]
          4. 调度目标探索 [Phase 5 新增]
          5. 目标冷却 [Phase 5 新增]
          6. 根据指标变化触发元业力熏习
          7. 提交所有数据库变更

        Returns:
            GuardianLoopResult
        """
        if not META_SEED_ENABLED:
            return GuardianLoopResult(success=True, meta_seeds_updated=0)

        start_time = time.monotonic()
        meta_seeds_updated = 0
        meta_karma_edges_created = 0
        goals_generated = 0
        goals_explored = 0
        goals_cooled = 0
        curiosity_new_associations = 0

        with self._loop_lock:
            if self._is_executing:
                return GuardianLoopResult(
                    success=False, error="guardian loop already executing"
                )
            self._is_executing = True
            try:
                self._graph.conn.execute("BEGIN IMMEDIATE")
                # ① 检查新增领域/关系类型
                new_domain_seeds = self._meta_seed_mgr.generate_domain_monitors()
                new_relation_seeds = self._meta_seed_mgr.generate_relation_monitors()
                new_system_seeds = self._meta_seed_mgr.generate_system_monitors()
                meta_seeds_updated += new_domain_seeds + new_relation_seeds + new_system_seeds

                # 超时检测
                elapsed = time.monotonic() - start_time
                if elapsed > GUARDIAN_LOOP_TIMEOUT:
                    log.warning("guardian loop timeout after step 1, elapsed=%.1fs", elapsed)
                    return self._make_timeout_result(start_time)

                # ② 更新所有元种子指标
                domain_updated = self._check_domain_health()
                relation_updated = self._check_relation_quality()
                system_updated = self._check_system_metrics()
                boundary_updated = self._update_self_boundary()
                unknown_updated = self._detect_unknown_domains()
                meta_seeds_updated += (
                    domain_updated + relation_updated + system_updated
                    + boundary_updated + unknown_updated
                )

                # 超时检测
                elapsed = time.monotonic() - start_time
                if elapsed > GUARDIAN_LOOP_TIMEOUT:
                    log.warning("guardian loop timeout after step 2, elapsed=%.1fs", elapsed)
                    return self._make_timeout_result(start_time)

                # ③ 生成认知目标 [Phase 5 新增]
                if COGNITIVE_GOAL_ENABLED and self._goal_mgr is not None:
                    try:
                        goals_generated = self._goal_mgr.generate_goals()
                    except Exception as e:
                        log.warning("认知目标生成失败: %s", e)

                    # 超时检测
                    elapsed = time.monotonic() - start_time
                    if elapsed > GUARDIAN_LOOP_TIMEOUT:
                        return self._make_timeout_result(start_time)

                # ④ 调度目标探索 [Phase 5 新增]
                if COGNITIVE_GOAL_ENABLED and CURIOSITY_ENGINE_ENABLED and self._curiosity_engine is not None:
                    try:
                        exploration_result = self._schedule_goal_exploration()
                        goals_explored = exploration_result["goals_explored"]
                        curiosity_new_associations = exploration_result["new_associations"]
                    except Exception as e:
                        log.warning("目标探索调度失败: %s", e)

                # ⑤ 目标冷却 [Phase 5 新增]
                if COGNITIVE_GOAL_ENABLED and self._goal_mgr is not None:
                    try:
                        goals_cooled = self._goal_mgr.cool_goals()
                    except Exception as e:
                        log.warning("目标冷却失败: %s", e)

                # ⑥ 根据指标变化触发元业力熏习
                meta_karma_edges_created = self._meta_seed_mgr.check_and_create_meta_karma()

                # ⑦ 提交所有数据库变更
                self._graph.conn.commit()

                # ⑧ Phase 6: 多模态对齐检查 [Phase 6 新增]
                if PERCEPTION_ENABLED:
                    try:
                        self._check_multimodal_alignment()
                    except Exception as e:
                        log.warning("多模态对齐检查失败: %s", e)

                duration_ms = int((time.monotonic() - start_time) * 1000)

                # 更新运行状态
                now = datetime.now(timezone.utc).isoformat()
                self._last_execution_time = now
                self._last_execution_result = "success"
                self._last_execution_duration_ms = duration_ms

                # 更新统计
                self._update_stats()

                log.info(
                    "guardian loop completed: meta_seeds_updated=%d, "
                    "goals_generated=%d, goals_explored=%d, goals_cooled=%d, "
                    "meta_karma_edges_created=%d, curiosity_new_associations=%d, "
                    "duration_ms=%d",
                    meta_seeds_updated, goals_generated, goals_explored,
                    goals_cooled, meta_karma_edges_created,
                    curiosity_new_associations, duration_ms,
                )

                return GuardianLoopResult(
                    success=True,
                    meta_seeds_updated=meta_seeds_updated,
                    meta_karma_edges_created=meta_karma_edges_created,
                    duration_ms=duration_ms,
                )

            except Exception as e:
                duration_ms = int((time.monotonic() - start_time) * 1000)
                self._last_execution_time = datetime.now(timezone.utc).isoformat()
                self._last_execution_result = "failed"
                self._last_execution_duration_ms = duration_ms
                self._consecutive_failures += 1

                log.error("guardian loop failed: %s, duration_ms=%d", e, duration_ms)

                return GuardianLoopResult(
                    success=False,
                    error=str(e),
                    duration_ms=duration_ms,
                )

            finally:
                self._is_executing = False

    # ── 指标计算 ──────────────────────────────────────

    def _check_domain_health(self) -> int:
        """检查各领域健康度 — 更新领域监控元种子指标

        对每个 domain_monitor 类别的元种子:
          - avg_karma_density: 该领域所有种子的平均出边权重
          - ripple_success_rate: 最近窗口内该领域被选中的成功率
          - conflict_frequency: 不重新计算，仅读取 meta_seeds 表当前值

        Returns:
            更新的元种子数量
        """
        try:
            domain_seeds = self._meta_seed_mgr.list_meta_seeds(
                category=MetaSeedCategory.DOMAIN_MONITOR
            )
        except Exception as e:
            log.warning("领域监控元种子查询失败: %s", e)
            return 0

        if not domain_seeds:
            return 0

        # 预加载 param_stats 最近窗口数据
        recent_stats: list[dict] = []
        try:
            rows = self._graph.conn.execute(
                "SELECT selected_domains, confidence FROM param_stats "
                "ORDER BY created_at DESC LIMIT ?",
                (GUARDIAN_METRICS_WINDOW,),
            ).fetchall()
            recent_stats = [dict(r) for r in rows]
        except Exception as e:
            log.warning("param_stats 查询失败: %s", e)

        updated = 0
        for ms in domain_seeds:
            domain = ms.label.replace("meta:", "")
            metrics = dict(ms.metrics)

            # avg_karma_density: 该领域所有种子的平均出边权重
            try:
                row = self._graph.conn.execute(
                    "SELECT AVG(ke.weight) as avg_w "
                    "FROM karma_edges ke "
                    "JOIN seeds s ON ke.source = s.label "
                    "WHERE s.domain = ? AND s.type != 'META'",
                    (domain,),
                ).fetchone()
                metrics["avg_karma_density"] = round(row["avg_w"], 4) if row and row["avg_w"] is not None else 0.0
            except Exception as e:
                log.warning("avg_karma_density 计算失败: %s, domain=%s", e, domain)

            # ripple_success_rate: 最近窗口内该领域被选中的成功率
            selected_count = 0
            for stat in recent_stats:
                try:
                    domains = json.loads(stat["selected_domains"]) if stat["selected_domains"] else []
                except (json.JSONDecodeError, TypeError):
                    continue
                if domain in domains:
                    selected_count += 1
            metrics["ripple_success_rate"] = round(selected_count / len(recent_stats), 4) if recent_stats else 0.0

            # conflict_frequency: 不重新计算，保留当前值

            if self._meta_seed_mgr.update_metrics(ms.label, metrics):
                updated += 1

        return updated

    def _check_relation_quality(self) -> int:
        """检查关系质量 — 更新关系质量元种子指标

        对每个 relation_quality 类别的元种子:
          - avg_weight: 该关系类型边的平均权重
          - verification_count: 最近窗口内该关系类型边被校验的总次数

        Returns:
            更新的元种子数量
        """
        try:
            relation_seeds = self._meta_seed_mgr.list_meta_seeds(
                category=MetaSeedCategory.RELATION_QUALITY
            )
        except Exception as e:
            log.warning("关系质量元种子查询失败: %s", e)
            return 0

        if not relation_seeds:
            return 0

        updated = 0
        for ms in relation_seeds:
            relation_type = ms.label.replace("meta:", "")
            metrics = dict(ms.metrics)

            # avg_weight: 该关系类型边的平均权重
            try:
                row = self._graph.conn.execute(
                    "SELECT AVG(weight) as avg_w FROM karma_edges WHERE relation = ?",
                    (relation_type,),
                ).fetchone()
                metrics["avg_weight"] = round(row["avg_w"], 4) if row and row["avg_w"] is not None else 0.0
            except Exception as e:
                log.warning("avg_weight 计算失败: %s, relation=%s", e, relation_type)

            # verification_count: 最近窗口内该关系类型边被校验的总次数
            try:
                row = self._graph.conn.execute(
                    "SELECT COUNT(*) as cnt FROM param_stats "
                    "WHERE karma_direction != 0 AND created_at > datetime('now', '-1 day')"
                ).fetchone()
                metrics["verification_count"] = row["cnt"] if row else 0
            except Exception as e:
                log.warning("verification_count 计算失败: %s, relation=%s", e, relation_type)

            if self._meta_seed_mgr.update_metrics(ms.label, metrics):
                updated += 1

        return updated

    def _check_system_metrics(self) -> int:
        """检查系统级指标 — 更新系统级元种子指标

        从数据库获取系统统计数据:
          - meta:system_total_nodes → total_seeds
          - meta:system_total_edges → total_karma_edges
          - meta:system_avg_confidence → 平均置信度
          - meta:system_distillation_rate → 熏习速率
          - meta:system_seed_growth_rate → 种子增长率

        Returns:
            更新的元种子数量
        """
        updated = 0

        # meta:system_total_nodes
        try:
            row = self._graph.conn.execute(
                "SELECT COUNT(*) as cnt FROM seeds WHERE type != 'META'"
            ).fetchone()
            total_seeds = row["cnt"] if row else 0

            # 读取之前值计算 delta_24h
            prev = self._meta_seed_mgr.get_meta_seed("meta:system_total_nodes")
            prev_value = prev.metrics.get("value", 0) if prev else 0
            delta_24h = total_seeds - prev_value

            self._meta_seed_mgr.update_metrics("meta:system_total_nodes", {
                "value": total_seeds, "delta_24h": delta_24h,
            })
            updated += 1
        except Exception as e:
            log.warning("system_total_nodes 更新失败: %s", e)

        # meta:system_total_edges
        try:
            row = self._graph.conn.execute(
                "SELECT COUNT(*) as cnt FROM karma_edges WHERE source NOT LIKE 'meta:%'"
            ).fetchone()
            total_edges = row["cnt"] if row else 0

            prev = self._meta_seed_mgr.get_meta_seed("meta:system_total_edges")
            prev_value = prev.metrics.get("value", 0) if prev else 0
            delta_24h = total_edges - prev_value

            self._meta_seed_mgr.update_metrics("meta:system_total_edges", {
                "value": total_edges, "delta_24h": delta_24h,
            })
            updated += 1
        except Exception as e:
            log.warning("system_total_edges 更新失败: %s", e)

        # meta:system_avg_confidence
        try:
            row = self._graph.conn.execute(
                "SELECT AVG(confidence) as avg_c FROM param_stats "
                "WHERE created_at > datetime('now', '-1 day')"
            ).fetchone()
            avg_conf = round(row["avg_c"], 4) if row and row["avg_c"] is not None else 0.0

            prev = self._meta_seed_mgr.get_meta_seed("meta:system_avg_confidence")
            prev_value = prev.metrics.get("value", 0) if prev else 0
            delta_24h = round(avg_conf - prev_value, 4) if isinstance(prev_value, (int, float)) else 0

            self._meta_seed_mgr.update_metrics("meta:system_avg_confidence", {
                "value": avg_conf, "delta_24h": delta_24h,
            })
            updated += 1
        except Exception as e:
            log.warning("system_avg_confidence 更新失败: %s", e)

        # meta:system_distillation_rate
        try:
            row = self._graph.conn.execute(
                "SELECT COUNT(*) as cnt FROM param_stats "
                "WHERE karma_direction != 0 AND created_at > datetime('now', '-1 day')"
            ).fetchone()
            distill_count = row["cnt"] if row else 0

            prev = self._meta_seed_mgr.get_meta_seed("meta:system_distillation_rate")
            prev_value = prev.metrics.get("value", 0) if prev else 0
            delta_24h = distill_count - prev_value

            self._meta_seed_mgr.update_metrics("meta:system_distillation_rate", {
                "value": distill_count, "delta_24h": delta_24h,
            })
            updated += 1
        except Exception as e:
            log.warning("system_distillation_rate 更新失败: %s", e)

        # meta:system_seed_growth_rate
        try:
            row = self._graph.conn.execute(
                "SELECT COUNT(*) as cnt FROM seeds "
                "WHERE created_at > datetime('now', '-1 day') AND type != 'META'"
            ).fetchone()
            new_seeds = row["cnt"] if row else 0

            prev = self._meta_seed_mgr.get_meta_seed("meta:system_seed_growth_rate")
            prev_value = prev.metrics.get("value", 0) if prev else 0
            delta_24h = new_seeds - prev_value

            self._meta_seed_mgr.update_metrics("meta:system_seed_growth_rate", {
                "value": new_seeds, "delta_24h": delta_24h,
            })
            updated += 1
        except Exception as e:
            log.warning("system_seed_growth_rate 更新失败: %s", e)

        return updated

    def _update_self_boundary(self) -> int:
        """更新自边界 — 更新 meta:unknown 指标

        从 candidate_seeds 表中提取状态为 candidate 的记录。

        Returns:
            更新的元种子数量（0 或 1）
        """
        try:
            return self._meta_seed_mgr.update_self_boundary()
        except Exception as e:
            log.warning("自边界更新失败: %s", e)
            return 0

    def _detect_unknown_domains(self) -> int:
        """探测未知领域 — 检查低置信度高频区域

        从 param_stats 表中计算各领域低置信度频率。

        Returns:
            新创建或更新的元种子数量
        """
        try:
            return self._meta_seed_mgr.detect_unknown_domains()
        except Exception as e:
            log.warning("未知领域探测失败: %s", e)
            return 0

    # ── 目标探索调度 ──────────────────────────────────

    def _schedule_goal_exploration(self) -> dict:
        """调度目标探索

        获取 pending 目标按优先级降序，取前 GOAL_MAX_EXPLORE_PER_CYCLE 个，
        优先级 > GOAL_AUTO_EXPLORE_THRESHOLD 时触发好奇心引擎。

        Returns:
            {"goals_explored": int, "new_associations": int}
        """
        if self._curiosity_engine is None or self._goal_mgr is None:
            return {"goals_explored": 0, "new_associations": 0}

        goals_explored = 0
        new_associations = 0

        try:
            # 获取 pending 目标按优先级降序
            from .cognitive_goal import GoalStatus, CognitiveGoalData
            pending_goals = self._goal_mgr.list_goals(status=GoalStatus.PENDING)

            # 取前 GOAL_MAX_EXPLORE_PER_CYCLE 个
            candidates = pending_goals[:GOAL_MAX_EXPLORE_PER_CYCLE]

            for goal in candidates:
                if goal.priority_weight > GOAL_AUTO_EXPLORE_THRESHOLD:
                    result = self._curiosity_engine.explore(goal)
                    if not result.error:
                        goals_explored += 1
                        new_associations += result.new_associations
                    else:
                        log.warning(
                            "目标探索失败: goal_id=%s, strategy=%s, error=%s",
                            goal.goal_id, result.strategy, result.error,
                        )
        except Exception as e:
            log.warning("目标探索调度异常: %s", e)

        return {"goals_explored": goals_explored, "new_associations": new_associations}

    # ── 状态查询 ──────────────────────────────────────

    def get_status(self) -> GuardianLoopStatus:
        """查询守护循环运行状态"""
        return GuardianLoopStatus(
            is_running=self._daemon_thread is not None and self._daemon_thread.is_alive(),
            last_execution_time=self._last_execution_time,
            last_execution_result=self._last_execution_result,
            last_execution_duration_ms=self._last_execution_duration_ms,
            total_meta_seeds=self._total_meta_seeds,
            total_meta_karma_edges=self._total_meta_karma_edges,
            interval_seconds=GUARDIAN_LOOP_INTERVAL,
            consecutive_failures=self._consecutive_failures,
        )

    def _update_stats(self) -> None:
        """更新内部统计（元种子总数、元业力边总数）"""
        try:
            row = self._graph.conn.execute(
                "SELECT COUNT(*) as cnt FROM meta_seeds"
            ).fetchone()
            self._total_meta_seeds = row["cnt"] if row else 0
        except Exception:
            pass

        try:
            row = self._graph.conn.execute(
                "SELECT COUNT(*) as cnt FROM karma_edges WHERE source_tag = 'meta_karma'"
            ).fetchone()
            self._total_meta_karma_edges = row["cnt"] if row else 0
        except Exception:
            pass

    @property
    def is_executing(self) -> bool:
        """当前是否正在执行"""
        return self._is_executing

    # ── 内部方法 ──────────────────────────────────────

    def _make_timeout_result(self, start_time: float) -> GuardianLoopResult:
        """构造超时结果"""
        duration_ms = int((time.monotonic() - start_time) * 1000)
        self._last_execution_time = datetime.now(timezone.utc).isoformat()
        self._last_execution_result = "failed"
        self._last_execution_duration_ms = duration_ms
        self._consecutive_failures += 1

        return GuardianLoopResult(
            success=False,
            error="guardian loop timeout",
            duration_ms=duration_ms,
        )

    # ── Phase 6: 多模态对齐 ──────────────────────────

    def _check_multimodal_alignment(self) -> None:
        """检查是否需要触发多模态对齐任务

        距上次对齐超过 MULTIMODAL_ALIGNMENT_INTERVAL 秒时触发。
        """
        if not PERCEPTION_ENABLED or not MULTIMODAL_ALIGNMENT_ENABLED:
            return

        try:
            # 检查上次对齐时间
            row = self._graph.conn.execute(
                "SELECT MAX(timestamp) as last_run FROM perception_events "
                "WHERE channel = 'multimodal_alignment'"
            ).fetchone()

            last_run = row["last_run"] if row else None
            if last_run:
                try:
                    last_dt = datetime.fromisoformat(last_run)
                    now_dt = datetime.now(timezone.utc)
                    elapsed = (now_dt - last_dt).total_seconds()
                    if elapsed < MULTIMODAL_ALIGNMENT_INTERVAL:
                        return  # 间隔未到
                except (ValueError, TypeError):
                    pass  # 解析失败，继续执行

            # 触发对齐
            from consciousness_sea.perception.multimodal_aligner import MultimodalAligner
            aligner = MultimodalAligner(self._graph)
            results = aligner.run_alignment()
            if results:
                log.info("多模态对齐完成: %d 条结果", len(results))
        except Exception as e:
            log.warning("多模态对齐检查失败: %s", e)