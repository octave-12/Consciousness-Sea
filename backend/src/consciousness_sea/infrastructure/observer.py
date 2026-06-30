"""
可观测性模块 (Observer)

系统监控统计查询、告警检测、HTML 监控面板渲染。
覆盖 REQ-P1R-013 ~ REQ-P1R-021。
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, TypedDict

from .connection_pool import ConnectionPool

if TYPE_CHECKING:
    pass

from .config import (
    COGNITIVE_GOAL_ENABLED,
    CURIOSITY_ENGINE_ENABLED,
    DEFAULT_DB_PATH,
    GOAL_POOL_MAX_SIZE,
    KARMA_ALERT_THRESHOLD,
    META_ALERT_CONFLICT_THRESHOLD,
    META_SEED_ENABLED,
    PERCEPTION_CHANNEL_FAILURE_ALERT_THRESHOLD,
    PERCEPTION_ENABLED,
    STATUS_TOP_N,
)

log = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════
#  数据模型
# ═══════════════════════════════════════════════════════════


@dataclass
class SeedRankItem:
    """种子排名项 — 按出边数排名"""

    label: str
    edge_count: int

    def __str__(self) -> str:
        return f"{self.label} ({self.edge_count} 出边)"


@dataclass
class KarmaRankItem:
    """业力边排名项 — 按权重排名"""

    source: str
    target: str
    weight: float

    def __str__(self) -> str:
        return f"{self.source}↔{self.target}:{self.weight:.2f}"


@dataclass
class QueryRecord:
    """查询历史记录"""

    query_text: str
    selected_domains: list[str]
    confidence: float


class DistillationPoolStatus(TypedDict, total=False):
    total_candidates: int
    upgraded_count: int
    pending_count: int
    cooled_count: int

class AliasExpansionStatus(TypedDict, total=False):
    total_aliases: int
    by_domain: dict[str, int]

class CandidateSeedsStatus(TypedDict, total=False):
    total_candidates: int
    by_status: dict[str, int]

class LatestCheckpointStatus(TypedDict, total=False):
    checkpoint_id: str
    tag: str
    edge_count: int
    created_at: str
    source: str

class MetaSeedsStatus(TypedDict, total=False):
    total_meta_seeds: int
    by_category: dict[str, int]

class GuardianLoopStatus(TypedDict, total=False):
    running: bool
    last_cycle_time: str

class CognitiveGoalsStatus(TypedDict, total=False):
    total_goals: int
    by_status: dict[str, int]
    by_type: dict[str, int]
    pool_usage_percent: float

class CuriosityEngineStatus(TypedDict, total=False):
    total_explorations: int
    total_new_associations: int
    total_external_queries: int
    last_exploration_time: str
    is_exploring: bool

class PerceptionStatus(TypedDict, total=False):
    enabled: bool
    total_perceptual_seeds: int
    total_hebbian_bindings: int

@dataclass
class StatusData:
    """系统完整监控状态"""

    total_seeds: int
    total_karma_edges: int
    hottest_seeds: list[SeedRankItem]
    coldest_seeds: list[SeedRankItem]
    heaviest_karma: list[KarmaRankItem]
    recent_queries: list[QueryRecord]
    alerts: list[str]
    domain_distribution: dict[str, int] = field(default_factory=dict)
    db_size_mb: float = 0.0
    distillation_pool: DistillationPoolStatus | None = None
    alias_expansion: AliasExpansionStatus | None = None
    candidate_seeds: CandidateSeedsStatus | None = None
    latest_checkpoint: LatestCheckpointStatus | None = None
    meta_seeds: MetaSeedsStatus | None = None
    guardian_loop: GuardianLoopStatus | None = None
    cognitive_goals: CognitiveGoalsStatus | None = None
    curiosity_engine: CuriosityEngineStatus | None = None
    perception: PerceptionStatus | None = None


# ═══════════════════════════════════════════════════════════
#  Observer 类
# ═══════════════════════════════════════════════════════════


class Observer:
    """系统可观测性 — 统计查询 + 告警检测 + HTML 渲染"""

    def __init__(self, pool: ConnectionPool) -> None:
        """
        初始化 Observer。

        Args:
            pool: ConnectionPool 实例（由另一个开发者实现），
                  提供 acquire() / release() 方法获取/归还数据库连接。
        """
        self._pool = pool

    # ── 种子排名查询 ───────────────────────────────────────

    def get_hottest_seeds(self, limit: int = STATUS_TOP_N) -> list[SeedRankItem]:
        """
        最热种子：出边数 Top-N。

        SQL: SELECT source, COUNT(*) as cnt FROM karma_edges
             GROUP BY source ORDER BY cnt DESC LIMIT ?

        Args:
            limit: 返回数量上限

        Returns:
            按出边数降序排列的种子列表
        """
        graph = None
        try:
            graph = self._pool.acquire()
            rows = graph.conn.execute(
                "SELECT source AS label, COUNT(*) AS edge_count "
                "FROM karma_edges GROUP BY source "
                "ORDER BY edge_count DESC LIMIT ?",
                (limit,),
            ).fetchall()
            return [SeedRankItem(label=r["label"], edge_count=r["edge_count"]) for r in rows]
        except Exception as e:
            log.warning("查询最热种子失败: %s", e)
            return []
        finally:
            if graph is not None:
                self._pool.release(graph)

    def get_coldest_seeds(self, limit: int = STATUS_TOP_N) -> list[SeedRankItem]:
        """
        最冷种子：出边数 Bottom-N。

        SQL: SELECT source, COUNT(*) as cnt FROM karma_edges
             GROUP BY source ORDER BY cnt ASC LIMIT ?

        Args:
            limit: 返回数量上限

        Returns:
            按出边数升序排列的种子列表
        """
        graph = None
        try:
            graph = self._pool.acquire()
            rows = graph.conn.execute(
                "SELECT source AS label, COUNT(*) AS edge_count "
                "FROM karma_edges GROUP BY source "
                "ORDER BY edge_count ASC LIMIT ?",
                (limit,),
            ).fetchall()
            return [SeedRankItem(label=r["label"], edge_count=r["edge_count"]) for r in rows]
        except Exception as e:
            log.warning("查询最冷种子失败: %s", e)
            return []
        finally:
            if graph is not None:
                self._pool.release(graph)

    # ── 业力边排名查询 ─────────────────────────────────────

    def get_heaviest_karma(self, limit: int = STATUS_TOP_N) -> list[KarmaRankItem]:
        """
        最重业力边：权重 Top-N。

        SQL: SELECT source, target, weight FROM karma_edges
             ORDER BY weight DESC LIMIT ?

        Args:
            limit: 返回数量上限

        Returns:
            按权重降序排列的业力边列表
        """
        graph = None
        try:
            graph = self._pool.acquire()
            rows = graph.conn.execute(
                "SELECT source, target, weight FROM karma_edges "
                "ORDER BY weight DESC LIMIT ?",
                (limit,),
            ).fetchall()
            return [
                KarmaRankItem(source=r["source"], target=r["target"], weight=r["weight"])
                for r in rows
            ]
        except Exception as e:
            log.warning("查询最重业力边失败: %s", e)
            return []
        finally:
            if graph is not None:
                self._pool.release(graph)

    # ── 查询历史 ───────────────────────────────────────────

    def get_recent_queries(self, limit: int = STATUS_TOP_N) -> list[QueryRecord]:
        """
        最近查询记录。

        从 query_history 表读取最近 N 条记录。

        Args:
            limit: 返回数量上限

        Returns:
            最近查询记录列表
        """
        graph = None
        try:
            graph = self._pool.acquire()
            # 确保 query_history 表存在
            from consciousness_sea.domain.query_history import ensure_history_table
            ensure_history_table(graph.conn)

            rows = graph.conn.execute(
                "SELECT query_text, selected_domains, confidence "
                "FROM query_history ORDER BY created_at DESC LIMIT ?",
                (limit,),
            ).fetchall()

            records: list[QueryRecord] = []
            for r in rows:
                # 解析 selected_domains JSON
                try:
                    domains = json.loads(r["selected_domains"]) if r["selected_domains"] else []
                except (json.JSONDecodeError, TypeError):
                    domains = []
                records.append(
                    QueryRecord(
                        query_text=r["query_text"],
                        selected_domains=domains,
                        confidence=r["confidence"],
                    )
                )
            return records
        except Exception as e:
            log.warning("查询最近记录失败: %s", e)
            return []
        finally:
            if graph is not None:
                self._pool.release(graph)

    # ── 告警检测 ───────────────────────────────────────────

    def detect_alerts(self) -> list[str]:
        """
        告警检测。

        规则:
          - 业力边权重 > KARMA_ALERT_THRESHOLD → "权重异常高"
          - 可扩展其他规则

        Returns:
            告警信息字符串列表
        """
        graph = None
        try:
            graph = self._pool.acquire()
            rows = graph.conn.execute(
                "SELECT source, target, weight FROM karma_edges "
                "WHERE weight > ?",
                (KARMA_ALERT_THRESHOLD,),
            ).fetchall()
            alerts: list[str] = []
            for r in rows:
                alerts.append(
                    f"{r['source']}↔{r['target']} 权重异常高 ({r['weight']:.2f})"
                )

            # Phase 4: 元种子冲突告警
            if META_SEED_ENABLED:
                try:
                    meta_rows = graph.conn.execute(
                        "SELECT label, json_extract(metrics_json, '$.conflict_frequency') as freq "
                        "FROM meta_seeds "
                        "WHERE json_extract(metrics_json, '$.conflict_frequency') > ?",
                        (META_ALERT_CONFLICT_THRESHOLD,)
                    ).fetchall()
                    for r in meta_rows:
                        alerts.append(
                            f"领域 '{r['label'].replace('meta:', '')}' 冲突频率异常高: {r['freq']}"
                        )
                except Exception as e:
                    log.warning("元种子告警检测失败: %s", e)

            # Phase 5: 认知目标池使用率告警
            if COGNITIVE_GOAL_ENABLED:
                try:
                    active_row = graph.conn.execute(
                        "SELECT COUNT(*) as cnt FROM cognitive_goals "
                        "WHERE status IN ('pending', 'exploring', 'querying_external')"
                    ).fetchone()
                    active_count = active_row["cnt"] if active_row else 0
                    if active_count > GOAL_POOL_MAX_SIZE * 0.8:
                        alerts.append(
                            f"认知目标池使用率过高: {active_count}/{GOAL_POOL_MAX_SIZE}"
                        )
                except Exception as e:
                    log.warning("认知目标池告警检测失败: %s", e)

            # Phase 6: 感知通道连续采集失败告警
            if PERCEPTION_ENABLED:
                try:
                    # 检查各通道的连续失败次数
                    for channel in ("visual", "auditory", "somatic"):
                        fail_row = graph.conn.execute(
                            "SELECT COUNT(*) as cnt FROM perception_events "
                            "WHERE channel = ? AND processed = 0 "
                            "AND timestamp > datetime('now', '-5 minutes')",
                            (channel,),
                        ).fetchone()
                        fail_count = fail_row["cnt"] if fail_row else 0
                        if fail_count > PERCEPTION_CHANNEL_FAILURE_ALERT_THRESHOLD:
                            alerts.append(
                                f"感知通道 '{channel}' 连续采集失败: {fail_count} 次"
                            )
                except Exception as e:
                    log.warning("感知通道告警检测失败: %s", e)

            return alerts
        except Exception as e:
            log.warning("告警检测失败: %s", e)
            return [f"告警检测异常: {e}"]
        finally:
            if graph is not None:
                self._pool.release(graph)

    # ── 聚合状态 ───────────────────────────────────────────

    def get_status(self) -> StatusData:
        """
        获取完整监控状态。

        聚合以上全部统计 + total_seeds / total_karma_edges +
        domain_distribution / db_size_mb。

        Returns:
            StatusData 包含全部统计信息
        """
        graph = None
        try:
            graph = self._pool.acquire()
            # 合并 COUNT 查询：种子总数 + 业力边总数
            count_row = graph.conn.execute(
                "SELECT (SELECT COUNT(*) FROM seeds) AS total_seeds, "
                "(SELECT COUNT(*) FROM karma_edges) AS total_karma_edges"
            ).fetchone()
            total_seeds = count_row["total_seeds"]
            total_karma_edges = count_row["total_karma_edges"]

            # 领域分布
            domain_distribution: dict[str, int] = {}
            domain_rows = graph.conn.execute(
                "SELECT domain, COUNT(*) AS cnt FROM seeds "
                "GROUP BY domain ORDER BY cnt DESC"
            ).fetchall()
            for r in domain_rows:
                domain_distribution[r["domain"] or "未分类"] = r["cnt"]

            # 数据库文件大小
            db_path = Path(DEFAULT_DB_PATH)
            db_size_mb = db_path.stat().st_size / (1024**2) if db_path.exists() else 0.0

            # 聚合各维度统计（每个方法内部独立获取/归还连接）
            hottest_seeds = self.get_hottest_seeds()
            coldest_seeds = self.get_coldest_seeds()
            heaviest_karma = self.get_heaviest_karma()
            recent_queries = self.get_recent_queries()
            alerts = self.detect_alerts()

            # Phase 2: 提炼池状态
            distillation_pool_status = None
            try:
                from consciousness_sea.learning.distillation_pool import DistillationPool
                distill = DistillationPool(graph)
                distillation_pool_status = distill.get_status()
            except Exception as e:
                log.warning("获取提炼池状态失败: %s", e)

            # Phase 3: 别名扩展状态
            alias_expansion_status = None
            try:
                from consciousness_sea.learning.alias_expander import AliasExpander
                expander = AliasExpander(graph)
                alias_expansion_status = expander.get_alias_stats()
            except Exception as e:
                log.warning("获取别名扩展状态失败: %s", e)

            # Phase 3: 候选种子状态
            candidate_seeds_status = None
            try:
                from consciousness_sea.learning.seed_candidate import SeedCandidateManager
                manager = SeedCandidateManager(graph)
                candidate_seeds_status = manager.get_status()
            except Exception as e:
                log.warning("获取候选种子状态失败: %s", e)

            # Phase 3: 最新检查点
            latest_checkpoint_status = None
            try:
                from consciousness_sea.learning.checkpoint import CheckpointManager
                cp_manager = CheckpointManager(graph)
                checkpoints = cp_manager.list_checkpoints(limit=1)
                if checkpoints:
                    cp = checkpoints[0]
                    latest_checkpoint_status = {
                        "checkpoint_id": cp.checkpoint_id,
                        "tag": cp.tag,
                        "edge_count": cp.edge_count,
                        "created_at": cp.created_at,
                        "source": cp.source.value if hasattr(cp.source, 'value') else cp.source,
                    }
            except Exception as e:
                log.warning("获取最新检查点失败: %s", e)

            # Phase 4: 元种子状态
            meta_seeds_status = None
            guardian_loop_status = None
            if META_SEED_ENABLED:
                try:
                    from consciousness_sea.metacognition.meta_seed import (
                        MetaSeedCategory,
                        MetaSeedManager,
                    )
                    mgr = MetaSeedManager(graph)
                    all_seeds = mgr.list_meta_seeds()
                    by_category: dict[str, int] = {}
                    for cat in MetaSeedCategory:
                        by_category[cat.value] = 0
                    for ms in all_seeds:
                        by_category[ms.category.value] = by_category.get(ms.category.value, 0) + 1
                    meta_seeds_status = {
                        "total_meta_seeds": len(all_seeds),
                        "by_category": by_category,
                    }
                except Exception as e:
                    log.warning("获取元种子状态失败: %s", e)

                try:
                    # 守护循环状态需要从 api.py 的全局实例获取
                    # 这里先返回 None，由 api.py 的 _status_to_dict 补充
                    pass
                except Exception as e:
                    log.warning("获取守护循环状态失败: %s", e)

            # Phase 5: 认知目标状态
            cognitive_goals_status = None
            if COGNITIVE_GOAL_ENABLED:
                try:
                    from consciousness_sea.metacognition.cognitive_goal import CognitiveGoalManager
                    goal_mgr = CognitiveGoalManager(graph)
                    stats = goal_mgr.get_goal_stats()
                    cognitive_goals_status = {
                        "total_goals": sum(stats.get("by_status", {}).values()),
                        "by_status": stats.get("by_status", {}),
                        "by_type": stats.get("by_type", {}),
                        "pool_usage_percent": stats.get("pool_usage", {}).get("usage_percent", 0.0),
                    }
                except Exception as e:
                    log.warning("获取认知目标状态失败: %s", e)

            # Phase 5: 好奇心引擎状态
            curiosity_engine_status = None
            if CURIOSITY_ENGINE_ENABLED:
                try:
                    from consciousness_sea.metacognition.cognitive_goal import CognitiveGoalManager
                    from consciousness_sea.metacognition.curiosity_engine import CuriosityEngine
                    goal_mgr = CognitiveGoalManager(graph)
                    engine = CuriosityEngine(graph, goal_mgr)
                    status = engine.get_status()
                    curiosity_engine_status = {
                        "total_explorations": status.total_explorations,
                        "total_new_associations": status.total_new_associations,
                        "total_external_queries": status.total_external_queries,
                        "last_exploration_time": status.last_exploration_time,
                        "is_exploring": status.is_exploring,
                    }
                except Exception as e:
                    log.warning("获取好奇心引擎状态失败: %s", e)

            # Phase 6: 感知状态
            perception_status = None
            if PERCEPTION_ENABLED:
                try:
                    # 感知元种子总数
                    total_perceptual_seeds = 0
                    try:
                        ps_row = graph.conn.execute(
                            "SELECT COUNT(*) as cnt FROM perceptual_seeds WHERE status = 'active'"
                        ).fetchone()
                        total_perceptual_seeds = ps_row["cnt"] if ps_row else 0
                    except Exception as _e:
                        log.debug("perceptual_seeds count failed", _e)

                    # Hebbian 绑定边总数
                    total_hebbian_bindings = 0
                    try:
                        hb_row = graph.conn.execute(
                            "SELECT COUNT(*) as cnt FROM karma_edges WHERE source_tag = 'hebbian_binding'"
                        ).fetchone()
                        total_hebbian_bindings = hb_row["cnt"] if hb_row else 0
                    except Exception as _e:
                        log.debug("hebbian bindings count failed", _e)

                    perception_status = {
                        "enabled": True,
                        "total_perceptual_seeds": total_perceptual_seeds,
                        "total_hebbian_bindings": total_hebbian_bindings,
                    }
                except Exception as e:
                    log.warning("获取感知状态失败: %s", e)
        except Exception as e:
            log.error("获取监控状态失败: %s", e)
            total_seeds = 0
            total_karma_edges = 0
            domain_distribution = {}
            db_size_mb = 0.0
            hottest_seeds = []
            coldest_seeds = []
            heaviest_karma = []
            recent_queries = []
            alerts = []
            distillation_pool_status = None
            alias_expansion_status = None
            candidate_seeds_status = None
            latest_checkpoint_status = None
            meta_seeds_status = None
            guardian_loop_status = None
            cognitive_goals_status = None
            curiosity_engine_status = None
            perception_status = None
        finally:
            if graph is not None:
                self._pool.release(graph)

        return StatusData(
            total_seeds=total_seeds,
            total_karma_edges=total_karma_edges,
            hottest_seeds=hottest_seeds,
            coldest_seeds=coldest_seeds,
            heaviest_karma=heaviest_karma,
            recent_queries=recent_queries,
            alerts=alerts,
            domain_distribution=domain_distribution,
            db_size_mb=round(db_size_mb, 2),
            distillation_pool=distillation_pool_status,
            alias_expansion=alias_expansion_status,
            candidate_seeds=candidate_seeds_status,
            latest_checkpoint=latest_checkpoint_status,
            meta_seeds=meta_seeds_status,
            guardian_loop=guardian_loop_status,
            cognitive_goals=cognitive_goals_status,
            curiosity_engine=curiosity_engine_status,
            perception=perception_status,
        )

    # ── HTML 渲染 ──────────────────────────────────────────

    def render_html(self, status: StatusData) -> str:
        """
        渲染 HTML 监控页面。

        纯标准库 f-string 拼装，内联 CSS，无外部依赖。
        深色主题，与识海项目风格一致。
        自动刷新 5 秒。

        Args:
            status: 系统监控状态数据

        Returns:
            完整 HTML 页面字符串
        """
        # 格式化最热种子表格行
        hottest_rows_html = ""
        for i, seed in enumerate(status.hottest_seeds, 1):
            hottest_rows_html += (
                f"<tr><td>{i}</td><td>{_escape_html(seed.label)}</td>"
                f"<td>{seed.edge_count:,}</td></tr>\n"
            )
        if not status.hottest_seeds:
            hottest_rows_html = '<tr><td colspan="3" class="empty">暂无数据</td></tr>'

        # 格式化最冷种子表格行
        coldest_rows_html = ""
        for i, seed in enumerate(status.coldest_seeds, 1):
            coldest_rows_html += (
                f"<tr><td>{i}</td><td>{_escape_html(seed.label)}</td>"
                f"<td>{seed.edge_count:,}</td></tr>\n"
            )
        if not status.coldest_seeds:
            coldest_rows_html = '<tr><td colspan="3" class="empty">暂无数据</td></tr>'

        # 格式化最重业力边表格行
        karma_rows_html = ""
        for i, edge in enumerate(status.heaviest_karma, 1):
            karma_rows_html += (
                f"<tr><td>{i}</td><td>{_escape_html(str(edge))}</td>"
                f"<td>{edge.weight:.4f}</td></tr>\n"
            )
        if not status.heaviest_karma:
            karma_rows_html = '<tr><td colspan="3" class="empty">暂无数据</td></tr>'

        # 格式化最近查询列表
        queries_html = ""
        for i, q in enumerate(status.recent_queries, 1):
            domains_str = ", ".join(q.selected_domains) if q.selected_domains else "—"
            queries_html += (
                f'<div class="query-item">'
                f'<span class="query-num">{i}.</span> '
                f'<span class="query-text">"{_escape_html(q.query_text)}"</span> '
                f'<span class="query-domains">[{_escape_html(domains_str)}]</span> '
                f'<span class="query-confidence">置信度: {q.confidence:.2f}</span>'
                f"</div>\n"
            )
        if not status.recent_queries:
            queries_html = '<div class="empty">暂无查询记录</div>'

        # 格式化告警区域
        alerts_html = ""
        for alert in status.alerts:
            alerts_html += f'<div class="alert-item">{_escape_html(alert)}</div>\n'
        if not status.alerts:
            alerts_html = '<div class="alert-ok">✓ 系统运行正常，无告警</div>'

        # 格式化提炼池状态（Phase 2）
        distill_html = ""
        if status.distillation_pool:
            dp = status.distillation_pool
            distill_html = (
                f'<div class="domain-item"><span>总候选数</span><span>{dp.get("total_candidates", 0):,}</span></div>\n'
                f'<div class="domain-item"><span>已升级</span><span>{dp.get("upgraded_count", 0):,}</span></div>\n'
                f'<div class="domain-item"><span>待审核</span><span>{dp.get("pending_count", 0):,}</span></div>\n'
                f'<div class="domain-item"><span>已冷却</span><span>{dp.get("cooled_count", 0):,}</span></div>\n'
            )
        else:
            distill_html = '<div class="empty">暂无数据</div>'

        # 领域分布（Top-10）
        domain_items_html = ""
        domain_top = list(status.domain_distribution.items())[:STATUS_TOP_N]
        for domain, count in domain_top:
            domain_items_html += (
                f'<div class="domain-item">'
                f"<span>{_escape_html(domain)}</span>"
                f"<span>{count:,}</span>"
                f"</div>\n"
            )

        html = f"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <meta http-equiv="refresh" content="5">
    <title>识海监控面板 — Consciousness Sea Dashboard</title>
    <style>
        /* ── 全局重置与基础 ─────────────────────────── */
        *, *::before, *::after {{
            box-sizing: border-box;
            margin: 0;
            padding: 0;
        }}

        body {{
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto,
                         "Noto Sans SC", sans-serif;
            background: #0a0e17;
            color: #c8d6e5;
            line-height: 1.6;
            padding: 20px;
            min-height: 100vh;
        }}

        /* ── 页面头部 ─────────────────────────────── */
        .header {{
            text-align: center;
            padding: 24px 0 16px;
            border-bottom: 1px solid #1e2a3a;
            margin-bottom: 24px;
        }}

        .header h1 {{
            font-size: 1.8rem;
            color: #54a0ff;
            font-weight: 600;
            letter-spacing: 0.5px;
        }}

        .header .subtitle {{
            color: #576574;
            font-size: 0.85rem;
            margin-top: 4px;
        }}

        /* ── 统计卡片 ─────────────────────────────── */
        .stats-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 16px;
            margin-bottom: 28px;
        }}

        .stat-card {{
            background: #111927;
            border: 1px solid #1e2a3a;
            border-radius: 8px;
            padding: 20px;
            text-align: center;
            transition: border-color 0.2s;
        }}

        .stat-card:hover {{
            border-color: #54a0ff;
        }}

        .stat-card .stat-value {{
            font-size: 2rem;
            font-weight: 700;
            color: #48dbfb;
        }}

        .stat-card .stat-label {{
            font-size: 0.85rem;
            color: #576574;
            margin-top: 4px;
        }}

        /* ── 区块容器 ─────────────────────────────── */
        .section {{
            background: #111927;
            border: 1px solid #1e2a3a;
            border-radius: 8px;
            margin-bottom: 20px;
            overflow: hidden;
        }}

        .section-header {{
            padding: 14px 20px;
            border-bottom: 1px solid #1e2a3a;
            font-size: 1rem;
            font-weight: 600;
            color: #f5f6fa;
            display: flex;
            align-items: center;
            gap: 8px;
        }}

        .section-header .icon {{
            font-size: 1.1rem;
        }}

        .section-body {{
            padding: 16px 20px;
        }}

        /* ── 表格 ─────────────────────────────────── */
        table {{
            width: 100%;
            border-collapse: collapse;
        }}

        th, td {{
            padding: 10px 14px;
            text-align: left;
            border-bottom: 1px solid #1a2332;
        }}

        th {{
            color: #8395a7;
            font-weight: 500;
            font-size: 0.85rem;
            text-transform: uppercase;
            letter-spacing: 0.5px;
        }}

        td {{
            color: #c8d6e5;
        }}

        tr:hover td {{
            background: #0d1520;
        }}

        td.num {{
            color: #48dbfb;
            font-weight: 600;
            font-variant-numeric: tabular-nums;
        }}

        .empty {{
            text-align: center;
            color: #576574;
            padding: 24px;
        }}

        /* ── 查询列表 ─────────────────────────────── */
        .query-item {{
            padding: 8px 0;
            border-bottom: 1px solid #1a2332;
        }}

        .query-item:last-child {{
            border-bottom: none;
        }}

        .query-num {{
            color: #576574;
            font-weight: 600;
            min-width: 28px;
            display: inline-block;
        }}

        .query-text {{
            color: #f5f6fa;
            font-weight: 500;
        }}

        .query-domains {{
            color: #54a0ff;
            font-size: 0.9rem;
        }}

        .query-confidence {{
            color: #576574;
            font-size: 0.85rem;
            float: right;
        }}

        /* ── 告警区域 ─────────────────────────────── */
        .alert-item {{
            padding: 10px 14px;
            margin-bottom: 8px;
            background: #2d1b1b;
            border-left: 3px solid #ee5a24;
            border-radius: 4px;
            color: #ff9f43;
            font-size: 0.9rem;
        }}

        .alert-ok {{
            color: #2ed573;
            padding: 10px 14px;
        }}

        /* ── 领域分布 ─────────────────────────────── */
        .domain-item {{
            display: flex;
            justify-content: space-between;
            padding: 6px 0;
            border-bottom: 1px solid #1a2332;
        }}

        .domain-item:last-child {{
            border-bottom: none;
        }}

        .domain-item span:last-child {{
            color: #48dbfb;
            font-variant-numeric: tabular-nums;
        }}

        /* ── 双栏布局 ─────────────────────────────── */
        .two-col {{
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 20px;
        }}

        @media (max-width: 768px) {{
            .two-col {{
                grid-template-columns: 1fr;
            }}

            .stats-grid {{
                grid-template-columns: repeat(2, 1fr);
            }}
        }}

        /* ── 页脚 ─────────────────────────────────── */
        .footer {{
            text-align: center;
            color: #576574;
            font-size: 0.8rem;
            margin-top: 28px;
            padding-top: 16px;
            border-top: 1px solid #1e2a3a;
        }}
    </style>
</head>
<body>

    <!-- 页面头部 -->
    <div class="header">
        <h1>🌊 识海监控面板</h1>
        <div class="subtitle">Consciousness Sea Dashboard — 自动刷新 5s</div>
    </div>

    <!-- 全局统计卡片 -->
    <div class="stats-grid">
        <div class="stat-card">
            <div class="stat-value">{status.total_seeds:,}</div>
            <div class="stat-label">种子总数</div>
        </div>
        <div class="stat-card">
            <div class="stat-value">{status.total_karma_edges:,}</div>
            <div class="stat-label">业力边总数</div>
        </div>
        <div class="stat-card">
            <div class="stat-value">{status.db_size_mb:.2f}</div>
            <div class="stat-label">数据库大小 (MB)</div>
        </div>
        <div class="stat-card">
            <div class="stat-value">{len(status.alerts)}</div>
            <div class="stat-label">活跃告警</div>
        </div>
    </div>

    <!-- 最热 / 最冷种子 -->
    <div class="two-col">
        <div class="section">
            <div class="section-header">
                <span class="icon">🔥</span> 最热种子 Top-{STATUS_TOP_N}
            </div>
            <div class="section-body">
                <table>
                    <thead><tr><th>#</th><th>种子</th><th>出边数</th></tr></thead>
                    <tbody>
                        {hottest_rows_html}
                    </tbody>
                </table>
            </div>
        </div>

        <div class="section">
            <div class="section-header">
                <span class="icon">❄️</span> 最冷种子 Top-{STATUS_TOP_N}
            </div>
            <div class="section-body">
                <table>
                    <thead><tr><th>#</th><th>种子</th><th>出边数</th></tr></thead>
                    <tbody>
                        {coldest_rows_html}
                    </tbody>
                </table>
            </div>
        </div>
    </div>

    <!-- 最重业力边 -->
    <div class="section">
        <div class="section-header">
            <span class="icon">⚖️</span> 最重业力边 Top-{STATUS_TOP_N}
        </div>
        <div class="section-body">
            <table>
                <thead><tr><th>#</th><th>业力边</th><th>权重</th></tr></thead>
                <tbody>
                    {karma_rows_html}
                </tbody>
            </table>
        </div>
    </div>

    <!-- 最近查询 + 领域分布 -->
    <div class="two-col">
        <div class="section">
            <div class="section-header">
                <span class="icon">📋</span> 最近查询
            </div>
            <div class="section-body">
                {queries_html}
            </div>
        </div>

        <div class="section">
            <div class="section-header">
                <span class="icon">📊</span> 领域分布 Top-{STATUS_TOP_N}
            </div>
            <div class="section-body">
                {domain_items_html}
            </div>
        </div>
    </div>

    <!-- 告警信息 -->
    <div class="section">
        <div class="section-header">
            <span class="icon">⚠️</span> 告警信息
        </div>
        <div class="section-body">
            {alerts_html}
        </div>
    </div>

    <!-- 提炼池状态（Phase 2） -->
    <div class="section">
        <div class="section-header">
            <span class="icon">🧪</span> 提炼池状态
        </div>
        <div class="section-body">
            {distill_html}
        </div>
    </div>

    <!-- 页脚 -->
    <div class="footer">
        识海 Consciousness Sea — 可观测性监控面板 | 阈值: KARMA_ALERT_THRESHOLD={KARMA_ALERT_THRESHOLD}
    </div>

</body>
</html>"""
        return html


# ═══════════════════════════════════════════════════════════
#  辅助函数
# ═══════════════════════════════════════════════════════════


def _escape_html(text: str) -> str:
    """
    转义 HTML 特殊字符，防止 XSS。

    Args:
        text: 原始文本

    Returns:
        转义后的安全文本
    """
    return (
        text.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
        .replace("'", "&#39;")
    )
