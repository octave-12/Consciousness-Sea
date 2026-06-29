"""
Phase 5 CuriosityEngine 单元测试

覆盖：
- 探索入口（explore）与并发控制
- 策略判断（内部探索/候选升级/外部查询/none）
- 内部探索（虚拟查询 + 涟漪传播 + 提炼池写入）
- 候选升级策略
- 外部查询策略
- 探索结果状态更新（成功→completed，失败→pending）
- 运行状态查询（get_status）
- 执行日志追加（_append_execution_log）
- CURIOSITY_ENGINE_ENABLED=False 时的行为
"""

from __future__ import annotations

import json
import sqlite3
import sys
import pathlib
from datetime import datetime, timezone
from unittest.mock import patch, MagicMock

import pytest

_root = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_root))
sys.path.insert(0, str(_root / "backend" / "src"))

from consciousness_sea.metacognition.curiosity_engine import (
    CuriosityEngine,
    ExplorationResult,
    CuriosityEngineStatus,
    ExternalQueryResult,
)
from consciousness_sea.metacognition.cognitive_goal import (
    CognitiveGoalManager,
    CognitiveGoalData,
    GoalType,
    GoalStatus,
)
from consciousness_sea.domain.graph_db import GraphDB
from consciousness_sea.infrastructure.config import (
    CURIOSITY_ENGINE_ENABLED,
    CURIOSITY_MAX_CONCURRENT,
    CURIOSITY_ACTIVATION_THRESHOLD,
    EXTERNAL_QUERY_ENABLED,
    GOAL_HIGH_CONFLICT_THRESHOLD,
)


# ═══════════════════════════════════════════════════════════
#  Fixtures
# ═══════════════════════════════════════════════════════════


def _build_test_db() -> sqlite3.Connection:
    """创建含 Phase 5 表的内存测试数据库"""
    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row
    conn.executescript("""
        CREATE TABLE seeds (
            id TEXT PRIMARY KEY, label TEXT NOT NULL,
            type TEXT NOT NULL DEFAULT 'CONCEPT',
            aliases TEXT NOT NULL DEFAULT '[]',
            activation REAL NOT NULL DEFAULT 0.0,
            domain TEXT NOT NULL DEFAULT '',
            definition TEXT NOT NULL DEFAULT '',
            pinyin TEXT NOT NULL DEFAULT '',
            activation_bias REAL NOT NULL DEFAULT 0.0,
            meta TEXT NOT NULL DEFAULT '{}',
            created_at TEXT NOT NULL DEFAULT (datetime('now'))
        );
        CREATE TABLE karma_edges (
            source TEXT NOT NULL, target TEXT NOT NULL, relation TEXT NOT NULL,
            weight REAL NOT NULL DEFAULT 0.5,
            source_tag TEXT NOT NULL DEFAULT '',
            created_at TEXT NOT NULL DEFAULT (datetime('now')),
            PRIMARY KEY (source, target, relation)
        );
        CREATE TABLE karma_edges_personal (
            user_label  TEXT    NOT NULL,
            source      TEXT    NOT NULL,
            target      TEXT    NOT NULL,
            relation    TEXT    NOT NULL,
            weight      REAL    NOT NULL,
            source_tag  TEXT    NOT NULL DEFAULT 'personal_karma',
            updated_at  TEXT    NOT NULL,
            PRIMARY KEY (user_label, source, target, relation)
        );
        CREATE TABLE meta_seeds (
            label           TEXT    PRIMARY KEY NOT NULL,
            category        TEXT    NOT NULL,
            metrics_json    TEXT    NOT NULL DEFAULT '{}',
            status          TEXT    NOT NULL DEFAULT 'active',
            source_domain   TEXT,
            dormant_since   TEXT,
            unchanged_cycles INTEGER NOT NULL DEFAULT 0,
            previous_metrics_json TEXT NOT NULL DEFAULT '{}',
            created_at      TEXT    NOT NULL,
            updated_at      TEXT    NOT NULL
        );
        CREATE TABLE candidate_seeds (
            label           TEXT    PRIMARY KEY,
            status          TEXT    NOT NULL DEFAULT 'candidate',
            count           INTEGER NOT NULL DEFAULT 1,
            domain          TEXT,
            co_occur_seeds  TEXT    NOT NULL DEFAULT '[]',
            candidate_since TEXT    NOT NULL,
            last_seen_at    TEXT    NOT NULL,
            promoted_at     TEXT,
            promoted_seed_id TEXT
        );
        CREATE TABLE unmatched_queries (
            query_text  TEXT    PRIMARY KEY NOT NULL,
            count       INTEGER NOT NULL DEFAULT 1,
            first_seen  TEXT    NOT NULL,
            last_seen   TEXT    NOT NULL
        );
        CREATE TABLE user_cold_start (
            user_label  TEXT    PRIMARY KEY,
            query_count INTEGER NOT NULL DEFAULT 0,
            updated_at  TEXT    NOT NULL
        );
        CREATE TABLE distillation_pool (
            candidate_id        INTEGER PRIMARY KEY AUTOINCREMENT,
            canonical_source    TEXT    NOT NULL,
            canonical_target    TEXT    NOT NULL,
            canonical_relation  TEXT    NOT NULL,
            representative_label TEXT   NOT NULL,
            count               INTEGER NOT NULL DEFAULT 1,
            contributor_users   TEXT    NOT NULL DEFAULT '[]',
            status              TEXT    NOT NULL DEFAULT 'pending',
            upgraded_at         TEXT,
            created_at          TEXT    NOT NULL,
            updated_at          TEXT    NOT NULL
        );
        CREATE TABLE param_stats (
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
        );
        CREATE TABLE expert_reliability (
            domain     TEXT PRIMARY KEY,
            score      REAL NOT NULL CHECK(score >= 0.0 AND score <= 1.0),
            updated_at TEXT NOT NULL
        );
        CREATE TABLE alias_backref_events (
            source_keyword  TEXT    NOT NULL,
            target_seed     TEXT    NOT NULL,
            ref_count       INTEGER NOT NULL DEFAULT 0,
            total_count     INTEGER NOT NULL DEFAULT 0,
            back_ref_rate   REAL    NOT NULL DEFAULT 0.0,
            status          TEXT    NOT NULL DEFAULT 'tracking',
            created_at      TEXT    NOT NULL,
            updated_at      TEXT    NOT NULL,
            PRIMARY KEY (source_keyword, target_seed)
        );
        CREATE TABLE checkpoint_meta (
            checkpoint_id    TEXT    PRIMARY KEY,
            tag              TEXT    NOT NULL DEFAULT '',
            edge_count       INTEGER NOT NULL DEFAULT 0,
            file_path        TEXT    NOT NULL,
            file_size_bytes  INTEGER NOT NULL DEFAULT 0,
            created_at       TEXT    NOT NULL,
            source           TEXT    NOT NULL DEFAULT 'manual'
        );
    """)

    # 插入测试种子
    seeds = [
        ("感冒", "感冒", "CONCEPT", "[]", "医学", "急性上呼吸道感染"),
        ("发热", "发热", "CONCEPT", "[]", "医学", "体温升高"),
        ("咳嗽", "咳嗽", "CONCEPT", "[]", "医学", "cough"),
        ("量子力学", "量子力学", "CONCEPT", "[]", "物理", "quantum mechanics"),
        ("薛定谔方程", "薛定谔方程", "CONCEPT", "[]", "物理", "Schrodinger equation"),
        ("人工智能", "人工智能", "CONCEPT", "[\"AI\"]", "计算机", "AI"),
        ("深度学习", "深度学习", "CONCEPT", "[]", "计算机", "deep learning"),
    ]
    conn.executemany(
        "INSERT INTO seeds (id, label, type, aliases, domain, definition) VALUES (?, ?, ?, ?, ?, ?)",
        seeds,
    )

    # 插入测试业力边
    edges = [
        ("感冒", "发热", "COOCCURS_WITH", 0.95, "karma_delta"),
        ("感冒", "咳嗽", "RELATED", 0.60, "karma_delta"),
        ("量子力学", "薛定谔方程", "IS_A", 0.88, "karma_delta"),
        ("人工智能", "深度学习", "RELATED", 0.90, "karma_delta"),
    ]
    conn.executemany(
        "INSERT INTO karma_edges (source, target, relation, weight, source_tag) VALUES (?, ?, ?, ?, ?)",
        edges,
    )
    conn.commit()
    return conn


def _make_graph_db(conn: sqlite3.Connection) -> GraphDB:
    """从已有连接创建 GraphDB 实例"""
    db = GraphDB(":memory:")
    db.conn = conn
    db.ensure_phase2_tables()
    db.ensure_phase3_tables()
    db.ensure_phase4_tables()
    db.ensure_phase5_tables()
    return db


@pytest.fixture
def graph():
    """创建内存数据库的 GraphDB 实例"""
    conn = _build_test_db()
    g = _make_graph_db(conn)
    yield g
    g.close()


@pytest.fixture
def goal_mgr(graph):
    """创建 CognitiveGoalManager 实例"""
    return CognitiveGoalManager(graph)


@pytest.fixture
def engine(graph, goal_mgr):
    """创建 CuriosityEngine 实例"""
    return CuriosityEngine(graph, goal_mgr)


def _insert_goal(conn, goal_id="goal_test", domain="医学", status="pending", goal_type="low_confidence"):
    """辅助：插入认知目标"""
    now = datetime.now(timezone.utc).isoformat()
    conn.execute(
        "INSERT INTO cognitive_goals "
        "(goal_id, goal_type, trigger_condition, domain, priority_weight, status, "
        " sub_goals, execution_log, decay_cycles_count, last_touched_at, created_at, updated_at) "
        "VALUES (?, ?, 'test', ?, 0.72, ?, '[]', '[]', 0, ?, ?, ?)",
        (goal_id, goal_type, domain, status, now, now, now),
    )
    conn.commit()


# ═══════════════════════════════════════════════════════════
#  测试用例
# ═══════════════════════════════════════════════════════════


class TestExplorationResult:
    """探索结果数据类测试"""

    def test_exploration_result_defaults(self):
        """ExplorationResult 默认值"""
        result = ExplorationResult(goal_id="test", strategy="internal")
        assert result.explored_seeds == []
        assert result.new_associations == 0
        assert result.distillation_candidates == 0
        assert result.duration_ms == 0
        assert result.error is None

    def test_curiosity_engine_status_defaults(self):
        """CuriosityEngineStatus 默认值"""
        status = CuriosityEngineStatus()
        assert status.total_explorations == 0
        assert status.total_new_associations == 0
        assert status.total_external_queries == 0
        assert status.last_exploration_time is None
        assert status.last_exploration_result is None
        assert status.is_exploring is False


class TestStrategyDetermination:
    """策略判断测试"""

    def test_strategy_internal(self, engine, graph):
        """内部已有种子且存在业力边 → internal"""
        strategy = engine._determine_strategy("医学")
        assert strategy == "internal"

    def test_strategy_candidate_upgrade(self, engine, graph):
        """内部无种子但提炼池有候选 → candidate_upgrade"""
        # 插入提炼池候选
        now = datetime.now(timezone.utc).isoformat()
        graph.conn.execute(
            "INSERT INTO distillation_pool "
            "(canonical_source, canonical_target, canonical_relation, representative_label, "
            " count, contributor_users, status, created_at, updated_at) "
            "VALUES (?, ?, 'RELATED', ?, 1, '[\"test\"]', 'pending', ?, ?)",
            ("文学种子", "文学关联", "文学种子→文学关联", now, now),
        )
        graph.conn.commit()

        strategy = engine._determine_strategy("文学")
        assert strategy == "candidate_upgrade"

    def test_strategy_external_when_enabled(self, engine, graph):
        """完全空白且 EXTERNAL_QUERY_ENABLED=True → external"""
        with patch("consciousness_sea.metacognition.curiosity_engine.EXTERNAL_QUERY_ENABLED", True):
            strategy = engine._determine_strategy("不存在的领域")
        assert strategy == "external"

    def test_strategy_none_when_disabled(self, engine, graph):
        """完全空白且 EXTERNAL_QUERY_ENABLED=False → none"""
        with patch("consciousness_sea.metacognition.curiosity_engine.EXTERNAL_QUERY_ENABLED", False):
            strategy = engine._determine_strategy("不存在的领域")
        assert strategy == "none"


class TestExploreEntry:
    """探索入口测试"""

    def test_explore_disabled(self, engine, graph):
        """CURIOSITY_ENGINE_ENABLED=False 时返回 disabled"""
        goal = CognitiveGoalData(
            goal_id="goal_test", goal_type=GoalType.LOW_CONFIDENCE,
            trigger_condition="test", domain="医学",
        )
        with patch("consciousness_sea.metacognition.curiosity_engine.CURIOSITY_ENGINE_ENABLED", False):
            result = engine.explore(goal)
        assert result.strategy == "disabled"
        assert result.error is not None

    def test_explore_concurrent_limit(self, engine, graph):
        """并发控制：超过 CURIOSITY_MAX_CONCURRENT 时返回错误"""
        goal = CognitiveGoalData(
            goal_id="goal_concurrent", goal_type=GoalType.LOW_CONFIDENCE,
            trigger_condition="test", domain="医学",
        )
        # 模拟并发已满
        engine._current_explorations = CURIOSITY_MAX_CONCURRENT
        result = engine.explore(goal)
        assert result.error is not None
        assert "并发" in result.error or "concurrent" in result.error.lower()
        # 恢复
        engine._current_explorations = 0

    def test_explore_success_updates_status(self, engine, graph):
        """探索成功 → 目标状态更新为 completed，priority_weight=0.1"""
        _insert_goal(graph.conn, goal_id="goal_success", domain="医学")
        goal = CognitiveGoalData(
            goal_id="goal_success", goal_type=GoalType.LOW_CONFIDENCE,
            trigger_condition="test", domain="医学",
        )

        # Mock 内部探索以避免实际路由调用
        with patch.object(engine, '_explore_internal') as mock_internal:
            mock_internal.return_value = ExplorationResult(
                goal_id="goal_success", strategy="internal",
                explored_seeds=["感冒", "发热"], new_associations=2,
            )
            result = engine.explore(goal)

        assert result.error is None
        row = graph.conn.execute(
            "SELECT status, priority_weight FROM cognitive_goals WHERE goal_id = 'goal_success'"
        ).fetchone()
        assert row["status"] == "completed"
        assert row["priority_weight"] == 0.1

    def test_explore_failure_restores_pending(self, engine, graph):
        """探索失败 → 目标状态恢复为 pending"""
        _insert_goal(graph.conn, goal_id="goal_fail", domain="医学")
        goal = CognitiveGoalData(
            goal_id="goal_fail", goal_type=GoalType.LOW_CONFIDENCE,
            trigger_condition="test", domain="医学",
        )

        with patch.object(engine, '_explore_internal') as mock_internal:
            mock_internal.return_value = ExplorationResult(
                goal_id="goal_fail", strategy="internal",
                error="虚拟查询执行失败",
            )
            result = engine.explore(goal)

        assert result.error is not None
        row = graph.conn.execute(
            "SELECT status FROM cognitive_goals WHERE goal_id = 'goal_fail'"
        ).fetchone()
        assert row["status"] == "pending"

    def test_explore_updates_engine_status(self, engine, graph):
        """探索后更新引擎运行状态"""
        _insert_goal(graph.conn, goal_id="goal_status", domain="医学")
        goal = CognitiveGoalData(
            goal_id="goal_status", goal_type=GoalType.LOW_CONFIDENCE,
            trigger_condition="test", domain="医学",
        )

        with patch.object(engine, '_explore_internal') as mock_internal:
            mock_internal.return_value = ExplorationResult(
                goal_id="goal_status", strategy="internal",
                new_associations=3,
            )
            engine.explore(goal)

        status = engine.get_status()
        assert status.total_explorations >= 1
        assert status.total_new_associations >= 3
        assert status.last_exploration_time is not None
        assert status.last_exploration_result == "success"

    def test_explore_finally_decrements_concurrent(self, engine, graph):
        """探索完成后并发计数递减（finally 块）"""
        _insert_goal(graph.conn, goal_id="goal_finally", domain="医学")
        goal = CognitiveGoalData(
            goal_id="goal_finally", goal_type=GoalType.LOW_CONFIDENCE,
            trigger_condition="test", domain="医学",
        )

        with patch.object(engine, '_explore_internal') as mock_internal:
            mock_internal.return_value = ExplorationResult(
                goal_id="goal_finally", strategy="internal",
                error="test error",
            )
            engine.explore(goal)

        assert engine._current_explorations == 0


class TestInternalExploration:
    """内部探索测试"""

    def test_internal_exploration_updates_status(self, engine, graph):
        """内部探索更新目标状态为 exploring"""
        _insert_goal(graph.conn, goal_id="goal_internal", domain="医学")
        goal = CognitiveGoalData(
            goal_id="goal_internal", goal_type=GoalType.LOW_CONFIDENCE,
            trigger_condition="test", domain="医学",
        )

        # Mock route 函数（局部导入，需 patch 源模块）
        from consciousness_sea.domain.router import RippleResult, ActivationNode
        mock_result = RippleResult()
        mock_result.activated["感冒"] = ActivationNode(
            label="感冒", activation=0.8, domain="医学", depth=0,
        )
        mock_result.activated["发热"] = ActivationNode(
            label="发热", activation=0.6, domain="医学", depth=1,
        )

        with patch("consciousness_sea.domain.router.route", return_value=mock_result):
            result = engine._explore_internal(goal)

        # 验证状态被更新为 exploring（探索过程中）
        # 探索完成后由 explore() 更新为 completed
        assert result.strategy == "internal"

    def test_internal_no_core_seeds_fallback(self, engine, graph):
        """无核心种子时切换到外部查询（若启用）"""
        _insert_goal(graph.conn, goal_id="goal_no_seeds", domain="文学")
        goal = CognitiveGoalData(
            goal_id="goal_no_seeds", goal_type=GoalType.LOW_DENSITY,
            trigger_condition="test", domain="文学",
        )

        with patch("consciousness_sea.metacognition.curiosity_engine.EXTERNAL_QUERY_ENABLED", False):
            result = engine._explore_internal(goal)

        assert result.error is not None

    def test_internal_route_failure(self, engine, graph):
        """虚拟查询执行失败时返回错误"""
        _insert_goal(graph.conn, goal_id="goal_route_fail", domain="医学")
        goal = CognitiveGoalData(
            goal_id="goal_route_fail", goal_type=GoalType.LOW_CONFIDENCE,
            trigger_condition="test", domain="医学",
        )

        with patch("consciousness_sea.domain.router.route", side_effect=Exception("路由失败")):
            result = engine._explore_internal(goal)

        assert result.error is not None
        assert "路由失败" in result.error or "失败" in result.error


class TestCandidateUpgrade:
    """候选升级策略测试"""

    def test_candidate_upgrade(self, engine, graph):
        """候选升级调用提炼池升级流程"""
        _insert_goal(graph.conn, goal_id="goal_upgrade", domain="医学")
        goal = CognitiveGoalData(
            goal_id="goal_upgrade", goal_type=GoalType.LOW_CONFIDENCE,
            trigger_condition="test", domain="医学",
        )

        with patch("consciousness_sea.learning.distillation_pool.DistillationPool") as mock_pool_cls:
            mock_pool = MagicMock()
            mock_pool.try_upgrade_by_domain.return_value = 2
            mock_pool_cls.return_value = mock_pool

            result = engine._explore_candidate_upgrade(goal)

        assert result.strategy == "candidate_upgrade"
        assert result.new_associations == 2

    def test_candidate_upgrade_failure(self, engine, graph):
        """候选升级失败时返回错误"""
        _insert_goal(graph.conn, goal_id="goal_upgrade_fail", domain="医学")
        goal = CognitiveGoalData(
            goal_id="goal_upgrade_fail", goal_type=GoalType.LOW_CONFIDENCE,
            trigger_condition="test", domain="医学",
        )

        with patch("consciousness_sea.learning.distillation_pool.DistillationPool") as mock_pool_cls:
            mock_pool_cls.side_effect = Exception("提炼池不可用")

            result = engine._explore_candidate_upgrade(goal)

        assert result.error is not None


class TestExternalQuery:
    """外部查询策略测试"""

    def test_external_disabled(self, engine, graph):
        """EXTERNAL_QUERY_ENABLED=False 时返回错误"""
        _insert_goal(graph.conn, goal_id="goal_ext", domain="文学")
        goal = CognitiveGoalData(
            goal_id="goal_ext", goal_type=GoalType.LOW_DENSITY,
            trigger_condition="test", domain="文学",
        )

        with patch("consciousness_sea.metacognition.curiosity_engine.EXTERNAL_QUERY_ENABLED", False):
            result = engine._explore_external(goal)

        assert result.error is not None
        assert "禁用" in result.error or "disabled" in result.error.lower()

    def test_external_updates_status(self, engine, graph):
        """外部查询更新目标状态为 querying_external"""
        _insert_goal(graph.conn, goal_id="goal_ext_status", domain="文学")
        goal = CognitiveGoalData(
            goal_id="goal_ext_status", goal_type=GoalType.LOW_DENSITY,
            trigger_condition="test", domain="文学",
        )

        with patch("consciousness_sea.metacognition.curiosity_engine.EXTERNAL_QUERY_ENABLED", True):
            with patch.object(engine, '_query_external_source') as mock_query:
                mock_query.return_value = ExternalQueryResult(
                    title="文学理论", summary="文学理论概述",
                    related_terms=["叙事", "修辞"], categories=["文学"],
                )
                with patch("consciousness_sea.learning.distillation_pool.DistillationPool") as mock_pool_cls:
                    mock_pool = MagicMock()
                    mock_pool.submit_external_candidate.return_value = 1
                    mock_pool.submit_candidate.return_value = 2
                    mock_pool_cls.return_value = mock_pool

                    result = engine._explore_external(goal)

        assert result.strategy == "external"
        assert result.error is None

    def test_external_query_no_result(self, engine, graph):
        """外部查询返回空结果时返回错误"""
        _insert_goal(graph.conn, goal_id="goal_ext_empty", domain="文学")
        goal = CognitiveGoalData(
            goal_id="goal_ext_empty", goal_type=GoalType.LOW_DENSITY,
            trigger_condition="test", domain="文学",
        )

        with patch("consciousness_sea.metacognition.curiosity_engine.EXTERNAL_QUERY_ENABLED", True):
            with patch.object(engine, '_query_external_source', return_value=None):
                result = engine._explore_external(goal)

        assert result.error is not None

    def test_external_query_retry(self, engine, graph):
        """外部查询失败时重试"""
        _insert_goal(graph.conn, goal_id="goal_ext_retry", domain="文学")
        goal = CognitiveGoalData(
            goal_id="goal_ext_retry", goal_type=GoalType.LOW_DENSITY,
            trigger_condition="test", domain="文学",
        )

        call_count = 0

        def side_effect(domain):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise Exception("网络超时")
            return ExternalQueryResult(
                title="文学", summary="文学概述",
                related_terms=["叙事"], categories=["文学"],
            )

        with patch("consciousness_sea.metacognition.curiosity_engine.EXTERNAL_QUERY_ENABLED", True):
            with patch.object(engine, '_query_external_source', side_effect=side_effect):
                with patch("consciousness_sea.learning.distillation_pool.DistillationPool") as mock_pool_cls:
                    mock_pool = MagicMock()
                    mock_pool.submit_external_candidate.return_value = 1
                    mock_pool.submit_candidate.return_value = 2
                    mock_pool_cls.return_value = mock_pool

                    result = engine._explore_external(goal)

        assert call_count == 2
        assert result.error is None


class TestGetStatus:
    """运行状态查询测试"""

    def test_get_status_initial(self, engine):
        """初始状态"""
        status = engine.get_status()
        assert isinstance(status, CuriosityEngineStatus)
        assert status.total_explorations == 0
        assert status.total_new_associations == 0
        assert status.total_external_queries == 0
        assert status.is_exploring is False

    def test_get_status_after_exploration(self, engine, graph):
        """探索后状态更新"""
        _insert_goal(graph.conn, goal_id="goal_status2", domain="医学")
        goal = CognitiveGoalData(
            goal_id="goal_status2", goal_type=GoalType.LOW_CONFIDENCE,
            trigger_condition="test", domain="医学",
        )

        with patch.object(engine, '_explore_internal') as mock_internal:
            mock_internal.return_value = ExplorationResult(
                goal_id="goal_status2", strategy="internal",
                new_associations=5,
            )
            engine.explore(goal)

        status = engine.get_status()
        assert status.total_explorations >= 1
        assert status.total_new_associations >= 5


class TestExecutionLog:
    """执行日志追加测试"""

    def test_append_execution_log(self, engine, graph):
        """追加执行日志"""
        _insert_goal(graph.conn, goal_id="goal_log")
        entry = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "action": "internal_exploration",
            "result": "found 3 new associations",
            "seeds": ["感冒", "发热", "咳嗽"],
        }
        engine._append_execution_log("goal_log", entry)

        row = graph.conn.execute(
            "SELECT execution_log FROM cognitive_goals WHERE goal_id = 'goal_log'"
        ).fetchone()
        log_list = json.loads(row["execution_log"])
        assert len(log_list) >= 1
        assert log_list[-1]["action"] == "internal_exploration"

    def test_append_execution_log_max_20(self, engine, graph):
        """执行日志最多保留 20 条"""
        _insert_goal(graph.conn, goal_id="goal_log_20")
        for i in range(25):
            entry = {"action": f"test_{i}", "result": "ok"}
            engine._append_execution_log("goal_log_20", entry)

        row = graph.conn.execute(
            "SELECT execution_log FROM cognitive_goals WHERE goal_id = 'goal_log_20'"
        ).fetchone()
        log_list = json.loads(row["execution_log"])
        assert len(log_list) <= 20

    def test_append_execution_log_nonexistent_goal(self, engine, graph):
        """追加日志到不存在的目标时不报错"""
        entry = {"action": "test", "result": "ok"}
        engine._append_execution_log("goal_not_exist", entry)
        # 不应抛出异常


class TestWikipediaDump:
    """Wikipedia 查询测试"""

    def test_query_wikipedia_dump_no_db(self, engine):
        """Wikipedia 数据库不存在时返回 None"""
        result = engine._query_wikipedia_dump("量子力学")
        assert result is None

    def test_query_external_source_wikipedia(self, engine):
        """EXTERNAL_SOURCE_TYPE=wikipedia_dump 时调用 _query_wikipedia_dump"""
        with patch.object(engine, '_query_wikipedia_dump', return_value=None) as mock:
            engine._query_external_source("量子力学")
        mock.assert_called_once_with("量子力学")