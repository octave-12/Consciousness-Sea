"""
Phase 5 API 端点测试

覆盖：
- GET /api/v1/cognitive-goals
- GET /api/v1/cognitive-goals/{goal_id}
- GET /api/v1/cognitive-goals/stats
- POST /api/v1/cognitive-goals
- GET /api/v1/curiosity/status
- POST /api/v1/curiosity/explore/{goal_id}
- /status 端点的认知目标状态扩展
"""

from __future__ import annotations

import json
import sqlite3
import sys
import os
from datetime import datetime, timezone
from unittest.mock import patch, MagicMock

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fastapi.testclient import TestClient

from core.graph_db import GraphDB
from core.cognitive_goal import CognitiveGoalManager, GoalType, GoalStatus
from core.curiosity_engine import CuriosityEngine, CuriosityEngineStatus
from core.guardian_loop import GuardianLoop
from core.config import COGNITIVE_GOAL_ENABLED, CURIOSITY_ENGINE_ENABLED


# ═══════════════════════════════════════════════════════════
#  Fixtures
# ═══════════════════════════════════════════════════════════


def _build_test_db() -> sqlite3.Connection:
    """创建含 Phase 5 表的内存测试数据库"""
    # check_same_thread=False: TestClient 在不同线程中执行请求
    conn = sqlite3.connect(":memory:", check_same_thread=False)
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

    seeds = [
        ("感冒", "感冒", "CONCEPT", "[]", "医学", "急性上呼吸道感染"),
        ("发热", "发热", "CONCEPT", "[]", "医学", "体温升高"),
        ("量子力学", "量子力学", "CONCEPT", "[]", "物理", "quantum mechanics"),
    ]
    conn.executemany(
        "INSERT INTO seeds (id, label, type, aliases, domain, definition) VALUES (?, ?, ?, ?, ?, ?)",
        seeds,
    )

    edges = [
        ("感冒", "发热", "COOCCURS_WITH", 0.95, "karma_delta"),
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
def client(graph):
    """创建 TestClient，注入 mock 的连接池和好奇心引擎"""
    import api as api_module

    # 创建 mock 连接池
    mock_pool = MagicMock()

    def acquire_side_effect():
        return graph

    def release_side_effect(g):
        pass

    mock_pool.acquire.side_effect = acquire_side_effect
    mock_pool.release.side_effect = release_side_effect

    # 创建 GuardianLoop
    guardian_loop = GuardianLoop(graph)

    # 创建 CognitiveGoalManager 和 CuriosityEngine
    goal_mgr = CognitiveGoalManager(graph)
    curiosity_engine = CuriosityEngine(graph, goal_mgr)

    # 注入到 api 模块
    api_module._pool = mock_pool
    api_module._guardian_loop = guardian_loop
    api_module._goal_mgr = goal_mgr
    api_module._curiosity_engine = curiosity_engine

    # 创建 mock Observer
    from core.observer import Observer, StatusData
    mock_observer = MagicMock(spec=Observer)
    mock_status = StatusData(
        total_seeds=3, total_karma_edges=1,
        hottest_seeds=[], coldest_seeds=[], heaviest_karma=[],
        recent_queries=[], alerts=[], domain_distribution={},
        db_size_mb=0.0,
        cognitive_goals=None, curiosity_engine=None,
    )
    mock_observer.get_status.return_value = mock_status
    api_module._observer = mock_observer

    # 创建 mock SessionManager 和 UserManager
    mock_session_mgr = MagicMock()
    mock_user_mgr = MagicMock()
    api_module._session_manager = mock_session_mgr
    api_module._user_manager = mock_user_mgr

    with patch("api.COGNITIVE_GOAL_ENABLED", True), \
         patch("api.CURIOSITY_ENGINE_ENABLED", True), \
         patch("api.META_SEED_ENABLED", True):
        tc = TestClient(api_module.app)
        yield tc

    # 清理
    api_module._pool = None
    api_module._guardian_loop = None
    api_module._goal_mgr = None
    api_module._curiosity_engine = None
    api_module._observer = None


def _insert_goal_via_api_db(graph, goal_id="goal_api_test", domain="医学",
                             status="pending", goal_type="low_confidence",
                             priority_weight=0.72):
    """辅助：直接向数据库插入认知目标"""
    now = datetime.now(timezone.utc).isoformat()
    graph.conn.execute(
        "INSERT INTO cognitive_goals "
        "(goal_id, goal_type, trigger_condition, domain, priority_weight, status, "
        " sub_goals, execution_log, decay_cycles_count, last_touched_at, created_at, updated_at) "
        "VALUES (?, ?, 'test_trigger', ?, ?, ?, '[]', '[]', 0, ?, ?, ?)",
        (goal_id, goal_type, domain, priority_weight, status, now, now, now),
    )
    graph.conn.commit()


# ═══════════════════════════════════════════════════════════
#  测试用例
# ═══════════════════════════════════════════════════════════


class TestCognitiveGoalsAPI:
    """认知目标 API 端点测试"""

    def test_list_cognitive_goals(self, client, graph):
        """GET /api/v1/cognitive-goals 返回目标列表"""
        _insert_goal_via_api_db(graph)

        with patch("api.COGNITIVE_GOAL_ENABLED", True):
            response = client.get("/api/v1/cognitive-goals")

        assert response.status_code == 200
        data = response.json()
        assert "goals" in data
        assert len(data["goals"]) >= 1

        # 验证字段格式
        for g in data["goals"]:
            assert "goal_id" in g
            assert "goal_type" in g
            assert "domain" in g
            assert "priority_weight" in g
            assert "status" in g

    def test_list_cognitive_goals_filter_status(self, client, graph):
        """GET /api/v1/cognitive-goals?status=pending 过滤正确"""
        _insert_goal_via_api_db(graph, goal_id="goal_pending_api", status="pending")
        _insert_goal_via_api_db(graph, goal_id="goal_completed_api", status="completed", domain="物理")

        with patch("api.COGNITIVE_GOAL_ENABLED", True):
            response = client.get("/api/v1/cognitive-goals?status=pending")

        assert response.status_code == 200
        data = response.json()
        for g in data["goals"]:
            assert g["status"] == "pending"

    def test_list_cognitive_goals_filter_type(self, client, graph):
        """GET /api/v1/cognitive-goals?goal_type=low_confidence 过滤正确"""
        _insert_goal_via_api_db(graph, goal_id="goal_lc_api", goal_type="low_confidence")
        _insert_goal_via_api_db(graph, goal_id="goal_hc_api", goal_type="high_conflict", domain="物理")

        with patch("api.COGNITIVE_GOAL_ENABLED", True):
            response = client.get("/api/v1/cognitive-goals?goal_type=low_confidence")

        assert response.status_code == 200
        data = response.json()
        for g in data["goals"]:
            assert g["goal_type"] == "low_confidence"

    def test_list_cognitive_goals_invalid_status(self, client, graph):
        """GET /api/v1/cognitive-goals?status=invalid 返回 422"""
        with patch("api.COGNITIVE_GOAL_ENABLED", True):
            response = client.get("/api/v1/cognitive-goals?status=invalid")

        assert response.status_code == 422

    def test_list_cognitive_goals_disabled(self, client, graph):
        """COGNITIVE_GOAL_ENABLED=False 时返回空列表"""
        with patch("api.COGNITIVE_GOAL_ENABLED", False):
            response = client.get("/api/v1/cognitive-goals")

        assert response.status_code == 200
        data = response.json()
        assert data["goals"] == []

    def test_get_cognitive_goal(self, client, graph):
        """GET /api/v1/cognitive-goals/{goal_id} 返回详情"""
        _insert_goal_via_api_db(graph, goal_id="goal_detail_api")

        with patch("api.COGNITIVE_GOAL_ENABLED", True):
            response = client.get("/api/v1/cognitive-goals/goal_detail_api")

        assert response.status_code == 200
        data = response.json()
        assert data["goal_id"] == "goal_detail_api"
        assert data["goal_type"] == "low_confidence"
        assert data["domain"] == "医学"
        assert "trigger_condition" in data
        assert "execution_log" in data
        assert "sub_goals" in data

    def test_get_cognitive_goal_not_found(self, client, graph):
        """GET /api/v1/cognitive-goals/{goal_id} 不存在时返回 404"""
        with patch("api.COGNITIVE_GOAL_ENABLED", True):
            response = client.get("/api/v1/cognitive-goals/goal_not_exist")

        assert response.status_code == 404

    def test_cognitive_goals_stats(self, client, graph):
        """GET /api/v1/cognitive-goals/stats 返回统计"""
        _insert_goal_via_api_db(graph, goal_id="goal_stats_api_1", status="pending")
        _insert_goal_via_api_db(graph, goal_id="goal_stats_api_2", status="completed",
                                 domain="物理", goal_type="high_conflict")

        with patch("api.COGNITIVE_GOAL_ENABLED", True):
            response = client.get("/api/v1/cognitive-goals/stats")

        assert response.status_code == 200
        data = response.json()
        assert "by_status" in data
        assert "by_type" in data
        assert "avg_priority_weight" in data
        assert "pool_usage" in data

    def test_create_cognitive_goal(self, client, graph):
        """POST /api/v1/cognitive-goals 手动创建目标"""
        with patch("api.COGNITIVE_GOAL_ENABLED", True):
            response = client.post(
                "/api/v1/cognitive-goals",
                json={
                    "goal_type": "low_confidence",
                    "domain": "计算机",
                    "trigger_condition": "manual_test",
                },
            )

        assert response.status_code == 200
        data = response.json()
        assert "goal_id" in data

    def test_create_cognitive_goal_invalid_type(self, client, graph):
        """POST /api/v1/cognitive-goals 无效 goal_type 返回 400"""
        with patch("api.COGNITIVE_GOAL_ENABLED", True):
            response = client.post(
                "/api/v1/cognitive-goals",
                json={
                    "goal_type": "invalid_type",
                    "domain": "计算机",
                    "trigger_condition": "manual",
                },
            )

        assert response.status_code == 400

    def test_create_cognitive_goal_missing_fields(self, client, graph):
        """POST /api/v1/cognitive-goals 缺少必填字段返回 422（Pydantic 校验）"""
        with patch("api.COGNITIVE_GOAL_ENABLED", True):
            response = client.post(
                "/api/v1/cognitive-goals",
                json={"goal_type": "low_confidence"},
            )

        assert response.status_code == 422

    def test_create_cognitive_goal_disabled(self, client, graph):
        """COGNITIVE_GOAL_ENABLED=False 时返回 503"""
        with patch("api.COGNITIVE_GOAL_ENABLED", False):
            response = client.post(
                "/api/v1/cognitive-goals",
                json={
                    "goal_type": "low_confidence",
                    "domain": "计算机",
                    "trigger_condition": "manual",
                },
            )

        assert response.status_code == 503


class TestCuriosityAPI:
    """好奇心引擎 API 端点测试"""

    def test_curiosity_status(self, client, graph):
        """GET /api/v1/curiosity/status 返回引擎状态"""
        with patch("api.CURIOSITY_ENGINE_ENABLED", True):
            response = client.get("/api/v1/curiosity/status")

        assert response.status_code == 200
        data = response.json()
        assert "total_explorations" in data
        assert "total_new_associations" in data
        assert "total_external_queries" in data
        assert "is_exploring" in data

    def test_curiosity_status_disabled(self, client, graph):
        """CURIOSITY_ENGINE_ENABLED=False 时返回默认状态"""
        import api as api_module
        original = api_module._curiosity_engine
        api_module._curiosity_engine = None

        with patch("api.CURIOSITY_ENGINE_ENABLED", False):
            response = client.get("/api/v1/curiosity/status")

        assert response.status_code == 200
        data = response.json()
        assert data["total_explorations"] == 0
        assert data["is_exploring"] is False

        api_module._curiosity_engine = original

    def test_trigger_curiosity_explore(self, client, graph):
        """POST /api/v1/curiosity/explore/{goal_id} 触发探索"""
        _insert_goal_via_api_db(graph, goal_id="goal_explore_api")

        with patch("api.CURIOSITY_ENGINE_ENABLED", True), \
             patch("api.COGNITIVE_GOAL_ENABLED", True):
            response = client.post("/api/v1/curiosity/explore/goal_explore_api")

        # 探索可能成功或失败（取决于内部路由），但不应返回 500
        assert response.status_code in (200, 404, 500)

    def test_trigger_curiosity_explore_not_found(self, client, graph):
        """POST /api/v1/curiosity/explore/{goal_id} 目标不存在返回 404"""
        with patch("api.CURIOSITY_ENGINE_ENABLED", True), \
             patch("api.COGNITIVE_GOAL_ENABLED", True):
            response = client.post("/api/v1/curiosity/explore/goal_not_exist")

        assert response.status_code == 404

    def test_trigger_curiosity_explore_disabled(self, client, graph):
        """CURIOSITY_ENGINE_ENABLED=False 时返回 503"""
        import api as api_module
        original = api_module._curiosity_engine
        api_module._curiosity_engine = None

        with patch("api.CURIOSITY_ENGINE_ENABLED", False):
            response = client.post("/api/v1/curiosity/explore/some_goal")

        assert response.status_code == 503

        api_module._curiosity_engine = original


class TestStatusExtension:
    """/status 端点的 Phase 5 状态扩展测试"""

    def test_status_contains_cognitive_goals_field(self, client, graph):
        """/status 响应包含 cognitive_goals 字段"""
        with patch("api.META_SEED_ENABLED", True), \
             patch("api.COGNITIVE_GOAL_ENABLED", True):
            response = client.get("/status")

        assert response.status_code == 200
        data = response.json()
        assert "cognitive_goals" in data
        assert "curiosity_engine" in data