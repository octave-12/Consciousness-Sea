"""
Phase 4 API 端点测试

覆盖：
- GET /api/v1/meta-seeds
- GET /api/v1/meta-seeds/{label}
- GET /api/v1/guardian/status
- POST /api/v1/guardian/trigger
- /status 端点的元种子状态扩展
"""

from __future__ import annotations

import json
import pathlib
import sqlite3
import sys
from unittest.mock import patch, MagicMock

import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent / 'backend' / 'src'))

from fastapi.testclient import TestClient

from consciousness_sea.domain.graph_db import GraphDB
from consciousness_sea.metacognition.meta_seed import MetaSeedManager, MetaSeedCategory, MetaSeedStatus
from consciousness_sea.metacognition.guardian_loop import GuardianLoop, GuardianLoopResult, GuardianLoopStatus
from consciousness_sea.infrastructure.config import META_SEED_ENABLED


# ═══════════════════════════════════════════════════════════
#  Fixtures
# ═══════════════════════════════════════════════════════════


def _build_test_db() -> sqlite3.Connection:
    """创建含 Phase 4 表的内存测试数据库"""
    # check_same_thread=False: TestClient 在不同线程中执行请求，
    # 必须允许跨线程使用 SQLite 连接
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
    """创建 TestClient，注入 mock 的连接池和 GuardianLoop"""
    import consciousness_sea.interfaces.api as api  # 触发 api.app 模块导入
    api_module = sys.modules['consciousness_sea.interfaces.api']

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

    # 注入到 api.app 模块（直接修改模块级变量）
    api_module._pool = mock_pool
    api_module._guardian_loop = guardian_loop

    # 创建 mock Observer
    from consciousness_sea.infrastructure.observer import Observer, StatusData
    mock_observer = MagicMock(spec=Observer)
    mock_status = StatusData(
        total_seeds=3,
        total_karma_edges=1,
        hottest_seeds=[],
        coldest_seeds=[],
        heaviest_karma=[],
        recent_queries=[],
        alerts=[],
        domain_distribution={},
        db_size_mb=0.0,
        meta_seeds=None,
        guardian_loop=None,
    )
    mock_observer.get_status.return_value = mock_status
    api_module._observer = mock_observer

    # 创建 mock SessionManager 和 UserManager
    mock_session_mgr = MagicMock()
    mock_user_mgr = MagicMock()
    api_module._session_manager = mock_session_mgr
    api_module._user_manager = mock_user_mgr

    # 确保 META_SEED_ENABLED
    with patch("consciousness_sea.interfaces.api.META_SEED_ENABLED", True):
        client = TestClient(api_module.app)
        yield client

    # 清理
    api_module._pool = None
    api_module._guardian_loop = None
    api_module._observer = None


# ═══════════════════════════════════════════════════════════
#  测试用例
# ═══════════════════════════════════════════════════════════


class TestMetaSeedsAPI:
    """元种子 API 端点测试"""

    def test_list_meta_seeds(self, client, graph):
        """GET /api/v1/meta-seeds 返回元种子列表"""
        # 先创建一些元种子
        mgr = MetaSeedManager(graph)
        mgr.generate_domain_monitors()
        mgr.generate_system_monitors()

        with patch("consciousness_sea.interfaces.api.META_SEED_ENABLED", True):
            response = client.get("/api/v1/meta-seeds")

        assert response.status_code == 200
        data = response.json()
        assert "meta_seeds" in data
        assert len(data["meta_seeds"]) > 0

        # 验证每个元素包含必要字段
        for ms in data["meta_seeds"]:
            assert "label" in ms
            assert "category" in ms
            assert "status" in ms
            assert "metrics" in ms
            assert "updated_at" in ms

    def test_list_meta_seeds_filter_category(self, client, graph):
        """GET /api/v1/meta-seeds?category=domain_monitor 过滤正确"""
        mgr = MetaSeedManager(graph)
        mgr.generate_domain_monitors()
        mgr.generate_system_monitors()

        with patch("consciousness_sea.interfaces.api.META_SEED_ENABLED", True):
            response = client.get("/api/v1/meta-seeds?category=domain_monitor")

        assert response.status_code == 200
        data = response.json()
        for ms in data["meta_seeds"]:
            assert ms["category"] == "domain_monitor"

    def test_get_meta_seed_detail(self, client, graph):
        """GET /api/v1/meta-seeds/{label} 返回详情含 meta_karma_edges"""
        mgr = MetaSeedManager(graph)
        mgr.generate_domain_monitors()

        with patch("consciousness_sea.interfaces.api.META_SEED_ENABLED", True):
            response = client.get("/api/v1/meta-seeds/meta:医学")

        assert response.status_code == 200
        data = response.json()
        assert data["label"] == "meta:医学"
        assert data["category"] == "domain_monitor"
        assert "meta_karma_edges" in data
        assert isinstance(data["meta_karma_edges"], list)

    def test_get_meta_seed_not_found(self, client, graph):
        """GET /api/v1/meta-seeds/{label} 元种子不存在时返回 404"""
        with patch("consciousness_sea.interfaces.api.META_SEED_ENABLED", True):
            response = client.get("/api/v1/meta-seeds/meta:不存在")

        assert response.status_code == 404


class TestGuardianAPI:
    """守护循环 API 端点测试"""

    def test_guardian_status(self, client, graph):
        """GET /api/v1/guardian/status 返回守护循环状态"""
        with patch("consciousness_sea.interfaces.api.META_SEED_ENABLED", True):
            response = client.get("/api/v1/guardian/status")

        assert response.status_code == 200
        data = response.json()
        assert "is_running" in data
        assert "interval_seconds" in data
        assert "consecutive_failures" in data
        assert "total_meta_seeds" in data
        assert "total_meta_karma_edges" in data

    def test_trigger_guardian(self, client, graph):
        """POST /api/v1/guardian/trigger 执行一次守护循环"""
        with patch("consciousness_sea.interfaces.api.META_SEED_ENABLED", True):
            response = client.post("/api/v1/guardian/trigger")

        assert response.status_code == 200
        data = response.json()
        assert data["status"] in ("success", "failed")
        assert "meta_seeds_updated" in data
        assert "meta_karma_edges_created" in data
        assert "duration_ms" in data

    def test_trigger_guardian_conflict(self, client, graph):
        """POST /api/v1/guardian/trigger 守护循环正在执行时返回 409"""
        api_module = sys.modules['consciousness_sea.interfaces.api']
        guardian = api_module._guardian_loop

        # 模拟正在执行
        guardian._is_executing = True
        try:
            with patch("consciousness_sea.interfaces.api.META_SEED_ENABLED", True):
                response = client.post("/api/v1/guardian/trigger")
            assert response.status_code == 409
        finally:
            guardian._is_executing = False

    def test_guardian_status_disabled(self, client, graph):
        """META_SEED_ENABLED=False 时守护循环状态返回默认值"""
        api_module = sys.modules['consciousness_sea.interfaces.api']
        api_module._guardian_loop = None

        with patch("consciousness_sea.interfaces.api.META_SEED_ENABLED", False):
            response = client.get("/api/v1/guardian/status")

        assert response.status_code == 200
        data = response.json()
        assert data["is_running"] is False
        assert data["total_meta_seeds"] == 0

        api_module._guardian_loop = GuardianLoop(graph)


class TestStatusExtension:
    """/status 端点的元种子状态扩展测试"""

    def test_status_contains_meta_seeds_field(self, client, graph):
        """/status 响应包含 meta_seeds 字段"""
        with patch("consciousness_sea.interfaces.api.META_SEED_ENABLED", True):
            response = client.get("/status")

        assert response.status_code == 200
        data = response.json()
        assert "meta_seeds" in data
        assert "guardian_loop" in data