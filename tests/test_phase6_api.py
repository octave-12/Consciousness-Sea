"""
Phase 6 API 端点测试

覆盖：
- GET /api/v1/perception/status
- GET /api/v1/perception/seeds
- GET /api/v1/perception/seeds/{label}
- GET /api/v1/perception/bindings
- GET /api/v1/perception/events
- POST /api/v1/perception/align
- 感知元种子不存在 404
- 多模态对齐正在运行时 409
- 现有 API 向后兼容性
"""

from __future__ import annotations

import sqlite3
import sys
import pathlib
from unittest.mock import patch, MagicMock

import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent / 'backend' / 'src'))

from fastapi.testclient import TestClient

from consciousness_sea.domain.graph_db import GraphDB
from consciousness_sea.perception.perception import PerceptionManager, PerceptionChannel, PerceptualSeedStatus
from consciousness_sea.infrastructure.config import PERCEPTION_ENABLED


# ═══════════════════════════════════════════════════════════
#  Fixtures
# ═══════════════════════════════════════════════════════════


def _build_test_db() -> sqlite3.Connection:
    """创建含 Phase 6 表的内存测试数据库"""
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
        CREATE TABLE meta_seeds (
            label TEXT PRIMARY KEY NOT NULL,
            category TEXT NOT NULL,
            metrics_json TEXT NOT NULL DEFAULT '{}',
            status TEXT NOT NULL DEFAULT 'active',
            source_domain TEXT,
            dormant_since TEXT,
            unchanged_cycles INTEGER NOT NULL DEFAULT 0,
            previous_metrics_json TEXT NOT NULL DEFAULT '{}',
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL
        );
        CREATE TABLE candidate_seeds (
            label TEXT PRIMARY KEY,
            status TEXT NOT NULL DEFAULT 'candidate',
            count INTEGER NOT NULL DEFAULT 1,
            domain TEXT,
            co_occur_seeds TEXT NOT NULL DEFAULT '[]',
            candidate_since TEXT NOT NULL,
            last_seen_at TEXT NOT NULL,
            promoted_at TEXT,
            promoted_seed_id TEXT
        );
        CREATE TABLE perceptual_seeds (
            label TEXT PRIMARY KEY NOT NULL,
            channel TEXT NOT NULL,
            feature_description TEXT NOT NULL DEFAULT '',
            activation_threshold REAL NOT NULL DEFAULT 0.3,
            status TEXT NOT NULL DEFAULT 'active',
            last_activation TEXT,
            activation_count INTEGER NOT NULL DEFAULT 0,
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL
        );
        CREATE TABLE perception_events (
            event_id INTEGER PRIMARY KEY AUTOINCREMENT,
            perceptual_seed TEXT NOT NULL,
            activation REAL NOT NULL,
            channel TEXT NOT NULL,
            timestamp TEXT NOT NULL,
            processed INTEGER NOT NULL DEFAULT 0
        );
        CREATE TABLE distillation_pool (
            candidate_id INTEGER PRIMARY KEY AUTOINCREMENT,
            canonical_source TEXT NOT NULL,
            canonical_target TEXT NOT NULL,
            canonical_relation TEXT NOT NULL,
            representative_label TEXT NOT NULL,
            count INTEGER NOT NULL DEFAULT 1,
            contributor_users TEXT NOT NULL DEFAULT '[]',
            status TEXT NOT NULL DEFAULT 'pending',
            upgraded_at TEXT,
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL
        );
        CREATE TABLE cognitive_goals (
            goal_id TEXT PRIMARY KEY NOT NULL,
            goal_type TEXT NOT NULL,
            trigger_condition TEXT NOT NULL,
            domain TEXT NOT NULL,
            priority_weight REAL NOT NULL DEFAULT 0.0,
            status TEXT NOT NULL DEFAULT 'pending',
            sub_goals TEXT NOT NULL DEFAULT '[]',
            execution_log TEXT NOT NULL DEFAULT '[]',
            associated_user TEXT,
            decay_cycles_count INTEGER NOT NULL DEFAULT 0,
            last_touched_at TEXT NOT NULL,
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL
        );
    """)

    seeds = [
        ("红色", "红色", "CONCEPT", "[]", "感知", "一种颜色"),
        ("发热", "发热", "CONCEPT", "[]", "医学", "体温升高"),
    ]
    conn.executemany(
        "INSERT INTO seeds (id, label, type, aliases, domain, definition) VALUES (?, ?, ?, ?, ?, ?)",
        seeds,
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
    """创建 TestClient，注入 mock 的连接池和 PerceptionManager"""
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

    # 创建 PerceptionManager
    pm = PerceptionManager(graph)
    pm._generate_preset_perceptual_seeds()

    # 注入到 api.app 模块（直接修改模块级变量）
    api_module._pool = mock_pool
    api_module._perception_manager = pm

    # 创建 mock Observer
    from consciousness_sea.infrastructure.observer import Observer, StatusData
    mock_observer = MagicMock(spec=Observer)
    mock_status = StatusData(
        total_seeds=2,
        total_karma_edges=0,
        hottest_seeds=[],
        coldest_seeds=[],
        heaviest_karma=[],
        recent_queries=[],
        alerts=[],
        domain_distribution={},
        db_size_mb=0.0,
        meta_seeds=None,
        guardian_loop=None,
        cognitive_goals=None,
        curiosity_engine=None,
        perception=None,
    )
    mock_observer.get_status.return_value = mock_status
    api_module._observer = mock_observer

    # 创建 mock SessionManager 和 UserManager
    mock_session_mgr = MagicMock()
    mock_user_mgr = MagicMock()
    api_module._session_manager = mock_session_mgr
    api_module._user_manager = mock_user_mgr

    # GuardianLoop
    api_module._guardian_loop = None

    with patch("consciousness_sea.interfaces.api.PERCEPTION_ENABLED", True):
        c = TestClient(api_module.app)
        yield c

    # 清理
    api_module._pool = None
    api_module._perception_manager = None
    api_module._observer = None


# ═══════════════════════════════════════════════════════════
#  测试用例
# ═══════════════════════════════════════════════════════════


class TestPerceptionStatusAPI:
    """感知状态查询 API 测试"""

    def test_perception_status(self, client):
        """GET /api/v1/perception/status 返回正确格式"""
        with patch("consciousness_sea.interfaces.api.PERCEPTION_ENABLED", True):
            response = client.get("/api/v1/perception/status")

        assert response.status_code == 200
        data = response.json()
        assert "enabled" in data
        assert "channels" in data
        assert "total_perceptual_seeds" in data
        assert "total_hebbian_bindings" in data
        assert "recent_activation_count" in data

    def test_perception_status_disabled(self, client):
        """PERCEPTION_ENABLED=False 时返回默认值"""
        api_module = sys.modules['consciousness_sea.interfaces.api']
        old_pm = api_module._perception_manager
        api_module._perception_manager = None

        with patch("consciousness_sea.interfaces.api.PERCEPTION_ENABLED", False):
            response = client.get("/api/v1/perception/status")

        assert response.status_code == 200
        data = response.json()
        assert data["enabled"] is False

        api_module._perception_manager = old_pm


class TestPerceptionSeedsAPI:
    """感知元种子 API 测试"""

    def test_list_perception_seeds(self, client):
        """GET /api/v1/perception/seeds 返回感知元种子列表"""
        with patch("consciousness_sea.interfaces.api.PERCEPTION_ENABLED", True):
            response = client.get("/api/v1/perception/seeds")

        assert response.status_code == 200
        data = response.json()
        assert "seeds" in data
        assert len(data["seeds"]) > 0

    def test_list_perception_seeds_filter(self, client):
        """GET /api/v1/perception/seeds?channel=visual 过滤正确"""
        with patch("consciousness_sea.interfaces.api.PERCEPTION_ENABLED", True):
            response = client.get("/api/v1/perception/seeds?channel=visual")

        assert response.status_code == 200
        data = response.json()
        for seed in data["seeds"]:
            assert seed["channel"] == "visual"

    def test_get_perception_seed_detail(self, client):
        """GET /api/v1/perception/seeds/{label} 返回详情"""
        with patch("consciousness_sea.interfaces.api.PERCEPTION_ENABLED", True):
            response = client.get("/api/v1/perception/seeds/percept:visual:red")

        assert response.status_code == 200
        data = response.json()
        assert data["label"] == "percept:visual:red"
        assert data["channel"] == "visual"
        assert "hebbian_bindings" in data

    def test_get_perception_seed_not_found(self, client):
        """GET /api/v1/perception/seeds/{label} 不存在时返回 404"""
        with patch("consciousness_sea.interfaces.api.PERCEPTION_ENABLED", True):
            response = client.get("/api/v1/perception/seeds/percept:not:exist")

        assert response.status_code == 404


class TestPerceptionBindingsAPI:
    """Hebbian 绑定边 API 测试"""

    def test_list_perception_bindings(self, client):
        """GET /api/v1/perception/bindings 返回绑定边列表"""
        with patch("consciousness_sea.interfaces.api.PERCEPTION_ENABLED", True):
            response = client.get("/api/v1/perception/bindings")

        assert response.status_code == 200
        data = response.json()
        assert "bindings" in data

    def test_list_perception_bindings_filter(self, client):
        """GET /api/v1/perception/bindings?channel=visual 过滤正确"""
        with patch("consciousness_sea.interfaces.api.PERCEPTION_ENABLED", True):
            response = client.get("/api/v1/perception/bindings?channel=visual")

        assert response.status_code == 200
        data = response.json()
        assert "bindings" in data


class TestPerceptionEventsAPI:
    """感知激活事件 API 测试"""

    def test_list_perception_events(self, client):
        """GET /api/v1/perception/events?limit=20 返回事件列表"""
        with patch("consciousness_sea.interfaces.api.PERCEPTION_ENABLED", True):
            response = client.get("/api/v1/perception/events?limit=20")

        assert response.status_code == 200
        data = response.json()
        assert "events" in data


class TestPerceptionAlignAPI:
    """多模态对齐 API 测试"""

    def test_trigger_alignment(self, client):
        """POST /api/v1/perception/align 执行一次多模态对齐"""
        with patch("consciousness_sea.interfaces.api.PERCEPTION_ENABLED", True):
            response = client.post("/api/v1/perception/align")

        assert response.status_code == 200
        data = response.json()
        assert "results" in data
        assert "count" in data

    def test_trigger_alignment_conflict(self, client):
        """POST /api/v1/perception/align 多模态对齐正在运行时返回 409"""
        api_module = sys.modules['consciousness_sea.interfaces.api']
        pm = api_module._perception_manager
        if pm and pm._multimodal_aligner is not None:
            pm._multimodal_aligner._running = True
            try:
                with patch("consciousness_sea.interfaces.api.PERCEPTION_ENABLED", True):
                    response = client.post("/api/v1/perception/align")
                assert response.status_code == 409
            finally:
                pm._multimodal_aligner._running = False


class TestExistingAPICompatibility:
    """现有 API 向后兼容性测试"""

    def test_health_endpoint(self, client):
        """GET /health 正常返回"""
        response = client.get("/health")
        assert response.status_code == 200

    def test_status_endpoint(self, client):
        """GET /status 正常返回"""
        response = client.get("/status")
        assert response.status_code == 200
        data = response.json()
        assert "total_seeds" in data
        assert "perception" in data