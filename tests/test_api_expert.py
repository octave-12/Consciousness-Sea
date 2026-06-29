"""
T-024: API 端点集成测试 + 向后兼容性测试

测试 API 端点的专家扩展：
- Phase 0 模式: /api/v1/query 响应包含 expert_available=false
- Phase 1 模式: 响应包含 expert_answer, expert_domain 等字段
- /status 包含 expert_status 字段
- 现有 test_api.py 全部通过（向后兼容）
- 使用 TestClient + MockExpertManager
"""

from __future__ import annotations

import sqlite3
import sys
import pathlib
from unittest.mock import patch

_root = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_root))
sys.path.insert(0, str(_root / "backend" / "src"))

from fastapi.testclient import TestClient

from consciousness_sea.interfaces.api import app
from consciousness_sea.domain.graph_db import GraphDB
from consciousness_sea.infrastructure.connection_pool import ConnectionPool
from consciousness_sea.infrastructure.user_manager import UserManager
from consciousness_sea.infrastructure.session_manager import SessionManager, SessionContext
from consciousness_sea.infrastructure.observer import Observer
from tests.conftest import MockExpertManager


# ═══════════════════════════════════════════════════════════
#  内存数据库构建
# ═══════════════════════════════════════════════════════════

def _build_test_db() -> GraphDB:
    """创建内存数据库"""
    conn = sqlite3.connect(':memory:', check_same_thread=False)
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
        CREATE TABLE IF NOT EXISTS karma_edges_personal (
            user_label  TEXT    NOT NULL,
            source      TEXT    NOT NULL,
            target      TEXT    NOT NULL,
            relation    TEXT    NOT NULL,
            weight      REAL    NOT NULL,
            source_tag  TEXT    NOT NULL DEFAULT 'personal_karma',
            updated_at  TEXT    NOT NULL,
            PRIMARY KEY (user_label, source, target, relation)
        );
        CREATE TABLE IF NOT EXISTS distillation_pool (
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
        CREATE TABLE IF NOT EXISTS param_stats (
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
        CREATE TABLE IF NOT EXISTS expert_reliability (
            domain     TEXT PRIMARY KEY,
            score      REAL NOT NULL CHECK(score >= 0.0 AND score <= 1.0),
            updated_at TEXT NOT NULL
        );
    """)

    seeds = [
        ('感冒', '感冒', 'CONCEPT', '[]', '医学', 'to catch cold'),
        ('发热', '发热', 'CONCEPT', '[]', '医学', 'fever'),
        ('咳嗽', '咳嗽', 'CONCEPT', '[]', '医学', 'cough'),
        ('量子力学', '量子力学', 'CONCEPT', '[]', '物理', 'quantum mechanics'),
        ('人工智能', '人工智能', 'CONCEPT', '["AI"]', '计算机', 'AI'),
        ('深度学习', '深度学习', 'CONCEPT', '[]', '计算机', 'deep learning'),
        ('水', '水', 'CONCEPT', '[]', '常识', 'water'),
    ]
    conn.executemany(
        "INSERT INTO seeds (id,label,type,aliases,domain,definition) VALUES (?,?,?,?,?,?)",
        seeds,
    )

    edges = [
        ('感冒', '发热', 'COOCCURS_WITH', 0.95),
        ('感冒', '咳嗽', 'COOCCURS_WITH', 0.90),
        ('量子力学', '薛定谔方程', 'RELATED', 0.85),
        ('人工智能', '深度学习', 'IS_A', 0.90),
    ]
    conn.executemany(
        "INSERT INTO karma_edges (source,target,relation,weight) VALUES (?,?,?,?)",
        edges,
    )
    conn.commit()

    db = GraphDB(':memory:')
    db.conn = conn
    db.ensure_phase2_tables()
    db.ensure_phase3_tables()
    return db


# ═══════════════════════════════════════════════════════════
#  依赖注入覆盖
# ═══════════════════════════════════════════════════════════

_test_db = _build_test_db()


class _InMemoryPool:
    def acquire(self):
        return _test_db

    def release(self, graph):
        pass

    def close_all(self):
        pass


_test_pool = _InMemoryPool()
_test_user_manager = UserManager(_test_pool)
_test_session_manager = SessionManager(_test_pool)
_test_observer = Observer(_test_pool)


def _override_get_pool():
    yield _test_pool


def _override_get_session_manager():
    yield _test_session_manager


def _override_get_user_manager():
    yield _test_user_manager


def _override_get_observer():
    yield _test_observer


from consciousness_sea.interfaces.api import get_pool, get_session_manager, get_user_manager, get_observer

app.dependency_overrides[get_pool] = _override_get_pool
app.dependency_overrides[get_session_manager] = _override_get_session_manager
app.dependency_overrides[get_user_manager] = _override_get_user_manager
app.dependency_overrides[get_observer] = _override_get_observer


# ═══════════════════════════════════════════════════════════
#  测试类
# ═══════════════════════════════════════════════════════════


class TestApiExpertPhase0:
    """Phase 0 模式 API 测试（无专家）"""

    def setup_method(self):
        self.client = TestClient(app)
        # 设置 _expert_manager 为 None（Phase 0）
        import sys
        api_module = sys.modules['consciousness_sea.interfaces.api']
        self._original_expert_manager = api_module._expert_manager
        api_module._expert_manager = None

    def teardown_method(self):
        import sys
        api_module = sys.modules['consciousness_sea.interfaces.api']
        api_module._expert_manager = self._original_expert_manager

    def test_phase0_query_response_contains_expert_available_false(self):
        """Phase 0 模式: /api/v1/query 响应包含 expert_available=false"""
        resp = self.client.post('/api/v1/query', json={'query': '感冒'})
        assert resp.status_code == 200
        data = resp.json()
        assert data['expert_available'] is False

    def test_phase0_query_response_no_expert_answer(self):
        """Phase 0 模式: 响应无 expert_answer"""
        resp = self.client.post('/api/v1/query', json={'query': '感冒'})
        assert resp.status_code == 200
        data = resp.json()
        assert data.get('expert_answer') is None

    def test_phase0_query_response_has_existing_fields(self):
        """Phase 0 模式: 现有字段完整"""
        resp = self.client.post('/api/v1/query', json={'query': '感冒'})
        assert resp.status_code == 200
        data = resp.json()
        assert 'query' in data
        assert 'activated_seeds' in data
        assert 'confidence' in data
        assert 'karma_direction' in data
        assert 'decision' in data

    def test_phase0_query_cross_validation_status_none(self):
        """Phase 0 模式: cross_validation_status='none'"""
        resp = self.client.post('/api/v1/query', json={'query': '感冒'})
        assert resp.status_code == 200
        data = resp.json()
        assert data['cross_validation_status'] == 'none'


class TestApiExpertPhase1:
    """Phase 1 模式 API 测试（有专家）"""

    def setup_method(self):
        self.client = TestClient(app)
        self.mock_manager = MockExpertManager(available=True, answer="专家测试回答")
        import sys
        api_module = sys.modules['consciousness_sea.interfaces.api']
        self._original_expert_manager = api_module._expert_manager
        api_module._expert_manager = self.mock_manager

    def teardown_method(self):
        import sys
        api_module = sys.modules['consciousness_sea.interfaces.api']
        api_module._expert_manager = self._original_expert_manager

    def test_phase1_query_response_contains_expert_available_true(self):
        """Phase 1 模式: 响应包含 expert_available=true"""
        resp = self.client.post('/api/v1/query', json={'query': '感冒'})
        assert resp.status_code == 200
        data = resp.json()
        assert data['expert_available'] is True

    def test_phase1_query_response_contains_expert_answer(self):
        """Phase 1 模式: 响应包含 expert_answer"""
        resp = self.client.post('/api/v1/query', json={'query': '感冒'})
        assert resp.status_code == 200
        data = resp.json()
        assert data.get('expert_answer') is not None

    def test_phase1_query_response_contains_expert_domain(self):
        """Phase 1 模式: 响应包含 expert_domain"""
        resp = self.client.post('/api/v1/query', json={'query': '感冒'})
        assert resp.status_code == 200
        data = resp.json()
        assert data.get('expert_domain') is not None

    def test_phase1_query_response_contains_reliability_score(self):
        """Phase 1 模式: 响应包含 reliability_score"""
        resp = self.client.post('/api/v1/query', json={'query': '感冒'})
        assert resp.status_code == 200
        data = resp.json()
        assert 'reliability_score' in data


class TestApiExpertStatus:
    """/status 端点专家状态测试"""

    def setup_method(self):
        self.client = TestClient(app)

    def test_status_contains_expert_status_field(self):
        """/status 包含 expert_status 字段"""
        resp = self.client.get('/status', headers={'Accept': 'application/json'})
        assert resp.status_code == 200
        data = resp.json()
        assert 'expert_status' in data

    def test_status_expert_status_has_expert_available(self):
        """expert_status 包含 expert_available 字段"""
        resp = self.client.get('/status', headers={'Accept': 'application/json'})
        assert resp.status_code == 200
        data = resp.json()
        assert 'expert_available' in data['expert_status']

    def test_status_expert_status_no_manager(self):
        """无 ExpertManager 时 expert_status 显示 not_initialized"""
        import sys
        api_module = sys.modules['consciousness_sea.interfaces.api']
        original = api_module._expert_manager
        api_module._expert_manager = None

        try:
            resp = self.client.get('/status', headers={'Accept': 'application/json'})
            assert resp.status_code == 200
            data = resp.json()
            assert data['expert_status']['expert_available'] is False
            assert data['expert_status']['unavailable_reason'] == 'not_initialized'
        finally:
            api_module._expert_manager = original

    def test_status_expert_status_with_mock_manager(self):
        """有 MockExpertManager 时 expert_status 显示详细信息"""
        mock_manager = MockExpertManager(available=True)
        import sys
        api_module = sys.modules['consciousness_sea.interfaces.api']
        original = api_module._expert_manager
        api_module._expert_manager = mock_manager

        try:
            resp = self.client.get('/status', headers={'Accept': 'application/json'})
            assert resp.status_code == 200
            data = resp.json()
            assert data['expert_status']['expert_available'] is True
            assert 'current_lora' in data['expert_status']
            assert 'reliability_scores' in data['expert_status']
        finally:
            api_module._expert_manager = original


class TestApiExpertBackwardCompat:
    """向后兼容性测试"""

    def setup_method(self):
        self.client = TestClient(app)
        import sys
        api_module = sys.modules['consciousness_sea.interfaces.api']
        self._original_expert_manager = api_module._expert_manager
        api_module._expert_manager = None

    def teardown_method(self):
        import sys
        api_module = sys.modules['consciousness_sea.interfaces.api']
        api_module._expert_manager = self._original_expert_manager

    def test_query_endpoint_still_works(self):
        """查询端点仍然正常工作"""
        resp = self.client.post('/api/v1/query', json={'query': '感冒'})
        assert resp.status_code == 200

    def test_stats_endpoint_still_works(self):
        """统计端点仍然正常工作"""
        resp = self.client.get('/api/v1/stats')
        assert resp.status_code == 200

    def test_health_endpoint_still_works(self):
        """健康检查端点仍然正常工作"""
        resp = self.client.get('/health')
        assert resp.status_code == 200
        assert resp.json()['status'] == 'ok'

    def test_history_endpoint_still_works(self):
        """历史端点仍然正常工作"""
        resp = self.client.get('/api/v1/history')
        assert resp.status_code == 200

    def test_query_response_structure_backward_compat(self):
        """查询响应结构与之前兼容"""
        resp = self.client.post('/api/v1/query', json={'query': '感冒'})
        assert resp.status_code == 200
        data = resp.json()

        # 原有字段
        assert 'query' in data
        assert 'activated_seeds' in data
        assert 'paths' in data
        assert 'domain_scores' in data
        assert 'selected_domains' in data
        assert 'matched_seeds' in data
        assert 'total_activated' in data
        assert 'confidence' in data
        assert 'karma_direction' in data
        assert 'decision' in data

        # 新增字段（有默认值，向后兼容）
        assert 'expert_available' in data
        assert 'cross_validation_status' in data


if __name__ == '__main__':
    import pytest
    pytest.main([__file__, '-v'])