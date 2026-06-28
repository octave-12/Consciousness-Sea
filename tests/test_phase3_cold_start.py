"""
Phase 3 冷启动管理器测试

测试 ColdStartManager 的所有公共方法：
- get_cold_factor: 获取冷启动衰减系数
- increment_query_count: 递增用户查询计数
- get_state: 查询用户冷启动状态
- invalidate_cache: 清除缓存
"""

from __future__ import annotations

import sqlite3
import sys
import os
from unittest.mock import patch

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.cold_start import ColdStartManager, ColdStartState
from core.graph_db import GraphDB
from core.config import COLD_START_ENABLED, COLD_START_QUERIES, KARMA_MAX_PAIRS


# ═══════════════════════════════════════════════════════════
#  Fixtures
# ═══════════════════════════════════════════════════════════


def _build_test_db() -> sqlite3.Connection:
    """创建含 Phase 3 表的内存测试数据库"""
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
        CREATE TABLE user_cold_start (
            user_label  TEXT    PRIMARY KEY,
            query_count INTEGER NOT NULL DEFAULT 0,
            updated_at  TEXT    NOT NULL
        );
    """)
    conn.commit()
    return conn


def _make_graph_db(conn: sqlite3.Connection) -> GraphDB:
    """从已有连接创建 GraphDB 实例"""
    db = GraphDB(":memory:")
    db.conn = conn
    db.ensure_phase2_tables()
    db.ensure_phase3_tables()
    return db


@pytest.fixture
def graph():
    """创建内存数据库的 GraphDB 实例"""
    conn = _build_test_db()
    g = _make_graph_db(conn)
    yield g
    g.close()


@pytest.fixture
def cold_manager(graph):
    """创建 ColdStartManager 实例"""
    return ColdStartManager(graph)


# ═══════════════════════════════════════════════════════════
#  测试用例
# ═══════════════════════════════════════════════════════════


class TestColdStartManager:
    """ColdStartManager 单元测试"""

    def test_cold_start_period(self, cold_manager, graph):
        """query_count <= 20 时 is_cold_start=True"""
        # 新用户，query_count=0
        state = cold_manager.get_state("user_001")
        assert state.is_cold_start is True
        assert state.query_count == 0

        # 增加到 10 次查询
        for _ in range(10):
            cold_manager.increment_query_count("user_001")

        state = cold_manager.get_state("user_001")
        assert state.is_cold_start is True
        assert state.query_count == 10

        # 增加到 19 次查询
        for _ in range(9):
            cold_manager.increment_query_count("user_001")

        state = cold_manager.get_state("user_001")
        assert state.is_cold_start is True
        assert state.query_count == 19

    def test_after_cold_start(self, cold_manager, graph):
        """query_count > 20 时 is_cold_start=False"""
        # 增加到 21 次查询
        for _ in range(21):
            cold_manager.increment_query_count("user_001")

        state = cold_manager.get_state("user_001")
        assert state.is_cold_start is False
        assert state.query_count == 21

    def test_decay_curve(self, cold_manager, graph):
        """第1次 cold_factor=0.05，第10次=0.5，第20次=1.0

        cold_factor = min(query_count / COLD_START_QUERIES, 1.0)
        COLD_START_QUERIES = 20
        """
        # 第1次: 1/20 = 0.05
        cold_manager.increment_query_count("user_curve")
        state = cold_manager.get_state("user_curve")
        assert abs(state.cold_factor - 0.05) < 0.01

        # 第10次: 10/20 = 0.5
        for _ in range(9):
            cold_manager.increment_query_count("user_curve")
        state = cold_manager.get_state("user_curve")
        assert abs(state.cold_factor - 0.5) < 0.01

        # 第20次: 20/20 = 1.0
        for _ in range(10):
            cold_manager.increment_query_count("user_curve")
        state = cold_manager.get_state("user_curve")
        assert abs(state.cold_factor - 1.0) < 0.01

    def test_disabled(self, cold_manager, graph):
        """COLD_START_ENABLED=False 时 cold_factor=1.0"""
        with patch("core.cold_start.COLD_START_ENABLED", False):
            factor = cold_manager.get_cold_factor("user_001")
            assert factor == 1.0

    def test_no_user(self, cold_manager, graph):
        """无用户标识时 cold_factor=1.0"""
        factor = cold_manager.get_cold_factor(None)
        assert factor == 1.0

        factor = cold_manager.get_cold_factor("")
        assert factor == 1.0

    def test_increment_persistence(self, cold_manager, graph):
        """写入→读取一致性"""
        # 递增5次
        for _ in range(5):
            cold_manager.increment_query_count("user_persist")

        # 从数据库直接读取
        row = graph.conn.execute(
            "SELECT query_count FROM user_cold_start WHERE user_label = ?",
            ("user_persist",),
        ).fetchone()
        assert row[0] == 5

        # 通过 get_state 读取
        state = cold_manager.get_state("user_persist")
        assert state.query_count == 5

    def test_cache_hit(self, cold_manager, graph):
        """连续查询时缓存命中"""
        # 首次查询，加载到缓存
        state1 = cold_manager.get_state("user_cache")

        # 再次查询，应从缓存读取
        state2 = cold_manager.get_state("user_cache")

        assert state1.query_count == state2.query_count
        assert state1.cold_factor == state2.cold_factor

        # 验证缓存存在
        assert "user_cache" in cold_manager._cache

    def test_estimate_from_karma(self, cold_manager, graph):
        """无记录时从个人业力边数量估算"""
        # 插入一些个人业力边
        now = "2025-01-01T00:00:00+00:00"
        for i in range(KARMA_MAX_PAIRS * 3):
            graph.conn.execute(
                "INSERT INTO karma_edges_personal "
                "(user_label, source, target, relation, weight, source_tag, updated_at) "
                "VALUES (?, ?, ?, 'RELATED', 0.5, 'personal_karma', ?)",
                ("user_est", f"seed_{i}", f"target_{i}", now),
            )
        graph.conn.commit()

        # 清除缓存确保从数据库加载
        cold_manager.invalidate_cache()

        # 从个人业力边估算
        estimated = cold_manager._estimate_count_from_karma("user_est")
        # 3 * KARMA_MAX_PAIRS / KARMA_MAX_PAIRS = 3
        assert estimated == 3

        # 通过 get_state 验证
        state = cold_manager.get_state("user_est")
        assert state.query_count == 3

    def test_invalidate_cache(self, cold_manager, graph):
        """清除缓存"""
        # 加载到缓存
        cold_manager.get_state("user_invalidate")
        assert "user_invalidate" in cold_manager._cache

        # 清除指定用户缓存
        cold_manager.invalidate_cache("user_invalidate")
        assert "user_invalidate" not in cold_manager._cache

        # 再次加载
        cold_manager.get_state("user_invalidate")
        assert "user_invalidate" in cold_manager._cache

        # 清除全部缓存
        cold_manager.invalidate_cache()
        assert len(cold_manager._cache) == 0

    def test_increment_returns_count(self, cold_manager, graph):
        """increment_query_count 返回递增后的查询计数"""
        count = cold_manager.increment_query_count("user_ret")
        assert count == 1

        count = cold_manager.increment_query_count("user_ret")
        assert count == 2

    def test_increment_no_user(self, cold_manager, graph):
        """user_label 为 None 时 increment_query_count 返回 0"""
        count = cold_manager.increment_query_count(None)
        assert count == 0

    def test_cold_factor_capped_at_one(self, cold_manager, graph):
        """cold_factor 上限为 1.0"""
        # 增加到 100 次查询
        for _ in range(100):
            cold_manager.increment_query_count("user_cap")

        state = cold_manager.get_state("user_cap")
        assert state.cold_factor == 1.0

    def test_estimate_no_karma_edges(self, cold_manager, graph):
        """无个人业力边时估算为 0"""
        estimated = cold_manager._estimate_count_from_karma("user_no_karma")
        assert estimated == 0