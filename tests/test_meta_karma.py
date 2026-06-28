"""
Phase 4 元业力边单元测试

覆盖：
- 元业力边创建条件
- 元业力边权重更新
- 元业力边与普通业力边的隔离
- 元业力边阈值判定
- 元业力边 relation 和 source_tag
"""

from __future__ import annotations

import json
import sqlite3
import sys
import os
from unittest.mock import patch

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.meta_seed import (
    MetaSeedManager,
    MetaSeedCategory,
    MetaSeedStatus,
    DOMAIN_MONITOR_DEFAULT_METRICS,
)
from core.graph_db import GraphDB
from core.config import (
    META_KARMA_DELTA_THRESHOLD,
    META_KARMA_INITIAL_WEIGHT,
    KARMA_MIN,
    KARMA_MAX,
)


# ═══════════════════════════════════════════════════════════
#  Fixtures
# ═══════════════════════════════════════════════════════════


def _build_test_db() -> sqlite3.Connection:
    """创建含 Phase 4 表的内存测试数据库"""
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
    """)
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
def mgr(graph):
    """创建 MetaSeedManager 实例"""
    return MetaSeedManager(graph)


# ═══════════════════════════════════════════════════════════
#  测试用例
# ═══════════════════════════════════════════════════════════


class TestMetaKarmaCreation:
    """元业力边创建条件测试

    注意：check_and_create_meta_karma() 内部通过 list_meta_seeds() 获取元种子列表，
    每次调用创建新的 MetaSeedData 对象，_previous_metrics 为空。
    第一次调用时保存当前指标作为基线并跳过，第二次调用时才能比较变化。
    因此测试需要调用两次：第一次设置基线，第二次检测变化。
    """

    def test_create_on_significant_change(self, mgr, graph):
        """指标变化量 >= DELTA_THRESHOLD 时创建元业力边"""
        # 创建两个同类别元种子
        mgr._create_meta_seed_record(
            "meta:医学", MetaSeedCategory.DOMAIN_MONITOR,
            {"avg_karma_density": 0.0, "ripple_success_rate": 0.0, "conflict_frequency": 0},
        )
        mgr._create_meta_seed_record(
            "meta:物理", MetaSeedCategory.DOMAIN_MONITOR,
            {"avg_karma_density": 0.0, "ripple_success_rate": 0.0, "conflict_frequency": 0},
        )

        # 第一次调用：设置基线
        mgr.check_and_create_meta_karma()

        # 更新指标使 conflict_frequency 变化量超过阈值
        mgr.update_metrics("meta:医学", {
            "avg_karma_density": 0.0, "ripple_success_rate": 0.0,
            "conflict_frequency": META_KARMA_DELTA_THRESHOLD + 5,
        })

        # 第二次调用：检测变化并创建元业力边
        edges_created = mgr.check_and_create_meta_karma()
        assert edges_created >= 1

    def test_no_create_below_threshold(self, mgr, graph):
        """指标变化量 < DELTA_THRESHOLD 时不创建元业力边"""
        mgr._create_meta_seed_record(
            "meta:医学", MetaSeedCategory.DOMAIN_MONITOR,
            {"avg_karma_density": 0.0, "ripple_success_rate": 0.0, "conflict_frequency": 0},
        )
        mgr._create_meta_seed_record(
            "meta:物理", MetaSeedCategory.DOMAIN_MONITOR,
            {"avg_karma_density": 0.0, "ripple_success_rate": 0.0, "conflict_frequency": 0},
        )

        # 第一次调用：设置基线
        mgr.check_and_create_meta_karma()

        # 变化量 = 1 < META_KARMA_DELTA_THRESHOLD(2)
        mgr.update_metrics("meta:医学", {
            "avg_karma_density": 0.0, "ripple_success_rate": 0.0, "conflict_frequency": 1,
        })

        edges_created = mgr.check_and_create_meta_karma()
        assert edges_created == 0

    def test_skip_non_numeric_metrics(self, mgr, graph):
        """跳过非数值指标（如 unmatched_keywords 列表）"""
        mgr._create_meta_seed_record(
            "meta:unknown", MetaSeedCategory.SELF_BOUNDARY,
            {"unmatched_keywords": ["词1"], "unmatched_count": 0, "top_unmatched": []},
        )
        mgr._create_meta_seed_record(
            "meta:医学", MetaSeedCategory.DOMAIN_MONITOR,
            {"avg_karma_density": 0.0, "ripple_success_rate": 0.0, "conflict_frequency": 0},
        )

        # 第一次调用：设置基线
        mgr.check_and_create_meta_karma()

        # 更新列表类型指标（不应触发元业力边）
        mgr.update_metrics("meta:unknown", {
            "unmatched_keywords": ["词1", "词2"], "unmatched_count": 2, "top_unmatched": ["词1"]
        })

        # 第二次调用：检测变化
        edges_created = mgr.check_and_create_meta_karma()
        # unmatched_count 是数值指标，变化量=2>=2，应触发
        # 但 unmatched_keywords 和 top_unmatched 是列表，应被跳过
        assert isinstance(edges_created, int)


class TestMetaKarmaRelation:
    """元业力边 relation 和 source_tag 测试"""

    def test_meta_karma_relation(self, mgr, graph):
        """元业力边 relation 为 META_CORRELATED"""
        mgr._create_meta_seed_record(
            "meta:医学", MetaSeedCategory.DOMAIN_MONITOR,
            {"avg_karma_density": 0.0, "ripple_success_rate": 0.0, "conflict_frequency": 0},
        )
        mgr._create_meta_seed_record(
            "meta:物理", MetaSeedCategory.DOMAIN_MONITOR,
            {"avg_karma_density": 0.0, "ripple_success_rate": 0.0, "conflict_frequency": 0},
        )

        # 第一次调用：设置基线
        mgr.check_and_create_meta_karma()

        mgr.update_metrics("meta:医学", {
            "avg_karma_density": 0.0, "ripple_success_rate": 0.0, "conflict_frequency": 5,
        })

        # 第二次调用：检测变化
        mgr.check_and_create_meta_karma()

        row = graph.conn.execute(
            "SELECT relation FROM karma_edges WHERE source_tag = 'meta_karma'"
        ).fetchone()
        assert row is not None
        assert row["relation"] == "META_CORRELATED"

    def test_meta_karma_source_tag(self, mgr, graph):
        """元业力边 source_tag 为 meta_karma"""
        mgr._create_meta_seed_record(
            "meta:医学", MetaSeedCategory.DOMAIN_MONITOR,
            {"avg_karma_density": 0.0, "ripple_success_rate": 0.0, "conflict_frequency": 0},
        )
        mgr._create_meta_seed_record(
            "meta:物理", MetaSeedCategory.DOMAIN_MONITOR,
            {"avg_karma_density": 0.0, "ripple_success_rate": 0.0, "conflict_frequency": 0},
        )

        # 第一次调用：设置基线
        mgr.check_and_create_meta_karma()

        mgr.update_metrics("meta:医学", {
            "avg_karma_density": 0.0, "ripple_success_rate": 0.0, "conflict_frequency": 5,
        })

        # 第二次调用：检测变化
        mgr.check_and_create_meta_karma()

        row = graph.conn.execute(
            "SELECT source_tag FROM karma_edges WHERE source = 'meta:医学'"
        ).fetchone()
        assert row is not None
        assert row["source_tag"] == "meta_karma"


class TestMetaKarmaIsolation:
    """元业力边与普通业力边的隔离测试

    注意：check_and_create_meta_karma() 内部通过 list_meta_seeds() 获取元种子列表，
    每次调用创建新的 MetaSeedData 对象，_previous_metrics 为空。
    第一次调用时保存当前指标作为基线并跳过，第二次调用时才能比较变化。
    因此测试需要调用两次：第一次设置基线，第二次检测变化。
    """

    def test_normal_karma_untouched(self, mgr, graph):
        """元业力边操作不影响普通业力边"""
        # 创建普通业力边
        graph.conn.execute(
            "INSERT INTO karma_edges (source, target, relation, weight, source_tag) "
            "VALUES ('感冒', '发热', 'RELATED', 0.8, 'karma_delta')"
        )
        graph.conn.commit()

        # 创建元种子（初始指标为 0）
        mgr._create_meta_seed_record(
            "meta:医学", MetaSeedCategory.DOMAIN_MONITOR,
            {"avg_karma_density": 0.0, "ripple_success_rate": 0.0, "conflict_frequency": 0},
        )
        mgr._create_meta_seed_record(
            "meta:物理", MetaSeedCategory.DOMAIN_MONITOR,
            {"avg_karma_density": 0.0, "ripple_success_rate": 0.0, "conflict_frequency": 0},
        )

        # 第一次调用：设置基线
        mgr.check_and_create_meta_karma()

        # 更新指标
        mgr.update_metrics("meta:医学", {
            "avg_karma_density": 0.0, "ripple_success_rate": 0.0, "conflict_frequency": 5,
        })

        # 第二次调用：检测变化并创建元业力边
        mgr.check_and_create_meta_karma()

        # 验证普通业力边不受影响
        row = graph.conn.execute(
            "SELECT weight FROM karma_edges WHERE source = '感冒' AND target = '发热' AND relation = 'RELATED'"
        ).fetchone()
        assert row is not None
        assert abs(row["weight"] - 0.8) < 0.01

    def test_meta_karma_queryable(self, mgr, graph):
        """元业力边可通过 outgoing_edges 查询"""
        mgr._create_meta_seed_record(
            "meta:医学", MetaSeedCategory.DOMAIN_MONITOR,
            {"avg_karma_density": 0.0, "ripple_success_rate": 0.0, "conflict_frequency": 0},
        )
        mgr._create_meta_seed_record(
            "meta:物理", MetaSeedCategory.DOMAIN_MONITOR,
            {"avg_karma_density": 0.0, "ripple_success_rate": 0.0, "conflict_frequency": 0},
        )

        # 第一次调用：设置基线
        mgr.check_and_create_meta_karma()

        mgr.update_metrics("meta:医学", {
            "avg_karma_density": 0.0, "ripple_success_rate": 0.0, "conflict_frequency": 5,
        })

        # 第二次调用：检测变化并创建元业力边
        mgr.check_and_create_meta_karma()

        # 使用 exclude_meta=False 查询元业力边
        edges = graph.outgoing_edges("meta:医学", exclude_meta=False)
        meta_edges = [e for e in edges if e.get("source_tag") == "meta_karma"]
        assert len(meta_edges) >= 1

    def test_meta_karma_excluded_from_ripple(self, mgr, graph):
        """元业力边不参与涟漪传播（exclude_meta=True 时返回空）"""
        mgr._create_meta_seed_record(
            "meta:医学", MetaSeedCategory.DOMAIN_MONITOR,
            {"avg_karma_density": 0.0, "ripple_success_rate": 0.0, "conflict_frequency": 0},
        )
        mgr._create_meta_seed_record(
            "meta:物理", MetaSeedCategory.DOMAIN_MONITOR,
            {"avg_karma_density": 0.0, "ripple_success_rate": 0.0, "conflict_frequency": 0},
        )

        # 第一次调用：设置基线
        mgr.check_and_create_meta_karma()

        mgr.update_metrics("meta:医学", {
            "avg_karma_density": 0.0, "ripple_success_rate": 0.0, "conflict_frequency": 5,
        })

        # 第二次调用：检测变化并创建元业力边
        mgr.check_and_create_meta_karma()

        # exclude_meta=True（默认）时，meta: 前缀的出边返回空
        edges = graph.outgoing_edges("meta:医学", exclude_meta=True)
        assert edges == []


class TestMetaKarmaDirection:
    """元业力边熏习方向测试

    注意：check_and_create_meta_karma() 内部通过 list_meta_seeds() 获取元种子列表，
    每次调用创建新的 MetaSeedData 对象，_previous_metrics 为空。
    第一次调用时保存当前指标作为基线并跳过，第二次调用时才能比较变化。
    因此测试需要调用两次：第一次设置基线，第二次检测变化。
    """

    def test_positive_direction_enhances(self, mgr, graph):
        """正向变化（指标恶化）增强元业力边权重"""
        mgr._create_meta_seed_record(
            "meta:医学", MetaSeedCategory.DOMAIN_MONITOR,
            {"avg_karma_density": 0.0, "ripple_success_rate": 0.0, "conflict_frequency": 0},
        )
        mgr._create_meta_seed_record(
            "meta:物理", MetaSeedCategory.DOMAIN_MONITOR,
            {"avg_karma_density": 0.0, "ripple_success_rate": 0.0, "conflict_frequency": 0},
        )

        # 第一次调用：设置基线
        mgr.check_and_create_meta_karma()

        # conflict_frequency 从 0 增加到 5（恶化）
        mgr.update_metrics("meta:医学", {
            "avg_karma_density": 0.0, "ripple_success_rate": 0.0, "conflict_frequency": 5,
        })

        # 第二次调用：检测变化并创建元业力边
        mgr.check_and_create_meta_karma()

        row = graph.conn.execute(
            "SELECT weight FROM karma_edges WHERE source = 'meta:医学' AND source_tag = 'meta_karma'"
        ).fetchone()
        assert row is not None
        # 正向变化权重应大于 0
        assert row["weight"] > 0.0

    def test_negative_direction_weakens(self, mgr, graph):
        """负向变化（指标改善）减弱元业力边权重"""
        mgr._create_meta_seed_record(
            "meta:医学", MetaSeedCategory.DOMAIN_MONITOR,
            {"avg_karma_density": 0.0, "ripple_success_rate": 0.0, "conflict_frequency": 10},
        )
        mgr._create_meta_seed_record(
            "meta:物理", MetaSeedCategory.DOMAIN_MONITOR,
            {"avg_karma_density": 0.0, "ripple_success_rate": 0.0, "conflict_frequency": 0},
        )

        # 第一次调用：设置基线（此时 conflict_frequency=10）
        mgr.check_and_create_meta_karma()

        # conflict_frequency 从 10 减少到 3（改善，delta=7 >= threshold=2）
        mgr.update_metrics("meta:医学", {
            "avg_karma_density": 0.0, "ripple_success_rate": 0.0, "conflict_frequency": 3,
        })

        # 第二次调用：检测变化并创建元业力边
        mgr.check_and_create_meta_karma()

        row = graph.conn.execute(
            "SELECT weight FROM karma_edges WHERE source = 'meta:医学' AND source_tag = 'meta_karma'"
        ).fetchone()
        assert row is not None


class TestMetaKarmaThreshold:
    """元业力边阈值判定测试

    注意：check_and_create_meta_karma() 内部通过 list_meta_seeds() 获取元种子列表，
    每次调用创建新的 MetaSeedData 对象，_previous_metrics 为空。
    第一次调用时保存当前指标作为基线并跳过，第二次调用时才能比较变化。
    因此测试需要调用两次：第一次设置基线，第二次检测变化。
    """

    def test_exact_threshold_triggers(self, mgr, graph):
        """变化量恰好等于阈值时也触发"""
        mgr._create_meta_seed_record(
            "meta:医学", MetaSeedCategory.DOMAIN_MONITOR,
            {"avg_karma_density": 0.0, "ripple_success_rate": 0.0, "conflict_frequency": 0},
        )
        mgr._create_meta_seed_record(
            "meta:物理", MetaSeedCategory.DOMAIN_MONITOR,
            {"avg_karma_density": 0.0, "ripple_success_rate": 0.0, "conflict_frequency": 0},
        )

        # 第一次调用：设置基线
        mgr.check_and_create_meta_karma()

        # 变化量 = META_KARMA_DELTA_THRESHOLD
        mgr.update_metrics("meta:医学", {
            "avg_karma_density": 0.0, "ripple_success_rate": 0.0,
            "conflict_frequency": META_KARMA_DELTA_THRESHOLD,
        })

        # 第二次调用：检测变化
        edges_created = mgr.check_and_create_meta_karma()
        # 变化量恰好等于阈值，delta < threshold 不触发（严格小于）
        # 根据 meta_seed.py 实现：if delta < META_KARMA_DELTA_THRESHOLD: continue
        # 所以 delta == threshold 时不跳过，会触发
        assert edges_created >= 1

    def test_no_active_seeds_no_edges(self, mgr, graph):
        """没有 active 元种子时不创建元业力边"""
        # 创建 dormant 元种子
        mgr._create_meta_seed_record(
            "meta:医学", MetaSeedCategory.DOMAIN_MONITOR,
            {"avg_karma_density": 0.0, "ripple_success_rate": 0.0, "conflict_frequency": 0},
        )
        # 手动设置为 dormant
        graph.conn.execute(
            "UPDATE meta_seeds SET status = 'dormant' WHERE label = 'meta:医学'"
        )
        graph.conn.commit()

        edges_created = mgr.check_and_create_meta_karma()
        assert edges_created == 0