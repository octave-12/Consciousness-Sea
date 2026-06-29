"""
Phase 3 别名自动扩展器测试

测试 AliasExpander 的所有公共方法：
- record_backref_events: 记录回指事件并执行阈值判定
- get_alias_stats: 查询别名扩展统计
"""

from __future__ import annotations

import json
import sqlite3
import sys
import pathlib
from unittest.mock import patch

import pytest

_root = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_root))
sys.path.insert(0, str(_root / "backend" / "src"))

from consciousness_sea.learning.alias_expander import (
    AliasExpander,
    AliasExpansionResult,
    BackrefEvent,
    BackrefStats,
    BackrefStatus,
)
from consciousness_sea.domain.graph_db import GraphDB
from consciousness_sea.infrastructure.config import (
    ALIAS_AUTO_EXTEND,
    ALIAS_BACK_REF_THRESHOLD,
    ALIAS_CONFLICT_MARGIN,
    ALIAS_MIN_COUNT,
)


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
    """)

    # 插入测试种子
    seeds = [
        ("感冒", "感冒", "CONCEPT", "[]", "医学", "急性上呼吸道感染"),
        ("炎症", "炎症", "CONCEPT", "[]", "医学", "组织对损伤的防御反应"),
        ("热证", "热证", "CONCEPT", "[]", "中医", "中医热性证候"),
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
    return db


@pytest.fixture
def graph():
    """创建内存数据库的 GraphDB 实例"""
    conn = _build_test_db()
    g = _make_graph_db(conn)
    yield g
    g.close()


@pytest.fixture
def expander(graph):
    """创建 AliasExpander 实例"""
    return AliasExpander(graph)


# ═══════════════════════════════════════════════════════════
#  测试用例
# ═══════════════════════════════════════════════════════════


class TestAliasExpander:
    """AliasExpander 单元测试"""

    def test_record_backref_event_basic(self, expander, graph):
        """记录 ("着凉", "感冒") 回指事件"""
        events = [BackrefEvent(source_keyword="着凉", target_seed="感冒")]
        results = expander.record_backref_events(events)

        assert len(results) == 1
        result = results[0]
        assert result.keyword == "着凉"
        assert result.seed_label == "感冒"
        assert result.back_ref_rate == 1.0  # 首次: 1/1
        assert result.total_count == 1

        # 验证数据库记录
        row = graph.conn.execute(
            "SELECT * FROM alias_backref_events WHERE source_keyword = ? AND target_seed = ?",
            ("着凉", "感冒"),
        ).fetchone()
        assert row is not None
        assert row["ref_count"] == 1
        assert row["total_count"] == 1
        assert row["back_ref_rate"] == 1.0
        assert row["status"] == "tracking"

    def test_backref_rate_calculation(self, expander, graph):
        """8次回指/10次总出现 = 0.8

        先记录8次回指事件（ref_count=8, total_count=8），
        再记录2次未匹配关键词（total_count+2 → 10），
        最终 back_ref_rate = 8/10 = 0.8
        """
        # 记录8次回指事件
        for _ in range(8):
            expander.record_backref_events(
                [BackrefEvent(source_keyword="着凉", target_seed="感冒")]
            )

        # 记录2次未匹配关键词（total_count +2）
        for _ in range(2):
            expander.record_backref_events(
                [], unmatched_keywords=["着凉"]
            )

        row = graph.conn.execute(
            "SELECT ref_count, total_count, back_ref_rate FROM alias_backref_events "
            "WHERE source_keyword = ? AND target_seed = ?",
            ("着凉", "感冒"),
        ).fetchone()

        assert row["ref_count"] == 8
        assert row["total_count"] == 10
        assert abs(row["back_ref_rate"] - 0.8) < 0.01

    def test_alias_auto_add_threshold(self, expander, graph):
        """back_ref_rate >= 0.6 且 count >= 5 时自动添加别名

        当回指率和出现次数均超过阈值时，应自动将关键词追加为种子的别名。
        使用独立数据库避免 fixture 污染。
        """
        conn = _build_test_db()
        g = _make_graph_db(conn)
        exp = AliasExpander(g)

        # 记录8次回指事件 + 2次未匹配 → back_ref_rate=0.8, total_count=10
        all_results = []
        for _ in range(8):
            results = exp.record_backref_events(
                [BackrefEvent(source_keyword="着凉", target_seed="感冒")]
            )
            all_results.extend(results)
        for _ in range(2):
            exp.record_backref_events(
                [], unmatched_keywords=["着凉"]
            )

        # 在第5次时 count=5 >= 5 且 rate=1.0 >= 0.6，应触发 aliased
        aliased_results = [r for r in all_results if r.action == "aliased"]
        assert len(aliased_results) >= 1

        # 验证种子别名已更新
        row = g.conn.execute(
            "SELECT aliases FROM seeds WHERE label = ?", ("感冒",)
        ).fetchone()
        aliases = json.loads(row["aliases"])
        assert "着凉" in aliases

        g.close()

    def test_alias_append_no_overwrite(self, expander, graph):
        """已有 "伤风,风寒" 追加 "着凉" 后为 ["伤风","风寒","着凉"]"""
        # 预设种子已有别名
        graph.conn.execute(
            "UPDATE seeds SET aliases = ? WHERE label = ?",
            (json.dumps(["伤风", "风寒"], ensure_ascii=False), "感冒"),
        )
        graph.conn.commit()

        # 记录足够回指事件使 "着凉" 达到阈值
        for _ in range(10):
            expander.record_backref_events(
                [BackrefEvent(source_keyword="着凉", target_seed="感冒")]
            )

        # 验证别名追加而非覆盖
        row = graph.conn.execute(
            "SELECT aliases FROM seeds WHERE label = ?", ("感冒",)
        ).fetchone()
        aliases = json.loads(row["aliases"])
        assert aliases == ["伤风", "风寒", "着凉"]

    def test_alias_deduplication(self, expander, graph):
        """别名已存在时不重复追加"""
        # 预设种子已有 "着凉" 别名
        graph.conn.execute(
            "UPDATE seeds SET aliases = ? WHERE label = ?",
            (json.dumps(["着凉"], ensure_ascii=False), "感冒"),
        )
        graph.conn.commit()

        # 记录足够回指事件
        for _ in range(10):
            expander.record_backref_events(
                [BackrefEvent(source_keyword="着凉", target_seed="感冒")]
            )

        # 验证别名不重复
        row = graph.conn.execute(
            "SELECT aliases FROM seeds WHERE label = ?", ("感冒",)
        ).fetchone()
        aliases = json.loads(row["aliases"])
        assert aliases.count("着凉") == 1

    def test_alias_conflict_detection(self, expander, graph):
        """同一词回指到多个种子时检测冲突

        "上火" 回指到 "炎症" 和 "热证"，当两者回指率差距 < ALIAS_CONFLICT_MARGIN 时标记冲突。
        使用独立数据库避免 fixture 污染，直接构造冲突场景。
        """
        conn = _build_test_db()
        g = _make_graph_db(conn)
        exp = AliasExpander(g)

        now = "2025-01-01T00:00:00+00:00"

        # 插入两条记录：rate 接近
        # "上火" → "炎症" rate=0.8
        g.conn.execute(
            "INSERT INTO alias_backref_events "
            "(source_keyword, target_seed, ref_count, total_count, back_ref_rate, status, created_at, updated_at) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            ("上火", "炎症", 8, 10, 0.8, "tracking", now, now),
        )
        # "上火" → "热证" rate=0.7
        g.conn.execute(
            "INSERT INTO alias_backref_events "
            "(source_keyword, target_seed, ref_count, total_count, back_ref_rate, status, created_at, updated_at) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            ("上火", "热证", 7, 10, 0.7, "tracking", now, now),
        )
        g.conn.commit()

        # 调用冲突检测
        # 差值 = 0.8 - 0.7 = 0.1 < 0.2 → 冲突
        has_conflict = exp._detect_conflict("上火", "炎症", 0.8)
        assert has_conflict is True

        # 再记录一次回指事件触发阈值判定
        results = exp.record_backref_events(
            [BackrefEvent(source_keyword="上火", target_seed="炎症")]
        )

        # 结果应为 conflicted
        result = results[0]
        assert result.action == "conflicted"

        # 验证种子别名未被修改
        row = g.conn.execute(
            "SELECT aliases FROM seeds WHERE label = ?", ("炎症",)
        ).fetchone()
        aliases = json.loads(row["aliases"])
        assert "上火" not in aliases

        g.close()

    def test_alias_conflict_margin(self, expander, graph):
        """差值 < ALIAS_CONFLICT_MARGIN 时标记待审核"""
        conn = _build_test_db()
        g = _make_graph_db(conn)
        exp = AliasExpander(g)
        now = "2025-01-01T00:00:00+00:00"

        # 差值 = 0.15 < 0.2 → 冲突
        g.conn.execute(
            "INSERT INTO alias_backref_events "
            "(source_keyword, target_seed, ref_count, total_count, back_ref_rate, status, created_at, updated_at) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            ("头晕", "感冒", 8, 10, 0.8, "tracking", now, now),
        )
        g.conn.execute(
            "INSERT INTO alias_backref_events "
            "(source_keyword, target_seed, ref_count, total_count, back_ref_rate, status, created_at, updated_at) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            ("头晕", "发热", 65, 100, 0.65, "tracking", now, now),
        )
        g.conn.commit()

        has_conflict = exp._detect_conflict("头晕", "感冒", 0.8)
        # 差值 = 0.8 - 0.65 = 0.15 < 0.2 → 冲突
        assert has_conflict is True

        # 差值 >= 0.2 → 不冲突
        g.conn.execute(
            "DELETE FROM alias_backref_events WHERE source_keyword = '头晕'"
        )
        g.conn.execute(
            "INSERT INTO alias_backref_events "
            "(source_keyword, target_seed, ref_count, total_count, back_ref_rate, status, created_at, updated_at) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            ("头晕", "感冒", 9, 10, 0.9, "tracking", now, now),
        )
        g.conn.execute(
            "INSERT INTO alias_backref_events "
            "(source_keyword, target_seed, ref_count, total_count, back_ref_rate, status, created_at, updated_at) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            ("头晕", "发热", 5, 10, 0.5, "tracking", now, now),
        )
        g.conn.commit()

        has_conflict = exp._detect_conflict("头晕", "感冒", 0.9)
        # 差值 = 0.9 - 0.5 = 0.4 >= 0.2 → 不冲突
        assert has_conflict is False

        g.close()

    def test_alias_disabled(self, expander, graph):
        """ALIAS_AUTO_EXTEND=False 时仅记录统计不修改 seeds 表"""
        with patch("consciousness_sea.learning.alias_expander.ALIAS_AUTO_EXTEND", False):
            # 记录足够回指事件
            for _ in range(10):
                results = expander.record_backref_events(
                    [BackrefEvent(source_keyword="着凉", target_seed="感冒")]
                )

            # 结果应为 disabled
            assert all(r.action == "disabled" for r in results)

            # 统计记录应存在
            row = graph.conn.execute(
                "SELECT COUNT(*) FROM alias_backref_events WHERE source_keyword = ?",
                ("着凉",),
            ).fetchone()
            assert row[0] > 0

            # 种子别名不应被修改
            row = graph.conn.execute(
                "SELECT aliases FROM seeds WHERE label = ?", ("感冒",)
            ).fetchone()
            aliases = json.loads(row["aliases"])
            assert "着凉" not in aliases

    def test_unmatched_keyword_tracking(self, expander, graph):
        """未匹配关键词仅更新 total_count"""
        # 先记录一次回指事件
        expander.record_backref_events(
            [BackrefEvent(source_keyword="着凉", target_seed="感冒")]
        )

        # 记录未匹配关键词
        expander.record_backref_events([], unmatched_keywords=["着凉"])

        # 验证 total_count 增加但 ref_count 不变
        row = graph.conn.execute(
            "SELECT ref_count, total_count FROM alias_backref_events "
            "WHERE source_keyword = ? AND target_seed = ?",
            ("着凉", "感冒"),
        ).fetchone()
        assert row["ref_count"] == 1
        assert row["total_count"] == 2

    def test_unmatched_keyword_new_entry(self, expander, graph):
        """全新未匹配关键词创建 __none__ 记录"""
        expander.record_backref_events([], unmatched_keywords=["未知词"])

        # 应创建 (未知词, __none__) 记录
        row = graph.conn.execute(
            "SELECT * FROM alias_backref_events WHERE source_keyword = ?",
            ("未知词",),
        ).fetchone()
        assert row is not None
        assert row["target_seed"] == "__none__"
        assert row["ref_count"] == 0
        assert row["total_count"] == 1
        assert row["back_ref_rate"] == 0.0

    def test_get_alias_stats(self, expander, graph):
        """统计查询返回正确格式"""
        # 创建一些数据 — 使用独立数据库避免 fixture 污染
        conn = _build_test_db()
        g = _make_graph_db(conn)
        exp = AliasExpander(g)

        # 记录3次回指事件（不会触发别名，因为 count=3 < 5）
        for _ in range(3):
            exp.record_backref_events(
                [BackrefEvent(source_keyword="着凉", target_seed="感冒")]
            )

        stats = exp.get_alias_stats()

        assert "total_tracked" in stats
        assert "total_aliased" in stats
        assert "total_conflicted" in stats
        assert "recent_aliases" in stats
        assert "conflicted_items" in stats
        assert isinstance(stats["total_tracked"], int)
        assert isinstance(stats["recent_aliases"], list)
        assert isinstance(stats["conflicted_items"], list)

        # 3次回指后 count=3 < 5，状态仍为 tracking
        assert stats["total_tracked"] >= 1

        g.close()

    def test_threshold_not_met_count(self, expander, graph):
        """total_count < ALIAS_MIN_COUNT 时返回 threshold_not_met"""
        # 只记录3次（< ALIAS_MIN_COUNT=5）
        for _ in range(3):
            results = expander.record_backref_events(
                [BackrefEvent(source_keyword="着凉", target_seed="感冒")]
            )

        # 结果应为 threshold_not_met（因为 total_count=3 < 5）
        assert all(r.action == "threshold_not_met" for r in results)

    def test_threshold_not_met_rate(self, expander, graph):
        """back_ref_rate < ALIAS_BACK_REF_THRESHOLD 时返回 threshold_not_met"""
        # 记录3次回指 + 7次未匹配 → rate = 3/10 = 0.3 < 0.6
        for _ in range(3):
            expander.record_backref_events(
                [BackrefEvent(source_keyword="着凉", target_seed="感冒")]
            )
        for _ in range(7):
            expander.record_backref_events([], unmatched_keywords=["着凉"])

        # 再记录1次回指触发判定
        results = expander.record_backref_events(
            [BackrefEvent(source_keyword="着凉", target_seed="感冒")]
        )

        # rate = 4/11 ≈ 0.36 < 0.6 → threshold_not_met
        assert results[0].action == "threshold_not_met"

    def test_already_aliased_status(self, expander, graph):
        """status == 'aliased' 时返回 already_aliased"""
        now = "2025-01-01T00:00:00+00:00"

        # 直接插入已 aliased 状态的记录
        graph.conn.execute(
            "INSERT INTO alias_backref_events "
            "(source_keyword, target_seed, ref_count, total_count, back_ref_rate, status, created_at, updated_at) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            ("着凉", "感冒", 8, 10, 0.8, "aliased", now, now),
        )
        graph.conn.commit()

        # 再记录回指事件
        results = expander.record_backref_events(
            [BackrefEvent(source_keyword="着凉", target_seed="感冒")]
        )

        assert results[0].action == "already_aliased"