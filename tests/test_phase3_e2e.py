"""
Phase 3 端到端场景验收测试

覆盖 spec.md 的 7 个验收场景：
- 场景1：别名自动扩展——"着凉"→"感冒"
- 场景2：别名冲突——"上火"回指到"炎症"和"热证"
- 场景3：候选种子——"DeepSeek"从候选到正式种子
- 场景4：新用户冷启动——第1次 cold_factor=0.05，第21次=1.0
- 场景5：全量回滚——恢复到检查点时刻的业力状态
- 场景6：单边回滚——仅恢复指定边的权重
- 场景7：功能降级兼容性——所有开关关闭时行为与 Phase 2 一致
"""

from __future__ import annotations

import json
import sqlite3
import sys
import os
import time
from pathlib import Path
from unittest.mock import patch

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.alias_expander import AliasExpander, BackrefEvent, BackrefStatus
from core.seed_candidate import SeedCandidateManager, CandidateStatus
from core.cold_start import ColdStartManager
from core.checkpoint import CheckpointManager, CheckpointSource
from core.graph_db import GraphDB
from core.config import (
    ALIAS_AUTO_EXTEND,
    ALIAS_BACK_REF_THRESHOLD,
    ALIAS_CONFLICT_MARGIN,
    ALIAS_MIN_COUNT,
    CANDIDATE_SEED_AUTO_CREATE,
    CANDIDATE_SEED_MIN_COUNT,
    CANDIDATE_SEED_PROMOTE_COUNT,
    COLD_START_ENABLED,
    COLD_START_QUERIES,
    CHECKPOINT_RETAIN_COUNT,
)


# ═══════════════════════════════════════════════════════════
#  Fixtures
# ═══════════════════════════════════════════════════════════


def _build_e2e_db() -> sqlite3.Connection:
    """创建端到端测试用内存数据库（含所有 Phase 3 表）"""
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
        CREATE TABLE user_cold_start (
            user_label  TEXT    PRIMARY KEY,
            query_count INTEGER NOT NULL DEFAULT 0,
            updated_at  TEXT    NOT NULL
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
        ("炎症", "炎症", "CONCEPT", "[]", "医学", "组织对损伤的防御反应"),
        ("热证", "热证", "CONCEPT", "[]", "中医", "中医热性证候"),
        ("发热", "发热", "CONCEPT", "[]", "医学", "体温升高"),
        ("咳嗽", "咳嗽", "CONCEPT", "[]", "医学", "cough"),
    ]
    conn.executemany(
        "INSERT INTO seeds (id, label, type, aliases, domain, definition) VALUES (?, ?, ?, ?, ?, ?)",
        seeds,
    )

    # 插入测试业力边
    edges = [
        ("感冒", "发热", "RELATED", 0.8, "karma_delta"),
        ("感冒", "咳嗽", "RELATED", 0.6, "karma_delta"),
        ("炎症", "发热", "RELATED", 0.7, "karma_delta"),
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
    return db


@pytest.fixture
def graph():
    """创建端到端测试 GraphDB 实例"""
    conn = _build_e2e_db()
    g = _make_graph_db(conn)
    yield g
    g.close()


# ═══════════════════════════════════════════════════════════
#  场景1：别名自动扩展——"着凉"→"感冒"
# ═══════════════════════════════════════════════════════════


class TestScenario1AliasExpansion:
    """场景1：别名自动扩展——"着凉"→"感冒" """

    def test_alias_auto_expand(self, graph):
        """完整流程：记录回指 → 统计回指率 → 达到阈值 → 自动追加别名"""
        expander = AliasExpander(graph)

        # 阶段1：记录回指事件，回指率逐步上升
        for i in range(8):
            results = expander.record_backref_events(
                [BackrefEvent(source_keyword="着凉", target_seed="感冒")]
            )

        # 阶段2：记录未匹配关键词，降低回指率
        for _ in range(2):
            expander.record_backref_events([], unmatched_keywords=["着凉"])

        # 此时 ref_count=8, total_count=10, rate=0.8 >= 0.6
        # 但 total_count=10 >= 5，应已触发自动添加
        # 检查最后一次有 target_seed 的结果
        last_result = results[-1] if results else None
        # 由于8次回指后 rate=1.0 > 0.6, count=8 >= 5，应在第5次时就已触发
        # 但第5次时 rate=1.0, count=5，满足条件

        # 验证别名已添加
        row = graph.conn.execute(
            "SELECT aliases FROM seeds WHERE label = ?", ("感冒",)
        ).fetchone()
        aliases = json.loads(row["aliases"])
        assert "着凉" in aliases

        # 验证回指事件状态为 aliased
        row = graph.conn.execute(
            "SELECT status FROM alias_backref_events "
            "WHERE source_keyword = ? AND target_seed = ?",
            ("着凉", "感冒"),
        ).fetchone()
        assert row["status"] == "aliased"


# ═══════════════════════════════════════════════════════════
#  场景2：别名冲突——"上火"回指到"炎症"和"热证"
# ═══════════════════════════════════════════════════════════


class TestScenario2AliasConflict:
    """场景2：别名冲突——"上火"回指到"炎症"和"热证" """

    def test_alias_conflict(self, graph):
        """当两个种子的回指率接近时，标记冲突而非自动添加"""
        expander = AliasExpander(graph)

        now = "2025-01-01T00:00:00+00:00"

        # 直接构造冲突场景：两个 target 回指率接近
        # "上火" → "炎症" rate=0.8
        graph.conn.execute(
            "INSERT INTO alias_backref_events "
            "(source_keyword, target_seed, ref_count, total_count, back_ref_rate, status, created_at, updated_at) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            ("上火", "炎症", 8, 10, 0.8, "tracking", now, now),
        )
        # "上火" → "热证" rate=0.7
        graph.conn.execute(
            "INSERT INTO alias_backref_events "
            "(source_keyword, target_seed, ref_count, total_count, back_ref_rate, status, created_at, updated_at) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            ("上火", "热证", 7, 10, 0.7, "tracking", now, now),
        )
        graph.conn.commit()

        # 再记录一次回指事件触发阈值判定
        results = expander.record_backref_events(
            [BackrefEvent(source_keyword="上火", target_seed="炎症")]
        )

        # 差值 = 0.8 - 0.7 = 0.1 < 0.2 → 冲突
        result = results[0]
        assert result.action == "conflicted"

        # 验证种子别名未被修改
        row = graph.conn.execute(
            "SELECT aliases FROM seeds WHERE label = ?", ("炎症",)
        ).fetchone()
        aliases = json.loads(row["aliases"])
        assert "上火" not in aliases

        # 验证冲突状态
        row = graph.conn.execute(
            "SELECT status FROM alias_backref_events "
            "WHERE source_keyword = ? AND target_seed = ?",
            ("上火", "炎症"),
        ).fetchone()
        assert row["status"] == "conflicted"


# ═══════════════════════════════════════════════════════════
#  场景3：候选种子——"DeepSeek"从候选到正式种子
# ═══════════════════════════════════════════════════════════


class TestScenario3CandidateSeed:
    """场景3：候选种子——"DeepSeek"从候选到正式种子"""

    def test_candidate_to_seed(self, graph):
        """完整流程：未匹配关键词 → 候选种子 → 正式种子"""
        manager = SeedCandidateManager(graph)

        # 阶段1：处理未匹配关键词，达到 MIN_COUNT 创建候选种子
        for _ in range(CANDIDATE_SEED_MIN_COUNT):
            manager.process_unmatched_keywords(
                ["DeepSeek"], co_occur_seeds=["感冒", "炎症"]
            )

        # 验证候选种子已创建
        row = graph.conn.execute(
            "SELECT * FROM candidate_seeds WHERE label = ?", ("DeepSeek",)
        ).fetchone()
        assert row is not None
        assert row["status"] == "candidate"

        # 阶段2：继续处理直到达到 PROMOTE_COUNT
        remaining = CANDIDATE_SEED_PROMOTE_COUNT - CANDIDATE_SEED_MIN_COUNT
        for _ in range(remaining):
            manager.process_unmatched_keywords(
                ["DeepSeek"], co_occur_seeds=["感冒", "炎症"]
            )

        # 阶段3：升级为正式种子
        result = manager.promote_candidate("DeepSeek")

        assert result.success is True
        assert result.domain == "医学"  # 基于共现种子推断
        assert result.initial_edges > 0

        # 验证正式种子已创建
        row = graph.conn.execute(
            "SELECT * FROM seeds WHERE label = ?", ("DeepSeek",)
        ).fetchone()
        assert row is not None
        assert row["domain"] == "医学"

        # 验证初始业力边
        row = graph.conn.execute(
            "SELECT * FROM karma_edges WHERE source = ? AND target = ?",
            ("DeepSeek", "感冒"),
        ).fetchone()
        assert row is not None
        assert row["source_tag"] == "candidate_promotion"

        # 验证候选种子状态已更新
        row = graph.conn.execute(
            "SELECT status FROM candidate_seeds WHERE label = ?", ("DeepSeek",)
        ).fetchone()
        assert row["status"] == "promoted"


# ═══════════════════════════════════════════════════════════
#  场景4：新用户冷启动
# ═══════════════════════════════════════════════════════════


class TestScenario4ColdStart:
    """场景4：新用户冷启动——第1次 cold_factor=0.05，第21次=1.0"""

    def test_cold_start_lifecycle(self, graph):
        """完整冷启动周期"""
        cold_mgr = ColdStartManager(graph)

        # 第1次查询
        count = cold_mgr.increment_query_count("new_user")
        assert count == 1
        factor = cold_mgr.get_cold_factor("new_user")
        assert abs(factor - 0.05) < 0.01  # 1/20 = 0.05

        # 第10次查询
        for _ in range(9):
            cold_mgr.increment_query_count("new_user")
        factor = cold_mgr.get_cold_factor("new_user")
        assert abs(factor - 0.5) < 0.01  # 10/20 = 0.5

        # 第20次查询（冷启动期最后）
        for _ in range(10):
            cold_mgr.increment_query_count("new_user")
        factor = cold_mgr.get_cold_factor("new_user")
        assert abs(factor - 1.0) < 0.01  # 20/20 = 1.0

        # 第21次查询（冷启动期结束）
        cold_mgr.increment_query_count("new_user")
        state = cold_mgr.get_state("new_user")
        assert state.is_cold_start is False
        assert state.cold_factor == 1.0


# ═══════════════════════════════════════════════════════════
#  场景5：全量回滚
# ═══════════════════════════════════════════════════════════


class TestScenario5FullRollback:
    """场景5：全量回滚——恢复到检查点时刻的业力状态"""

    def test_full_rollback(self, tmp_path):
        """完整流程：创建检查点 → 修改业力 → 回滚 → 验证恢复"""
        # 使用独立数据库避免 fixture 污染
        conn = _build_e2e_db()
        g = _make_graph_db(conn)
        cp_mgr = CheckpointManager(g, checkpoint_dir=str(tmp_path / "cp"))

        # 阶段1：创建检查点
        meta = cp_mgr.create_checkpoint(tag="before_modification")
        assert meta.edge_count == 3  # 3条业力边

        # 阶段2：修改业力边
        g.conn.execute(
            "UPDATE karma_edges SET weight = 0.99 WHERE source = '感冒' AND target = '发热'"
        )
        g.conn.execute(
            "INSERT INTO karma_edges (source, target, relation, weight, source_tag) "
            "VALUES ('热证', '咳嗽', 'RELATED', 0.5, 'test')"
        )
        g.conn.commit()

        # 验证修改已生效
        row = g.conn.execute(
            "SELECT weight FROM karma_edges WHERE source = '感冒' AND target = '发热' AND relation = 'RELATED'"
        ).fetchone()
        assert abs(row["weight"] - 0.99) < 0.01

        row_count = g.conn.execute("SELECT COUNT(*) FROM karma_edges").fetchone()
        assert row_count[0] == 4  # 原来3条 + 新增1条

        # 等待1秒确保 pre_rollback 检查点 ID 不冲突
        time.sleep(1.0)

        # 阶段3：执行全量回滚
        result = cp_mgr.rollback(meta.checkpoint_id, mode="full")
        assert result.success is True
        assert result.edges_affected == 3

        # 阶段4：验证业力状态已恢复
        row = g.conn.execute(
            "SELECT weight FROM karma_edges WHERE source = '感冒' AND target = '发热' AND relation = 'RELATED'"
        ).fetchone()
        assert abs(row["weight"] - 0.8) < 0.01

        # 新增的边应被删除
        row = g.conn.execute(
            "SELECT * FROM karma_edges WHERE source = '热证' AND target = '咳嗽' AND relation = 'RELATED'"
        ).fetchone()
        assert row is None

        # 边数应恢复到3
        row_count = g.conn.execute("SELECT COUNT(*) FROM karma_edges").fetchone()
        assert row_count[0] == 3

        g.close()


# ═══════════════════════════════════════════════════════════
#  场景6：单边回滚
# ═══════════════════════════════════════════════════════════


class TestScenario6SingleRollback:
    """场景6：单边回滚——仅恢复指定边的权重"""

    def test_single_rollback(self, tmp_path):
        """完整流程：创建检查点 → 修改多条边 → 单边回滚 → 验证仅指定边恢复"""
        # 使用独立数据库避免 fixture 污染
        conn = _build_e2e_db()
        g = _make_graph_db(conn)
        cp_mgr = CheckpointManager(g, checkpoint_dir=str(tmp_path / "cp"))

        # 阶段1：创建检查点
        meta = cp_mgr.create_checkpoint(tag="single_before")

        # 阶段2：修改多条业力边
        g.conn.execute(
            "UPDATE karma_edges SET weight = 0.99 WHERE source = '感冒' AND target = '发热'"
        )
        g.conn.execute(
            "UPDATE karma_edges SET weight = 0.01 WHERE source = '感冒' AND target = '咳嗽'"
        )
        g.conn.commit()

        # 等待1秒确保 pre_rollback 检查点 ID 不冲突
        time.sleep(1.0)

        # 阶段3：仅回滚 "感冒→发热" 边
        result = cp_mgr.rollback(
            meta.checkpoint_id,
            mode="single",
            edges=[{"source": "感冒", "target": "发热", "relation": "RELATED"}],
        )
        assert result.success is True
        assert result.edges_affected == 1

        # 阶段4：验证仅指定边恢复
        row = g.conn.execute(
            "SELECT weight FROM karma_edges WHERE source = '感冒' AND target = '发热' AND relation = 'RELATED'"
        ).fetchone()
        assert abs(row["weight"] - 0.8) < 0.01  # 已恢复

        # 另一条边未恢复
        row = g.conn.execute(
            "SELECT weight FROM karma_edges WHERE source = '感冒' AND target = '咳嗽' AND relation = 'RELATED'"
        ).fetchone()
        assert abs(row["weight"] - 0.01) < 0.01  # 未恢复

        g.close()


# ═══════════════════════════════════════════════════════════
#  场景7：功能降级兼容性
# ═══════════════════════════════════════════════════════════


class TestScenario7DegradationCompat:
    """场景7：功能降级兼容性——所有开关关闭时行为与 Phase 2 一致"""

    def test_all_features_disabled(self, graph):
        """所有 Phase 3 开关关闭时，系统行为与 Phase 2 一致"""
        expander = AliasExpander(graph)
        candidate_mgr = SeedCandidateManager(graph)
        cold_mgr = ColdStartManager(graph)

        with (
            patch("core.alias_expander.ALIAS_AUTO_EXTEND", False),
            patch("core.seed_candidate.CANDIDATE_SEED_AUTO_CREATE", False),
            patch("core.cold_start.COLD_START_ENABLED", False),
        ):
            # 1. 别名扩展：仅记录统计，不修改 seeds 表
            for _ in range(10):
                results = expander.record_backref_events(
                    [BackrefEvent(source_keyword="着凉", target_seed="感冒")]
                )
            assert all(r.action == "disabled" for r in results)

            # 验证种子别名未修改
            row = graph.conn.execute(
                "SELECT aliases FROM seeds WHERE label = ?", ("感冒",)
            ).fetchone()
            aliases = json.loads(row["aliases"])
            assert "着凉" not in aliases

            # 2. 候选种子：不创建候选种子
            processed = candidate_mgr.process_unmatched_keywords(
                ["DeepSeek"], co_occur_seeds=["感冒"]
            )
            assert processed == 0

            # 3. 冷启动：cold_factor 始终为 1.0
            factor = cold_mgr.get_cold_factor("user_001")
            assert factor == 1.0

            # 新用户也返回 1.0
            cold_mgr.increment_query_count("new_user")
            factor = cold_mgr.get_cold_factor("new_user")
            assert factor == 1.0

    def test_alias_disabled_stats_still_recorded(self, graph):
        """别名扩展禁用时，统计仍被记录"""
        expander = AliasExpander(graph)

        with patch("core.alias_expander.ALIAS_AUTO_EXTEND", False):
            expander.record_backref_events(
                [BackrefEvent(source_keyword="着凉", target_seed="感冒")]
            )

            # 统计记录应存在
            row = graph.conn.execute(
                "SELECT * FROM alias_backref_events WHERE source_keyword = ? AND target_seed = ?",
                ("着凉", "感冒"),
            ).fetchone()
            assert row is not None
            assert row["ref_count"] == 1

    def test_cold_start_disabled_increment_still_works(self, graph):
        """冷启动禁用时，increment_query_count 仍正常工作"""
        cold_mgr = ColdStartManager(graph)

        with patch("core.cold_start.COLD_START_ENABLED", False):
            count = cold_mgr.increment_query_count("user_001")
            assert count == 1

            # 数据库记录应存在
            row = graph.conn.execute(
                "SELECT query_count FROM user_cold_start WHERE user_label = ?",
                ("user_001",),
            ).fetchone()
            assert row[0] == 1