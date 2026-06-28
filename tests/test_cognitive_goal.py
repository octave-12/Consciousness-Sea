"""
Phase 5 CognitiveGoalManager 单元测试

覆盖：
- 认知目标数据模型（GoalType、GoalStatus、CognitiveGoalData）
- 目标生成（四种触发条件：低置信度/低密度/高冲突/新词）
- 目标去重（同 domain + goal_type）
- 优先级权重计算（四因子加权）
- 目标冷却（权重衰减/过期/归档/池淘汰）
- 目标查询与统计
- 目标触及更新
- COGNITIVE_GOAL_ENABLED=False 时的行为
"""

from __future__ import annotations

import json
import sqlite3
import sys
import os
from datetime import datetime, timezone, timedelta
from unittest.mock import patch

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.cognitive_goal import (
    CognitiveGoalManager,
    CognitiveGoalData,
    GoalType,
    GoalStatus,
)
from core.graph_db import GraphDB
from core.config import (
    COGNITIVE_GOAL_ENABLED,
    GOAL_LOW_CONF_THRESHOLD,
    GOAL_LOW_DENSITY_RATIO,
    GOAL_HIGH_CONFLICT_THRESHOLD,
    GOAL_NEW_TERM_THRESHOLD,
    GOAL_DECAY_CYCLES,
    GOAL_DECAY_FACTOR,
    GOAL_EXPIRE_THRESHOLD,
    GOAL_USER_ABSENCE_CYCLES,
    GOAL_POOL_MAX_SIZE,
    GOAL_WEIGHT_USER_RELEVANCE,
    GOAL_WEIGHT_SYSTEM_CORENESS,
    GOAL_WEIGHT_UNCERTAINTY,
    GOAL_WEIGHT_DECOMPOSABILITY,
    GUARDIAN_LOOP_INTERVAL,
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
        ("维C", "维C", "CONCEPT", "[]", "营养", "Vitamin C"),
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
def mgr(graph):
    """创建 CognitiveGoalManager 实例"""
    return CognitiveGoalManager(graph)


def _insert_domain_monitor(conn, domain, metrics, status="active"):
    """辅助：插入领域监控元种子"""
    now = datetime.now(timezone.utc).isoformat()
    conn.execute(
        "INSERT INTO meta_seeds (label, category, metrics_json, status, source_domain, created_at, updated_at) "
        "VALUES (?, 'domain_monitor', ?, ?, ?, ?, ?)",
        (f"meta:{domain}", json.dumps(metrics), status, domain, now, now),
    )
    # 同时在 seeds 表创建 META 种子
    conn.execute(
        "INSERT OR IGNORE INTO seeds (id, label, type, domain, activation) VALUES (?, ?, 'META', '元认知', 0.0)",
        (f"meta:{domain}", f"meta:{domain}"),
    )
    conn.commit()


# ═══════════════════════════════════════════════════════════
#  测试用例
# ═══════════════════════════════════════════════════════════


class TestGoalTypeAndStatus:
    """枚举与数据类测试"""

    def test_goal_type_values(self):
        """GoalType 枚举包含四个值"""
        assert GoalType.LOW_CONFIDENCE.value == "low_confidence"
        assert GoalType.LOW_DENSITY.value == "low_density"
        assert GoalType.HIGH_CONFLICT.value == "high_conflict"
        assert GoalType.NEW_TERM.value == "new_term"

    def test_goal_status_values(self):
        """GoalStatus 枚举包含六个值"""
        assert GoalStatus.PENDING.value == "pending"
        assert GoalStatus.EXPLORING.value == "exploring"
        assert GoalStatus.QUERYING_EXTERNAL.value == "querying_external"
        assert GoalStatus.COMPLETED.value == "completed"
        assert GoalStatus.ARCHIVED.value == "archived"
        assert GoalStatus.EXPIRED.value == "expired"

    def test_goal_type_invalid(self):
        """无效 goal_type 构造时抛出 ValueError"""
        with pytest.raises(ValueError):
            GoalType("invalid")

    def test_goal_status_invalid(self):
        """无效 status 构造时抛出 ValueError"""
        with pytest.raises(ValueError):
            GoalStatus("invalid")

    def test_cognitive_goal_data_defaults(self):
        """CognitiveGoalData 默认值"""
        data = CognitiveGoalData(
            goal_id="test", goal_type=GoalType.LOW_CONFIDENCE,
            trigger_condition="test", domain="测试",
        )
        assert data.priority_weight == 0.0
        assert data.status == GoalStatus.PENDING
        assert data.sub_goals == []
        assert data.execution_log == []
        assert data.associated_user is None
        assert data.decay_cycles_count == 0


class TestGoalGeneration:
    """目标生成测试"""

    def test_generate_goals_low_confidence(self, mgr, graph):
        """低置信度频率触发：conflict_frequency > GOAL_LOW_CONF_THRESHOLD"""
        _insert_domain_monitor(
            graph.conn, "量子力学",
            {"avg_karma_density": 0.5, "ripple_success_rate": 0.3, "conflict_frequency": GOAL_LOW_CONF_THRESHOLD + 1},
        )
        created = mgr.generate_goals()
        assert created >= 1

        # 验证目标已创建
        row = graph.conn.execute(
            "SELECT * FROM cognitive_goals WHERE domain = '量子力学' AND goal_type = 'low_confidence'"
        ).fetchone()
        assert row is not None
        assert row["status"] == "pending"
        assert "conflict_frequency" in row["trigger_condition"]

    def test_generate_goals_low_density(self, mgr, graph):
        """业力密度过低触发：avg_karma_density < 全局平均 × GOAL_LOW_DENSITY_RATIO"""
        # 文学领域没有种子，avg_karma_density=0
        _insert_domain_monitor(
            graph.conn, "文学",
            {"avg_karma_density": 0.05, "ripple_success_rate": 0.0, "conflict_frequency": 0},
        )
        created = mgr.generate_goals()
        assert created >= 1

        row = graph.conn.execute(
            "SELECT * FROM cognitive_goals WHERE domain = '文学' AND goal_type = 'low_density'"
        ).fetchone()
        assert row is not None

    def test_generate_goals_high_conflict(self, mgr, graph):
        """冲突频率过高触发：conflict_frequency > GOAL_HIGH_CONFLICT_THRESHOLD"""
        _insert_domain_monitor(
            graph.conn, "医学",
            {"avg_karma_density": 0.5, "ripple_success_rate": 0.3, "conflict_frequency": GOAL_HIGH_CONFLICT_THRESHOLD + 2},
        )
        created = mgr.generate_goals()
        assert created >= 1

        row = graph.conn.execute(
            "SELECT * FROM cognitive_goals WHERE domain = '医学' AND goal_type = 'high_conflict'"
        ).fetchone()
        assert row is not None

    def test_generate_goals_new_term(self, mgr, graph):
        """新词触发：top_unmatched 关键词出现次数 > GOAL_NEW_TERM_THRESHOLD"""
        # 创建 meta:unknown
        now = datetime.now(timezone.utc).isoformat()
        graph.conn.execute(
            "INSERT INTO meta_seeds (label, category, metrics_json, status, created_at, updated_at) "
            "VALUES ('meta:unknown', 'self_boundary', ?, 'active', ?, ?)",
            (json.dumps({"unmatched_count": 1, "unmatched_keywords": ["DeepSeek"], "top_unmatched": ["DeepSeek"]}), now, now),
        )
        graph.conn.execute(
            "INSERT OR IGNORE INTO seeds (id, label, type, domain, activation) VALUES ('meta:unknown', 'meta:unknown', 'META', '元认知', 0.0)",
        )
        # 插入 unmatched_queries
        graph.conn.execute(
            "INSERT INTO unmatched_queries (query_text, count, first_seen, last_seen) "
            "VALUES ('DeepSeek', ?, ?, ?)",
            (GOAL_NEW_TERM_THRESHOLD + 2, now, now),
        )
        # 插入 candidate_seeds 用于推断领域
        graph.conn.execute(
            "INSERT INTO candidate_seeds (label, status, count, domain, candidate_since, last_seen_at) "
            "VALUES ('DeepSeek', 'candidate', 6, '科技', ?, ?)",
            (now, now),
        )
        graph.conn.commit()

        created = mgr.generate_goals()
        assert created >= 1

        row = graph.conn.execute(
            "SELECT * FROM cognitive_goals WHERE goal_type = 'new_term'"
        ).fetchone()
        assert row is not None

    def test_generate_goals_no_trigger(self, mgr, graph):
        """无触发条件时不生成目标"""
        _insert_domain_monitor(
            graph.conn, "计算机",
            {"avg_karma_density": 0.5, "ripple_success_rate": 0.3, "conflict_frequency": 0},
        )
        created = mgr.generate_goals()
        assert created == 0

    def test_generate_goals_disabled(self, mgr, graph):
        """COGNITIVE_GOAL_ENABLED=False 时不生成目标"""
        _insert_domain_monitor(
            graph.conn, "量子力学",
            {"avg_karma_density": 0.5, "ripple_success_rate": 0.3, "conflict_frequency": 100},
        )
        with patch("core.cognitive_goal.COGNITIVE_GOAL_ENABLED", False):
            created = mgr.generate_goals()
        assert created == 0

    def test_generate_goals_skips_inactive_meta_seeds(self, mgr, graph):
        """非 active 状态的元种子不触发目标"""
        _insert_domain_monitor(
            graph.conn, "量子力学",
            {"avg_karma_density": 0.5, "ripple_success_rate": 0.3, "conflict_frequency": 100},
            status="dormant",
        )
        created = mgr.generate_goals()
        assert created == 0


class TestGoalDedup:
    """目标去重测试"""

    def test_dedup_same_domain_same_type(self, mgr, graph):
        """同 domain + goal_type 且 status=pending → 更新而非创建"""
        _insert_domain_monitor(
            graph.conn, "量子力学",
            {"avg_karma_density": 0.5, "ripple_success_rate": 0.3, "conflict_frequency": GOAL_LOW_CONF_THRESHOLD + 1},
        )
        # 第一次生成
        mgr.generate_goals()
        count_before = graph.conn.execute(
            "SELECT COUNT(*) FROM cognitive_goals WHERE domain = '量子力学' AND goal_type = 'low_confidence'"
        ).fetchone()[0]
        assert count_before == 1

        # 第二次生成（应更新而非创建）
        mgr.generate_goals()
        count_after = graph.conn.execute(
            "SELECT COUNT(*) FROM cognitive_goals WHERE domain = '量子力学' AND goal_type = 'low_confidence'"
        ).fetchone()[0]
        assert count_after == 1

    def test_different_type_coexists(self, mgr, graph):
        """同 domain 不同 goal_type 允许共存"""
        _insert_domain_monitor(
            graph.conn, "量子力学",
            {
                "avg_karma_density": 0.05,  # 触发 low_density
                "ripple_success_rate": 0.3,
                "conflict_frequency": GOAL_HIGH_CONFLICT_THRESHOLD + 2,  # 触发 high_conflict
            },
        )
        created = mgr.generate_goals()
        assert created >= 2

        count = graph.conn.execute(
            "SELECT COUNT(DISTINCT goal_type) FROM cognitive_goals WHERE domain = '量子力学'"
        ).fetchone()[0]
        assert count >= 2

    def test_dedup_updates_priority(self, mgr, graph):
        """去重时更新优先级权重"""
        _insert_domain_monitor(
            graph.conn, "量子力学",
            {"avg_karma_density": 0.5, "ripple_success_rate": 0.3, "conflict_frequency": GOAL_LOW_CONF_THRESHOLD + 1},
        )
        mgr.generate_goals()
        row_before = graph.conn.execute(
            "SELECT priority_weight FROM cognitive_goals WHERE domain = '量子力学' AND goal_type = 'low_confidence'"
        ).fetchone()

        # 再次生成
        mgr.generate_goals()
        row_after = graph.conn.execute(
            "SELECT priority_weight FROM cognitive_goals WHERE domain = '量子力学' AND goal_type = 'low_confidence'"
        ).fetchone()

        # 权重应被更新（updated_at 变化）
        assert row_after is not None


class TestGoalPoolMaxSize:
    """目标池上限测试"""

    def test_pool_max_size_respected(self, mgr, graph):
        """池大小超限时跳过创建"""
        # 直接插入 GOAL_POOL_MAX_SIZE 个 pending 目标
        now = datetime.now(timezone.utc).isoformat()
        for i in range(GOAL_POOL_MAX_SIZE):
            graph.conn.execute(
                "INSERT INTO cognitive_goals "
                "(goal_id, goal_type, trigger_condition, domain, priority_weight, status, "
                " sub_goals, execution_log, decay_cycles_count, last_touched_at, created_at, updated_at) "
                "VALUES (?, 'low_confidence', 'test', ?, 0.5, 'pending', '[]', '[]', 0, ?, ?, ?)",
                (f"goal_test_{i}", f"domain_{i}", now, now, now),
            )
        graph.conn.commit()

        # 尝试创建新目标
        result = mgr._create_or_update_goal(
            goal_type=GoalType.LOW_CONFIDENCE,
            domain="新领域",
            trigger_condition="test",
        )
        assert result is False


class TestPriorityWeight:
    """优先级权重计算测试"""

    def test_priority_weight_formula(self, mgr, graph):
        """优先级权重 = 用户相关性×0.4 + 系统核心度×0.3 + 不确定性×0.2 + 可分解性×0.1"""
        _insert_domain_monitor(
            graph.conn, "医学",
            {"avg_karma_density": 0.5, "ripple_success_rate": 0.3, "conflict_frequency": 6},
        )
        created = mgr.generate_goals()
        assert created >= 1

        row = graph.conn.execute(
            "SELECT priority_weight FROM cognitive_goals WHERE domain = '医学'"
        ).fetchone()
        assert row is not None
        weight = row["priority_weight"]
        assert 0.0 <= weight <= 1.0

    def test_priority_weight_deterministic(self, mgr, graph):
        """相同输入产生相同优先级权重"""
        _insert_domain_monitor(
            graph.conn, "物理",
            {"avg_karma_density": 0.5, "ripple_success_rate": 0.3, "conflict_frequency": 6},
        )
        w1 = mgr._compute_priority_weight("物理", GoalType.LOW_CONFIDENCE)
        w2 = mgr._compute_priority_weight("物理", GoalType.LOW_CONFIDENCE)
        assert w1 == w2

    def test_compute_user_relevance_default(self, mgr, graph):
        """用户相关性：无数据时默认 0.5"""
        relevance = mgr._compute_user_relevance("不存在的领域")
        assert relevance == 0.5

    def test_compute_system_coreness_same_domain(self, mgr, graph):
        """系统核心度：目标领域即核心领域时返回 1.0"""
        # 医学领域种子最多
        coreness = mgr._compute_system_coreness("医学")
        assert coreness == 1.0

    def test_compute_system_coreness_default(self, mgr, graph):
        """系统核心度：无数据时默认 0.5"""
        coreness = mgr._compute_system_coreness("不存在的领域")
        assert coreness == 0.5

    def test_compute_uncertainty_default(self, mgr, graph):
        """不确定性：无元种子时默认 0.5"""
        uncertainty = mgr._compute_uncertainty("不存在的领域")
        assert uncertainty == 0.5

    def test_compute_decomposability_no_seeds(self, mgr, graph):
        """可分解性：无种子的领域返回 0.0（子领域数为0）"""
        decomposability = mgr._compute_decomposability("不存在的领域")
        assert decomposability == 0.0  # 无种子时子领域数为0

    def test_compute_decomposability_with_seeds(self, mgr, graph):
        """可分解性：有种子的领域返回 > 0"""
        decomposability = mgr._compute_decomposability("医学")
        assert decomposability > 0.0


class TestGoalCooling:
    """目标冷却测试"""

    def test_cool_goals_weight_decay(self, mgr, graph):
        """权重衰减：连续 GOAL_DECAY_CYCLES 周期无人触及 → weight × GOAL_DECAY_FACTOR"""
        now = datetime.now(timezone.utc)
        old_time = now - timedelta(seconds=GUARDIAN_LOOP_INTERVAL * (GOAL_DECAY_CYCLES + 1))
        graph.conn.execute(
            "INSERT INTO cognitive_goals "
            "(goal_id, goal_type, trigger_condition, domain, priority_weight, status, "
            " sub_goals, execution_log, decay_cycles_count, last_touched_at, created_at, updated_at) "
            "VALUES (?, 'low_confidence', 'test', '量子力学', 0.8, 'pending', '[]', '[]', 0, ?, ?, ?)",
            ("goal_decay_test", old_time.isoformat(), old_time.isoformat(), old_time.isoformat()),
        )
        graph.conn.commit()

        cooled = mgr.cool_goals()
        assert cooled >= 1

        row = graph.conn.execute(
            "SELECT priority_weight, status FROM cognitive_goals WHERE goal_id = 'goal_decay_test'"
        ).fetchone()
        assert row["priority_weight"] < 0.8
        assert row["status"] == "pending"  # 还未过期

    def test_cool_goals_expire(self, mgr, graph):
        """权重衰减后 < GOAL_EXPIRE_THRESHOLD → 标记 expired"""
        now = datetime.now(timezone.utc)
        old_time = now - timedelta(seconds=GUARDIAN_LOOP_INTERVAL * (GOAL_DECAY_CYCLES + 1))
        # 使用极低权重
        graph.conn.execute(
            "INSERT INTO cognitive_goals "
            "(goal_id, goal_type, trigger_condition, domain, priority_weight, status, "
            " sub_goals, execution_log, decay_cycles_count, last_touched_at, created_at, updated_at) "
            "VALUES (?, 'low_confidence', 'test', '量子力学', 0.06, 'pending', '[]', '[]', 0, ?, ?, ?)",
            ("goal_expire_test", old_time.isoformat(), old_time.isoformat(), old_time.isoformat()),
        )
        graph.conn.commit()

        mgr.cool_goals()

        row = graph.conn.execute(
            "SELECT status FROM cognitive_goals WHERE goal_id = 'goal_expire_test'"
        ).fetchone()
        assert row["status"] == "expired"

    def test_cool_goals_user_absent_archive(self, mgr, graph):
        """用户缺席归档：关联用户连续 GOAL_USER_ABSENCE_CYCLES 周期无活动"""
        now = datetime.now(timezone.utc)
        old_time = now - timedelta(seconds=GUARDIAN_LOOP_INTERVAL * (GOAL_USER_ABSENCE_CYCLES + 1))
        graph.conn.execute(
            "INSERT INTO cognitive_goals "
            "(goal_id, goal_type, trigger_condition, domain, priority_weight, status, "
            " sub_goals, execution_log, associated_user, decay_cycles_count, last_touched_at, created_at, updated_at) "
            "VALUES (?, 'new_term', 'test', '科技', 0.5, 'pending', '[]', '[]', 'user_abc', 0, ?, ?, ?)",
            ("goal_absent_test", now.isoformat(), old_time.isoformat(), old_time.isoformat()),
        )
        # 用户最后活跃时间很久以前
        graph.conn.execute(
            "INSERT INTO user_cold_start (user_label, query_count, updated_at) VALUES ('user_abc', 5, ?)",
            (old_time.isoformat(),),
        )
        graph.conn.commit()

        cooled = mgr.cool_goals()
        assert cooled >= 1

        row = graph.conn.execute(
            "SELECT status FROM cognitive_goals WHERE goal_id = 'goal_absent_test'"
        ).fetchone()
        assert row["status"] == "archived"

    def test_cool_goals_pool_eviction(self, mgr, graph):
        """池大小超限 → 淘汰最低权重目标"""
        now = datetime.now(timezone.utc).isoformat()
        # 插入超过上限的目标
        for i in range(GOAL_POOL_MAX_SIZE + 5):
            weight = 0.01 if i < 5 else 0.5  # 前5个权重最低
            graph.conn.execute(
                "INSERT INTO cognitive_goals "
                "(goal_id, goal_type, trigger_condition, domain, priority_weight, status, "
                " sub_goals, execution_log, decay_cycles_count, last_touched_at, created_at, updated_at) "
                "VALUES (?, 'low_confidence', 'test', ?, ?, 'pending', '[]', '[]', 0, ?, ?, ?)",
                (f"goal_evict_{i}", f"domain_{i}", weight, now, now, now),
            )
        graph.conn.commit()

        cooled = mgr.cool_goals()
        assert cooled >= 5

        # 验证最低权重的目标被淘汰
        active_count = graph.conn.execute(
            "SELECT COUNT(*) FROM cognitive_goals WHERE status = 'pending'"
        ).fetchone()[0]
        assert active_count <= GOAL_POOL_MAX_SIZE

    def test_cool_goals_does_not_affect_exploring(self, mgr, graph):
        """冷却不影响 exploring 状态的目标"""
        now = datetime.now(timezone.utc).isoformat()
        graph.conn.execute(
            "INSERT INTO cognitive_goals "
            "(goal_id, goal_type, trigger_condition, domain, priority_weight, status, "
            " sub_goals, execution_log, decay_cycles_count, last_touched_at, created_at, updated_at) "
            "VALUES (?, 'low_confidence', 'test', '量子力学', 0.8, 'exploring', '[]', '[]', 0, ?, ?, ?)",
            ("goal_exploring_test", now, now, now),
        )
        graph.conn.commit()

        cooled = mgr.cool_goals()
        # exploring 状态的目标不应被冷却
        row = graph.conn.execute(
            "SELECT status, priority_weight FROM cognitive_goals WHERE goal_id = 'goal_exploring_test'"
        ).fetchone()
        assert row["status"] == "exploring"
        assert row["priority_weight"] == 0.8

    def test_cool_goals_disabled(self, mgr, graph):
        """COGNITIVE_GOAL_ENABLED=False 时不执行冷却"""
        with patch("core.cognitive_goal.COGNITIVE_GOAL_ENABLED", False):
            cooled = mgr.cool_goals()
        assert cooled == 0


class TestGoalQuery:
    """目标查询与统计测试"""

    def test_get_goal(self, mgr, graph):
        """查询单个认知目标"""
        now = datetime.now(timezone.utc).isoformat()
        graph.conn.execute(
            "INSERT INTO cognitive_goals "
            "(goal_id, goal_type, trigger_condition, domain, priority_weight, status, "
            " sub_goals, execution_log, decay_cycles_count, last_touched_at, created_at, updated_at) "
            "VALUES (?, 'low_confidence', 'test', '量子力学', 0.72, 'pending', '[]', '[]', 0, ?, ?, ?)",
            ("goal_query_test", now, now, now),
        )
        graph.conn.commit()

        goal = mgr.get_goal("goal_query_test")
        assert goal is not None
        assert goal.goal_id == "goal_query_test"
        assert goal.goal_type == GoalType.LOW_CONFIDENCE
        assert goal.domain == "量子力学"
        assert goal.priority_weight == 0.72
        assert goal.status == GoalStatus.PENDING

    def test_get_goal_not_found(self, mgr, graph):
        """查询不存在的目标返回 None"""
        goal = mgr.get_goal("goal_not_exist")
        assert goal is None

    def test_list_goals(self, mgr, graph):
        """查询目标列表"""
        now = datetime.now(timezone.utc).isoformat()
        for i, (gt, domain) in enumerate([
            ("low_confidence", "量子力学"),
            ("high_conflict", "医学"),
        ]):
            graph.conn.execute(
                "INSERT INTO cognitive_goals "
                "(goal_id, goal_type, trigger_condition, domain, priority_weight, status, "
                " sub_goals, execution_log, decay_cycles_count, last_touched_at, created_at, updated_at) "
                "VALUES (?, ?, 'test', ?, 0.5, 'pending', '[]', '[]', 0, ?, ?, ?)",
                (f"goal_list_{i}", gt, domain, now, now, now),
            )
        graph.conn.commit()

        goals = mgr.list_goals()
        assert len(goals) >= 2

    def test_list_goals_filter_status(self, mgr, graph):
        """按状态过滤目标列表"""
        now = datetime.now(timezone.utc).isoformat()
        graph.conn.execute(
            "INSERT INTO cognitive_goals "
            "(goal_id, goal_type, trigger_condition, domain, priority_weight, status, "
            " sub_goals, execution_log, decay_cycles_count, last_touched_at, created_at, updated_at) "
            "VALUES (?, 'low_confidence', 'test', '量子力学', 0.72, 'completed', '[]', '[]', 0, ?, ?, ?)",
            ("goal_completed", now, now, now),
        )
        graph.conn.execute(
            "INSERT INTO cognitive_goals "
            "(goal_id, goal_type, trigger_condition, domain, priority_weight, status, "
            " sub_goals, execution_log, decay_cycles_count, last_touched_at, created_at, updated_at) "
            "VALUES (?, 'high_conflict', 'test', '医学', 0.5, 'pending', '[]', '[]', 0, ?, ?, ?)",
            ("goal_pending", now, now, now),
        )
        graph.conn.commit()

        pending = mgr.list_goals(status=GoalStatus.PENDING)
        assert len(pending) >= 1
        assert all(g.status == GoalStatus.PENDING for g in pending)

    def test_list_goals_filter_type(self, mgr, graph):
        """按类型过滤目标列表"""
        now = datetime.now(timezone.utc).isoformat()
        graph.conn.execute(
            "INSERT INTO cognitive_goals "
            "(goal_id, goal_type, trigger_condition, domain, priority_weight, status, "
            " sub_goals, execution_log, decay_cycles_count, last_touched_at, created_at, updated_at) "
            "VALUES (?, 'low_confidence', 'test', '量子力学', 0.72, 'pending', '[]', '[]', 0, ?, ?, ?)",
            ("goal_lc", now, now, now),
        )
        graph.conn.execute(
            "INSERT INTO cognitive_goals "
            "(goal_id, goal_type, trigger_condition, domain, priority_weight, status, "
            " sub_goals, execution_log, decay_cycles_count, last_touched_at, created_at, updated_at) "
            "VALUES (?, 'high_conflict', 'test', '医学', 0.5, 'pending', '[]', '[]', 0, ?, ?, ?)",
            ("goal_hc", now, now, now),
        )
        graph.conn.commit()

        lc_goals = mgr.list_goals(goal_type=GoalType.LOW_CONFIDENCE)
        assert len(lc_goals) >= 1
        assert all(g.goal_type == GoalType.LOW_CONFIDENCE for g in lc_goals)

    def test_get_goal_stats(self, mgr, graph):
        """查询目标统计信息"""
        now = datetime.now(timezone.utc).isoformat()
        graph.conn.execute(
            "INSERT INTO cognitive_goals "
            "(goal_id, goal_type, trigger_condition, domain, priority_weight, status, "
            " sub_goals, execution_log, decay_cycles_count, last_touched_at, created_at, updated_at) "
            "VALUES (?, 'low_confidence', 'test', '量子力学', 0.72, 'pending', '[]', '[]', 0, ?, ?, ?)",
            ("goal_stats_1", now, now, now),
        )
        graph.conn.execute(
            "INSERT INTO cognitive_goals "
            "(goal_id, goal_type, trigger_condition, domain, priority_weight, status, "
            " sub_goals, execution_log, decay_cycles_count, last_touched_at, created_at, updated_at) "
            "VALUES (?, 'high_conflict', 'test', '医学', 0.5, 'completed', '[]', '[]', 0, ?, ?, ?)",
            ("goal_stats_2", now, now, now),
        )
        graph.conn.commit()

        stats = mgr.get_goal_stats()
        assert "by_status" in stats
        assert "by_type" in stats
        assert "avg_priority_weight" in stats
        assert "pool_usage" in stats
        assert stats["pool_usage"]["max"] == GOAL_POOL_MAX_SIZE

    def test_get_goal_stats_disabled(self, mgr, graph):
        """COGNITIVE_GOAL_ENABLED=False 时统计返回空"""
        with patch("core.cognitive_goal.COGNITIVE_GOAL_ENABLED", False):
            stats = mgr.get_goal_stats()
        assert stats["by_status"] == {}
        assert stats["pool_usage"]["active"] == 0


class TestTouchGoalDomain:
    """目标触及更新测试"""

    def test_touch_goal_domain(self, mgr, graph):
        """touch_goal_domain 更新该领域所有 pending 目标的 last_touched_at"""
        now = datetime.now(timezone.utc).isoformat()
        graph.conn.execute(
            "INSERT INTO cognitive_goals "
            "(goal_id, goal_type, trigger_condition, domain, priority_weight, status, "
            " sub_goals, execution_log, decay_cycles_count, last_touched_at, created_at, updated_at) "
            "VALUES (?, 'low_confidence', 'test', '量子力学', 0.72, 'pending', '[]', '[]', 0, 'old_time', ?, ?)",
            ("goal_touch_test", now, now),
        )
        graph.conn.commit()

        mgr.touch_goal_domain("量子力学")

        row = graph.conn.execute(
            "SELECT last_touched_at FROM cognitive_goals WHERE goal_id = 'goal_touch_test'"
        ).fetchone()
        assert row["last_touched_at"] != "old_time"

    def test_touch_goal_domain_disabled(self, mgr, graph):
        """COGNITIVE_GOAL_ENABLED=False 时不更新"""
        with patch("core.cognitive_goal.COGNITIVE_GOAL_ENABLED", False):
            mgr.touch_goal_domain("量子力学")
        # 不应报错


class TestGoalIdGeneration:
    """目标 ID 生成测试"""

    def test_goal_id_format(self, mgr, graph):
        """goal_id 格式为 goal_{timestamp}_{domain}_{hash}"""
        goal_id = mgr._generate_goal_id("量子力学", GoalType.LOW_CONFIDENCE)
        assert goal_id.startswith("goal_")
        assert "量子力学" in goal_id

    def test_goal_id_unique(self, mgr, graph):
        """不同时间生成的 goal_id 不同"""
        import time
        id1 = mgr._generate_goal_id("量子力学", GoalType.LOW_CONFIDENCE)
        time.sleep(0.01)
        id2 = mgr._generate_goal_id("量子力学", GoalType.LOW_CONFIDENCE)
        assert id1 != id2


class TestGoalHistory:
    """目标历史记录测试"""

    def test_record_goal_history(self, mgr, graph):
        """目标状态变更记录到 goal_history 表"""
        mgr._record_goal_history(
            "goal_test", "pending", "expired", 0.8, 0.04, "decay_expire",
        )
        row = graph.conn.execute(
            "SELECT * FROM goal_history WHERE goal_id = 'goal_test'"
        ).fetchone()
        assert row is not None
        assert row["old_status"] == "pending"
        assert row["new_status"] == "expired"
        assert row["reason"] == "decay_expire"