"""
Phase 4 MetaSeedManager 单元测试

覆盖：
- 元种子生成（领域监控、关系质量、系统级、自边界、未知领域）
- 元种子查询和列表
- 指标更新（update_metrics、increment_metric）
- 元业力边创建（check_and_create_meta_karma）
- 休眠/退役判定（check_dormant_status）
- _ensure_meta_prefix 辅助方法
- META_SEED_ENABLED=False 时的行为
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

from consciousness_sea.metacognition.meta_seed import (
    MetaSeedManager,
    MetaSeedData,
    MetaSeedCategory,
    MetaSeedStatus,
    DOMAIN_MONITOR_DEFAULT_METRICS,
    RELATION_QUALITY_DEFAULT_METRICS,
    SYSTEM_MONITOR_DEFAULT_METRICS,
    SELF_BOUNDARY_DEFAULT_METRICS,
    PERFORMANCE_MONITOR_DEFAULT_METRICS,
    SYSTEM_META_SEEDS,
)
from consciousness_sea.domain.graph_db import GraphDB
from consciousness_sea.infrastructure.config import (
    META_SEED_ENABLED,
    META_KARMA_DELTA_THRESHOLD,
    META_SEED_DORMANT_CYCLES,
    META_EXPLORE_LOW_CONF_THRESHOLD,
    CONFIDENCE_LOW,
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


class TestMetaSeedGeneration:
    """元种子生成测试"""

    def test_generate_domain_monitors(self, mgr, graph):
        """领域监控元种子自动生成：从 seeds 表不同 domain 值生成"""
        created = mgr.generate_domain_monitors()
        # 领域: 医学, 营养, 物理, 计算机 → 4 个
        assert created == 4

        # 验证 seeds 表和 meta_seeds 表均有记录
        row = graph.conn.execute(
            "SELECT * FROM seeds WHERE label = 'meta:医学'"
        ).fetchone()
        assert row is not None
        assert row["type"] == "META"
        assert row["domain"] == "元认知"
        assert row["activation"] == 0.0

        row = graph.conn.execute(
            "SELECT * FROM meta_seeds WHERE label = 'meta:医学'"
        ).fetchone()
        assert row is not None
        assert row["category"] == "domain_monitor"
        metrics = json.loads(row["metrics_json"])
        assert metrics == DOMAIN_MONITOR_DEFAULT_METRICS

    def test_generate_domain_monitors_excludes_meta_domain(self, mgr, graph):
        """元种子不监控元种子：domain='元认知' 被排除"""
        # 先创建一个元认知领域的种子
        graph.conn.execute(
            "INSERT INTO seeds (id, label, type, domain) VALUES ('元认知种子', '元认知种子', 'CONCEPT', '元认知')"
        )
        graph.conn.commit()

        created = mgr.generate_domain_monitors()
        # 不应创建 meta:元认知
        row = graph.conn.execute(
            "SELECT * FROM meta_seeds WHERE label = 'meta:元认知'"
        ).fetchone()
        assert row is None

    def test_generate_domain_monitors_dedup(self, mgr, graph):
        """元种子去重：已存在时跳过创建"""
        mgr.generate_domain_monitors()
        # 第二次调用应返回 0（全部已存在）
        created = mgr.generate_domain_monitors()
        assert created == 0

    def test_generate_relation_monitors(self, mgr, graph):
        """关系质量元种子自动生成：从 karma_edges 表不同 relation 值生成"""
        created = mgr.generate_relation_monitors()
        # 关系: COOCCURS_WITH, RELATED, IS_A → 3 个
        assert created == 3

        row = graph.conn.execute(
            "SELECT * FROM meta_seeds WHERE label = 'meta:IS_A'"
        ).fetchone()
        assert row is not None
        assert row["category"] == "relation_quality"
        metrics = json.loads(row["metrics_json"])
        assert metrics == RELATION_QUALITY_DEFAULT_METRICS

    def test_generate_relation_monitors_dedup(self, mgr, graph):
        """关系质量元种子去重"""
        mgr.generate_relation_monitors()
        created = mgr.generate_relation_monitors()
        assert created == 0

    def test_generate_system_monitors(self, mgr, graph):
        """系统级元种子固定生成：5 个"""
        created = mgr.generate_system_monitors()
        assert created == 5

        for label, desc in SYSTEM_META_SEEDS:
            row = graph.conn.execute(
                "SELECT * FROM meta_seeds WHERE label = ?", (label,)
            ).fetchone()
            assert row is not None
            assert row["category"] == "system_monitor"
            metrics = json.loads(row["metrics_json"])
            assert metrics == SYSTEM_MONITOR_DEFAULT_METRICS

    def test_generate_system_monitors_dedup(self, mgr, graph):
        """系统级元种子去重"""
        mgr.generate_system_monitors()
        created = mgr.generate_system_monitors()
        assert created == 0

    def test_update_self_boundary(self, mgr, graph):
        """自边界元种子更新：从 candidate_seeds 表提取未匹配关键词"""
        # 先确保 meta:unknown 存在
        mgr._create_meta_seed_record(
            "meta:unknown", MetaSeedCategory.SELF_BOUNDARY,
            dict(SELF_BOUNDARY_DEFAULT_METRICS),
        )

        # 插入候选种子
        now = "2025-01-01T00:00:00+00:00"
        graph.conn.execute(
            "INSERT INTO candidate_seeds (label, status, count, candidate_since, last_seen_at) "
            "VALUES (?, 'candidate', ?, ?, ?)",
            ("DeepSeek", 5, now, now),
        )
        graph.conn.execute(
            "INSERT INTO candidate_seeds (label, status, count, candidate_since, last_seen_at) "
            "VALUES (?, 'candidate', ?, ?, ?)",
            ("Transformer", 3, now, now),
        )
        graph.conn.execute(
            "INSERT INTO candidate_seeds (label, status, count, candidate_since, last_seen_at) "
            "VALUES (?, 'promoted', ?, ?, ?)",
            ("已升级", 10, now, now),
        )
        graph.conn.commit()

        updated = mgr.update_self_boundary()
        assert updated == 1

        ms = mgr.get_meta_seed("meta:unknown")
        assert ms is not None
        assert ms.metrics["unmatched_count"] == 2
        assert "DeepSeek" in ms.metrics["unmatched_keywords"]
        assert "Transformer" in ms.metrics["unmatched_keywords"]
        # promoted 状态的不应出现
        assert "已升级" not in ms.metrics["unmatched_keywords"]
        # top_unmatched 按 count 降序
        assert ms.metrics["top_unmatched"][0] == "DeepSeek"

    def test_detect_unknown_domains(self, mgr, graph):
        """未知领域探测元种子：低置信度频率超过阈值时创建"""
        # 插入 param_stats 数据：物理领域低置信度高
        now = "2025-01-01T00:00:00+00:00"
        for i in range(40):
            confidence = 0.1 if i < 35 else 0.8  # 35/40 = 0.875 > 0.3
            graph.conn.execute(
                "INSERT INTO param_stats "
                "(query_text, decay_factor, domain_threshold, confidence_high, "
                "ripple_depth, activated_count, selected_domains, confidence, karma_direction, created_at) "
                "VALUES (?, 1.0, 0.3, 0.7, 2, 5, ?, ?, 0, ?)",
                (f"query_{i}", json.dumps(["物理"]), confidence, now),
            )
        graph.conn.commit()

        created = mgr.detect_unknown_domains()
        assert created >= 1

        ms = mgr.get_meta_seed("meta:explore_物理")
        assert ms is not None
        assert ms.category == MetaSeedCategory.PERFORMANCE_MONITOR
        assert ms.metrics["low_confidence_rate"] > META_EXPLORE_LOW_CONF_THRESHOLD

    def test_detect_unknown_domains_dedup(self, mgr, graph):
        """未知领域探测元种子去重：已存在时仅更新指标"""
        # 先创建探测元种子
        mgr._create_meta_seed_record(
            "meta:explore_物理", MetaSeedCategory.PERFORMANCE_MONITOR,
            {"low_confidence_rate": 0.35, "query_count": 100},
            source_domain="物理",
        )

        # 插入 param_stats 数据
        now = "2025-01-01T00:00:00+00:00"
        for i in range(40):
            confidence = 0.1 if i < 35 else 0.8
            graph.conn.execute(
                "INSERT INTO param_stats "
                "(query_text, decay_factor, domain_threshold, confidence_high, "
                "ripple_depth, activated_count, selected_domains, confidence, karma_direction, created_at) "
                "VALUES (?, 1.0, 0.3, 0.7, 2, 5, ?, ?, 0, ?)",
                (f"query_{i}", json.dumps(["物理"]), confidence, now),
            )
        graph.conn.commit()

        created = mgr.detect_unknown_domains()
        assert created >= 1

        # 验证只有一条 meta:explore_物理
        count = graph.conn.execute(
            "SELECT COUNT(*) FROM meta_seeds WHERE label = 'meta:explore_物理'"
        ).fetchone()[0]
        assert count == 1


class TestMetaSeedQuery:
    """元种子查询测试"""

    def test_get_meta_seed(self, mgr, graph):
        """查询单个元种子"""
        mgr.generate_domain_monitors()
        ms = mgr.get_meta_seed("meta:医学")
        assert ms is not None
        assert ms.label == "meta:医学"
        assert ms.category == MetaSeedCategory.DOMAIN_MONITOR
        assert ms.status == MetaSeedStatus.ACTIVE

    def test_get_meta_seed_not_found(self, mgr, graph):
        """查询不存在的元种子返回 None"""
        ms = mgr.get_meta_seed("meta:不存在")
        assert ms is None

    def test_list_meta_seeds(self, mgr, graph):
        """查询元种子列表"""
        mgr.generate_domain_monitors()
        mgr.generate_system_monitors()
        seeds = mgr.list_meta_seeds()
        assert len(seeds) >= 9  # 4 domain + 5 system

    def test_list_meta_seeds_filter_category(self, mgr, graph):
        """按类别过滤元种子列表"""
        mgr.generate_domain_monitors()
        mgr.generate_system_monitors()
        seeds = mgr.list_meta_seeds(category=MetaSeedCategory.DOMAIN_MONITOR)
        assert len(seeds) == 4
        assert all(s.category == MetaSeedCategory.DOMAIN_MONITOR for s in seeds)

    def test_list_meta_seeds_filter_status(self, mgr, graph):
        """按状态过滤元种子列表"""
        mgr.generate_system_monitors()
        seeds = mgr.list_meta_seeds(status=MetaSeedStatus.ACTIVE)
        assert len(seeds) == 5
        assert all(s.status == MetaSeedStatus.ACTIVE for s in seeds)

    def test_get_meta_seed_json_parse_error(self, mgr, graph):
        """metrics JSON 格式异常时重置为空对象"""
        # 直接插入非法 JSON
        now = "2025-01-01T00:00:00+00:00"
        graph.conn.execute(
            "INSERT INTO meta_seeds (label, category, metrics_json, status, created_at, updated_at) "
            "VALUES (?, 'domain_monitor', 'invalid_json', 'active', ?, ?)",
            ("meta:测试", now, now),
        )
        graph.conn.commit()

        ms = mgr.get_meta_seed("meta:测试")
        assert ms is not None
        assert ms.metrics == {}


class TestMetaSeedMetrics:
    """元种子指标更新测试"""

    def test_update_metrics(self, mgr, graph):
        """原子更新指标"""
        mgr._create_meta_seed_record(
            "meta:医学", MetaSeedCategory.DOMAIN_MONITOR,
            dict(DOMAIN_MONITOR_DEFAULT_METRICS),
        )

        new_metrics = {"avg_karma_density": 0.45, "ripple_success_rate": 0.78, "conflict_frequency": 3}
        result = mgr.update_metrics("meta:医学", new_metrics)
        assert result is True

        ms = mgr.get_meta_seed("meta:医学")
        assert ms is not None
        assert ms.metrics["avg_karma_density"] == 0.45
        assert ms.metrics["conflict_frequency"] == 3

    def test_update_metrics_not_found(self, mgr, graph):
        """更新不存在的元种子返回 False"""
        result = mgr.update_metrics("meta:不存在", {"value": 1})
        assert result is False

    def test_increment_metric(self, mgr, graph):
        """递增元种子的某个指标"""
        mgr._create_meta_seed_record(
            "meta:医学", MetaSeedCategory.DOMAIN_MONITOR,
            dict(DOMAIN_MONITOR_DEFAULT_METRICS),
        )

        result = mgr.increment_metric("meta:医学", "conflict_frequency", delta=1)
        assert result is True

        ms = mgr.get_meta_seed("meta:医学")
        assert ms is not None
        assert ms.metrics["conflict_frequency"] == 1

        # 再次递增
        mgr.increment_metric("meta:医学", "conflict_frequency", delta=2)
        ms = mgr.get_meta_seed("meta:医学")
        assert ms.metrics["conflict_frequency"] == 3

    def test_increment_metric_not_found(self, mgr, graph):
        """递增不存在的元种子返回 False"""
        result = mgr.increment_metric("meta:不存在", "value", delta=1)
        assert result is False


class TestMetaSeedDormant:
    """元种子休眠/退役判定测试

    _unchanged_cycles 和 _previous_metrics 已持久化到数据库，
    因此多次调用 check_dormant_status() 可以正确累积无变化周期数。
    使用空 metrics {} 创建元种子，确保初始时 metrics_changed=False。
    """

    def test_dormant_transition(self, mgr, graph):
        """连续 DORMANT_CYCLES 个周期无变化 → dormant

        使用空 metrics {} 确保 metrics_changed=False
        """
        mgr._create_meta_seed_record(
            "meta:医学", MetaSeedCategory.DOMAIN_MONITOR,
            {},  # 空 metrics，确保与 _previous_metrics={} 相等
        )

        # 模拟连续无变化周期
        for _ in range(META_SEED_DORMANT_CYCLES):
            mgr.check_dormant_status()

        ms = mgr.get_meta_seed("meta:医学")
        assert ms is not None
        assert ms.status == MetaSeedStatus.DORMANT

    def test_retired_transition(self, mgr, graph):
        """dormant 超过 3 × DORMANT_CYCLES → retired"""
        mgr._create_meta_seed_record(
            "meta:医学", MetaSeedCategory.DOMAIN_MONITOR,
            {},  # 空 metrics
        )

        # 先进入 dormant
        for _ in range(META_SEED_DORMANT_CYCLES):
            mgr.check_dormant_status()

        ms = mgr.get_meta_seed("meta:医学")
        assert ms.status == MetaSeedStatus.DORMANT

        # 继续无变化直到 retired
        for _ in range(META_SEED_DORMANT_CYCLES * 3):
            mgr.check_dormant_status()

        ms = mgr.get_meta_seed("meta:医学")
        assert ms is not None
        assert ms.status == MetaSeedStatus.RETIRED

    def test_dormant_to_active_on_change(self, mgr, graph):
        """dormant 状态指标变化 → 恢复 active"""
        mgr._create_meta_seed_record(
            "meta:医学", MetaSeedCategory.DOMAIN_MONITOR,
            {},  # 空 metrics
        )

        # 先进入 dormant
        for _ in range(META_SEED_DORMANT_CYCLES):
            mgr.check_dormant_status()

        ms = mgr.get_meta_seed("meta:医学")
        assert ms.status == MetaSeedStatus.DORMANT

        # 更新指标使指标发生变化（从 {} 变为有值）
        mgr.update_metrics("meta:医学", {"avg_karma_density": 0.5, "ripple_success_rate": 0.0, "conflict_frequency": 0})

        # 再次检查
        mgr.check_dormant_status()

        ms = mgr.get_meta_seed("meta:医学")
        assert ms.status == MetaSeedStatus.ACTIVE

    def test_retired_not_changed(self, mgr, graph):
        """退役状态不再变更"""
        mgr._create_meta_seed_record(
            "meta:医学", MetaSeedCategory.DOMAIN_MONITOR,
            {},  # 空 metrics
        )

        # 进入 retired
        total_cycles = META_SEED_DORMANT_CYCLES * 4
        for _ in range(total_cycles):
            mgr.check_dormant_status()

        ms = mgr.get_meta_seed("meta:医学")
        assert ms.status == MetaSeedStatus.RETIRED

        # 更新指标后检查，retired 不应恢复
        mgr.update_metrics("meta:医学", {"avg_karma_density": 0.9, "ripple_success_rate": 0.0, "conflict_frequency": 0})
        mgr.check_dormant_status()

        ms = mgr.get_meta_seed("meta:医学")
        assert ms.status == MetaSeedStatus.RETIRED


class TestMetaKarmaEdge:
    """元业力边创建测试

    _previous_metrics 已持久化到数据库。第一次调用 check_and_create_meta_karma()
    时 _previous_metrics 为空，保存当前指标作为基线并跳过。
    第二次调用时从数据库读取基线，可以比较变化并创建元业力边。
    因此测试需要调用两次：第一次设置基线，第二次检测变化。
    """

    def test_check_and_create_meta_karma(self, mgr, graph):
        """指标变化量 >= DELTA_THRESHOLD 时触发元业力边创建"""
        # 创建两个同类别元种子
        mgr._create_meta_seed_record(
            "meta:医学", MetaSeedCategory.DOMAIN_MONITOR,
            {"avg_karma_density": 0.0, "ripple_success_rate": 0.0, "conflict_frequency": 0},
        )
        mgr._create_meta_seed_record(
            "meta:物理", MetaSeedCategory.DOMAIN_MONITOR,
            {"avg_karma_density": 0.0, "ripple_success_rate": 0.0, "conflict_frequency": 0},
        )

        # 第一次调用：设置基线（_previous_metrics 为空，保存当前指标作为基线）
        mgr.check_and_create_meta_karma()

        # 更新指标使 conflict_frequency 变化量超过阈值
        mgr.update_metrics("meta:医学", {
            "avg_karma_density": 0.0, "ripple_success_rate": 0.0,
            "conflict_frequency": META_KARMA_DELTA_THRESHOLD + 1,
        })

        # 第二次调用：检测变化并创建元业力边
        edges_created = mgr.check_and_create_meta_karma()
        assert edges_created >= 1

        # 验证元业力边已创建
        row = graph.conn.execute(
            "SELECT * FROM karma_edges WHERE source = 'meta:医学' AND source_tag = 'meta_karma'"
        ).fetchone()
        assert row is not None
        assert row["relation"] == "META_CORRELATED"

    def test_meta_karma_direction_positive(self, mgr, graph):
        """正向变化（指标恶化）增强元业力边"""
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

        # conflict_frequency 从 0 增加到 5（正向变化/恶化）
        mgr.update_metrics("meta:医学", {
            "avg_karma_density": 0.0, "ripple_success_rate": 0.0, "conflict_frequency": 5,
        })

        # 第二次调用：检测变化
        mgr.check_and_create_meta_karma()

        row = graph.conn.execute(
            "SELECT weight FROM karma_edges WHERE source = 'meta:医学' AND source_tag = 'meta_karma'"
        ).fetchone()
        assert row is not None
        # 正向变化权重应大于 0
        assert row["weight"] > 0.0

    def test_meta_karma_direction_negative(self, mgr, graph):
        """负向变化（指标改善）减弱元业力边"""
        mgr._create_meta_seed_record(
            "meta:医学", MetaSeedCategory.DOMAIN_MONITOR,
            {"avg_karma_density": 0.0, "ripple_success_rate": 0.0, "conflict_frequency": 10},
        )
        mgr._create_meta_seed_record(
            "meta:物理", MetaSeedCategory.DOMAIN_MONITOR,
            {"avg_karma_density": 0.0, "ripple_success_rate": 0.0, "conflict_frequency": 0},
        )

        # 第一次调用：设置基线
        mgr.check_and_create_meta_karma()

        # conflict_frequency 从 10 减少到 3（负向变化/改善）
        mgr.update_metrics("meta:医学", {
            "avg_karma_density": 0.0, "ripple_success_rate": 0.0, "conflict_frequency": 3,
        })

        # 第二次调用：检测变化
        mgr.check_and_create_meta_karma()

        row = graph.conn.execute(
            "SELECT weight FROM karma_edges WHERE source = 'meta:医学' AND source_tag = 'meta_karma'"
        ).fetchone()
        assert row is not None

    def test_meta_karma_below_threshold(self, mgr, graph):
        """指标变化量 < DELTA_THRESHOLD 时不触发元业力边"""
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

    def test_meta_karma_isolation(self, mgr, graph):
        """元业力边与普通业力边的隔离：source_tag 区分"""
        # 创建普通业力边
        graph.conn.execute(
            "INSERT INTO karma_edges (source, target, relation, weight, source_tag) "
            "VALUES ('感冒', '发热', 'RELATED', 0.5, 'karma_delta')"
        )
        graph.conn.commit()

        # 创建元业力边
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

        # 验证普通业力边不受影响
        normal_edges = graph.conn.execute(
            "SELECT COUNT(*) FROM karma_edges WHERE source_tag = 'karma_delta'"
        ).fetchone()[0]
        assert normal_edges >= 1

        # 验证元业力边有 meta_karma 标记
        meta_edges = graph.conn.execute(
            "SELECT COUNT(*) FROM karma_edges WHERE source_tag = 'meta_karma'"
        ).fetchone()[0]
        assert meta_edges >= 1


class TestEnsureMetaPrefix:
    """_ensure_meta_prefix 辅助方法测试"""

    def test_add_prefix(self, mgr):
        """不以 meta: 开头时自动添加"""
        assert mgr._ensure_meta_prefix("物理") == "meta:物理"

    def test_already_has_prefix(self, mgr):
        """已有 meta: 前缀时不重复添加"""
        assert mgr._ensure_meta_prefix("meta:物理") == "meta:物理"

    def test_empty_string(self, mgr):
        """空字符串也添加前缀"""
        assert mgr._ensure_meta_prefix("") == "meta:"


class TestMetaSeedDisabled:
    """META_SEED_ENABLED=False 时的行为测试"""

    def test_generate_domain_monitors_disabled(self, mgr, graph):
        """禁用时 generate_domain_monitors 返回 0"""
        with patch("consciousness_sea.metacognition.meta_seed.META_SEED_ENABLED", False):
            assert mgr.generate_domain_monitors() == 0

    def test_generate_relation_monitors_disabled(self, mgr, graph):
        """禁用时 generate_relation_monitors 返回 0"""
        with patch("consciousness_sea.metacognition.meta_seed.META_SEED_ENABLED", False):
            assert mgr.generate_relation_monitors() == 0

    def test_generate_system_monitors_disabled(self, mgr, graph):
        """禁用时 generate_system_monitors 返回 0"""
        with patch("consciousness_sea.metacognition.meta_seed.META_SEED_ENABLED", False):
            assert mgr.generate_system_monitors() == 0

    def test_get_meta_seed_disabled(self, mgr, graph):
        """禁用时 get_meta_seed 返回 None"""
        with patch("consciousness_sea.metacognition.meta_seed.META_SEED_ENABLED", False):
            assert mgr.get_meta_seed("meta:医学") is None

    def test_list_meta_seeds_disabled(self, mgr, graph):
        """禁用时 list_meta_seeds 返回空列表"""
        with patch("consciousness_sea.metacognition.meta_seed.META_SEED_ENABLED", False):
            assert mgr.list_meta_seeds() == []

    def test_update_metrics_disabled(self, mgr, graph):
        """禁用时 update_metrics 返回 False"""
        with patch("consciousness_sea.metacognition.meta_seed.META_SEED_ENABLED", False):
            assert mgr.update_metrics("meta:医学", {}) is False

    def test_increment_metric_disabled(self, mgr, graph):
        """禁用时 increment_metric 返回 False"""
        with patch("consciousness_sea.metacognition.meta_seed.META_SEED_ENABLED", False):
            assert mgr.increment_metric("meta:医学", "conflict_frequency") is False

    def test_check_and_create_meta_karma_disabled(self, mgr, graph):
        """禁用时 check_and_create_meta_karma 返回 0"""
        with patch("consciousness_sea.metacognition.meta_seed.META_SEED_ENABLED", False):
            assert mgr.check_and_create_meta_karma() == 0

    def test_check_dormant_status_disabled(self, mgr, graph):
        """禁用时 check_dormant_status 返回 0"""
        with patch("consciousness_sea.metacognition.meta_seed.META_SEED_ENABLED", False):
            assert mgr.check_dormant_status() == 0

    def test_update_self_boundary_disabled(self, mgr, graph):
        """禁用时 update_self_boundary 返回 0"""
        with patch("consciousness_sea.metacognition.meta_seed.META_SEED_ENABLED", False):
            assert mgr.update_self_boundary() == 0

    def test_detect_unknown_domains_disabled(self, mgr, graph):
        """禁用时 detect_unknown_domains 返回 0"""
        with patch("consciousness_sea.metacognition.meta_seed.META_SEED_ENABLED", False):
            assert mgr.detect_unknown_domains() == 0


class TestCreateMetaSeedRecord:
    """_create_meta_seed_record 原子创建测试"""

    def test_create_both_tables(self, mgr, graph):
        """seeds 表和 meta_seeds 表同时创建"""
        result = mgr._create_meta_seed_record(
            "meta:测试领域", MetaSeedCategory.DOMAIN_MONITOR,
            dict(DOMAIN_MONITOR_DEFAULT_METRICS),
            source_domain="测试领域",
        )
        assert result is True

        # 验证 seeds 表
        row = graph.conn.execute(
            "SELECT * FROM seeds WHERE label = 'meta:测试领域'"
        ).fetchone()
        assert row is not None
        assert row["type"] == "META"
        assert row["domain"] == "元认知"
        assert row["activation"] == 0.0
        assert row["aliases"] == "[]"

        # 验证 meta_seeds 表
        row = graph.conn.execute(
            "SELECT * FROM meta_seeds WHERE label = 'meta:测试领域'"
        ).fetchone()
        assert row is not None
        assert row["category"] == "domain_monitor"
        assert row["status"] == "active"
        assert row["source_domain"] == "测试领域"

    def test_create_auto_prefix(self, mgr, graph):
        """label 不以 meta: 开头时自动添加前缀"""
        result = mgr._create_meta_seed_record(
            "测试领域", MetaSeedCategory.DOMAIN_MONITOR,
            dict(DOMAIN_MONITOR_DEFAULT_METRICS),
        )
        assert result is True

        # 验证自动添加了前缀
        row = graph.conn.execute(
            "SELECT * FROM meta_seeds WHERE label = 'meta:测试领域'"
        ).fetchone()
        assert row is not None

    def test_create_invalid_category(self, mgr, graph):
        """category 不合法时拒绝创建"""
        result = mgr._create_meta_seed_record(
            "meta:测试", "invalid_category",  # type: ignore
            {},
        )
        assert result is False

    def test_create_already_exists(self, mgr, graph):
        """元种子已存在时跳过创建"""
        mgr._create_meta_seed_record(
            "meta:测试领域", MetaSeedCategory.DOMAIN_MONITOR,
            dict(DOMAIN_MONITOR_DEFAULT_METRICS),
        )
        result = mgr._create_meta_seed_record(
            "meta:测试领域", MetaSeedCategory.DOMAIN_MONITOR,
            dict(DOMAIN_MONITOR_DEFAULT_METRICS),
        )
        assert result is False