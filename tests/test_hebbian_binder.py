"""
Phase 6 HebbianBinder 单元测试

覆盖：
- Hebbian 绑定器启动和停止
- 共同激活检测
- Hebbian 学习规则
- 反复共同激活→权重攀升
- Hebbian 绑定边权重上界
- Hebbian 绑定边 source_tag 标记
- Hebbian 绑定边关系类型
- Hebbian 负向衰减
- Hebbian 绑定器状态查询
- 时间窗口内大量共同激活
- 时间戳解析
"""

from __future__ import annotations

import sqlite3
import sys
import pathlib
from datetime import datetime, timezone
from unittest.mock import patch

import pytest

_root = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_root))
sys.path.insert(0, str(_root / "backend" / "src"))

from consciousness_sea.perception.hebbian_binder import HebbianBinder, HebbianBinderStatus
from consciousness_sea.perception.perception import PerceptActivationEvent, ConceptActivationEvent, PerceptionChannel
from consciousness_sea.domain.graph_db import GraphDB
from consciousness_sea.infrastructure.config import (
    HEBBIAN_TIME_WINDOW,
    HEBBIAN_LEARNING_RATE,
    HEBBIAN_NEGATIVE_DECAY_ENABLED,
    HEBBIAN_NEGATIVE_RATE,
    HEBBIAN_MAX_BINDINGS_PER_WINDOW,
    KARMA_MIN,
    KARMA_MAX,
)


# ═══════════════════════════════════════════════════════════
#  Fixtures
# ═══════════════════════════════════════════════════════════


def _build_test_db() -> sqlite3.Connection:
    """创建含 Phase 6 表的内存测试数据库"""
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
    """)

    # 插入测试种子
    seeds = [
        ("红色", "红色", "CONCEPT", "[]", "感知", "一种颜色"),
        ("percept:visual:red", "percept:visual:red", "PERCEPTUAL", "[]", "感知", ""),
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
    return db


@pytest.fixture
def graph():
    """创建内存数据库的 GraphDB 实例"""
    conn = _build_test_db()
    g = _make_graph_db(conn)
    yield g
    g.close()


@pytest.fixture
def binder(graph):
    """创建 HebbianBinder 实例"""
    return HebbianBinder(graph)


def _make_percept_event(seed: str = "percept:visual:red", ts_offset_ms: int = 0) -> PerceptActivationEvent:
    """创建感知激活事件"""
    now = datetime.now(timezone.utc)
    ts = now.isoformat(timespec='milliseconds')
    return PerceptActivationEvent(
        perceptual_seed=seed,
        activation=0.8,
        timestamp=ts,
        channel=PerceptionChannel.VISUAL,
    )


def _make_concept_event(seeds: list[str] | None = None, ts_offset_ms: int = 0) -> ConceptActivationEvent:
    """创建概念激活事件"""
    if seeds is None:
        seeds = ["红色"]
    now = datetime.now(timezone.utc)
    ts = now.isoformat(timespec='milliseconds')
    return ConceptActivationEvent(
        activated_seeds=seeds,
        timestamp=ts,
    )


# ═══════════════════════════════════════════════════════════
#  测试用例
# ═══════════════════════════════════════════════════════════


class TestHebbianBinderStartStop:
    """Hebbian 绑定器启动和停止测试"""

    def test_start(self, binder):
        """start() 启动 daemon 线程"""
        binder.start()
        assert binder._daemon_thread is not None
        assert binder._daemon_thread.name == "hebbian-binder"
        assert binder._daemon_thread.daemon is True
        binder.stop()

    def test_stop(self, binder):
        """stop() 优雅停止"""
        binder.start()
        binder.stop()
        assert binder._shutdown_event.is_set()


class TestCoActivationDetection:
    """共同激活检测测试"""

    def test_co_activation_creates_binding(self, binder, graph):
        """时间窗口内感知+概念事件 → 共同激活 → 创建绑定边"""
        binder.on_percept_activation(_make_percept_event())
        binder.on_concept_activation(_make_concept_event())

        # 手动触发一次检测
        binder._check_co_activation()

        # 验证绑定边已创建
        row = graph.conn.execute(
            "SELECT * FROM karma_edges WHERE source_tag = 'hebbian_binding'"
        ).fetchone()
        assert row is not None
        assert row["source"] == "percept:visual:red"
        assert row["target"] == "红色"
        assert row["relation"] == "HEBBIAN_BIND"

    def test_no_co_activation_without_events(self, binder, graph):
        """无事件时不创建绑定边"""
        binder._check_co_activation()
        row = graph.conn.execute(
            "SELECT * FROM karma_edges WHERE source_tag = 'hebbian_binding'"
        ).fetchone()
        assert row is None

    def test_no_co_activation_only_percept(self, binder, graph):
        """仅有感知事件时不创建绑定边"""
        binder.on_percept_activation(_make_percept_event())
        binder._check_co_activation()
        row = graph.conn.execute(
            "SELECT * FROM karma_edges WHERE source_tag = 'hebbian_binding'"
        ).fetchone()
        assert row is None


class TestHebbianLearningRule:
    """Hebbian 学习规则测试"""

    def test_initial_weight_is_learning_rate(self, binder, graph):
        """首次共同激活 → 初始权重为 HEBBIAN_LEARNING_RATE"""
        binder.on_percept_activation(_make_percept_event())
        binder.on_concept_activation(_make_concept_event())
        binder._check_co_activation()

        row = graph.conn.execute(
            "SELECT weight FROM karma_edges WHERE source_tag = 'hebbian_binding'"
        ).fetchone()
        assert row is not None
        assert abs(row["weight"] - HEBBIAN_LEARNING_RATE) < 0.001

    def test_repeated_co_activation_increases_weight(self, binder, graph):
        """反复共同激活 → 权重攀升"""
        for _ in range(8):
            binder.on_percept_activation(_make_percept_event())
            binder.on_concept_activation(_make_concept_event())
            binder._check_co_activation()

        row = graph.conn.execute(
            "SELECT weight FROM karma_edges WHERE source_tag = 'hebbian_binding'"
        ).fetchone()
        assert row is not None
        expected = HEBBIAN_LEARNING_RATE * 8
        assert abs(row["weight"] - expected) < 0.01


class TestWeightBounds:
    """权重边界测试"""

    def test_weight_upper_bound(self, binder, graph):
        """权重超过 KARMA_MAX → 裁剪到 2.0"""
        # 先创建一条绑定边
        binder.on_percept_activation(_make_percept_event())
        binder.on_concept_activation(_make_concept_event())
        binder._check_co_activation()

        # 手动设置权重接近上限
        graph.conn.execute(
            "UPDATE karma_edges SET weight = ? WHERE source_tag = 'hebbian_binding'",
            (KARMA_MAX - 0.005,),
        )
        graph.conn.commit()

        # 再次共同激活
        binder.on_percept_activation(_make_percept_event())
        binder.on_concept_activation(_make_concept_event())
        binder._check_co_activation()

        row = graph.conn.execute(
            "SELECT weight FROM karma_edges WHERE source_tag = 'hebbian_binding'"
        ).fetchone()
        assert row["weight"] <= KARMA_MAX


class TestBindingEdgeProperties:
    """绑定边属性测试"""

    def test_source_tag(self, binder, graph):
        """source_tag = 'hebbian_binding'"""
        binder.on_percept_activation(_make_percept_event())
        binder.on_concept_activation(_make_concept_event())
        binder._check_co_activation()

        row = graph.conn.execute(
            "SELECT source_tag FROM karma_edges WHERE source = 'percept:visual:red'"
        ).fetchone()
        assert row["source_tag"] == "hebbian_binding"

    def test_relation_type(self, binder, graph):
        """relation = 'HEBBIAN_BIND'"""
        binder.on_percept_activation(_make_percept_event())
        binder.on_concept_activation(_make_concept_event())
        binder._check_co_activation()

        row = graph.conn.execute(
            "SELECT relation FROM karma_edges WHERE source = 'percept:visual:red'"
        ).fetchone()
        assert row["relation"] == "HEBBIAN_BIND"


class TestNegativeDecay:
    """负向衰减测试"""

    def test_negative_decay_enabled(self, binder, graph):
        """感知激活但概念未激活 → 减弱绑定边权重

        负向衰减在 _check_co_activation 中，仅在存在共同激活时才运行。
        策略：同时发送两个感知事件（一个与概念共同激活，一个不在时间窗口内），
        非共同激活的感知元种子的绑定边权重应被衰减。
        """
        # 先为 percept:visual:green 创建与"红色"的绑定边（多次共同激活使权重高于 KARMA_MIN）
        for _ in range(5):
            binder.on_percept_activation(PerceptActivationEvent(
                perceptual_seed="percept:visual:green",
                activation=0.7,
                timestamp=datetime.now(timezone.utc).isoformat(timespec='milliseconds'),
                channel=PerceptionChannel.VISUAL,
            ))
            binder.on_concept_activation(ConceptActivationEvent(
                activated_seeds=["红色"],
                timestamp=datetime.now(timezone.utc).isoformat(timespec='milliseconds'),
            ))
            binder._check_co_activation()

        # 验证绑定边存在且有权重
        row = graph.conn.execute(
            "SELECT weight FROM karma_edges WHERE source = 'percept:visual:green' AND source_tag = 'hebbian_binding'"
        ).fetchone()
        assert row is not None
        weight_before = row["weight"]
        assert weight_before > KARMA_MIN

        # 启用负向衰减：
        # 发送 percept:visual:red（与"红色"共同激活）+ percept:visual:green（时间窗口外）
        # percept:visual:green 不在共同激活列表中 → 其绑定边权重应被衰减
        now_ts = datetime.now(timezone.utc).isoformat(timespec='milliseconds')
        far_past_ts = "2020-01-01T00:00:00.000+00:00"

        with patch("consciousness_sea.perception.hebbian_binder.HEBBIAN_NEGATIVE_DECAY_ENABLED", True):
            # percept:visual:red 与"红色"在时间窗口内 → 共同激活
            binder.on_percept_activation(PerceptActivationEvent(
                perceptual_seed="percept:visual:red",
                activation=0.8,
                timestamp=now_ts,
                channel=PerceptionChannel.VISUAL,
            ))
            # percept:visual:green 时间戳在窗口外 → 不共同激活
            binder.on_percept_activation(PerceptActivationEvent(
                perceptual_seed="percept:visual:green",
                activation=0.7,
                timestamp=far_past_ts,
                channel=PerceptionChannel.VISUAL,
            ))
            binder.on_concept_activation(ConceptActivationEvent(
                activated_seeds=["红色"],
                timestamp=now_ts,
            ))
            binder._check_co_activation()

        # percept:visual:green 的权重应该减少（负向衰减）
        row_after = graph.conn.execute(
            "SELECT weight FROM karma_edges WHERE source = 'percept:visual:green' AND source_tag = 'hebbian_binding'"
        ).fetchone()
        if row_after is not None:
            assert row_after["weight"] < weight_before

    def test_negative_decay_disabled_by_default(self, binder, graph):
        """默认不启用负向衰减"""
        assert HEBBIAN_NEGATIVE_DECAY_ENABLED is False


class TestMaxBindingsPerWindow:
    """时间窗口内大量共同激活测试"""

    def test_max_bindings_limit(self, binder, graph):
        """按 MAX_BINDINGS_PER_WINDOW 限制绑定数"""
        # 创建多个感知事件和概念事件
        for i in range(100):
            binder.on_percept_activation(PerceptActivationEvent(
                perceptual_seed=f"percept:visual:color_{i}",
                activation=0.8,
                timestamp=datetime.now(timezone.utc).isoformat(timespec='milliseconds'),
                channel=PerceptionChannel.VISUAL,
            ))
        binder.on_concept_activation(ConceptActivationEvent(
            activated_seeds=["红色"],
            timestamp=datetime.now(timezone.utc).isoformat(timespec='milliseconds'),
        ))

        binder._check_co_activation()

        count = graph.conn.execute(
            "SELECT COUNT(*) FROM karma_edges WHERE source_tag = 'hebbian_binding'"
        ).fetchone()[0]
        assert count <= HEBBIAN_MAX_BINDINGS_PER_WINDOW


class TestBinderStatus:
    """绑定器状态查询测试"""

    def test_get_status(self, binder, graph):
        """get_status() 返回 HebbianBinderStatus"""
        binder.on_percept_activation(_make_percept_event())
        binder.on_concept_activation(_make_concept_event())
        binder._check_co_activation()

        status = binder.get_status()
        assert isinstance(status, HebbianBinderStatus)
        assert status.total_bindings >= 1
        assert "visual" in status.bindings_by_channel

    def test_get_status_not_running(self, binder):
        """未启动时 is_running 为 False"""
        status = binder.get_status()
        assert status.is_running is False


class TestTimestampParsing:
    """时间戳解析测试"""

    def test_parse_valid_timestamp(self):
        """解析有效 ISO 8601 时间戳"""
        ts = "2025-01-01T12:00:00.000+00:00"
        result = HebbianBinder._parse_timestamp(ts)
        assert result is not None

    def test_parse_empty_timestamp(self):
        """空时间戳返回 None"""
        result = HebbianBinder._parse_timestamp("")
        assert result is None

    def test_parse_invalid_timestamp(self):
        """无效时间戳返回 None"""
        result = HebbianBinder._parse_timestamp("not-a-timestamp")
        assert result is None