"""
Phase 6 PerceptionManager 单元测试

覆盖：
- 感知管理器启动和停止
- 感知通道启停控制
- 感知激活事件分发
- 概念种子激活事件转发
- 感知通道状态查询
- 感知通道硬件不可用降级
- 感知功能开关
- 感知元种子创建（seeds 表 + perceptual_seeds 表）
- 感知元种子 label 前缀约束
- 感知元种子通道类型枚举
- 感知元种子去重
- 感知元种子可扩展
- 预设感知元种子自动生成
"""

from __future__ import annotations

import sqlite3
import sys
import pathlib
from datetime import datetime, timezone
from unittest.mock import patch, MagicMock

import pytest

_root = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_root))
sys.path.insert(0, str(_root / "backend" / "src"))

from consciousness_sea.perception.perception import (
    PerceptionManager,
    PerceptionChannel,
    PerceptualSeedStatus,
    PerceptActivationEvent,
    ConceptActivationEvent,
    ChannelStatus,
    PerceptionManagerStatus,
)
from consciousness_sea.domain.graph_db import GraphDB
from consciousness_sea.infrastructure.config import PERCEPTION_ENABLED


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

    # 插入测试种子
    seeds = [
        ("红色", "红色", "CONCEPT", "[]", "感知", "一种颜色"),
        ("发热", "发热", "CONCEPT", "[]", "医学", "体温升高"),
        ("量子力学", "量子力学", "CONCEPT", "[]", "物理", "quantum mechanics"),
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
def pm(graph):
    """创建 PerceptionManager 实例"""
    return PerceptionManager(graph)


# ═══════════════════════════════════════════════════════════
#  测试用例
# ═══════════════════════════════════════════════════════════


class TestPerceptionEnums:
    """枚举与数据类测试"""

    def test_perception_channel_values(self):
        """PerceptionChannel 枚举包含三个值"""
        assert PerceptionChannel.VISUAL.value == "visual"
        assert PerceptionChannel.AUDITORY.value == "auditory"
        assert PerceptionChannel.SOMATIC.value == "somatic"

    def test_perceptual_seed_status_values(self):
        """PerceptualSeedStatus 枚举包含三个值"""
        assert PerceptualSeedStatus.ACTIVE.value == "active"
        assert PerceptualSeedStatus.DISABLED.value == "disabled"
        assert PerceptualSeedStatus.RETIRED.value == "retired"

    def test_percept_activation_event(self):
        """PerceptActivationEvent 数据类"""
        now = datetime.now(timezone.utc).isoformat(timespec='milliseconds')
        event = PerceptActivationEvent(
            perceptual_seed="percept:visual:red",
            activation=0.8,
            timestamp=now,
            channel=PerceptionChannel.VISUAL,
        )
        assert event.perceptual_seed == "percept:visual:red"
        assert event.activation == 0.8
        assert event.channel == PerceptionChannel.VISUAL

    def test_concept_activation_event(self):
        """ConceptActivationEvent 数据类"""
        now = datetime.now(timezone.utc).isoformat(timespec='milliseconds')
        event = ConceptActivationEvent(
            activated_seeds=["红色", "颜色"],
            timestamp=now,
        )
        assert event.activated_seeds == ["红色", "颜色"]

    def test_channel_status(self):
        """ChannelStatus 数据类"""
        cs = ChannelStatus(running=True, last_activation="T1", mock_mode=True)
        assert cs.running is True
        assert cs.mock_mode is True
        assert cs.consecutive_failures == 0

    def test_perception_manager_status(self):
        """PerceptionManagerStatus 数据类"""
        status = PerceptionManagerStatus(enabled=True, total_perceptual_seeds=16)
        assert status.enabled is True
        assert status.total_perceptual_seeds == 16


class TestPerceptionManagerStartStop:
    """感知管理器启动和停止测试"""

    def test_start_generates_seeds(self, pm, graph):
        """start() 生成预设感知元种子"""
        with patch("consciousness_sea.perception.perception.PERCEPTION_ENABLED", True), \
             patch("consciousness_sea.perception.visual_anchor.VisualAnchor") as MockVA, \
             patch("consciousness_sea.perception.audio_anchor.AudioAnchor") as MockAA, \
             patch("consciousness_sea.perception.somatic_anchor.SomaticAnchor") as MockSA, \
             patch("consciousness_sea.perception.hebbian_binder.HebbianBinder") as MockHB, \
             patch("consciousness_sea.perception.multimodal_aligner.MultimodalAligner") as MockMA:
            MockVA.return_value._mock_mode = True
            MockAA.return_value._mock_mode = True
            pm.start()

        # 验证 16 个预设种子已创建
        count = graph.conn.execute(
            "SELECT COUNT(*) FROM perceptual_seeds"
        ).fetchone()[0]
        assert count == 16

        # 验证 seeds 表中也有对应的 PERCEPTUAL 记录
        seed_count = graph.conn.execute(
            "SELECT COUNT(*) FROM seeds WHERE type = 'PERCEPTUAL'"
        ).fetchone()[0]
        assert seed_count == 16

    def test_start_disabled(self, pm, graph):
        """PERCEPTION_ENABLED=False 时 start() 不执行任何操作"""
        with patch("consciousness_sea.perception.perception.PERCEPTION_ENABLED", False):
            pm.start()

        count = graph.conn.execute(
            "SELECT COUNT(*) FROM perceptual_seeds"
        ).fetchone()[0]
        assert count == 0

    def test_stop(self, pm, graph):
        """stop() 优雅停止所有组件"""
        with patch("consciousness_sea.perception.perception.PERCEPTION_ENABLED", True), \
             patch("consciousness_sea.perception.visual_anchor.VisualAnchor") as MockVA, \
             patch("consciousness_sea.perception.audio_anchor.AudioAnchor") as MockAA, \
             patch("consciousness_sea.perception.somatic_anchor.SomaticAnchor") as MockSA, \
             patch("consciousness_sea.perception.hebbian_binder.HebbianBinder") as MockHB, \
             patch("consciousness_sea.perception.multimodal_aligner.MultimodalAligner") as MockMA:
            MockVA.return_value._mock_mode = True
            MockAA.return_value._mock_mode = True
            pm.start()
            pm.stop()

        assert pm._shutdown_event.is_set()


class TestChannelControl:
    """感知通道启停控制测试"""

    def test_start_channel_visual(self, pm, graph):
        """start_channel('visual') 启动视觉通道"""
        with patch("consciousness_sea.perception.visual_anchor.VisualAnchor") as MockVA, \
             patch("consciousness_sea.perception.perception.PERCEPTION_ENABLED", True):
            MockVA.return_value._mock_mode = True
            result = pm.start_channel("visual")
        assert result is True
        assert pm._channel_status["visual"].running is True

    def test_start_channel_unknown(self, pm, graph):
        """start_channel('unknown') 返回 False"""
        with patch("consciousness_sea.perception.perception.PERCEPTION_ENABLED", True):
            result = pm.start_channel("olfactory")
        assert result is False

    def test_stop_channel(self, pm, graph):
        """stop_channel('visual') 停止视觉通道"""
        with patch("consciousness_sea.perception.visual_anchor.VisualAnchor") as MockVA, \
             patch("consciousness_sea.perception.perception.PERCEPTION_ENABLED", True):
            MockVA.return_value._mock_mode = True
            pm.start_channel("visual")
            result = pm.stop_channel("visual")
        assert result is True
        assert pm._channel_status["visual"].running is False

    def test_stop_channel_not_running(self, pm, graph):
        """stop_channel 对未运行的通道返回 False"""
        result = pm.stop_channel("visual")
        assert result is False

    def test_start_channel_disabled(self, pm, graph):
        """PERCEPTION_ENABLED=False 时 start_channel 返回 False"""
        with patch("consciousness_sea.perception.perception.PERCEPTION_ENABLED", False):
            result = pm.start_channel("visual")
        assert result is False


class TestEventDispatch:
    """事件分发测试"""

    def test_on_percept_activation(self, pm, graph):
        """on_percept_activation() 分发事件给 Hebbian 绑定器"""
        now = datetime.now(timezone.utc).isoformat(timespec='milliseconds')
        event = PerceptActivationEvent(
            perceptual_seed="percept:visual:red",
            activation=0.8,
            timestamp=now,
            channel=PerceptionChannel.VISUAL,
        )

        # 先创建感知元种子
        pm._create_perceptual_seed_record(
            "percept:visual:red", "visual", "红色通道占比", 0.3
        )

        mock_binder = MagicMock()
        pm._hebbian_binder = mock_binder

        pm.on_percept_activation(event)

        # 验证转发给 Hebbian 绑定器
        mock_binder.on_percept_activation.assert_called_once_with(event)

        # 验证 seeds 表激活值更新
        row = graph.conn.execute(
            "SELECT activation FROM seeds WHERE label = 'percept:visual:red'"
        ).fetchone()
        assert row is not None
        assert row["activation"] == 0.8

        # 验证 perception_events 表写入
        row = graph.conn.execute(
            "SELECT * FROM perception_events WHERE perceptual_seed = 'percept:visual:red'"
        ).fetchone()
        assert row is not None
        assert row["activation"] == 0.8

    def test_on_percept_activation_updates_channel_status(self, pm, graph):
        """on_percept_activation() 更新通道状态"""
        now = datetime.now(timezone.utc).isoformat(timespec='milliseconds')
        event = PerceptActivationEvent(
            perceptual_seed="percept:visual:red",
            activation=0.8,
            timestamp=now,
            channel=PerceptionChannel.VISUAL,
        )
        pm._hebbian_binder = MagicMock()
        pm.on_percept_activation(event)

        assert pm._channel_status["visual"].last_activation == now

    def test_on_concept_activation(self, pm, graph):
        """on_concept_activation() 转发给 Hebbian 绑定器"""
        now = datetime.now(timezone.utc).isoformat(timespec='milliseconds')
        event = ConceptActivationEvent(
            activated_seeds=["红色", "颜色"],
            timestamp=now,
        )

        mock_binder = MagicMock()
        pm._hebbian_binder = mock_binder

        pm.on_concept_activation(event)
        mock_binder.on_concept_activation.assert_called_once_with(event)

    def test_on_concept_activation_no_binder(self, pm, graph):
        """Hebbian 绑定器不存在时 on_concept_activation 不报错"""
        now = datetime.now(timezone.utc).isoformat(timespec='milliseconds')
        event = ConceptActivationEvent(
            activated_seeds=["红色"],
            timestamp=now,
        )
        pm._hebbian_binder = None
        pm.on_concept_activation(event)  # 不应抛异常


class TestStatusQuery:
    """状态查询测试"""

    def test_get_status(self, pm, graph):
        """get_status() 返回 PerceptionManagerStatus"""
        pm._create_perceptual_seed_record(
            "percept:visual:red", "visual", "红色通道占比", 0.3
        )
        status = pm.get_status()
        assert isinstance(status, PerceptionManagerStatus)
        assert status.total_perceptual_seeds >= 1
        assert "visual" in status.channels
        assert "auditory" in status.channels
        assert "somatic" in status.channels


class TestPerceptualSeedCreation:
    """感知元种子创建测试"""

    def test_create_perceptual_seed_both_tables(self, pm, graph):
        """创建感知元种子：seeds 表和 perceptual_seeds 表同时创建"""
        result = pm._create_perceptual_seed_record(
            "percept:visual:red", "visual", "红色通道占比", 0.3
        )
        assert result is True

        # 验证 seeds 表
        row = graph.conn.execute(
            "SELECT * FROM seeds WHERE label = 'percept:visual:red'"
        ).fetchone()
        assert row is not None
        assert row["type"] == "PERCEPTUAL"
        assert row["domain"] == "感知"
        assert row["activation"] == 0.0

        # 验证 perceptual_seeds 表
        row = graph.conn.execute(
            "SELECT * FROM perceptual_seeds WHERE label = 'percept:visual:red'"
        ).fetchone()
        assert row is not None
        assert row["channel"] == "visual"
        assert row["feature_description"] == "红色通道占比"
        assert row["activation_threshold"] == 0.3
        assert row["status"] == "active"

    def test_create_perceptual_seed_auto_prefix(self, pm, graph):
        """label 不以 "percept:" 开头时自动添加前缀"""
        result = pm._create_perceptual_seed_record(
            "visual:yellow", "visual", "黄色通道占比", 0.25
        )
        assert result is True

        row = graph.conn.execute(
            "SELECT * FROM perceptual_seeds WHERE label = 'percept:visual:yellow'"
        ).fetchone()
        assert row is not None

    def test_create_perceptual_seed_invalid_channel(self, pm, graph):
        """channel 不合法时拒绝创建"""
        result = pm._create_perceptual_seed_record(
            "percept:olfactory:rose", "olfactory", "玫瑰气味", 0.3
        )
        assert result is False

    def test_create_perceptual_seed_dedup(self, pm, graph):
        """感知元种子去重：已存在时跳过创建"""
        pm._create_perceptual_seed_record(
            "percept:visual:red", "visual", "红色通道占比", 0.3
        )
        result = pm._create_perceptual_seed_record(
            "percept:visual:red", "visual", "红色通道占比", 0.3
        )
        assert result is False

    def test_add_perceptual_seed(self, pm, graph):
        """add_perceptual_seed() 创建自定义感知元种子"""
        result = pm.add_perceptual_seed(
            label="percept:visual:yellow",
            channel="visual",
            feature_description="黄色通道占比",
            activation_threshold=0.25,
        )
        assert result is True

    def test_ensure_percept_prefix(self):
        """_ensure_percept_prefix() 辅助方法"""
        assert PerceptionManager._ensure_percept_prefix("visual:red") == "percept:visual:red"
        assert PerceptionManager._ensure_percept_prefix("percept:visual:red") == "percept:visual:red"


class TestPresetSeeds:
    """预设感知元种子自动生成测试"""

    def test_generate_preset_seeds(self, pm, graph):
        """_generate_preset_perceptual_seeds() 生成 16 个预设种子"""
        created = pm._generate_preset_perceptual_seeds()
        assert created == 16

        # 验证视觉 6 个
        visual_count = graph.conn.execute(
            "SELECT COUNT(*) FROM perceptual_seeds WHERE channel = 'visual'"
        ).fetchone()[0]
        assert visual_count == 6

        # 验证听觉 5 个
        auditory_count = graph.conn.execute(
            "SELECT COUNT(*) FROM perceptual_seeds WHERE channel = 'auditory'"
        ).fetchone()[0]
        assert auditory_count == 5

        # 验证本体 5 个
        somatic_count = graph.conn.execute(
            "SELECT COUNT(*) FROM perceptual_seeds WHERE channel = 'somatic'"
        ).fetchone()[0]
        assert somatic_count == 5

    def test_generate_preset_seeds_idempotent(self, pm, graph):
        """预设种子生成幂等：第二次调用返回 0"""
        pm._generate_preset_perceptual_seeds()
        created = pm._generate_preset_perceptual_seeds()
        assert created == 0


class TestHardwareDegradation:
    """感知通道硬件不可用降级测试"""

    def test_visual_channel_unavailable(self, pm, graph):
        """视觉通道启动失败时进入 disabled 状态"""
        with patch("consciousness_sea.perception.visual_anchor.VisualAnchor", side_effect=Exception("camera not found")), \
             patch("consciousness_sea.perception.perception.PERCEPTION_ENABLED", True):
            result = pm._start_visual_channel()
        assert result is False
        assert pm._channel_status["visual"].running is False


class TestQueryMethods:
    """查询方法测试"""

    def test_get_perceptual_seed(self, pm, graph):
        """get_perceptual_seed() 返回种子详情"""
        pm._create_perceptual_seed_record(
            "percept:visual:red", "visual", "红色通道占比", 0.3
        )
        seed = pm.get_perceptual_seed("percept:visual:red")
        assert seed is not None
        assert seed["label"] == "percept:visual:red"
        assert seed["channel"] == "visual"
        assert "hebbian_bindings" in seed

    def test_get_perceptual_seed_not_found(self, pm, graph):
        """get_perceptual_seed() 不存在时返回 None"""
        seed = pm.get_perceptual_seed("percept:not:exist")
        assert seed is None

    def test_list_perceptual_seeds(self, pm, graph):
        """list_perceptual_seeds() 返回种子列表"""
        pm._generate_preset_perceptual_seeds()
        seeds = pm.list_perceptual_seeds()
        assert len(seeds) == 16

    def test_list_perceptual_seeds_filter(self, pm, graph):
        """list_perceptual_seeds(channel='visual') 过滤正确"""
        pm._generate_preset_perceptual_seeds()
        seeds = pm.list_perceptual_seeds(channel="visual")
        assert len(seeds) == 6
        assert all(s["channel"] == "visual" for s in seeds)

    def test_list_hebbian_bindings(self, pm, graph):
        """list_hebbian_bindings() 返回绑定边列表"""
        bindings = pm.list_hebbian_bindings()
        assert isinstance(bindings, list)

    def test_list_perception_events(self, pm, graph):
        """list_perception_events() 返回事件列表"""
        events = pm.list_perception_events(limit=10)
        assert isinstance(events, list)

    def test_run_multimodal_alignment_no_aligner(self, pm, graph):
        """multimodal_aligner 为 None 时返回空列表"""
        pm._multimodal_aligner = None
        result = pm.run_multimodal_alignment()
        assert result == []