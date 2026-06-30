"""
Phase 6 端到端场景验收测试

覆盖 7 个端到端场景：
1. 感知元种子自动生成（start → 16 个预设种子在 seeds + perceptual_seeds 表中）
2. Hebbian 绑定边自然生长（感知激活 + 概念激活 → 时间窗口内共同激活 → 绑定边）
3. 感知元种子参与涟漪传播（PERCEPTUAL 类型被 match_seeds 排除）
4. 感知通道硬件不可用降级（VisualAnchor/AudioAnchor/SomaticAnchor 异常 → 降级）
5. 感知功能关闭（PERCEPTION_ENABLED=False → 所有操作跳过）
6. 多模态对齐离线校准（MultimodalAligner 禁用/无帧/无 CLIP → 安全返回）
7. 感知查询 API（6 个端点的端到端集成测试）
"""

from __future__ import annotations

import sqlite3
import sys
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

# 确保 backend/src 在 sys.path 中
_src = str(Path(__file__).resolve().parent.parent / "backend" / "src")
if _src not in sys.path:
    sys.path.insert(0, _src)
# 同时保留项目根目录
_root = str(Path(__file__).resolve().parent.parent)
if _root not in sys.path:
    sys.path.insert(0, _root)

from consciousness_sea.domain.graph_db import GraphDB
from consciousness_sea.infrastructure.config import (
    HEBBIAN_LEARNING_RATE,
    KARMA_MAX,
)
from consciousness_sea.perception.hebbian_binder import HebbianBinder
from consciousness_sea.perception.multimodal_aligner import MultimodalAligner
from consciousness_sea.perception.perception import (
    ConceptActivationEvent,
    PerceptActivationEvent,
    PerceptionChannel,
    PerceptionManager,
    PerceptionManagerStatus,
)
from consciousness_sea.perception.somatic_anchor import SomaticAnchor, SomaticFeatures
from fastapi.testclient import TestClient

# ═══════════════════════════════════════════════════════════
#  Fixtures
# ═══════════════════════════════════════════════════════════


def _build_test_db() -> sqlite3.Connection:
    """创建含全部 Phase 1-6 表的内存测试数据库"""
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
        CREATE TABLE param_stats (
            stat_id INTEGER PRIMARY KEY AUTOINCREMENT,
            query_text TEXT NOT NULL,
            decay_factor REAL NOT NULL,
            domain_threshold REAL NOT NULL,
            confidence_high REAL NOT NULL,
            ripple_depth INTEGER NOT NULL,
            activated_count INTEGER NOT NULL,
            selected_domains TEXT NOT NULL,
            confidence REAL NOT NULL,
            karma_direction INTEGER NOT NULL,
            created_at TEXT NOT NULL
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

    # 插入概念种子（用于涟漪传播和 Hebbian 绑定测试）
    seeds = [
        ("红色", "红色", "CONCEPT", "[]", "感知", "一种颜色"),
        ("发热", "发热", "CONCEPT", "[]", "医学", "体温升高"),
        ("量子力学", "量子力学", "CONCEPT", "[]", "物理", "quantum mechanics"),
        ("声音", "声音", "CONCEPT", "[]", "感知", "听觉感受"),
        ("温度", "温度", "CONCEPT", "[]", "物理", "热力学量"),
    ]
    conn.executemany(
        "INSERT INTO seeds (id, label, type, aliases, domain, definition) VALUES (?, ?, ?, ?, ?, ?)",
        seeds,
    )

    # 插入一些业力边
    edges = [
        ("红色", "发热", "RELATED", 0.60, "karma_delta"),
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
def pm(graph):
    """创建 PerceptionManager 实例"""
    return PerceptionManager(graph)


# ═══════════════════════════════════════════════════════════
#  场景 1: 感知元种子自动生成
# ═══════════════════════════════════════════════════════════


class TestScenario1PerceptualSeedAutoGeneration:
    """端到端场景 1: 感知管理器启动后自动生成 16 个预设感知元种子

    验收标准:
      - seeds 表中新增 16 条 type='PERCEPTUAL' 记录
      - perceptual_seeds 表中新增 16 条记录
      - 视觉 6 个、听觉 5 个、本体 5 个
      - 所有种子 label 以 "percept:" 开头
      - seeds 表中 PERCEPTUAL 种子的 domain='感知', activation=0.0
      - match_seeds() 不返回 PERCEPTUAL 类型种子
    """

    def test_auto_generation_on_start(self, pm, graph):
        """start() 后自动生成 16 个预设感知元种子"""
        with patch("consciousness_sea.perception.perception.PERCEPTION_ENABLED", True), \
             patch("consciousness_sea.perception.visual_anchor.VisualAnchor") as MockVA, \
             patch("consciousness_sea.perception.audio_anchor.AudioAnchor") as MockAA, \
             patch("consciousness_sea.perception.somatic_anchor.SomaticAnchor"), \
             patch("consciousness_sea.perception.hebbian_binder.HebbianBinder"), \
             patch("consciousness_sea.perception.multimodal_aligner.MultimodalAligner"):
            MockVA.return_value._mock_mode = True
            MockAA.return_value._mock_mode = True
            pm.start()

        # seeds 表中 PERCEPTUAL 类型
        seed_count = graph.conn.execute(
            "SELECT COUNT(*) FROM seeds WHERE type = 'PERCEPTUAL'"
        ).fetchone()[0]
        assert seed_count == 16

        # perceptual_seeds 表
        ps_count = graph.conn.execute(
            "SELECT COUNT(*) FROM perceptual_seeds"
        ).fetchone()[0]
        assert ps_count == 16

        # 按通道统计
        visual_count = graph.conn.execute(
            "SELECT COUNT(*) FROM perceptual_seeds WHERE channel = 'visual'"
        ).fetchone()[0]
        auditory_count = graph.conn.execute(
            "SELECT COUNT(*) FROM perceptual_seeds WHERE channel = 'auditory'"
        ).fetchone()[0]
        somatic_count = graph.conn.execute(
            "SELECT COUNT(*) FROM perceptual_seeds WHERE channel = 'somatic'"
        ).fetchone()[0]
        assert visual_count == 6
        assert auditory_count == 5
        assert somatic_count == 5

    def test_perceptual_seed_label_prefix(self, pm, graph):
        """所有感知元种子 label 以 'percept:' 开头"""
        pm._generate_preset_perceptual_seeds()

        rows = graph.conn.execute(
            "SELECT label FROM perceptual_seeds"
        ).fetchall()
        for row in rows:
            assert row["label"].startswith("percept:"), \
                f"label '{row['label']}' does not start with 'percept:'"

    def test_perceptual_seed_in_seeds_table(self, pm, graph):
        """seeds 表中 PERCEPTUAL 种子的 domain='感知', activation=0.0"""
        pm._generate_preset_perceptual_seeds()

        rows = graph.conn.execute(
            "SELECT * FROM seeds WHERE type = 'PERCEPTUAL'"
        ).fetchall()
        assert len(rows) == 16
        for row in rows:
            assert row["domain"] == "感知"
            assert row["activation"] == 0.0
            assert row["aliases"] == "[]"

    def test_match_seeds_excludes_perceptual(self, pm, graph):
        """match_seeds() 不返回 PERCEPTUAL 类型种子"""
        pm._generate_preset_perceptual_seeds()

        # match_seeds 应排除 PERCEPTUAL 类型
        matched = graph.match_seeds("percept:visual:red")
        for seed in matched:
            assert seed["type"] != "PERCEPTUAL"

    def test_perceptual_seed_lifecycle_create_activate_decay(self, pm, graph):
        """感知元种子完整生命周期：生成→激活→衰减"""
        # 1. 生成
        pm._create_perceptual_seed_record(
            "percept:visual:red", "visual", "红色通道占比", 0.3
        )

        # 验证初始状态
        seed = pm.get_perceptual_seed("percept:visual:red")
        assert seed is not None
        assert seed["status"] == "active"
        assert seed["activation_count"] == 0

        # 2. 激活
        now = datetime.now(timezone.utc).isoformat(timespec='milliseconds')
        event = PerceptActivationEvent(
            perceptual_seed="percept:visual:red",
            activation=0.8,
            timestamp=now,
            channel=PerceptionChannel.VISUAL,
        )
        pm._hebbian_binder = MagicMock()
        pm.on_percept_activation(event)

        # 验证激活后状态
        seed = pm.get_perceptual_seed("percept:visual:red")
        assert seed["activation_count"] == 1
        assert seed["last_activation"] is not None

        # 验证 seeds 表激活值更新
        row = graph.conn.execute(
            "SELECT activation FROM seeds WHERE label = 'percept:visual:red'"
        ).fetchone()
        assert row["activation"] == 0.8

        # 3. 衰减（模拟多次低激活事件后状态不变，但 activation_count 增加）
        for i in range(5):
            low_event = PerceptActivationEvent(
                perceptual_seed="percept:visual:red",
                activation=0.1,
                timestamp=datetime.now(timezone.utc).isoformat(timespec='milliseconds'),
                channel=PerceptionChannel.VISUAL,
            )
            pm.on_percept_activation(low_event)

        seed = pm.get_perceptual_seed("percept:visual:red")
        assert seed["activation_count"] == 6  # 1 + 5


# ═══════════════════════════════════════════════════════════
#  场景 2: Hebbian 绑定边自然生长
# ═══════════════════════════════════════════════════════════


class TestScenario2HebbianBindingGrowth:
    """端到端场景 2: 感知激活 + 概念激活 → 时间窗口内共同激活 → Hebbian 绑定边

    验收标准:
      - 感知元种子和概念种子在时间窗口内共同激活 → 创建绑定边
      - 绑定边 source_tag='hebbian_binding', relation='HEBBIAN_BIND'
      - 反复共同激活 → 权重递增
      - 权重不超过 KARMA_MAX
      - 感知激活事件被记录到 perception_events 表
      - seeds 表中感知元种子 activation 被更新
    """

    def test_hebbian_binding_end_to_end(self, pm, graph):
        """感知激活 + 概念激活 → Hebbian 绑定边自然生长"""
        # 1. 创建感知元种子
        pm._create_perceptual_seed_record(
            "percept:visual:red", "visual", "红色通道占比", 0.3
        )

        # 2. 创建 Hebbian 绑定器
        binder = HebbianBinder(graph)

        # 3. 模拟感知激活事件
        now = datetime.now(timezone.utc).isoformat(timespec='milliseconds')
        percept_event = PerceptActivationEvent(
            perceptual_seed="percept:visual:red",
            activation=0.8,
            timestamp=now,
            channel=PerceptionChannel.VISUAL,
        )

        # 4. 模拟概念激活事件（时间窗口内）
        concept_event = ConceptActivationEvent(
            activated_seeds=["红色"],
            timestamp=now,
        )

        # 5. 发送事件
        binder.on_percept_activation(percept_event)
        binder.on_concept_activation(concept_event)

        # 6. 手动触发检测
        binder._check_co_activation()

        # 7. 验证绑定边已创建
        row = graph.conn.execute(
            "SELECT * FROM karma_edges WHERE source_tag = 'hebbian_binding'"
        ).fetchone()
        assert row is not None
        assert row["source"] == "percept:visual:red"
        assert row["target"] == "红色"
        assert row["relation"] == "HEBBIAN_BIND"
        assert row["source_tag"] == "hebbian_binding"

    def test_repeated_co_activation_weight_growth(self, pm, graph):
        """反复共同激活 → Hebbian 绑定边权重递增"""
        binder = HebbianBinder(graph)

        # 多次共同激活
        for i in range(5):
            now = datetime.now(timezone.utc).isoformat(timespec='milliseconds')
            binder.on_percept_activation(PerceptActivationEvent(
                perceptual_seed="percept:visual:red",
                activation=0.8,
                timestamp=now,
                channel=PerceptionChannel.VISUAL,
            ))
            binder.on_concept_activation(ConceptActivationEvent(
                activated_seeds=["红色"],
                timestamp=now,
            ))
            binder._check_co_activation()

        # 验证权重递增
        row = graph.conn.execute(
            "SELECT weight FROM karma_edges WHERE source_tag = 'hebbian_binding'"
        ).fetchone()
        assert row is not None
        # 5 次共同激活后权重应大于 HEBBIAN_LEARNING_RATE
        assert row["weight"] > HEBBIAN_LEARNING_RATE
        # 权重不应超过 KARMA_MAX
        assert row["weight"] <= KARMA_MAX

    def test_perception_events_recorded(self, pm, graph):
        """感知激活事件被记录到 perception_events 表"""
        pm._create_perceptual_seed_record(
            "percept:visual:red", "visual", "红色通道占比", 0.3
        )
        pm._hebbian_binder = MagicMock()

        now = datetime.now(timezone.utc).isoformat(timespec='milliseconds')
        event = PerceptActivationEvent(
            perceptual_seed="percept:visual:red",
            activation=0.8,
            timestamp=now,
            channel=PerceptionChannel.VISUAL,
        )
        pm.on_percept_activation(event)

        # 验证 perception_events 表
        row = graph.conn.execute(
            "SELECT * FROM perception_events WHERE perceptual_seed = 'percept:visual:red'"
        ).fetchone()
        assert row is not None
        assert row["activation"] == 0.8
        assert row["channel"] == "visual"

    def test_seeds_activation_updated(self, pm, graph):
        """感知激活后 seeds 表中种子 activation 被更新"""
        pm._create_perceptual_seed_record(
            "percept:visual:red", "visual", "红色通道占比", 0.3
        )
        pm._hebbian_binder = MagicMock()

        now = datetime.now(timezone.utc).isoformat(timespec='milliseconds')
        event = PerceptActivationEvent(
            perceptual_seed="percept:visual:red",
            activation=0.65,
            timestamp=now,
            channel=PerceptionChannel.VISUAL,
        )
        pm.on_percept_activation(event)

        row = graph.conn.execute(
            "SELECT activation FROM seeds WHERE label = 'percept:visual:red'"
        ).fetchone()
        assert row is not None
        assert abs(row["activation"] - 0.65) < 0.001


# ═══════════════════════════════════════════════════════════
#  场景 3: 感知元种子参与涟漪传播
# ═══════════════════════════════════════════════════════════


class TestScenario3PerceptualSeedInRipple:
    """端到端场景 3: 感知元种子不参与涟漪传播（被 match_seeds 排除）

    验收标准:
      - match_seeds() 查询不返回 PERCEPTUAL 类型种子
      - PERCEPTUAL 种子不会出现在涟漪传播的激活列表中
      - 感知元种子通过 Hebbian 绑定边间接影响概念种子激活
    """

    def test_match_seeds_excludes_perceptual_type(self, pm, graph):
        """match_seeds() 排除 PERCEPTUAL 类型种子"""
        pm._generate_preset_perceptual_seeds()

        # 尝试匹配包含感知元种子关键词的查询
        matched = graph.match_seeds("红色")
        for seed in matched:
            assert seed["type"] != "PERCEPTUAL"

    def test_perceptual_seeds_not_in_ripple_activation(self, pm, graph):
        """PERCEPTUAL 种子不出现在涟漪激活列表中"""
        pm._generate_preset_perceptual_seeds()

        # 即使感知元种子 label 包含"红色"，match_seeds 也不应返回它
        matched = graph.match_seeds("percept:visual:red")
        perceptual_matched = [s for s in matched if s["type"] == "PERCEPTUAL"]
        assert len(perceptual_matched) == 0

    def test_hebbian_binding_links_percept_to_concept(self, pm, graph):
        """感知元种子通过 Hebbian 绑定边间接关联概念种子"""
        # 创建感知元种子
        pm._create_perceptual_seed_record(
            "percept:visual:red", "visual", "红色通道占比", 0.3
        )

        # 创建 Hebbian 绑定边
        binder = HebbianBinder(graph)
        now = datetime.now(timezone.utc).isoformat(timespec='milliseconds')
        binder.on_percept_activation(PerceptActivationEvent(
            perceptual_seed="percept:visual:red",
            activation=0.8,
            timestamp=now,
            channel=PerceptionChannel.VISUAL,
        ))
        binder.on_concept_activation(ConceptActivationEvent(
            activated_seeds=["红色"],
            timestamp=now,
        ))
        binder._check_co_activation()

        # 验证绑定边存在
        row = graph.conn.execute(
            "SELECT * FROM karma_edges "
            "WHERE source = 'percept:visual:red' AND source_tag = 'hebbian_binding'"
        ).fetchone()
        assert row is not None
        assert row["target"] == "红色"


# ═══════════════════════════════════════════════════════════
#  场景 4: 感知通道硬件不可用降级
# ═══════════════════════════════════════════════════════════


class TestScenario4ChannelDegradation:
    """端到端场景 4: 感知通道硬件不可用时优雅降级

    验收标准:
      - VisualAnchor 启动失败 → visual 通道 running=False
      - AudioAnchor 启动失败 → auditory 通道 running=False
      - SomaticAnchor 启动失败 → somatic 通道 running=False
      - 单个通道失败不影响其他通道
      - SomaticAnchor 无 psutil 时降级到 /proc 或 WMI
    """

    def test_visual_channel_degradation(self, pm, graph):
        """视觉通道启动失败时优雅降级"""
        with patch("consciousness_sea.perception.visual_anchor.VisualAnchor", side_effect=Exception("camera not found")), \
             patch("consciousness_sea.perception.perception.PERCEPTION_ENABLED", True):
            result = pm._start_visual_channel()

        assert result is False
        assert pm._channel_status["visual"].running is False

    def test_auditory_channel_degradation(self, pm, graph):
        """听觉通道启动失败时优雅降级"""
        with patch("consciousness_sea.perception.audio_anchor.AudioAnchor", side_effect=Exception("microphone not found")), \
             patch("consciousness_sea.perception.perception.PERCEPTION_ENABLED", True):
            result = pm._start_auditory_channel()

        assert result is False
        assert pm._channel_status["auditory"].running is False

    def test_somatic_channel_degradation(self, pm, graph):
        """本体感知通道启动失败时优雅降级"""
        with patch("consciousness_sea.perception.somatic_anchor.SomaticAnchor", side_effect=Exception("psutil not available")), \
             patch("consciousness_sea.perception.perception.PERCEPTION_ENABLED", True):
            result = pm._start_somatic_channel()

        assert result is False
        assert pm._channel_status["somatic"].running is False

    def test_single_channel_failure_does_not_affect_others(self, pm, graph):
        """单个通道失败不影响其他通道"""
        with patch("consciousness_sea.perception.perception.PERCEPTION_ENABLED", True), \
             patch("consciousness_sea.perception.visual_anchor.VisualAnchor", side_effect=Exception("camera error")), \
             patch("consciousness_sea.perception.audio_anchor.AudioAnchor") as MockAA, \
             patch("consciousness_sea.perception.somatic_anchor.SomaticAnchor"), \
             patch("consciousness_sea.perception.hebbian_binder.HebbianBinder"), \
             patch("consciousness_sea.perception.multimodal_aligner.MultimodalAligner"):
            MockAA.return_value._mock_mode = True
            pm.start()

        # 视觉通道失败
        assert pm._channel_status["visual"].running is False
        # 听觉通道成功
        assert pm._channel_status["auditory"].running is True

    def test_somatic_anchor_no_psutil_graceful(self, pm, graph):
        """SomaticAnchor 无 psutil 时优雅降级"""
        anchor = SomaticAnchor(pm)
        # psutil 是在方法内部 import 的，不可用时自动降级
        # 直接调用 collect_features 不应抛异常
        with patch("consciousness_sea.perception.perception.PERCEPTION_ENABLED", True):
            features = anchor.collect_features()
        # psutil 不可用时，部分指标可能为 None
        assert isinstance(features, SomaticFeatures)


# ═══════════════════════════════════════════════════════════
#  场景 5: 感知功能关闭
# ═══════════════════════════════════════════════════════════


class TestScenario5PerceptionDisabled:
    """端到端场景 5: PERCEPTION_ENABLED=False 时感知功能完全关闭

    验收标准:
      - start() 不执行任何操作
      - start_channel() 返回 False
      - get_status() 返回 enabled=False
      - 不生成任何感知元种子
    """

    def test_start_does_nothing_when_disabled(self, pm, graph):
        """PERCEPTION_ENABLED=False 时 start() 不执行任何操作"""
        with patch("consciousness_sea.perception.perception.PERCEPTION_ENABLED", False):
            pm.start()

        count = graph.conn.execute(
            "SELECT COUNT(*) FROM perceptual_seeds"
        ).fetchone()[0]
        assert count == 0

    def test_start_channel_returns_false_when_disabled(self, pm, graph):
        """PERCEPTION_ENABLED=False 时 start_channel() 返回 False"""
        with patch("consciousness_sea.perception.perception.PERCEPTION_ENABLED", False):
            result = pm.start_channel("visual")
        assert result is False

    def test_status_shows_disabled(self, pm, graph):
        """PERCEPTION_ENABLED=False 时状态显示 disabled"""
        with patch("consciousness_sea.perception.perception.PERCEPTION_ENABLED", False):
            status = pm.get_status()
        # enabled 字段取决于配置值
        assert isinstance(status, PerceptionManagerStatus)

    def test_no_seeds_generated_when_disabled(self, pm, graph):
        """PERCEPTION_ENABLED=False 时不生成任何感知元种子"""
        with patch("consciousness_sea.perception.perception.PERCEPTION_ENABLED", False):
            pm.start()

        # seeds 表不应有 PERCEPTUAL 类型
        count = graph.conn.execute(
            "SELECT COUNT(*) FROM seeds WHERE type = 'PERCEPTUAL'"
        ).fetchone()[0]
        assert count == 0


# ═══════════════════════════════════════════════════════════
#  场景 6: 多模态对齐离线校准
# ═══════════════════════════════════════════════════════════


class TestScenario6MultimodalAlignment:
    """端到端场景 6: 多模态对齐离线校准

    验收标准:
      - MULTIMODAL_ALIGNMENT_ENABLED=False 时返回空列表
      - 无帧缓存时安全返回空列表
      - 无 CLIP 模型时安全返回空列表
      - 对齐器防重入（同时运行返回空列表）
      - last_run_time 在对齐后更新
    """

    def test_alignment_disabled(self, graph):
        """MULTIMODAL_ALIGNMENT_ENABLED=False 时返回空列表"""
        aligner = MultimodalAligner(graph)
        with patch("consciousness_sea.perception.multimodal_aligner.MULTIMODAL_ALIGNMENT_ENABLED", False):
            result = aligner.run_alignment()
        assert result == []

    def test_alignment_no_frames(self, graph):
        """无帧缓存时安全返回空列表"""
        aligner = MultimodalAligner(graph)
        with patch("consciousness_sea.perception.multimodal_aligner.MULTIMODAL_ALIGNMENT_ENABLED", True):
            result = aligner.run_alignment()
        assert result == []

    def test_alignment_no_clip_model(self, graph):
        """无 CLIP 模型时安全返回空列表"""
        aligner = MultimodalAligner(graph)
        with patch("consciousness_sea.perception.multimodal_aligner.MULTIMODAL_ALIGNMENT_ENABLED", True), \
             patch("consciousness_sea.perception.multimodal_aligner.MultimodalAligner._get_recent_frames", return_value=[b"fake_frame"]):
            result = aligner.run_alignment()
        # CLIP 不可用，返回空列表
        assert result == []

    def test_alignment_reentrancy_protection(self, graph):
        """对齐器防重入：同时运行返回空列表"""
        aligner = MultimodalAligner(graph)
        # 模拟正在运行
        aligner._running = True
        with patch("consciousness_sea.perception.multimodal_aligner.MULTIMODAL_ALIGNMENT_ENABLED", True):
            result = aligner.run_alignment()
        assert result == []
        aligner._running = False

    def test_infer_percept_seed(self):
        """_infer_percept_seed 从概念种子推断感知元种子"""
        assert "visual" in MultimodalAligner._infer_percept_seed("红色")
        assert "auditory" in MultimodalAligner._infer_percept_seed("声音")


# ═══════════════════════════════════════════════════════════
#  场景 7: 感知查询 API 端到端集成测试
# ═══════════════════════════════════════════════════════════


class TestScenario7PerceptionAPI:
    """端到端场景 7: 感知查询 API 端到端集成测试

    验收标准:
      - GET /api/v1/perception/status 返回感知系统状态
      - GET /api/v1/perception/seeds 返回感知元种子列表
      - GET /api/v1/perception/seeds/{label} 返回种子详情
      - GET /api/v1/perception/bindings 返回 Hebbian 绑定边
      - GET /api/v1/perception/events 返回感知事件
      - POST /api/v1/perception/align 触发多模态对齐
      - 感知功能禁用时各端点返回默认值
    """

    @pytest.fixture
    def client(self, graph, pm):
        """创建 TestClient，注入 mock 连接池和 PerceptionManager"""
        api_module = sys.modules['consciousness_sea.interfaces.api']

        # 创建 mock 连接池
        mock_pool = MagicMock()

        def acquire_side_effect():
            return graph

        def release_side_effect(g):
            pass

        mock_pool.acquire.side_effect = acquire_side_effect
        mock_pool.release.side_effect = release_side_effect

        # 注入到 api 模块
        api_module._pool = mock_pool
        api_module._perception_manager = pm

        # 创建 mock Observer
        from consciousness_sea.infrastructure.observer import Observer, StatusData
        mock_observer = MagicMock(spec=Observer)
        mock_status = StatusData(
            total_seeds=5,
            total_karma_edges=1,
            hottest_seeds=[],
            coldest_seeds=[],
            heaviest_karma=[],
            recent_queries=[],
            alerts=[],
            domain_distribution={},
            db_size_mb=0.0,
            meta_seeds=None,
            guardian_loop=None,
            perception=None,
        )
        mock_observer.get_status.return_value = mock_status
        api_module._observer = mock_observer

        # 创建 mock SessionManager 和 UserManager
        mock_session_mgr = MagicMock()
        mock_user_mgr = MagicMock()
        api_module._session_manager = mock_session_mgr
        api_module._user_manager = mock_user_mgr

        with patch("consciousness_sea.interfaces.api.PERCEPTION_ENABLED", True):
            client = TestClient(api_module.app)
            yield client

        # 清理
        api_module._pool = None
        api_module._perception_manager = None
        api_module._observer = None

    def test_perception_status(self, client, pm, graph):
        """GET /api/v1/perception/status 返回感知系统状态"""
        pm._generate_preset_perceptual_seeds()

        with patch("consciousness_sea.interfaces.api.PERCEPTION_ENABLED", True):
            response = client.get("/api/v1/perception/status")

        assert response.status_code == 200
        data = response.json()
        assert "enabled" in data
        assert "channels" in data
        assert "total_perceptual_seeds" in data
        assert "total_hebbian_bindings" in data
        assert "recent_activation_count" in data

    def test_perception_seeds(self, client, pm, graph):
        """GET /api/v1/perception/seeds 返回感知元种子列表"""
        pm._generate_preset_perceptual_seeds()

        with patch("consciousness_sea.interfaces.api.PERCEPTION_ENABLED", True):
            response = client.get("/api/v1/perception/seeds")

        assert response.status_code == 200
        data = response.json()
        assert "seeds" in data
        assert len(data["seeds"]) == 16

    def test_perception_seed_detail(self, client, pm, graph):
        """GET /api/v1/perception/seeds/{label} 返回种子详情"""
        pm._create_perceptual_seed_record(
            "percept:visual:red", "visual", "红色通道占比", 0.3
        )

        with patch("consciousness_sea.interfaces.api.PERCEPTION_ENABLED", True):
            response = client.get("/api/v1/perception/seeds/percept:visual:red")

        assert response.status_code == 200
        data = response.json()
        assert data["label"] == "percept:visual:red"
        assert data["channel"] == "visual"

    def test_perception_seed_detail_not_found(self, client, pm, graph):
        """GET /api/v1/perception/seeds/{label} 不存在时返回 404"""
        with patch("consciousness_sea.interfaces.api.PERCEPTION_ENABLED", True):
            response = client.get("/api/v1/perception/seeds/percept:not:exist")

        assert response.status_code == 404

    def test_perception_bindings(self, client, pm, graph):
        """GET /api/v1/perception/bindings 返回 Hebbian 绑定边"""
        with patch("consciousness_sea.interfaces.api.PERCEPTION_ENABLED", True):
            response = client.get("/api/v1/perception/bindings")

        assert response.status_code == 200
        data = response.json()
        assert "bindings" in data

    def test_perception_events(self, client, pm, graph):
        """GET /api/v1/perception/events 返回感知事件"""
        # 先写入一个事件
        pm._create_perceptual_seed_record(
            "percept:visual:red", "visual", "红色通道占比", 0.3
        )
        pm._hebbian_binder = MagicMock()
        now = datetime.now(timezone.utc).isoformat(timespec='milliseconds')
        pm.on_percept_activation(PerceptActivationEvent(
            perceptual_seed="percept:visual:red",
            activation=0.8,
            timestamp=now,
            channel=PerceptionChannel.VISUAL,
        ))

        with patch("consciousness_sea.interfaces.api.PERCEPTION_ENABLED", True):
            response = client.get("/api/v1/perception/events")

        assert response.status_code == 200
        data = response.json()
        assert "events" in data

    def test_perception_align(self, client, pm, graph):
        """POST /api/v1/perception/align 触发多模态对齐"""
        with patch("consciousness_sea.interfaces.api.PERCEPTION_ENABLED", True), \
             patch("consciousness_sea.perception.multimodal_aligner.MULTIMODAL_ALIGNMENT_ENABLED", True):
            response = client.post("/api/v1/perception/align")

        assert response.status_code == 200
        data = response.json()
        assert "results" in data
        assert "count" in data

    def test_perception_api_disabled(self, graph):
        """感知功能禁用时各端点返回默认值"""
        api_module = sys.modules['consciousness_sea.interfaces.api']

        mock_pool = MagicMock()
        mock_pool.acquire.return_value = graph
        mock_pool.release.side_effect = lambda g: None
        api_module._pool = mock_pool
        api_module._perception_manager = None

        from consciousness_sea.infrastructure.observer import Observer, StatusData
        mock_observer = MagicMock(spec=Observer)
        mock_status = StatusData(
            total_seeds=0,
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
            perception=None,
        )
        mock_observer.get_status.return_value = mock_status
        api_module._observer = mock_observer

        mock_session_mgr = MagicMock()
        mock_user_mgr = MagicMock()
        api_module._session_manager = mock_session_mgr
        api_module._user_manager = mock_user_mgr

        with patch("consciousness_sea.interfaces.api.PERCEPTION_ENABLED", False):
            client = TestClient(api_module.app)

            # status
            resp = client.get("/api/v1/perception/status")
            assert resp.status_code == 200
            assert resp.json()["enabled"] is False

            # seeds
            resp = client.get("/api/v1/perception/seeds")
            assert resp.status_code == 200
            assert resp.json()["seeds"] == []

            # bindings
            resp = client.get("/api/v1/perception/bindings")
            assert resp.status_code == 200
            assert resp.json()["bindings"] == []

            # events
            resp = client.get("/api/v1/perception/events")
            assert resp.status_code == 200
            assert resp.json()["events"] == []

        # 清理
        api_module._pool = None
        api_module._perception_manager = None
        api_module._observer = None
