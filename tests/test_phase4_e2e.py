"""
Phase 4 端到端场景验收测试

覆盖 7 个验收场景（参考 spec.md）：
- 场景1：元种子自动生成
- 场景2：守护循环健康检查
- 场景3：元业力边自然形成
- 场景4：自边界追踪
- 场景5：涟漪传播排除元种子
- 场景6：API 查询元种子
- 场景7：元种子休眠与退役
"""

from __future__ import annotations

import json
import sqlite3
import sys
from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest

# 确保 backend/src 在 sys.path 中
_src = str(Path(__file__).resolve().parent.parent / "backend" / "src")
if _src not in sys.path:
    sys.path.insert(0, _src)
# 同时保留项目根目录（tests 中部分 import 依赖它）
_root = str(Path(__file__).resolve().parent.parent)
if _root not in sys.path:
    sys.path.insert(0, _root)

from consciousness_sea.metacognition.meta_seed import (
    MetaSeedManager,
    MetaSeedCategory,
    MetaSeedStatus,
    DOMAIN_MONITOR_DEFAULT_METRICS,
    RELATION_QUALITY_DEFAULT_METRICS,
    SYSTEM_MONITOR_DEFAULT_METRICS,
    SELF_BOUNDARY_DEFAULT_METRICS,
    SYSTEM_META_SEEDS,
)
from consciousness_sea.metacognition.guardian_loop import GuardianLoop, GuardianLoopResult
from consciousness_sea.domain.graph_db import GraphDB
from consciousness_sea.domain.router import route
from consciousness_sea.infrastructure.config import (
    META_SEED_ENABLED,
    META_KARMA_DELTA_THRESHOLD,
    META_SEED_DORMANT_CYCLES,
    CONFIDENCE_LOW,
)


# ═══════════════════════════════════════════════════════════
#  Fixtures
# ═══════════════════════════════════════════════════════════


def _build_e2e_db() -> sqlite3.Connection:
    """创建端到端测试用内存数据库（含所有 Phase 4 表）"""
    # check_same_thread=False: TestClient 在不同线程中执行请求，
    # 必须允许跨线程使用 SQLite 连接
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
        ("姜汤", "姜汤", "CONCEPT", "[]", "常识", "ginger soup"),
        ("量子力学", "量子力学", "CONCEPT", "[]", "物理", "quantum mechanics"),
        ("薛定谔方程", "薛定谔方程", "CONCEPT", "[]", "物理", "Schrodinger equation"),
        ("人工智能", "人工智能", "CONCEPT", "[\"AI\"]", "计算机", "AI"),
        ("深度学习", "深度学习", "CONCEPT", "[]", "计算机", "deep learning"),
        ("水", "水", "CONCEPT", "[]", "常识", "water"),
    ]
    conn.executemany(
        "INSERT INTO seeds (id, label, type, aliases, domain, definition) VALUES (?, ?, ?, ?, ?, ?)",
        seeds,
    )

    # 插入测试业力边
    edges = [
        ("感冒", "发热", "COOCCURS_WITH", 0.95, "karma_delta"),
        ("感冒", "咳嗽", "RELATED", 0.60, "karma_delta"),
        ("感冒", "维C", "RELATED", 0.60, "karma_delta"),
        ("量子力学", "薛定谔方程", "IS_A", 0.88, "karma_delta"),
        ("人工智能", "深度学习", "IS_A", 0.90, "karma_delta"),
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
    """创建端到端测试 GraphDB 实例"""
    conn = _build_e2e_db()
    g = _make_graph_db(conn)
    yield g
    g.close()


# ═══════════════════════════════════════════════════════════
#  场景1：元种子自动生成
# ═══════════════════════════════════════════════════════════


class TestScenario1AutoGeneration:
    """场景1：元种子自动生成——系统启动后约 50 个元种子"""

    def test_meta_seeds_auto_generated(self, graph):
        """守护循环执行后元种子自动生成，seeds 表和 meta_seeds 表均有记录"""
        guardian = GuardianLoop(graph)
        result = guardian.execute_once()
        assert result.success is True

        # 验证 meta_seeds 表有记录
        count = graph.conn.execute("SELECT COUNT(*) FROM meta_seeds").fetchone()[0]
        # 4 domain(医学/营养/物理/计算机) + 3 relation(COOCCURS_WITH/RELATED/IS_A) + 5 system + 1 self_boundary = 13
        assert count >= 13

        # 验证 seeds 表有 META 记录
        meta_count = graph.conn.execute(
            "SELECT COUNT(*) FROM seeds WHERE type = 'META'"
        ).fetchone()[0]
        assert meta_count >= 13

        # 验证 META 种子的 domain 为 '元认知'
        row = graph.conn.execute(
            "SELECT domain FROM seeds WHERE label = 'meta:医学'"
        ).fetchone()
        assert row["domain"] == "元认知"

        # 验证 META 种子的 activation 为 0.0
        row = graph.conn.execute(
            "SELECT activation FROM seeds WHERE label = 'meta:医学'"
        ).fetchone()
        assert row["activation"] == 0.0

    def test_all_categories_generated(self, graph):
        """所有类别的元种子都被生成"""
        mgr = MetaSeedManager(graph)
        mgr.generate_domain_monitors()
        mgr.generate_relation_monitors()
        mgr.generate_system_monitors()
        mgr._create_meta_seed_record(
            "meta:unknown", MetaSeedCategory.SELF_BOUNDARY,
            dict(SELF_BOUNDARY_DEFAULT_METRICS),
        )

        categories = set()
        for ms in mgr.list_meta_seeds():
            categories.add(ms.category)

        assert MetaSeedCategory.DOMAIN_MONITOR in categories
        assert MetaSeedCategory.RELATION_QUALITY in categories
        assert MetaSeedCategory.SYSTEM_MONITOR in categories
        assert MetaSeedCategory.SELF_BOUNDARY in categories


# ═══════════════════════════════════════════════════════════
#  场景2：守护循环健康检查
# ═══════════════════════════════════════════════════════════


class TestScenario2GuardianHealthCheck:
    """场景2：守护循环健康检查——领域/关系/系统指标更新"""

    def test_domain_health_updated(self, graph):
        """守护循环更新领域监控元种子指标"""
        guardian = GuardianLoop(graph)
        guardian.execute_once()

        mgr = MetaSeedManager(graph)
        ms = mgr.get_meta_seed("meta:医学")
        assert ms is not None
        # 指标应已被更新（不再是初始值）
        assert "avg_karma_density" in ms.metrics
        assert "ripple_success_rate" in ms.metrics

    def test_relation_quality_updated(self, graph):
        """守护循环更新关系质量元种子指标"""
        guardian = GuardianLoop(graph)
        guardian.execute_once()

        mgr = MetaSeedManager(graph)
        ms = mgr.get_meta_seed("meta:RELATED")
        assert ms is not None
        assert "avg_weight" in ms.metrics

    def test_system_metrics_updated(self, graph):
        """守护循环更新系统级元种子指标"""
        guardian = GuardianLoop(graph)
        guardian.execute_once()

        mgr = MetaSeedManager(graph)
        ms = mgr.get_meta_seed("meta:system_total_nodes")
        assert ms is not None
        assert ms.metrics["value"] >= 0


# ═══════════════════════════════════════════════════════════
#  场景3：元业力边自然形成
# ═══════════════════════════════════════════════════════════


class TestScenario3MetaKarmaFormation:
    """场景3：元业力边自然形成——conflict_frequency 变化量 ≥ 阈值时创建元业力边

    注意：check_and_create_meta_karma() 内部通过 list_meta_seeds() 获取元种子列表，
    每次调用创建新的 MetaSeedData 对象，_previous_metrics 为空。
    第一次调用时保存当前指标作为基线并跳过，第二次调用时才能比较变化。
    因此测试需要调用两次：第一次设置基线，第二次检测变化。
    """

    def test_meta_karma_naturally_formed(self, graph):
        """量子力学领域低置信度 → conflict_frequency 变化 → 元业力边"""
        mgr = MetaSeedManager(graph)

        # 创建元种子（初始指标为 0）
        mgr.generate_domain_monitors()

        # 第一次调用：设置基线
        mgr.check_and_create_meta_karma()

        # 模拟 conflict_frequency 增加（量子力学领域遇到低置信度）
        mgr.update_metrics("meta:物理", {
            "avg_karma_density": 0.0, "ripple_success_rate": 0.0,
            "conflict_frequency": META_KARMA_DELTA_THRESHOLD + 3,
        })

        # 第二次调用：检测变化并创建元业力边
        edges_created = mgr.check_and_create_meta_karma()
        assert edges_created >= 1

        # 验证元业力边
        row = graph.conn.execute(
            "SELECT * FROM karma_edges WHERE source = 'meta:物理' AND source_tag = 'meta_karma'"
        ).fetchone()
        assert row is not None
        assert row["relation"] == "META_CORRELATED"


# ═══════════════════════════════════════════════════════════
#  场景4：自边界追踪
# ═══════════════════════════════════════════════════════════


class TestScenario4SelfBoundary:
    """场景4：自边界追踪——meta:unknown 的 unmatched_keywords 从 candidate_seeds 更新"""

    def test_self_boundary_tracks_unknown(self, graph):
        """meta:unknown 的 unmatched_keywords 从 candidate_seeds 更新"""
        mgr = MetaSeedManager(graph)
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
        graph.conn.commit()

        # 更新自边界
        updated = mgr.update_self_boundary()
        assert updated == 1

        ms = mgr.get_meta_seed("meta:unknown")
        assert ms is not None
        assert ms.metrics["unmatched_count"] == 2
        assert "DeepSeek" in ms.metrics["unmatched_keywords"]
        assert "Transformer" in ms.metrics["unmatched_keywords"]


# ═══════════════════════════════════════════════════════════
#  场景5：涟漪传播排除元种子
# ═══════════════════════════════════════════════════════════


class TestScenario5RippleExclusion:
    """场景5：涟漪传播排除元种子——查询"物理"不涉及"meta:物理" """

    def test_match_seeds_excludes_meta(self, graph):
        """match_seeds 不匹配 META 类型种子"""
        # 创建元种子
        mgr = MetaSeedManager(graph)
        mgr.generate_domain_monitors()

        # 验证 meta:物理 存在于 seeds 表
        row = graph.conn.execute(
            "SELECT * FROM seeds WHERE label = 'meta:物理'"
        ).fetchone()
        assert row is not None
        assert row["type"] == "META"

        # match_seeds 不应匹配到 meta:物理
        matched = graph.match_seeds("物理")
        labels = [s["label"] for s in matched]
        assert "meta:物理" not in labels
        # 但应匹配到 "量子力学"（物理领域的种子）
        # 注意：match_seeds 使用分词器，"物理"可能匹配到 domain=物理 的种子

    def test_outgoing_edges_excludes_meta(self, graph):
        """outgoing_edges 排除 meta: 前缀的源"""
        # 创建元种子和元业力边
        mgr = MetaSeedManager(graph)
        mgr.generate_domain_monitors()

        graph.conn.execute(
            "INSERT INTO karma_edges (source, target, relation, weight, source_tag) "
            "VALUES ('meta:物理', 'meta:医学', 'META_CORRELATED', 0.05, 'meta_karma')"
        )
        graph.conn.commit()

        # exclude_meta=True（默认）时返回空
        edges = graph.outgoing_edges("meta:物理", exclude_meta=True)
        assert edges == []

        # exclude_meta=False 时返回元业力边
        edges = graph.outgoing_edges("meta:物理", exclude_meta=False)
        assert len(edges) >= 1

    def test_ripple_does_not_reach_meta(self, graph):
        """涟漪传播不涉及元种子"""
        # 创建元种子
        mgr = MetaSeedManager(graph)
        mgr.generate_domain_monitors()

        # 创建元业力边
        graph.conn.execute(
            "INSERT INTO karma_edges (source, target, relation, weight, source_tag) "
            "VALUES ('meta:物理', 'meta:医学', 'META_CORRELATED', 0.05, 'meta_karma')"
        )
        graph.conn.commit()

        # 执行路由
        result = route("感冒", graph)

        # 涟漪传播不应涉及 meta: 前缀的种子
        for label in result.activated:
            assert not label.startswith("meta:")


# ═══════════════════════════════════════════════════════════
#  场景6：API 查询元种子
# ═══════════════════════════════════════════════════════════


class TestScenario6APIQuery:
    """场景6：API 查询元种子——所有端点返回正确格式"""

    def test_list_meta_seeds_api(self, graph):
        """GET /api/v1/meta-seeds 返回正确格式"""
        from fastapi.testclient import TestClient
        import consciousness_sea.interfaces.api as api
        api_module = sys.modules['consciousness_sea.interfaces.api']

        # 创建 mock 连接池
        mock_pool = MagicMock()
        mock_pool.acquire.return_value = graph
        mock_pool.release.return_value = None

        guardian_loop = GuardianLoop(graph)

        api_module._pool = mock_pool
        api_module._guardian_loop = guardian_loop

        # 创建 mock Observer
        from consciousness_sea.infrastructure.observer import Observer, StatusData
        mock_observer = MagicMock(spec=Observer)
        mock_status = StatusData(
            total_seeds=10, total_karma_edges=5,
            hottest_seeds=[], coldest_seeds=[], heaviest_karma=[],
            recent_queries=[], alerts=[], domain_distribution={},
        )
        mock_observer.get_status.return_value = mock_status
        api_module._observer = mock_observer

        # 先生成元种子
        mgr = MetaSeedManager(graph)
        mgr.generate_domain_monitors()
        mgr.generate_system_monitors()

        with patch("consciousness_sea.interfaces.api.META_SEED_ENABLED", True):
            client = TestClient(api_module.app)
            response = client.get("/api/v1/meta-seeds")

        assert response.status_code == 200
        data = response.json()
        assert "meta_seeds" in data
        assert len(data["meta_seeds"]) > 0

        # 验证字段格式
        for ms in data["meta_seeds"]:
            assert "label" in ms
            assert "category" in ms
            assert "status" in ms
            assert "metrics" in ms
            assert "updated_at" in ms

        api_module._pool = None
        api_module._guardian_loop = None

    def test_guardian_status_api(self, graph):
        """GET /api/v1/guardian/status 返回守护循环状态"""
        from fastapi.testclient import TestClient
        import consciousness_sea.interfaces.api as api
        api_module = sys.modules['consciousness_sea.interfaces.api']

        mock_pool = MagicMock()
        mock_pool.acquire.return_value = graph
        mock_pool.release.return_value = None

        guardian_loop = GuardianLoop(graph)
        api_module._pool = mock_pool
        api_module._guardian_loop = guardian_loop

        with patch("consciousness_sea.interfaces.api.META_SEED_ENABLED", True):
            client = TestClient(api_module.app)
            response = client.get("/api/v1/guardian/status")

        assert response.status_code == 200
        data = response.json()
        assert "is_running" in data
        assert "interval_seconds" in data

        api_module._pool = None
        api_module._guardian_loop = None

    def test_trigger_guardian_api(self, graph):
        """POST /api/v1/guardian/trigger 立即执行"""
        from fastapi.testclient import TestClient
        import consciousness_sea.interfaces.api as api
        api_module = sys.modules['consciousness_sea.interfaces.api']

        mock_pool = MagicMock()
        mock_pool.acquire.return_value = graph
        mock_pool.release.return_value = None

        guardian_loop = GuardianLoop(graph)
        api_module._pool = mock_pool
        api_module._guardian_loop = guardian_loop

        with patch("consciousness_sea.interfaces.api.META_SEED_ENABLED", True):
            client = TestClient(api_module.app)
            response = client.post("/api/v1/guardian/trigger")

        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "success"
        assert "meta_seeds_updated" in data

        api_module._pool = None
        api_module._guardian_loop = None


# ═══════════════════════════════════════════════════════════
#  场景7：元种子休眠与退役
# ═══════════════════════════════════════════════════════════


class TestScenario7DormantAndRetired:
    """场景7：元种子休眠与退役——连续无变化 → dormant → retired

    注意：_unchanged_cycles 和 _previous_metrics 已持久化到数据库，
    因此多次调用 check_dormant_status() 可以正确累积无变化周期数。
    """

    def test_dormant_lifecycle(self, graph):
        """完整休眠/退役生命周期"""
        mgr = MetaSeedManager(graph)
        # 使用空 metrics 创建
        mgr._create_meta_seed_record(
            "meta:医学", MetaSeedCategory.DOMAIN_MONITOR,
            {},
        )

        # 阶段1：连续无变化 → dormant
        for _ in range(META_SEED_DORMANT_CYCLES):
            mgr.check_dormant_status()

        ms = mgr.get_meta_seed("meta:医学")
        assert ms is not None
        assert ms.status == MetaSeedStatus.DORMANT

        # 阶段2：指标变化 → 恢复 active
        mgr.update_metrics("meta:医学", {
            "avg_karma_density": 0.5, "ripple_success_rate": 0.0, "conflict_frequency": 0
        })
        mgr.check_dormant_status()

        ms = mgr.get_meta_seed("meta:医学")
        assert ms.status == MetaSeedStatus.ACTIVE

        # 阶段3：再次连续无变化 → dormant → 继续无变化 → retired
        # 不再更新 metrics，使 metrics_changed=False，_unchanged_cycles 自然累积
        for _ in range(META_SEED_DORMANT_CYCLES):
            mgr.check_dormant_status()

        ms = mgr.get_meta_seed("meta:医学")
        assert ms.status == MetaSeedStatus.DORMANT

        # 继续无变化直到 retired
        for _ in range(META_SEED_DORMANT_CYCLES * 3):
            mgr.check_dormant_status()

        ms = mgr.get_meta_seed("meta:医学")
        assert ms.status == MetaSeedStatus.RETIRED

    def test_functional_degradation(self, graph):
        """功能降级兼容性：META_SEED_ENABLED=False 时行为与 Phase 3 一致"""
        with patch("consciousness_sea.metacognition.meta_seed.META_SEED_ENABLED", False):
            mgr = MetaSeedManager(graph)

            # 所有生成方法返回 0
            assert mgr.generate_domain_monitors() == 0
            assert mgr.generate_relation_monitors() == 0
            assert mgr.generate_system_monitors() == 0
            assert mgr.update_self_boundary() == 0
            assert mgr.detect_unknown_domains() == 0

            # 查询方法返回空
            assert mgr.get_meta_seed("meta:医学") is None
            assert mgr.list_meta_seeds() == []

            # 更新方法返回 False
            assert mgr.update_metrics("meta:医学", {}) is False
            assert mgr.increment_metric("meta:医学", "conflict_frequency") is False

            # 元业力边和休眠判定返回 0
            assert mgr.check_and_create_meta_karma() == 0
            assert mgr.check_dormant_status() == 0

        # 验证没有创建任何元种子
        count = graph.conn.execute("SELECT COUNT(*) FROM meta_seeds").fetchone()[0]
        assert count == 0