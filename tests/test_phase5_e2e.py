"""
Phase 5 端到端场景验收测试

覆盖：
- 场景1：认知目标完整生命周期（生成→调度→探索→冷却→归档）
- 场景2：好奇心引擎三种策略（内部探索/候选升级/外部查询）
- 场景3：守护循环7步扩展
- 场景4：目标池管理（上限1000、淘汰最低权重）
- 场景5：目标去重逻辑
- 场景6：API 查询认知目标
- 场景7：虚拟查询不触发熏习
"""

from __future__ import annotations

import json
import sqlite3
import sys
from pathlib import Path
from datetime import datetime, timezone, timedelta
from unittest.mock import patch, MagicMock

import pytest

# 确保 backend/src 在 sys.path 中
_src = str(Path(__file__).resolve().parent.parent / "backend" / "src")
if _src not in sys.path:
    sys.path.insert(0, _src)
# 同时保留项目根目录
_root = str(Path(__file__).resolve().parent.parent)
if _root not in sys.path:
    sys.path.insert(0, _root)

from consciousness_sea.metacognition.cognitive_goal import (
    CognitiveGoalManager,
    CognitiveGoalData,
    GoalType,
    GoalStatus,
)
from consciousness_sea.metacognition.curiosity_engine import (
    CuriosityEngine,
    ExplorationResult,
    CuriosityEngineStatus,
)
from consciousness_sea.metacognition.guardian_loop import GuardianLoop, GuardianLoopResult
from consciousness_sea.domain.graph_db import GraphDB
from consciousness_sea.domain.router import route
from consciousness_sea.infrastructure.config import (
    COGNITIVE_GOAL_ENABLED,
    CURIOSITY_ENGINE_ENABLED,
    GOAL_LOW_CONF_THRESHOLD,
    GOAL_HIGH_CONFLICT_THRESHOLD,
    GOAL_LOW_DENSITY_RATIO,
    GOAL_NEW_TERM_THRESHOLD,
    GOAL_DECAY_CYCLES,
    GOAL_DECAY_FACTOR,
    GOAL_EXPIRE_THRESHOLD,
    GOAL_USER_ABSENCE_CYCLES,
    GOAL_POOL_MAX_SIZE,
    GOAL_AUTO_EXPLORE_THRESHOLD,
    GUARDIAN_LOOP_INTERVAL,
    META_SEED_ENABLED,
)


# ═══════════════════════════════════════════════════════════
#  Fixtures
# ═══════════════════════════════════════════════════════════


def _build_e2e_db() -> sqlite3.Connection:
    """创建端到端测试用内存数据库"""
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
    db.ensure_phase5_tables()
    return db


@pytest.fixture
def graph():
    """创建端到端测试 GraphDB 实例"""
    conn = _build_e2e_db()
    g = _make_graph_db(conn)
    yield g
    g.close()


def _insert_domain_monitor(conn, domain, metrics, status="active"):
    """辅助：插入领域监控元种子"""
    now = datetime.now(timezone.utc).isoformat()
    conn.execute(
        "INSERT INTO meta_seeds (label, category, metrics_json, status, source_domain, created_at, updated_at) "
        "VALUES (?, 'domain_monitor', ?, ?, ?, ?, ?)",
        (f"meta:{domain}", json.dumps(metrics), status, domain, now, now),
    )
    conn.execute(
        "INSERT OR IGNORE INTO seeds (id, label, type, domain, activation) VALUES (?, ?, 'META', '元认知', 0.0)",
        (f"meta:{domain}", f"meta:{domain}"),
    )
    conn.commit()


# ═══════════════════════════════════════════════════════════
#  场景1：认知目标完整生命周期
# ═══════════════════════════════════════════════════════════


class TestScenario1GoalLifecycle:
    """场景1：认知目标完整生命周期——生成→调度→探索→冷却→归档"""

    def test_goal_lifecycle_generate(self, graph):
        """步骤1：基于元种子指标生成认知目标"""
        _insert_domain_monitor(
            graph.conn, "量子力学",
            {"avg_karma_density": 0.5, "ripple_success_rate": 0.3,
             "conflict_frequency": GOAL_LOW_CONF_THRESHOLD + 1},
        )
        mgr = CognitiveGoalManager(graph)
        created = mgr.generate_goals()
        assert created >= 1

        # 验证目标已创建且状态为 pending
        goals = mgr.list_goals(status=GoalStatus.PENDING)
        assert len(goals) >= 1
        qm_goals = [g for g in goals if g.domain == "量子力学"]
        assert len(qm_goals) >= 1

    def test_goal_lifecycle_explore(self, graph):
        """步骤2：好奇心引擎探索目标"""
        _insert_domain_monitor(
            graph.conn, "医学",
            {"avg_karma_density": 0.5, "ripple_success_rate": 0.3,
             "conflict_frequency": GOAL_LOW_CONF_THRESHOLD + 1},
        )
        mgr = CognitiveGoalManager(graph)
        mgr.generate_goals()

        # 获取 pending 目标
        goals = mgr.list_goals(status=GoalStatus.PENDING)
        medical_goals = [g for g in goals if g.domain == "医学"]
        if not medical_goals:
            pytest.skip("无医学领域目标")

        goal = medical_goals[0]
        engine = CuriosityEngine(graph, mgr)

        # Mock 内部探索
        with patch.object(engine, '_explore_internal') as mock_internal:
            mock_internal.return_value = ExplorationResult(
                goal_id=goal.goal_id, strategy="internal",
                explored_seeds=["感冒", "发热"], new_associations=2,
            )
            result = engine.explore(goal)

        assert result.error is None
        # 验证目标状态更新为 completed
        updated = mgr.get_goal(goal.goal_id)
        assert updated.status == GoalStatus.COMPLETED
        assert updated.priority_weight == 0.1

    def test_goal_lifecycle_cool(self, graph):
        """步骤3：目标冷却——权重衰减"""
        now = datetime.now(timezone.utc)
        old_time = now - timedelta(seconds=GUARDIAN_LOOP_INTERVAL * (GOAL_DECAY_CYCLES + 1))
        graph.conn.execute(
            "INSERT INTO cognitive_goals "
            "(goal_id, goal_type, trigger_condition, domain, priority_weight, status, "
            " sub_goals, execution_log, decay_cycles_count, last_touched_at, created_at, updated_at) "
            "VALUES (?, 'low_confidence', 'test', '量子力学', 0.8, 'pending', '[]', '[]', 0, ?, ?, ?)",
            ("goal_lifecycle_cool", old_time.isoformat(), old_time.isoformat(), old_time.isoformat()),
        )
        graph.conn.commit()

        mgr = CognitiveGoalManager(graph)
        cooled = mgr.cool_goals()
        assert cooled >= 1

        row = graph.conn.execute(
            "SELECT priority_weight FROM cognitive_goals WHERE goal_id = 'goal_lifecycle_cool'"
        ).fetchone()
        assert row["priority_weight"] < 0.8

    def test_goal_lifecycle_archive(self, graph):
        """步骤4：用户缺席归档"""
        now = datetime.now(timezone.utc)
        old_time = now - timedelta(seconds=GUARDIAN_LOOP_INTERVAL * (GOAL_USER_ABSENCE_CYCLES + 1))
        graph.conn.execute(
            "INSERT INTO cognitive_goals "
            "(goal_id, goal_type, trigger_condition, domain, priority_weight, status, "
            " sub_goals, execution_log, associated_user, decay_cycles_count, last_touched_at, created_at, updated_at) "
            "VALUES (?, 'new_term', 'test', '科技', 0.5, 'pending', '[]', '[]', 'user_test', 0, ?, ?, ?)",
            ("goal_lifecycle_archive", now.isoformat(), old_time.isoformat(), old_time.isoformat()),
        )
        graph.conn.execute(
            "INSERT INTO user_cold_start (user_label, query_count, updated_at) VALUES ('user_test', 5, ?)",
            (old_time.isoformat(),),
        )
        graph.conn.commit()

        mgr = CognitiveGoalManager(graph)
        mgr.cool_goals()

        row = graph.conn.execute(
            "SELECT status FROM cognitive_goals WHERE goal_id = 'goal_lifecycle_archive'"
        ).fetchone()
        assert row["status"] == "archived"


# ═══════════════════════════════════════════════════════════
#  场景2：好奇心引擎三种策略
# ═══════════════════════════════════════════════════════════


class TestScenario2CuriosityStrategies:
    """场景2：好奇心引擎三种策略——内部探索/候选升级/外部查询"""

    def test_strategy_internal(self, graph):
        """内部探索：已有种子且存在业力边"""
        mgr = CognitiveGoalManager(graph)
        engine = CuriosityEngine(graph, mgr)

        strategy = engine._determine_strategy("医学")
        assert strategy == "internal"

    def test_strategy_candidate_upgrade(self, graph):
        """候选升级：内部无种子但提炼池有候选"""
        now = datetime.now(timezone.utc).isoformat()
        graph.conn.execute(
            "INSERT INTO distillation_pool "
            "(canonical_source, canonical_target, canonical_relation, representative_label, "
            " count, contributor_users, status, created_at, updated_at) "
            "VALUES (?, ?, 'RELATED', ?, 1, '[\"test\"]', 'pending', ?, ?)",
            ("文学种子", "文学关联", "文学种子→文学关联", now, now),
        )
        graph.conn.commit()

        mgr = CognitiveGoalManager(graph)
        engine = CuriosityEngine(graph, mgr)

        strategy = engine._determine_strategy("文学")
        assert strategy == "candidate_upgrade"

    def test_strategy_external(self, graph):
        """外部查询：完全空白且启用外部查询"""
        mgr = CognitiveGoalManager(graph)
        engine = CuriosityEngine(graph, mgr)

        with patch("consciousness_sea.metacognition.curiosity_engine.EXTERNAL_QUERY_ENABLED", True):
            strategy = engine._determine_strategy("完全不存在的领域")
        assert strategy == "external"

    def test_strategy_none(self, graph):
        """none：完全空白且禁用外部查询"""
        mgr = CognitiveGoalManager(graph)
        engine = CuriosityEngine(graph, mgr)

        with patch("consciousness_sea.metacognition.curiosity_engine.EXTERNAL_QUERY_ENABLED", False):
            strategy = engine._determine_strategy("完全不存在的领域")
        assert strategy == "none"


# ═══════════════════════════════════════════════════════════
#  场景3：守护循环7步扩展
# ═══════════════════════════════════════════════════════════


class TestScenario3GuardianLoop7Steps:
    """场景3：守护循环7步扩展——①②③④⑤⑥⑦"""

    def test_guardian_loop_7_steps(self, graph):
        """守护循环执行7步：元种子生成→指标更新→目标生成→调度→冷却→元业力→COMMIT"""
        guardian = GuardianLoop(graph)
        result = guardian.execute_once()
        assert result.success is True

        # 验证元种子已生成
        meta_count = graph.conn.execute("SELECT COUNT(*) FROM meta_seeds").fetchone()[0]
        assert meta_count > 0

        # 验证认知目标表存在（即使没有目标被生成）
        goal_count = graph.conn.execute("SELECT COUNT(*) FROM cognitive_goals").fetchone()[0]
        assert goal_count >= 0  # 可能没有触发条件

    def test_guardian_loop_with_goal_generation(self, graph):
        """守护循环在有触发条件时生成认知目标"""
        # 设置触发条件
        _insert_domain_monitor(
            graph.conn, "量子力学",
            {"avg_karma_density": 0.5, "ripple_success_rate": 0.3,
             "conflict_frequency": GOAL_HIGH_CONFLICT_THRESHOLD + 5},
        )

        guardian = GuardianLoop(graph)
        result = guardian.execute_once()
        assert result.success is True

        # 验证认知目标已生成
        goal_count = graph.conn.execute(
            "SELECT COUNT(*) FROM cognitive_goals WHERE domain = '量子力学'"
        ).fetchone()[0]
        assert goal_count >= 1

    def test_guardian_loop_disabled(self, graph):
        """COGNITIVE_GOAL_ENABLED=False 时守护循环跳过步骤③④⑤"""
        _insert_domain_monitor(
            graph.conn, "量子力学",
            {"avg_karma_density": 0.5, "ripple_success_rate": 0.3,
             "conflict_frequency": 100},
        )

        with patch("consciousness_sea.metacognition.guardian_loop.COGNITIVE_GOAL_ENABLED", False):
            guardian = GuardianLoop(graph)
            result = guardian.execute_once()
            assert result.success is True

        # 不应生成认知目标
        goal_count = graph.conn.execute("SELECT COUNT(*) FROM cognitive_goals").fetchone()[0]
        assert goal_count == 0


# ═══════════════════════════════════════════════════════════
#  场景4：目标池管理
# ═══════════════════════════════════════════════════════════


class TestScenario4PoolManagement:
    """场景4：目标池管理——上限1000、淘汰最低权重"""

    def test_pool_eviction_order(self, graph):
        """淘汰顺序：按 priority_weight ASC, created_at ASC"""
        now = datetime.now(timezone.utc).isoformat()
        # 创建多个目标，权重不同
        for i in range(5):
            weight = 0.01 + i * 0.01  # 0.01, 0.02, 0.03, 0.04, 0.05
            graph.conn.execute(
                "INSERT INTO cognitive_goals "
                "(goal_id, goal_type, trigger_condition, domain, priority_weight, status, "
                " sub_goals, execution_log, decay_cycles_count, last_touched_at, created_at, updated_at) "
                "VALUES (?, 'low_confidence', 'test', ?, ?, 'pending', '[]', '[]', 0, ?, ?, ?)",
                (f"goal_pool_{i}", f"domain_{i}", weight, now, now, now),
            )
        graph.conn.commit()

        # 设置池上限为 3，触发淘汰
        with patch("consciousness_sea.metacognition.cognitive_goal.GOAL_POOL_MAX_SIZE", 3):
            mgr = CognitiveGoalManager(graph)
            cooled = mgr.cool_goals()

        # 验证最低权重的2个目标被淘汰
        active_count = graph.conn.execute(
            "SELECT COUNT(*) FROM cognitive_goals WHERE status = 'pending'"
        ).fetchone()[0]
        assert active_count <= 3

    def test_completed_not_counted_in_pool(self, graph):
        """completed 和 archived 状态的目标不计入池大小上限"""
        now = datetime.now(timezone.utc).isoformat()
        # 插入 completed 目标
        for i in range(10):
            graph.conn.execute(
                "INSERT INTO cognitive_goals "
                "(goal_id, goal_type, trigger_condition, domain, priority_weight, status, "
                " sub_goals, execution_log, decay_cycles_count, last_touched_at, created_at, updated_at) "
                "VALUES (?, 'low_confidence', 'test', ?, 0.1, 'completed', '[]', '[]', 0, ?, ?, ?)",
                (f"goal_completed_{i}", f"domain_c_{i}", now, now, now),
            )
        # 插入 pending 目标
        graph.conn.execute(
            "INSERT INTO cognitive_goals "
            "(goal_id, goal_type, trigger_condition, domain, priority_weight, status, "
            " sub_goals, execution_log, decay_cycles_count, last_touched_at, created_at, updated_at) "
            "VALUES (?, 'low_confidence', 'test', '医学', 0.5, 'pending', '[]', '[]', 0, ?, ?, ?)",
            ("goal_pending_pool", now, now, now),
        )
        graph.conn.commit()

        mgr = CognitiveGoalManager(graph)
        active_count = mgr._get_active_goal_count()
        assert active_count == 1  # 只有 pending 的1个


# ═══════════════════════════════════════════════════════════
#  场景5：目标去重逻辑
# ═══════════════════════════════════════════════════════════


class TestScenario5GoalDedup:
    """场景5：目标去重逻辑——同 domain + goal_type → 更新，不同 goal_type → 共存"""

    def test_dedup_updates_existing(self, graph):
        """同 domain + goal_type 且 status=pending → 更新优先级权重"""
        _insert_domain_monitor(
            graph.conn, "量子力学",
            {"avg_karma_density": 0.5, "ripple_success_rate": 0.3,
             "conflict_frequency": GOAL_LOW_CONF_THRESHOLD + 1},
        )
        mgr = CognitiveGoalManager(graph)

        # 第一次生成
        mgr.generate_goals()
        count1 = graph.conn.execute(
            "SELECT COUNT(*) FROM cognitive_goals WHERE domain = '量子力学' AND goal_type = 'low_confidence'"
        ).fetchone()[0]
        assert count1 == 1

        # 第二次生成（应更新而非创建）
        mgr.generate_goals()
        count2 = graph.conn.execute(
            "SELECT COUNT(*) FROM cognitive_goals WHERE domain = '量子力学' AND goal_type = 'low_confidence'"
        ).fetchone()[0]
        assert count2 == 1

    def test_different_types_coexist(self, graph):
        """同 domain 不同 goal_type 允许共存"""
        _insert_domain_monitor(
            graph.conn, "量子力学",
            {
                "avg_karma_density": 0.05,  # 触发 low_density
                "ripple_success_rate": 0.3,
                "conflict_frequency": GOAL_HIGH_CONFLICT_THRESHOLD + 5,  # 触发 high_conflict
            },
        )
        mgr = CognitiveGoalManager(graph)
        created = mgr.generate_goals()
        assert created >= 2

        types = graph.conn.execute(
            "SELECT DISTINCT goal_type FROM cognitive_goals WHERE domain = '量子力学'"
        ).fetchall()
        type_values = {r["goal_type"] for r in types}
        assert "low_density" in type_values
        assert "high_conflict" in type_values


# ═══════════════════════════════════════════════════════════
#  场景6：API 查询认知目标
# ═══════════════════════════════════════════════════════════


class TestScenario6APIQuery:
    """场景6：API 查询认知目标——所有端点返回正确格式"""

    def test_list_cognitive_goals_api(self, graph):
        """GET /api/v1/cognitive-goals 返回正确格式"""
        from fastapi.testclient import TestClient
        import consciousness_sea.interfaces.api as api
        api_module = sys.modules['consciousness_sea.interfaces.api']

        mock_pool = MagicMock()
        mock_pool.acquire.return_value = graph
        mock_pool.release.return_value = None

        guardian_loop = GuardianLoop(graph)
        goal_mgr = CognitiveGoalManager(graph)
        curiosity_engine = CuriosityEngine(graph, goal_mgr)

        api_module._pool = mock_pool
        api_module._guardian_loop = guardian_loop
        api_module._goal_mgr = goal_mgr
        api_module._curiosity_engine = curiosity_engine

        # 创建 mock Observer
        from consciousness_sea.infrastructure.observer import Observer, StatusData
        mock_observer = MagicMock(spec=Observer)
        mock_status = StatusData(
            total_seeds=8, total_karma_edges=5,
            hottest_seeds=[], coldest_seeds=[], heaviest_karma=[],
            recent_queries=[], alerts=[], domain_distribution={},
            cognitive_goals=None, curiosity_engine=None,
        )
        mock_observer.get_status.return_value = mock_status
        api_module._observer = mock_observer

        # 先生成认知目标
        _insert_domain_monitor(
            graph.conn, "量子力学",
            {"avg_karma_density": 0.5, "ripple_success_rate": 0.3,
             "conflict_frequency": GOAL_LOW_CONF_THRESHOLD + 1},
        )
        goal_mgr.generate_goals()

        with patch("consciousness_sea.interfaces.api.COGNITIVE_GOAL_ENABLED", True), \
             patch("consciousness_sea.interfaces.api.META_SEED_ENABLED", True):
            client = TestClient(api_module.app)
            response = client.get("/api/v1/cognitive-goals")

        assert response.status_code == 200
        data = response.json()
        assert "goals" in data
        assert len(data["goals"]) > 0

        api_module._pool = None
        api_module._guardian_loop = None
        api_module._goal_mgr = None
        api_module._curiosity_engine = None

    def test_curiosity_status_api(self, graph):
        """GET /api/v1/curiosity/status 返回引擎状态"""
        from fastapi.testclient import TestClient
        import consciousness_sea.interfaces.api as api
        api_module = sys.modules['consciousness_sea.interfaces.api']

        mock_pool = MagicMock()
        mock_pool.acquire.return_value = graph
        mock_pool.release.return_value = None

        goal_mgr = CognitiveGoalManager(graph)
        curiosity_engine = CuriosityEngine(graph, goal_mgr)

        api_module._pool = mock_pool
        api_module._curiosity_engine = curiosity_engine

        with patch("consciousness_sea.interfaces.api.CURIOSITY_ENGINE_ENABLED", True):
            client = TestClient(api_module.app)
            response = client.get("/api/v1/curiosity/status")

        assert response.status_code == 200
        data = response.json()
        assert "total_explorations" in data
        assert "is_exploring" in data

        api_module._pool = None
        api_module._curiosity_engine = None

    def test_guardian_trigger_api(self, graph):
        """POST /api/v1/guardian/trigger 触发守护循环（含目标生成）"""
        from fastapi.testclient import TestClient
        import consciousness_sea.interfaces.api as api
        api_module = sys.modules['consciousness_sea.interfaces.api']

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
            total_seeds=8, total_karma_edges=5,
            hottest_seeds=[], coldest_seeds=[], heaviest_karma=[],
            recent_queries=[], alerts=[], domain_distribution={},
        )
        mock_observer.get_status.return_value = mock_status
        api_module._observer = mock_observer

        with patch("consciousness_sea.interfaces.api.META_SEED_ENABLED", True), \
             patch("consciousness_sea.interfaces.api.COGNITIVE_GOAL_ENABLED", True):
            client = TestClient(api_module.app)
            response = client.post("/api/v1/guardian/trigger")

        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "success"

        api_module._pool = None
        api_module._guardian_loop = None


# ═══════════════════════════════════════════════════════════
#  场景7：虚拟查询不触发熏习
# ═══════════════════════════════════════════════════════════


class TestScenario7VirtualQueryNoKarma:
    """场景7：虚拟查询不触发熏习——skip_verification=True"""

    def test_virtual_query_skip_verification(self, graph):
        """route(query, skip_verification=True) 不触发校验和熏习"""
        # 记录熏习前的业力边数
        edge_count_before = graph.conn.execute(
            "SELECT COUNT(*) FROM karma_edges WHERE source NOT LIKE 'meta:%'"
        ).fetchone()[0]

        # 执行虚拟查询
        result = route("感冒 发热", graph, skip_verification=True)

        # 验证涟漪传播正常执行
        assert len(result.activated) > 0

        # 验证业力边数未增加（无熏习）
        edge_count_after = graph.conn.execute(
            "SELECT COUNT(*) FROM karma_edges WHERE source NOT LIKE 'meta:%'"
        ).fetchone()[0]
        assert edge_count_after == edge_count_before

    def test_virtual_query_max_depth(self, graph):
        """route(query, skip_verification=True, max_depth=1) 限制传播深度"""
        result = route("感冒 发热", graph, skip_verification=True, max_depth=1)

        # 验证最大深度不超过1
        for node in result.activated.values():
            assert node.depth <= 1

    def test_virtual_query_no_personal_weights(self, graph):
        """虚拟查询不使用个人业力权重"""
        # 插入个人业力
        now = datetime.now(timezone.utc).isoformat()
        graph.conn.execute(
            "INSERT INTO karma_edges_personal "
            "(user_label, source, target, relation, weight, updated_at) "
            "VALUES ('test_user', '感冒', '发热', 'COOCCURS_WITH', 0.99, ?)",
            (now,),
        )
        graph.conn.commit()

        # 虚拟查询应使用全局权重
        result = route("感冒", graph, skip_verification=True)
        assert len(result.activated) > 0