"""
Phase 4 GuardianLoop 单元测试

覆盖：
- 守护循环启动/停止
- 单次执行（execute_once）
- 领域健康度检查
- 关系质量检查
- 系统级指标检查
- 自边界更新
- 未知领域探测
- 状态查询
"""

from __future__ import annotations

import json
import pathlib
import sqlite3
import sys
import time
from unittest.mock import patch

import pytest

_root = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_root))
sys.path.insert(0, str(_root / "backend" / "src"))

from consciousness_sea.domain.graph_db import GraphDB
from consciousness_sea.infrastructure.config import (
    GUARDIAN_LOOP_INTERVAL,
)
from consciousness_sea.metacognition.guardian_loop import (
    GuardianLoop,
    GuardianLoopResult,
    GuardianLoopStatus,
)
from consciousness_sea.metacognition.meta_seed import (
    MetaSeedCategory,
    MetaSeedManager,
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
        CREATE TABLE user_cold_start (
            user_label  TEXT    PRIMARY KEY,
            query_count INTEGER NOT NULL DEFAULT 0,
            updated_at  TEXT    NOT NULL
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
def guardian(graph):
    """创建 GuardianLoop 实例"""
    return GuardianLoop(graph)


# ═══════════════════════════════════════════════════════════
#  测试用例
# ═══════════════════════════════════════════════════════════


class TestGuardianLoopControl:
    """守护循环启动/停止测试"""

    def test_start_creates_daemon_thread(self, guardian):
        """启动守护线程：daemon 线程，名称 'guardian-loop'"""
        with patch("consciousness_sea.metacognition.guardian_loop.GUARDIAN_LOOP_INITIAL_DELAY", 0.01):
            guardian.start()
            try:
                assert guardian._daemon_thread is not None
                assert guardian._daemon_thread.is_alive()
                assert guardian._daemon_thread.daemon is True
                assert guardian._daemon_thread.name == "guardian-loop"
            finally:
                guardian.stop()

    def test_start_idempotent(self, guardian):
        """守护循环已在运行时不重复启动"""
        with patch("consciousness_sea.metacognition.guardian_loop.GUARDIAN_LOOP_INITIAL_DELAY", 0.01):
            guardian.start()
            try:
                thread1 = guardian._daemon_thread
                guardian.start()  # 再次调用
                thread2 = guardian._daemon_thread
                assert thread1 is thread2  # 同一个线程
            finally:
                guardian.stop()

    def test_stop_terminates_thread(self, guardian):
        """停止守护线程"""
        with patch("consciousness_sea.metacognition.guardian_loop.GUARDIAN_LOOP_INITIAL_DELAY", 0.01):
            guardian.start()
            assert guardian._daemon_thread is not None
            assert guardian._daemon_thread.is_alive()

            guardian.stop()
            time.sleep(0.1)
            assert guardian._daemon_thread is None or not guardian._daemon_thread.is_alive()

    def test_stop_when_not_running(self, guardian):
        """未运行时 stop 不报错"""
        guardian.stop()  # 不应抛异常


class TestGuardianExecuteOnce:
    """单次执行流程测试"""

    def test_execute_once_success(self, guardian, graph):
        """单次执行成功返回 GuardianLoopResult"""
        result = guardian.execute_once()
        assert isinstance(result, GuardianLoopResult)
        assert result.success is True
        assert result.duration_ms >= 0

    def test_execute_once_generates_meta_seeds(self, guardian, graph):
        """单次执行生成元种子"""
        result = guardian.execute_once()
        assert result.meta_seeds_updated > 0

        # 验证元种子已创建
        count = graph.conn.execute("SELECT COUNT(*) FROM meta_seeds").fetchone()[0]
        assert count > 0

    def test_execute_once_commits(self, guardian, graph):
        """单次执行提交所有数据库变更"""
        guardian.execute_once()

        # 验证数据已提交（可读取到）
        count = graph.conn.execute("SELECT COUNT(*) FROM meta_seeds").fetchone()[0]
        assert count > 0

    def test_execute_once_disabled(self, guardian, graph):
        """META_SEED_ENABLED=False 时返回空结果"""
        with patch("consciousness_sea.metacognition.guardian_loop.META_SEED_ENABLED", False):
            result = guardian.execute_once()
            assert result.success is True
            assert result.meta_seeds_updated == 0

    def test_execute_once_logs_result(self, guardian, graph):
        """单次执行记录 INFO 日志"""
        with patch("consciousness_sea.metacognition.guardian_loop.log") as mock_log:
            result = guardian.execute_once()
            if result.success:
                mock_log.info.assert_called()
                call_args = str(mock_log.info.call_args)
                assert "guardian loop completed" in call_args


class TestGuardianDomainHealth:
    """领域健康度检查测试"""

    def test_check_domain_health(self, guardian, graph):
        """领域健康度指标更新"""
        # 先生成领域监控元种子
        mgr = MetaSeedManager(graph)
        mgr.generate_domain_monitors()

        # 插入 param_stats 数据
        now = "2025-01-01T00:00:00+00:00"
        for i in range(10):
            graph.conn.execute(
                "INSERT INTO param_stats "
                "(query_text, decay_factor, domain_threshold, confidence_high, "
                "ripple_depth, activated_count, selected_domains, confidence, karma_direction, created_at) "
                "VALUES (?, 1.0, 0.3, 0.7, 2, 5, ?, 0.5, 0, ?)",
                (f"query_{i}", json.dumps(["医学"]), now),
            )
        graph.conn.commit()

        updated = guardian._check_domain_health()
        assert updated > 0

        # 验证指标已更新
        ms = mgr.get_meta_seed("meta:医学")
        assert ms is not None
        assert ms.metrics["avg_karma_density"] >= 0.0
        assert ms.metrics["ripple_success_rate"] >= 0.0


class TestGuardianRelationQuality:
    """关系质量检查测试"""

    def test_check_relation_quality(self, guardian, graph):
        """关系质量指标更新"""
        mgr = MetaSeedManager(graph)
        mgr.generate_relation_monitors()

        updated = guardian._check_relation_quality()
        assert updated > 0

        ms = mgr.get_meta_seed("meta:RELATED")
        assert ms is not None
        assert ms.metrics["avg_weight"] >= 0.0


class TestGuardianSystemMetrics:
    """系统级指标检查测试"""

    def test_check_system_metrics(self, guardian, graph):
        """系统级指标更新"""
        mgr = MetaSeedManager(graph)
        mgr.generate_system_monitors()

        updated = guardian._check_system_metrics()
        assert updated >= 5  # 5 个系统级元种子

        ms = mgr.get_meta_seed("meta:system_total_nodes")
        assert ms is not None
        assert ms.metrics["value"] >= 0

        ms = mgr.get_meta_seed("meta:system_total_edges")
        assert ms is not None
        assert ms.metrics["value"] >= 0


class TestGuardianSelfBoundary:
    """自边界更新测试"""

    def test_update_self_boundary(self, guardian, graph):
        """自边界元种子指标更新"""
        mgr = MetaSeedManager(graph)
        mgr._create_meta_seed_record(
            "meta:unknown", MetaSeedCategory.SELF_BOUNDARY,
            {"unmatched_keywords": [], "unmatched_count": 0, "top_unmatched": []},
        )

        # 插入候选种子
        now = "2025-01-01T00:00:00+00:00"
        graph.conn.execute(
            "INSERT INTO candidate_seeds (label, status, count, candidate_since, last_seen_at) "
            "VALUES (?, 'candidate', ?, ?, ?)",
            ("DeepSeek", 5, now, now),
        )
        graph.conn.commit()

        updated = guardian._update_self_boundary()
        assert updated == 1


class TestGuardianUnknownDomains:
    """未知领域探测测试"""

    def test_detect_unknown_domains(self, guardian, graph):
        """低置信度高频区域探测"""
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

        updated = guardian._detect_unknown_domains()
        assert updated >= 1


class TestGuardianStatus:
    """守护循环状态查询测试"""

    def test_get_status(self, guardian):
        """查询守护循环运行状态"""
        status = guardian.get_status()
        assert isinstance(status, GuardianLoopStatus)
        assert status.is_running is False
        assert status.interval_seconds == GUARDIAN_LOOP_INTERVAL
        assert status.consecutive_failures == 0

    def test_get_status_after_execution(self, guardian, graph):
        """执行后状态更新"""
        guardian.execute_once()
        status = guardian.get_status()
        assert status.last_execution_result == "success"
        assert status.last_execution_duration_ms is not None
        assert status.last_execution_duration_ms >= 0

    def test_is_executing_property(self, guardian):
        """is_executing 属性"""
        assert guardian.is_executing is False


class TestGuardianIntegration:
    """守护循环集成测试"""

    def test_full_execute_once_flow(self, guardian, graph):
        """完整单次执行流程：生成 → 更新 → 熏习 → 休眠判定"""
        result = guardian.execute_once()
        assert result.success is True
        assert result.meta_seeds_updated > 0

        # 验证元种子已创建
        count = graph.conn.execute("SELECT COUNT(*) FROM meta_seeds").fetchone()[0]
        # 4 domain + 3 relation + 5 system = 12
        assert count >= 12

        # 验证 seeds 表中的 META 记录
        meta_count = graph.conn.execute(
            "SELECT COUNT(*) FROM seeds WHERE type = 'META'"
        ).fetchone()[0]
        assert meta_count >= 12

    def test_execute_once_idempotent(self, guardian, graph):
        """多次执行幂等：不重复创建元种子"""
        guardian.execute_once()
        count1 = graph.conn.execute("SELECT COUNT(*) FROM meta_seeds").fetchone()[0]

        guardian.execute_once()
        count2 = graph.conn.execute("SELECT COUNT(*) FROM meta_seeds").fetchone()[0]

        # 第二次不应创建新的元种子
        assert count2 == count1

    def test_execute_once_with_new_domain(self, guardian, graph):
        """新增领域后下次执行自动生成元种子"""
        guardian.execute_once()
        count1 = graph.conn.execute("SELECT COUNT(*) FROM meta_seeds WHERE category = 'domain_monitor'").fetchone()[0]

        # 新增领域
        graph.conn.execute(
            "INSERT INTO seeds (id, label, type, domain) VALUES ('热力学', '热力学', 'CONCEPT', '物理新领域')"
        )
        graph.conn.commit()

        guardian.execute_once()
        count2 = graph.conn.execute("SELECT COUNT(*) FROM meta_seeds WHERE category = 'domain_monitor'").fetchone()[0]
        assert count2 > count1

        # 验证新元种子
        ms = MetaSeedManager(graph).get_meta_seed("meta:物理新领域")
        assert ms is not None
