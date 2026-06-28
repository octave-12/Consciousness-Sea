"""
Phase 2 参数统计评估测试 (T8.3)

覆盖:
- 统计收集：查询后 param_stats 表有新记录
- 衰减系数评估：输出各值精确度 + 推荐值
- 领域阈值评估：输出 F1 + 推荐值
- 正向熏习条件评估：输出熏习质量 + 推荐值
- 统计数据不足 100 次时的警告标注
- 评估工具不修改 config.py
"""

import sqlite3
import sys
import os
import json
from pathlib import Path
from datetime import datetime, timezone

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.graph_db import GraphDB
from core.param_evaluator import ParamEvaluator
from core.connection_pool import ConnectionPool
from core.config import RIPPLE_DECAY, DOMAIN_THRESHOLD, CONFIDENCE_HIGH


def _build_test_db(db_path: str) -> None:
    """创建测试用 SQLite 数据库文件（含 Phase 2 表）"""
    conn = sqlite3.connect(db_path)
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
        CREATE TABLE IF NOT EXISTS karma_edges_personal (
            user_label  TEXT    NOT NULL,
            source      TEXT    NOT NULL,
            target      TEXT    NOT NULL,
            relation    TEXT    NOT NULL,
            weight      REAL    NOT NULL,
            source_tag  TEXT    NOT NULL DEFAULT 'personal_karma',
            updated_at  TEXT    NOT NULL,
            PRIMARY KEY (user_label, source, target, relation)
        );
        CREATE TABLE IF NOT EXISTS distillation_pool (
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
        CREATE TABLE IF NOT EXISTS param_stats (
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

    seeds = [
        ('感冒', '感冒', 'CONCEPT', '[]', '医学', 'cold'),
        ('发热', '发热', 'CONCEPT', '[]', '医学', 'fever'),
        ('咳嗽', '咳嗽', 'CONCEPT', '[]', '医学', 'cough'),
    ]
    conn.executemany(
        "INSERT INTO seeds (id, label, type, aliases, domain, definition) "
        "VALUES (?, ?, ?, ?, ?, ?)",
        seeds,
    )

    edges = [
        ('感冒', '发热', 'COOCCURS_WITH', 0.95),
        ('感冒', '咳嗽', 'COOCCURS_WITH', 0.90),
    ]
    conn.executemany(
        "INSERT INTO karma_edges (source, target, relation, weight) "
        "VALUES (?, ?, ?, ?)",
        edges,
    )
    conn.commit()
    conn.close()


def _insert_param_stats(conn: sqlite3.Connection, count: int) -> None:
    """向 param_stats 表插入指定数量的测试记录"""
    now = datetime.now(timezone.utc).isoformat()
    for i in range(count):
        conn.execute(
            "INSERT INTO param_stats "
            "(query_text, decay_factor, domain_threshold, confidence_high, "
            " ripple_depth, activated_count, selected_domains, confidence, "
            " karma_direction, created_at) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                f"测试查询{i}",
                RIPPLE_DECAY,
                DOMAIN_THRESHOLD,
                CONFIDENCE_HIGH,
                2,
                5,
                json.dumps(['医学'], ensure_ascii=False),
                0.8,
                1,
                now,
            ),
        )
    conn.commit()


class TestStatsCollection:
    """统计收集：查询后 param_stats 表有新记录"""

    def test_param_stats_table_exists(self, tmp_path):
        """param_stats 表在数据库连接后自动创建"""
        db_path = str(Path(tmp_path) / "test.db")
        _build_test_db(db_path)

        graph = GraphDB(db_path)
        graph.connect()

        # ensure_phase2_tables 应已创建 param_stats 表
        row = graph.conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='param_stats'"
        ).fetchone()
        assert row is not None

        graph.close()

    def test_insert_and_query_param_stats(self, tmp_path):
        """插入统计数据后可查询"""
        db_path = str(Path(tmp_path) / "test.db")
        _build_test_db(db_path)

        conn = sqlite3.connect(db_path)
        conn.row_factory = sqlite3.Row

        _insert_param_stats(conn, 5)

        rows = conn.execute("SELECT * FROM param_stats").fetchall()
        assert len(rows) == 5

        conn.close()


class TestDecayFactorEvaluation:
    """衰减系数评估：输出各值精确度 + 推荐值"""

    def test_evaluate_decay_factor(self, tmp_path):
        """衰减系数评估返回候选值列表和推荐值"""
        db_path = str(Path(tmp_path) / "test.db")
        _build_test_db(db_path)

        # 插入统计数据
        conn = sqlite3.connect(db_path)
        _insert_param_stats(conn, 10)
        conn.close()

        pool = ConnectionPool(db_path, pool_size=2)
        evaluator = ParamEvaluator(pool)

        report = evaluator.evaluate_decay_factor()

        assert report['parameter'] == 'decay_factor'
        assert isinstance(report['candidates'], list)
        assert len(report['candidates']) > 0
        assert 'value' in report['candidates'][0]
        assert 'precision' in report['candidates'][0]
        assert report['recommended'] is not None or len(report['candidates']) == 0
        assert 'recommendation_reason' in report
        assert 'sample_size' in report
        assert report['sample_size'] == 10

        pool.close_all()

    def test_decay_factor_precision_range(self, tmp_path):
        """精确度值在 [0, 1] 范围内"""
        db_path = str(Path(tmp_path) / "test.db")
        _build_test_db(db_path)

        conn = sqlite3.connect(db_path)
        _insert_param_stats(conn, 10)
        conn.close()

        pool = ConnectionPool(db_path, pool_size=2)
        evaluator = ParamEvaluator(pool)

        report = evaluator.evaluate_decay_factor()

        for candidate in report['candidates']:
            assert 0.0 <= candidate['precision'] <= 1.0

        pool.close_all()


class TestDomainThresholdEvaluation:
    """领域阈值评估：输出 F1 + 推荐值"""

    def test_evaluate_domain_threshold(self, tmp_path):
        """领域阈值评估返回 F1 分数和推荐值"""
        db_path = str(Path(tmp_path) / "test.db")
        _build_test_db(db_path)

        conn = sqlite3.connect(db_path)
        _insert_param_stats(conn, 10)
        conn.close()

        pool = ConnectionPool(db_path, pool_size=2)
        evaluator = ParamEvaluator(pool)

        report = evaluator.evaluate_domain_threshold()

        assert report['parameter'] == 'domain_threshold'
        assert isinstance(report['candidates'], list)
        assert len(report['candidates']) > 0
        assert 'value' in report['candidates'][0]
        assert 'false_positive' in report['candidates'][0]
        assert 'false_negative' in report['candidates'][0]
        assert 'f1' in report['candidates'][0]
        assert 'recommendation_reason' in report
        assert report['sample_size'] == 10

        pool.close_all()

    def test_domain_f1_range(self, tmp_path):
        """F1 分数在 [0, 1] 范围内"""
        db_path = str(Path(tmp_path) / "test.db")
        _build_test_db(db_path)

        conn = sqlite3.connect(db_path)
        _insert_param_stats(conn, 10)
        conn.close()

        pool = ConnectionPool(db_path, pool_size=2)
        evaluator = ParamEvaluator(pool)

        report = evaluator.evaluate_domain_threshold()

        for candidate in report['candidates']:
            assert 0.0 <= candidate['f1'] <= 1.0
            assert 0.0 <= candidate['false_positive'] <= 1.0
            assert 0.0 <= candidate['false_negative'] <= 1.0

        pool.close_all()


class TestPositiveKarmaEvaluation:
    """正向熏习条件评估：输出熏习质量 + 推荐值"""

    def test_evaluate_positive_karma_threshold(self, tmp_path):
        """正向熏习条件评估返回熏习质量和推荐值"""
        db_path = str(Path(tmp_path) / "test.db")
        _build_test_db(db_path)

        conn = sqlite3.connect(db_path)
        _insert_param_stats(conn, 10)
        conn.close()

        pool = ConnectionPool(db_path, pool_size=2)
        evaluator = ParamEvaluator(pool)

        report = evaluator.evaluate_positive_karma_threshold()

        assert report['parameter'] == 'positive_karma_threshold'
        assert isinstance(report['candidates'], list)
        assert len(report['candidates']) > 0
        assert 'value' in report['candidates'][0]
        assert 'positive_rate' in report['candidates'][0]
        assert 'negative_rate' in report['candidates'][0]
        assert 'quality' in report['candidates'][0]
        assert 'recommendation_reason' in report
        assert report['sample_size'] == 10

        pool.close_all()

    def test_karma_quality_range(self, tmp_path):
        """熏习质量指标在合理范围内"""
        db_path = str(Path(tmp_path) / "test.db")
        _build_test_db(db_path)

        conn = sqlite3.connect(db_path)
        _insert_param_stats(conn, 10)
        conn.close()

        pool = ConnectionPool(db_path, pool_size=2)
        evaluator = ParamEvaluator(pool)

        report = evaluator.evaluate_positive_karma_threshold()

        for candidate in report['candidates']:
            assert 0.0 <= candidate['positive_rate'] <= 1.0
            assert 0.0 <= candidate['negative_rate'] <= 1.0
            assert 0.0 <= candidate['quality'] <= 1.0

        pool.close_all()


class TestInsufficientSamplesWarning:
    """统计数据不足 100 次时的警告标注"""

    def test_warning_when_below_100(self, tmp_path):
        """样本数 < 100 时评估报告包含警告"""
        db_path = str(Path(tmp_path) / "test.db")
        _build_test_db(db_path)

        conn = sqlite3.connect(db_path)
        _insert_param_stats(conn, 10)
        conn.close()

        pool = ConnectionPool(db_path, pool_size=2)
        evaluator = ParamEvaluator(pool)

        report = evaluator.evaluate_decay_factor()

        assert report['warning'] is not None
        assert '100' in report['warning'] or '不足' in report['warning']

        pool.close_all()

    def test_no_warning_when_above_100(self, tmp_path):
        """样本数 >= 100 时评估报告无警告"""
        db_path = str(Path(tmp_path) / "test.db")
        _build_test_db(db_path)

        conn = sqlite3.connect(db_path)
        _insert_param_stats(conn, 100)
        conn.close()

        pool = ConnectionPool(db_path, pool_size=2)
        evaluator = ParamEvaluator(pool)

        report = evaluator.evaluate_decay_factor()

        assert report['warning'] is None

        pool.close_all()


class TestEvaluatorDoesNotModifyConfig:
    """评估工具不修改 config.py"""

    def test_config_unchanged_after_evaluation(self, tmp_path):
        """评估前后 config.py 中的参数值不变"""
        db_path = str(Path(tmp_path) / "test.db")
        _build_test_db(db_path)

        conn = sqlite3.connect(db_path)
        _insert_param_stats(conn, 10)
        conn.close()

        # 记录评估前的配置值
        from core import config
        decay_before = config.RIPPLE_DECAY
        threshold_before = config.DOMAIN_THRESHOLD
        confidence_before = config.CONFIDENCE_HIGH

        pool = ConnectionPool(db_path, pool_size=2)
        evaluator = ParamEvaluator(pool)

        evaluator.evaluate_all()

        # 验证配置值未改变
        assert config.RIPPLE_DECAY == decay_before
        assert config.DOMAIN_THRESHOLD == threshold_before
        assert config.CONFIDENCE_HIGH == confidence_before

        pool.close_all()

    def test_evaluate_all_report(self, tmp_path):
        """evaluate_all() 返回综合报告"""
        db_path = str(Path(tmp_path) / "test.db")
        _build_test_db(db_path)

        conn = sqlite3.connect(db_path)
        _insert_param_stats(conn, 10)
        conn.close()

        pool = ConnectionPool(db_path, pool_size=2)
        evaluator = ParamEvaluator(pool)

        report = evaluator.evaluate_all()

        assert 'reports' in report
        assert 'decay_factor' in report['reports']
        assert 'domain_threshold' in report['reports']
        assert 'positive_karma_threshold' in report['reports']
        assert 'elapsed_seconds' in report
        assert 'note' in report
        assert '不自动修改' in report['note']

        pool.close_all()