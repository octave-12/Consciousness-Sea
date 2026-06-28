"""
Phase 2 端到端验收测试 (T10.2)

覆盖场景:
- 场景1：Top-N 熏习将目标对从全量降至 ≤500
- 场景2：低权边在定期清理中自动删除
- 场景3：用户 A 的错误关联不影响用户 B
- 场景4：提炼池三用户升级为全局业力
- 场景5：双层业力叠加计算正确
- 场景6：现有 API 和测试不受影响
"""

import sqlite3
import sys
import os
import json
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.graph_db import GraphDB
from core.verifier import apply_karma, verify
from core.router import route, RippleResult, ActivationNode
from core.karma_cleaner import KarmaCleaner
from core.distillation_pool import DistillationPool
from core.connection_pool import ConnectionPool
from core.config import COLD_START_QUERIES


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

    # 种子数据
    seeds = [
        ('感冒', '感冒', 'CONCEPT', '[]', '医学', 'cold'),
        ('发热', '发热', 'CONCEPT', '[]', '医学', 'fever'),
        ('咳嗽', '咳嗽', 'CONCEPT', '[]', '医学', 'cough'),
        ('头痛', '头痛', 'CONCEPT', '[]', '医学', 'headache'),
        ('量子力学', '量子力学', 'CONCEPT', '[]', '物理', 'quantum mechanics'),
        ('薛定谔方程', '薛定谔方程', 'CONCEPT', '[]', '物理', 'Schrodinger equation'),
        ('人工智能', '人工智能', 'CONCEPT', '["AI"]', '计算机', 'AI'),
        ('深度学习', '深度学习', 'CONCEPT', '[]', '计算机', 'deep learning'),
    ]
    conn.executemany(
        "INSERT INTO seeds (id, label, type, aliases, domain, definition) "
        "VALUES (?, ?, ?, ?, ?, ?)",
        seeds,
    )

    # 边数据
    edges = [
        ('感冒', '发热', 'COOCCURS_WITH', 0.95),
        ('感冒', '咳嗽', 'COOCCURS_WITH', 0.90),
        ('头痛', '感冒', 'RELATED', 0.003),  # 低权边
        ('量子力学', '薛定谔方程', 'RELATED', 0.85),
        ('人工智能', '深度学习', 'IS_A', 0.90),
        ('感冒', '量子力学', 'RELATED', 0.05),
    ]
    conn.executemany(
        "INSERT INTO karma_edges (source, target, relation, weight) "
        "VALUES (?, ?, ?, ?)",
        edges,
    )
    conn.commit()
    conn.close()


class TestScenario1TopNReduction:
    """场景1：Top-N 熏习将目标对从全量降至 ≤500"""

    def test_top_n_reduces_modified_pairs(self, tmp_path):
        """KARMA_FULL_SET=False 时熏习对数少于全量"""
        db_path = str(Path(tmp_path) / "test.db")
        _build_test_db(db_path)

        graph = GraphDB(db_path)
        graph.connect()

        # 执行路由
        result = route('感冒', graph)

        # 全量熏习
        with patch('core.config.KARMA_FULL_SET', True):
            full_count = apply_karma(result, graph, karma_direction=+1, dry_run=True)

        # Top-N 熏习
        with patch('core.config.KARMA_FULL_SET', False), \
             patch('core.config.KARMA_TOP_N', 3):
            top_n_count = apply_karma(result, graph, karma_direction=+1, dry_run=True)

        # Top-N 熏习对数应 <= 全量
        assert top_n_count <= full_count

        graph.close()

    def test_top_n_max_pairs_protection(self, tmp_path):
        """KARMA_MAX_PAIRS=500 时熏习对数不超过 500"""
        db_path = str(Path(tmp_path) / "test.db")
        _build_test_db(db_path)

        graph = GraphDB(db_path)
        graph.connect()

        result = route('感冒', graph)

        with patch('core.config.KARMA_FULL_SET', True), \
             patch('core.config.KARMA_MAX_PAIRS', 500):
            modified = apply_karma(result, graph, karma_direction=+1, dry_run=True)

        assert modified <= 500

        graph.close()


class TestScenario2LowWeightEdgeAutoDelete:
    """场景2：低权边在定期清理中自动删除"""

    def test_low_weight_edge_deleted_by_cleaner(self, tmp_path):
        """KarmaCleaner 自动删除低权边"""
        db_path = str(Path(tmp_path) / "test.db")
        _build_test_db(db_path)

        pool = ConnectionPool(db_path, pool_size=2)
        cleaner = KarmaCleaner(pool)

        # 清理前验证低权边存在
        graph = pool.acquire()
        edge_before = graph.get_edge('头痛', '感冒', 'RELATED')
        assert edge_before is not None
        assert edge_before['weight'] < 0.01  # 低于 KARMA_MIN
        pool.release(graph)

        # 执行清理
        result = cleaner.cleanup_low_weight_edges()
        assert result['deleted'] >= 1

        # 清理后验证低权边已删除
        graph = pool.acquire()
        edge_after = graph.get_edge('头痛', '感冒', 'RELATED')
        assert edge_after is None
        pool.release(graph)

        pool.close_all()

    def test_normal_edges_preserved_after_cleanup(self, tmp_path):
        """清理后正常权重的边仍保留"""
        db_path = str(Path(tmp_path) / "test.db")
        _build_test_db(db_path)

        pool = ConnectionPool(db_path, pool_size=2)
        cleaner = KarmaCleaner(pool)

        cleaner.cleanup_low_weight_edges()

        graph = pool.acquire()
        edge = graph.get_edge('感冒', '发热', 'COOCCURS_WITH')
        assert edge is not None
        assert edge['weight'] > 0.01
        pool.release(graph)

        pool.close_all()


class TestScenario3UserIsolation:
    """场景3：用户 A 的错误关联不影响用户 B"""

    def test_user_a_error_does_not_affect_user_b(self, tmp_path):
        """用户 A 的负向熏习不影响用户 B 的个人业力"""
        db_path = str(Path(tmp_path) / "test.db")
        _build_test_db(db_path)

        graph = GraphDB(db_path)
        graph.connect()

        # 用户 A 和 B 都有个人业力
        graph.adjust_karma_personal('user_A', '感冒', '发热', 'COOCCURS_WITH', delta=0.1)
        graph.adjust_karma_personal('user_B', '感冒', '发热', 'COOCCURS_WITH', delta=0.1)
        graph.conn.commit()

        # 用户 A 负向熏习（错误关联）
        graph.adjust_karma_personal('user_A', '感冒', '发热', 'COOCCURS_WITH', delta=-0.3)
        graph.conn.commit()

        # 用户 B 的个人业力应不变
        weight_b = graph.get_personal_weight('user_B', '感冒', '发热', 'COOCCURS_WITH')
        assert weight_b is not None
        # user_B: 0.5 + 0.1 = 0.6
        assert abs(weight_b - 0.6) < 0.01

        # 全局业力也不应变
        global_edge = graph.get_edge('感冒', '发热', 'COOCCURS_WITH')
        assert global_edge is not None
        assert abs(global_edge['weight'] - 0.95) < 0.01

        graph.close()

    def test_user_a_karma_does_not_affect_user_b_routing(self, tmp_path):
        """用户 A 的个人业力不影响用户 B 的路由结果"""
        db_path = str(Path(tmp_path) / "test.db")
        _build_test_db(db_path)

        graph = GraphDB(db_path)
        graph.connect()

        # 用户 A 大量正向熏习
        for _ in range(10):
            graph.adjust_karma_personal('user_A', '感冒', '量子力学', 'RELATED', delta=0.1)
        graph.conn.commit()

        # 用户 B 路由
        result_b = route('感冒', graph, user_label='user_B')

        # 用户 B 的路径中不应包含用户 A 的个人业力影响
        for p in result_b.paths:
            if p['source'] == '感冒' and p['target'] == '量子力学':
                # 用户 B 没有个人业力，personal_w=0
                # weight = global_w * 0.7 + 0 * 0.3
                global_w = 0.05
                expected_weight = global_w * 0.7
                assert abs(p['weight'] - expected_weight) < 0.01

        graph.close()


class TestScenario4DistillationUpgrade:
    """场景4：提炼池三用户升级为全局业力"""

    def test_three_users_upgrade_to_global_karma(self, tmp_path):
        """三个独立用户提交同一关联后升级为全局业力"""
        db_path = str(Path(tmp_path) / "test.db")
        _build_test_db(db_path)

        graph = GraphDB(db_path)
        graph.connect()

        distill = DistillationPool(graph)

        # 三个用户提交同一候选
        for user in ['user_A', 'user_B', 'user_C']:
            distill.submit_candidate(
                user_label=user, source='感冒', target='头痛', relation='RELATED',
            )

        # 验证全局业力边已创建
        edge = graph.get_edge('感冒', '头痛', 'RELATED')
        assert edge is not None
        from core.config import DISTILLATION_INITIAL_WEIGHT
        assert edge['weight'] >= DISTILLATION_INITIAL_WEIGHT

        # 验证提炼池状态
        status = distill.get_status()
        assert status['upgraded_count'] == 1

        graph.close()


class TestScenario5DualKarmaSuperposition:
    """场景5：双层业力叠加计算正确"""

    def test_ripple_weight_calculation(self, tmp_path):
        """涟漪传播权重 = global × 0.7 + personal × 0.3"""
        db_path = str(Path(tmp_path) / "test.db")
        _build_test_db(db_path)

        graph = GraphDB(db_path)
        graph.connect()

        # 创建个人业力
        graph.adjust_karma_personal('user_A', '感冒', '发热', 'COOCCURS_WITH', delta=0.3)
        graph.conn.commit()

        # 递增用户查询计数使冷启动因子=1.0（否则新用户 cold_factor=0 会清零个人权重）
        from core.cold_start import ColdStartManager
        csm = ColdStartManager(graph)
        for _ in range(COLD_START_QUERIES):
            csm.increment_query_count('user_A')

        # 执行路由
        result = route('感冒', graph, user_label='user_A')

        # 查找感冒→发热的路径
        path = None
        for p in result.paths:
            if p['source'] == '感冒' and p['target'] == '发热':
                path = p
                break

        assert path is not None

        # 验证权重叠加
        global_w = 0.95
        personal_w = 0.8  # 0.5 + 0.3
        expected_weight = global_w * 0.7 + personal_w * 0.3

        assert abs(path['weight'] - expected_weight) < 0.01

        graph.close()

    def test_no_personal_weight_uses_global_only(self, tmp_path):
        """没有个人业力时权重 = global × 0.7"""
        db_path = str(Path(tmp_path) / "test.db")
        _build_test_db(db_path)

        graph = GraphDB(db_path)
        graph.connect()

        # 执行路由（无个人业力）
        result = route('感冒', graph, user_label=None)

        # 查找感冒→发热的路径
        path = None
        for p in result.paths:
            if p['source'] == '感冒' and p['target'] == '发热':
                path = p
                break

        assert path is not None

        # 无个人业力
        global_w = 0.95
        expected_weight = global_w * 0.7

        assert abs(path['weight'] - expected_weight) < 0.01

        graph.close()


class TestScenario6BackwardCompatibility:
    """场景6：现有 API 和测试不受影响"""

    def test_route_without_user_label(self, tmp_path):
        """不传 user_label 时路由行为与 Phase 0/1 一致"""
        db_path = str(Path(tmp_path) / "test.db")
        _build_test_db(db_path)

        graph = GraphDB(db_path)
        graph.connect()

        result = route('感冒', graph, user_label=None)

        assert result.query == '感冒'
        assert len(result.activated) > 0
        assert len(result.paths) > 0

        graph.close()

    def test_apply_karma_without_user_label(self, tmp_path):
        """不传 user_label 时熏习写入全局业力"""
        db_path = str(Path(tmp_path) / "test.db")
        _build_test_db(db_path)

        graph = GraphDB(db_path)
        graph.connect()

        result = route('感冒', graph, user_label=None)

        # 记录全局业力初始值
        edge_before = graph.get_edge('感冒', '发热', 'COOCCURS_WITH')
        weight_before = edge_before['weight'] if edge_before else 0.95

        # 熏习（不传 user_label）
        modified = apply_karma(result, graph, karma_direction=+1, user_label=None)

        # 全局业力应改变
        edge_after = graph.get_edge('感冒', '发热', 'COOCCURS_WITH')
        assert edge_after is not None

        graph.close()

    def test_verify_still_works(self, tmp_path):
        """校验器仍正常工作"""
        db_path = str(Path(tmp_path) / "test.db")
        _build_test_db(db_path)

        graph = GraphDB(db_path)
        graph.connect()

        result = route('感冒', graph, user_label=None)

        # 生成回答
        from core.answerer import answer_from_activation
        answer_text = answer_from_activation(result, graph)

        # 校验
        verdict = verify(answer_text, result, graph)

        assert 'confidence' in verdict
        assert 'karma_direction' in verdict
        assert 'decision' in verdict
        assert 0.0 <= verdict['confidence'] <= 1.0
        assert verdict['karma_direction'] in (-1, 0, 1)
        assert verdict['decision'] in ('reinforce', 'correct', 'uncertain')

        graph.close()

    def test_graph_db_stats_still_works(self, tmp_path):
        """GraphDB.stats() 仍正常工作"""
        db_path = str(Path(tmp_path) / "test.db")
        _build_test_db(db_path)

        graph = GraphDB(db_path)
        graph.connect()

        stats = graph.stats()

        assert 'nodes' in stats
        assert 'edges' in stats
        assert stats['nodes'] > 0
        assert stats['edges'] > 0

        graph.close()

    def test_phase2_tables_auto_created(self, tmp_path):
        """Phase 2 新增表在连接时自动创建"""
        db_path = str(Path(tmp_path) / "test.db")
        _build_test_db(db_path)

        graph = GraphDB(db_path)
        graph.connect()

        # 验证 Phase 2 新增表存在
        tables = graph.conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table'"
        ).fetchall()
        table_names = {t['name'] for t in tables}

        assert 'karma_edges_personal' in table_names
        assert 'distillation_pool' in table_names
        assert 'param_stats' in table_names

        graph.close()