"""
Phase 2 双层业力架构测试 (T6.2)

覆盖:
- 个人业力层隔离写入：用户 A 的熏习不影响用户 B
- 涟漪传播业力叠加：ripple_weight = global × 0.7 + personal × 0.3
- 新用户纯全局业力：personal_weight=0 时 ripple_weight = global × 0.7
- apply_karma() 传入 user_label 时走个人业力路径
- 用户删除后个人业力清理
"""

import sqlite3
import sys
import os
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.graph_db import GraphDB
from core.verifier import apply_karma
from core.router import route, RippleResult, ActivationNode
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

    seeds = [
        ('感冒', '感冒', 'CONCEPT', '[]', '医学', 'cold'),
        ('发热', '发热', 'CONCEPT', '[]', '医学', 'fever'),
        ('咳嗽', '咳嗽', 'CONCEPT', '[]', '医学', 'cough'),
        ('头痛', '头痛', 'CONCEPT', '[]', '医学', 'headache'),
    ]
    conn.executemany(
        "INSERT INTO seeds (id, label, type, aliases, domain, definition) "
        "VALUES (?, ?, ?, ?, ?, ?)",
        seeds,
    )

    edges = [
        ('感冒', '发热', 'COOCCURS_WITH', 0.95),
        ('感冒', '咳嗽', 'COOCCURS_WITH', 0.90),
        ('头痛', '感冒', 'RELATED', 0.50),
    ]
    conn.executemany(
        "INSERT INTO karma_edges (source, target, relation, weight) "
        "VALUES (?, ?, ?, ?)",
        edges,
    )
    conn.commit()
    conn.close()


class TestPersonalKarmaIsolation:
    """个人业力层隔离写入：用户 A 的熏习不影响用户 B"""

    def test_user_a_karma_not_affect_user_b(self, tmp_path):
        """用户 A 的个人业力写入不影响用户 B 的个人业力"""
        db_path = str(Path(tmp_path) / "test.db")
        _build_test_db(db_path)

        graph = GraphDB(db_path)
        graph.connect()

        # 用户 A 熏习
        graph.adjust_karma_personal('user_A', '感冒', '发热', 'COOCCURS_WITH', delta=0.1)
        graph.conn.commit()

        # 用户 B 熏习不同权重
        graph.adjust_karma_personal('user_B', '感冒', '发热', 'COOCCURS_WITH', delta=-0.05)
        graph.conn.commit()

        # 验证两个用户的个人业力独立
        weight_a = graph.get_personal_weight('user_A', '感冒', '发热', 'COOCCURS_WITH')
        weight_b = graph.get_personal_weight('user_B', '感冒', '发热', 'COOCCURS_WITH')

        assert weight_a is not None
        assert weight_b is not None
        # user_A: 0.5 + 0.1 = 0.6
        assert abs(weight_a - 0.6) < 0.01
        # user_B: 0.5 - 0.05 = 0.45
        assert abs(weight_b - 0.45) < 0.01

        graph.close()

    def test_user_a_karma_not_affect_global(self, tmp_path):
        """用户 A 的个人业力写入不影响全局业力"""
        db_path = str(Path(tmp_path) / "test.db")
        _build_test_db(db_path)

        graph = GraphDB(db_path)
        graph.connect()

        # 记录全局业力初始值
        global_before = graph.get_edge('感冒', '发热', 'COOCCURS_WITH')
        weight_before = global_before['weight']

        # 用户 A 熏习个人层
        graph.adjust_karma_personal('user_A', '感冒', '发热', 'COOCCURS_WITH', delta=0.1)
        graph.conn.commit()

        # 全局业力应不变
        global_after = graph.get_edge('感冒', '发热', 'COOCCURS_WITH')
        assert global_after['weight'] == weight_before

        graph.close()

    def test_batch_get_personal_weights(self, tmp_path):
        """批量获取个人业力权重"""
        db_path = str(Path(tmp_path) / "test.db")
        _build_test_db(db_path)

        graph = GraphDB(db_path)
        graph.connect()

        # 为用户 A 创建多条个人业力
        graph.adjust_karma_personal('user_A', '感冒', '发热', 'COOCCURS_WITH', delta=0.1)
        graph.adjust_karma_personal('user_A', '感冒', '咳嗽', 'COOCCURS_WITH', delta=0.2)
        graph.conn.commit()

        # 批量查询
        weights = graph.batch_get_personal_weights('user_A', ['感冒'])

        assert ('感冒', '发热', 'COOCCURS_WITH') in weights
        assert ('感冒', '咳嗽', 'COOCCURS_WITH') in weights

        graph.close()


class TestRippleWeightSuperposition:
    """涟漪传播业力叠加：ripple_weight = global × 0.7 + personal × 0.3"""

    def test_dual_weight_superposition(self, tmp_path):
        """双层权重叠加计算正确"""
        db_path = str(Path(tmp_path) / "test.db")
        _build_test_db(db_path)

        graph = GraphDB(db_path)
        graph.connect()

        # 为用户 A 创建个人业力
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

        # 验证叠加权重
        global_w = 0.95  # 全局权重
        personal_w = 0.8  # 0.5 + 0.3
        expected_weight = global_w * 0.7 + personal_w * 0.3

        assert abs(path['weight'] - expected_weight) < 0.01

        graph.close()

    def test_no_personal_weight_uses_global_only(self, tmp_path):
        """没有个人业力时使用纯全局权重"""
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

        # 无个人业力时，personal_w=0
        # weight = global_w * 0.7 + 0 * 0.3 = global_w * 0.7
        global_w = 0.95
        expected_weight = global_w * 0.7

        assert abs(path['weight'] - expected_weight) < 0.01

        graph.close()


class TestNewUserPureGlobalKarma:
    """新用户纯全局业力：personal_weight=0 时 ripple_weight = global × 0.7"""

    def test_new_user_ripple_weight(self, tmp_path):
        """新用户没有个人业力，ripple_weight = global × 0.7"""
        db_path = str(Path(tmp_path) / "test.db")
        _build_test_db(db_path)

        graph = GraphDB(db_path)
        graph.connect()

        # 新用户执行路由
        result = route('感冒', graph, user_label='new_user')

        # 查找感冒→发热的路径
        path = None
        for p in result.paths:
            if p['source'] == '感冒' and p['target'] == '发热':
                path = p
                break

        assert path is not None

        # 新用户没有个人业力，personal_w=0
        global_w = 0.95
        expected_weight = global_w * 0.7

        assert abs(path['weight'] - expected_weight) < 0.01

        graph.close()

    def test_new_user_no_personal_edges(self, tmp_path):
        """新用户在个人业力表中没有记录"""
        db_path = str(Path(tmp_path) / "test.db")
        _build_test_db(db_path)

        graph = GraphDB(db_path)
        graph.connect()

        # 新用户查询个人业力
        weight = graph.get_personal_weight('new_user', '感冒', '发热', 'COOCCURS_WITH')
        assert weight is None

        graph.close()


class TestApplyKarmaWithUserLabel:
    """apply_karma() 传入 user_label 时走个人业力路径"""

    def test_apply_karma_with_user_label_writes_personal(self, tmp_path):
        """传入 user_label 时熏习写入个人业力层"""
        db_path = str(Path(tmp_path) / "test.db")
        _build_test_db(db_path)

        graph = GraphDB(db_path)
        graph.connect()

        # 构造 RippleResult
        result = RippleResult()
        result.query = '感冒'
        result.activated['感冒'] = ActivationNode(
            label='感冒', activation=1.0, domain='医学', definition='cold', depth=0
        )
        result.activated['发热'] = ActivationNode(
            label='发热', activation=0.7, domain='医学', definition='fever', depth=1
        )
        result.paths.append({
            'source': '感冒', 'target': '发热', 'relation': 'COOCCURS_WITH',
            'weight': 0.95, 'depth': 1, 'ripple_activation': 0.7,
        })

        # 记录全局业力初始值
        global_before = graph.get_edge('感冒', '发热', 'COOCCURS_WITH')['weight']

        # 使用 user_label 熏习
        apply_karma(result, graph, karma_direction=+1, user_label='user_A')

        # 个人业力应有记录
        personal_w = graph.get_personal_weight('user_A', '感冒', '发热', 'COOCCURS_WITH')
        assert personal_w is not None

        # 全局业力应不变
        global_after = graph.get_edge('感冒', '发热', 'COOCCURS_WITH')
        assert global_after['weight'] == global_before

        graph.close()

    def test_apply_karma_without_user_label_writes_global(self, tmp_path):
        """不传 user_label 时熏习写入全局业力层"""
        db_path = str(Path(tmp_path) / "test.db")
        _build_test_db(db_path)

        graph = GraphDB(db_path)
        graph.connect()

        # 构造 RippleResult
        result = RippleResult()
        result.query = '感冒'
        result.activated['感冒'] = ActivationNode(
            label='感冒', activation=1.0, domain='医学', definition='cold', depth=0
        )
        result.activated['发热'] = ActivationNode(
            label='发热', activation=0.7, domain='医学', definition='fever', depth=1
        )
        result.paths.append({
            'source': '感冒', 'target': '发热', 'relation': 'COOCCURS_WITH',
            'weight': 0.95, 'depth': 1, 'ripple_activation': 0.7,
        })

        # 记录全局业力初始值
        global_before = graph.get_edge('感冒', '发热', 'COOCCURS_WITH')['weight']

        # 不传 user_label 熏习
        apply_karma(result, graph, karma_direction=+1, user_label=None)

        # 全局业力应改变
        global_after = graph.get_edge('感冒', '发热', 'COOCCURS_WITH')
        assert global_after['weight'] != global_before

        graph.close()


class TestUserDeletionCleanup:
    """用户删除后个人业力清理"""

    def test_delete_user_personal_karma(self, tmp_path):
        """删除用户后其个人业力记录被清理"""
        db_path = str(Path(tmp_path) / "test.db")
        _build_test_db(db_path)

        graph = GraphDB(db_path)
        graph.connect()

        # 创建用户 A 的个人业力
        graph.adjust_karma_personal('user_A', '感冒', '发热', 'COOCCURS_WITH', delta=0.1)
        graph.adjust_karma_personal('user_A', '感冒', '咳嗽', 'COOCCURS_WITH', delta=0.2)
        graph.conn.commit()

        # 验证个人业力存在
        w1 = graph.get_personal_weight('user_A', '感冒', '发热', 'COOCCURS_WITH')
        assert w1 is not None

        # 删除用户 A 的个人业力
        graph.conn.execute(
            "DELETE FROM karma_edges_personal WHERE user_label = ?",
            ('user_A',)
        )
        graph.conn.commit()

        # 验证个人业力已清理
        w2 = graph.get_personal_weight('user_A', '感冒', '发热', 'COOCCURS_WITH')
        assert w2 is None

        w3 = graph.get_personal_weight('user_A', '感冒', '咳嗽', 'COOCCURS_WITH')
        assert w3 is None

        graph.close()

    def test_delete_user_does_not_affect_other_users(self, tmp_path):
        """删除用户 A 不影响用户 B 的个人业力"""
        db_path = str(Path(tmp_path) / "test.db")
        _build_test_db(db_path)

        graph = GraphDB(db_path)
        graph.connect()

        # 创建两个用户的个人业力
        graph.adjust_karma_personal('user_A', '感冒', '发热', 'COOCCURS_WITH', delta=0.1)
        graph.adjust_karma_personal('user_B', '感冒', '发热', 'COOCCURS_WITH', delta=0.2)
        graph.conn.commit()

        # 删除用户 A 的个人业力
        graph.conn.execute(
            "DELETE FROM karma_edges_personal WHERE user_label = ?",
            ('user_A',)
        )
        graph.conn.commit()

        # 用户 B 的个人业力应不受影响
        w_b = graph.get_personal_weight('user_B', '感冒', '发热', 'COOCCURS_WITH')
        assert w_b is not None
        assert abs(w_b - 0.7) < 0.01  # 0.5 + 0.2

        graph.close()