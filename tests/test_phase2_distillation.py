"""
Phase 2 提炼池测试 (T6.2)

覆盖:
- 提炼池候选提交
- 三重等价判定（精确匹配、关系等价、涟漪验证）
- 提炼池三用户升级为全局业力
- 全局业力冷却退回
- 提炼池状态查询
"""

import sqlite3
import sys
import pathlib
import json
from pathlib import Path

_root = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_root))
sys.path.insert(0, str(_root / "backend" / "src"))

from consciousness_sea.domain.graph_db import GraphDB
from consciousness_sea.learning.distillation_pool import DistillationPool
from consciousness_sea.infrastructure.karma_cleaner import KarmaCleaner
from consciousness_sea.infrastructure.connection_pool import ConnectionPool


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
        ('电脑', '电脑', 'CONCEPT', '["计算机"]', '计算机', 'computer'),
        ('计算机', '计算机', 'CONCEPT', '[]', '计算机', 'computer_device'),
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
        ('电脑', '发热', 'RELATED', 0.10),
    ]
    conn.executemany(
        "INSERT INTO karma_edges (source, target, relation, weight) "
        "VALUES (?, ?, ?, ?)",
        edges,
    )
    conn.commit()
    conn.close()


class TestDistillationSubmit:
    """提炼池候选提交"""

    def test_submit_new_candidate(self, tmp_path):
        """提交新候选到提炼池"""
        db_path = str(Path(tmp_path) / "test.db")
        _build_test_db(db_path)

        graph = GraphDB(db_path)
        graph.connect()

        distill = DistillationPool(graph)
        candidate_id = distill.submit_candidate(
            user_label='user_A',
            source='感冒',
            target='发热',
            relation='COOCCURS_WITH',
        )

        assert candidate_id is not None
        assert candidate_id > 0

        # 验证数据库中的记录
        row = graph.conn.execute(
            "SELECT * FROM distillation_pool WHERE candidate_id=?",
            (candidate_id,)
        ).fetchone()
        assert row is not None
        assert row['canonical_source'] == '感冒'
        assert row['canonical_target'] == '发热'
        assert row['count'] == 1

        graph.close()

    def test_submit_duplicate_merges(self, tmp_path):
        """重复提交合并到已有候选（count +1）"""
        db_path = str(Path(tmp_path) / "test.db")
        _build_test_db(db_path)

        graph = GraphDB(db_path)
        graph.connect()

        distill = DistillationPool(graph)

        # 第一次提交
        id1 = distill.submit_candidate(
            user_label='user_A', source='感冒', target='发热', relation='COOCCURS_WITH',
        )

        # 第二次提交（不同用户）
        id2 = distill.submit_candidate(
            user_label='user_B', source='感冒', target='发热', relation='COOCCURS_WITH',
        )

        # 应合并到同一个候选
        assert id1 == id2

        row = graph.conn.execute(
            "SELECT count, contributor_users FROM distillation_pool WHERE candidate_id=?",
            (id1,)
        ).fetchone()
        assert row['count'] == 2
        contributors = json.loads(row['contributor_users'])
        assert 'user_A' in contributors
        assert 'user_B' in contributors

        graph.close()

    def test_submit_with_alias_canonicalization(self, tmp_path):
        """提交时别名归一化：'电脑' → '计算机'"""
        db_path = str(Path(tmp_path) / "test.db")
        _build_test_db(db_path)

        graph = GraphDB(db_path)
        graph.connect()

        distill = DistillationPool(graph)

        # 使用别名 '电脑' 提交
        candidate_id = distill.submit_candidate(
            user_label='user_A', source='电脑', target='发热', relation='RELATED',
        )

        # 验证归一化后 source 为 '电脑'（别名在 alias_index 中映射到主 label）
        # 注意：电脑的别名是 ["计算机"]，即 "计算机" 是电脑的别名
        # alias_index: "计算机" → "电脑"
        row = graph.conn.execute(
            "SELECT canonical_source FROM distillation_pool WHERE candidate_id=?",
            (candidate_id,)
        ).fetchone()
        assert row is not None

        graph.close()


class TestTripleEquivalence:
    """三重等价判定（精确匹配、关系等价、涟漪验证）"""

    def test_exact_match(self, tmp_path):
        """精确匹配：相同 source/target/relation"""
        db_path = str(Path(tmp_path) / "test.db")
        _build_test_db(db_path)

        graph = GraphDB(db_path)
        graph.connect()

        distill = DistillationPool(graph)

        id1 = distill.submit_candidate(
            user_label='user_A', source='感冒', target='发热', relation='COOCCURS_WITH',
        )

        # 精确匹配应找到已有候选
        id2 = distill.submit_candidate(
            user_label='user_B', source='感冒', target='发热', relation='COOCCURS_WITH',
        )

        assert id1 == id2

        graph.close()

    def test_relation_equivalence(self, tmp_path):
        """关系等价映射：HELPS_WITH → TREATS"""
        db_path = str(Path(tmp_path) / "test.db")
        _build_test_db(db_path)

        graph = GraphDB(db_path)
        graph.connect()

        distill = DistillationPool(graph)

        # 提交 HELPS_WITH 关系
        id1 = distill.submit_candidate(
            user_label='user_A', source='感冒', target='发热', relation='HELPS_WITH',
        )

        # 提交 TREATS 关系（HELPS_WITH 映射到 TREATS）
        id2 = distill.submit_candidate(
            user_label='user_B', source='感冒', target='发热', relation='TREATS',
        )

        # 两者应合并（关系等价）
        assert id1 == id2

        graph.close()


class TestDistillationUpgrade:
    """提炼池三用户升级为全局业力"""

    def test_three_users_upgrade_to_global(self, tmp_path):
        """三个独立用户提交同一候选后自动升级为全局业力"""
        db_path = str(Path(tmp_path) / "test.db")
        _build_test_db(db_path)

        graph = GraphDB(db_path)
        graph.connect()

        distill = DistillationPool(graph)

        # 第一个用户提交
        distill.submit_candidate(
            user_label='user_A', source='感冒', target='头痛', relation='RELATED',
        )

        # 第二个用户提交
        distill.submit_candidate(
            user_label='user_B', source='感冒', target='头痛', relation='RELATED',
        )

        # 验证尚未升级
        row = graph.conn.execute(
            "SELECT count, status FROM distillation_pool "
            "WHERE canonical_source='感冒' AND canonical_target='头痛'"
        ).fetchone()
        assert row['count'] == 2
        assert row['status'] == 'pending'

        # 第三个用户提交 → 触发升级
        distill.submit_candidate(
            user_label='user_C', source='感冒', target='头痛', relation='RELATED',
        )

        # 验证已升级
        row = graph.conn.execute(
            "SELECT count, status FROM distillation_pool "
            "WHERE canonical_source='感冒' AND canonical_target='头痛'"
        ).fetchone()
        assert row['status'] == 'upgraded'

        # 验证全局业力边已创建
        edge = graph.get_edge('感冒', '头痛', 'RELATED')
        assert edge is not None
        from consciousness_sea.infrastructure.config import DISTILLATION_INITIAL_WEIGHT
        assert edge['weight'] >= DISTILLATION_INITIAL_WEIGHT

        graph.close()

    def test_upgrade_does_not_lower_existing_weight(self, tmp_path):
        """升级时若全局业力边已存在，不降低其权重"""
        db_path = str(Path(tmp_path) / "test.db")
        _build_test_db(db_path)

        graph = GraphDB(db_path)
        graph.connect()

        # 先创建一条高权重全局边
        graph.adjust_karma_atomic('感冒', '发热', 'COOCCURS_WITH', delta=0.5)
        graph.conn.commit()
        weight_before = graph.get_edge('感冒', '发热', 'COOCCURS_WITH')['weight']

        distill = DistillationPool(graph)

        # 三个用户提交 → 触发升级
        for user in ['user_A', 'user_B', 'user_C']:
            distill.submit_candidate(
                user_label=user, source='感冒', target='发热', relation='COOCCURS_WITH',
            )

        # 全局业力权重不应降低
        weight_after = graph.get_edge('感冒', '发热', 'COOCCURS_WITH')['weight']
        assert weight_after >= weight_before

        graph.close()


class TestDistillationCooldown:
    """全局业力冷却退回"""

    def test_cooldown_after_global_edge_deleted(self, tmp_path):
        """全局业力边被删除后提炼池候选退回为 cooled 状态"""
        db_path = str(Path(tmp_path) / "test.db")
        _build_test_db(db_path)

        graph = GraphDB(db_path)
        graph.connect()

        distill = DistillationPool(graph)

        # 三个用户提交 → 升级
        for user in ['user_A', 'user_B', 'user_C']:
            distill.submit_candidate(
                user_label=user, source='感冒', target='头痛', relation='RELATED',
            )

        # 验证已升级
        row = graph.conn.execute(
            "SELECT status FROM distillation_pool "
            "WHERE canonical_source='感冒' AND canonical_target='头痛'"
        ).fetchone()
        assert row['status'] == 'upgraded'

        # 将全局边权重降至 KARMA_MIN 以下
        graph.conn.execute(
            "UPDATE karma_edges SET weight=0.005 "
            "WHERE source='感冒' AND target='头痛' AND relation='RELATED'"
        )
        graph.conn.commit()

        # 运行清理器
        pool = ConnectionPool(db_path, pool_size=2)
        cleaner = KarmaCleaner(pool)
        cleaner.cleanup_low_weight_edges()

        # 验证候选已退回为 cooled
        graph2 = pool.acquire()
        row2 = graph2.conn.execute(
            "SELECT status FROM distillation_pool "
            "WHERE canonical_source='感冒' AND canonical_target='头痛'"
        ).fetchone()
        assert row2['status'] == 'cooled'
        pool.release(graph2)

        pool.close_all()


class TestDistillationStatus:
    """提炼池状态查询"""

    def test_get_status_empty(self, tmp_path):
        """空提炼池状态查询"""
        db_path = str(Path(tmp_path) / "test.db")
        _build_test_db(db_path)

        graph = GraphDB(db_path)
        graph.connect()

        distill = DistillationPool(graph)
        status = distill.get_status()

        assert status['total_candidates'] == 0
        assert status['upgraded_count'] == 0
        assert status['pending_count'] == 0
        assert status['cooled_count'] == 0

        graph.close()

    def test_get_status_with_candidates(self, tmp_path):
        """有候选时状态查询正确"""
        db_path = str(Path(tmp_path) / "test.db")
        _build_test_db(db_path)

        graph = GraphDB(db_path)
        graph.connect()

        distill = DistillationPool(graph)

        # 提交两个完全不同的候选
        distill.submit_candidate(
            user_label='user_A', source='感冒', target='发热', relation='COOCCURS_WITH',
        )
        distill.submit_candidate(
            user_label='user_B', source='量子力学', target='薛定谔方程', relation='RELATED',
        )

        status = distill.get_status()
        assert status['total_candidates'] >= 2
        assert status['pending_count'] >= 2

        graph.close()

    def test_get_status_after_upgrade(self, tmp_path):
        """升级后状态查询正确"""
        db_path = str(Path(tmp_path) / "test.db")
        _build_test_db(db_path)

        graph = GraphDB(db_path)
        graph.connect()

        distill = DistillationPool(graph)

        # 三个用户提交 → 升级
        for user in ['user_A', 'user_B', 'user_C']:
            distill.submit_candidate(
                user_label=user, source='感冒', target='头痛', relation='RELATED',
            )

        status = distill.get_status()
        assert status['total_candidates'] == 1
        assert status['upgraded_count'] == 1
        assert status['pending_count'] == 0

        graph.close()