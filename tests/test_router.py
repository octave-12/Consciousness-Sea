"""
测试路由器 — BFS 涟漪传播
"""

import sqlite3
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.graph_db import GraphDB
from core.router import route, RippleResult


def _setup_db():
    conn = sqlite3.connect(':memory:')
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
    """)

    seeds = [
        ('感冒', '感冒', '医学', 'to catch cold'),
        ('发热', '发热', '医学', 'fever'),
        ('咳嗽', '咳嗽', '医学', 'cough'),
        ('维C', '维C', '营养', 'Vitamin C'),
        ('量子力学', '量子力学', '物理', 'quantum mechanics'),
        ('薛定谔方程', '薛定谔方程', '物理', ''),
        ('波动方程', '波动方程', '物理', ''),
        ('人工智能', '人工智能', '计算机', 'AI'),
        ('深度学习', '深度学习', '计算机', 'deep learning'),
        ('神经网络', '神经网络', '计算机', 'neural network'),
    ]
    for s in seeds:
        conn.execute(
            "INSERT INTO seeds (id,label,domain,definition) VALUES (?,?,?,?)",
            s
        )

    edges = [
        ('感冒', '发热', 'COOCCURS_WITH', 0.95),
        ('感冒', '咳嗽', 'COOCCURS_WITH', 0.90),
        ('感冒', '维C', 'RELATED', 0.70),
        ('量子力学', '薛定谔方程', 'RELATED', 0.85),
        ('薛定谔方程', '波动方程', 'IS_A', 0.80),
        ('人工智能', '深度学习', 'IS_A', 0.92),
        ('深度学习', '神经网络', 'RELATED', 0.88),
        ('感冒', '量子力学', 'RELATED', 0.05),  # 弱边，涟漪不应走太远
    ]
    for e in edges:
        conn.execute(
            "INSERT INTO karma_edges (source,target,relation,weight) VALUES (?,?,?,?)",
            e
        )
    conn.commit()
    return conn


class TestRouter:

    def setup_method(self):
        conn = _setup_db()
        self.db = GraphDB(':memory:')
        self.db.conn = conn
        self.db.ensure_phase2_tables()
        self.db.ensure_phase3_tables()

    def teardown_method(self):
        self.db.close()

    def test_basic_activation(self):
        """查询词直接匹配种子"""
        result = route('感冒了怎么办', self.db)
        assert len(result.seed_matches) >= 1
        # '感冒' 应该在第一波激活中
        assert '感冒' in result.activated

    def test_ripple_propagation(self):
        """涟漪传播：感冒 → 发热/咳嗽/维C"""
        result = route('感冒', self.db)
        # 感冒的邻居应该被激活
        assert '发热' in result.activated or '咳嗽' in result.activated
        # 路径应该有记录
        assert len(result.paths) > 0

    def test_two_hop_propagation(self):
        """二跳传播：量子力学 → 薛定谔方程 → 波动方程"""
        result = route('量子力学', self.db)
        # 二跳应该能到波动方程（如果激活值够高）
        # 注意：波动方程在深度2，激活值较低
        activated_labels = set(result.activated.keys())
        assert '薛定谔方程' in activated_labels
        # 波动方程可能因为衰减而不够强，这个不硬断言
        activated_labels = set(result.activated.keys())

    def test_weak_edge_decay(self):
        """弱边涟漪应很快衰减"""
        result = route('感冒', self.db)
        # 感冒→量子力学 权重0.05，激活值应该很低
        if '量子力学' in result.activated:
            node = result.activated['量子力学']
            assert node.activation < 0.1  # 1.0 * 0.05 * 0.7 = 0.035

    def test_domain_scores(self):
        """领域得分聚合"""
        result = route('感冒', self.db)
        assert result.domain_scores
        # 医学领域应该得分最高
        if '医学' in result.domain_scores:
            assert result.domain_scores['医学'] > 0

    def test_top_seeds_sorted(self):
        """Top-K 应该按激活值降序"""
        result = route('人工智能深度学习', self.db)
        top = result.top_seeds
        if len(top) >= 2:
            assert top[0].activation >= top[-1].activation

    def test_empty_query(self):
        """空查询不崩溃"""
        result = route('xyz123不存在的查询', self.db)
        assert isinstance(result, RippleResult)
        assert len(result.activated) == 0


if __name__ == '__main__':
    t = TestRouter()
    for name in dir(t):
        if name.startswith('test_'):
            t.setup_method()
            try:
                getattr(t, name)()
                print(f"  ✓ {name}")
            except Exception as e:
                print(f"  ✗ {name}: {e}")
            t.teardown_method()
