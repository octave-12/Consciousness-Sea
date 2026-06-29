"""
测试 GraphDB — 使用内存数据库模拟
"""

import sqlite3
import json
import sys
import pathlib
_root = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_root))
sys.path.insert(0, str(_root / "backend" / "src"))

from consciousness_sea.domain.graph_db import GraphDB


def _setup_test_db():
    """创建测试用内存数据库，带 seeds 和 karma_edges 表"""
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

    # 插入测试数据
    seeds = [
        ('感冒', '感冒', 'CONCEPT', '[]', '医学', 'to catch cold; (common) cold'),
        ('发热', '发热', 'CONCEPT', '[]', '医学', 'fever'),
        ('咳嗽', '咳嗽', 'CONCEPT', '[]', '医学', 'cough'),
        ('维C', '维C', 'CONCEPT', '[]', '营养', 'Vitamin C'),
        ('量子力学', '量子力学', 'CONCEPT', '[]', '物理', 'quantum mechanics'),
        ('薛定谔方程', '薛定谔方程', 'CONCEPT', '[]', '物理', 'Schrodinger equation'),
        ('人工智能', '人工智能', 'CONCEPT', '[]', '计算机', 'artificial intelligence'),
        ('深度学习', '深度学习', 'CONCEPT', '[]', '计算机', 'deep learning'),
        ('苏轼', '苏轼', 'CONCEPT', '[]', '文学', 'Su Shi (1037-1101)'),
        ('电脑', '电脑', 'CONCEPT', '["计算机"]', '计算机', 'computer'),
    ]
    conn.executemany(
        "INSERT INTO seeds (id,label,type,aliases,domain,definition) VALUES (?,?,?,?,?,?)",
        seeds
    )

    edges = [
        ('感冒', '发热', 'COOCCURS_WITH', 0.95),
        ('感冒', '咳嗽', 'COOCCURS_WITH', 0.90),
        ('感冒', '维C', 'RELATED', 0.70),
        ('量子力学', '薛定谔方程', 'IS_A', 0.85),
        ('人工智能', '深度学习', 'IS_A', 0.90),
        ('感冒', '量子力学', 'RELATED', 0.10),  # 弱关联
    ]
    conn.executemany(
        "INSERT INTO karma_edges (source,target,relation,weight) VALUES (?,?,?,?)",
        edges
    )
    conn.commit()
    return conn


class TestGraphDB:
    """GraphDB 单元测试"""

    def setup_method(self):
        self.real_conn = _setup_test_db()
        self.db = GraphDB(':memory:')
        self.db.conn = self.real_conn

    def teardown_method(self):
        self.db.close()

    def test_get_seed_exact(self):
        seed = self.db.get_seed('感冒')
        assert seed is not None
        assert seed['label'] == '感冒'
        assert seed['domain'] == '医学'

    def test_get_seed_alias(self):
        seed = self.db.get_seed('计算机')  # 电脑的别名
        assert seed is not None
        assert seed['label'] == '电脑'

    def test_get_seed_not_found(self):
        seed = self.db.get_seed('不存在的概念')
        assert seed is None

    def test_match_seeds_simple(self):
        seeds = self.db.match_seeds('感冒了吃什么药')
        labels = {s['label'] for s in seeds}
        assert '感冒' in labels

    def test_match_seeds_complex(self):
        # Phase 0 tokenizer: 整体中文 span + 2-gram 拆分
        # "感冒发热" → 整体匹配 "感冒" 或 "发热"（都不对，但 2-gram "感冒"和"发热"能匹配）
        seeds = self.db.match_seeds('感冒发热')
        labels = {s['label'] for s in seeds}
        assert '感冒' in labels or '发热' in labels

    def test_outgoing_edges(self):
        edges = self.db.outgoing_edges('感冒')
        assert len(edges) >= 3  # 发热, 咳嗽, 维C, 量子力学(弱)
        relations = {e['relation'] for e in edges}
        assert 'COOCCURS_WITH' in relations

    def test_adjust_karma_new(self):
        self.db.adjust_karma('感冒', '新概念', 'RELATED', 0.01)
        edge = self.db.get_edge('感冒', '新概念', 'RELATED')
        assert edge is not None
        assert 0.50 < edge['weight'] < 0.52

    def test_adjust_karma_existing(self):
        self.db.adjust_karma('感冒', '发热', 'COOCCURS_WITH', 0.01)
        edge = self.db.get_edge('感冒', '发热', 'COOCCURS_WITH')
        assert edge['weight'] > 0.95  # 0.95 + 0.01

    def test_stats(self):
        s = self.db.stats()
        assert s['nodes'] == 10
        assert s['edges'] >= 6


if __name__ == '__main__':
    # 简单运行
    t = TestGraphDB()
    for name in dir(t):
        if name.startswith('test_'):
            t.setup_method()
            try:
                getattr(t, name)()
                print(f"  ✓ {name}")
            except Exception as e:
                print(f"  ✗ {name}: {e}")
            t.teardown_method()
