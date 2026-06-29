"""
测试校验器 + 熏习引擎
"""

import sqlite3
import sys
import pathlib
_root = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_root))
sys.path.insert(0, str(_root / "backend" / "src"))

from consciousness_sea.domain.graph_db import GraphDB
from consciousness_sea.domain.router import route
from consciousness_sea.domain.answerer import answer_from_activation
from consciousness_sea.domain.verifier import verify, apply_karma


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
        ('量子力学', '量子力学', '物理', 'quantum mechanics'),
    ]
    for s in seeds:
        conn.execute("INSERT INTO seeds (id,label,domain,definition) VALUES (?,?,?,?)", s)

    edges = [
        ('感冒', '发热', 'COOCCURS_WITH', 0.95),
        ('感冒', '咳嗽', 'COOCCURS_WITH', 0.90),
    ]
    for e in edges:
        conn.execute(
            "INSERT INTO karma_edges (source,target,relation,weight) VALUES (?,?,?,?)",
            e
        )
    conn.commit()
    return conn


class TestVerifier:

    def setup_method(self):
        conn = _setup_db()
        self.db = GraphDB(':memory:')
        self.db.conn = conn
        self.db.ensure_phase2_tables()
        self.db.ensure_phase3_tables()

    def teardown_method(self):
        self.db.close()

    def test_high_confidence(self):
        """回答关键词在激活区域内 → 高置信度"""
        result = route('感冒', self.db)
        answer = answer_from_activation(result, self.db)
        verdict = verify(answer, result, self.db)
        assert verdict['decision'] == 'reinforce'
        assert verdict['karma_direction'] == 1
        assert verdict['confidence'] >= 0.7

    def test_low_confidence(self):
        """回答关键词不在激活区域 → 低置信度"""
        result = route('感冒', self.db)
        # 伪造一个不相关的回答
        verdict = verify('黑洞 引力 相对论', result, self.db)
        assert verdict['decision'] == 'correct'
        assert verdict['karma_direction'] == -1

    def test_empty_answer(self):
        result = route('感冒', self.db)
        verdict = verify('', result, self.db)
        assert verdict['confidence'] == 0.5
        assert verdict['karma_direction'] == 0

    def test_apply_karma_reinforce(self):
        """正向熏习：边权重 +0.01"""
        result = route('感冒', self.db)
        n = apply_karma(result, self.db, karma_direction=1, dry_run=False)
        assert n > 0  # 传播路径上的边都被熏习

        # 验证边已被修改
        edge = self.db.get_edge('感冒', '发热', 'COOCCURS_WITH')
        assert edge['weight'] > 0.95

    def test_apply_karma_correct(self):
        """负向熏习：边权重 -0.01"""
        result = route('感冒', self.db)
        n = apply_karma(result, self.db, karma_direction=-1, dry_run=False)
        assert n > 0

        edge = self.db.get_edge('感冒', '发热', 'COOCCURS_WITH')
        assert edge['weight'] < 0.95

    def test_apply_karma_dry_run(self):
        """Dry run 不修改数据"""
        result = route('感冒', self.db)
        original = self.db.get_edge('感冒', '发热', 'COOCCURS_WITH')['weight']
        n = apply_karma(result, self.db, karma_direction=1, dry_run=True)
        assert n > 0
        after = self.db.get_edge('感冒', '发热', 'COOCCURS_WITH')['weight']
        assert original == after  # dry run 不应修改

    def test_end_to_end(self):
        """端到端：查询 → 路由 → 回答 → 校验 → 熏习"""
        result = route('感冒', self.db)
        answer = answer_from_activation(result, self.db)
        verdict = verify(answer, result, self.db)
        n = apply_karma(result, self.db, verdict['karma_direction'])

        assert result.selected_domains  # 有选定领域
        assert len(answer) > 0  # 有回答
        assert verdict['confidence'] > 0  # 有置信度
        assert n > 0  # 有熏习


if __name__ == '__main__':
    t = TestVerifier()
    for name in dir(t):
        if name.startswith('test_'):
            t.setup_method()
            try:
                getattr(t, name)()
                print(f"  ✓ {name}")
            except Exception as e:
                import traceback
                print(f"  ✗ {name}: {e}")
                traceback.print_exc()
            t.teardown_method()
