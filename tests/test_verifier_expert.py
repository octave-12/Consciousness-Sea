"""
T-022: verifier 扩展单元测试

测试 verify() 函数的新增参数和可靠性加权逻辑：
- verify(answer_text, result, graph) 无新参数时行为不变
- reliability=0.6 → confidence = raw_confidence × 0.6
- cv_discount=0.7 → confidence = raw_confidence × 0.7
- reliability=0.6, cv_discount=0.7 → confidence = raw × 0.7 × 0.6
- 返回值包含 raw_confidence 字段
- Phase 0 模式下 raw_confidence == confidence
"""

from __future__ import annotations

import sqlite3
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.graph_db import GraphDB
from core.verifier import verify, _reset_stopwords_cache
from core.router import RippleResult, ActivationNode


def _setup_db():
    """创建测试用内存数据库"""
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
        ('感冒', '感冒', 'CONCEPT', '[]', '医学', 'to catch cold'),
        ('发热', '发热', 'CONCEPT', '[]', '医学', 'fever'),
        ('咳嗽', '咳嗽', 'CONCEPT', '[]', '医学', 'cough'),
        ('量子力学', '量子力学', 'CONCEPT', '[]', '物理', 'quantum mechanics'),
    ]
    conn.executemany(
        "INSERT INTO seeds (id,label,type,aliases,domain,definition) VALUES (?,?,?,?,?,?)",
        seeds,
    )

    edges = [
        ('感冒', '发热', 'COOCCURS_WITH', 0.95),
        ('感冒', '咳嗽', 'COOCCURS_WITH', 0.90),
    ]
    conn.executemany(
        "INSERT INTO karma_edges (source,target,relation,weight) VALUES (?,?,?,?)",
        edges,
    )
    conn.commit()
    return conn


def _make_ripple_result(activated_labels: list[str] | None = None) -> RippleResult:
    """辅助函数：构造 RippleResult 对象"""
    result = RippleResult()
    if activated_labels:
        for label in activated_labels:
            result.activated[label] = ActivationNode(label=label, activation=1.0, depth=0)
    return result


class TestVerifierExpertNoNewParams:
    """无新参数时行为不变测试"""

    def setup_method(self):
        _reset_stopwords_cache()
        self.conn = _setup_db()
        self.db = GraphDB(':memory:')
        self.db.conn = self.conn

    def teardown_method(self):
        self.db.close()
        _reset_stopwords_cache()

    def test_verify_without_new_params_returns_confidence(self):
        """无新参数时返回 confidence"""
        result = _make_ripple_result(['感冒'])
        verdict = verify('感冒', result, self.db)
        assert 'confidence' in verdict
        assert 0.0 <= verdict['confidence'] <= 1.0

    def test_verify_without_new_params_returns_karma_direction(self):
        """无新参数时返回 karma_direction"""
        result = _make_ripple_result(['感冒'])
        verdict = verify('感冒', result, self.db)
        assert 'karma_direction' in verdict
        assert verdict['karma_direction'] in (-1, 0, 1)

    def test_verify_without_new_params_returns_decision(self):
        """无新参数时返回 decision"""
        result = _make_ripple_result(['感冒'])
        verdict = verify('感冒', result, self.db)
        assert 'decision' in verdict
        assert verdict['decision'] in ('reinforce', 'correct', 'uncertain')


class TestVerifierExpertReliability:
    """reliability 加权测试"""

    def setup_method(self):
        _reset_stopwords_cache()
        self.conn = _setup_db()
        self.db = GraphDB(':memory:')
        self.db.conn = self.conn

    def teardown_method(self):
        self.db.close()
        _reset_stopwords_cache()

    def test_reliability_0_6_discounts_confidence(self):
        """reliability=0.6 → confidence = raw_confidence × 0.6"""
        result = _make_ripple_result(['感冒'])
        verdict = verify('感冒', result, self.db, reliability=0.6)
        raw = verdict['raw_confidence']
        actual = verdict['confidence']
        assert abs(actual - raw * 0.6) < 0.01

    def test_reliability_0_85_preserves_confidence(self):
        """reliability=0.85 → confidence = raw_confidence × 0.85"""
        result = _make_ripple_result(['感冒'])
        verdict = verify('感冒', result, self.db, reliability=0.85)
        raw = verdict['raw_confidence']
        actual = verdict['confidence']
        assert abs(actual - raw * 0.85) < 0.01

    def test_reliability_1_0_no_change(self):
        """reliability=1.0 → confidence 不变"""
        result = _make_ripple_result(['感冒'])
        verdict = verify('感冒', result, self.db, reliability=1.0)
        raw = verdict['raw_confidence']
        actual = verdict['confidence']
        assert abs(actual - raw) < 0.01

    def test_reliability_field_in_return(self):
        """返回值包含 reliability 字段"""
        result = _make_ripple_result(['感冒'])
        verdict = verify('感冒', result, self.db, reliability=0.6)
        assert 'reliability' in verdict
        assert verdict['reliability'] == 0.6


class TestVerifierExpertCvDiscount:
    """cv_discount 加权测试"""

    def setup_method(self):
        _reset_stopwords_cache()
        self.conn = _setup_db()
        self.db = GraphDB(':memory:')
        self.db.conn = self.conn

    def teardown_method(self):
        self.db.close()
        _reset_stopwords_cache()

    def test_cv_discount_0_7_discounts_confidence(self):
        """cv_discount=0.7 → confidence = raw_confidence × 0.7"""
        result = _make_ripple_result(['感冒'])
        verdict = verify('感冒', result, self.db, cv_discount=0.7)
        raw = verdict['raw_confidence']
        actual = verdict['confidence']
        assert abs(actual - raw * 0.7) < 0.01

    def test_cv_discount_1_0_no_change(self):
        """cv_discount=1.0 → confidence 不变"""
        result = _make_ripple_result(['感冒'])
        verdict = verify('感冒', result, self.db, cv_discount=1.0)
        raw = verdict['raw_confidence']
        actual = verdict['confidence']
        assert abs(actual - raw) < 0.01

    def test_cv_discount_field_in_return(self):
        """返回值包含 cv_discount 字段"""
        result = _make_ripple_result(['感冒'])
        verdict = verify('感冒', result, self.db, cv_discount=0.7)
        assert 'cv_discount' in verdict
        assert verdict['cv_discount'] == 0.7


class TestVerifierExpertStacking:
    """reliability + cv_discount 叠加测试"""

    def setup_method(self):
        _reset_stopwords_cache()
        self.conn = _setup_db()
        self.db = GraphDB(':memory:')
        self.db.conn = self.conn

    def teardown_method(self):
        self.db.close()
        _reset_stopwords_cache()

    def test_reliability_and_cv_discount_stacking(self):
        """reliability=0.6, cv_discount=0.7 → confidence = raw × 0.7 × 0.6"""
        result = _make_ripple_result(['感冒'])
        verdict = verify('感冒', result, self.db, reliability=0.6, cv_discount=0.7)
        raw = verdict['raw_confidence']
        actual = verdict['confidence']
        expected = raw * 0.7 * 0.6
        assert abs(actual - expected) < 0.01

    def test_stacking_order_cv_then_reliability(self):
        """叠加顺序: 先 cv_discount 后 reliability"""
        result = _make_ripple_result(['感冒'])
        verdict = verify('感冒', result, self.db, reliability=0.6, cv_discount=0.7)
        raw = verdict['raw_confidence']
        # actual_confidence = raw × cv_discount × reliability
        expected = raw * 0.7 * 0.6
        assert abs(verdict['confidence'] - expected) < 0.01

    def test_stacking_with_expert_domain(self):
        """叠加 + expert_domain"""
        result = _make_ripple_result(['感冒'])
        verdict = verify(
            '感冒', result, self.db,
            expert_domain="医学",
            reliability=0.85,
            cv_discount=0.7,
        )
        raw = verdict['raw_confidence']
        expected = raw * 0.7 * 0.85
        assert abs(verdict['confidence'] - expected) < 0.01
        assert verdict['expert_domain'] == "医学"


class TestVerifierExpertRawConfidence:
    """raw_confidence 字段测试"""

    def setup_method(self):
        _reset_stopwords_cache()
        self.conn = _setup_db()
        self.db = GraphDB(':memory:')
        self.db.conn = self.conn

    def teardown_method(self):
        self.db.close()
        _reset_stopwords_cache()

    def test_raw_confidence_field_exists(self):
        """返回值包含 raw_confidence 字段"""
        result = _make_ripple_result(['感冒'])
        verdict = verify('感冒', result, self.db)
        assert 'raw_confidence' in verdict

    def test_phase0_raw_confidence_equals_confidence(self):
        """Phase 0 模式下 raw_confidence == confidence"""
        result = _make_ripple_result(['感冒'])
        verdict = verify('感冒', result, self.db)
        assert verdict['raw_confidence'] == verdict['confidence']

    def test_phase1_raw_confidence_differs_from_confidence(self):
        """Phase 1 模式下 raw_confidence != confidence（当 reliability < 1.0）"""
        result = _make_ripple_result(['感冒'])
        verdict = verify('感冒', result, self.db, reliability=0.6)
        assert verdict['raw_confidence'] != verdict['confidence']
        assert verdict['raw_confidence'] > verdict['confidence']

    def test_raw_confidence_range(self):
        """raw_confidence 在 [0, 1] 范围内"""
        result = _make_ripple_result(['感冒'])
        verdict = verify('感冒', result, self.db, reliability=0.6)
        assert 0.0 <= verdict['raw_confidence'] <= 1.0


class TestVerifierExpertDomain:
    """expert_domain 字段测试"""

    def setup_method(self):
        _reset_stopwords_cache()
        self.conn = _setup_db()
        self.db = GraphDB(':memory:')
        self.db.conn = self.conn

    def teardown_method(self):
        self.db.close()
        _reset_stopwords_cache()

    def test_expert_domain_field_in_return(self):
        """返回值包含 expert_domain 字段"""
        result = _make_ripple_result(['感冒'])
        verdict = verify('感冒', result, self.db, expert_domain="医学")
        assert 'expert_domain' in verdict
        assert verdict['expert_domain'] == "医学"

    def test_expert_domain_default_none(self):
        """默认 expert_domain=None"""
        result = _make_ripple_result(['感冒'])
        verdict = verify('感冒', result, self.db)
        assert verdict['expert_domain'] is None


if __name__ == '__main__':
    import pytest
    pytest.main([__file__, '-v'])