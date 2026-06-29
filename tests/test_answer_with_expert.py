"""
T-023: answer_with_expert 集成测试

测试 answer_with_expert() 的完整流程：
- 专家可用时返回 expert_answer 和 retrieval_answer
- 专家不可用时降级到检索式回答
- 空激活时不调用专家推理
- 多领域时执行交叉验证
- 返回字典包含所有必要字段
- 使用 MockExpertManager，无 GPU 依赖
"""

from __future__ import annotations

import sqlite3
import sys
import pathlib

_root = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_root))
sys.path.insert(0, str(_root / "backend" / "src"))

from consciousness_sea.domain.answerer import answer_with_expert, answer_from_activation
from consciousness_sea.domain.router import RippleResult, ActivationNode
from consciousness_sea.domain.graph_db import GraphDB
from consciousness_sea.expert.cross_validator import CrossValidationStatus

# 导入 conftest 中的 MockExpertManager
from tests.conftest import MockExpertManager


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
        ('感冒', '感冒', 'CONCEPT', '[]', '医学', '急性上呼吸道感染'),
        ('发热', '发热', 'CONCEPT', '[]', '医学', '体温升高的生理反应'),
        ('咳嗽', '咳嗽', 'CONCEPT', '[]', '医学', 'cough'),
        ('维C', '维C', 'CONCEPT', '[]', '营养', 'Vitamin C'),
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


def _make_ripple_result(
    seeds: list[tuple[str, float, str, str]] | None = None,
    domain_scores: dict[str, float] | None = None,
    paths: list[dict] | None = None,
    query: str = "感冒了吃什么",
) -> RippleResult:
    """构造测试用 RippleResult"""
    result = RippleResult()
    result.query = query

    if seeds:
        for label, activation, domain, definition in seeds:
            result.activated[label] = ActivationNode(
                label=label,
                activation=activation,
                domain=domain,
                definition=definition,
                depth=0,
            )

    if domain_scores:
        result.domain_scores = domain_scores

    if paths:
        result.paths = paths

    return result


class TestAnswerWithExpertAvailable:
    """专家可用时测试"""

    def setup_method(self):
        self.conn = _setup_db()
        self.db = GraphDB(':memory:')
        self.db.conn = conn = self.conn
        self.db.ensure_phase2_tables()
        self.db.ensure_phase3_tables()
        self.mock_manager = MockExpertManager(available=True, answer="感冒是常见的呼吸道疾病，建议多休息")

    def teardown_method(self):
        self.db.close()

    def test_expert_available_returns_expert_answer(self):
        """专家可用时返回 expert_answer"""
        result = _make_ripple_result(
            seeds=[("感冒", 0.9, "医学", "急性上呼吸道感染")],
            domain_scores={"医学": 0.85},
        )

        answer_result = answer_with_expert(result, self.db, self.mock_manager)

        assert answer_result['expert_answer'] is not None
        assert len(answer_result['expert_answer']) > 0

    def test_expert_available_returns_retrieval_answer(self):
        """专家可用时同时返回 retrieval_answer"""
        result = _make_ripple_result(
            seeds=[("感冒", 0.9, "医学", "急性上呼吸道感染")],
            domain_scores={"医学": 0.85},
        )

        answer_result = answer_with_expert(result, self.db, self.mock_manager)

        assert answer_result['retrieval_answer'] is not None
        assert len(answer_result['retrieval_answer']) > 0

    def test_expert_available_flag_is_true(self):
        """专家可用时 expert_available=True"""
        result = _make_ripple_result(
            seeds=[("感冒", 0.9, "医学", "急性上呼吸道感染")],
            domain_scores={"医学": 0.85},
        )

        answer_result = answer_with_expert(result, self.db, self.mock_manager)

        assert answer_result['expert_available'] is True

    def test_expert_available_returns_domain(self):
        """专家可用时返回 expert_domain"""
        result = _make_ripple_result(
            seeds=[("感冒", 0.9, "医学", "急性上呼吸道感染")],
            domain_scores={"医学": 0.85},
        )

        answer_result = answer_with_expert(result, self.db, self.mock_manager)

        assert answer_result['expert_domain'] is not None


class TestAnswerWithExpertUnavailable:
    """专家不可用时测试"""

    def setup_method(self):
        self.conn = _setup_db()
        self.db = GraphDB(':memory:')
        self.db.conn = self.conn
        self.mock_manager = MockExpertManager(available=False)

    def teardown_method(self):
        self.db.close()

    def test_expert_unavailable_falls_back_to_retrieval(self):
        """专家不可用时降级到检索式回答"""
        result = _make_ripple_result(
            seeds=[("感冒", 0.9, "医学", "急性上呼吸道感染")],
            domain_scores={"医学": 0.85},
        )

        answer_result = answer_with_expert(result, self.db, self.mock_manager)

        assert answer_result['expert_answer'] is None
        assert answer_result['retrieval_answer'] is not None
        assert answer_result['expert_available'] is False

    def test_expert_unavailable_with_none_manager(self):
        """expert_manager 为 None 时降级"""
        result = _make_ripple_result(
            seeds=[("感冒", 0.9, "医学", "急性上呼吸道感染")],
            domain_scores={"医学": 0.85},
        )

        answer_result = answer_with_expert(result, self.db, None)

        assert answer_result['expert_answer'] is None
        assert answer_result['expert_available'] is False


class TestAnswerWithExpertEmptyActivation:
    """空激活测试"""

    def setup_method(self):
        self.conn = _setup_db()
        self.db = GraphDB(':memory:')
        self.db.conn = self.conn
        self.mock_manager = MockExpertManager(available=True)

    def teardown_method(self):
        self.db.close()

    def test_empty_activation_does_not_call_expert(self):
        """空激活时不调用专家推理"""
        result = _make_ripple_result(
            seeds=None,
            domain_scores={},
        )

        answer_result = answer_with_expert(result, self.db, self.mock_manager)

        # 空激活 → expert_available=False（降级）
        assert answer_result['expert_available'] is False
        assert answer_result['expert_answer'] is None

        # 验证 mock 未被调用
        assert len(self.mock_manager._infer_calls) == 0

    def test_empty_activation_returns_retrieval_answer(self):
        """空激活时返回检索式回答"""
        result = _make_ripple_result(
            seeds=None,
            domain_scores={},
        )

        answer_result = answer_with_expert(result, self.db, self.mock_manager)

        assert answer_result['retrieval_answer'] is not None


class TestAnswerWithExpertMultiDomain:
    """多领域交叉验证测试"""

    def setup_method(self):
        self.conn = _setup_db()
        self.db = GraphDB(':memory:')
        self.db.conn = self.conn
        self.mock_manager = MockExpertManager(available=True, answer="专家回答")

    def teardown_method(self):
        self.db.close()

    def test_multi_domain_executes_cross_validation(self):
        """多领域时执行交叉验证"""
        result = _make_ripple_result(
            seeds=[
                ("感冒", 0.9, "医学", "急性上呼吸道感染"),
                ("维C", 0.6, "营养", "Vitamin C"),
            ],
            domain_scores={"医学": 0.85, "营养": 0.5},
        )

        answer_result = answer_with_expert(result, self.db, self.mock_manager)

        # 多领域 → 应有交叉验证状态
        assert answer_result['cross_validation_status'] is not None

    def test_single_domain_no_cross_validation(self):
        """单领域时无交叉验证"""
        result = _make_ripple_result(
            seeds=[("感冒", 0.9, "医学", "急性上呼吸道感染")],
            domain_scores={"医学": 0.85},
        )

        answer_result = answer_with_expert(result, self.db, self.mock_manager)

        assert answer_result['cross_validation_status'] == 'none'


class TestAnswerWithExpertReturnFields:
    """返回字段完整性测试"""

    def setup_method(self):
        self.conn = _setup_db()
        self.db = GraphDB(':memory:')
        self.db.conn = self.conn
        self.mock_manager = MockExpertManager(available=True)

    def teardown_method(self):
        self.db.close()

    def test_return_dict_contains_all_required_fields(self):
        """返回字典包含所有必要字段"""
        result = _make_ripple_result(
            seeds=[("感冒", 0.9, "医学", "急性上呼吸道感染")],
            domain_scores={"医学": 0.85},
        )

        answer_result = answer_with_expert(result, self.db, self.mock_manager)

        required_fields = [
            'expert_answer',
            'retrieval_answer',
            'expert_domain',
            'expert_available',
            'reliability_score',
            'cross_validation_status',
            'cross_validation_discount',
        ]

        for field in required_fields:
            assert field in answer_result, f"Missing field: {field}"

    def test_return_dict_fallback_contains_all_required_fields(self):
        """降级时返回字典也包含所有必要字段"""
        result = _make_ripple_result(
            seeds=[("感冒", 0.9, "医学", "急性上呼吸道感染")],
            domain_scores={"医学": 0.85},
        )

        answer_result = answer_with_expert(result, self.db, None)

        required_fields = [
            'expert_answer',
            'retrieval_answer',
            'expert_domain',
            'expert_available',
            'reliability_score',
            'cross_validation_status',
            'cross_validation_discount',
        ]

        for field in required_fields:
            assert field in answer_result, f"Missing field: {field}"


class TestAnswerWithExpertBackwardCompat:
    """向后兼容测试"""

    def setup_method(self):
        self.conn = _setup_db()
        self.db = GraphDB(':memory:')
        self.db.conn = self.conn

    def teardown_method(self):
        self.db.close()

    def test_answer_from_activation_unchanged(self):
        """answer_from_activation() 行为不变"""
        result = _make_ripple_result(
            seeds=[("感冒", 0.9, "医学", "急性上呼吸道感染")],
            domain_scores={"医学": 0.85},
        )

        text = answer_from_activation(result, self.db)
        assert isinstance(text, str)
        assert "感冒" in text

    def test_answer_from_activation_empty_seeds(self):
        """answer_from_activation() 空种子"""
        result = _make_ripple_result(seeds=None, domain_scores={})
        text = answer_from_activation(result, self.db)
        assert "未匹配" in text


if __name__ == '__main__':
    import pytest
    pytest.main([__file__, '-v'])