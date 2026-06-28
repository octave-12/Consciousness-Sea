"""
conftest.py — 共享测试 fixtures

提供 MockExpertManager 和通用测试辅助函数。
"""

from __future__ import annotations

import sqlite3
import sys
import os
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.expert_manager import ExpertStatus, InferenceResult
from core.graph_db import GraphDB
from core.router import RippleResult, ActivationNode


# ═══════════════════════════════════════════════════════════
#  MockExpertManager — 模拟 ExpertManager 的测试替身
# ═══════════════════════════════════════════════════════════

class MockExpertManager:
    """Mock ExpertManager for testing without GPU

    模拟 ExpertManager 的核心接口，无需 GPU/PyTorch 依赖。
    """

    def __init__(self, available: bool = True, answer: str = "测试专家回答"):
        self._expert_available = available
        self._answer = answer
        self._current_lora = "医学" if available else None
        self._reliability_scores = {"医学": 0.85, "常识": 0.6}
        self._lora_switch_count = 0
        self._inference_count = 0
        self._fallback_count = 0 if available else 1
        self._infer_calls: list[dict] = []  # 记录推理调用

    @property
    def expert_available(self) -> bool:
        return self._expert_available

    @property
    def status(self) -> ExpertStatus:
        return ExpertStatus(
            expert_available=self._expert_available,
            current_lora=self._current_lora,
            vram_usage_mb=0,
            reliability_scores=dict(self._reliability_scores),
            lora_switch_count=self._lora_switch_count,
            inference_count=self._inference_count,
            fallback_count=self._fallback_count,
            unavailable_reason=None if self._expert_available else "no_torch",
        )

    def infer(
        self,
        prompt: str,
        target_domain: str,
        max_new_tokens: int = 512,
    ) -> InferenceResult:
        self._infer_calls.append({
            "prompt": prompt,
            "target_domain": target_domain,
            "max_new_tokens": max_new_tokens,
        })
        if not self._expert_available:
            self._fallback_count += 1
            return InferenceResult(
                answer_text="",
                domain=target_domain,
                reliability=0.7,
                inference_time_ms=0,
                fallback=True,
            )
        self._inference_count += 1
        return InferenceResult(
            answer_text=self._answer,
            domain=target_domain,
            reliability=self._reliability_scores.get(target_domain, 0.7),
            inference_time_ms=100,
            fallback=False,
        )

    def infer_multi_domain(
        self,
        prompt: str,
        domains: list[str],
        max_new_tokens: int = 512,
    ) -> list[InferenceResult]:
        results: list[InferenceResult] = []
        for domain in domains:
            result = self.infer(prompt, domain, max_new_tokens=max_new_tokens)
            if not result.fallback and result.answer_text:
                results.append(result)
        return results

    def shutdown(self) -> None:
        pass


# ═══════════════════════════════════════════════════════════
#  通用测试辅助
# ═══════════════════════════════════════════════════════════

def _setup_test_db() -> sqlite3.Connection:
    """创建测试用内存数据库（含基础表和种子数据）"""
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
        ('姜汤', '姜汤', 'CONCEPT', '[]', '常识', 'ginger soup'),
        ('量子力学', '量子力学', 'CONCEPT', '[]', '物理', 'quantum mechanics'),
        ('薛定谔方程', '薛定谔方程', 'CONCEPT', '[]', '物理', 'Schrodinger equation'),
        ('人工智能', '人工智能', 'CONCEPT', '["AI"]', '计算机', 'AI'),
        ('深度学习', '深度学习', 'CONCEPT', '[]', '计算机', 'deep learning'),
        ('水', '水', 'CONCEPT', '[]', '常识', 'water'),
    ]
    conn.executemany(
        "INSERT INTO seeds (id,label,type,aliases,domain,definition) VALUES (?,?,?,?,?,?)",
        seeds,
    )

    edges = [
        ('感冒', '发热', 'COOCCURS_WITH', 0.95),
        ('感冒', '咳嗽', 'COOCCURS_WITH', 0.90),
        ('感冒', '维C', 'RELATED', 0.60),
        ('量子力学', '薛定谔方程', 'RELATED', 0.88),
        ('人工智能', '深度学习', 'IS_A', 0.90),
    ]
    conn.executemany(
        "INSERT INTO karma_edges (source,target,relation,weight) VALUES (?,?,?,?)",
        edges,
    )
    conn.commit()
    return conn


def _make_graph_db(conn: sqlite3.Connection) -> GraphDB:
    """从已有连接创建 GraphDB 实例

    注意：绕过 connect() 方法直接赋值 conn，因此需要手动调用
    ensure_phase2_tables() / ensure_phase3_tables() 确保所有表存在，
    否则 route() 等调用 ColdStartManager 时会因 user_cold_start 表
    不存在而报错。
    """
    db = GraphDB(':memory:')
    db.conn = conn
    db.ensure_phase2_tables()
    db.ensure_phase3_tables()
    return db


def _make_ripple_result(
    activated_labels: list[str] | None = None,
    domain_scores: dict[str, float] | None = None,
    paths: list[dict] | None = None,
    query: str = "测试查询",
    selected_domains: list[str] | None = None,
) -> RippleResult:
    """辅助函数：构造 RippleResult 对象"""
    result = RippleResult()
    result.query = query
    if activated_labels:
        for i, label in enumerate(activated_labels):
            result.activated[label] = ActivationNode(
                label=label,
                activation=1.0 - i * 0.1,
                domain=domain_scores and list(domain_scores.keys())[0] or "医学",
                definition=f"{label}的定义",
                depth=0,
            )
    if domain_scores:
        result.domain_scores = domain_scores
    if paths:
        result.paths = paths
    if selected_domains:
        # selected_domains 是 property，通过 domain_scores 控制
        pass
    return result


# ═══════════════════════════════════════════════════════════
#  pytest fixtures
# ═══════════════════════════════════════════════════════════

@pytest.fixture
def mock_expert_manager_available():
    """可用状态的 MockExpertManager"""
    return MockExpertManager(available=True, answer="这是专家测试回答")


@pytest.fixture
def mock_expert_manager_unavailable():
    """不可用状态的 MockExpertManager"""
    return MockExpertManager(available=False)


@pytest.fixture
def test_db():
    """测试用内存数据库连接"""
    return _setup_test_db()


@pytest.fixture
def test_graph(test_db):
    """测试用 GraphDB 实例"""
    return _make_graph_db(test_db)