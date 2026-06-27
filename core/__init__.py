"""
识海 (Consciousness Sea) — 去中心化多智能体认知架构

Phase 0: 无专家运行 — 文本匹配 + BFS 涟漪传播 + 检索式回答
"""

from .graph_db import GraphDB
from .router import route, RippleResult, ActivationNode
from .answerer import answer_from_activation, answer_as_dict
from .verifier import verify, apply_karma
from .tokenizer import tokenize, TokenMatch
from .domain_inference import infer_domains, infer_single_domain
from .query_history import record_query, get_history

__all__ = [
    'GraphDB',
    'route', 'RippleResult', 'ActivationNode',
    'answer_from_activation', 'answer_as_dict',
    'verify', 'apply_karma',
    'tokenize', 'TokenMatch',
    'infer_domains', 'infer_single_domain',
    'record_query', 'get_history',
]
