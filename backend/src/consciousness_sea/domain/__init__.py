"""领域层：图谱、路由、校验、回答、分词、领域推断、查询历史"""

from .answerer import answer_as_dict, answer_from_activation, answer_with_expert
from .domain_inference import infer_domains, infer_single_domain
from .graph_db import GraphDB
from .query_history import get_history, record_query
from .router import ActivationNode, RippleResult, route
from .tokenizer import TokenMatch, tokenize
from .verifier import apply_karma, verify

__all__ = [
    'GraphDB',
    'route', 'RippleResult', 'ActivationNode',
    'answer_from_activation', 'answer_as_dict', 'answer_with_expert',
    'verify', 'apply_karma',
    'tokenize', 'TokenMatch',
    'infer_domains', 'infer_single_domain',
    'record_query', 'get_history',
]
