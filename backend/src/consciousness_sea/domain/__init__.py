"""领域层：图谱、路由、校验、回答、分词、领域推断、查询历史"""

from .graph_db import GraphDB
from .router import route, RippleResult, ActivationNode
from .answerer import answer_from_activation, answer_as_dict, answer_with_expert
from .verifier import verify, apply_karma
from .tokenizer import tokenize, TokenMatch
from .domain_inference import infer_domains, infer_single_domain
from .query_history import record_query, get_history

__all__ = [
    'GraphDB',
    'route', 'RippleResult', 'ActivationNode',
    'answer_from_activation', 'answer_as_dict', 'answer_with_expert',
    'verify', 'apply_karma',
    'tokenize', 'TokenMatch',
    'infer_domains', 'infer_single_domain',
    'record_query', 'get_history',
]