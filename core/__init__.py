"""
识海 (Consciousness Sea) — 去中心化多智能体认知架构

Phase 0: 无专家运行 — 文本匹配 + BFS 涟漪传播 + 检索式回答
Phase 1 剩余: 并发隔离 + 用户识别 + 可观测性
Phase 1 专家组: 基座模型 + LoRA 热切换 + 交叉验证 + 可靠性建模
Phase 2: 熏习粒度 + 业力边界 + 双层业力 + 参数统计
"""

from .graph_db import GraphDB
from .router import route, RippleResult, ActivationNode
from .answerer import answer_from_activation, answer_as_dict, answer_with_expert
from .verifier import verify, apply_karma
from .tokenizer import tokenize, TokenMatch
from .domain_inference import infer_domains, infer_single_domain
from .query_history import record_query, get_history
from .connection_pool import ConnectionPool, ConnectionPoolExhausted
from .user_manager import UserManager
from .session_manager import SessionManager, SessionContext
from .observer import Observer, StatusData, SeedRankItem, KarmaRankItem, QueryRecord
from .karma_cleaner import KarmaCleaner
from .distillation_pool import DistillationPool
from .expert_manager import ExpertManager, ExpertStatus, InferenceResult
from .context_injector import ContextInjector, PromptResult
from .cross_validator import CrossValidator, CrossValidationResult, CrossValidationStatus
from .expert_reliability import ExpertReliabilityStore

# Phase 3: 自生长
from .alias_expander import AliasExpander, BackrefEvent, BackrefStats, AliasExpansionResult, BackrefStatus
from .seed_candidate import SeedCandidateManager, CandidateSeed, PromotionResult, CandidateStatus
from .cold_start import ColdStartManager, ColdStartState
from .checkpoint import CheckpointManager, CheckpointMeta, CheckpointData, RollbackResult, CheckpointSource

# Phase 4: 元种子体系
from .meta_seed import MetaSeedManager, MetaSeedData, MetaSeedCategory, MetaSeedStatus
from .guardian_loop import GuardianLoop, GuardianLoopResult, GuardianLoopStatus

# Phase 5: 认知目标与好奇心引擎
from .cognitive_goal import CognitiveGoalManager, CognitiveGoalData, GoalType, GoalStatus
from .curiosity_engine import CuriosityEngine, ExplorationResult, CuriosityEngineStatus, ExternalQueryResult

# Phase 6: 具身化/多模态感知 + Hebbian 关联
from .perception import (
    PerceptionManager, PerceptionManagerStatus,
    PerceptActivationEvent, ConceptActivationEvent,
    PerceptionChannel, PerceptualSeedStatus,
)
from .visual_anchor import VisualAnchor, VisualFeatures
from .audio_anchor import AudioAnchor, AudioFeatures
from .somatic_anchor import SomaticAnchor, SomaticFeatures
from .hebbian_binder import HebbianBinder, HebbianBinderStatus
from .multimodal_aligner import MultimodalAligner, AlignmentResult

__all__ = [
    'GraphDB',
    'route', 'RippleResult', 'ActivationNode',
    'answer_from_activation', 'answer_as_dict', 'answer_with_expert',
    'verify', 'apply_karma',
    'tokenize', 'TokenMatch',
    'infer_domains', 'infer_single_domain',
    'record_query', 'get_history',
    'ConnectionPool', 'ConnectionPoolExhausted',
    'UserManager',
    'SessionManager', 'SessionContext',
    'Observer', 'StatusData', 'SeedRankItem', 'KarmaRankItem', 'QueryRecord',
    'KarmaCleaner',
    'DistillationPool',
    'ExpertManager', 'ExpertStatus', 'InferenceResult',
    'ContextInjector', 'PromptResult',
    'CrossValidator', 'CrossValidationResult', 'CrossValidationStatus',
    'ExpertReliabilityStore',
    # Phase 3: 自生长
    'AliasExpander', 'BackrefEvent', 'BackrefStats', 'AliasExpansionResult', 'BackrefStatus',
    'SeedCandidateManager', 'CandidateSeed', 'PromotionResult', 'CandidateStatus',
    'ColdStartManager', 'ColdStartState',
    'CheckpointManager', 'CheckpointMeta', 'CheckpointData', 'RollbackResult', 'CheckpointSource',
    # Phase 4: 元种子体系
    'MetaSeedManager', 'MetaSeedData', 'MetaSeedCategory', 'MetaSeedStatus',
    'GuardianLoop', 'GuardianLoopResult', 'GuardianLoopStatus',
    # Phase 5: 认知目标与好奇心引擎
    'CognitiveGoalManager', 'CognitiveGoalData', 'GoalType', 'GoalStatus',
    'CuriosityEngine', 'ExplorationResult', 'CuriosityEngineStatus', 'ExternalQueryResult',
    # Phase 6: 具身化/多模态感知 + Hebbian 关联
    'PerceptionManager', 'PerceptionManagerStatus',
    'PerceptActivationEvent', 'ConceptActivationEvent',
    'PerceptionChannel', 'PerceptualSeedStatus',
    'VisualAnchor', 'VisualFeatures',
    'AudioAnchor', 'AudioFeatures',
    'SomaticAnchor', 'SomaticFeatures',
    'HebbianBinder', 'HebbianBinderStatus',
    'MultimodalAligner', 'AlignmentResult',
]
