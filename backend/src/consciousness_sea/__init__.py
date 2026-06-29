"""
识海 (Consciousness Sea) — 去中心化多智能体认知架构

Phase 0: 无专家运行 — 文本匹配 + BFS 涟漪传播 + 检索式回答
Phase 1 剩余: 并发隔离 + 用户识别 + 可观测性
Phase 1 专家组: 基座模型 + LoRA 热切换 + 交叉验证 + 可靠性建模
Phase 2: 熏习粒度 + 业力边界 + 双层业力 + 参数统计
"""

# domain
from .domain import (
    GraphDB,
    route, RippleResult, ActivationNode,
    answer_from_activation, answer_as_dict, answer_with_expert,
    verify, apply_karma,
    tokenize, TokenMatch,
    infer_domains, infer_single_domain,
    record_query, get_history,
)

# infrastructure
from .infrastructure import (
    ConnectionPool, ConnectionPoolExhausted,
    UserManager,
    SessionManager, SessionContext,
    Observer, StatusData, SeedRankItem, KarmaRankItem, QueryRecord,
    KarmaCleaner,
    ensure_param_stats_table, record_param_stats,
    ParamEvaluator,
)

# learning
from .learning import (
    DistillationPool,
    AliasExpander, BackrefEvent, BackrefStats, AliasExpansionResult, BackrefStatus,
    SeedCandidateManager, CandidateSeed, PromotionResult, CandidateStatus,
    ColdStartManager, ColdStartState,
    CheckpointManager, CheckpointMeta, CheckpointData, RollbackResult, CheckpointSource,
)

# expert
from .expert import (
    ExpertManager, ExpertStatus, InferenceResult,
    ContextInjector, PromptResult,
    CrossValidator, CrossValidationResult, CrossValidationStatus,
    ExpertReliabilityStore,
)

# metacognition
from .metacognition import (
    MetaSeedManager, MetaSeedData, MetaSeedCategory, MetaSeedStatus,
    GuardianLoop, GuardianLoopResult, GuardianLoopStatus,
    CognitiveGoalManager, CognitiveGoalData, GoalType, GoalStatus,
    CuriosityEngine, ExplorationResult, CuriosityEngineStatus, ExternalQueryResult,
)

# perception
from .perception import (
    PerceptionManager, PerceptionManagerStatus,
    PerceptActivationEvent, ConceptActivationEvent,
    PerceptionChannel, PerceptualSeedStatus,
    VisualAnchor, VisualFeatures,
    AudioAnchor, AudioFeatures,
    SomaticAnchor, SomaticFeatures,
    HebbianBinder, HebbianBinderStatus,
    MultimodalAligner, AlignmentResult,
)

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
    'ensure_param_stats_table', 'record_param_stats',
    'ParamEvaluator',
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