"""
识海 (Consciousness Sea) — 去中心化多智能体认知架构

Phase 0: 无专家运行 — 文本匹配 + BFS 涟漪传播 + 检索式回答
Phase 1 剩余: 并发隔离 + 用户识别 + 可观测性
Phase 1 专家组: 基座模型 + LoRA 热切换 + 交叉验证 + 可靠性建模
Phase 2: 熏习粒度 + 业力边界 + 双层业力 + 参数统计
"""

# domain
from .domain import (
    ActivationNode,
    GraphDB,
    RippleResult,
    TokenMatch,
    answer_as_dict,
    answer_from_activation,
    answer_with_expert,
    apply_karma,
    get_history,
    infer_domains,
    infer_single_domain,
    record_query,
    route,
    tokenize,
    verify,
)

# expert
from .expert import (
    ContextInjector,
    CrossValidationResult,
    CrossValidationStatus,
    CrossValidator,
    ExpertManager,
    ExpertReliabilityStore,
    ExpertStatus,
    InferenceResult,
    PromptResult,
)

# infrastructure
from .infrastructure import (
    ConnectionPool,
    ConnectionPoolExhausted,
    KarmaCleaner,
    KarmaRankItem,
    Observer,
    ParamEvaluator,
    QueryRecord,
    SeedRankItem,
    SessionContext,
    SessionManager,
    StatusData,
    UserManager,
    ensure_param_stats_table,
    record_param_stats,
)

# learning
from .learning import (
    AliasExpander,
    AliasExpansionResult,
    BackrefEvent,
    BackrefStats,
    BackrefStatus,
    CandidateSeed,
    CandidateStatus,
    CheckpointData,
    CheckpointManager,
    CheckpointMeta,
    CheckpointSource,
    ColdStartManager,
    ColdStartState,
    DistillationPool,
    PromotionResult,
    RollbackResult,
    SeedCandidateManager,
)

# metacognition
from .metacognition import (
    CognitiveGoalData,
    CognitiveGoalManager,
    CuriosityEngine,
    CuriosityEngineStatus,
    ExplorationResult,
    ExternalQueryResult,
    GoalStatus,
    GoalType,
    GuardianLoop,
    GuardianLoopResult,
    GuardianLoopStatus,
    MetaSeedCategory,
    MetaSeedData,
    MetaSeedManager,
    MetaSeedStatus,
)

# perception
from .perception import (
    AlignmentResult,
    AudioAnchor,
    AudioFeatures,
    ConceptActivationEvent,
    HebbianBinder,
    HebbianBinderStatus,
    MultimodalAligner,
    PerceptActivationEvent,
    PerceptionChannel,
    PerceptionManager,
    PerceptionManagerStatus,
    PerceptualSeedStatus,
    SomaticAnchor,
    SomaticFeatures,
    VisualAnchor,
    VisualFeatures,
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
