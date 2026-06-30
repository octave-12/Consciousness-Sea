"""
识海核心常数配置

所有可调参数集中在这里。
Phase 2 会对这些常数做统计观测调优。
"""

from __future__ import annotations

import os
import pathlib

# ── 路径配置 ───────────────────────────────────────────────
DEFAULT_DATA_DIR = "data"                  # 默认数据目录（相对于项目根目录）


def resolve_data_dir(cli_data_dir: str | None = None) -> pathlib.Path:
    """
    解析数据目录路径。

    优先级：
      1. 命令行参数 --data-dir 传入的路径
      2. 环境变量 CONSCIOUSNESS_SEA_DATA_DIR
      3. 包根目录下的 data/ 目录（基于 consciousness_sea 包位置推断）
      4. 当前工作目录下的 data/ 目录

    Args:
        cli_data_dir: 命令行传入的数据目录路径，可选。

    Returns:
        pathlib.Path: 解析后的数据目录绝对路径。
    """
    if cli_data_dir is not None:
        return pathlib.Path(cli_data_dir).resolve()

    env_dir = os.environ.get("CONSCIOUSNESS_SEA_DATA_DIR")
    if env_dir:
        return pathlib.Path(env_dir).resolve()

    try:
        package_dir = pathlib.Path(__file__).resolve().parent.parent
        candidate = package_dir / DEFAULT_DATA_DIR
        if candidate.is_dir():
            return candidate
    except Exception:
        pass

    return pathlib.Path.cwd() / DEFAULT_DATA_DIR


# ── 数据库 ─────────────────────────────────────────────────
DEFAULT_DB_PATH = str(resolve_data_dir() / 'consciousness_sea.db')

# ── 涟漪传播 ──────────────────────────────────────────────
RIPPLE_DEPTH = 2           # BFS 最大深度
RIPPLE_DECAY = 0.7         # 每跳衰减系数
INITIAL_ACTIVATION = 1.0   # 查询词匹配种子的初始激活值
MAX_ACTIVATION = 2.0       # 单节点激活值上限（防止多路径叠加爆炸）

# ── 路由器 ─────────────────────────────────────────────────
DOMAIN_THRESHOLD = 0.3     # 领域激活值低于此阈值的不考虑
TOP_K_SEEDS = 20           # 选取激活值最高的 K 个种子进入回答
TOP_K_PATHS = 10           # 选取最多 K 条路径展示

# ── 校验器 ─────────────────────────────────────────────────
CONFIDENCE_HIGH = 0.7      # 高于此值 → 正向熏习
CONFIDENCE_LOW = 0.3       # 低于此值 → 负向熏习
KARMA_DELTA = 0.01         # 每次熏习的权重修改量

# ── 业力边界 ───────────────────────────────────────────────
KARMA_MIN = 0.01           # 低于此值 → 删除边（Phase 2 从 0.005 调整为 0.01）
KARMA_MAX = 2.0            # 高于此值 → 裁剪
KARMA_DECAY_PER_YEAR = 0.001  # 冷路径年度衰减（Phase 2 启用）

# ── 熏习粒度控制 ────────────────────────────────────────────
KARMA_TOP_N = 20           # 只熏激活值最高的 N 个种子之间的边（Phase 2）
KARMA_FULL_SET = True      # True=全量熏习(向后兼容), False=Top-N
KARMA_MAX_PAIRS = 500      # 单次熏习最大修改边数

# ── 路径与关系 ─────────────────────────────────────────────
RELATION_NAMES = {
    'IS_A': '是一种',
    'PART_OF': '是...的组成部分',
    'RELATED': '与...相关',
    'COOCCURS_WITH': '常与...共现',
    'CAUSE': '导致',
    'HAS': '拥有',
    'DEFINED_AS': '定义为',
    'LOCATED_IN': '位于',
    'BEFORE': '先于',
    'FOLLOWS': '跟随',
    'HAS_SUBEVENT': '包含子事件',
    'HAS_CAPABILITY': '具备能力',
    'HAS_PROPERTY': '具有属性',
    'MADE_OF': '由...构成',
    'PAR': '是...的一部分',
    'COOCCURS_IN': '在...中共现',
    'SYNONYM': '别名',
}

# ── 用户种子 ───────────────────────────────────────────────
DEFAULT_USER_SEED = 'user_default'
USER_PREACTIVATION = 0.5   # 用户种子的固定预激活值

# ══════════════════════════════════════════════════════════
#  以下为 Phase 1 新增配置段
# ══════════════════════════════════════════════════════════

# ── HTTP API 配置 ─────────────────────────────────────────
API_HOST = "127.0.0.1"     # API 监听地址（仅本地访问）
API_PORT = 8111            # API 监听端口
API_WORKERS = 1            # Uvicorn worker 数量
API_TIMEOUT = 5.0          # 请求超时时间（秒）

# ── API 认证配置 ─────────────────────────────────────────
API_AUTH_ENABLED: bool = False  # API Key 认证开关（默认关闭，向后兼容）
API_KEY: str = ""               # API Key 值（启用认证时必须配置）

# ── CORS 配置 ─────────────────────────────────────────────
CORS_ALLOWED_ORIGINS: list[str] = ["http://localhost:3000", "http://localhost:5173"]  # 默认开发环境
_cors_env = os.environ.get("CORS_ALLOWED_ORIGINS")
if _cors_env:
    CORS_ALLOWED_ORIGINS = [o.strip() for o in _cors_env.split(",") if o.strip()]

# ── 分词器配置 ─────────────────────────────────────────────
MAX_COMPOUND_LEN = 8       # 组合词最大匹配长度（字符数）
ENABLE_FUZZY = True        # 启用模糊匹配（Phase 1 开启，覆盖上方同名变量）
FUZZY_EDIT_DISTANCE = 1    # 模糊匹配编辑距离阈值（覆盖上方同名变量）
MIN_KEYWORD_LENGTH = 2     # 关键词最小长度，低于此值的单字词被过滤

# ── 否定词配置 ─────────────────────────────────────────────
NEGATION_SCOPE = 4         # 否定词作用范围（否定词后的字符数）

# ── 停用词配置 ─────────────────────────────────────────────
STOPWORDS_PATH = "data/stopwords.txt"  # 扩展停用词文件路径

# ── 查询历史配置 ───────────────────────────────────────────
HISTORY_DEFAULT_LIMIT = 20  # 查询历史默认返回条数
HISTORY_MAX_LIMIT = 100     # 查询历史最大返回条数

# ── 数据修复配置 ───────────────────────────────────────────
RELATED_MIN_CONFIDENCE = 0.5   # RELATED 边最低置信度阈值
ISA_PRUNE_THRESHOLD = 0.3      # IS_A 边裁剪阈值（低于此值的边被删除）
ISA_MAX_COUNT = 1_500_000      # IS_A 边最大保留数量
DOMAIN_COVERAGE_TARGET = 0.95  # 领域覆盖率目标（95%）
REPAIR_BATCH_SIZE = 50000      # 修复脚本批量处理大小

CG_DB_FILENAME = "concept_graph.db"        # 龙珠概念图数据库文件名
CEDICT_FILENAME = "cedict_parsed.json"     # CC-CEDICT 解析后 JSON 文件名
ZHWIKI_FILENAME = "zhwiki.db"              # Wikipedia 中文数据库文件名

# ══════════════════════════════════════════════════════════
#  以下为 Phase 1 剩余功能配置段
# ══════════════════════════════════════════════════════════

# ── 连接池配置 ─────────────────────────────────────────────
CONNECTION_POOL_SIZE = 5       # 连接池最大连接数
CONNECTION_POOL_TIMEOUT = 5.0  # 连接池获取超时时间（秒）
BUSY_TIMEOUT_MS = 5000         # SQLite busy_timeout（毫秒）

# ── 用户管理配置 ───────────────────────────────────────────
VALID_SOURCES = frozenset({'wechat', 'web', 'api'})  # 合法的来源平台
USER_ID_HASH_LENGTH = 8        # 用户 ID 哈希截取长度
MAX_SOURCE_ID_LENGTH = 256     # 来源平台用户标识最大长度

# ── 可观测性配置 ───────────────────────────────────────────
STATUS_TOP_N = 10              # 监控面板排名显示数量
KARMA_ALERT_THRESHOLD = 1.8    # 业力边权重告警阈值
STATUS_QUERY_TIMEOUT = 5.0     # 统计查询超时时间（秒）


# ══════════════════════════════════════════════════════════
#  以下为 Phase 2 新增配置段
# ══════════════════════════════════════════════════════════

# ── Phase 2: 参数统计评估 (#4) ─────────────────────────────
PARAM_EVAL_MIN_SAMPLES = 100   # 评估所需最少样本数
PARAM_EVAL_TIMEOUT_SEC = 300   # 评估超时时间（秒）

# ── Phase 2: 双层业力架构 (#19) ────────────────────────────
GLOBAL_WEIGHT_RATIO = 0.7      # 全局业力权重占比
PERSONAL_WEIGHT_RATIO = 0.3    # 个人业力权重占比
DISTILLATION_THRESHOLD = 3     # 提炼池升级阈值（N 个独立用户）
DISTILLATION_INITIAL_WEIGHT = 0.05  # 升级为全局业力的初始权重
NEIGHBOR_OVERLAP_THRESHOLD = 0.4    # 涟漪验证邻居重叠度阈值

# ── Phase 2: 关系等价映射 (#19) ────────────────────────────
RELATION_EQUIVALENCE_MAP: dict[str, str] = {
    'HELPS_WITH': 'TREATS',
    'HELPS': 'TREATS',
    'USED_FOR': 'TREATS',
    'COOCCURS_IN': 'COOCCURS_WITH',
    'PAR': 'PART_OF',
    'SYNONYM': 'RELATED',
}


# ══════════════════════════════════════════════════════════
#  以下为 Phase 1 专家组配置段
# ══════════════════════════════════════════════════════════

# ── 专家模型配置 ─────────────────────────────────────────
EXPERT_MODEL_PATH: str = ""  # 基座模型路径（空字符串表示无模型）
LORA_ADAPTERS: dict[str, str] = {}  # 领域→LoRA路径映射，如 {"医学": "models/medical_lora"}
EXPERT_RELIABILITY: dict[str, float] = {}  # 领域→可靠性映射，如 {"医学": 0.85, "常识": 0.6}
DEFAULT_LORA: str | None = None  # 默认LoRA领域名
EXPERT_MAX_VRAM_GB: float = 5.5  # VRAM预算上限(GB)
EXPERT_INFERENCE_TIMEOUT: float = 10.0  # 推理超时(秒)
EXPERT_MAX_NEW_TOKENS: int = 512  # 最大生成token数

# ── Ollama 后端配置 ─────────────────────────────────────
OLLAMA_BASE_URL: str = os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434")  # Ollama API 地址
OLLAMA_MODEL: str = "deepseek-r1-7b"             # Ollama 模型名
OLLAMA_TIMEOUT: float = 60.0                      # Ollama 请求超时（秒）
OLLAMA_ENABLED: bool = False                       # 是否启用 Ollama 后端
EXPERT_BACKEND: str = "auto"                       # 专家后端选择: "auto" | "ollama" | "pytorch" | "none"

# ── 交叉验证配置 ─────────────────────────────────────────
CROSS_VALIDATION_DISCOUNT: float = 0.7  # 交叉验证折扣系数
CROSS_VALIDATION_CONSISTENCY_THRESHOLD: float = 0.8  # 一致性阈值

# ── 上下文注入配置 ───────────────────────────────────────
CONTEXT_MAX_TOKENS: int = 2048  # 上下文最大token数
CONTEXT_MAX_QUERY_LENGTH: int = 500  # 用户查询文本最大字符数


# ══════════════════════════════════════════════════════════
#  以下为 Phase 3 自生长配置段
# ══════════════════════════════════════════════════════════

# ── 别名自动扩展 (#3) ─────────────────────────────────
ALIAS_AUTO_EXTEND: bool = True          # 别名自动扩展开关
ALIAS_BACK_REF_THRESHOLD: float = 0.6   # 回指率阈值（超过此值自动添加别名）
ALIAS_MIN_COUNT: int = 5                # 最小出现次数（低于此值不判定）
ALIAS_CONFLICT_MARGIN: float = 0.2      # 冲突检测容差（次高与最高回指率之差小于此值时标记待审核）

# ── 候选种子 (#18) ─────────────────────────────────────
CANDIDATE_SEED_AUTO_CREATE: bool = True  # 候选种子自动创建开关
CANDIDATE_SEED_MIN_COUNT: int = 5        # 候选种子最小出现次数（内存预计数器阈值）
CANDIDATE_SEED_PROMOTE_COUNT: int = 10   # 候选种子升级为正式种子的出现次数阈值
CANDIDATE_SEED_EXPIRE_DAYS: int = 30     # 候选种子过期天数（超过标记为 expired）
CANDIDATE_SEED_PURGE_DAYS: int = 90      # 过期候选种子清理天数（超过则删除记录）

# ── 新用户冷启动 (#14) ────────────────────────────────
COLD_START_ENABLED: bool = True          # 冷启动功能开关
COLD_START_QUERIES: int = 20             # 冷启动期查询次数阈值

# ── 业力检查点与回滚 (#13) ────────────────────────────
CHECKPOINT_CRON_HOUR: int = 3            # 自动检查点创建时间（小时，24小时制）
CHECKPOINT_RETAIN_COUNT: int = 30        # 保留最近 N 个检查点
CHECKPOINT_DIR: str = "checkpoints/"     # 检查点文件存储目录


# ══════════════════════════════════════════════════════════
#  以下为 Phase 4 元种子体系配置段
# ══════════════════════════════════════════════════════════

# ── Phase 4: 元种子体系 ──────────────────────────────────
META_SEED_ENABLED: bool = True                    # 元种子功能开关
GUARDIAN_LOOP_INTERVAL: int = 60                  # 守护循环间隔（秒）
GUARDIAN_LOOP_TIMEOUT: int = 120                  # 守护循环超时（秒）
GUARDIAN_LOOP_INITIAL_DELAY: int = 10             # 守护循环首次执行延迟（秒）
GUARDIAN_METRICS_WINDOW: int = 1000               # 指标计算窗口大小（查询次数）
META_KARMA_INITIAL_WEIGHT: float = 0.05           # 元业力边初始权重
META_KARMA_DELTA_THRESHOLD: int = 2               # 元业力熏习触发阈值（指标变化量）
META_SEED_DORMANT_CYCLES: int = 100               # 元种子休眠周期数
META_EXPLORE_WINDOW: int = 100                    # 未知领域探测窗口大小（查询次数）
META_EXPLORE_LOW_CONF_THRESHOLD: float = 0.3      # 未知领域探测低置信度频率阈值
META_ALERT_CONFLICT_THRESHOLD: int = 10           # 元种子冲突告警阈值


# ══════════════════════════════════════════════════════════
#  以下为 Phase 5 认知目标与好奇心引擎配置段
# ══════════════════════════════════════════════════════════

# ── Phase 5: 认知目标功能开关 ──────────────────────────
COGNITIVE_GOAL_ENABLED: bool = True                    # 认知目标功能开关
CURIOSITY_ENGINE_ENABLED: bool = True                  # 好奇心引擎功能开关
EXTERNAL_QUERY_ENABLED: bool = False                   # 外部查询功能开关（默认关闭）

# ── Phase 5: 目标生成阈值 ──────────────────────────────
GOAL_LOW_CONF_THRESHOLD: int = 5                       # 低置信度目标触发阈值
GOAL_LOW_CONF_WINDOW: int = 100                        # 低置信度检测窗口（查询次数）
GOAL_LOW_DENSITY_RATIO: float = 0.5                    # 低密度目标触发比例
GOAL_HIGH_CONFLICT_THRESHOLD: int = 10                 # 高冲突目标触发阈值
GOAL_NEW_TERM_THRESHOLD: int = 5                       # 新词目标触发阈值（出现次数）
GOAL_NEW_TERM_WINDOW: int = 24                         # 新词检测窗口（小时数）

# ── Phase 5: 目标调度参数 ──────────────────────────────
GOAL_AUTO_EXPLORE_THRESHOLD: float = 0.6               # 自动探索优先级阈值
GOAL_MAX_EXPLORE_PER_CYCLE: int = 3                    # 单次守护循环最大探索目标数

# ── Phase 5: 目标冷却参数 ──────────────────────────────
GOAL_DECAY_CYCLES: int = 10                            # 权重衰减周期数
GOAL_DECAY_FACTOR: float = 0.8                         # 权重衰减因子
GOAL_EXPIRE_THRESHOLD: float = 0.05                    # 目标过期权重阈值
GOAL_USER_ABSENCE_CYCLES: int = 50                     # 用户缺席归档周期数
GOAL_POOL_MAX_SIZE: int = 1000                         # 池大小上限

# ── Phase 5: 优先级权重系数 ──────────────────────────────
GOAL_WEIGHT_USER_RELEVANCE: float = 0.4                # 用户相关性权重
GOAL_WEIGHT_SYSTEM_CORENESS: float = 0.3               # 系统核心度权重
GOAL_WEIGHT_UNCERTAINTY: float = 0.2                   # 不确定性权重
GOAL_WEIGHT_DECOMPOSABILITY: float = 0.1               # 可分解性权重
GOAL_DECOMPOSABILITY_NORM: int = 5                     # 可分解性归一化因子

# ── Phase 5: 好奇心引擎参数 ──────────────────────────────
CURIOSITY_MAX_CONCURRENT: int = 1                      # 最大并发探索数
CURIOSITY_EXPLORE_TIMEOUT: int = 30                    # 探索超时秒数
CURIOSITY_MAX_DEPTH: int = 3                           # 内部探索最大深度
CURIOSITY_ACTIVATION_THRESHOLD: float = 0.1            # 探索激活值阈值

# ── Phase 5: 外部查询参数 ──────────────────────────────
EXTERNAL_SOURCE_TYPE: str = "wikipedia_dump"            # 外部知识源类型
EXTERNAL_QUERY_TIMEOUT: int = 15                       # 外部查询超时秒数
EXTERNAL_QUERY_MAX_RETRIES: int = 2                    # 外部查询最大重试次数
EXTERNAL_QUERY_MAX_PER_CYCLE: int = 1                  # 单次守护循环最大外部查询数


# ══════════════════════════════════════════════════════════
#  以下为 Phase 6 具身化/多模态感知 + Hebbian 关联配置段
# ══════════════════════════════════════════════════════════

# ── Phase 6: 感知功能总开关 ──────────────────────────────
PERCEPTION_ENABLED: bool = True                     # 感知功能开关
PERCEPTION_SHUTDOWN_TIMEOUT: int = 10               # 感知管理器停止超时（秒）
PERCEPTION_CHANNEL_FAILURE_ALERT_THRESHOLD: int = 5  # 感知通道连续失败告警阈值

# ── Phase 6: Hebbian 绑定器参数 ──────────────────────────
HEBBIAN_TIME_WINDOW: int = 1000                     # 共同激活时间窗口（毫秒）
HEBBIAN_LEARNING_RATE: float = 0.01                 # Hebbian 学习率
HEBBIAN_NEGATIVE_DECAY_ENABLED: bool = False         # 负向衰减开关
HEBBIAN_NEGATIVE_RATE: float = 0.001                # 负向衰减率
HEBBIAN_MAX_BINDINGS_PER_WINDOW: int = 50           # 单次窗口最大绑定数
HEBBIAN_CHECK_INTERVAL: int = 100                   # 绑定器检查间隔（毫秒）

# ── Phase 6: 视觉锚定器参数 ──────────────────────────────
VISUAL_FRAME_INTERVAL: int = 1000                   # 摄像头帧采集间隔（毫秒）
VISUAL_MOCK_MODE: bool = False                      # 视觉 mock 模式
VISUAL_RED_THRESHOLD: float = 0.3                   # 红色激活阈值
VISUAL_GREEN_THRESHOLD: float = 0.3                 # 绿色激活阈值
VISUAL_BLUE_THRESHOLD: float = 0.3                  # 蓝色激活阈值
VISUAL_BRIGHT_THRESHOLD: float = 0.7                # 亮度激活阈值
VISUAL_DARK_THRESHOLD: float = 0.3                  # 暗度激活阈值
VISUAL_EDGE_DENSE_THRESHOLD: float = 0.4            # 边缘密度激活阈值

# ── Phase 6: 听觉锚定器参数 ──────────────────────────────
AUDITORY_SAMPLE_RATE: int = 16000                   # 音频采样率（Hz）
AUDITORY_MOCK_MODE: bool = False                    # 听觉 mock 模式
AUDITORY_HIGH_FREQ_THRESHOLD: float = 500           # 高频激活阈值（Hz）
AUDITORY_LOW_FREQ_THRESHOLD: float = 200            # 低频激活阈值（Hz）
AUDITORY_BRIGHT_THRESHOLD: float = 3000             # 明亮音色阈值（Hz）

# ── Phase 6: 本体感知锚定器参数 ──────────────────────────
SOMATIC_SAMPLE_INTERVAL: int = 1000                 # 系统指标采集间隔（毫秒）
SOMATIC_HIGH_TEMP_THRESHOLD: float = 70             # CPU 高温阈值（°C）
SOMATIC_HIGH_MEMORY_THRESHOLD: float = 80           # 内存占用阈值（%）
SOMATIC_SLOW_RESPONSE_THRESHOLD: float = 300        # 响应延迟阈值（ms）

# ── Phase 6: 多模态对齐器参数 ────────────────────────────
MULTIMODAL_ALIGNMENT_ENABLED: bool = False          # 多模态对齐开关（默认关闭）
MULTIMODAL_ALIGNMENT_INTERVAL: int = 3600           # 对齐运行间隔（秒）
MULTIMODAL_ALIGNMENT_SAMPLE_COUNT: int = 100        # 对齐采样数量
MULTIMODAL_ALIGNMENT_PER_IMAGE_TIMEOUT: int = 3000  # 单张图像对齐超时（毫秒）


def _validate_phase5_config() -> None:
    """校验 Phase 5 配置项合法性"""
    import logging
    _log = logging.getLogger(__name__)

    errors: list[str] = []

    if GOAL_POOL_MAX_SIZE <= 0:
        errors.append(f"GOAL_POOL_MAX_SIZE={GOAL_POOL_MAX_SIZE} ≤ 0")
    if GOAL_DECAY_FACTOR >= 1.0:
        errors.append(f"GOAL_DECAY_FACTOR={GOAL_DECAY_FACTOR} ≥ 1.0")
    if GOAL_DECAY_FACTOR <= 0.0:
        errors.append(f"GOAL_DECAY_FACTOR={GOAL_DECAY_FACTOR} ≤ 0.0")
    if GOAL_EXPIRE_THRESHOLD < 0.0:
        errors.append(f"GOAL_EXPIRE_THRESHOLD={GOAL_EXPIRE_THRESHOLD} < 0.0")
    if GOAL_LOW_DENSITY_RATIO <= 0.0 or GOAL_LOW_DENSITY_RATIO > 1.0:
        errors.append(f"GOAL_LOW_DENSITY_RATIO={GOAL_LOW_DENSITY_RATIO} 不在 (0, 1] 范围内")
    if CURIOSITY_MAX_CONCURRENT <= 0:
        errors.append(f"CURIOSITY_MAX_CONCURRENT={CURIOSITY_MAX_CONCURRENT} ≤ 0")
    if CURIOSITY_MAX_DEPTH <= 0:
        errors.append(f"CURIOSITY_MAX_DEPTH={CURIOSITY_MAX_DEPTH} ≤ 0")
    if CURIOSITY_ACTIVATION_THRESHOLD < 0.0:
        errors.append(f"CURIOSITY_ACTIVATION_THRESHOLD={CURIOSITY_ACTIVATION_THRESHOLD} < 0.0")
    if EXTERNAL_QUERY_MAX_RETRIES < 0:
        errors.append(f"EXTERNAL_QUERY_MAX_RETRIES={EXTERNAL_QUERY_MAX_RETRIES} < 0")
    if EXTERNAL_QUERY_TIMEOUT <= 0:
        errors.append(f"EXTERNAL_QUERY_TIMEOUT={EXTERNAL_QUERY_TIMEOUT} ≤ 0")
    if GOAL_AUTO_EXPLORE_THRESHOLD < 0.0 or GOAL_AUTO_EXPLORE_THRESHOLD > 1.0:
        errors.append(f"GOAL_AUTO_EXPLORE_THRESHOLD={GOAL_AUTO_EXPLORE_THRESHOLD} 不在 [0, 1] 范围内")
    if GOAL_MAX_EXPLORE_PER_CYCLE <= 0:
        errors.append(f"GOAL_MAX_EXPLORE_PER_CYCLE={GOAL_MAX_EXPLORE_PER_CYCLE} ≤ 0")
    if GOAL_DECAY_CYCLES <= 0:
        errors.append(f"GOAL_DECAY_CYCLES={GOAL_DECAY_CYCLES} ≤ 0")
    if GOAL_USER_ABSENCE_CYCLES <= 0:
        errors.append(f"GOAL_USER_ABSENCE_CYCLES={GOAL_USER_ABSENCE_CYCLES} ≤ 0")
    if GOAL_DECOMPOSABILITY_NORM <= 0:
        errors.append(f"GOAL_DECOMPOSABILITY_NORM={GOAL_DECOMPOSABILITY_NORM} ≤ 0")

    weight_sum = (GOAL_WEIGHT_USER_RELEVANCE + GOAL_WEIGHT_SYSTEM_CORENESS
                  + GOAL_WEIGHT_UNCERTAINTY + GOAL_WEIGHT_DECOMPOSABILITY)
    if abs(weight_sum - 1.0) > 0.001:
        errors.append(f"优先级权重系数之和={weight_sum} ≠ 1.0")

    for error in errors:
        _log.error("Phase 5 配置校验失败: %s", error)

    if errors:
        raise ValueError("Phase 5 配置校验失败: " + ", ".join(errors))


def _validate_all_config() -> None:
    errors: list[str] = []

    if CONNECTION_POOL_SIZE <= 0:
        errors.append(f"CONNECTION_POOL_SIZE={CONNECTION_POOL_SIZE} <= 0")
    if CONNECTION_POOL_TIMEOUT <= 0:
        errors.append(f"CONNECTION_POOL_TIMEOUT={CONNECTION_POOL_TIMEOUT} <= 0")
    if DISTILLATION_THRESHOLD <= 0:
        errors.append(f"DISTILLATION_THRESHOLD={DISTILLATION_THRESHOLD} <= 0")
    if DISTILLATION_INITIAL_WEIGHT <= 0 or DISTILLATION_INITIAL_WEIGHT > 1:
        errors.append(f"DISTILLATION_INITIAL_WEIGHT={DISTILLATION_INITIAL_WEIGHT} not in (0, 1]")
    if GLOBAL_WEIGHT_RATIO < 0 or GLOBAL_WEIGHT_RATIO > 1:
        errors.append(f"GLOBAL_WEIGHT_RATIO={GLOBAL_WEIGHT_RATIO} not in [0, 1]")
    if PERSONAL_WEIGHT_RATIO < 0 or PERSONAL_WEIGHT_RATIO > 1:
        errors.append(f"PERSONAL_WEIGHT_RATIO={PERSONAL_WEIGHT_RATIO} not in [0, 1]")
    if abs(GLOBAL_WEIGHT_RATIO + PERSONAL_WEIGHT_RATIO - 1.0) > 0.01:
        errors.append(f"GLOBAL_WEIGHT_RATIO({GLOBAL_WEIGHT_RATIO}) + PERSONAL_WEIGHT_RATIO({PERSONAL_WEIGHT_RATIO}) != 1.0")
    if KARMA_ALERT_THRESHOLD <= 0:
        errors.append(f"KARMA_ALERT_THRESHOLD={KARMA_ALERT_THRESHOLD} <= 0")
    if ALIAS_BACK_REF_THRESHOLD <= 0 or ALIAS_BACK_REF_THRESHOLD > 1:
        errors.append(f"ALIAS_BACK_REF_THRESHOLD={ALIAS_BACK_REF_THRESHOLD} not in (0, 1]")
    if CANDIDATE_SEED_MIN_COUNT <= 0:
        errors.append(f"CANDIDATE_SEED_MIN_COUNT={CANDIDATE_SEED_MIN_COUNT} <= 0")
    if CANDIDATE_SEED_PROMOTE_COUNT < CANDIDATE_SEED_MIN_COUNT:
        errors.append(f"CANDIDATE_SEED_PROMOTE_COUNT({CANDIDATE_SEED_PROMOTE_COUNT}) < CANDIDATE_SEED_MIN_COUNT({CANDIDATE_SEED_MIN_COUNT})")
    if CHECKPOINT_RETAIN_COUNT <= 0:
        errors.append(f"CHECKPOINT_RETAIN_COUNT={CHECKPOINT_RETAIN_COUNT} <= 0")
    if GUARDIAN_LOOP_INTERVAL <= 0:
        errors.append(f"GUARDIAN_LOOP_INTERVAL={GUARDIAN_LOOP_INTERVAL} <= 0")
    if GUARDIAN_LOOP_TIMEOUT < GUARDIAN_LOOP_INTERVAL:
        errors.append(f"GUARDIAN_LOOP_TIMEOUT({GUARDIAN_LOOP_TIMEOUT}) < GUARDIAN_LOOP_INTERVAL({GUARDIAN_LOOP_INTERVAL})")
    if GOAL_POOL_MAX_SIZE <= 0:
        errors.append(f"GOAL_POOL_MAX_SIZE={GOAL_POOL_MAX_SIZE} <= 0")
    if GOAL_DECAY_FACTOR <= 0 or GOAL_DECAY_FACTOR >= 1:
        errors.append(f"GOAL_DECAY_FACTOR={GOAL_DECAY_FACTOR} not in (0, 1)")
    if HEBBIAN_LEARNING_RATE <= 0 or HEBBIAN_LEARNING_RATE > 1:
        errors.append(f"HEBBIAN_LEARNING_RATE={HEBBIAN_LEARNING_RATE} not in (0, 1]")

    if errors:
        raise ValueError("配置校验失败: " + ", ".join(errors))


_config_validated = False


def validate_config() -> None:
    """延迟执行配置校验（在应用启动时调用，而非 import 时）

    避免配置错误导致整个模块无法 import，
    允许应用在启动阶段捕获并优雅处理配置问题。
    """
    global _config_validated
    if _config_validated:
        return
    _config_validated = True
    _validate_all_config()
    _validate_phase5_config()