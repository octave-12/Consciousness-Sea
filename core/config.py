"""
识海核心常数配置

所有可调参数集中在这里。
Phase 2 会对这些常数做统计观测调优。
"""

from __future__ import annotations

import os
import pathlib

# ── 涟漪传播 ──────────────────────────────────────────────
RIPPLE_DEPTH = 2           # BFS 最大深度
RIPPLE_DECAY = 0.7         # 每跳衰减系数
INITIAL_ACTIVATION = 1.0   # 查询词匹配种子的初始激活值
MAX_ACTIVATION = 2.0       # 单节点激活值上限（防止多路径叠加爆炸）

# ── 路由器 ─────────────────────────────────────────────────
DOMAIN_THRESHOLD = 0.3     # 领域激活值低于此阈值的不考虑
TOP_K_SEEDS = 20           # 选取激活值最高的 K 个种子进入回答
TOP_K_PATHS = 10           # 选取最多 K 条路径展示

# ── 文本匹配 ───────────────────────────────────────────────
# ENABLE_FUZZY 和 FUZZY_EDIT_DISTANCE 已移至 Phase 1 配置段（当前阶段已启用模糊匹配）

# ── 校验器 ─────────────────────────────────────────────────
CONFIDENCE_HIGH = 0.7      # 高于此值 → 正向熏习
CONFIDENCE_LOW = 0.3       # 低于此值 → 负向熏习
KARMA_DELTA = 0.01         # 每次熏习的权重修改量

# ── 业力边界 ───────────────────────────────────────────────
KARMA_MIN = 0.005          # 低于此值 → 删除边
KARMA_MAX = 2.0            # 高于此值 → 裁剪
KARMA_DECAY_PER_YEAR = 0.001  # 冷路径年度衰减（Phase 2 启用）

# ── 熏习粒度控制 ────────────────────────────────────────────
KARMA_TOP_N = 20           # 只熏激活值最高的 N 个种子之间的边（Phase 2）
KARMA_FULL_SET = True      # Phase 0: 熏所有 co-activated pairs

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

# ── 数据库 ─────────────────────────────────────────────────
DEFAULT_DB_PATH = 'data/consciousness_sea.db'

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

# ── 路径配置 ───────────────────────────────────────────────
DEFAULT_DATA_DIR = "data"                  # 默认数据目录（相对于项目根目录）
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


def resolve_data_dir(cli_data_dir: str | None = None) -> pathlib.Path:
    """
    解析数据目录路径。

    优先级：
      1. 命令行参数 --data-dir 传入的路径
      2. 环境变量 CONSCIOUSNESS_SEA_DATA_DIR
      3. 脚本相对的 data/ 目录

    Args:
        cli_data_dir: 命令行传入的数据目录路径，可选。

    Returns:
        pathlib.Path: 解析后的数据目录绝对路径。
    """
    # 优先级 1: 命令行参数
    if cli_data_dir is not None:
        return pathlib.Path(cli_data_dir).resolve()

    # 优先级 2: 环境变量
    env_dir = os.environ.get("CONSCIOUSNESS_SEA_DATA_DIR")
    if env_dir:
        return pathlib.Path(env_dir).resolve()

    # 优先级 3: 脚本相对的 data/ 目录
    # 使用调用栈找到主脚本位置，若无法确定则回退到当前工作目录
    import __main__

    main_file = getattr(__main__, "__file__", None)
    if main_file:
        project_root = pathlib.Path(main_file).resolve().parent
    else:
        project_root = pathlib.Path.cwd()

    return project_root / DEFAULT_DATA_DIR
