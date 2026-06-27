"""
答案校验器 + 熏习引擎

校验: 回答关键词 vs 激活种子一致性 → 置信度
熏习: 置信度 → 决定业力偏移方向 (+0.01 / 0 / -0.01)

闭环:
  查询 → 路由 → 回答 → 校验 → 置信度 → 熏习 ↶
"""

from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Optional

from .graph_db import GraphDB
from .router import RippleResult
from .config import STOPWORDS_PATH, MIN_KEYWORD_LENGTH

log = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════
#  内置停用词表（≥ 200 个词）
# ═══════════════════════════════════════════════════════════════

BUILTIN_STOP_WORDS: frozenset[str] = frozenset({
    # ── 格式词 ──────────────────────────────────────────────
    '关于', '关联概念', '传播路径', '涟漪波及', '领域得分', '选定领域',
    '激活值', '深度', '激活',

    # ── 结构词 ──────────────────────────────────────────────
    '常与', '共现', '这是一种', '定义为', '位于', '先于', '跟随',
    '包含', '具备', '具有', '由构成',

    # ── 领域名（不是内容关键词） ────────────────────────────
    '医学', '物理', '计算机', '文学', '营养', '常识',
    '数学', '法律', '化学', '生物', '历史', '成语',

    # ── 常见虚词 ────────────────────────────────────────────
    '的', '了', '是', '在', '有', '和', '与', '或', '不', '也',
    '都', '而', '及', '等', '为', '中', '对', '上', '下', '这',
    '那', '个', '被', '把', '让', '给', '从', '到', '以', '之',
    '可', '能', '会', '要', '将', '已', '所', '其', '此', '该',

    # ── 其他高频无意义词 ────────────────────────────────────
    '什么', '怎么', '如何', '哪些', '为什么', '哪里', '哪个',
    '可以', '应该', '需要', '使用', '进行', '通过',

    # ── 代词/指示词 ─────────────────────────────────────────
    '我', '你', '他', '她', '它', '我们', '你们', '他们',
    '自己', '大家', '别人', '对方', '彼此', '相互',

    # ── 连词/助词 ───────────────────────────────────────────
    '但是', '而且', '因为', '所以', '如果', '虽然', '不过',
    '然而', '因此', '于是', '然后', '接着', '并且', '或者',
    '还是', '只要', '只有', '无论', '不管', '即使', '尽管',
    '不但', '不仅', '而且', '何况', '况且', '以致', '从而',

    # ── 量词/数词 ───────────────────────────────────────────
    '一', '二', '三', '四', '五', '六', '七', '八', '九', '十',
    '百', '千', '万', '亿', '几', '多', '少', '些', '点',
    '次', '遍', '回', '趟', '场', '件', '条', '只', '头', '匹',

    # ── 介词 ────────────────────────────────────────────────
    '按', '照', '比', '朝', '向', '往', '沿', '经', '过',
    '除', '离', '距', '凭', '据', '靠', '用', '拿', '趁',

    # ── 副词 ────────────────────────────────────────────────
    '很', '非常', '特别', '十分', '极其', '最', '更', '越',
    '太', '真', '确实', '实在', '简直', '几乎', '差不多',
    '稍微', '略微', '有点', '稍', '略', '颇', '甚',
    '就', '才', '刚', '正', '在', '已', '曾', '将', '要',
    '还', '再', '又', '也', '都', '总', '全', '仅', '只',
    '不', '没', '别', '未', '勿', '莫', '非', '无',

    # ── 动词虚化词 ──────────────────────────────────────────
    '做', '作', '使', '让', '叫', '请', '给', '带',
    '来', '去', '起', '出', '回', '过', '开', '上', '下',

    # ── 常见疑问/语气词 ────────────────────────────────────
    '吗', '呢', '吧', '啊', '呀', '哇', '哎', '哦',
    '嗯', '哈', '嘛', '呗', '咧', '喽', '嘞',

    # ── 补充高频无意义词 ────────────────────────────────────
    '时候', '地方', '样子', '东西', '办法', '方面', '问题',
    '情况', '关系', '条件', '原因', '结果', '目的', '意义',
    '作用', '影响', '特点', '性质', '内容', '形式', '方法',
    '过程', '状态', '程度', '范围', '部分', '整体', '结构',
    '功能', '效果', '价值', '水平', '质量', '标准', '规则',
    '原则', '基础', '根据', '来源', '类型', '种类', '类别',
    '概念', '观点', '态度', '立场', '角度', '方向', '目标',
})


# ═══════════════════════════════════════════════════════════════
#  领域名排除集合
# ═══════════════════════════════════════════════════════════════

DOMAIN_NAMES: frozenset[str] = frozenset({
    '医学', '物理', '计算机', '文学', '营养', '常识',
    '数学', '法律', '化学', '生物', '历史', '成语',
})


# ═══════════════════════════════════════════════════════════════
#  停用词加载
# ═══════════════════════════════════════════════════════════════

# 模块级缓存
_stopwords_cache: set[str] | None = None


def load_stopwords(stopwords_path: str | Path | None = None) -> set[str]:
    """
    加载停用词集合：内置停用词 + 从 data/stopwords.txt 加载扩展。

    Args:
        stopwords_path: 扩展停用词文件路径，默认使用 config.STOPWORDS_PATH

    Returns:
        完整停用词集合
    """
    global _stopwords_cache

    if _stopwords_cache is not None:
        return _stopwords_cache

    # 内置停用词为基础
    result = set(BUILTIN_STOP_WORDS)

    # 从扩展文件加载
    if stopwords_path is None:
        stopwords_path = Path(STOPWORDS_PATH)
    else:
        stopwords_path = Path(stopwords_path)

    if stopwords_path.exists():
        try:
            with open(stopwords_path, 'r', encoding='utf-8') as f:
                for line in f:
                    word = line.strip()
                    if word and word not in result:
                        result.add(word)
            log.debug("扩展停用词加载成功: %s", stopwords_path)
        except OSError as e:
            log.warning("停用词文件读取失败: %s，仅使用内置停用词", e)
    else:
        log.warning("停用词文件不存在: %s，仅使用内置停用词", stopwords_path)

    _stopwords_cache = result
    return result


def _reset_stopwords_cache() -> None:
    """重置停用词缓存（仅用于测试）"""
    global _stopwords_cache
    _stopwords_cache = None


# ═══════════════════════════════════════════════════════════════
#  关键词质量权重
# ═══════════════════════════════════════════════════════════════

def _keyword_quality_weight(keyword: str, graph: GraphDB) -> float:
    """
    计算关键词的质量权重。

    规则:
      - 长度 ≥ 4 且在知识库中存在对应种子 → 权重 1.5
      - 长度 ≥ 3 且在知识库中存在 → 权重 1.2
      - 其他 → 权重 1.0

    Args:
        keyword: 关键词
        graph: 知识图谱连接（用于检查种子是否存在）

    Returns:
        质量权重
    """
    seed = graph.get_seed(keyword)
    if seed is not None:
        if len(keyword) >= 4:
            return 1.5
        elif len(keyword) >= 3:
            return 1.2
    return 1.0


# ═══════════════════════════════════════════════════════════════
#  关键词提取 V2
# ═══════════════════════════════════════════════════════════════

def _extract_keywords_v2(text: str, graph: GraphDB) -> list[tuple[str, float]]:
    """
    从文本中提取中文关键词（升级版）。

    改进:
      - 停用词过滤（≥ 200 词）
      - 最小长度 ≥ 2（过滤单字词）
      - 领域名排除
      - 质量权重

    Args:
        text: 输入文本
        graph: 知识图谱连接

    Returns:
        [(keyword, weight), ...] 列表
    """
    stopwords = load_stopwords()

    # 提取中文连续段（长度 ≥ MIN_KEYWORD_LENGTH）
    chinese = re.findall(r'[\u4e00-\u9fff]{2,}', text)

    seen: set[str] = set()
    result: list[tuple[str, float]] = []

    for w in chinese:
        # 去重
        if w in seen:
            continue
        seen.add(w)

        # 停用词过滤
        if w in stopwords:
            continue

        # 最小长度过滤（单字词已在正则中过滤，这里确保 ≥ MIN_KEYWORD_LENGTH）
        if len(w) < MIN_KEYWORD_LENGTH:
            continue

        # 领域名排除
        if w in DOMAIN_NAMES:
            continue

        # 质量权重
        weight = _keyword_quality_weight(w, graph)
        result.append((w, weight))

    return result


# ═══════════════════════════════════════════════════════════════
#  旧版关键词提取（保留向后兼容）
# ═══════════════════════════════════════════════════════════════

def _extract_keywords(text: str) -> list[str]:
    """[DEPRECATED] 从文本中提取中文关键词（去停用词、去重）。

    此方法已被 _extract_keywords_v2() 替代，保留仅为向后兼容。
    """
    chinese = re.findall(r'[\u4e00-\u9fff]{2,}', text)
    seen = set()
    result = []
    for w in chinese:
        if w not in seen and w not in BUILTIN_STOP_WORDS:
            seen.add(w)
            result.append(w)
    return result


# ═══════════════════════════════════════════════════════════════
#  校验主函数
# ═══════════════════════════════════════════════════════════════

def verify(
    answer_text: str,
    result: RippleResult,
    graph: GraphDB,
) -> dict:
    """
    校验回答质量，返回置信度和熏习决策。

    Args:
        answer_text: 生成的回答文本（Phase 0 是检索式回答）
        result: 路由器返回的涟漪结果
        graph: 知识图谱连接（用于写回业力）

    Returns:
        {
            'confidence': float,      # 0~1
            'karma_direction': int,   # +1 正向, -1 负向, 0 不动
            'matched_keywords': int,
            'total_keywords': int,
            'decision': str,          # 'reinforce' | 'correct' | 'uncertain'
        }
    """
    from .config import CONFIDENCE_HIGH, CONFIDENCE_LOW

    # ── 1. 提取回答中的关键词（升级版） ──────────────────
    keywords = _extract_keywords_v2(answer_text, graph)
    if not keywords:
        return {
            'confidence': 0.5,
            'karma_direction': 0,
            'matched_keywords': 0,
            'total_keywords': 0,
            'decision': 'uncertain',
        }

    # ── 2. 匹配关键词到激活区域（加权） ───────────────────
    active_labels = set(result.activated.keys())
    weighted_matched = 0.0
    weighted_total = 0.0

    for kw, weight in keywords:
        weighted_total += weight
        matched = False

        # 精确匹配 + 别名匹配
        if kw in active_labels:
            matched = True
        else:
            seed = graph.get_seed(kw)
            if seed and seed.get('aliases'):
                import json
                try:
                    aliases = json.loads(seed['aliases'])
                    if any(a in active_labels for a in aliases):
                        matched = True
                except (json.JSONDecodeError, TypeError):
                    pass

        if matched:
            weighted_matched += weight

    # ── 3. 加权置信度计算 ─────────────────────────────────
    confidence = weighted_matched / weighted_total if weighted_total > 0 else 0.0

    # ── 4. 熏习决策 ───────────────────────────────────────
    if confidence >= CONFIDENCE_HIGH:
        karma_direction = +1
        decision = 'reinforce'
    elif confidence < CONFIDENCE_LOW:
        karma_direction = -1
        decision = 'correct'
    else:
        karma_direction = 0
        decision = 'uncertain'

    return {
        'confidence': round(confidence, 4),
        'karma_direction': karma_direction,
        'matched_keywords': int(weighted_matched),
        'total_keywords': len(keywords),
        'decision': decision,
    }


def apply_karma(
    result: RippleResult,
    graph: GraphDB,
    karma_direction: int,
    dry_run: bool = False,
) -> int:
    """
    应用熏习：对本次查询中 co-activated 的种子对修改业力。

    Args:
        result: 涟漪传播结果
        graph: 知识图谱连接
        karma_direction: +1 / -1 / 0
        dry_run: True 时只统计不写入

    Returns:
        实际修改的边数
    """
    from .config import KARMA_DELTA, KARMA_FULL_SET, KARMA_TOP_N

    if karma_direction == 0:
        return 0

    delta = KARMA_DELTA * karma_direction

    # 选取要熏习的种子
    if KARMA_FULL_SET:
        # Phase 0: 所有 co-activated pairs
        targets = sorted(
            result.activated.values(),
            key=lambda n: n.activation, reverse=True
        )
    else:
        # Phase 2: 只选 Top-N
        targets = sorted(
            result.activated.values(),
            key=lambda n: n.activation, reverse=True
        )[:KARMA_TOP_N]

    modified = 0
    # 对 targets 中的所有 pair 修改业力
    # （只熏在本次查询中被涟漪传播路径覆盖的边，而非所有可能的 pairs）
    seen_pairs = set()
    for path in result.paths:
        pair = (path['source'], path['target'])
        if pair not in seen_pairs:
            seen_pairs.add(pair)
            if not dry_run:
                graph.adjust_karma(
                    path['source'], path['target'],
                    relation=path['relation'],
                    delta=delta,
                )
            modified += 1

    if not dry_run:
        graph.conn.commit()

    return modified
