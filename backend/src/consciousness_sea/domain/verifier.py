"""
答案校验器 + 熏习引擎

校验: 回答关键词 vs 激活种子一致性 → 置信度
熏习: 置信度 → 决定业力偏移方向 (+0.01 / 0 / -0.01)

闭环:
  查询 → 路由 → 回答 → 校验 → 置信度 → 熏习 ↶
"""

from __future__ import annotations

import json
import logging
import re
from pathlib import Path

from consciousness_sea.infrastructure.config import MIN_KEYWORD_LENGTH, STOPWORDS_PATH

from .graph_db import GraphDB
from .router import RippleResult

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
#  校验主函数
# ═══════════════════════════════════════════════════════════════

def verify(
    answer_text: str,
    result: RippleResult,
    graph: GraphDB,
    *,
    expert_domain: str | None = None,
    reliability: float = 1.0,
    cv_discount: float = 1.0,
) -> dict:
    """
    校验回答质量，返回置信度和熏习决策。

    Args:
        answer_text: 生成的回答文本（Phase 0 是检索式回答）
        result: 路由器返回的涟漪结果
        graph: 知识图谱连接（用于写回业力）
        expert_domain: 专家领域名（Phase 1 可选）
        reliability: 专家可靠性分数 [0.0, 1.0]（默认 1.0，Phase 0 不打折）
        cv_discount: 交叉验证折扣系数 [0.0, 1.0]（默认 1.0，无打折）

    Returns:
        {
            'confidence': float,      # 0~1 (actual = raw × cv_discount × reliability)
            'raw_confidence': float,  # 原始置信度（Phase 0 时等于 confidence）
            'karma_direction': int,   # +1 正向, -1 负向, 0 不动
            'matched_keywords': int,
            'total_keywords': int,
            'decision': str,          # 'reinforce' | 'correct' | 'uncertain'
            'expert_domain': str | None,
            'reliability': float,
            'cv_discount': float,
        }
    """
    from consciousness_sea.infrastructure.config import CONFIDENCE_HIGH, CONFIDENCE_LOW

    # ── 1. 提取回答中的关键词（升级版） ──────────────────
    keywords = _extract_keywords_v2(answer_text, graph)
    if not keywords:
        return {
            'confidence': 0.5,
            'raw_confidence': 0.5,
            'karma_direction': 0,
            'matched_keywords': 0,
            'total_keywords': 0,
            'decision': 'uncertain',
            'expert_domain': expert_domain,
            'reliability': reliability,
            'cv_discount': cv_discount,
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
                try:
                    aliases = json.loads(seed['aliases'])
                    if any(a in active_labels for a in aliases):
                        matched = True
                except (json.JSONDecodeError, TypeError):
                    pass

        if matched:
            weighted_matched += weight

    # ── 3. 加权置信度计算 ─────────────────────────────────
    raw_confidence = weighted_matched / weighted_total if weighted_total > 0 else 0.0

    # 可靠性加权: actual_confidence = raw_confidence × cv_discount × reliability
    actual_confidence = raw_confidence * cv_discount * reliability

    # ── 4. 熏习决策（基于 actual_confidence） ─────────────
    if actual_confidence >= CONFIDENCE_HIGH:
        karma_direction = +1
        decision = 'reinforce'
    elif actual_confidence < CONFIDENCE_LOW:
        karma_direction = -1
        decision = 'correct'
    else:
        karma_direction = 0
        decision = 'uncertain'

    return {
        'confidence': round(actual_confidence, 4),
        'raw_confidence': round(raw_confidence, 4),
        'karma_direction': karma_direction,
        'matched_keywords': int(weighted_matched),
        'total_keywords': len(keywords),
        'decision': decision,
        'expert_domain': expert_domain,
        'reliability': reliability,
        'cv_discount': cv_discount,
    }


def apply_karma(
    result: RippleResult,
    graph: GraphDB,
    karma_direction: int,
    dry_run: bool = False,
    user_label: str | None = None,
    answer_text: str | None = None,
    verify_result: dict | None = None,
) -> int:
    """
    应用熏习：对本次查询中 co-activated 的种子对修改业力。

    Phase 2 变更:
      - Top-N 种子筛选（KARMA_FULL_SET=False 时）
      - 路径级过滤：仅 source 和 target 均在 Top-N 集合内的路径
      - KARMA_MAX_PAIRS 上限保护
      - 双层业力写入（user_label 不为 None 时写入个人层）
      - 提炼池候选提交

    Args:
        result: 涟漪传播结果
        graph: 知识图谱连接
        karma_direction: +1 / -1 / 0
        dry_run: True 时只统计不写入
        user_label: 用户标识（可选，Phase 2 新增）
        answer_text: 专家答案文本（可选，Phase 3 新增，用于别名回指和候选种子提取）

    Returns:
        实际修改的边数
    """
    from consciousness_sea.infrastructure.config import (
        KARMA_DELTA,
        KARMA_FULL_SET,
        KARMA_MAX_PAIRS,
        KARMA_TOP_N,
    )

    if karma_direction == 0:
        return 0

    delta = KARMA_DELTA * karma_direction

    # ── 1. 选取 Top-N 种子 ──────────────────────────────
    if KARMA_FULL_SET:
        # Phase 0: 所有 co-activated pairs（向后兼容）
        target_labels = set(result.activated.keys())
    else:
        # Phase 2: 只选 Top-N
        top_nodes = sorted(
            result.activated.values(),
            key=lambda n: n.activation, reverse=True
        )[:KARMA_TOP_N]
        target_labels = {n.label for n in top_nodes}

    # ── 2. 路径级筛选：仅 source 和 target 均在 target_labels 中的路径 ──
    filtered_paths = []
    seen_pairs = set()
    for path in result.paths:
        pair = (path['source'], path['target'])
        if pair in seen_pairs:
            continue
        if path['source'] in target_labels and path['target'] in target_labels:
            seen_pairs.add(pair)
            filtered_paths.append(path)

    # ── 3. 上限保护（KARMA_MAX_PAIRS） ──────────────────
    if len(filtered_paths) > KARMA_MAX_PAIRS:
        filtered_paths = filtered_paths[:KARMA_MAX_PAIRS]

    # ── 4. 执行熏习 ────────────────────────────────────
    modified = 0
    for path in filtered_paths:
        if not dry_run:
            # Phase 2: 双层业力写入
            if user_label:
                graph.adjust_karma_personal(
                    user_label, path['source'], path['target'],
                    relation=path['relation'], delta=delta,
                )
            else:
                graph.adjust_karma_atomic(
                    path['source'], path['target'],
                    relation=path['relation'], delta=delta,
                )
        modified += 1

    # ── 5. 提炼池候选提交（Phase 2） ───────────────────
    if not dry_run and user_label and karma_direction != 0:
        try:
            from consciousness_sea.learning.distillation_pool import DistillationPool
            distill = DistillationPool(graph)
            for path in filtered_paths:
                distill.submit_candidate(
                    user_label=user_label,
                    source=path['source'],
                    target=path['target'],
                    relation=path['relation'],
                )
        except Exception as e:
            # 提炼池提交失败不影响熏习结果
            log.warning("提炼池候选提交失败: %s", e)

    # ── 6. commit 重试逻辑 ──────────────────────────────
    if not dry_run:
        _commit_with_retry(graph)

    # ── 7. Phase 3 后处理 ──────────────────────────────
    if not dry_run and karma_direction != 0:
        _post_karma_phase3(result, graph, user_label, answer_text)

    # ── 8. Phase 4 后处理 ──────────────────────────────
    if not dry_run and karma_direction != 0:
        _post_karma_phase4(result, graph, verify_result or {})

    # ── 9. Phase 6 后处理 ──────────────────────────────
    if not dry_run and karma_direction != 0:
        _post_karma_phase6(result, graph, verify_result or {})

    return modified


def _commit_with_retry(graph: GraphDB, max_retries: int = 3) -> None:
    """commit 重试逻辑 — 失败后先 rollback 再重试

    SQLite 不支持重试失败的 commit，必须先 rollback。
    delta 只有 0.01，丢失可接受。
    """
    for attempt in range(1, max_retries + 1):
        try:
            graph.conn.commit()
            break
        except Exception as e:
            # 先 rollback，清除失败的事务状态
            try:
                graph.conn.rollback()
            except Exception:
                pass
            if attempt < max_retries:
                log.warning("commit 失败 (第 %d 次)，已 rollback，重试中: %s", attempt, e)
            else:
                # delta 只有 0.01，丢失可接受
                log.warning("commit 重试 %d 次后仍失败，跳过本次熏习: %s", max_retries, e)


def _post_karma_phase3(
    result: RippleResult,
    graph: GraphDB,
    user_label: str | None = None,
    answer_text: str | None = None,
) -> None:
    """Phase 3 后处理：别名回指记录 + 候选种子提取"""
    # 1. 别名回指记录
    try:
        if answer_text:
            from consciousness_sea.learning.alias_expander import AliasExpander
            events, unmatched = _extract_backref_events(result, answer_text, graph)
            if events or unmatched:
                expander = AliasExpander(graph)
                results = expander.record_backref_events(events, unmatched)
                for r in results:
                    if r.action == 'aliased':
                        log.info("alias auto-extended: '%s' → seed '%s'", r.keyword, r.seed_label)
    except Exception as e:
        log.warning("别名回指记录失败: %s", e)

    # 2. 候选种子提取
    try:
        if answer_text:
            from consciousness_sea.learning.seed_candidate import SeedCandidateManager
            unmatched = _extract_unmatched_keywords(result, answer_text, graph)
            if unmatched:
                co_occur = list(result.activated.keys())[:10]
                manager = SeedCandidateManager(graph)
                manager.process_unmatched_keywords(unmatched, co_occur)
    except Exception as e:
        log.warning("候选种子提取失败: %s", e)


def _extract_backref_events(
    result: RippleResult,
    answer_text: str,
    graph: GraphDB,
) -> tuple[list, list[str]]:
    """从涟漪结果和专家答案中提取回指事件"""
    from consciousness_sea.learning.alias_expander import BackrefEvent

    events = []
    unmatched = []
    active_labels = set(result.activated.keys())

    # 从答案中提取关键词
    keywords = _extract_keywords_v2(answer_text, graph)

    for kw, _ in keywords:
        if kw in active_labels:
            continue  # 已匹配种子，跳过

        # 检查是否通过别名匹配到种子
        seed = graph.get_seed(kw)
        if seed and seed['label'] in active_labels:
            # 回指事件：kw → seed_label
            events.append(BackrefEvent(source_keyword=kw, target_seed=seed['label']))
        else:
            # 未匹配关键词
            unmatched.append(kw)

    return events, unmatched


def _extract_unmatched_keywords(
    result: RippleResult,
    answer_text: str | None,
    graph: GraphDB,
) -> list[str]:
    """从查询结果中提取未匹配关键词"""
    if not answer_text:
        return []

    active_labels = set(result.activated.keys())
    keywords = _extract_keywords_v2(answer_text, graph)

    unmatched = []
    for kw, _ in keywords:
        if len(kw) < MIN_KEYWORD_LENGTH:
            continue
        if kw in active_labels:
            continue
        # 检查是否通过别名匹配到种子
        seed = graph.get_seed(kw)
        if seed:
            continue  # 已通过别名关联
        unmatched.append(kw)

    return unmatched


def _post_karma_phase4(
    result: RippleResult,
    graph: GraphDB,
    verify_result: dict,
) -> None:
    """Phase 4 后处理：元种子指标更新 + 目标触及

    在熏习完成后执行，不阻塞查询响应。
    异常时仅记录 WARNING 日志，不影响查询结果。
    """
    from consciousness_sea.infrastructure.config import COGNITIVE_GOAL_ENABLED, META_SEED_ENABLED

    if not META_SEED_ENABLED:
        return

    try:
        from consciousness_sea.metacognition.meta_seed import MetaSeedManager
        mgr = MetaSeedManager(graph)

        karma_direction = verify_result.get('karma_direction', 0)
        expert_domain = verify_result.get('expert_domain')

        # 1. 更新领域元种子 conflict_frequency
        if expert_domain and karma_direction == -1:
            meta_label = f"meta:{expert_domain}"
            mgr.increment_metric(meta_label, "conflict_frequency", delta=1)

        # 2. [Phase 5 新增] 更新认知目标触及时间
        if COGNITIVE_GOAL_ENABLED and expert_domain:
            try:
                from consciousness_sea.metacognition.cognitive_goal import CognitiveGoalManager
                goal_mgr = CognitiveGoalManager(graph)
                goal_mgr.touch_goal_domain(expert_domain)
            except Exception as e:
                log.warning("目标触及更新失败: %s", e)

    except Exception as e:
        log.warning("元种子指标更新失败: %s", e)


def _post_karma_phase6(
    result: RippleResult,
    graph: GraphDB,
    verify_result: dict,
) -> None:
    """Phase 6 后处理：概念种子激活事件通知 Hebbian 绑定器

    在熏习完成后执行，不阻塞查询响应。
    异常时仅记录 WARNING 日志，不影响查询结果。

    注意: 概念激活事件通过 api.py 中的 _perception_manager 单例转发，
    不写入 perception_events 表（该表仅用于感知激活事件）。
    """
    from consciousness_sea.infrastructure.config import PERCEPTION_ENABLED

    if not PERCEPTION_ENABLED:
        return

    try:
        # 获取激活的概念种子列表
        activated_seeds = list(result.activated.keys())
        if not activated_seeds:
            return

        # 构造概念种子激活事件
        from datetime import datetime, timezone

        from consciousness_sea.perception.perception import ConceptActivationEvent
        now = datetime.now(timezone.utc).isoformat(timespec='milliseconds')

        event = ConceptActivationEvent(
            activated_seeds=activated_seeds,
            timestamp=now,
        )

        # 通过模块级变量间接通知 PerceptionManager
        # 在 api.py 的 query_endpoint 中，熏习完成后会调用此函数，
        # 而 api.py 持有 _perception_manager 单例，可以在那里转发事件
        # 此处将事件暂存到 verify_result 中，由 api.py 转发
        verify_result["_concept_activation_event"] = event

    except Exception as e:
        log.warning("Phase 6 后处理失败: %s", e)
