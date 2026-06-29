"""
tokenizer — 升级版分词器

替代 graph_db._tokenize() 的简陋实现，提供：
- 组合词优先匹配（最大正向匹配算法）
- 同义词/别名扩展
- 模糊匹配（编辑距离）
- 否定词识别

不依赖 jieba 或任何第三方分词库。
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass

from consciousness_sea.infrastructure.config import MAX_COMPOUND_LEN, NEGATION_SCOPE

log = logging.getLogger(__name__)

# ═══════════════════════════════════════════════════════════════
#  预编译正则表达式
# ═══════════════════════════════════════════════════════════════

_RE_CHINESE_SPAN = re.compile(r'[\u4e00-\u9fff]+')
_RE_ALPHANUM = re.compile(r'[a-zA-Z0-9]+')

# ═══════════════════════════════════════════════════════════════
#  否定词表
# ═══════════════════════════════════════════════════════════════

NEGATION_WORDS: frozenset[str] = frozenset({
    '不是', '并非', '没有', '不会', '不能', '不要', '并非是',
    '不', '没', '非', '无', '别', '莫', '勿',
})

# 反问标记（Phase 1 后续处理，暂不参与否定逻辑）
RHETORICAL_MARKERS: frozenset[str] = frozenset({'怎么', '难道', '岂', '何'})


# ═══════════════════════════════════════════════════════════════
#  数据类
# ═══════════════════════════════════════════════════════════════

@dataclass
class TokenMatch:
    """分词匹配结果

    Attributes:
        text: 原始查询词
        match_type: 匹配类型 — 'exact' | 'alias' | 'fuzzy' | 'unmatched'
        seed_label: 匹配到的种子 label（unmatched 时为空字符串）
        excluded: 是否被否定词排除
    """
    text: str
    match_type: str  # 'exact' | 'alias' | 'fuzzy' | 'unmatched'
    seed_label: str = ''
    excluded: bool = False

    def __post_init__(self):
        # unmatched 时确保 seed_label 为空
        if self.match_type == 'unmatched':
            object.__setattr__(self, 'seed_label', '')


# ═══════════════════════════════════════════════════════════════
#  最大正向匹配
# ═══════════════════════════════════════════════════════════════

def _max_forward_match(
    span: str,
    label_index: set[str],
    alias_index: dict[str, str] | None = None,
    max_len: int = MAX_COMPOUND_LEN,
) -> list[TokenMatch]:
    """对单个中文 span 执行最大正向匹配。

    算法：
      1. 若 span 整体在 label_index 中 → 整体匹配，跳过拆分
      2. 否则从 pos=0 开始，逐步尝试从最大长度到 1 的子串匹配
      3. 匹配时同时检查 label_index（精确匹配）和 alias_index（别名匹配）
      4. 完全无法匹配的单字标记为 unmatched

    Args:
        span: 中文连续段
        label_index: 所有种子 label 的集合
        alias_index: alias → seed_label 的映射，可选
        max_len: 最大匹配长度（字符数）

    Returns:
        匹配结果列表
    """
    if not span:
        return []

    # 整体匹配：span 本身就是知识库中的种子
    if span in label_index:
        return [TokenMatch(text=span, match_type='exact', seed_label=span)]

    # 整体别名匹配
    if alias_index and span in alias_index:
        return [TokenMatch(text=span, match_type='alias', seed_label=alias_index[span])]

    results: list[TokenMatch] = []
    pos = 0
    span_len = len(span)

    while pos < span_len:
        # 当前窗口最大匹配长度
        window = min(max_len, span_len - pos)
        matched = False

        while window >= 1:
            candidate = span[pos:pos + window]
            # 优先精确匹配
            if candidate in label_index:
                results.append(
                    TokenMatch(text=candidate, match_type='exact', seed_label=candidate)
                )
                pos += window
                matched = True
                break
            # 其次别名匹配
            if alias_index and candidate in alias_index:
                results.append(
                    TokenMatch(text=candidate, match_type='alias', seed_label=alias_index[candidate])
                )
                pos += window
                matched = True
                break
            window -= 1

        if not matched:
            # 单字无法匹配 → unmatched
            results.append(
                TokenMatch(text=span[pos], match_type='unmatched')
            )
            pos += 1

    return results


# ═══════════════════════════════════════════════════════════════
#  别名匹配
# ═══════════════════════════════════════════════════════════════

def match_with_aliases(
    token: str,
    alias_index: dict[str, str],
) -> TokenMatch | None:
    """通过别名索引匹配种子。

    Args:
        token: 待匹配的词
        alias_index: alias → seed_label 的映射

    Returns:
        匹配成功返回 TokenMatch(match_type='alias')，否则 None
    """
    if token in alias_index:
        return TokenMatch(
            text=token,
            match_type='alias',
            seed_label=alias_index[token],
            excluded=False,
        )
    return None


# ═══════════════════════════════════════════════════════════════
#  模糊匹配（编辑距离）
# ═══════════════════════════════════════════════════════════════

def _levenshtein(s1: str, s2: str) -> int:
    """计算两个字符串之间的编辑距离（Levenshtein distance）。

    使用动态规划，时间复杂度 O(m*n)，空间复杂度 O(n)。

    Args:
        s1: 第一个字符串
        s2: 第二个字符串

    Returns:
        编辑距离（非负整数）
    """
    m, n = len(s1), len(s2)

    # 快速排除：长度差超过 1 时，编辑距离至少为长度差
    # 调用方已做长度差预筛选，此处保留通用逻辑

    # 退化情况
    if m == 0:
        return n
    if n == 0:
        return m

    # 使用一维 DP 数组，空间优化为 O(n)
    dp = list(range(n + 1))

    for i in range(1, m + 1):
        prev = dp[0]  # dp[i-1][j-1]
        dp[0] = i     # dp[i][0] = i

        for j in range(1, n + 1):
            temp = dp[j]  # 保存 dp[i-1][j] 供下一轮使用
            if s1[i - 1] == s2[j - 1]:
                dp[j] = prev
            else:
                dp[j] = 1 + min(prev, dp[j], dp[j - 1])
            prev = temp

    return dp[n]


def fuzzy_match(
    token: str,
    label_index: set[str],
    max_distance: int = 1,
    edge_count_map: dict[str, int] | None = None,
    label_length_buckets: dict[int, set[str]] | None = None,
) -> TokenMatch | None:
    """编辑距离 ≤ max_distance 的模糊匹配。

    优化策略：
      - 长度分桶：如果提供了 label_length_buckets，只搜索长度在
        [token_len - max_distance, token_len + max_distance] 范围内的桶
      - 只对长度差 ≤ max_distance 的候选词计算编辑距离
      - 多个候选时选择在知识库中出边数最多的种子（需要 edge_count_map）

    Args:
        token: 待匹配的词
        label_index: 所有种子 label 的集合
        max_distance: 最大允许编辑距离
        edge_count_map: seed_label → 出边数 的映射（用于歧义消解），可选
        label_length_buckets: 按长度分桶的 label 索引 {len: {label1, label2, ...}}，可选

    Returns:
        匹配成功返回 TokenMatch(match_type='fuzzy')，否则 None
    """
    candidates: list[str] = []
    token_len = len(token)

    if label_length_buckets is not None:
        # 长度分桶优化：只搜索长度差在 max_distance 以内的桶
        for bucket_len in range(token_len - max_distance, token_len + max_distance + 1):
            bucket = label_length_buckets.get(bucket_len)
            if bucket is None:
                continue
            for label in bucket:
                if _levenshtein(token, label) <= max_distance:
                    candidates.append(label)
    else:
        # 无分桶时回退到全量遍历（性能警告）
        log.warning(
            "fuzzy_match 未提供 label_length_buckets，回退到全量遍历，"
            "label_index 大小=%d，建议调用方构建并传入分桶索引",
            len(label_index),
        )
        for label in label_index:
            # 长度差预筛选：编辑距离至少等于长度差
            if abs(len(label) - token_len) > max_distance:
                continue
            if _levenshtein(token, label) <= max_distance:
                candidates.append(label)

    if not candidates:
        return None

    # 歧义消解：选择出边数最多的种子
    if edge_count_map and len(candidates) > 1:
        best = max(candidates, key=lambda lbl: edge_count_map.get(lbl, 0))
    else:
        # 无 edge_count_map 或只有一个候选，取第一个
        best = candidates[0]

    return TokenMatch(
        text=token,
        match_type='fuzzy',
        seed_label=best,
        excluded=False,
    )


def _try_combine_fuzzy(
    tokens: list[TokenMatch],
    label_index: set[str],
    max_distance: int = 1,
    edge_count_map: dict[str, int] | None = None,
    label_length_buckets: dict[int, set[str]] | None = None,
) -> None:
    """对连续的 unmatched 中文单字尝试组合模糊匹配。

    例如 tokens 中连续出现 [unmatched:'感', unmatched:'昌']，
    组合为 '感昌' 后模糊匹配到 '感冒'。

    匹配成功时，将第一个 token 替换为组合结果，后续 token 标记为
    空 TokenMatch（后续清理时移除）。

    Args:
        tokens: TokenMatch 列表（原地修改）
        label_index: 所有种子 label 的集合
        max_distance: 最大编辑距离
        edge_count_map: 出边数映射（歧义消解用），可选
        label_length_buckets: 按长度分桶的 label 索引，可选
    """
    i = 0
    while i < len(tokens):
        if tokens[i].match_type != 'unmatched':
            i += 1
            continue

        # 收集从位置 i 开始的连续 unmatched 中文单字
        j = i
        while j < len(tokens) and tokens[j].match_type == 'unmatched' and len(tokens[j].text) == 1 and _RE_CHINESE_SPAN.fullmatch(tokens[j].text):
            j += 1

        consecutive_count = j - i
        if consecutive_count < 2:
            i += 1
            continue

        # 尝试不同长度的组合（从长到短），寻找模糊匹配
        combined = ''.join(tokens[k].text for k in range(i, j))
        found = False

        for length in range(min(consecutive_count, MAX_COMPOUND_LEN), 1, -1):
            for start in range(consecutive_count - length + 1):
                candidate = combined[start:start + length]
                fuzzy_result = fuzzy_match(
                    candidate, label_index,
                    max_distance=max_distance,
                    edge_count_map=edge_count_map,
                    label_length_buckets=label_length_buckets,
                )
                if fuzzy_result:
                    # 替换第一个 token 为组合结果
                    tokens[i + start] = fuzzy_result
                    # 将被合并的后续 token 标记为空
                    for k in range(i + start + 1, i + start + length):
                        tokens[k] = TokenMatch(text='', match_type='unmatched')
                    found = True
                    break
            if found:
                break

        i = j

    # 移除空 token
    empty_indices = [idx for idx, t in enumerate(tokens) if t.text == '' and t.match_type == 'unmatched']
    for idx in reversed(empty_indices):
        tokens.pop(idx)


# ═══════════════════════════════════════════════════════════════
#  否定词识别
# ═══════════════════════════════════════════════════════════════

def _split_by_negation(span: str) -> list[str]:
    """按否定词切分中文 span，返回不含否定词的子段列表。

    否定词本身不作为子段返回（它们会在 detect_negation 阶段
    通过原始 query 文本的位置信息来识别）。

    例如：
      "不是感冒" → ["感冒"]
      "没感冒" → ["感冒"]
      "感冒不是发热" → ["感冒", "发热"]

    Args:
        span: 中文连续段

    Returns:
        切分后的子段列表（不含否定词本身）
    """
    # 按否定词长度降序排列，优先匹配长否定词
    sorted_neg_words = sorted(NEGATION_WORDS, key=len, reverse=True)

    # 找到所有否定词的位置
    neg_ranges: list[tuple[int, int]] = []  # (start, end) 左闭右开
    for word in sorted_neg_words:
        idx = span.find(word)
        while idx != -1:
            # 检查是否与已找到的否定词范围重叠
            end = idx + len(word)
            overlaps = any(s < end and e > idx for s, e in neg_ranges)
            if not overlaps:
                neg_ranges.append((idx, end))
            idx = span.find(word, idx + 1)

    if not neg_ranges:
        return [span]

    # 按位置排序
    neg_ranges.sort()

    # 提取否定词之间的子段
    result: list[str] = []
    prev_end = 0
    for start, end in neg_ranges:
        if start > prev_end:
            sub = span[prev_end:start]
            if sub:
                result.append(sub)
        prev_end = end

    # 最后一段
    if prev_end < len(span):
        sub = span[prev_end:]
        if sub:
            result.append(sub)

    return result


def detect_negation(query: str, tokens: list[TokenMatch]) -> list[TokenMatch]:
    """识别否定词并标记排除语义。

    规则：
      - 否定词后的第一个匹配种子标记为 excluded=True
      - 作用范围：否定词后 NEGATION_SCOPE（默认 4）个字符以内

    Args:
        query: 原始查询文本
        tokens: 已完成匹配的 TokenMatch 列表

    Returns:
        更新了 excluded 标记的 tokens 列表（原地修改并返回）
    """
    # 收集所有否定词的结束位置
    negation_end_positions: list[int] = []

    for word in NEGATION_WORDS:
        idx = query.find(word)
        while idx != -1:
            # 否定词结束位置（即否定词后第一个字符的索引）
            negation_end_positions.append(idx + len(word))
            idx = query.find(word, idx + 1)

    if not negation_end_positions:
        return tokens

    # 对每个 token，检查其起始位置是否在某个否定词的作用范围内
    for token in tokens:
        if token.match_type == 'unmatched':
            continue  # 未匹配的词无需标记排除

        # 查找 token 在 query 中的位置
        token_start = query.find(token.text)
        if token_start == -1:
            continue

        for neg_end in negation_end_positions:
            # 否定词后 NEGATION_SCOPE 个字符以内
            if neg_end <= token_start < neg_end + NEGATION_SCOPE:
                token.excluded = True
                break  # 一个 token 只需被一个否定词标记

    return tokens


# ═══════════════════════════════════════════════════════════════
#  主函数：完整分词流程
# ═══════════════════════════════════════════════════════════════

def tokenize(
    query: str,
    label_index: set[str],
    alias_index: dict[str, str],
    enable_fuzzy: bool = True,
    max_edit_distance: int = 1,
    edge_count_map: dict[str, int] | None = None,
    label_length_buckets: dict[int, set[str]] | None = None,
) -> list[TokenMatch]:
    """完整分词流程。

    算法流程：
      1. 提取中文连续段 → 最大正向匹配
      2. 对未精确匹配的词尝试别名匹配
      3. 对仍无匹配的词尝试模糊匹配（可选）
      4. 英文/数字词匹配
      5. 否定词识别（在所有匹配完成后执行）

    Args:
        query: 查询文本
        label_index: 所有种子 label 的集合
        alias_index: alias → seed_label 的映射
        enable_fuzzy: 是否启用模糊匹配，默认 True
        max_edit_distance: 模糊匹配最大编辑距离，默认 1
        edge_count_map: seed_label → 出边数 的映射（模糊匹配歧义消解用），可选
        label_length_buckets: 按长度分桶的 label 索引 {len: {label1, label2, ...}}，可选

    Returns:
        TokenMatch 列表
    """
    if not query or not query.strip():
        return []

    tokens: list[TokenMatch] = []

    # ── Step 1: 中文连续段 → 否定词切分 → 匹配 ──
    chinese_spans = _RE_CHINESE_SPAN.findall(query)
    for span in chinese_spans:
        # 先按否定词切分 span，保留否定词位置信息供后续 detect_negation 使用
        sub_spans = _split_by_negation(span)
        for sub in sub_spans:
            if not sub:
                continue

            # 优先检查整体是否在 label_index 中（精确匹配）
            if sub in label_index:
                tokens.append(
                    TokenMatch(text=sub, match_type='exact', seed_label=sub)
                )
                continue

            # 其次检查整体是否在 alias_index 中（别名匹配）
            # 这避免了"着凉"被拆成"着"+"凉"再逐字别名匹配的问题
            alias_result = match_with_aliases(sub, alias_index)
            if alias_result:
                tokens.append(alias_result)
                continue

            # 尝试整体模糊匹配（仅当启用模糊匹配且 sub 长度合理时）
            # 注意：由于已按否定词切分，sub 中不包含否定词
            if enable_fuzzy and 2 <= len(sub) <= MAX_COMPOUND_LEN:
                fuzzy_result = fuzzy_match(
                    sub, label_index,
                    max_distance=max_edit_distance,
                    edge_count_map=edge_count_map,
                    label_length_buckets=label_length_buckets,
                )
                if fuzzy_result:
                    tokens.append(fuzzy_result)
                    continue

            # 整体都不匹配 → 最大正向匹配拆分
            span_tokens = _max_forward_match(sub, label_index, alias_index)
            tokens.extend(span_tokens)

    # ── Step 2: 别名匹配（对拆分后仍为 unmatched 的 token） ──
    for i, token in enumerate(tokens):
        if token.match_type == 'unmatched':
            alias_result = match_with_aliases(token.text, alias_index)
            if alias_result:
                tokens[i] = alias_result

    # ── Step 3: 模糊匹配（仅对仍为 unmatched 的中文 token） ──
    if enable_fuzzy:
        # 3a: 对连续的 unmatched 中文单字尝试组合模糊匹配
        # 例如 "感"+"昌" 组合为 "感昌" → 模糊匹配到 "感冒"
        _try_combine_fuzzy(tokens, label_index, max_edit_distance, edge_count_map, label_length_buckets)

        # 3b: 对仍为 unmatched 的单个中文 token 尝试模糊匹配
        for i, token in enumerate(tokens):
            if token.match_type == 'unmatched' and _RE_CHINESE_SPAN.fullmatch(token.text):
                fuzzy_result = fuzzy_match(
                    token.text,
                    label_index,
                    max_distance=max_edit_distance,
                    edge_count_map=edge_count_map,
                    label_length_buckets=label_length_buckets,
                )
                if fuzzy_result:
                    tokens[i] = fuzzy_result

    # ── Step 4: 英文/数字词匹配 ──
    alphanum_words = _RE_ALPHANUM.findall(query)
    for word in alphanum_words:
        if word in label_index:
            tokens.append(
                TokenMatch(text=word, match_type='exact', seed_label=word)
            )
        else:
            # 英文词也尝试别名匹配
            alias_result = match_with_aliases(word, alias_index)
            if alias_result:
                tokens.append(alias_result)
            else:
                tokens.append(
                    TokenMatch(text=word, match_type='unmatched')
                )

    # ── Step 5: 否定词识别（所有匹配完成后执行） ──
    tokens = detect_negation(query, tokens)

    return tokens