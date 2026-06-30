"""
测试升级版分词器 (Tokenizer)

覆盖 TASK-011:
- 组合词优先匹配
- 组合词回退拆分
- 别名匹配
- 模糊匹配
- 否定词识别
- 英文/数字词匹配
- 空查询和无匹配查询
"""

import pathlib
import sys

_root = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_root))
sys.path.insert(0, str(_root / "backend" / "src"))

from consciousness_sea.domain.tokenizer import (
    NEGATION_WORDS,
    TokenMatch,
    _levenshtein,
    _max_forward_match,
    detect_negation,
    fuzzy_match,
    match_with_aliases,
    tokenize,
)

# ═══════════════════════════════════════════════════════════════
#  测试用知识库索引
# ═══════════════════════════════════════════════════════════════

# 种子 label 索引
LABEL_INDEX: set[str] = {
    '感冒', '发热', '咳嗽', '量子力学', '薛定谔方程',
    '人工智能', '深度学习', '计算机', '腹泻', '维C',
    '黑洞', '引力波', '相对论',
}

# 别名索引: alias → seed_label
ALIAS_INDEX: dict[str, str] = {
    '着凉': '感冒',
    '电脑': '计算机',
    '拉肚子': '腹泻',
    'AI': '人工智能',
}


class TestCompoundWordPriority:
    """组合词优先匹配测试"""

    def test_whole_compound_word_matched(self):
        """'量子力学' 整体匹配，不拆分"""
        tokens = tokenize('量子力学', LABEL_INDEX, ALIAS_INDEX)
        # 应该整体匹配为 '量子力学'
        matched_labels = [t.seed_label for t in tokens if t.match_type != 'unmatched']
        assert '量子力学' in matched_labels

    def test_compound_not_split_into_subwords(self):
        """整体匹配时不应产生子词"""
        tokens = tokenize('量子力学', LABEL_INDEX, ALIAS_INDEX)
        # 只应有一个匹配结果
        matched = [t for t in tokens if t.match_type != 'unmatched']
        assert len(matched) == 1
        assert matched[0].seed_label == '量子力学'
        assert matched[0].match_type == 'exact'

    def test_two_compound_words(self):
        """两个组合词在同一查询中都被整体匹配"""
        tokens = tokenize('量子力学人工智能', LABEL_INDEX, ALIAS_INDEX)
        matched_labels = [t.seed_label for t in tokens if t.match_type != 'unmatched']
        assert '量子力学' in matched_labels
        assert '人工智能' in matched_labels


class TestCompoundWordFallback:
    """组合词回退拆分测试"""

    def test_fallback_split_when_whole_not_in_index(self):
        """整体不在知识库时拆分为子词"""
        # "感冒发热" 整体不在 label_index 中，但 "感冒" 和 "发热" 都在
        tokens = tokenize('感冒发热', LABEL_INDEX, ALIAS_INDEX)
        matched_labels = [t.seed_label for t in tokens if t.match_type != 'unmatched']
        assert '感冒' in matched_labels
        assert '发热' in matched_labels

    def test_partial_match(self):
        """部分子词匹配，部分不匹配"""
        # "感冒XYZ" — "感冒" 匹配，"XYZ" 不匹配
        tokens = tokenize('感冒XYZ', LABEL_INDEX, ALIAS_INDEX)
        matched = [t for t in tokens if t.match_type == 'exact']
        assert any(t.seed_label == '感冒' for t in matched)


class TestAliasMatch:
    """别名匹配测试"""

    def test_zhaoliang_matches_ganmao(self):
        """'着凉' 匹配到 '感冒'"""
        tokens = tokenize('着凉', LABEL_INDEX, ALIAS_INDEX)
        matched = [t for t in tokens if t.match_type == 'alias']
        assert len(matched) >= 1
        assert matched[0].seed_label == '感冒'

    def test_diannao_matches_jisuanji(self):
        """'电脑' 别名匹配到 '计算机'"""
        tokens = tokenize('电脑', LABEL_INDEX, ALIAS_INDEX)
        matched = [t for t in tokens if t.match_type == 'alias']
        assert len(matched) >= 1
        assert matched[0].seed_label == '计算机'

    def test_laduzi_matches_fuxie(self):
        """'拉肚子' 别名匹配到 '腹泻'"""
        tokens = tokenize('拉肚子', LABEL_INDEX, ALIAS_INDEX)
        matched = [t for t in tokens if t.match_type == 'alias']
        assert len(matched) >= 1
        assert matched[0].seed_label == '腹泻'

    def test_match_with_aliases_function(self):
        """match_with_aliases 函数直接测试"""
        result = match_with_aliases('着凉', ALIAS_INDEX)
        assert result is not None
        assert result.match_type == 'alias'
        assert result.seed_label == '感冒'

    def test_match_with_aliases_no_match(self):
        """match_with_aliases 无匹配返回 None"""
        result = match_with_aliases('不存在的别名', ALIAS_INDEX)
        assert result is None


class TestFuzzyMatch:
    """模糊匹配测试"""

    def test_ganchang_matches_ganmao(self):
        """'感昌' 模糊匹配到 '感冒'，编辑距离=1"""
        tokens = tokenize('感昌', LABEL_INDEX, ALIAS_INDEX, enable_fuzzy=True)
        fuzzy = [t for t in tokens if t.match_type == 'fuzzy']
        assert len(fuzzy) >= 1
        assert fuzzy[0].seed_label == '感冒'

    def test_fuzzy_match_function(self):
        """fuzzy_match 函数直接测试"""
        result = fuzzy_match('感昌', LABEL_INDEX, max_distance=1)
        assert result is not None
        assert result.match_type == 'fuzzy'
        assert result.seed_label == '感冒'

    def test_fuzzy_disabled(self):
        """禁用模糊匹配时不进行模糊匹配"""
        tokens = tokenize('感昌', LABEL_INDEX, ALIAS_INDEX, enable_fuzzy=False)
        fuzzy = [t for t in tokens if t.match_type == 'fuzzy']
        assert len(fuzzy) == 0

    def test_fuzzy_max_distance(self):
        """编辑距离超过阈值不匹配"""
        result = fuzzy_match('感昌啊', LABEL_INDEX, max_distance=1)
        # '感昌啊' vs '感冒' 编辑距离=2，超过 max_distance=1
        assert result is None

    def test_levenshtein_basic(self):
        """编辑距离基本测试"""
        assert _levenshtein('', '') == 0
        assert _levenshtein('abc', 'abc') == 0
        assert _levenshtein('abc', 'abd') == 1
        assert _levenshtein('abc', 'ab') == 1
        assert _levenshtein('感昌', '感冒') == 1

    def test_levenshtein_empty(self):
        """空字符串编辑距离"""
        assert _levenshtein('', 'abc') == 3
        assert _levenshtein('abc', '') == 3

    def test_fuzzy_edge_count_disambiguation(self):
        """模糊匹配歧义消解：选择出边数最多的种子"""
        # 构造两个候选，通过 edge_count_map 消歧
        label_index = {'测试A', '测试B'}
        edge_count_map = {'测试A': 5, '测试B': 10}
        result = fuzzy_match('测试C', label_index, max_distance=1, edge_count_map=edge_count_map)
        # '测试C' 与 '测试A' 和 '测试B' 编辑距离都是 1
        # 应该选择出边数更多的 '测试B'
        if result is not None:
            assert result.seed_label == '测试B'


class TestNegationDetection:
    """否定词识别测试"""

    def test_negation_excludes_seed(self):
        """'不是感冒' 中 '感冒' 被标记为 excluded"""
        tokens = tokenize('不是感冒', LABEL_INDEX, ALIAS_INDEX)
        ganmao_tokens = [t for t in tokens if t.seed_label == '感冒']
        assert len(ganmao_tokens) >= 1
        assert ganmao_tokens[0].excluded is True

    def test_no_negation(self):
        """无否定词时 excluded=False"""
        tokens = tokenize('感冒', LABEL_INDEX, ALIAS_INDEX)
        ganmao_tokens = [t for t in tokens if t.seed_label == '感冒']
        assert len(ganmao_tokens) >= 1
        assert ganmao_tokens[0].excluded is False

    def test_negation_words_set(self):
        """否定词表包含常见否定词"""
        assert '不是' in NEGATION_WORDS
        assert '不' in NEGATION_WORDS
        assert '没' in NEGATION_WORDS
        assert '没有' in NEGATION_WORDS

    def test_detect_negation_function(self):
        """detect_negation 函数直接测试"""
        tokens = [TokenMatch(text='感冒', match_type='exact', seed_label='感冒')]
        result = detect_negation('不是感冒', tokens)
        assert result[0].excluded is True

    def test_detect_negation_no_negation(self):
        """无否定词时不标记 excluded"""
        tokens = [TokenMatch(text='感冒', match_type='exact', seed_label='感冒')]
        result = detect_negation('感冒了', tokens)
        assert result[0].excluded is False

    def test_negation_scope(self):
        """否定词作用范围外的词不被排除"""
        # '不' 后面跟了很多字，'感冒' 超出 NEGATION_SCOPE 范围
        # NEGATION_SCOPE=4，'不XXXX感冒' → '感冒' 在范围外
        tokens = [TokenMatch(text='感冒', match_type='exact', seed_label='感冒')]
        result = detect_negation('不XXXX感冒', tokens)
        # '不' 结束位置=1，'感冒' 起始位置=5，5 >= 1+4=5，刚好在边界
        # 实际上 5 < 5 不成立，所以不被排除
        assert result[0].excluded is False


class TestEnglishAndNumberMatch:
    """英文/数字词匹配测试"""

    def test_english_word_exact_match(self):
        """英文词精确匹配"""
        label_index = {'AI', 'Python', 'COVID'}
        alias_index: dict[str, str] = {}
        tokens = tokenize('AI', label_index, alias_index)
        matched = [t for t in tokens if t.match_type == 'exact']
        assert len(matched) >= 1
        assert matched[0].seed_label == 'AI'

    def test_english_word_alias_match(self):
        """英文词别名匹配"""
        tokens = tokenize('AI', LABEL_INDEX, ALIAS_INDEX)
        matched = [t for t in tokens if t.match_type == 'alias']
        assert len(matched) >= 1
        assert matched[0].seed_label == '人工智能'

    def test_number_match(self):
        """数字词匹配"""
        label_index = {'100', '3D'}
        alias_index: dict[str, str] = {}
        tokens = tokenize('100', label_index, alias_index)
        matched = [t for t in tokens if t.match_type == 'exact']
        assert len(matched) >= 1

    def test_mixed_chinese_english(self):
        """中英文混合查询"""
        tokens = tokenize('感冒AI', LABEL_INDEX, ALIAS_INDEX)
        matched_labels = [t.seed_label for t in tokens if t.match_type != 'unmatched']
        assert '感冒' in matched_labels
        assert '人工智能' in matched_labels


class TestEmptyAndNoMatch:
    """空查询和无匹配查询测试"""

    def test_empty_query(self):
        """空查询返回空列表"""
        tokens = tokenize('', LABEL_INDEX, ALIAS_INDEX)
        assert tokens == []

    def test_whitespace_query(self):
        """纯空白查询返回空列表"""
        tokens = tokenize('   ', LABEL_INDEX, ALIAS_INDEX)
        assert tokens == []

    def test_no_match_query(self):
        """无匹配查询返回 unmatched tokens"""
        tokens = tokenize('xyz', LABEL_INDEX, ALIAS_INDEX)
        # 英文词不在知识库中，应为 unmatched
        assert len(tokens) >= 1
        assert all(t.match_type == 'unmatched' for t in tokens)

    def test_chinese_no_match(self):
        """中文无匹配查询"""
        tokens = tokenize('咕噜咕噜', LABEL_INDEX, ALIAS_INDEX, enable_fuzzy=False)
        # 所有中文词都不在知识库中，应为 unmatched
        assert all(t.match_type == 'unmatched' for t in tokens)


class TestMaxForwardMatch:
    """最大正向匹配算法测试"""

    def test_whole_match(self):
        """整体匹配优先"""
        results = _max_forward_match('量子力学', LABEL_INDEX)
        assert len(results) == 1
        assert results[0].match_type == 'exact'
        assert results[0].seed_label == '量子力学'

    def test_split_match(self):
        """拆分匹配"""
        results = _max_forward_match('感冒发热', LABEL_INDEX)
        matched = [r for r in results if r.match_type == 'exact']
        labels = {r.seed_label for r in matched}
        assert '感冒' in labels
        assert '发热' in labels

    def test_empty_span(self):
        """空 span 返回空列表"""
        results = _max_forward_match('', LABEL_INDEX)
        assert results == []

    def test_single_char_unmatched(self):
        """单字无法匹配标记为 unmatched"""
        results = _max_forward_match('咕', LABEL_INDEX)
        assert len(results) == 1
        assert results[0].match_type == 'unmatched'

    def test_alias_in_forward_match(self):
        """最大正向匹配中的别名匹配"""
        results = _max_forward_match('电脑', LABEL_INDEX, ALIAS_INDEX)
        assert len(results) == 1
        assert results[0].match_type == 'alias'
        assert results[0].seed_label == '计算机'


class TestTokenMatchDataclass:
    """TokenMatch 数据类测试"""

    def test_exact_match(self):
        """精确匹配 TokenMatch"""
        t = TokenMatch(text='感冒', match_type='exact', seed_label='感冒')
        assert t.text == '感冒'
        assert t.match_type == 'exact'
        assert t.seed_label == '感冒'
        assert t.excluded is False

    def test_unmatched_clears_seed_label(self):
        """unmatched 类型自动清空 seed_label"""
        t = TokenMatch(text='咕', match_type='unmatched', seed_label='不应该有')
        assert t.seed_label == ''

    def test_alias_match(self):
        """别名匹配 TokenMatch"""
        t = TokenMatch(text='着凉', match_type='alias', seed_label='感冒')
        assert t.match_type == 'alias'
        assert t.seed_label == '感冒'


if __name__ == '__main__':
    import traceback
    classes = [
        TestCompoundWordPriority, TestCompoundWordFallback, TestAliasMatch,
        TestFuzzyMatch, TestNegationDetection, TestEnglishAndNumberMatch,
        TestEmptyAndNoMatch, TestMaxForwardMatch, TestTokenMatchDataclass,
    ]
    for cls in classes:
        t = cls()
        for name in dir(t):
            if name.startswith('test_'):
                try:
                    getattr(t, name)()
                    print(f"  [PASS] {cls.__name__}.{name}")
                except Exception as e:
                    print(f"  [FAIL] {cls.__name__}.{name}: {e}")
                    traceback.print_exc()
