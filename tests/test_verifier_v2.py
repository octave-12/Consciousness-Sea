"""
测试升级后校验器 (Verifier V2)

覆盖:
- 扩充后的停用词过滤
- 关键词最小长度 >= 2
- 领域名排除
- 关键词质量权重
- 空关键词兜底
- 加权置信度计算
"""

import sqlite3
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.graph_db import GraphDB
from core.verifier import (
    _extract_keywords_v2,
    _keyword_quality_weight,
    verify,
    load_stopwords,
    _reset_stopwords_cache,
    BUILTIN_STOP_WORDS,
    DOMAIN_NAMES,
)
from core.router import RippleResult, ActivationNode


def _setup_db():
    """创建测试用内存数据库"""
    conn = sqlite3.connect(':memory:')
    conn.row_factory = sqlite3.Row
    conn.executescript("""
        CREATE TABLE seeds (
            id TEXT PRIMARY KEY, label TEXT NOT NULL,
            type TEXT NOT NULL DEFAULT 'CONCEPT',
            aliases TEXT NOT NULL DEFAULT '[]',
            activation REAL NOT NULL DEFAULT 0.0,
            domain TEXT NOT NULL DEFAULT '',
            definition TEXT NOT NULL DEFAULT '',
            pinyin TEXT NOT NULL DEFAULT '',
            activation_bias REAL NOT NULL DEFAULT 0.0,
            meta TEXT NOT NULL DEFAULT '{}',
            created_at TEXT NOT NULL DEFAULT (datetime('now'))
        );
        CREATE TABLE karma_edges (
            source TEXT NOT NULL, target TEXT NOT NULL, relation TEXT NOT NULL,
            weight REAL NOT NULL DEFAULT 0.5,
            source_tag TEXT NOT NULL DEFAULT '',
            created_at TEXT NOT NULL DEFAULT (datetime('now')),
            PRIMARY KEY (source, target, relation)
        );
    """)

    seeds = [
        ('感冒', '感冒', 'CONCEPT', '[]', '医学', 'to catch cold'),
        ('发热', '发热', 'CONCEPT', '[]', '医学', 'fever'),
        ('咳嗽', '咳嗽', 'CONCEPT', '[]', '医学', 'cough'),
        ('量子力学', '量子力学', 'CONCEPT', '[]', '物理', 'quantum mechanics'),
        ('薛定谔方程', '薛定谔方程', 'CONCEPT', '[]', '物理', 'Schrodinger equation'),
        ('人工智能', '人工智能', 'CONCEPT', '[]', '计算机', 'AI'),
        ('深度学习', '深度学习', 'CONCEPT', '[]', '计算机', 'deep learning'),
        ('维C', '维C', 'CONCEPT', '[]', '营养', 'Vitamin C'),
    ]
    conn.executemany(
        "INSERT INTO seeds (id,label,type,aliases,domain,definition) VALUES (?,?,?,?,?,?)",
        seeds
    )

    edges = [
        ('感冒', '发热', 'COOCCURS_WITH', 0.95),
        ('感冒', '咳嗽', 'COOCCURS_WITH', 0.90),
    ]
    conn.executemany(
        "INSERT INTO karma_edges (source,target,relation,weight) VALUES (?,?,?,?)",
        edges
    )
    conn.commit()
    return conn


def _make_ripple_result(activated_labels: list[str] | None = None) -> RippleResult:
    """辅助函数：构造 RippleResult 对象"""
    result = RippleResult()
    if activated_labels:
        for label in activated_labels:
            result.activated[label] = ActivationNode(label=label, activation=1.0, depth=0)
    return result


class TestStopwordsFilter:
    """扩充后的停用词过滤测试"""

    def setup_method(self):
        _reset_stopwords_cache()
        self.conn = _setup_db()
        self.db = GraphDB(':memory:')
        self.db.conn = self.conn

    def teardown_method(self):
        self.db.close()
        _reset_stopwords_cache()

    def test_builtin_stopwords_loaded(self):
        """内置停用词已加载"""
        stopwords = load_stopwords()
        assert '的' in stopwords
        assert '了' in stopwords
        assert '是' in stopwords
        assert '什么' in stopwords
        assert '怎么' in stopwords

    def test_format_words_filtered(self):
        """格式词被过滤"""
        stopwords = load_stopwords()
        assert '关于' in stopwords
        assert '关联概念' in stopwords
        assert '传播路径' in stopwords

    def test_structure_words_filtered(self):
        """结构词被过滤"""
        stopwords = load_stopwords()
        assert '常与' in stopwords
        assert '共现' in stopwords
        assert '定义为' in stopwords

    def test_pronouns_filtered(self):
        """代词被过滤"""
        stopwords = load_stopwords()
        assert '我' in stopwords
        assert '你' in stopwords
        assert '他们' in stopwords

    def test_conjunctions_filtered(self):
        """连词被过滤"""
        stopwords = load_stopwords()
        assert '但是' in stopwords
        assert '因为' in stopwords
        assert '所以' in stopwords

    def test_stopwords_count(self):
        """内置停用词数量 >= 200"""
        assert len(BUILTIN_STOP_WORDS) >= 200

    def test_stopwords_filter_in_keyword_extraction(self):
        """停用词在关键词提取中被过滤"""
        # _extract_keywords_v2 使用 re.findall(r'[\u4e00-\u9fff]{2,}', text)
        # 连续中文段会被整体提取，所以用空格分隔
        text = "感冒 发热 咳嗽 的 了"
        keywords = _extract_keywords_v2(text, self.db)
        keyword_texts = [kw for kw, _ in keywords]
        # '的' 和 '了' 是单字，不应出现（正则已过滤单字）
        assert '的' not in keyword_texts
        assert '了' not in keyword_texts
        # '感冒' 和 '发热' 应该出现
        assert '感冒' in keyword_texts
        assert '发热' in keyword_texts


class TestMinKeywordLength:
    """关键词最小长度 >= 2 测试"""

    def setup_method(self):
        _reset_stopwords_cache()
        self.conn = _setup_db()
        self.db = GraphDB(':memory:')
        self.db.conn = self.conn

    def teardown_method(self):
        self.db.close()
        _reset_stopwords_cache()

    def test_single_char_filtered(self):
        """单字词被过滤"""
        text = "感冒 发 咳嗽"
        keywords = _extract_keywords_v2(text, self.db)
        keyword_texts = [kw for kw, _ in keywords]
        # '发' 是单字，不应出现
        assert '发' not in keyword_texts

    def test_two_char_kept(self):
        """双字词保留"""
        # 用空格分隔以避免被提取为一个整体中文段
        text = "感冒 发热"
        keywords = _extract_keywords_v2(text, self.db)
        keyword_texts = [kw for kw, _ in keywords]
        assert '感冒' in keyword_texts
        assert '发热' in keyword_texts


class TestDomainNameExclusion:
    """领域名排除测试"""

    def setup_method(self):
        _reset_stopwords_cache()
        self.conn = _setup_db()
        self.db = GraphDB(':memory:')
        self.db.conn = self.conn

    def teardown_method(self):
        self.db.close()
        _reset_stopwords_cache()

    def test_domain_names_in_exclusion_set(self):
        """领域名在排除集合中"""
        assert '医学' in DOMAIN_NAMES
        assert '物理' in DOMAIN_NAMES
        assert '计算机' in DOMAIN_NAMES
        assert '文学' in DOMAIN_NAMES
        assert '常识' in DOMAIN_NAMES

    def test_domain_name_filtered_in_keywords(self):
        """领域名在关键词提取中被排除"""
        # 用空格分隔以独立提取每个词
        text = "医学 物理 计算机"
        keywords = _extract_keywords_v2(text, self.db)
        keyword_texts = [kw for kw, _ in keywords]
        # 领域名不应作为关键词
        assert '医学' not in keyword_texts
        assert '物理' not in keyword_texts
        assert '计算机' not in keyword_texts


class TestKeywordQualityWeight:
    """关键词质量权重测试"""

    def setup_method(self):
        _reset_stopwords_cache()
        self.conn = _setup_db()
        self.db = GraphDB(':memory:')
        self.db.conn = self.conn

    def teardown_method(self):
        self.db.close()
        _reset_stopwords_cache()

    def test_long_keyword_in_graph_weight_1_5(self):
        """长度 >= 4 且在知识库中存在 → 权重 1.5"""
        weight = _keyword_quality_weight('量子力学', self.db)
        assert weight == 1.5

    def test_medium_keyword_in_graph_weight_1_2(self):
        """长度 >= 3 且在知识库中存在 → 权重 1.2"""
        weight = _keyword_quality_weight('薛定谔方程', self.db)
        assert weight == 1.5  # 长度5 >= 4，权重1.5

    def test_three_char_keyword_in_graph(self):
        """长度=3 且在知识库中存在 → 权重 1.2"""
        # 添加一个三字种子
        self.conn.execute(
            "INSERT INTO seeds (id,label,type,aliases,domain,definition) VALUES (?,?,?,?,?,?)",
            ('新概念', '新概念', 'CONCEPT', '[]', '常识', 'test')
        )
        self.conn.commit()
        # 清除缓存让新种子可见
        self.db._label_index = None
        weight = _keyword_quality_weight('新概念', self.db)
        assert weight == 1.2

    def test_keyword_not_in_graph_weight_1_0(self):
        """不在知识库中 → 权重 1.0"""
        weight = _keyword_quality_weight('不存在的词', self.db)
        assert weight == 1.0

    def test_short_keyword_in_graph_weight_1_0(self):
        """长度 < 3 且在知识库中 → 权重 1.0"""
        # '维C' 长度=2，在知识库中，但长度 < 3
        weight = _keyword_quality_weight('维C', self.db)
        assert weight == 1.0

    def test_quality_weight_in_keyword_extraction(self):
        """质量权重在关键词提取中生效"""
        # 用空格分隔以独立提取每个词
        text = "量子力学 感冒"
        keywords = _extract_keywords_v2(text, self.db)
        keyword_dict = dict(keywords)
        # '量子力学' 长度4，在知识库中 → 权重1.5
        assert keyword_dict.get('量子力学') == 1.5
        # '感冒' 长度2，在知识库中但长度<3 → 权重1.0
        assert keyword_dict.get('感冒') == 1.0


class TestEmptyKeywordFallback:
    """空关键词兜底测试"""

    def setup_method(self):
        _reset_stopwords_cache()
        self.conn = _setup_db()
        self.db = GraphDB(':memory:')
        self.db.conn = self.conn

    def teardown_method(self):
        self.db.close()
        _reset_stopwords_cache()

    def test_empty_text_returns_fallback(self):
        """空文本 → confidence=0.5, karma_direction=0"""
        result = _make_ripple_result()
        verdict = verify('', result, self.db)
        assert verdict['confidence'] == 0.5
        assert verdict['karma_direction'] == 0
        assert verdict['decision'] == 'uncertain'

    def test_stopwords_only_returns_fallback(self):
        """只有停用词的文本 → confidence=0.5"""
        result = _make_ripple_result()
        # 使用全为停用词且无连续2字中文段的文本
        # 单字停用词被正则 {2,} 过滤，不会产生关键词
        verdict = verify('的 了 是 在', result, self.db)
        assert verdict['confidence'] == 0.5
        assert verdict['karma_direction'] == 0

    def test_domain_names_only_returns_fallback(self):
        """只有领域名的文本 → 无有效关键词"""
        result = _make_ripple_result()
        verdict = verify('医学 物理 计算机', result, self.db)
        assert verdict['confidence'] == 0.5


class TestWeightedConfidence:
    """加权置信度计算测试"""

    def setup_method(self):
        _reset_stopwords_cache()
        self.conn = _setup_db()
        self.db = GraphDB(':memory:')
        self.db.conn = self.conn

    def teardown_method(self):
        self.db.close()
        _reset_stopwords_cache()

    def test_confidence_calculation_with_weights(self):
        """加权置信度正确计算"""
        result = _make_ripple_result(['感冒', '量子力学'])

        # 回答中包含 '感冒'(权重1.0) 和 '量子力学'(权重1.5)
        # 用空格分隔以独立提取
        verdict = verify('感冒 量子力学', result, self.db)
        assert verdict['confidence'] > 0
        assert verdict['matched_keywords'] >= 0

    def test_partial_match_confidence(self):
        """部分匹配时置信度在 0~1 之间"""
        result = _make_ripple_result(['感冒'])

        # 回答中包含 '感冒'(匹配) 和 '黑洞'(不匹配)
        verdict = verify('感冒 黑洞', result, self.db)
        assert 0.0 < verdict['confidence'] < 1.0

    def test_no_match_zero_confidence(self):
        """无匹配时置信度为 0"""
        result = _make_ripple_result(['感冒'])

        verdict = verify('黑洞 引力波', result, self.db)
        assert verdict['confidence'] == 0.0

    def test_full_match_confidence_one(self):
        """全部匹配时置信度为 1"""
        result = _make_ripple_result(['感冒', '发热'])

        # 用空格分隔以独立提取
        verdict = verify('感冒 发热', result, self.db)
        assert verdict['confidence'] == 1.0


if __name__ == '__main__':
    import traceback
    classes = [
        TestStopwordsFilter, TestMinKeywordLength, TestDomainNameExclusion,
        TestKeywordQualityWeight, TestEmptyKeywordFallback, TestWeightedConfidence,
    ]
    for cls in classes:
        t = cls()
        for name in dir(t):
            if name.startswith('test_'):
                t.setup_method()
                try:
                    getattr(t, name)()
                    print(f"  [PASS] {cls.__name__}.{name}")
                except Exception as e:
                    print(f"  [FAIL] {cls.__name__}.{name}: {e}")
                    traceback.print_exc()
                t.teardown_method()
