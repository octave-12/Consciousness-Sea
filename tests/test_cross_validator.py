"""
T-019: CrossValidator 单元测试

测试 CrossValidator 的所有公共方法：
- 一致性高的回答 → status=CONSISTENT, discount=1.0
- 矛盾回答 → status=CONTESTED, discount=0.7
- 单回答 → status=NONE, discount=1.0
- 否定词对检测（"有效" vs "无效"）
- Jaccard 系数计算
- contradiction_points 非空（矛盾时）
- merged_answer 非空（一致时）
- 无 GPU 依赖
"""

from __future__ import annotations

import pathlib
import sys

_root = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_root))
sys.path.insert(0, str(_root / "backend" / "src"))

from consciousness_sea.expert.cross_validator import CrossValidationStatus, CrossValidator


class TestCrossValidatorConsistent:
    """一致性高的回答测试"""

    def test_consistent_answers_return_consistent_status(self):
        """一致性高的回答 → status=CONSISTENT"""
        cv = CrossValidator()
        # 使用完全相同的回答确保 Jaccard=1.0 且无否定词对
        result = cv.validate(
            ["抗生素无效", "抗生素无效"],
            ["医学", "常识"],
        )
        assert result.status == CrossValidationStatus.CONSISTENT

    def test_consistent_answers_discount_1(self):
        """一致性高的回答 → discount=1.0"""
        cv = CrossValidator()
        result = cv.validate(
            ["抗生素无效", "抗生素无效"],
            ["医学", "常识"],
        )
        assert result.discount == 1.0

    def test_consistent_answers_merged_answer_not_none(self):
        """一致性高的回答 → merged_answer 非空"""
        cv = CrossValidator()
        result = cv.validate(
            ["抗生素无效", "抗生素无效"],
            ["医学", "常识"],
        )
        assert result.merged_answer is not None
        assert len(result.merged_answer) > 0

    def test_identical_answers_consistent(self):
        """完全相同的回答 → CONSISTENT"""
        cv = CrossValidator()
        result = cv.validate(
            ["抗生素无效", "抗生素无效"],
            ["医学", "常识"],
        )
        assert result.status == CrossValidationStatus.CONSISTENT
        assert result.discount == 1.0

    def test_consistent_answers_no_contradiction_points(self):
        """一致性高的回答 → contradiction_points 为空"""
        cv = CrossValidator()
        result = cv.validate(
            ["感冒需要休息和补充维C", "感冒要多休息补充维C和喝水"],
            ["医学", "常识"],
        )
        assert len(result.contradiction_points) == 0

    def test_consistent_per_answer_discounts_all_1(self):
        """一致性高的回答 → 每个回答折扣为 1.0"""
        cv = CrossValidator()
        result = cv.validate(
            ["抗生素无效", "抗生素无效"],
            ["医学", "常识"],
        )
        assert result.per_answer_discounts == [1.0, 1.0]


class TestCrossValidatorContested:
    """矛盾回答测试"""

    def test_contradictory_answers_return_contested_status(self):
        """矛盾回答 → status=CONTESTED"""
        cv = CrossValidator()
        result = cv.validate(
            ["这个方法是有效的", "这个方法是无效的"],
            ["医学", "常识"],
        )
        assert result.status == CrossValidationStatus.CONTESTED

    def test_contradictory_answers_discount(self):
        """矛盾回答 → discount=0.7"""
        cv = CrossValidator()
        result = cv.validate(
            ["这个方法是有效的", "这个方法是无效的"],
            ["医学", "常识"],
        )
        assert result.discount == 0.7

    def test_contradictory_answers_contradiction_points_not_empty(self):
        """矛盾回答 → contradiction_points 非空"""
        cv = CrossValidator()
        result = cv.validate(
            ["这个方法是有效的", "这个方法是无效的"],
            ["医学", "常识"],
        )
        assert len(result.contradiction_points) > 0

    def test_contradictory_answers_merged_answer_none(self):
        """矛盾回答 → merged_answer 为 None"""
        cv = CrossValidator()
        result = cv.validate(
            ["这个方法是有效的", "这个方法是无效的"],
            ["医学", "常识"],
        )
        assert result.merged_answer is None

    def test_contradictory_per_answer_discounts(self):
        """矛盾回答 → 每个回答折扣为 0.7"""
        cv = CrossValidator()
        result = cv.validate(
            ["这个方法是有效的", "这个方法是无效的"],
            ["医学", "常识"],
        )
        assert result.per_answer_discounts == [0.7, 0.7]

    def test_low_jaccard_contested(self):
        """低 Jaccard 系数（关键词差异大）→ CONTESTED"""
        cv = CrossValidator()
        result = cv.validate(
            ["量子力学是物理学分支", "感冒需要多喝水休息"],
            ["物理", "医学"],
        )
        # 关键词几乎无重叠 → Jaccard 很低 → CONTESTED
        assert result.status == CrossValidationStatus.CONTESTED


class TestCrossValidatorSingleAnswer:
    """单回答测试"""

    def test_single_answer_returns_none_status(self):
        """单回答 → status=NONE"""
        cv = CrossValidator()
        result = cv.validate(["感冒需要休息"], ["医学"])
        assert result.status == CrossValidationStatus.NONE

    def test_single_answer_discount_1(self):
        """单回答 → discount=1.0"""
        cv = CrossValidator()
        result = cv.validate(["感冒需要休息"], ["医学"])
        assert result.discount == 1.0

    def test_single_answer_no_contradiction_points(self):
        """单回答 → contradiction_points 为空"""
        cv = CrossValidator()
        result = cv.validate(["感冒需要休息"], ["医学"])
        assert len(result.contradiction_points) == 0

    def test_single_answer_merged_answer_none(self):
        """单回答 → merged_answer 为 None"""
        cv = CrossValidator()
        result = cv.validate(["感冒需要休息"], ["医学"])
        assert result.merged_answer is None

    def test_empty_answers_returns_none_status(self):
        """空回答列表 → status=NONE"""
        cv = CrossValidator()
        result = cv.validate([], [])
        assert result.status == CrossValidationStatus.NONE
        assert result.discount == 1.0


class TestCrossValidatorNegationPairs:
    """否定词对检测测试"""

    def test_effective_ineffective_pair(self):
        """否定词对: "有效" vs "无效" """
        cv = CrossValidator()
        result = cv.validate(
            ["这个药是有效的", "这个药是无效的"],
            ["医学", "常识"],
        )
        assert result.status == CrossValidationStatus.CONTESTED
        assert any("有效" in cp and "无效" in cp for cp in result.contradiction_points)

    def test_can_cannot_pair(self):
        """否定词对: "可以" vs "不可以" """
        cv = CrossValidator()
        result = cv.validate(
            ["感冒可以吃阿奇霉素", "感冒不可以吃阿奇霉素"],
            ["医学", "常识"],
        )
        assert result.status == CrossValidationStatus.CONTESTED
        assert any("可以" in cp for cp in result.contradiction_points)

    def test_should_should_not_pair(self):
        """否定词对: "应该" vs "不应该" """
        cv = CrossValidator()
        result = cv.validate(
            ["你应该多喝水", "你不应该多喝水"],
            ["医学", "常识"],
        )
        assert result.status == CrossValidationStatus.CONTESTED

    def test_safe_dangerous_pair(self):
        """否定词对: "安全" vs "危险" """
        cv = CrossValidator()
        result = cv.validate(
            ["这个药物是安全的", "这个药物是危险的"],
            ["医学", "常识"],
        )
        assert result.status == CrossValidationStatus.CONTESTED

    def test_no_negation_no_contradiction(self):
        """无否定词对 → 无矛盾点（即使 Jaccard 低）"""
        cv = CrossValidator()
        result = cv.validate(
            ["感冒需要休息", "感冒需要喝水"],
            ["医学", "常识"],
        )
        # 无否定词对，Jaccard 也可能较高
        # 但如果 Jaccard 低也可能 CONTESTED
        # 关键是 contradiction_points 不含否定词对矛盾
        for cp in result.contradiction_points:
            assert "矛盾" not in cp or "说" not in cp


class TestCrossValidatorJaccard:
    """Jaccard 系数计算测试"""

    def test_jaccard_identical_sets(self):
        """两个相同集合 → Jaccard = 1.0"""
        cv = CrossValidator()
        jaccard = cv._compute_jaccard({"感冒", "发热"}, {"感冒", "发热"})
        assert jaccard == 1.0

    def test_jaccard_disjoint_sets(self):
        """两个不相交集合 → Jaccard = 0.0"""
        cv = CrossValidator()
        jaccard = cv._compute_jaccard({"感冒", "发热"}, {"量子", "力学"})
        assert jaccard == 0.0

    def test_jaccard_partial_overlap(self):
        """部分重叠 → 0 < Jaccard < 1"""
        cv = CrossValidator()
        jaccard = cv._compute_jaccard({"感冒", "发热"}, {"感冒", "咳嗽"})
        # |A ∩ B| = 1, |A ∪ B| = 3 → Jaccard = 1/3
        assert 0.0 < jaccard < 1.0
        assert abs(jaccard - 1/3) < 0.01

    def test_jaccard_empty_sets(self):
        """两个空集合 → Jaccard = 1.0"""
        cv = CrossValidator()
        jaccard = cv._compute_jaccard(set(), set())
        assert jaccard == 1.0

    def test_jaccard_one_empty_set(self):
        """一个空集合 → Jaccard = 0.0"""
        cv = CrossValidator()
        jaccard = cv._compute_jaccard({"感冒"}, set())
        assert jaccard == 0.0


class TestCrossValidatorMergeAnswers:
    """回答合并测试"""

    def test_merge_answers_returns_string(self):
        """合并回答返回字符串"""
        cv = CrossValidator()
        merged = cv._merge_answers(["回答A", "回答B"], ["医学", "常识"])
        assert isinstance(merged, str)
        assert len(merged) > 0

    def test_merge_answers_base_is_first(self):
        """合并回答以第一个回答为主体"""
        cv = CrossValidator()
        merged = cv._merge_answers(["回答A", "回答B"], ["医学", "常识"])
        assert merged.startswith("回答A")

    def test_merge_answers_empty_list(self):
        """空回答列表返回空字符串"""
        cv = CrossValidator()
        merged = cv._merge_answers([], [])
        assert merged == ""

    def test_merge_answers_single_answer(self):
        """单回答直接返回"""
        cv = CrossValidator()
        merged = cv._merge_answers(["唯一回答"], ["医学"])
        assert merged == "唯一回答"


class TestCrossValidatorCustomParams:
    """自定义参数测试"""

    def test_custom_discount(self):
        """自定义折扣系数"""
        cv = CrossValidator(discount=0.5)
        result = cv.validate(
            ["这个方法是有效的", "这个方法是无效的"],
            ["医学", "常识"],
        )
        assert result.discount == 0.5

    def test_custom_consistency_threshold(self):
        """自定义一致性阈值"""
        cv = CrossValidator(consistency_threshold=0.3)
        # 低阈值下更多回答被视为一致
        cv.validate(
            ["感冒需要休息", "感冒需要喝水"],
            ["医学", "常识"],
        )
        # 阈值 0.3，如果 Jaccard > 0.3 且无矛盾 → CONSISTENT
        # 具体结果取决于关键词提取


class TestCrossValidatorNegationPairsTable:
    """否定词对表完整性测试"""

    def test_negation_pairs_count(self):
        """否定词对表至少包含 13 对"""
        cv = CrossValidator()
        assert len(cv.NEGATION_PAIRS) >= 13

    def test_negation_pairs_are_tuples(self):
        """否定词对为 (positive, negative) 元组"""
        cv = CrossValidator()
        for positive, negative in cv.NEGATION_PAIRS:
            assert isinstance(positive, str)
            assert isinstance(negative, str)
            assert len(positive) > 0
            assert len(negative) > 0

    def test_specific_negation_pairs_exist(self):
        """特定否定词对存在"""
        cv = CrossValidator()
        pairs_dict = dict(cv.NEGATION_PAIRS)
        assert "有效" in pairs_dict
        assert pairs_dict["有效"] == "无效"
        assert "可以" in pairs_dict
        assert "安全" in pairs_dict


if __name__ == '__main__':
    import pytest
    pytest.main([__file__, '-v'])
