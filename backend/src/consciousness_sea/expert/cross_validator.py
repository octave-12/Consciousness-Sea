"""
CrossValidator — 多专家答案对比 + 矛盾检测

职责:
  - 提取多个专家回答的关键词
  - 检测关键词间的矛盾（否定词对检测）
  - 计算一致性得分（Jaccard 系数）
  - 决定合并/打折/分别呈现策略

CrossValidator 为无状态组件，每次 validate() 调用独立执行。
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from enum import Enum

from consciousness_sea.infrastructure.config import (
    CROSS_VALIDATION_DISCOUNT,
    CROSS_VALIDATION_CONSISTENCY_THRESHOLD,
    MIN_KEYWORD_LENGTH,
)
from consciousness_sea.domain.verifier import BUILTIN_STOP_WORDS, DOMAIN_NAMES

log = logging.getLogger(__name__)


class CrossValidationStatus(str, Enum):
    """交叉验证状态枚举"""

    NONE = "none"  # 未执行交叉验证
    CONSISTENT = "consistent"  # 一致性高
    CONTESTED = "contested"  # 存在矛盾


@dataclass
class CrossValidationResult:
    """交叉验证结果"""

    status: CrossValidationStatus = CrossValidationStatus.NONE
    discount: float = 1.0  # 置信度折扣系数 (1.0 = 无折扣)
    contradiction_points: list[str] = field(default_factory=list)  # 矛盾点描述
    merged_answer: str | None = None  # 合并后的回答（一致性高时）
    per_answer_discounts: list[float] = field(default_factory=list)  # 每个回答的折扣


class CrossValidator:
    """交叉验证器 — 多专家答案对比 + 矛盾检测

    职责:
      - 提取多个专家回答的关键词
      - 检测关键词间的矛盾（否定词对检测）
      - 计算一致性得分（Jaccard 系数）
      - 决定合并/打折/分别呈现策略

    Args:
        discount: 矛盾折扣系数 (默认 0.7)
        consistency_threshold: 一致性阈值 (默认 0.8)
    """

    # 否定词对表（用于矛盾检测）
    NEGATION_PAIRS: list[tuple[str, str]] = [
        ("有效", "无效"),
        ("可以", "不可以"),
        ("应该", "不应该"),
        ("能", "不能"),
        ("会", "不会"),
        ("是", "不是"),
        ("有", "没有"),
        ("需要", "不需要"),
        ("正确", "错误"),
        ("安全", "危险"),
        ("推荐", "不推荐"),
        ("适合", "不适合"),
        ("必须", "禁止"),
    ]

    def __init__(
        self,
        discount: float = CROSS_VALIDATION_DISCOUNT,
        consistency_threshold: float = CROSS_VALIDATION_CONSISTENCY_THRESHOLD,
    ) -> None:
        self._discount = discount
        self._consistency_threshold = consistency_threshold

    def validate(
        self,
        answers: list[str],
        domains: list[str],
    ) -> CrossValidationResult:
        """执行交叉验证

        流程:
          1. 仅 1 个回答 → 跳过，返回 NONE
          2. 提取每个回答的关键词集合
          3. 计算 Jaccard 一致性
          4. 检测否定词对矛盾
          5. 决策:
             - 一致性 > 0.8 且无矛盾 → CONSISTENT, 合并回答
             - 一致性 ≤ 0.8 或有矛盾 → CONTESTED, 打折

        Args:
            answers: 专家回答文本列表
            domains: 对应领域列表

        Returns:
            CrossValidationResult
        """
        # 1. 单回答 → 跳过
        if len(answers) <= 1:
            return CrossValidationResult(
                status=CrossValidationStatus.NONE,
                discount=1.0,
                per_answer_discounts=[1.0] * len(answers),
            )

        # 2. 提取每个回答的关键词集合
        keyword_sets = [self._extract_answer_keywords(a) for a in answers]

        # 3. 两两比较
        min_jaccard = 1.0
        all_contradictions: list[str] = []

        for i in range(len(answers)):
            for j in range(i + 1, len(answers)):
                jaccard = self._compute_jaccard(keyword_sets[i], keyword_sets[j])
                min_jaccard = min(min_jaccard, jaccard)
                contradictions = self._detect_negation_contradiction(
                    answers[i], answers[j]
                )
                all_contradictions.extend(contradictions)

        # 4. 决策
        if min_jaccard > self._consistency_threshold and not all_contradictions:
            # 一致性高 → 合并
            merged = self._merge_answers(answers, domains)
            return CrossValidationResult(
                status=CrossValidationStatus.CONSISTENT,
                discount=1.0,
                merged_answer=merged,
                per_answer_discounts=[1.0] * len(answers),
            )
        else:
            # 存在矛盾 → 打折
            return CrossValidationResult(
                status=CrossValidationStatus.CONTESTED,
                discount=self._discount,
                contradiction_points=all_contradictions,
                per_answer_discounts=[self._discount] * len(answers),
            )

    def _extract_answer_keywords(self, text: str) -> set[str]:
        """从回答文本中提取关键词集合

        复用 verifier 的停用词过滤逻辑。
        使用 2~3 字滑动窗口从中文文本中提取 n-gram，
        以获得适合 Jaccard 比较的关键词粒度。
        """
        # 提取中文连续段
        chinese_spans = re.findall(r"[\u4e00-\u9fff]+", text)

        keywords: set[str] = set()
        for span in chinese_spans:
            # 使用滑动窗口生成 2~3 字 n-gram
            for n in (2, 3):
                for i in range(len(span) - n + 1):
                    gram = span[i : i + n]
                    # 停用词过滤
                    if gram in BUILTIN_STOP_WORDS:
                        continue
                    # 最小长度过滤
                    if len(gram) < MIN_KEYWORD_LENGTH:
                        continue
                    # 领域名排除
                    if gram in DOMAIN_NAMES:
                        continue
                    keywords.add(gram)

        return keywords

    def _compute_jaccard(self, set_a: set[str], set_b: set[str]) -> float:
        """计算 Jaccard 相似系数

        J(A, B) = |A ∩ B| / |A ∪ B|
        """
        if not set_a and not set_b:
            return 1.0  # 两个空集视为完全一致
        union = set_a | set_b
        if not union:
            return 1.0
        intersection = set_a & set_b
        return len(intersection) / len(union)

    def _detect_negation_contradiction(
        self,
        answer_a: str,
        answer_b: str,
    ) -> list[str]:
        """检测否定词对矛盾

        对每对否定词 (p, n):
          - 若 answer_a 包含 p 且 answer_b 包含 n → 矛盾
          - 若 answer_a 包含 n 且 answer_b 包含 p → 矛盾

        Returns:
            矛盾点描述列表
        """
        contradictions: list[str] = []

        for positive, negative in self.NEGATION_PAIRS:
            a_has_pos = positive in answer_a
            a_has_neg = negative in answer_a
            b_has_pos = positive in answer_b
            b_has_neg = negative in answer_b

            # 当正向词是否定词的子串时（如 "可以" 在 "不可以" 中），
            # 需要排除否定词对正向词的遮蔽：若否定词存在，
            # 则视为正向词不存在（因为正向词是否定词的一部分）
            if a_has_neg and positive in negative:
                a_has_pos = False
            if b_has_neg and positive in negative:
                b_has_pos = False

            # A 说肯定且不含否定，B 说否定且不含肯定
            if a_has_pos and not a_has_neg and b_has_neg and not b_has_pos:
                contradictions.append(
                    f"矛盾: 一方说「{positive}」，另一方说「{negative}」"
                )
            # A 说否定且不含肯定，B 说肯定且不含否定
            elif a_has_neg and not a_has_pos and b_has_pos and not b_has_neg:
                contradictions.append(
                    f"矛盾: 一方说「{negative}」，另一方说「{positive}」"
                )

        return contradictions

    def _merge_answers(
        self,
        answers: list[str],
        domains: list[str],
    ) -> str:
        """合并一致性高的多个回答

        策略: 取第一个回答为主体，补充其他回答中的独特信息。
        """
        if not answers:
            return ""

        if len(answers) == 1:
            return answers[0]

        # 取第一个回答为主体
        base = answers[0]
        base_keywords = self._extract_answer_keywords(base)

        # 补充其他回答中的独特信息
        supplements: list[str] = []
        for i in range(1, len(answers)):
            answer_keywords = self._extract_answer_keywords(answers[i])
            unique_keywords = answer_keywords - base_keywords
            if unique_keywords:
                domain_label = domains[i] if i < len(domains) else f"专家{i + 1}"
                supplements.append(
                    f"[{domain_label}补充] {answers[i]}"
                )

        if supplements:
            return base + "\n\n" + "\n\n".join(supplements)
        return base