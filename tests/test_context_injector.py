"""
T-018: ContextInjector 单元测试

测试 ContextInjector 的所有公共方法：
- 正常 prompt 构造（含种子 + 路径 + 查询）
- Token 预算截断（超长种子列表按激活值截断）
- 查询文本截断（>500 字符时截断）
- 空激活种子 → 合理 prompt
- PromptResult.truncated 标志
- full_prompt 包含 system + context + query
- 无 GPU 依赖
"""

from __future__ import annotations

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.context_injector import ContextInjector, PromptResult
from core.router import RippleResult, ActivationNode
from core.config import CONTEXT_MAX_TOKENS, CONTEXT_MAX_QUERY_LENGTH


def _make_ripple_result(
    seeds: list[tuple[str, float, str, str]] | None = None,
    paths: list[dict] | None = None,
    domain_scores: dict[str, float] | None = None,
    query: str = "感冒了吃什么",
) -> RippleResult:
    """构造测试用 RippleResult

    Args:
        seeds: [(label, activation, domain, definition), ...]
        paths: [{source, target, relation, weight, ripple_activation}, ...]
        domain_scores: {domain: score}
        query: 查询文本
    """
    result = RippleResult()
    result.query = query

    if seeds:
        for label, activation, domain, definition in seeds:
            result.activated[label] = ActivationNode(
                label=label,
                activation=activation,
                domain=domain,
                definition=definition,
                depth=0,
            )

    if paths:
        result.paths = paths

    if domain_scores:
        result.domain_scores = domain_scores

    return result


class TestContextInjectorBuildPrompt:
    """build_prompt() 正常构造测试"""

    def test_normal_prompt_with_seeds_and_paths(self):
        """正常 prompt 构造（含种子 + 路径 + 查询）"""
        result = _make_ripple_result(
            seeds=[
                ("感冒", 0.9, "医学", "急性上呼吸道感染"),
                ("发热", 0.8, "医学", "体温升高的生理反应"),
            ],
            paths=[
                {
                    "source": "感冒",
                    "target": "发热",
                    "relation": "COOCCURS_WITH",
                    "weight": 0.95,
                    "ripple_activation": 0.85,
                },
            ],
            domain_scores={"医学": 0.85},
        )

        injector = ContextInjector()
        prompt_result = injector.build_prompt(result, "感冒了吃什么")

        assert isinstance(prompt_result, PromptResult)
        # full_prompt 包含三部分
        assert injector.SYSTEM_PROMPT_TEMPLATE in prompt_result.full_prompt
        assert "感冒" in prompt_result.context_block
        assert "发热" in prompt_result.context_block
        assert prompt_result.user_query == "感冒了吃什么"

    def test_prompt_contains_system_context_query(self):
        """full_prompt 包含 system + context + query 三部分"""
        result = _make_ripple_result(
            seeds=[("感冒", 0.9, "医学", "急性上呼吸道感染")],
            domain_scores={"医学": 0.5},
        )

        injector = ContextInjector()
        prompt_result = injector.build_prompt(result, "感冒了吃什么")

        # 验证三部分
        assert prompt_result.system_prompt == injector.SYSTEM_PROMPT_TEMPLATE
        assert len(prompt_result.context_block) > 0
        assert prompt_result.user_query == "感冒了吃什么"
        assert prompt_result.full_prompt.startswith(injector.SYSTEM_PROMPT_TEMPLATE)

    def test_seeds_block_format(self):
        """种子文本块格式: - 感冒 [医学]: 急性上呼吸道感染"""
        result = _make_ripple_result(
            seeds=[("感冒", 0.9, "医学", "急性上呼吸道感染")],
            domain_scores={"医学": 0.5},
        )

        injector = ContextInjector()
        prompt_result = injector.build_prompt(result, "测试")

        assert "- 感冒 [医学]: 急性上呼吸道感染" in prompt_result.context_block

    def test_paths_block_format(self):
        """路径文本块格式: - 感冒 --[常与...共现]--> 发热 (关联度: 0.95)"""
        result = _make_ripple_result(
            seeds=[
                ("感冒", 0.9, "医学", "cold"),
                ("发热", 0.8, "医学", "fever"),
            ],
            paths=[
                {
                    "source": "感冒",
                    "target": "发热",
                    "relation": "COOCCURS_WITH",
                    "weight": 0.95,
                    "ripple_activation": 0.85,
                },
            ],
            domain_scores={"医学": 0.5},
        )

        injector = ContextInjector()
        prompt_result = injector.build_prompt(result, "测试")

        assert "感冒" in prompt_result.context_block
        assert "发热" in prompt_result.context_block
        assert "0.95" in prompt_result.context_block

    def test_seeds_sorted_by_activation_desc(self):
        """种子按激活值降序排列"""
        result = _make_ripple_result(
            seeds=[
                ("低激活", 0.3, "医学", "低"),
                ("高激活", 0.9, "医学", "高"),
                ("中激活", 0.6, "医学", "中"),
            ],
            domain_scores={"医学": 0.5},
        )

        injector = ContextInjector()
        prompt_result = injector.build_prompt(result, "测试")

        # 高激活种子应出现在低激活之前
        high_pos = prompt_result.context_block.find("高激活")
        low_pos = prompt_result.context_block.find("低激活")
        assert high_pos < low_pos


class TestContextInjectorTruncation:
    """Token 预算截断测试"""

    def test_truncation_flag_set_when_content_exceeds_budget(self):
        """超长种子列表触发截断，truncated=True"""
        # 构造大量种子，使 token 数超过预算
        many_seeds = [
            (f"种子{i:03d}", 1.0 - i * 0.001, "医学", f"这是第{i}个种子的定义说明文字" * 10)
            for i in range(200)
        ]

        result = _make_ripple_result(
            seeds=many_seeds,
            domain_scores={"医学": 0.5},
        )

        # 使用较小的 token 预算以确保截断
        injector = ContextInjector(max_context_tokens=500)
        prompt_result = injector.build_prompt(result, "测试")

        assert prompt_result.truncated is True

    def test_truncation_flag_not_set_when_within_budget(self):
        """少量种子不触发截断，truncated=False"""
        result = _make_ripple_result(
            seeds=[("感冒", 0.9, "医学", "急性上呼吸道感染")],
            domain_scores={"医学": 0.5},
        )

        injector = ContextInjector()
        prompt_result = injector.build_prompt(result, "测试")

        assert prompt_result.truncated is False

    def test_truncated_content_within_budget(self):
        """截断后 context token 数在预算内"""
        many_seeds = [
            (f"种子{i:03d}", 1.0 - i * 0.001, "医学", f"定义{i}" * 20)
            for i in range(100)
        ]

        result = _make_ripple_result(
            seeds=many_seeds,
            domain_scores={"医学": 0.5},
        )

        budget = 500
        injector = ContextInjector(max_context_tokens=budget)
        prompt_result = injector.build_prompt(result, "测试")

        # 截断后 token 数应 <= 预算（允许一定误差因为估算是近似的）
        assert prompt_result.context_token_count <= budget * 1.5

    def test_high_activation_seeds_preserved_during_truncation(self):
        """截断时保留高激活值种子"""
        seeds = [
            (f"低种子{i}", 0.1 + i * 0.01, "医学", f"定义{i}" * 20)
            for i in range(50)
        ]
        # 添加一个高激活种子
        seeds.insert(0, ("高激活种子", 1.0, "医学", "非常重要的定义" * 20))

        result = _make_ripple_result(
            seeds=seeds,
            domain_scores={"医学": 0.5},
        )

        injector = ContextInjector(max_context_tokens=300)
        prompt_result = injector.build_prompt(result, "测试")

        # 高激活种子应被保留
        assert "高激活种子" in prompt_result.context_block


class TestContextInjectorQueryTruncation:
    """查询文本截断测试"""

    def test_short_query_not_truncated(self):
        """短查询不截断"""
        result = _make_ripple_result(
            seeds=[("感冒", 0.9, "医学", "cold")],
            domain_scores={"医学": 0.5},
        )

        injector = ContextInjector()
        prompt_result = injector.build_prompt(result, "感冒")

        assert prompt_result.user_query == "感冒"

    def test_long_query_truncated_to_max_length(self):
        """超长查询截断到 max_query_length"""
        result = _make_ripple_result(
            seeds=[("感冒", 0.9, "医学", "cold")],
            domain_scores={"医学": 0.5},
        )

        long_query = "感冒" * 300  # 600 字符
        injector = ContextInjector(max_query_length=500)
        prompt_result = injector.build_prompt(result, long_query)

        assert len(prompt_result.user_query) == 500
        assert prompt_result.user_query == long_query[:500]

    def test_custom_max_query_length(self):
        """自定义 max_query_length"""
        result = _make_ripple_result(
            seeds=[("感冒", 0.9, "医学", "cold")],
            domain_scores={"医学": 0.5},
        )

        injector = ContextInjector(max_query_length=100)
        prompt_result = injector.build_prompt(result, "A" * 200)

        assert len(prompt_result.user_query) == 100

    def test_query_exactly_at_max_length(self):
        """查询恰好等于 max_length 不截断"""
        result = _make_ripple_result(
            seeds=[("感冒", 0.9, "医学", "cold")],
            domain_scores={"医学": 0.5},
        )

        query = "A" * 500
        injector = ContextInjector(max_query_length=500)
        prompt_result = injector.build_prompt(result, query)

        assert prompt_result.user_query == query


class TestContextInjectorEmptyActivation:
    """空激活种子测试"""

    def test_empty_seeds_returns_reasonable_prompt(self):
        """空激活种子返回合理 prompt"""
        result = _make_ripple_result(
            seeds=None,
            domain_scores={},
        )

        injector = ContextInjector()
        prompt_result = injector.build_prompt(result, "测试查询")

        assert isinstance(prompt_result, PromptResult)
        assert len(prompt_result.full_prompt) > 0
        # 空激活应显示"无激活概念"
        assert "无激活概念" in prompt_result.context_block
        assert prompt_result.user_query == "测试查询"

    def test_empty_seeds_no_paths(self):
        """空激活无路径"""
        result = _make_ripple_result(
            seeds=None,
            paths=None,
            domain_scores={},
        )

        injector = ContextInjector()
        prompt_result = injector.build_prompt(result, "测试")

        assert "无概念间关系" in prompt_result.context_block

    def test_empty_seeds_truncated_is_false(self):
        """空激活不触发截断标志"""
        result = _make_ripple_result(
            seeds=None,
            domain_scores={},
        )

        injector = ContextInjector()
        prompt_result = injector.build_prompt(result, "测试")

        assert prompt_result.truncated is False


class TestContextInjectorEstimateTokens:
    """Token 估算测试"""

    def test_estimate_tokens_positive(self):
        """Token 估算返回正整数"""
        injector = ContextInjector()
        tokens = injector._estimate_tokens("这是一段测试文本")
        assert tokens > 0
        assert isinstance(tokens, int)

    def test_estimate_tokens_empty_string(self):
        """空字符串 token 数为 0"""
        injector = ContextInjector()
        tokens = injector._estimate_tokens("")
        assert tokens == 0

    def test_estimate_tokens_formula(self):
        """Token 估算公式: len(text) * 0.6"""
        injector = ContextInjector()
        text = "测试文本"
        expected = int(len(text) * 0.6)
        assert injector._estimate_tokens(text) == expected


class TestContextInjectorStateless:
    """ContextInjector 无状态测试"""

    def test_multiple_calls_independent(self):
        """多次调用 build_prompt 互不影响"""
        injector = ContextInjector()

        result1 = _make_ripple_result(
            seeds=[("感冒", 0.9, "医学", "cold")],
            domain_scores={"医学": 0.5},
        )
        result2 = _make_ripple_result(
            seeds=[("量子力学", 0.8, "物理", "quantum")],
            domain_scores={"物理": 0.6},
        )

        pr1 = injector.build_prompt(result1, "感冒")
        pr2 = injector.build_prompt(result2, "量子力学")

        assert "感冒" in pr1.context_block
        assert "量子力学" not in pr1.context_block
        assert "量子力学" in pr2.context_block
        assert "感冒" not in pr2.context_block


if __name__ == '__main__':
    import pytest
    pytest.main([__file__, '-v'])