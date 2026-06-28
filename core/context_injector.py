"""
ContextInjector — 将涟漪传播激活区域构造为专家 prompt

职责:
  - 从 RippleResult 提取激活种子和传播路径
  - 按 token 预算截断（≤ CONTEXT_MAX_TOKENS）
  - 构造结构化 prompt（system + context + query）
  - 用户查询文本长度截断保护

ContextInjector 为无状态组件，每次 build_prompt() 调用独立构造 prompt，
不持有可变状态。
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

from .router import RippleResult
from .config import CONTEXT_MAX_TOKENS, CONTEXT_MAX_QUERY_LENGTH, RELATION_NAMES

log = logging.getLogger(__name__)


@dataclass
class PromptResult:
    """Prompt 构造结果"""

    system_prompt: str
    context_block: str
    user_query: str
    full_prompt: str
    context_token_count: int
    truncated: bool  # 是否发生了截断


class ContextInjector:
    """上下文注入器 — 将激活区域构造为专家 prompt

    职责:
      - 从 RippleResult 提取激活种子和传播路径
      - 按 token 预算截断（≤ CONTEXT_MAX_TOKENS）
      - 构造结构化 prompt（system + context + query）
      - 用户查询文本长度截断保护

    Args:
        max_context_tokens: 上下文最大 token 数
        max_query_length: 用户查询文本最大字符数
    """

    # 系统提示模板
    SYSTEM_PROMPT_TEMPLATE: str = (
        "你是识海知识系统的领域专家。请严格基于下方提供的知识上下文回答用户问题。"
        "不要编造上下文中未提及的信息。如果上下文不足以回答问题，请明确说明。"
        "回答应简洁、准确，使用自然语言。"
    )

    # 上下文块模板
    CONTEXT_TEMPLATE: str = (
        "## 知识上下文\n\n"
        "### 激活的概念\n{seeds_block}\n\n"
        "### 概念间关系\n{paths_block}\n"
    )

    def __init__(
        self,
        max_context_tokens: int = CONTEXT_MAX_TOKENS,
        max_query_length: int = CONTEXT_MAX_QUERY_LENGTH,
    ) -> None:
        self._max_context_tokens = max_context_tokens
        self._max_query_length = max_query_length

    def build_prompt(
        self,
        result: RippleResult,
        query: str,
    ) -> PromptResult:
        """构造完整的专家推理 prompt

        流程:
          1. 提取激活种子（按激活值降序）
          2. 提取传播路径（按 ripple_activation 降序）
          3. 构造 seeds_block 和 paths_block
          4. 按 token 预算截断（优先保留高激活值内容）
          5. 截断用户查询文本
          6. 拼装完整 prompt

        Args:
            result: 涟漪传播结果
            query: 用户查询文本

        Returns:
            PromptResult 包含完整 prompt 和构造元信息
        """
        # 1. 提取激活种子（按激活值降序）
        seeds = sorted(
            result.activated.values(),
            key=lambda n: n.activation,
            reverse=True,
        )
        seed_dicts = [
            {
                "label": s.label,
                "domain": s.domain,
                "definition": s.definition,
                "activation": s.activation,
            }
            for s in seeds
        ]

        # 2. 提取传播路径（按 ripple_activation 降序）
        paths = sorted(
            result.paths,
            key=lambda p: p.get("ripple_activation", 0.0),
            reverse=True,
        )

        # 3. 按 token 预算截断
        truncated_seeds, truncated_paths, was_truncated = self._truncate_to_budget(
            seed_dicts, paths, self._max_context_tokens
        )

        # 4. 构造文本块
        seeds_block = self._build_seeds_block(truncated_seeds)
        paths_block = self._build_paths_block(truncated_paths)

        # 5. 构造上下文块
        context_block = self.CONTEXT_TEMPLATE.format(
            seeds_block=seeds_block or "（无激活概念）",
            paths_block=paths_block or "（无概念间关系）",
        )

        # 6. 截断用户查询
        user_query = self._truncate_query(query, self._max_query_length)

        # 7. 拼装完整 prompt
        full_prompt = (
            f"{self.SYSTEM_PROMPT_TEMPLATE}\n\n"
            f"{context_block}\n"
            f"{user_query}"
        )

        # 8. 估算 token 数
        context_token_count = self._estimate_tokens(context_block)

        return PromptResult(
            system_prompt=self.SYSTEM_PROMPT_TEMPLATE,
            context_block=context_block,
            user_query=user_query,
            full_prompt=full_prompt,
            context_token_count=context_token_count,
            truncated=was_truncated,
        )

    def _build_seeds_block(self, seeds: list[dict]) -> str:
        """构造激活种子文本块

        格式:
          - 感冒 [医学]: 急性上呼吸道感染...
          - 发热 [医学]: 体温升高的生理反应...
        """
        if not seeds:
            return ""

        lines = []
        for seed in seeds:
            label = seed.get("label", "")
            domain = seed.get("domain", "")
            definition = seed.get("definition", "")

            if domain and definition:
                lines.append(f"- {label} [{domain}]: {definition}")
            elif domain:
                lines.append(f"- {label} [{domain}]")
            elif definition:
                lines.append(f"- {label}: {definition}")
            else:
                lines.append(f"- {label}")

        return "\n".join(lines)

    def _build_paths_block(self, paths: list[dict]) -> str:
        """构造传播路径文本块

        格式:
          - 感冒 --[常与...共现]--> 发热 (关联度: 0.85)
          - 感冒 --[导致]--> 头痛 (关联度: 0.72)
        """
        if not paths:
            return ""


        lines = []
        for path in paths:
            source = path.get("source", "")
            target = path.get("target", "")
            relation = path.get("relation", "")
            weight = path.get("weight", 0.0)

            # 翻译关系名
            relation_display = RELATION_NAMES.get(relation, relation)

            lines.append(
                f"- {source} --[{relation_display}]--> {target} (关联度: {weight:.2f})"
            )

        return "\n".join(lines)

    def _estimate_tokens(self, text: str) -> int:
        """估算文本 token 数

        策略: 中文约 1.5 字符/token，英文约 4 字符/token。
        简化估算: len(text) * 0.6（混合文本的经验系数）。
        实际推理时由 tokenizer 精确计算，此处仅用于预算控制。
        """
        return int(len(text) * 0.6)

    def _truncate_to_budget(
        self,
        seeds: list[dict],
        paths: list[dict],
        budget: int,
    ) -> tuple[list[dict], list[dict], bool]:
        """按 token 预算截断种子和路径

        策略:
          1. 先填入种子（按激活值降序），直到接近预算的 60%
          2. 再填入路径（按 ripple_activation 降序），直到接近预算的 40%
          3. 确保总 token 数 ≤ budget

        Args:
            seeds: 激活种子列表（已按激活值降序排序）
            paths: 传播路径列表（已按 ripple_activation 降序排序）
            budget: token 预算

        Returns:
            (截断后种子, 截断后路径, 是否发生了截断)
        """
        seeds_budget = int(budget * 0.6)
        paths_budget = int(budget * 0.4)

        # ── 截断种子 ──
        truncated_seeds: list[dict] = []
        cumulative_seeds_tokens = 0
        for seed in seeds:
            # 估算单条种子的 token 数
            label = seed.get("label", "")
            domain = seed.get("domain", "")
            definition = seed.get("definition", "")
            seed_text = f"- {label} [{domain}]: {definition}"
            current_seed_tokens = self._estimate_tokens(seed_text)

            if cumulative_seeds_tokens + current_seed_tokens <= seeds_budget:
                truncated_seeds.append(seed)
                cumulative_seeds_tokens += current_seed_tokens
            else:
                # 预算已满，停止添加
                break

        # ── 截断路径 ──
        truncated_paths: list[dict] = []
        paths_tokens = 0
        for path in paths:
            source = path.get("source", "")
            target = path.get("target", "")
            relation = path.get("relation", "")
            weight = path.get("weight", 0.0)
            path_text = f"- {source} --[{relation}]--> {target} (关联度: {weight:.2f})"
            path_tokens = self._estimate_tokens(path_text)

            if paths_tokens + path_tokens <= paths_budget:
                truncated_paths.append(path)
                paths_tokens += path_tokens
            else:
                break

        # 判断是否发生了截断
        was_truncated = len(truncated_seeds) < len(seeds) or len(truncated_paths) < len(paths)

        return truncated_seeds, truncated_paths, was_truncated

    def _truncate_query(self, query: str, max_length: int = CONTEXT_MAX_QUERY_LENGTH) -> str:
        """截断用户查询文本

        Args:
            query: 原始查询文本
            max_length: 最大字符数

        Returns:
            截断后的查询文本
        """
        if len(query) <= max_length:
            return query
        truncated = query[:max_length]
        log.debug("查询文本已截断: %d → %d 字符", len(query), max_length)
        return truncated