"""
检索式回答引擎 (Phase 0)

不经过任何 LLM。从激活种子列表和传播路径直接拼装结构化回答。

Phase 1 升级: 激活种子 + 路径作为 context 注入 LoRA 专家模型。
"""

from __future__ import annotations

import logging
from typing import Optional

from .graph_db import GraphDB
from .router import RippleResult, ActivationNode
from .config import RELATION_NAMES, TOP_K_SEEDS, TOP_K_PATHS

log = logging.getLogger(__name__)


def answer_from_activation(
    result: RippleResult,
    graph: Optional[GraphDB] = None,
) -> str:
    """
    从涟漪传播结果拼装可读回答。

    Args:
        result: 路由器返回的 RippleResult
        graph: 可选，用于补充释义（CC-CEDICT）

    Returns:
        格式化的文本回答
    """
    lines = []

    # ── 标题 ──────────────────────────────────────────
    lines.append(f"关于「{result.query}」\n")

    # ── 激活的顶层种子（depth=0，即查询词直接匹配的）───
    top_seeds = result.top_seeds
    if not top_seeds:
        lines.append("  (未匹配到相关知识种子)")
        return '\n'.join(lines)

    depth0 = [n for n in top_seeds if n.depth == 0]
    other = [n for n in top_seeds if n.depth > 0]

    # ── 关联概念列表 ──────────────────────────────────
    all_labels = [n.label for n in top_seeds]
    lines.append(f"  关联概念: {'、'.join(all_labels)}")
    lines.append("")

    # ── 直接匹配的种子 + 释义 ─────────────────────────
    for node in depth0:
        if node.definition:
            lines.append(f"  「{node.label}」: {node.definition[:120]}")
        else:
            lines.append(f"  「{node.label}」"
                         f"{f' [{node.domain}]' if node.domain else ''}")
    if depth0:
        lines.append("")

    # ── 传播路径（Top-K，按激活值排序）────────────────
    paths = sorted(
        result.paths,
        key=lambda p: p['ripple_activation'], reverse=True
    )[:TOP_K_PATHS]

    if paths:
        lines.append("  传播路径:")
        for p in paths:
            rel_name = RELATION_NAMES.get(p['relation'], p['relation'])
            lines.append(
                f"    「{p['source']}」--[{rel_name}, "
                f"w={p['weight']:.2f}]-->「{p['target']}」"
                f"  (激活: {p['ripple_activation']:.4f})"
            )
        lines.append("")

    # ── 涟漪波及的其他概念 ────────────────────────────
    if other:
        lines.append("  涟漪波及:")
        for node in other[:10]:
            d = f" [{node.domain}]" if node.domain else ""
            lines.append(
                f"    「{node.label}」{d} "
                f"(激活值: {node.activation:.4f}, 深度: {node.depth})"
            )
        lines.append("")

    # ── 领域得分 ──────────────────────────────────────
    if result.domain_scores:
        sorted_domains = sorted(
            result.domain_scores.items(),
            key=lambda x: x[1], reverse=True
        )
        lines.append("  领域得分:")
        for domain, score in sorted_domains[:5]:
            bar = '█' * min(int(score * 10), 20)
            lines.append(f"    {domain:8s}: {score:.3f} {bar}")
        lines.append("")

    # ── 选定领域 ──────────────────────────────────────
    selected = result.selected_domains
    if selected:
        lines.append(f"  选定领域: {', '.join(selected)}")
    else:
        lines.append("  选定领域: 常识 (无领域超过阈值)")

    return '\n'.join(lines)


def answer_as_dict(result: RippleResult) -> dict:
    """
    以结构化 JSON 形式返回结果（用于 API / Phase 1 的 context 注入）。
    """
    return {
        'query': result.query,
        'activated_seeds': [
            {
                'label': n.label,
                'activation': round(n.activation, 4),
                'domain': n.domain,
                'definition': n.definition[:200] if n.definition else '',
                'depth': n.depth,
            }
            for n in result.top_seeds
        ],
        'paths': [
            {
                'source': p['source'],
                'target': p['target'],
                'relation': p['relation'],
                'weight': p['weight'],
                'depth': p['depth'],
            }
            for p in sorted(
                result.paths,
                key=lambda x: x['ripple_activation'], reverse=True
            )[:TOP_K_PATHS]
        ],
        'domain_scores': {
            d: round(s, 4) for d, s in
            sorted(result.domain_scores.items(), key=lambda x: x[1], reverse=True)
        },
        'selected_domains': result.selected_domains,
        'matched_seeds': len(result.seed_matches),
        'total_activated': len(result.activated),
    }


def answer_with_expert(
    result: RippleResult,
    graph: Optional[GraphDB],
    expert_manager: Optional[object] = None,
) -> dict:
    """使用专家模型生成回答（Phase 1），不可用时降级到 Phase 0。

    核心流程:
      1. expert_manager 为 None 或 expert_available=False → 降级到 answer_from_activation()
      2. 空激活（top_seeds 为空）→ 降级到 Phase 0
      3. 使用 ContextInjector 构造 prompt
      4. 单领域 → ExpertManager.infer()
      5. 多领域 → ExpertManager.infer_multi_domain() + CrossValidator
      6. 组装混合回答（expert_answer + retrieval_answer）

    Args:
        result: 路由器返回的 RippleResult
        graph: 知识图谱连接（用于补充释义）
        expert_manager: ExpertManager 实例（可选）

    Returns:
        包含 expert_answer, retrieval_answer, expert_domain, expert_available,
        reliability_score, cross_validation_status, cross_validation_discount 的字典
    """
    from .context_injector import ContextInjector
    from .cross_validator import CrossValidator, CrossValidationStatus
    from .config import EXPERT_MAX_NEW_TOKENS

    # ── 降级条件 1: expert_manager 为 None 或不可用 ──
    if expert_manager is None or not expert_manager.expert_available:
        retrieval_text = answer_from_activation(result, graph)
        return {
            'expert_answer': None,
            'retrieval_answer': retrieval_text,
            'expert_domain': None,
            'expert_available': False,
            'reliability_score': None,
            'cross_validation_status': 'none',
            'cross_validation_discount': 1.0,
        }

    # ── 降级条件 2: 空激活 ──
    if not result.top_seeds:
        retrieval_text = answer_from_activation(result, graph)
        return {
            'expert_answer': None,
            'retrieval_answer': retrieval_text,
            'expert_domain': None,
            'expert_available': False,
            'reliability_score': None,
            'cross_validation_status': 'none',
            'cross_validation_discount': 1.0,
        }

    # ── 构造上下文 prompt ──
    try:
        injector = ContextInjector()
        prompt_result = injector.build_prompt(result, result.query)
        prompt = prompt_result.full_prompt
    except Exception as e:
        log.error("构造上下文 prompt 失败，降级到 Phase 0: %s", e, exc_info=True)
        retrieval_text = answer_from_activation(result, graph)
        return {
            'expert_answer': None,
            'retrieval_answer': retrieval_text,
            'expert_domain': None,
            'expert_available': False,
            'reliability_score': None,
            'cross_validation_status': 'none',
            'cross_validation_discount': 1.0,
        }

    # ── 检索式回答（始终保留）──
    retrieval_text = answer_from_activation(result, graph)

    # ── 专家推理 ──
    selected_domains = result.selected_domains or []

    try:
        if len(selected_domains) <= 1:
            # 单领域推理
            target_domain = selected_domains[0] if selected_domains else ""
            inference_result = expert_manager.infer(
                prompt=prompt,
                target_domain=target_domain,
                max_new_tokens=EXPERT_MAX_NEW_TOKENS,
            )

            if inference_result.fallback or not inference_result.answer_text:
                # 专家推理降级
                return {
                    'expert_answer': None,
                    'retrieval_answer': retrieval_text,
                    'expert_domain': None,
                    'expert_available': False,
                    'reliability_score': None,
                    'cross_validation_status': 'none',
                    'cross_validation_discount': 1.0,
                }

            return {
                'expert_answer': inference_result.answer_text,
                'retrieval_answer': retrieval_text,
                'expert_domain': inference_result.domain,
                'expert_available': True,
                'reliability_score': inference_result.reliability,
                'cross_validation_status': 'none',
                'cross_validation_discount': 1.0,
            }

        else:
            # 多领域推理 + 交叉验证
            inference_results = expert_manager.infer_multi_domain(
                prompt=prompt,
                domains=selected_domains,
                max_new_tokens=EXPERT_MAX_NEW_TOKENS,
            )

            if not inference_results:
                # 所有领域推理均降级
                return {
                    'expert_answer': None,
                    'retrieval_answer': retrieval_text,
                    'expert_domain': None,
                    'expert_available': False,
                    'reliability_score': None,
                    'cross_validation_status': 'none',
                    'cross_validation_discount': 1.0,
                }

            # 执行交叉验证
            answers = [r.answer_text for r in inference_results]
            domains = [r.domain for r in inference_results]

            validator = CrossValidator()
            cv_result = validator.validate(answers, domains)

            # 选择主回答
            if cv_result.merged_answer:
                expert_answer = cv_result.merged_answer
            else:
                # 取第一个非降级结果作为主回答
                expert_answer = inference_results[0].answer_text

            # 主领域和可靠性取第一个推理结果
            primary = inference_results[0]

            return {
                'expert_answer': expert_answer,
                'retrieval_answer': retrieval_text,
                'expert_domain': primary.domain,
                'expert_available': True,
                'reliability_score': primary.reliability,
                'cross_validation_status': cv_result.status.value,
                'cross_validation_discount': cv_result.discount,
            }

    except Exception as e:
        log.error("专家推理异常，降级到 Phase 0: %s", e, exc_info=True)
        return {
            'expert_answer': None,
            'retrieval_answer': retrieval_text,
            'expert_domain': None,
            'expert_available': False,
            'reliability_score': None,
            'cross_validation_status': 'none',
            'cross_validation_discount': 1.0,
        }
