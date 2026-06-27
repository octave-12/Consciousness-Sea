"""
检索式回答引擎 (Phase 0)

不经过任何 LLM。从激活种子列表和传播路径直接拼装结构化回答。

Phase 1 升级: 激活种子 + 路径作为 context 注入 LoRA 专家模型。
"""

from typing import Optional
from .graph_db import GraphDB
from .router import RippleResult, ActivationNode
from .config import RELATION_NAMES, TOP_K_SEEDS, TOP_K_PATHS


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
