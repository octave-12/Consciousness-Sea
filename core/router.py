"""
确定性路由器 — 文本匹配 + BFS 涟漪传播

不调用任何模型，纯规则 + 图遍历。
一次查询的延迟 < 10ms（BFS 深度 2，内存传播）。

流程:
  查询词 → 匹配种子 → BFS 涟漪传播（深度2，衰减0.7）
  → 按领域聚合激活值 → 返回激活分布 + Top-K 种子 + 传播路径
"""

from collections import defaultdict, deque
from typing import Optional
from .graph_db import GraphDB
from .config import (
    RIPPLE_DEPTH, RIPPLE_DECAY, INITIAL_ACTIVATION,
    DOMAIN_THRESHOLD, TOP_K_SEEDS, MAX_ACTIVATION,
)


class ActivationNode:
    """激活状态中的一个节点"""
    __slots__ = ('label', 'activation', 'domain', 'definition', 'depth', 'parent')

    def __init__(self, label: str, activation: float = 0.0,
                 domain: str = '', definition: str = '',
                 depth: int = 0, parent: Optional[str] = None):
        self.label = label
        self.activation = activation
        self.domain = domain
        self.definition = definition
        self.depth = depth
        self.parent = parent

    def __repr__(self):
        return f"ActNode({self.label}, act={self.activation:.3f})"


class RippleResult:
    """一次涟漪传播的完整结果"""

    def __init__(self):
        self.activated: dict[str, ActivationNode] = {}  # label → ActivationNode
        self.paths: list[dict] = []       # 传播路径记录
        self.seed_matches: list[dict] = []  # 原始匹配到的种子
        self.domain_scores: dict[str, float] = defaultdict(float)
        self.query: str = ''

    @property
    def top_seeds(self) -> list[ActivationNode]:
        """激活值最高的 K 个种子"""
        return sorted(
            self.activated.values(),
            key=lambda n: n.activation, reverse=True
        )[:TOP_K_SEEDS]

    @property
    def selected_domains(self) -> list[str]:
        """激活值超过阈值的领域"""
        return sorted(
            [d for d, s in self.domain_scores.items() if s >= DOMAIN_THRESHOLD],
            key=lambda d: self.domain_scores[d], reverse=True
        )


def route(query: str, graph: GraphDB, user_label: Optional[str] = None) -> RippleResult:
    """
    执行一次完整的查询路由。

    Args:
        query: 用户查询文本
        graph: 知识图谱连接
        user_label: 可选，用户种子 label（用于个人偏向）

    Returns:
        RippleResult 包含激活节点、路径、领域得分
    """
    result = RippleResult()
    result.query = query

    # ── 0. 用户种子预激活（如果有） ─────────────────────
    if user_label:
        user_seed = graph.get_seed(user_label)
        if user_seed:
            _activate_user_seed(result, graph, user_seed)

    # ── 1. 文本匹配种子 ────────────────────────────────
    seeds = graph.match_seeds(query)
    result.seed_matches = seeds

    # ── 2. 第一波激活：查询词匹配的种子直接激活 ──────
    bfs_queue: deque[str] = deque()
    for seed in seeds:
        label = seed['label']
        node = ActivationNode(
            label=label,
            activation=INITIAL_ACTIVATION,
            domain=seed.get('domain', ''),
            definition=seed.get('definition', ''),
            depth=0,
        )
        result.activated[label] = node
        bfs_queue.append(label)

    # ── 3. BFS 涟漪传播 ────────────────────────────────
    for wave in range(RIPPLE_DEPTH):
        # 批量预加载：收集本轮所有边的目标和对应种子信息
        all_targets = set()
        node_edges: dict[str, list[dict]] = {}  # 缓存每节点的出边
        
        current_wave = list(bfs_queue)
        for src_label in current_wave:
            src_node = result.activated.get(src_label)
            if not src_node:
                continue
            edges = graph.outgoing_edges(src_label)
            node_edges[src_label] = edges
            for e in edges:
                all_targets.add(e['target'])
        
        # 批量加载目标种子信息
        target_info = graph.batch_get_seeds(list(all_targets))
        
        next_wave: deque[str] = deque()
        for src_label in current_wave:
            src_node = result.activated.get(src_label)
            if not src_node:
                continue
            
            for e in node_edges.get(src_label, []):
                target = e['target']
                relation = e['relation']
                weight = e.get('weight', 0.5)
                depth = src_node.depth + 1

                ripple_activation = (
                    src_node.activation * weight * (RIPPLE_DECAY ** depth)
                )

                if target in result.activated:
                    existing = result.activated[target]
                    existing.activation = min(existing.activation + ripple_activation, MAX_ACTIVATION)
                else:
                    info = target_info.get(target, {})
                    node = ActivationNode(
                        label=target,
                        activation=min(ripple_activation, MAX_ACTIVATION),
                        domain=info.get('domain', ''),
                        definition=info.get('definition', ''),
                        depth=depth,
                        parent=src_label,
                    )
                    result.activated[target] = node
                    if depth < RIPPLE_DEPTH:
                        next_wave.append(target)

                result.paths.append({
                    'source': src_label,
                    'target': target,
                    'relation': relation,
                    'weight': weight,
                    'depth': depth,
                    'ripple_activation': round(ripple_activation, 4),
                })
        
        bfs_queue = next_wave

    # ── 4. 按领域聚合激活值 ────────────────────────────
    for node in result.activated.values():
        domain = node.domain or '常识'
        result.domain_scores[domain] += node.activation

    return result


def _activate_user_seed(result: RippleResult, graph: GraphDB, user_seed: dict):
    """
    激活用户种子 → 用户关注的领域/概念预激活。

    用户种子本身的出边（关注边）会给目标种子带来预激活。
    这样"那个方程怎么推导"不需要用户再说"我关心量子力学"。
    """
    from .config import USER_PREACTIVATION

    user_label = user_seed['label']
    result.activated[user_label] = ActivationNode(
        label=user_label,
        activation=USER_PREACTIVATION,
        domain='用户',
    )

    # 用户种子的出边 → 预激活目标
    edges = graph.outgoing_edges(user_label)
    for e in edges:
        target = e['target']
        weight = e.get('weight', 0.5)
        pre_act = USER_PREACTIVATION * weight * RIPPLE_DECAY

        t_seed = graph.get_seed(target)
        result.activated[target] = ActivationNode(
            label=target,
            activation=pre_act,
            domain=t_seed.get('domain', '') if t_seed else '',
            definition=t_seed.get('definition', '') if t_seed else '',
            depth=1,
            parent=user_label,
        )
        result.paths.append({
            'source': user_label,
            'target': target,
            'relation': e['relation'],
            'weight': weight,
            'depth': 1,
            'ripple_activation': round(pre_act, 4),
        })
