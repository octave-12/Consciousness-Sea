# 架构 — 第一步实现

> 这一步今天就可以开始。需要的只有 GPU + Graph DB + 几个小模型。

---

## 系统拓扑

```
                          ┌──────────┐
                          │   用户    │
                          └────┬─────┘
                               │ 自然语言
                               ▼
┌──────────────────────────────────────────────────────────────┐
│                    确定性路由器                               │
│                                                               │
│  1. 分词 → 查询词匹配知识库种子 → 第一波激活                   │
│  2. 激活沿边传播（BFS，深度2）→ 第二波激活                     │
│  3. 聚合激活值 → 阈值判定 → 选出领域                          │
│  4. 激活种子 Top-K + 传播路径 → 拼装检索式回答                 │
└──────────────────────────────────────────────────────────────┘
                               │
                               ▼
┌──────────────────────────────────────────────────────────────┐
│                    检索式回答引擎                              │
│                                                               │
│  Phase 0: 无模型。激活种子列表 + 关系路径 → 结构化输出         │
│  Phase 1: 单模型 + LoRA → 在激活种子约束下生成自然语言         │
└──────────────────────────┬───────────────────────────────────┘
                           │
                           ▼
                ┌─────────────────────┐
                │    答案校验器        │
                │  回答关键词 vs       │
                │  激活种子一致性校验  │
                │  → 置信度 → 熏习方向 │
                └──────────┬──────────┘
                           │
                           ▼
                       用户
```

---

## 组件详设

### 1. 共享知识库

**选型**: SQLite + 邻接表（复用龙珠概念图结构，无需新数据库）

**节点结构**:
```
// 概念节点
{
  id: "electricity",
  label: "电",
  type: "CONCEPT",
  aliases: ["电力", "电能", "电流"],
  activation: 0.0,
  domain: "物理",
  meta: {
    source: "wikipedia",
    frequency: 2341
  }
}

// 用户节点
{
  id: "user_lzk",
  label: "李泽坤",
  type: "USER",
  activation: 0.0,
  preferences: {
    style: "concise",
    level: "expert"
  }
}
```

**边结构**:
```
{
  source: "electricity",
  target: "magnetism",
  relation: "RELATED",
  weight: 0.85
}
```

**关系类型**: IS_A, PART_OF, RELATED, COOCCURS, OPPOSES, CAUSES, DEFINED_AS

### 2. 确定性路由器

**不是模型，是规则**。核心只有一个函数：

```python
def route(query: str, graph: GraphDB) -> list[Expert]:
    # 1. 分词 → 文本匹配种子（匹配 label 和 aliases，不是向量相似度）
    seeds = graph.match_seeds(query)  # ["感冒", "咳嗽", "药"]

    # 2. 第一波激活
    for seed in seeds:
        graph.activate(seed, value=1.0)

    # 3. 第二波激活 — 沿边传播 (BFS depth=2)
    for _ in range(2):
        for node in graph.active_nodes():
            for edge in graph.outgoing(node):
                graph.activate(edge.target,
                    value=edge.weight * node.activation * 0.7)

    # 4. 按领域聚合激活值
    domain_scores = {}
    for node in graph.active_nodes():
        domain_scores[node.domain] += node.activation

    # 5. 阈值 → 选定专家
    threshold = 0.3
    selected = [d for d, score in domain_scores.items()
                if score > threshold]

    # 6. 无命中 → 常识兜底
    if not selected:
        selected = ["常识"]

    return selected
```

**关键**: 路由不调用模型，只查图和传播激活值。查询延迟 = 图遍历延迟（毫秒级）。

### 3. 回答引擎

#### Phase 0：检索式回答（无专家模型）

不经过任何 LLM。从激活种子和子图直接拼装回答：

```python
def answer_from_activation(graph: GraphDB, active_seeds: list) -> str:
    """从激活种子拼装可读回答，不调用任何模型"""
    # 激活种子按 activation 降序
    top = sorted(active_seeds, key=lambda s: s.activation, reverse=True)[:10]

    # 给相关种子加上关系解释
    lines = []
    for seed in top:
        if seed.definition:
            lines.append(f"「{seed.label}」: {seed.definition}")

    # 找出 Top-5 种子之间的路径
    paths = graph.paths_between(top[:5])

    # 拼装自然可读的输出
    for path in paths:
        lines.append(
            f"「{path.source.label}」{path.relation}「{path.target.label}」"
            f"—— {relation_explanation(path.relation, path.source, path.target)}"
        )

    return "\n".join(lines)
```

**为什么 Phase 0 没有自然语言生成**: Phase 0 的目标是验证架构——路由、涟漪传播、校验器。这些和模型质量无关。激活种子列表 + 传播路径本身就是答案——能看到系统在想什么，比黑盒回答更有价值。

#### Phase 1 升级：单模型 + LoRA 切换

一个 7B Q4 基座模型常驻显存（~5GB），6 个 LoRA 适配器按需切换（<0.1s）：

```python
class ExpertEngine:
    def __init__(self):
        self.base_model = load_quantized("7B", "Q4_K_M")  # ~5GB
        self.loras = {
            "医学": load_lora("medical_lora.bin"),    # ~50MB
            "法律": load_lora("legal_lora.bin"),
            "诗词": load_lora("poetry_lora.bin"),
            "物理": load_lora("physics_lora.bin"),
            "数学": load_lora("math_lora.bin"),
            "常识": load_lora("common_lora.bin"),
        }
        self.current_lora = None

    def switch_to(self, domain: str):
        if self.current_lora:
            self.base_model.unload_lora(self.current_lora)
        self.base_model.load_lora(self.loras[domain])
        self.current_lora = domain

    def answer(self, query: str, context: dict) -> dict:
        # context = {激活种子, 传播路径, 释义}
        # 激活种子列表约束专家的回答范围 → 抑制幻觉
        prompt = self._build_prompt(query, context)
        response = self.base_model.generate(prompt)
        return {
            "answer": response,
            "domain": self.current_lora,
            "uncertainties": []
        }
```

**混合回答策略**: 检索式回答（Phase 0 路径）的输出作为 context 注入专家模型 → 专家在这个约束下生成自然语言 → 校验器检查回答关键词是否在激活区域内。

### 4. 答案校验器 + 熏习引擎

校验器的输出不只是"可信/不可信"，它驱动熏习的方向：

```python
def verify(answer: str, active_nodes: list, graph: GraphDB) -> dict:
    # 提取答案中的关键词
    keywords = extract_keywords(answer)

    # 匹配关键词到知识库节点
    matched = 0
    for kw in keywords:
        if graph.has_node(kw) and graph.is_active(kw):
            matched += 1

    # 置信度 = 匹配数 / 关键词数
    confidence = matched / len(keywords) if keywords else 0.5

    # 根据置信度决定熏习方向
    if confidence > 0.7:
        karma_delta = +0.01   # 正向熏习
    elif confidence < 0.3:
        karma_delta = -0.01   # 负向熏习（纠正错误关联）
    else:
        karma_delta = 0       # 不确定，不做修改

    # 熏习：对所有在此次查询中被激活的种子对，按 karma_delta 修改业力
    for pair in graph.co_activated_pairs():
        graph.adjust_karma(pair, karma_delta)

    return {"confidence": confidence, "karma_delta": karma_delta}
```

**负熏习不会立刻抹掉错误关联**——十次正熏习建立的业力需要十次负熏习来翻回。这是有意设计的：系统不会因为一次低置信度就丢弃一条路径，但持续的负面信号会自然冷却它。

---

## 数据流（一次完整查询）

```
用户李泽坤: "那个方程怎么推导"

1. 用户种子激活:
   user_lzk → 激活1.0
   业力边: user_lzk —[关注:0.9]→ 量子力学
             user_lzk —[关注:0.7]→ 认知架构
   量子力学种子被预激活至 0.9（用户种子的业力偏向）

2. 分词: ["那个", "方程", "推导"]

3. 知识库激活:
   方程 → 激活1.0
     → RELATED → 数学公式 (激活0.85)
     → RELATED → 薛定谔方程 (激活0.72)  ← 被量子力学的预激活叠加
   量子力学 → 激活0.9（来自用户种子）
     → EQUATION → 薛定谔方程 (激活0.85)

   薛定谔方程被两个路径激活：0.72 + 0.85 → 总和最高

4. 聚合激活 → 领域得分:
   物理: 4.1
   数学: 2.7
   其他: <0.3

5. 选定专家: [物理, 数学]

6. 专家回答后 → 熏习:
   被激活过的种子间业力 +0.01:
   薛定谔方程 ↔ 量子力学: +0.01
   薛定谔方程 ↔ 推导: +0.01
   user_lzk ↔ 量子力学: +0.01  (用户业力也在熏习)

7. 激活值衰减归零。业力落盘永久保留。

8. 返回用户
```

### 无用户种子 vs 有用户种子的区别

```
无用户种子:
  "那个方程怎么推导" → 分词("方程","推导")
  → 不知道"那个"指什么 → 物理/数学/工程全匹配 → 通用回答

有用户种子:
  "那个方程怎么推导" → 用户种子预激活"量子力学"
  → 涟漪在"量子力学"和"薛定谔方程"附近形成共振
  → 精确命中
```

---

## 性能预估

| 操作 | Phase 0 | Phase 1 |
|------|:--:|:--:|
| 知识库激活 + 传播 | <10ms | <10ms |
| 路由器判定 | <5ms | <5ms |
| 回答生成 | <50ms（检索式拼装） | 2-5s（LoRA 推理） |
| 答案校验 + 熏习 | <50ms | <50ms |
| **一次查询 total** | **< 100ms** | **2-5s** |

---

## 启动清单 (Phase 0)

- [ ] 导入知识库 (`import_knowledge_base.py` — 四层混合, ~2 小时)
- [ ] 实现确定性路由器（文本匹配 + BFS 涟漪传播）
- [ ] 实现检索式回答引擎（激活种子 + 路径拼装）
- [ ] 实现答案校验器 + 熏习引擎（置信度 → ±0.01）
- [ ] 端到端集成测试（10 个查询全部路由正确）

## 升级清单 (Phase 1)

- [ ] 部署 7B Q4 基座模型 (llama.cpp / Ollama)
- [ ] 训练 6 个领域 LoRA（或先用 system prompt 代替 → Phase 2 再训）
- [ ] 实现 LoRA 热切换
- [ ] 混合回答策略：检索式 context 注入 → 自然语言生成
- [ ] 校验器闭环验证自然语言回答

---

*实验中的故事 · 工程*
