# 识海 (Consciousness Sea)

> **一种新范式：以图节点为知识种子、以激活传播为检索机制、以小专家模型为回答引擎的去中心化认知架构。**
>
> 没有字场。没有能量景观。没有梯度下降推理。
> 只有种子（概念节点）、涟漪（激活传播）、业力（连接权重）。在每一次查询中临时组织自己。

---

## 不是什么

不是 LLM。不是 RAG。不是 MoE。不是 prompt engineering。不是更好的 chatbot。

## 是什么

一个**去中心化、自组织、涌现认知**的多智能体架构。从一群独立小专家开始，逐步拆掉所有人为设计的边界，最终汇成一片激活与衰减的连续识海。

---

## 三步演进

| 步 | 状态 | 做什么 |
|:--:|------|--------|
| **第一步** | 现在就可以用 | 共享知识库 + 一群领域小专家 + 确定性路由器 |
| **第二步** | 1-3年 | 专家学会自己叫人，路由从运行时经验中生长 |
| **第三步** | 3-10年 | 连"专家"都消解掉，只剩节点间的激活与衰减 |

每一步减少一个"人做决定"的地方。第一步人不再写答案（那是 prompt engineering），第二步人不再写路由规则，第三步人不再定义专家边界。

详细路线：[`docs/roadmap.md`](docs/roadmap.md)

## 已知缺陷与补齐计划

架构不是完美的。9 个已知问题，分四个阶段补齐——从"跑得起来"到"自我生长"。

详见：[`docs/supplement-roadmap.md`](docs/supplement-roadmap.md)

---

## 为什么叫识海

"场"是物理学术语——精确，但没有生命感。

"识海"来自唯识宗的阿赖耶识——一切种子的储存处，不是思考本身，而是**思考得以发生的那个无限基底**。

| 概念 | 在识海中的对应 |
|------|-------------|
| 种子 | 知识节点（概念/事实/规则） |
| 涟漪 | 一次查询激发的激活传播 |
| 业力 | 节点间的连接权重（Hebbian 更新） |
| 熏习 | 反复激活 → 连接增强 → 形成稳定路径 |
| 现行 | 当前激活状态（推理的瞬时快照） |

详见：[`docs/consciousness-sea-as-framework.md`](docs/consciousness-sea-as-framework.md)

---

## 架构概览

```
         ┌──────────────────────────────────┐
         │      共享知识库 (Graph DB)        │
         │   概念节点 + 关系边 + 激活值       │
         └──────────────┬───────────────────┘
                        │
    ┌───────┬───────┬───┴───┬──────┬──────┐
  诗词    法律    医学    数学    物理    常识
    │       │       │       │      │      │
    └───────┴───────┴───┬───┴──────┴──────┘
                        │
              ┌─────────┴──────────┐
              │   确定性路由器      │
              │ (概念激活模式匹配)  │
              └────────────────────┘
```

详见：[`docs/architecture.md`](docs/architecture.md)

---

## 项目结构

```
consciousness-sea/
├── backend/
│   ├── cli.py                          # CLI 入口
│   ├── config/
│   │   └── .env.example                # 环境变量模板
│   ├── scripts/
│   │   ├── import_knowledge_base.py     # 知识库导入
│   │   └── import_related.py           # 关系导入
│   └── src/
│       └── consciousness_sea/          # 核心包
│           ├── domain/                  # 领域层：路由、回答、图数据库、校验
│           ├── expert/                  # 专家层：专家管理、上下文注入、交叉验证
│           ├── infrastructure/          # 基础设施：连接池、会话、观测、配置
│           ├── interfaces/              # 接口层：FastAPI 服务
│           ├── learning/                # 学习层：熏习、冷启动、检查点、别名扩展
│           ├── metacognition/           # 元认知层：元种子、守卫环、认知目标、好奇心
│           └── perception/              # 感知层：多模态锚定、Hebbian 关联
├── frontend/                            # Vue3 + Vite + TypeScript 前端
├── tests/                               # 996+ 测试用例
├── docs/                                # 文档
├── data/                                # 数据文件
├── pyproject.toml                       # 项目配置
└── LICENSE                              # Apache 2.0
```

---

## 六大阶段

| Phase | 主题 | 核心模块 |
|:-----:|------|---------|
| 0-1 | 基座 | GraphDB、Router、Verifier、Tokenizer、DomainInference |
| 1 | 专家组 | ExpertManager、ContextInjector、CrossValidator、ExpertReliability |
| 2 | 熏习 | DistillationPool、KarmaCleaner、ParamEvaluator、ParamStats |
| 3 | 自生长 | AliasExpander、SeedCandidate、ColdStart、Checkpoint |
| 4 | 元种子 | MetaSeedManager、GuardianLoop |
| 5 | 认知目标 | CognitiveGoalManager、CuriosityEngine |
| 6 | 具身感知 | PerceptionManager、VisualAnchor、AudioAnchor、SomaticAnchor、HebbianBinder、MultimodalAligner |

---

## 快速开始

### 环境要求

- Python 3.12+
- SQLite（内置）

### 安装

```bash
git clone https://gitee.com/octave-12/consciousness-sea.git
cd consciousness-sea

pip install -e ".[dev]"
```

### 运行

```bash
consciousness-sea-api

python -m backend.cli query "量子力学的基本原理"
```

### 测试

```bash
pytest
pytest tests/test_router.py
```

---

## 技术栈

| 层 | 技术 |
|----|------|
| 后端 | Python 3.12+ / FastAPI / Pydantic |
| 数据库 | SQLite（邻接表） |
| 前端 | Vue 3 / Vite / TypeScript |
| AI | LoRA 热切换 / Ollama（可选） |
| 测试 | pytest / pytest-asyncio / httpx |

---

## 与龙珠的关系

龙珠 = **一个引擎的进化史**。从 GNG 到 94K 字场到能量景观到连续场——四代都在追问"认知能否在一个结构中完成"。

识海 = **同一个人、不同的问题**。不再追问"一个结构够不够"，而是追问"结构有没有可能不是预设的，而是自己长出来的"。

龙珠是**做出来的**。识海是**剩下来的**。

---

## 许可证

[Apache License 2.0](LICENSE)

---

*实验中的故事 · 第三代*
