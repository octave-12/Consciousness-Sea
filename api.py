#!/usr/bin/env python3
"""
识海 HTTP API 服务 — FastAPI 实现

端点:
  POST /api/v1/query    — 执行查询
  GET  /api/v1/stats    — 数据库统计
  GET  /api/v1/history  — 查询历史
  GET  /health          — 健康检查
"""

from __future__ import annotations

import logging
import sqlite3
from pathlib import Path
from typing import Optional

import uvicorn
from fastapi import Depends, FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from core import (
    GraphDB,
    route,
    answer_from_activation,
    answer_as_dict,
    verify,
    apply_karma,
)
from core.config import (
    DEFAULT_DB_PATH,
    API_HOST,
    API_PORT,
    API_TIMEOUT,
    HISTORY_DEFAULT_LIMIT,
    HISTORY_MAX_LIMIT,
    TOP_K_PATHS,
)
from core.query_history import record_query, get_history

log = logging.getLogger(__name__)

# ═══════════════════════════════════════════════════════════
#  FastAPI 应用初始化
# ═══════════════════════════════════════════════════════════

app = FastAPI(title="识海 API", version="0.1.0")

# CORS: 仅允许 http://localhost:*
app.add_middleware(
    CORSMiddleware,
    allow_origin_regex=r"http://localhost:\d+",
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ═══════════════════════════════════════════════════════════
#  Pydantic 请求/响应模型
# ═══════════════════════════════════════════════════════════


class QueryRequest(BaseModel):
    """查询请求"""

    query: str = Field(..., min_length=1, max_length=1000, description="查询文本")
    user: Optional[str] = Field(None, description="用户种子 label")
    dry_run: bool = Field(False, description="dry-run 模式不写回业力")


class ActivatedSeed(BaseModel):
    """激活的种子"""

    label: str
    activation: float
    domain: str
    definition: str
    depth: int


class PropagationPath(BaseModel):
    """传播路径"""

    source: str
    target: str
    relation: str
    weight: float
    depth: int
    ripple_activation: float


class QueryResponse(BaseModel):
    """查询响应"""

    query: str
    activated_seeds: list[ActivatedSeed]
    paths: list[PropagationPath]
    domain_scores: dict[str, float]
    selected_domains: list[str]
    matched_seeds: int
    total_activated: int
    confidence: float
    karma_direction: int
    decision: str


class StatsResponse(BaseModel):
    """统计响应"""

    nodes: int
    edges: int
    relations: dict[str, int]
    domain_distribution: dict[str, int]
    db_size_mb: float


class HistoryRecord(BaseModel):
    """查询历史记录"""

    query_id: int
    query_text: str
    matched_seeds_count: int
    selected_domains: list[str]  # 领域列表
    confidence: float
    karma_direction: int
    created_at: str


class HistoryResponse(BaseModel):
    """查询历史响应"""

    records: list[HistoryRecord]
    total: int
    limit: int
    offset: int


# ═══════════════════════════════════════════════════════════
#  依赖注入
# ═══════════════════════════════════════════════════════════


def get_graph_db() -> GraphDB:
    """每个请求获取数据库连接"""
    graph = GraphDB(DEFAULT_DB_PATH)
    graph.connect()
    try:
        yield graph
    finally:
        graph.close()


# ═══════════════════════════════════════════════════════════
#  端点实现
# ═══════════════════════════════════════════════════════════


@app.get("/health")
def health_check():
    """健康检查"""
    return {"status": "ok"}


@app.post("/api/v1/query", response_model=QueryResponse)
def query_endpoint(
    req: QueryRequest,
    graph: GraphDB = Depends(get_graph_db),
):
    """
    执行一次查询。

    流程: route() → answer_from_activation() → verify() → apply_karma()
    """
    # 空查询校验（Pydantic min_length=1 已拦截，这里做防御性检查）
    if not req.query or not req.query.strip():
        raise HTTPException(status_code=400, detail="query 不能为空")

    try:
        # 1. 路由
        result = route(req.query, graph, user_label=req.user)

        # 2. 回答
        answer_text = answer_from_activation(result, graph)

        # 3. 校验
        verdict = verify(answer_text, result, graph)

        # 4. 熏习
        karma_count = apply_karma(
            result, graph, verdict["karma_direction"], dry_run=req.dry_run
        )
        log.info(
            f"查询完成: query='{req.query}', confidence={verdict['confidence']}, "
            f"karma_direction={verdict['karma_direction']:+d}, "
            f"karma_edges={karma_count}, dry_run={req.dry_run}"
        )

        # 5. 记录查询历史（失败不影响响应）
        try:
            record_query(
                conn=graph.conn,
                query_text=req.query,
                matched_seeds_count=len(result.seed_matches),
                selected_domains=result.selected_domains if result.selected_domains else [],
                confidence=verdict["confidence"],
                karma_direction=verdict["karma_direction"],
            )
        except Exception as e:
            log.warning(f"记录查询历史失败: {e}")

        # 6. 组装响应
        activated_seeds = [
            ActivatedSeed(
                label=n.label,
                activation=round(n.activation, 4),
                domain=n.domain,
                definition=n.definition[:200] if n.definition else "",
                depth=n.depth,
            )
            for n in result.top_seeds
        ]


        paths = [
            PropagationPath(
                source=p["source"],
                target=p["target"],
                relation=p["relation"],
                weight=p["weight"],
                depth=p["depth"],
                ripple_activation=p["ripple_activation"],
            )
            for p in sorted(
                result.paths, key=lambda x: x["ripple_activation"], reverse=True
            )[:TOP_K_PATHS]
        ]

        domain_scores = {
            d: round(s, 4)
            for d, s in sorted(
                result.domain_scores.items(), key=lambda x: x[1], reverse=True
            )
        }

        return QueryResponse(
            query=result.query,
            activated_seeds=activated_seeds,
            paths=paths,
            domain_scores=domain_scores,
            selected_domains=result.selected_domains,
            matched_seeds=len(result.seed_matches),
            total_activated=len(result.activated),
            confidence=verdict["confidence"],
            karma_direction=verdict["karma_direction"],
            decision=verdict["decision"],
        )

    except sqlite3.OperationalError as e:
        log.error(f"数据库操作失败: {e}")
        raise HTTPException(
            status_code=503,
            detail={"error": "service_unavailable", "message": "数据库暂时不可用"},
        )
    except TimeoutError as e:
        log.error(f"请求超时: {e}")
        raise HTTPException(
            status_code=504,
            detail={"error": "timeout", "message": "请求超时"},
        )
    except HTTPException:
        raise
    except Exception as e:
        log.exception(f"查询处理异常: {e}")
        raise HTTPException(
            status_code=500,
            detail={"error": "internal_error", "message": "服务器内部错误，请稍后重试"},
        )


@app.get("/api/v1/stats", response_model=StatsResponse)
def stats_endpoint(graph: GraphDB = Depends(get_graph_db)):
    """返回数据库统计信息"""
    try:
        s = graph.stats()

        # 关系类型分布
        relations: dict[str, int] = {}
        rows = graph.conn.execute(
            "SELECT relation, COUNT(*) as cnt FROM karma_edges "
            "GROUP BY relation ORDER BY cnt DESC"
        ).fetchall()
        for r in rows:
            relations[r["relation"]] = r["cnt"]

        # 领域分布
        domain_distribution: dict[str, int] = {}
        domain_rows = graph.conn.execute(
            "SELECT domain, COUNT(*) as cnt FROM seeds "
            "GROUP BY domain ORDER BY cnt DESC"
        ).fetchall()
        for r in domain_rows:
            domain_distribution[r["domain"] or "未分类"] = r["cnt"]

        # 数据库文件大小
        db_path = Path(DEFAULT_DB_PATH)
        db_size_mb = db_path.stat().st_size / (1024**2) if db_path.exists() else 0.0

        return StatsResponse(
            nodes=s["nodes"],
            edges=s["edges"],
            relations=relations,
            domain_distribution=domain_distribution,
            db_size_mb=round(db_size_mb, 2),
        )

    except sqlite3.OperationalError as e:
        log.error(f"数据库操作失败: {e}")
        raise HTTPException(
            status_code=503,
            detail={"error": "service_unavailable", "message": "数据库暂时不可用"},
        )
    except Exception as e:
        log.exception(f"统计查询异常: {e}")
        raise HTTPException(
            status_code=500,
            detail={"error": "internal_error", "message": "服务器内部错误，请稍后重试"},
        )


@app.get("/api/v1/history", response_model=HistoryResponse)
def history_endpoint(
    limit: int = Query(HISTORY_DEFAULT_LIMIT, ge=1, le=HISTORY_MAX_LIMIT, description="返回条数"),
    offset: int = Query(0, ge=0, description="偏移量"),
    graph: GraphDB = Depends(get_graph_db),
):
    """查询历史记录"""
    # 参数截断：limit 超过最大值时截断
    if limit > HISTORY_MAX_LIMIT:
        limit = HISTORY_MAX_LIMIT

    try:
        result = get_history(graph.conn, limit=limit, offset=offset)

        records = [
            HistoryRecord(
                query_id=r["query_id"],
                query_text=r["query_text"],
                matched_seeds_count=r["matched_seeds_count"],
                selected_domains=r["selected_domains"],
                confidence=r["confidence"],
                karma_direction=r["karma_direction"],
                created_at=r["created_at"],
            )
            for r in result["records"]
        ]

        return HistoryResponse(
            records=records,
            total=result["total"],
            limit=result["limit"],
            offset=result["offset"],
        )

    except sqlite3.OperationalError as e:
        log.error(f"数据库操作失败: {e}")
        raise HTTPException(
            status_code=503,
            detail={"error": "service_unavailable", "message": "数据库暂时不可用"},
        )
    except Exception as e:
        log.exception(f"查询历史异常: {e}")
        raise HTTPException(
            status_code=500,
            detail={"error": "internal_error", "message": "服务器内部错误，请稍后重试"},
        )


# ═══════════════════════════════════════════════════════════
#  启动入口
# ═══════════════════════════════════════════════════════════


def main():
    """启动 uvicorn 服务"""
    uvicorn.run(
        "api:app",
        host=API_HOST,
        port=API_PORT,
        workers=1,
        timeout_keep_alive=API_TIMEOUT,
    )


if __name__ == "__main__":
    main()