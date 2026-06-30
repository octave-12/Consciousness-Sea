#!/usr/bin/env python3
"""
识海 HTTP API 服务 — FastAPI 实现

端点:
  POST /api/v1/query    — 执行查询
  GET  /api/v1/stats    — 数据库统计
  GET  /api/v1/history  — 查询历史
  GET  /status          — 可观测性监控面板（JSON/HTML 内容协商）
  GET  /health          — 健康检查
"""

from __future__ import annotations

import logging
import os
import sqlite3
from contextlib import asynccontextmanager
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal, Optional

import uvicorn
from fastapi import Depends, FastAPI, Header, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse
from pydantic import BaseModel, Field

from consciousness_sea import (
    GraphDB,
    answer_with_expert,
    apply_karma,
    route,
    verify,
)
from consciousness_sea.domain.query_history import get_history, record_query
from consciousness_sea.infrastructure.audit_logger import record_audit
from consciousness_sea.infrastructure.auth import (
    create_access_token,
    verify_api_key,
)
from consciousness_sea.infrastructure.config import (
    API_HOST,
    API_PORT,
    API_TIMEOUT,
    COGNITIVE_GOAL_ENABLED,
    CORS_ALLOWED_ORIGINS,
    CURIOSITY_ENGINE_ENABLED,
    DEFAULT_DB_PATH,
    DEFAULT_LORA,
    EXPERT_BACKEND,
    EXPERT_INFERENCE_TIMEOUT,
    EXPERT_MAX_VRAM_GB,
    EXPERT_MODEL_PATH,
    EXPERT_RELIABILITY,
    HISTORY_DEFAULT_LIMIT,
    HISTORY_MAX_LIMIT,
    LORA_ADAPTERS,
    MAX_SOURCE_ID_LENGTH,
    META_SEED_ENABLED,
    OLLAMA_BASE_URL,
    OLLAMA_MODEL,
    OLLAMA_TIMEOUT,
    PERCEPTION_ENABLED,
    TOP_K_PATHS,
    VALID_SOURCES,
)
from consciousness_sea.infrastructure.connection_pool import ConnectionPool, ConnectionPoolExhausted
from consciousness_sea.infrastructure.observer import Observer, StatusData
from consciousness_sea.infrastructure.param_stats import (
    record_param_stats as _record_param_stats_core,
)
from consciousness_sea.infrastructure.rate_limiter import (
    IP_RATE_LIMIT,
    IP_RATE_WINDOW,
    rate_limiter,
)
from consciousness_sea.infrastructure.session_manager import SessionManager
from consciousness_sea.infrastructure.user_manager import UserManager

if TYPE_CHECKING:
    from consciousness_sea.expert.expert_manager import ExpertManager
    from consciousness_sea.learning.checkpoint import CheckpointManager
    from consciousness_sea.metacognition.cognitive_goal import CognitiveGoalManager
    from consciousness_sea.metacognition.curiosity_engine import CuriosityEngine
    from consciousness_sea.metacognition.guardian_loop import GuardianLoop
    from consciousness_sea.perception.perception import PerceptionManager

log = logging.getLogger(__name__)

# ═══════════════════════════════════════════════════════════
#  FastAPI 应用初始化
# ═══════════════════════════════════════════════════════════

# ═══════════════════════════════════════════════════════════
#  模块级单例 — 连接池 + 用户管理 + Session + 可观测性
# ═══════════════════════════════════════════════════════════

_pool: Optional[ConnectionPool] = None
_user_manager: Optional[UserManager] = None
_session_manager: Optional[SessionManager] = None
_observer: Optional[Observer] = None
_expert_manager: Optional[ExpertManager] = None
_checkpoint_manager: Optional[CheckpointManager] = None
_checkpoint_graph: Optional[GraphDB] = None  # CheckpointManager 专用独立连接
_guardian_loop: Optional[GuardianLoop] = None
_guardian_graph: Optional[GraphDB] = None  # GuardianLoop 专用独立连接
_goal_mgr: Optional[CognitiveGoalManager] = None
_curiosity_engine: Optional[CuriosityEngine] = None
_perception_manager: Optional[PerceptionManager] = None
_perception_graph: Optional[GraphDB] = None  # PerceptionManager 专用独立连接


@asynccontextmanager
async def lifespan(app: FastAPI):
    """应用生命周期管理：启动时初始化资源，关闭时清理资源"""
    global _pool, _user_manager, _session_manager, _observer, _expert_manager, _checkpoint_manager, _checkpoint_graph, _guardian_loop, _guardian_graph, _goal_mgr, _curiosity_engine, _perception_manager, _perception_graph

    # ── Startup ──
    from consciousness_sea.infrastructure.config import validate_config
    validate_config()
    _pool = ConnectionPool(DEFAULT_DB_PATH)
    _user_manager = UserManager(_pool)
    _session_manager = SessionManager(_pool)
    _observer = Observer(_pool)
    # 懒加载创建 ExpertManager（不阻塞启动，实际加载在首次推理时触发）
    _expert_manager = _create_expert_manager()
    # Phase 3: 创建 CheckpointManager 并启动守护线程
    # 使用独立数据库连接，避免守护线程与连接池中的连接冲突
    try:
        from consciousness_sea.learning.checkpoint import CheckpointManager
        _checkpoint_graph = GraphDB(DEFAULT_DB_PATH)
        _checkpoint_graph.connect()
        _checkpoint_manager = CheckpointManager(_checkpoint_graph)
        _checkpoint_manager.start_daemon()
        log.info("CheckpointManager 已创建并启动守护线程（独立连接）")
    except Exception as e:
        log.warning("CheckpointManager 创建失败: %s", e)
        _checkpoint_manager = None
        if _checkpoint_graph is not None:
            try:
                _checkpoint_graph.close()
            except Exception:
                pass
            _checkpoint_graph = None
    # Phase 4: 创建 GuardianLoop 并启动守护线程
    if META_SEED_ENABLED:
        try:
            from consciousness_sea.metacognition.guardian_loop import GuardianLoop
            _guardian_graph = GraphDB(DEFAULT_DB_PATH)
            _guardian_graph.connect()
            _guardian_loop = GuardianLoop(_guardian_graph)
            _guardian_loop.start()
            log.info("GuardianLoop 已创建并启动守护线程（独立连接）")
        except Exception as e:
            log.warning("GuardianLoop 创建失败: %s", e)
            _guardian_loop = None
            if _guardian_graph is not None:
                try:
                    _guardian_graph.close()
                except Exception:
                    pass
                _guardian_graph = None
    # Phase 5: 创建 CognitiveGoalManager 和 CuriosityEngine
    if COGNITIVE_GOAL_ENABLED and _guardian_graph is not None:
        try:
            from consciousness_sea.metacognition.cognitive_goal import CognitiveGoalManager
            _goal_mgr = CognitiveGoalManager(_guardian_graph)
            log.info("CognitiveGoalManager 已创建（使用 GuardianLoop 独立连接）")
        except Exception as e:
            log.warning("CognitiveGoalManager 创建失败: %s", e)
            _goal_mgr = None
    if CURIOSITY_ENGINE_ENABLED and _goal_mgr is not None and _guardian_graph is not None:
        try:
            from consciousness_sea.metacognition.curiosity_engine import CuriosityEngine
            _curiosity_engine = CuriosityEngine(_guardian_graph, _goal_mgr)
            log.info("CuriosityEngine 已创建（使用 GuardianLoop 独立连接）")
        except Exception as e:
            log.warning("CuriosityEngine 创建失败: %s", e)
            _curiosity_engine = None
    # Phase 6: 创建 PerceptionManager 并启动感知通道
    if PERCEPTION_ENABLED:
        try:
            from consciousness_sea.perception.perception import PerceptionManager
            _perception_graph = GraphDB(DEFAULT_DB_PATH)
            _perception_graph.connect()
            _perception_manager = PerceptionManager(_perception_graph)
            _perception_manager.start()
            log.info("PerceptionManager 已创建并启动（独立连接）")
        except Exception as e:
            log.warning("PerceptionManager 创建失败: %s", e)
            _perception_manager = None
            if _perception_graph is not None:
                try:
                    _perception_graph.close()
                except Exception:
                    pass
                _perception_graph = None
    log.info("连接池和管理器已初始化: pool_size=5")

    yield

    # ── Shutdown ──
    # Phase 6: 停止 PerceptionManager 并关闭独立连接
    if _perception_manager is not None:
        try:
            _perception_manager.stop()
            log.info("PerceptionManager 已停止")
        except Exception as e:
            log.warning("PerceptionManager 停止异常: %s", e)
        _perception_manager = None
    if _perception_graph is not None:
        try:
            _perception_graph.close()
            log.info("PerceptionManager 独立连接已关闭")
        except Exception as e:
            log.warning("PerceptionManager 独立连接关闭异常: %s", e)
        _perception_graph = None
    # Phase 5: 清理好奇心引擎和目标管理器
    _curiosity_engine = None
    _goal_mgr = None
    # Phase 4: 停止 GuardianLoop 守护线程并关闭独立连接
    if _guardian_loop is not None:
        try:
            _guardian_loop.stop()
            log.info("GuardianLoop 守护线程已停止")
        except Exception as e:
            log.warning("GuardianLoop 停止异常: %s", e)
        _guardian_loop = None
    if _guardian_graph is not None:
        try:
            _guardian_graph.close()
            log.info("GuardianLoop 独立连接已关闭")
        except Exception as e:
            log.warning("GuardianLoop 独立连接关闭异常: %s", e)
        _guardian_graph = None
    # Phase 3: 停止 CheckpointManager 守护线程并关闭独立连接
    if _checkpoint_manager is not None:
        try:
            _checkpoint_manager.stop_daemon()
            log.info("CheckpointManager 守护线程已停止")
        except Exception as e:
            log.warning("CheckpointManager 停止异常: %s", e)
        _checkpoint_manager = None
    if _checkpoint_graph is not None:
        try:
            _checkpoint_graph.close()
            log.info("CheckpointManager 独立连接已关闭")
        except Exception as e:
            log.warning("CheckpointManager 独立连接关闭异常: %s", e)
        _checkpoint_graph = None
    # 关闭 ExpertManager（释放 GPU 资源）
    if _expert_manager is not None:
        try:
            _expert_manager.shutdown()
            log.info("ExpertManager 已关闭")
        except Exception as e:
            log.warning("ExpertManager 关闭异常: %s", e)
        _expert_manager = None
    # 关闭连接池
    if _pool is not None:
        _pool.close_all()
        log.info("连接池已关闭")



app = FastAPI(
    title="识海 API",
    version="0.1.0",
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_url="/openapi.json",
    description="识海 — 去中心化多智能体认知架构 API",
)



@app.middleware("http")
async def rate_limit_and_audit_middleware(request, call_next):
    import time as _time
    start = _time.time()

    client_ip = request.client.host if request.client else "unknown"
    if not rate_limiter.check_ip(client_ip):
        return JSONResponse(
            status_code=429,
            content={"error": "rate_limit_exceeded", "message": "IP rate limit exceeded"},
            headers={
                "X-RateLimit-Limit": str(IP_RATE_LIMIT),
                "X-RateLimit-Window": f"{IP_RATE_WINDOW}s",
                "Retry-After": str(IP_RATE_WINDOW),
            },
        )

    response = await call_next(request)
    elapsed_ms = (_time.time() - start) * 1000

    ip_remaining = rate_limiter.get_ip_remaining(client_ip)
    response.headers["X-RateLimit-Limit"] = str(IP_RATE_LIMIT)
    response.headers["X-RateLimit-Remaining"] = str(ip_remaining)

    try:
        record_audit(
            action="api_request",
            method=request.method,
            path=str(request.url.path),
            status_code=response.status_code,
            ip_address=client_ip,
            user_agent=request.headers.get("user-agent"),
            response_time_ms=round(elapsed_ms, 2),
        )
    except Exception:
        pass

    return response


_cors_origins = CORS_ALLOWED_ORIGINS
_has_regex = any("*" in o for o in _cors_origins)
if _has_regex:
    import re
    _regex_parts = [re.escape(o).replace(r"\*", r"\d+") for o in _cors_origins if "*" in o]
    _plain_origins = [o for o in _cors_origins if "*" not in o]
    _combined_regex = "|".join(_regex_parts) if _regex_parts else None
    app.add_middleware(
        CORSMiddleware,
        allow_origin_regex=_combined_regex,
        allow_origins=_plain_origins if _plain_origins else [],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
else:
    app.add_middleware(
        CORSMiddleware,
        allow_origins=_cors_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )


def _create_expert_manager() -> Optional[ExpertManager]:
    """工厂函数：从配置创建 ExpertManager 实例

    读取 config.py 中的专家模型配置，创建 ExpertManager。
    创建失败时返回 None，不抛出异常。

    Returns:
        ExpertManager 实例或 None
    """
    try:
        from pathlib import Path

        from consciousness_sea.expert.expert_manager import ExpertManager
        from consciousness_sea.expert.expert_reliability import ExpertReliabilityStore

        # 解析 LoRA 适配器路径映射
        lora_adapters: dict[str, Path] = {}
        for domain, path_str in LORA_ADAPTERS.items():
            lora_adapters[domain] = Path(path_str)

        # 解析基座模型路径
        model_path = Path(EXPERT_MODEL_PATH) if EXPERT_MODEL_PATH else Path("")

        # 创建可靠性持久化存储
        reliability_store = ExpertReliabilityStore()
        # 使用连接池初始化存储表
        if _pool is not None:
            graph = _pool.acquire()
            try:
                reliability_store.initialize_table(graph.conn)
            finally:
                _pool.release(graph)

        manager = ExpertManager(
            model_path=model_path,
            lora_adapters=lora_adapters if lora_adapters else None,
            reliability=EXPERT_RELIABILITY if EXPERT_RELIABILITY else None,
            default_lora=DEFAULT_LORA,
            max_vram_gb=EXPERT_MAX_VRAM_GB,
            inference_timeout=EXPERT_INFERENCE_TIMEOUT,
            reliability_store=reliability_store,
            ollama_base_url=OLLAMA_BASE_URL,
            ollama_model=OLLAMA_MODEL,
            ollama_timeout=OLLAMA_TIMEOUT,
            expert_backend=EXPERT_BACKEND,
        )
        log.info("ExpertManager 已创建（懒加载模式，基座模型将在首次推理时加载）")
        return manager
    except Exception as e:
        log.warning("ExpertManager 创建失败: %s，将以 Phase 0 模式运行", e)
        return None


# ═══════════════════════════════════════════════════════════
#  Pydantic 请求/响应模型
# ═══════════════════════════════════════════════════════════


class QueryRequest(BaseModel):
    """查询请求"""

    query: str = Field(..., min_length=1, max_length=1000, description="查询文本")
    user: Optional[str] = Field(None, description="用户种子 label（向后兼容）")
    source: Optional[str] = Field(None, description="来源平台（wechat/web/api）")
    source_id: Optional[str] = Field(None, max_length=256, description="来源平台用户标识")
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
    # Phase 1 专家字段（均有默认值，向后兼容）
    expert_answer: Optional[str] = None
    expert_domain: Optional[str] = None
    expert_available: bool = False
    reliability_score: Optional[float] = None
    cross_validation_status: str = "none"
    cross_validation_discount: float = 1.0
    # Phase 3: 自生长字段
    alias_extended: Optional[list[dict]] = None
    cold_start_factor: Optional[float] = None


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
    user_id: Optional[str] = None  # M-8: 新增 user_id 字段


class HistoryResponse(BaseModel):
    """查询历史响应"""

    records: list[HistoryRecord]
    total: int
    limit: int
    offset: int


class CreateCognitiveGoalRequest(BaseModel):
    """手动创建认知目标请求"""

    goal_type: str = Field(..., description="目标类型（low_confidence/low_density/high_conflict/new_term）")
    domain: str = Field(..., min_length=1, max_length=100, description="关联领域")
    trigger_condition: str = Field("manual", max_length=500, description="触发条件")


class RollbackRequest(BaseModel):
    """回滚请求"""

    checkpoint_id: str = Field(..., min_length=1, max_length=100)
    mode: Literal["full", "single"] = Field("full")
    edges: Optional[list[dict]] = Field(None, max_length=1000)


def _error_response(status_code: int, error: str, message: str) -> JSONResponse:
    return JSONResponse(
        status_code=status_code,
        content={"error": error, "message": message},
    )


# ═══════════════════════════════════════════════════════════
#  依赖注入
# ═══════════════════════════════════════════════════════════


def get_pool() -> ConnectionPool:
    """获取连接池实例"""
    if _pool is None:
        raise HTTPException(status_code=503, detail={"error": "service_unavailable", "message": "连接池未初始化"})
    return _pool


def get_session_manager() -> SessionManager:
    """获取 Session 管理器实例"""
    if _session_manager is None:
        raise HTTPException(status_code=503, detail={"error": "service_unavailable", "message": "Session 管理器未初始化"})
    return _session_manager


def get_user_manager() -> UserManager:
    """获取用户管理器实例"""
    if _user_manager is None:
        raise HTTPException(status_code=503, detail={"error": "service_unavailable", "message": "用户管理器未初始化"})
    return _user_manager


def get_observer() -> Observer:
    """获取可观测性实例"""
    if _observer is None:
        raise HTTPException(status_code=503, detail={"error": "service_unavailable", "message": "可观测性模块未初始化"})
    return _observer


# ═══════════════════════════════════════════════════════════
#  用户标识解析与校验
# ═══════════════════════════════════════════════════════════


def _validate_source(source: str, source_id: str) -> None:
    """校验来源标识格式

    Args:
        source: 来源平台
        source_id: 来源平台用户标识

    Raises:
        HTTPException: 422 校验失败
    """
    if source not in VALID_SOURCES:
        raise HTTPException(
            status_code=422,
            detail={"error": "validation_error", "message": f"source 必须为 {VALID_SOURCES} 之一"},
        )
    if not source_id or len(source_id) > MAX_SOURCE_ID_LENGTH:
        raise HTTPException(
            status_code=422,
            detail={"error": "validation_error", "message": "source_id 格式不合法"},
        )


def _resolve_user_label(req: QueryRequest, user_mgr: UserManager) -> Optional[str]:
    """解析用户标识，返回 user_label

    优先级:
      1. source + source_id 均存在 → UserManager.resolve_user()
      2. 仅 user 存在 → 直接使用（向后兼容）
      3. 均不存在 → None

    Args:
        req: 查询请求
        user_mgr: 用户管理器

    Returns:
        user_label 或 None
    """
    if req.source and req.source_id:
        _validate_source(req.source, req.source_id)
        return user_mgr.resolve_user(req.source, req.source_id)
    elif req.user:
        return req.user
    return None


# ═══════════════════════════════════════════════════════════
#  端点实现
# ═══════════════════════════════════════════════════════════


class LoginRequest(BaseModel):
    username: str = Field(..., min_length=1, max_length=100)
    password: str = Field(..., min_length=1, max_length=200)


class LoginResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"
    expires_in: int


class AuditLogQuery(BaseModel):
    user_id: Optional[str] = None
    action: Optional[str] = None
    start_time: Optional[str] = None
    end_time: Optional[str] = None
    limit: int = Field(100, ge=1, le=1000)
    offset: int = Field(0, ge=0)


@app.post("/api/v1/auth/login", response_model=LoginResponse)
def login_endpoint(req: LoginRequest):
    from consciousness_sea.infrastructure.auth import _JWT_EXPIRATION_HOURS, _auth
    if not _auth.enabled:
        token = create_access_token(req.username)
        return LoginResponse(access_token=token, expires_in=_JWT_EXPIRATION_HOURS * 3600)
    if not _auth.validate(req.password):
        raise HTTPException(status_code=401, detail={"error": "unauthorized", "message": "Invalid credentials"})
    token = create_access_token(req.username)
    return LoginResponse(access_token=token, expires_in=_JWT_EXPIRATION_HOURS * 3600)


@app.get("/health")
def health_check():
    db_ok = False
    try:
        if _pool is not None:
            graph = _pool.acquire()
            try:
                graph.stats()
                db_ok = True
            finally:
                _pool.release(graph)
    except Exception:
        pass

    return {
        "status": "ok" if db_ok else "degraded",
        "version": "0.1.0",
        "database": "connected" if db_ok else "unavailable",
        "modules": {
            "expert": _expert_manager is not None,
            "checkpoint": _checkpoint_manager is not None,
            "guardian": _guardian_loop is not None,
            "cognitive_goal": _goal_mgr is not None,
            "curiosity": _curiosity_engine is not None,
            "perception": _perception_manager is not None,
        },
    }


@app.get("/metrics")
def metrics_endpoint():
    from consciousness_sea.infrastructure.rate_limiter import circuit_breaker, rate_limiter
    return {
        "rate_limiter": {
            "ip_entries": len(rate_limiter._ip_entries),
            "user_entries": len(rate_limiter._user_entries),
        },
        "circuit_breaker": {
            "circuits": {
                k: v for k, v in circuit_breaker._circuits.items()
            }
        },
    }


@app.post("/api/v1/audit/query")
def query_audit_logs(body: AuditLogQuery, _auth: None = Depends(verify_api_key)):
    from consciousness_sea.infrastructure.audit_logger import query_audit_log as _query_audit_log
    return {
        "records": _query_audit_log(
            user_id=body.user_id,
            action=body.action,
            start_time=body.start_time,
            end_time=body.end_time,
            limit=body.limit,
            offset=body.offset,
        )
    }



@app.post("/api/v1/query", response_model=QueryResponse)
def query_endpoint(
    req: QueryRequest,
    session_mgr: SessionManager = Depends(get_session_manager),
    user_mgr: UserManager = Depends(get_user_manager),
    _auth: None = Depends(verify_api_key),
):
    """
    执行一次查询。

    流程: resolve_user → create_session → route() → answer() → verify() → apply_karma()
    """
    if not req.query or not req.query.strip():
        raise HTTPException(status_code=400, detail={"error": "bad_request", "message": "query 不能为空"})

    # 解析用户标识
    user_label = _resolve_user_label(req, user_mgr)

    # 从连接池获取独立连接
    try:
        ctx = session_mgr.create_session(user_label=user_label)
    except ConnectionPoolExhausted as e:
        log.error("连接池耗尽: %s", e)
        raise HTTPException(
            status_code=503,
            detail={"error": "service_unavailable", "message": "服务繁忙，请稍后重试"},
        )

    try:
        graph = ctx.graph

        # 1. 路由
        result = route(req.query, graph, user_label=user_label)

        # 2. 回答（使用专家模式）
        expert_result = answer_with_expert(result, graph, _expert_manager)

        # 确定回答文本（专家回答优先，否则检索式回答）
        answer_text = expert_result.get('expert_answer') or expert_result.get('retrieval_answer', '')

        # 3. 校验（传入专家相关参数）
        verdict = verify(
            answer_text, result, graph,
            expert_domain=expert_result.get('expert_domain'),
            reliability=expert_result.get('reliability_score') if expert_result.get('reliability_score') is not None else 1.0,
            cv_discount=expert_result.get('cross_validation_discount') if expert_result.get('cross_validation_discount') is not None else 1.0,
        )

        # 4. 熏习
        karma_count = apply_karma(
            result, graph, verdict["karma_direction"],
            dry_run=req.dry_run, user_label=user_label,
            answer_text=answer_text,
            verify_result=verdict,
        )
        log.info(
            "查询完成: query='%s', confidence=%s, "
            "karma_direction=%+d, "
            "karma_edges=%s, dry_run=%s, "
            "expert_available=%s",
            req.query,
            verdict["confidence"],
            verdict["karma_direction"],
            karma_count,
            req.dry_run,
            expert_result.get("expert_available", False),
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
                user_id=user_label,
                expert_domain=expert_result.get('expert_domain'),
                expert_available=expert_result.get('expert_available', False),
                cross_validation_status=expert_result.get('cross_validation_status', 'none'),
            )
        except Exception as e:
            log.warning("记录查询历史失败: %s", e)

        # 5.5 记录参数统计（Phase 2，失败不影响响应）
        try:
            _record_param_stats(
                graph=graph,
                query_text=req.query,
                result=result,
                verdict=verdict,
            )
        except Exception as e:
            log.warning("记录参数统计失败: %s", e)

        # 5.6 Phase 3: 冷启动计数递增
        try:
            if user_label:
                user_mgr.post_query_increment(user_label)
        except Exception as e:
            log.warning("冷启动计数递增失败: %s", e)

        # 5.7 Phase 3: 获取冷启动因子
        cold_start_factor = None
        try:
            from consciousness_sea.learning.cold_start import ColdStartManager
            cold_factor = ColdStartManager(graph).get_cold_factor(user_label)
            cold_start_factor = cold_factor if user_label else None
        except Exception as e:
            log.warning("获取冷启动因子失败: %s", e)

        # 5.8 Phase 6: 转发概念激活事件到 PerceptionManager
        if PERCEPTION_ENABLED and _perception_manager is not None:
            try:
                concept_event = verdict.get("_concept_activation_event")
                if concept_event is not None:
                    _perception_manager.on_concept_activation(concept_event)
            except Exception as e:
                log.warning("概念激活事件转发失败: %s", e)

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
            expert_answer=expert_result.get('expert_answer'),
            expert_domain=expert_result.get('expert_domain'),
            expert_available=expert_result.get('expert_available', False),
            reliability_score=expert_result.get('reliability_score'),
            cross_validation_status=expert_result.get('cross_validation_status', 'none'),
            cross_validation_discount=expert_result.get('cross_validation_discount', 1.0),
            # Phase 3: 自生长字段
            alias_extended=None,
            cold_start_factor=cold_start_factor,
        )

    except ConnectionPoolExhausted as e:
        log.error("连接池耗尽: %s", e)
        raise HTTPException(
            status_code=503,
            detail={"error": "service_unavailable", "message": "服务繁忙，请稍后重试"},
        )
    except sqlite3.OperationalError as e:
        log.error("数据库操作失败: %s", e)
        raise HTTPException(
            status_code=503,
            detail={"error": "service_unavailable", "message": "数据库暂时不可用"},
        )
    except TimeoutError as e:
        log.error("请求超时: %s", e)
        raise HTTPException(
            status_code=504,
            detail={"error": "timeout", "message": "请求超时"},
        )
    except HTTPException:
        raise
    except Exception as e:
        log.exception("查询处理异常: %s", e)
        raise HTTPException(
            status_code=500,
            detail={"error": "internal_error", "message": "服务器内部错误"},
        )
    finally:
        session_mgr.end_session(ctx)


@app.get("/api/v1/stats", response_model=StatsResponse)
def stats_endpoint(
    pool: ConnectionPool = Depends(get_pool),
    _auth: None = Depends(verify_api_key),
):
    """返回数据库统计信息"""
    graph = None
    try:
        graph = pool.acquire()
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
        log.error("数据库操作失败: %s", e)
        raise HTTPException(
            status_code=503,
            detail={"error": "service_unavailable", "message": "数据库暂时不可用"},
        )
    except Exception as e:
        log.exception("统计查询异常: %s", e)
        raise HTTPException(
            status_code=500,
            detail={"error": "internal_error", "message": "服务器内部错误"},
        )
    finally:
        if graph is not None:
            pool.release(graph)


@app.get("/api/v1/history", response_model=HistoryResponse)
def history_endpoint(
    limit: int = Query(HISTORY_DEFAULT_LIMIT, ge=1, le=HISTORY_MAX_LIMIT, description="返回条数"),
    offset: int = Query(0, ge=0, description="偏移量"),
    pool: ConnectionPool = Depends(get_pool),
    _auth: None = Depends(verify_api_key),
):
    """查询历史记录"""
    # 参数截断：limit 超过最大值时截断
    if limit > HISTORY_MAX_LIMIT:
        limit = HISTORY_MAX_LIMIT

    graph = None
    try:
        graph = pool.acquire()
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
                user_id=r.get("user_id"),  # M-8: 传入 user_id
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
        log.error("数据库操作失败: %s", e)
        raise HTTPException(
            status_code=503,
            detail={"error": "service_unavailable", "message": "数据库暂时不可用"},
        )
    except Exception as e:
        log.exception("查询历史异常: %s", e)
        raise HTTPException(
            status_code=500,
            detail={"error": "internal_error", "message": "服务器内部错误"},
        )
    finally:
        if graph is not None:
            pool.release(graph)


# ═══════════════════════════════════════════════════════════
#  Phase 3: 自生长端点
# ═══════════════════════════════════════════════════════════


@app.get("/api/v1/aliases")
def get_aliases(pool: ConnectionPool = Depends(get_pool), _auth: None = Depends(verify_api_key)):
    """查询别名扩展状态"""
    graph = None
    try:
        graph = pool.acquire()
        from consciousness_sea.learning.alias_expander import AliasExpander
        expander = AliasExpander(graph)
        return expander.get_alias_stats()
    except Exception as e:
        log.exception("别名查询异常: %s", e)
        raise HTTPException(
            status_code=500,
            detail={"error": "internal_error", "message": "服务器内部错误"},
        )
    finally:
        if graph is not None:
            pool.release(graph)


@app.get("/api/v1/candidate-seeds")
def get_candidate_seeds(pool: ConnectionPool = Depends(get_pool), _auth: None = Depends(verify_api_key)):
    """查询候选种子状态"""
    graph = None
    try:
        graph = pool.acquire()
        from consciousness_sea.learning.seed_candidate import SeedCandidateManager
        manager = SeedCandidateManager(graph)
        return manager.get_status()
    except Exception as e:
        log.exception("候选种子查询异常: %s", e)
        raise HTTPException(
            status_code=500,
            detail={"error": "internal_error", "message": "服务器内部错误"},
        )
    finally:
        if graph is not None:
            pool.release(graph)


@app.post("/api/v1/checkpoint")
def create_checkpoint(tag: str = "", pool: ConnectionPool = Depends(get_pool), _auth: None = Depends(verify_api_key)):
    """创建手动检查点"""
    graph = None
    try:
        graph = pool.acquire()
        from consciousness_sea.learning.checkpoint import CheckpointManager, CheckpointSource
        manager = CheckpointManager(graph)
        meta = manager.create_checkpoint(tag=tag, source=CheckpointSource.MANUAL)
        return {
            "checkpoint_id": meta.checkpoint_id,
            "edge_count": meta.edge_count,
            "file_size_bytes": meta.file_size_bytes,
            "created_at": meta.created_at,
        }
    except Exception as e:
        log.exception("创建检查点异常: %s", e)
        raise HTTPException(
            status_code=500,
            detail={"error": "internal_error", "message": "服务器内部错误"},
        )
    finally:
        if graph is not None:
            pool.release(graph)


@app.get("/api/v1/checkpoints")
def list_checkpoints(limit: int = 20, pool: ConnectionPool = Depends(get_pool), _auth: None = Depends(verify_api_key)):
    """查询检查点列表"""
    graph = None
    try:
        graph = pool.acquire()
        from consciousness_sea.learning.checkpoint import CheckpointManager
        manager = CheckpointManager(graph)
        checkpoints = manager.list_checkpoints(limit=limit)
        return [
            {
                "checkpoint_id": cp.checkpoint_id,
                "tag": cp.tag,
                "edge_count": cp.edge_count,
                "created_at": cp.created_at,
                "source": cp.source.value if hasattr(cp.source, 'value') else cp.source,
            }
            for cp in checkpoints
        ]
    except Exception as e:
        log.exception("查询检查点列表异常: %s", e)
        raise HTTPException(
            status_code=500,
            detail={"error": "internal_error", "message": "服务器内部错误"},
        )
    finally:
        if graph is not None:
            pool.release(graph)


@app.post("/api/v1/rollback")
def rollback(req: RollbackRequest, pool: ConnectionPool = Depends(get_pool), _auth: None = Depends(verify_api_key)):
    """执行回滚操作"""
    graph = None
    try:
        graph = pool.acquire()
        from consciousness_sea.learning.checkpoint import CheckpointManager
        manager = CheckpointManager(graph)
        result = manager.rollback(checkpoint_id=req.checkpoint_id, mode=req.mode, edges=req.edges)
        return {
            "status": "success" if result.success else "failed",
            "mode": result.mode,
            "checkpoint_id": result.checkpoint_id,
            "edges_affected": result.edges_affected,
            "error": result.error,
        }
    except Exception as e:
        log.exception("回滚操作异常: %s", e)
        raise HTTPException(
            status_code=500,
            detail={"error": "internal_error", "message": "服务器内部错误"},
        )
    finally:
        if graph is not None:
            pool.release(graph)


@app.get("/api/v1/cold-start/{user_label}")
def get_cold_start_status(user_label: str, pool: ConnectionPool = Depends(get_pool), _auth: None = Depends(verify_api_key)):
    """查询冷启动状态"""
    graph = None
    try:
        graph = pool.acquire()
        from consciousness_sea.infrastructure.config import COLD_START_QUERIES
        from consciousness_sea.learning.cold_start import ColdStartManager
        manager = ColdStartManager(graph)
        state = manager.get_state(user_label)
        return {
            "user_label": state.user_label,
            "query_count": state.query_count,
            "is_cold_start": state.is_cold_start,
            "cold_factor": state.cold_factor,
            "cold_start_queries": COLD_START_QUERIES,
        }
    except Exception as e:
        log.exception("冷启动状态查询异常: %s", e)
        raise HTTPException(
            status_code=500,
            detail={"error": "internal_error", "message": "服务器内部错误"},
        )
    finally:
        if graph is not None:
            pool.release(graph)


# ═══════════════════════════════════════════════════════════
#  Phase 4: 元种子体系端点
# ═══════════════════════════════════════════════════════════


@app.get("/api/v1/meta-seeds")
def list_meta_seeds(
    category: str | None = None,
    limit: int = Query(100, ge=1, le=1000),
    offset: int = Query(0, ge=0),
    pool: ConnectionPool = Depends(get_pool),
    _auth: None = Depends(verify_api_key),
):
    """查询所有元种子

    Query Parameters:
        category: 按类别过滤（domain_monitor / system_monitor / self_boundary /
                  performance_monitor / relation_quality）
    """
    if not META_SEED_ENABLED:
        return {"meta_seeds": []}

    graph = None
    try:
        graph = pool.acquire()
        from consciousness_sea.metacognition.meta_seed import MetaSeedCategory, MetaSeedManager
        mgr = MetaSeedManager(graph)

        cat_enum = None
        if category:
            try:
                cat_enum = MetaSeedCategory(category)
            except ValueError:
                raise HTTPException(
                    status_code=422,
                    detail={"error": "validation_error", "message": f"无效的 category 值: {category}"},
                )

        seeds = mgr.list_meta_seeds(category=cat_enum, limit=limit, offset=offset)
        return {
            "meta_seeds": [
                {
                    "label": ms.label,
                    "category": ms.category.value,
                    "status": ms.status.value,
                    "metrics": ms.metrics,
                    "updated_at": ms.updated_at,
                }
                for ms in seeds
            ]
        }
    except HTTPException:
        raise
    except Exception as e:
        log.exception("元种子列表查询异常: %s", e)
        raise HTTPException(
            status_code=500,
            detail={"error": "internal_error", "message": "服务器内部错误"},
        )
    finally:
        if graph is not None:
            pool.release(graph)


@app.get("/api/v1/meta-seeds/{label}")
def get_meta_seed(label: str, pool: ConnectionPool = Depends(get_pool), _auth: None = Depends(verify_api_key)):
    """查询单个元种子详情"""
    if not META_SEED_ENABLED:
        raise HTTPException(status_code=404, detail={"error": "not_found", "message": f"元种子不存在: {label}"})

    graph = None
    try:
        graph = pool.acquire()
        from consciousness_sea.metacognition.meta_seed import MetaSeedManager
        mgr = MetaSeedManager(graph)

        ms = mgr.get_meta_seed(label)
        if ms is None:
            raise HTTPException(status_code=404, detail={"error": "not_found", "message": f"元种子不存在: {label}"})

        meta_karma_edges = []
        try:
            rows = graph.conn.execute(
                "SELECT source, target, relation, weight FROM karma_edges "
                "WHERE (source = ? OR target = ?) AND source_tag = 'meta_karma'",
                (label, label),
            ).fetchall()
            meta_karma_edges = [dict(r) for r in rows]
        except Exception:
            pass

        return {
            "label": ms.label,
            "category": ms.category.value,
            "status": ms.status.value,
            "metrics": ms.metrics,
            "meta_karma_edges": meta_karma_edges,
            "updated_at": ms.updated_at,
        }
    except HTTPException:
        raise
    except Exception as e:
        log.exception("元种子详情查询异常: %s", e)
        raise HTTPException(
            status_code=500,
            detail={"error": "internal_error", "message": "服务器内部错误"},
        )
    finally:
        if graph is not None:
            pool.release(graph)


@app.get("/api/v1/guardian/status")
def guardian_status(_auth: None = Depends(verify_api_key)):
    """查询守护循环运行状态"""
    if not META_SEED_ENABLED or _guardian_loop is None:
        return {
            "is_running": False,
            "last_execution_time": None,
            "last_execution_result": None,
            "last_execution_duration_ms": None,
            "total_meta_seeds": 0,
            "total_meta_karma_edges": 0,
            "interval_seconds": 60,
            "consecutive_failures": 0,
        }

    try:
        status = _guardian_loop.get_status()
        return {
            "is_running": status.is_running,
            "last_execution_time": status.last_execution_time,
            "last_execution_result": status.last_execution_result,
            "last_execution_duration_ms": status.last_execution_duration_ms,
            "total_meta_seeds": status.total_meta_seeds,
            "total_meta_karma_edges": status.total_meta_karma_edges,
            "interval_seconds": status.interval_seconds,
            "consecutive_failures": status.consecutive_failures,
        }
    except Exception as e:
        log.exception("守护循环状态查询异常: %s", e)
        raise HTTPException(
            status_code=500,
            detail={"error": "internal_error", "message": "服务器内部错误"},
        )


@app.post("/api/v1/guardian/trigger")
def trigger_guardian(_auth: None = Depends(verify_api_key)):
    """手动触发一次守护循环"""
    if not META_SEED_ENABLED or _guardian_loop is None:
        return {
            "status": "disabled",
            "meta_seeds_updated": 0,
            "meta_karma_edges_created": 0,
            "duration_ms": 0,
        }

    if _guardian_loop.is_executing:
        raise HTTPException(
            status_code=409,
            detail={"error": "conflict", "message": "守护循环正在执行中"},
        )

    try:
        result = _guardian_loop.execute_once()
        return {
            "status": "success" if result.success else "failed",
            "meta_seeds_updated": result.meta_seeds_updated,
            "meta_karma_edges_created": result.meta_karma_edges_created,
            "duration_ms": result.duration_ms,
            "error": result.error,
        }
    except Exception as e:
        log.exception("守护循环触发异常: %s", e)
        raise HTTPException(
            status_code=500,
            detail={"error": "internal_error", "message": "服务器内部错误"},
        )


@app.get("/status")
def status_endpoint(
    accept: str = Header(default="application/json"),
    observer: Observer = Depends(get_observer),
):
    """
    可观测性监控面板。

    内容协商:
      - Accept: text/html → 返回 HTML 监控页面
      - 其他 → 返回 JSON 格式状态数据

    异常处理:
      - 数据库不可用返回 503
    """
    try:
        status = observer.get_status()

        # 内容协商：浏览器请求返回 HTML 页面
        if "text/html" in accept:
            html_content = observer.render_html(status)
            return HTMLResponse(content=html_content)

        # 默认返回 JSON
        return _status_to_dict(status)

    except sqlite3.OperationalError as e:
        log.error("/status 数据库操作失败: %s", e)
        raise HTTPException(
            status_code=503,
            detail={"error": "service_unavailable", "message": "数据库暂时不可用"},
        )
    except Exception as e:
        log.exception("/status 查询异常: %s", e)
        raise HTTPException(
            status_code=503,
            detail={"error": "service_unavailable", "message": "监控服务暂时不可用"},
        )


def _status_to_dict(status: StatusData) -> dict[str, Any]:
    """
    将 StatusData 转换为 JSON 可序列化的字典。

    Args:
        status: 系统监控状态数据

    Returns:
        符合 StatusData 结构的字典
    """
    result: dict[str, Any] = {
        "total_seeds": status.total_seeds,
        "total_karma_edges": status.total_karma_edges,
        "hottest_seeds": [
            {"label": item.label, "edge_count": item.edge_count}
            for item in status.hottest_seeds
        ],
        "coldest_seeds": [
            {"label": item.label, "edge_count": item.edge_count}
            for item in status.coldest_seeds
        ],
        "heaviest_karma": [str(item) for item in status.heaviest_karma],
        "recent_queries": [
            {
                "query_text": q.query_text,
                "selected_domains": q.selected_domains,
                "confidence": q.confidence,
            }
            for q in status.recent_queries
        ],
        "alerts": status.alerts,
        "domain_distribution": status.domain_distribution,
        "db_size_mb": status.db_size_mb,
        "distillation_pool": status.distillation_pool,
        # Phase 3: 自生长状态
        "alias_expansion": status.alias_expansion,
        "candidate_seeds": status.candidate_seeds,
        "latest_checkpoint": status.latest_checkpoint,
        # Phase 4: 元种子状态
        "meta_seeds": status.meta_seeds,
        "guardian_loop": status.guardian_loop,
        # Phase 5: 认知目标与好奇心引擎状态
        "cognitive_goals": status.cognitive_goals,
        "curiosity_engine": status.curiosity_engine,
        # Phase 6: 感知状态
        "perception": status.perception,
    }

    # ── Phase 4: 补充守护循环状态 ──
    if META_SEED_ENABLED and _guardian_loop is not None:
        try:
            g_status = _guardian_loop.get_status()
            result["guardian_loop"] = {
                "is_running": g_status.is_running,
                "last_execution_time": g_status.last_execution_time,
                "last_execution_result": g_status.last_execution_result,
                "consecutive_failures": g_status.consecutive_failures,
            }
        except Exception as e:
            log.warning("获取守护循环状态失败: %s", e)

    # ── Phase 5: 补充好奇心引擎状态 ──
    if CURIOSITY_ENGINE_ENABLED and _curiosity_engine is not None:
        try:
            c_status = _curiosity_engine.get_status()
            result["curiosity_engine"] = {
                "total_explorations": c_status.total_explorations,
                "total_new_associations": c_status.total_new_associations,
                "total_external_queries": c_status.total_external_queries,
                "last_exploration_time": c_status.last_exploration_time,
                "is_exploring": c_status.is_exploring,
            }
        except Exception as e:
            log.warning("获取好奇心引擎状态失败: %s", e)

    # ── 专家状态信息 ──
    if _expert_manager is not None:
        try:
            expert_status = _expert_manager.status
            result["expert_status"] = {
                "expert_available": expert_status.expert_available,
                "current_lora": expert_status.current_lora,
                "vram_usage_mb": expert_status.vram_usage_mb,
                "reliability_scores": expert_status.reliability_scores,
                "lora_switch_count": expert_status.lora_switch_count,
                "inference_count": expert_status.inference_count,
                "fallback_count": expert_status.fallback_count,
                "unavailable_reason": expert_status.unavailable_reason,
                "active_backend": expert_status.active_backend,
            }
        except Exception as e:
            log.warning("获取专家状态失败: %s", e)
            result["expert_status"] = {
                "expert_available": False,
                "unavailable_reason": f"status_error: {e}",
            }
    else:
        result["expert_status"] = {
            "expert_available": False,
            "unavailable_reason": "not_initialized",
        }

    return result


def _record_param_stats(
    graph: GraphDB,
    query_text: str,
    result: Any,
    verdict: dict,
) -> None:
    """记录参数统计到 param_stats 表（Phase 2）

    委托给 consciousness_sea.infrastructure.param_stats.record_param_stats 实现。
    记录失败不影响查询结果返回。

    Args:
        graph: 知识图谱连接
        query_text: 查询文本
        result: 涟漪传播结果
        verdict: 校验结果
    """
    _record_param_stats_core(graph, query_text, result, verdict)


# ═══════════════════════════════════════════════════════════
#  Phase 5: 认知目标与好奇心引擎端点
# ═══════════════════════════════════════════════════════════


@app.get("/api/v1/cognitive-goals")
def list_cognitive_goals(
    status: str | None = None,
    goal_type: str | None = None,
    pool: ConnectionPool = Depends(get_pool),
    _auth: None = Depends(verify_api_key),
):
    """查询所有认知目标

    Query Parameters:
        status: 按状态过滤（pending/exploring/querying_external/completed/archived/expired）
        goal_type: 按类型过滤（low_confidence/low_density/high_conflict/new_term）
    """
    if not COGNITIVE_GOAL_ENABLED:
        return {"goals": []}

    graph = None
    try:
        graph = pool.acquire()
        from consciousness_sea.metacognition.cognitive_goal import CognitiveGoalManager, GoalStatus
        from consciousness_sea.metacognition.cognitive_goal import GoalType as GT

        mgr = CognitiveGoalManager(graph)

        status_enum = None
        if status:
            try:
                status_enum = GoalStatus(status)
            except ValueError:
                raise HTTPException(
                    status_code=422,
                    detail={"error": "validation_error", "message": f"无效的 status 值: {status}"},
                )

        type_enum = None
        if goal_type:
            try:
                type_enum = GT(goal_type)
            except ValueError:
                raise HTTPException(
                    status_code=422,
                    detail={"error": "validation_error", "message": f"无效的 goal_type 值: {goal_type}"},
                )

        goals = mgr.list_goals(status=status_enum, goal_type=type_enum)
        return {
            "goals": [
                {
                    "goal_id": g.goal_id,
                    "goal_type": g.goal_type.value,
                    "domain": g.domain,
                    "priority_weight": g.priority_weight,
                    "status": g.status.value,
                    "created_at": g.created_at,
                    "updated_at": g.updated_at,
                }
                for g in goals
            ]
        }
    except HTTPException:
        raise
    except Exception as e:
        log.exception("认知目标列表查询异常: %s", e)
        raise HTTPException(
            status_code=500,
            detail={"error": "internal_error", "message": "服务器内部错误"},
        )
    finally:
        if graph is not None:
            pool.release(graph)


@app.get("/api/v1/cognitive-goals/stats")
def cognitive_goals_stats(pool: ConnectionPool = Depends(get_pool), _auth: None = Depends(verify_api_key)):
    """查询认知目标统计信息"""
    if not COGNITIVE_GOAL_ENABLED:
        return {
            "by_status": {},
            "by_type": {},
            "avg_priority_weight": 0.0,
            "pool_usage": {"active": 0, "max": 1000, "usage_percent": 0.0},
        }

    graph = None
    try:
        graph = pool.acquire()
        from consciousness_sea.metacognition.cognitive_goal import CognitiveGoalManager
        mgr = CognitiveGoalManager(graph)
        return mgr.get_goal_stats()
    except Exception as e:
        log.exception("认知目标统计查询异常: %s", e)
        raise HTTPException(
            status_code=500,
            detail={"error": "internal_error", "message": "服务器内部错误"},
        )
    finally:
        if graph is not None:
            pool.release(graph)


@app.get("/api/v1/cognitive-goals/{goal_id}")
def get_cognitive_goal(goal_id: str, pool: ConnectionPool = Depends(get_pool), _auth: None = Depends(verify_api_key)):
    """查询单个认知目标详情"""
    if not COGNITIVE_GOAL_ENABLED:
        raise HTTPException(status_code=404, detail={"error": "not_found", "message": f"认知目标不存在: {goal_id}"})

    graph = None
    try:
        graph = pool.acquire()
        from consciousness_sea.metacognition.cognitive_goal import CognitiveGoalManager
        mgr = CognitiveGoalManager(graph)

        goal = mgr.get_goal(goal_id)
        if goal is None:
            raise HTTPException(status_code=404, detail={"error": "not_found", "message": f"认知目标不存在: {goal_id}"})

        return {
            "goal_id": goal.goal_id,
            "goal_type": goal.goal_type.value,
            "trigger_condition": goal.trigger_condition,
            "domain": goal.domain,
            "priority_weight": goal.priority_weight,
            "status": goal.status.value,
            "sub_goals": goal.sub_goals,
            "execution_log": goal.execution_log,
            "associated_user": goal.associated_user,
            "decay_cycles_count": goal.decay_cycles_count,
            "last_touched_at": goal.last_touched_at,
            "created_at": goal.created_at,
            "updated_at": goal.updated_at,
        }
    except HTTPException:
        raise
    except Exception as e:
        log.exception("认知目标详情查询异常: %s", e)
        raise HTTPException(
            status_code=500,
            detail={"error": "internal_error", "message": "服务器内部错误"},
        )
    finally:
        if graph is not None:
            pool.release(graph)


@app.post("/api/v1/cognitive-goals")
def create_cognitive_goal(body: CreateCognitiveGoalRequest, pool: ConnectionPool = Depends(get_pool), _auth: None = Depends(verify_api_key)):
    """手动创建认知目标

    Request Body:
        {
            "goal_type": "low_confidence",
            "domain": "量子力学",
            "trigger_condition": "manual"
        }
    """
    if not COGNITIVE_GOAL_ENABLED:
        raise HTTPException(status_code=503, detail={"error": "service_unavailable", "message": "认知目标功能已禁用"})

    from consciousness_sea.metacognition.cognitive_goal import GoalType

    goal_type_str = body.goal_type
    domain = body.domain
    trigger_condition = body.trigger_condition

    try:
        goal_type = GoalType(goal_type_str)
    except ValueError:
        raise HTTPException(
            status_code=400,
            detail={"error": "validation_error", "message": f"无效的 goal_type 值: {goal_type_str}，合法值: {[t.value for t in GoalType]}"},
        )

    graph = None
    try:
        graph = pool.acquire()
        from consciousness_sea.metacognition.cognitive_goal import CognitiveGoalManager
        mgr = CognitiveGoalManager(graph)

        success = mgr._create_or_update_goal(
            goal_type=goal_type,
            domain=domain,
            trigger_condition=trigger_condition,
        )

        if not success:
            raise HTTPException(status_code=409, detail={"error": "conflict", "message": "目标创建失败（可能池已满）"})

        # 查找刚创建的目标
        goals = mgr.list_goals()
        for g in goals:
            if g.domain == domain and g.goal_type == goal_type and g.status.value == "pending":
                return {"goal_id": g.goal_id}

        return {"goal_id": None, "message": "目标已创建或更新"}
    except HTTPException:
        raise
    except Exception as e:
        log.exception("认知目标创建异常: %s", e)
        raise HTTPException(
            status_code=500,
            detail={"error": "internal_error", "message": "服务器内部错误"},
        )
    finally:
        if graph is not None:
            pool.release(graph)


@app.get("/api/v1/curiosity/status")
def curiosity_status(_auth: None = Depends(verify_api_key)):
    """查询好奇心引擎运行状态"""
    if not CURIOSITY_ENGINE_ENABLED or _curiosity_engine is None:
        return {
            "total_explorations": 0,
            "total_new_associations": 0,
            "total_external_queries": 0,
            "last_exploration_time": None,
            "last_exploration_result": None,
            "is_exploring": False,
        }

    try:
        status = _curiosity_engine.get_status()
        return {
            "total_explorations": status.total_explorations,
            "total_new_associations": status.total_new_associations,
            "total_external_queries": status.total_external_queries,
            "last_exploration_time": status.last_exploration_time,
            "last_exploration_result": status.last_exploration_result,
            "is_exploring": status.is_exploring,
        }
    except Exception as e:
        log.exception("好奇心引擎状态查询异常: %s", e)
        raise HTTPException(
            status_code=500,
            detail={"error": "internal_error", "message": "服务器内部错误"},
        )


@app.post("/api/v1/curiosity/explore/{goal_id}")
def trigger_curiosity_explore(goal_id: str, pool: ConnectionPool = Depends(get_pool), _auth: None = Depends(verify_api_key)):
    """手动触发好奇心引擎对指定目标执行探索"""
    if not CURIOSITY_ENGINE_ENABLED or _curiosity_engine is None:
        raise HTTPException(status_code=503, detail={"error": "service_unavailable", "message": "好奇心引擎未启用"})

    try:
        status = _curiosity_engine.get_status()
        if status.is_exploring:
            raise HTTPException(status_code=409, detail={"error": "conflict", "message": "好奇心引擎正在执行探索"})
    except HTTPException:
        raise
    except Exception:
        pass

    graph_to_release = None
    try:
        from consciousness_sea.metacognition.cognitive_goal import CognitiveGoalManager
        if _guardian_graph is not None:
            goal_mgr = CognitiveGoalManager(_guardian_graph)
        else:
            graph_to_release = pool.acquire()
            goal_mgr = CognitiveGoalManager(graph_to_release)

        goal = goal_mgr.get_goal(goal_id)
        if goal is None:
            raise HTTPException(status_code=404, detail={"error": "not_found", "message": f"认知目标不存在: {goal_id}"})

        # 执行探索
        result = _curiosity_engine.explore(goal)

        return {
            "status": "success" if not result.error else "failed",
            "strategy": result.strategy,
            "new_associations": result.new_associations,
            "distillation_candidates": result.distillation_candidates,
            "duration_ms": result.duration_ms,
            "error": result.error,
        }
    except HTTPException:
        raise
    except Exception as e:
        log.exception("好奇心探索异常: %s", e)
        raise HTTPException(
            status_code=500,
            detail={"error": "internal_error", "message": "服务器内部错误"},
        )
    finally:
        if graph_to_release is not None:
            pool.release(graph_to_release)


# ═══════════════════════════════════════════════════════════
#  Phase 6: 感知通道端点
# ═══════════════════════════════════════════════════════════


@app.get("/api/v1/perception/status")
def perception_status(_auth: None = Depends(verify_api_key)):
    """查询感知系统整体状态"""
    if not PERCEPTION_ENABLED or _perception_manager is None:
        return {
            "enabled": False,
            "channels": {},
            "total_perceptual_seeds": 0,
            "total_hebbian_bindings": 0,
            "recent_activation_count": 0,
            "last_multimodal_alignment": None,
        }

    try:
        status = _perception_manager.get_status()
        return {
            "enabled": status.enabled,
            "channels": {
                ch: {
                    "running": cs.running,
                    "last_activation": cs.last_activation,
                    "mock_mode": cs.mock_mode,
                    "consecutive_failures": cs.consecutive_failures,
                }
                for ch, cs in status.channels.items()
            },
            "total_perceptual_seeds": status.total_perceptual_seeds,
            "total_hebbian_bindings": status.total_hebbian_bindings,
            "recent_activation_count": status.recent_activation_count,
            "last_multimodal_alignment": status.last_multimodal_alignment,
        }
    except Exception as e:
        log.exception("感知状态查询异常: %s", e)
        raise HTTPException(
            status_code=500,
            detail={"error": "internal_error", "message": "服务器内部错误"},
        )


@app.get("/api/v1/perception/seeds")
def perception_seeds(channel: str | None = None, _auth: None = Depends(verify_api_key)):
    """查询所有感知元种子

    Query Parameters:
        channel: 按通道过滤（visual / auditory / somatic）
    """
    if not PERCEPTION_ENABLED or _perception_manager is None:
        return {"seeds": []}

    # 验证 channel 参数
    valid_channels = {"visual", "auditory", "somatic"}
    if channel is not None and channel not in valid_channels:
        raise HTTPException(
            status_code=422,
            detail={"error": "validation_error", "message": f"无效的 channel 值: {channel}，合法值: {sorted(valid_channels)}"},
        )

    try:
        seeds = _perception_manager.list_perceptual_seeds(channel=channel)
        return {"seeds": seeds}
    except Exception as e:
        log.exception("感知种子查询异常: %s", e)
        raise HTTPException(
            status_code=500,
            detail={"error": "internal_error", "message": "服务器内部错误"},
        )


@app.get("/api/v1/perception/seeds/{label}")
def perception_seed_detail(label: str, _auth: None = Depends(verify_api_key)):
    """查询单个感知元种子详情"""
    if not PERCEPTION_ENABLED or _perception_manager is None:
        raise HTTPException(status_code=404, detail={"error": "not_found", "message": f"感知元种子不存在: {label}"})

    try:
        seed = _perception_manager.get_perceptual_seed(label)
        if seed is None:
            raise HTTPException(status_code=404, detail={"error": "not_found", "message": f"感知元种子不存在: {label}"})
        return seed
    except HTTPException:
        raise
    except Exception as e:
        log.exception("感知种子详情查询异常: %s", e)
        raise HTTPException(
            status_code=500,
            detail={"error": "internal_error", "message": "服务器内部错误"},
        )


@app.get("/api/v1/perception/bindings")
def perception_bindings(channel: str | None = None, _auth: None = Depends(verify_api_key)):
    """查询所有 Hebbian 绑定边

    Query Parameters:
        channel: 按通道过滤（visual / auditory / somatic）
    """
    if not PERCEPTION_ENABLED or _perception_manager is None:
        return {"bindings": []}

    # 验证 channel 参数
    valid_channels = {"visual", "auditory", "somatic"}
    if channel is not None and channel not in valid_channels:
        raise HTTPException(
            status_code=422,
            detail={"error": "validation_error", "message": f"无效的 channel 值: {channel}，合法值: {sorted(valid_channels)}"},
        )

    try:
        bindings = _perception_manager.list_hebbian_bindings(channel=channel)
        return {"bindings": bindings}
    except Exception as e:
        log.exception("Hebbian绑定查询异常: %s", e)
        raise HTTPException(
            status_code=500,
            detail={"error": "internal_error", "message": "服务器内部错误"},
        )


@app.get("/api/v1/perception/events")
def perception_events(limit: int = Query(20, ge=1, le=1000), _auth: None = Depends(verify_api_key)):
    """查询最近的感知激活事件"""
    if not PERCEPTION_ENABLED or _perception_manager is None:
        return {"events": []}

    try:
        events = _perception_manager.list_perception_events(limit=limit)
        return {"events": events}
    except Exception as e:
        log.exception("感知事件查询异常: %s", e)
        raise HTTPException(
            status_code=500,
            detail={"error": "internal_error", "message": "服务器内部错误"},
        )


@app.post("/api/v1/perception/align")
def perception_align(_auth: None = Depends(verify_api_key)):
    """手动触发一次多模态对齐"""
    if not PERCEPTION_ENABLED or _perception_manager is None:
        raise HTTPException(status_code=503, detail={"error": "service_unavailable", "message": "感知功能已禁用"})

    try:
        aligner = _perception_manager.multimodal_aligner
        if aligner is not None and aligner.is_running:
            raise HTTPException(status_code=409, detail={"error": "conflict", "message": "多模态对齐正在运行中"})

        results = _perception_manager.run_multimodal_alignment()
        return {"results": results, "count": len(results)}
    except HTTPException:
        raise
    except Exception as e:
        log.exception("多模态对齐异常: %s", e)
        raise HTTPException(
            status_code=500,
            detail={"error": "internal_error", "message": "服务器内部错误"},
        )


def main():
    ssl_keyfile = os.environ.get("SSL_KEYFILE")
    ssl_certfile = os.environ.get("SSL_CERTFILE")

    ssl_kwargs: dict = {}
    if ssl_certfile and ssl_keyfile:
        ssl_kwargs["ssl_keyfile"] = ssl_keyfile
        ssl_kwargs["ssl_certfile"] = ssl_certfile
        log.info("HTTPS 已启用: cert=%s", ssl_certfile)

    uvicorn.run(
        "consciousness_sea.interfaces.api:app",
        host=API_HOST,
        port=API_PORT,
        workers=1,
        timeout_keep_alive=API_TIMEOUT,
        **ssl_kwargs,
    )


if __name__ == "__main__":
    main()
