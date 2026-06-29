"""
会话管理器 — Session 级上下文隔离与资源生命周期管理

管理每次查询请求的完整上下文，包括：
  - 从连接池获取/归还独立连接
  - Session ID 生成
  - 用户标识绑定
  - 资源清理保证

设计要点:
  - 不引入全局 Session 存储，Session 上下文仅在请求处理期间存在
  - RippleResult 天然隔离（route() 每次调用创建新实例）
  - 使用 try/finally 模式确保连接总是归还
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Optional
from uuid import uuid4

from typing import TYPE_CHECKING

from .connection_pool import ConnectionPool

if TYPE_CHECKING:
    from consciousness_sea.domain.graph_db import GraphDB

log = logging.getLogger(__name__)


@dataclass
class SessionContext:
    """一次查询请求的完整上下文

    Attributes:
        session_id: 唯一会话标识
        graph: 本次请求的独立连接
        user_label: 用户种子 label（可选）
        created_at: 创建时间（ISO 8601）
    """

    session_id: str
    graph: Optional[GraphDB] = None
    user_label: Optional[str] = None
    created_at: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )

    def cleanup(self) -> None:
        """清理 Session 资源（断开引用，允许 GC）

        将 graph 和 user_label 置为 None，断开对连接的引用。
        注意：连接的实际归还由 SessionManager.end_session() 调用 pool.release() 完成。
        """
        self.graph = None
        self.user_label = None


class SessionManager:
    """Session 生命周期管理

    负责创建和结束 Session，管理连接池的连接获取与归还。

    Args:
        pool: 连接池实例
    """

    def __init__(self, pool: ConnectionPool):
        self._pool = pool

    def create_session(self, user_label: Optional[str] = None) -> SessionContext:
        """创建新 Session：从连接池获取连接

        Args:
            user_label: 用户种子 label（可选）

        Returns:
            SessionContext 实例
        """
        graph = self._pool.acquire()
        session_id = uuid4().hex[:12]
        ctx = SessionContext(
            session_id=session_id,
            graph=graph,
            user_label=user_label,
        )
        log.debug("Session 已创建: session_id=%s, user_label=%s", session_id, user_label)
        return ctx

    def end_session(self, ctx: SessionContext) -> None:
        """结束 Session：清理资源 + 归还连接到池中

        先调用 ctx.cleanup() 断开引用，再将连接归还到连接池。
        即使 ctx.graph 已被 cleanup 置为 None，也安全处理。

        Args:
            ctx: 要结束的 SessionContext 实例
        """
        # 保存引用再清理
        graph = ctx.graph
        ctx.cleanup()

        # 归还连接到池中
        if graph is not None:
            self._pool.release(graph)
            log.debug("Session 已结束: session_id=%s", ctx.session_id)