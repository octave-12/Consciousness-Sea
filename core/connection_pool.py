"""
SQLite 连接池 — 线程安全连接管理

为识海系统提供 SQLite 连接复用和生命周期管理。
基于 threading.Lock + queue.Queue 实现线程安全的连接池。

设计要点:
  - 每个请求线程从池中获取独立连接
  - SQLite WAL 模式 + check_same_thread=False 确保多连接并发读写安全
  - 归还的连接重置缓存状态，确保下次使用时重新加载最新数据
  - 超时保护：acquire() 超时抛出 ConnectionPoolExhausted 异常
"""

from __future__ import annotations

import logging
import queue
import threading
from typing import Optional

from .config import (
    CONNECTION_POOL_SIZE,
    CONNECTION_POOL_TIMEOUT,
    BUSY_TIMEOUT_MS,
    DEFAULT_DB_PATH,
)
from .graph_db import GraphDB

log = logging.getLogger(__name__)


class ConnectionPoolExhausted(Exception):
    """连接池耗尽异常 — 所有连接都在使用中且等待超时"""

    pass


class ConnectionPool:
    """SQLite 连接池 — 线程安全连接管理

    使用 queue.Queue 管理空闲连接，threading.Lock 保护使用中集合。
    连接创建时设置 WAL 模式、busy_timeout、synchronous=NORMAL。
    归还连接时调用 invalidate_cache() 重置缓存。

    Args:
        db_path: 数据库文件路径
        pool_size: 最大连接数
    """

    def __init__(self, db_path: str = DEFAULT_DB_PATH, pool_size: int = CONNECTION_POOL_SIZE):
        self._db_path = db_path
        self._pool_size = pool_size
        self._idle: queue.Queue[GraphDB] = queue.Queue()
        self._in_use: set[int] = set()  # 存储 id(graph) 用于追踪
        self._lock = threading.Lock()
        self._created_count = 0  # 已创建的连接总数

    def _create_connection(self) -> GraphDB:
        """创建新的 GraphDB 连接并设置 PRAGMA"""
        graph = GraphDB(self._db_path)
        graph.connect()
        # 设置 busy_timeout（连接池场景下必须设置）
        graph.conn.execute(f"PRAGMA busy_timeout={BUSY_TIMEOUT_MS}")
        log.debug("连接池创建新连接: db_path=%s", self._db_path)
        return graph

    def acquire(self, timeout: float = CONNECTION_POOL_TIMEOUT) -> GraphDB:
        """获取一个可用连接（线程安全）

        流程:
          1. 优先从空闲队列取连接
          2. 空队列则新建（未超上限）
          3. 超上限则阻塞等待（超时抛 ConnectionPoolExhausted）

        Args:
            timeout: 等待超时时间（秒）

        Returns:
            GraphDB 连接实例

        Raises:
            ConnectionPoolExhausted: 所有连接都在使用中且等待超时
        """
        # 1. 尝试从空闲队列取
        try:
            graph = self._idle.get_nowait()
            with self._lock:
                self._in_use.add(id(graph))
            return graph
        except queue.Empty:
            pass

        # 2. 尝试新建连接（未超上限）
        with self._lock:
            if self._created_count < self._pool_size:
                self._created_count += 1
                graph = self._create_connection()
                self._in_use.add(id(graph))
                return graph

        # 3. 阻塞等待归还
        try:
            graph = self._idle.get(timeout=timeout)
            with self._lock:
                self._in_use.add(id(graph))
            return graph
        except queue.Empty:
            raise ConnectionPoolExhausted(
                f"连接池耗尽: pool_size={self._pool_size}, "
                f"in_use={len(self._in_use)}, timeout={timeout}s"
            )

    def release(self, graph: GraphDB) -> None:
        """归还连接到池中

        归还前重置缓存状态，确保下次使用时重新加载最新数据。

        Args:
            graph: 要归还的 GraphDB 连接实例
        """
        if graph is None:
            return

        with self._lock:
            self._in_use.discard(id(graph))

        # 重置缓存，确保数据新鲜度
        graph.invalidate_cache()
        self._idle.put(graph)
        log.debug("连接已归还到池中")

    def close_all(self) -> None:
        """关闭所有连接（应用关闭时调用）

        关闭空闲队列中的所有连接，并清空使用中集合。
        使用中的连接由于已被请求线程持有，无法强制关闭。
        """
        with self._lock:
            # 关闭所有空闲连接
            while not self._idle.empty():
                try:
                    graph = self._idle.get_nowait()
                    graph.close()
                except queue.Empty:
                    break
            self._in_use.clear()
            self._created_count = 0
        log.info("连接池已关闭所有连接")