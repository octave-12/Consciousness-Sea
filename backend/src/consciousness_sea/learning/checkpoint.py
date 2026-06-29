"""
CheckpointManager — 业力检查点与回滚管理器

负责创建、管理和回滚业力检查点。检查点以 JSON 文件形式存储全局业力边快照，
元数据记录在 checkpoint_meta 表中。支持全量回滚和单边回滚，并提供守护线程
实现定时自动检查点。

设计原则：
  - 原子写入：临时文件 + os.replace() 确保检查点文件不会损坏
  - 回滚前自动创建 pre_rollback 检查点，防止误操作
  - 回滚仅影响 karma_edges 表，不影响 karma_edges_personal 表
  - 线程安全：所有检查点操作在 _checkpoint_lock 保护下执行
"""

from __future__ import annotations

import json
import logging
import os
import threading
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Optional

from consciousness_sea.infrastructure.config import CHECKPOINT_CRON_HOUR, CHECKPOINT_RETAIN_COUNT, CHECKPOINT_DIR
from consciousness_sea.domain.graph_db import GraphDB

log = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════
#  数据类与枚举
# ═══════════════════════════════════════════════════════════


class CheckpointSource(str, Enum):
    """检查点来源类型"""

    AUTO_CRON = "auto_cron"
    MANUAL = "manual"
    PRE_ROLLBACK = "pre_rollback"


@dataclass
class CheckpointMeta:
    """检查点元数据（对应 checkpoint_meta 表记录）"""

    checkpoint_id: str
    tag: str
    edge_count: int
    file_path: str
    file_size_bytes: int
    created_at: str
    source: CheckpointSource


@dataclass
class CheckpointData:
    """检查点文件内容结构"""

    version: int = 1
    checkpoint_id: str = ""
    created_at: str = ""
    tag: str = ""
    source: str = ""
    edge_count: int = 0
    edges: list[dict] | None = None  # [{source, target, relation, weight, source_tag}, ...]


@dataclass
class RollbackResult:
    """回滚操作结果"""

    mode: str  # "full" / "single"
    checkpoint_id: str
    edges_affected: int
    success: bool
    error: str | None = None


# ═══════════════════════════════════════════════════════════
#  CheckpointManager
# ═══════════════════════════════════════════════════════════


class CheckpointManager:
    """业力检查点与回滚管理器

    功能：
      1. 创建业力检查点（JSON 格式，原子写入）
      2. 全量回滚和单边回滚
      3. 自动定时检查点（守护线程）
      4. 检查点保留策略

    线程安全：所有检查点操作在 _checkpoint_lock 保护下执行。
    """

    def __init__(self, graph: GraphDB, checkpoint_dir: str | None = None) -> None:
        self._graph = graph
        self._checkpoint_dir = Path(checkpoint_dir or CHECKPOINT_DIR)
        self._checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self._checkpoint_lock = threading.Lock()
        self._shutdown_event = threading.Event()
        self._daemon_thread: threading.Thread | None = None
        self._last_auto_checkpoint_date: str | None = None

    # ───────────────────────────────────────────────────────
    #  创建检查点
    # ───────────────────────────────────────────────────────

    def create_checkpoint(
        self,
        tag: str = "",
        source: CheckpointSource = CheckpointSource.MANUAL,
    ) -> CheckpointMeta:
        """创建业力检查点

        读取所有全局业力边，写入 JSON 文件（原子写入），并记录元数据。

        Args:
            tag: 检查点标签（可选描述）
            source: 检查点来源

        Returns:
            CheckpointMeta: 创建的检查点元数据
        """
        with self._checkpoint_lock:
            return self._create_checkpoint_locked(tag, source)

    def _create_checkpoint_locked(
        self, tag: str, source: CheckpointSource
    ) -> CheckpointMeta:
        """在锁内执行检查点创建

        流程：
          1. 生成 checkpoint_id（格式 cp_{YYYYMMDD_HHmmss}）
          2. 读取所有全局业力边
          3. 构造 CheckpointData
          4. 原子写入 JSON 文件（临时文件 + os.replace）
          5. 记录 checkpoint_meta 表
          6. 清理旧检查点
        """
        now = datetime.now(timezone.utc)
        checkpoint_id = f"cp_{now.strftime('%Y%m%d_%H%M%S')}"
        created_at = now.isoformat()

        # 1. 读取所有全局业力边
        rows = self._graph.conn.execute(
            "SELECT source, target, relation, weight, source_tag FROM karma_edges"
        ).fetchall()
        edges = [dict(r) for r in rows]

        # 2. 构造 CheckpointData
        checkpoint_data = CheckpointData(
            version=1,
            checkpoint_id=checkpoint_id,
            created_at=created_at,
            tag=tag,
            source=source.value,
            edge_count=len(edges),
            edges=edges,
        )

        # 3. 原子写入 JSON 文件
        data_dict = {
            "version": checkpoint_data.version,
            "checkpoint_id": checkpoint_data.checkpoint_id,
            "created_at": checkpoint_data.created_at,
            "tag": checkpoint_data.tag,
            "source": checkpoint_data.source,
            "edge_count": checkpoint_data.edge_count,
            "edges": checkpoint_data.edges,
        }

        tmp_path = self._checkpoint_dir / f"checkpoint_tmp_{checkpoint_id}.json"
        final_path = self._checkpoint_dir / f"checkpoint_{checkpoint_id}.json"

        with open(tmp_path, "w", encoding="utf-8") as f:
            json.dump(data_dict, f, ensure_ascii=False, indent=2)

        os.replace(str(tmp_path), str(final_path))

        # 4. 获取文件大小
        file_size_bytes = final_path.stat().st_size

        # 5. 记录 checkpoint_meta 表
        self._graph.conn.execute(
            "INSERT INTO checkpoint_meta "
            "(checkpoint_id, tag, edge_count, file_path, file_size_bytes, created_at, source) "
            "VALUES (?, ?, ?, ?, ?, ?, ?)",
            (
                checkpoint_id,
                tag,
                len(edges),
                str(final_path),
                file_size_bytes,
                created_at,
                source.value,
            ),
        )
        self._graph.conn.commit()

        log.info(
            "checkpoint created: id=%s, edges=%d, source=%s, tag=%s",
            checkpoint_id,
            len(edges),
            source.value,
            tag,
        )

        # 6. 清理旧检查点
        cleaned = self._cleanup_old_checkpoints()
        if cleaned > 0:
            log.info("cleaned up %d old checkpoint(s)", cleaned)

        return CheckpointMeta(
            checkpoint_id=checkpoint_id,
            tag=tag,
            edge_count=len(edges),
            file_path=str(final_path),
            file_size_bytes=file_size_bytes,
            created_at=created_at,
            source=source,
        )

    # ───────────────────────────────────────────────────────
    #  保留策略
    # ───────────────────────────────────────────────────────

    def _cleanup_old_checkpoints(self) -> int:
        """保留最近 CHECKPOINT_RETAIN_COUNT 个检查点

        超出时删除最旧的检查点（文件 + 元数据）。

        Returns:
            删除的检查点数量
        """
        rows = self._graph.conn.execute(
            "SELECT checkpoint_id, file_path FROM checkpoint_meta "
            "ORDER BY created_at ASC"
        ).fetchall()

        total = len(rows)
        if total <= CHECKPOINT_RETAIN_COUNT:
            return 0

        delete_count = total - CHECKPOINT_RETAIN_COUNT
        to_delete = rows[:delete_count]

        for row in to_delete:
            checkpoint_id = row["checkpoint_id"]
            file_path = row["file_path"]

            # 删除文件
            try:
                path = Path(file_path)
                if path.exists():
                    path.unlink()
                    log.debug("deleted checkpoint file: %s", file_path)
            except OSError as e:
                log.warning(
                    "failed to delete checkpoint file %s: %s", file_path, e
                )

            # 删除元数据
            self._graph.conn.execute(
                "DELETE FROM checkpoint_meta WHERE checkpoint_id = ?",
                (checkpoint_id,),
            )

        self._graph.conn.commit()
        return delete_count

    # ───────────────────────────────────────────────────────
    #  查询检查点
    # ───────────────────────────────────────────────────────

    def _load_checkpoint_meta(self, checkpoint_id: str) -> CheckpointMeta | None:
        """从数据库加载检查点元数据

        Args:
            checkpoint_id: 检查点 ID

        Returns:
            CheckpointMeta 或 None（不存在时）
        """
        row = self._graph.conn.execute(
            "SELECT checkpoint_id, tag, edge_count, file_path, "
            "file_size_bytes, created_at, source "
            "FROM checkpoint_meta WHERE checkpoint_id = ?",
            (checkpoint_id,),
        ).fetchone()

        if not row:
            return None

        return CheckpointMeta(
            checkpoint_id=row["checkpoint_id"],
            tag=row["tag"],
            edge_count=row["edge_count"],
            file_path=row["file_path"],
            file_size_bytes=row["file_size_bytes"],
            created_at=row["created_at"],
            source=CheckpointSource(row["source"]),
        )

    def list_checkpoints(self, limit: int = 20) -> list[CheckpointMeta]:
        """查询检查点列表（按创建时间降序）

        Args:
            limit: 最大返回数量

        Returns:
            CheckpointMeta 列表
        """
        rows = self._graph.conn.execute(
            "SELECT checkpoint_id, tag, edge_count, file_path, "
            "file_size_bytes, created_at, source "
            "FROM checkpoint_meta "
            "ORDER BY created_at DESC "
            "LIMIT ?",
            (limit,),
        ).fetchall()

        result: list[CheckpointMeta] = []
        for row in rows:
            result.append(
                CheckpointMeta(
                    checkpoint_id=row["checkpoint_id"],
                    tag=row["tag"],
                    edge_count=row["edge_count"],
                    file_path=row["file_path"],
                    file_size_bytes=row["file_size_bytes"],
                    created_at=row["created_at"],
                    source=CheckpointSource(row["source"]),
                )
            )
        return result

    # ───────────────────────────────────────────────────────
    #  回滚
    # ───────────────────────────────────────────────────────

    def rollback(
        self,
        checkpoint_id: str,
        mode: str = "full",
        edges: list[dict] | None = None,
    ) -> RollbackResult:
        """执行回滚操作

        回滚前自动创建 pre_rollback 检查点（失败时仅 WARNING，继续回滚）。
        回滚仅修改 karma_edges 表，不影响 karma_edges_personal 表。

        Args:
            checkpoint_id: 目标检查点 ID
            mode: 回滚模式 — "full"（全量回滚）或 "single"（单边回滚）
            edges: 单边回滚时指定的边列表，格式 [{source, target, relation}, ...]

        Returns:
            RollbackResult: 回滚结果
        """
        # 1. 回滚前自动创建 pre_rollback 检查点
        try:
            with self._checkpoint_lock:
                self._create_checkpoint_locked(
                    tag=f"pre_rollback_{checkpoint_id}",
                    source=CheckpointSource.PRE_ROLLBACK,
                )
        except Exception as e:
            log.warning(
                "failed to create pre_rollback checkpoint, continuing rollback: %s", e
            )

        # 2. 加载检查点元数据
        meta = self._load_checkpoint_meta(checkpoint_id)
        if meta is None:
            return RollbackResult(
                mode=mode,
                checkpoint_id=checkpoint_id,
                edges_affected=0,
                success=False,
                error=f"checkpoint not found: {checkpoint_id}",
            )

        # 3. 加载检查点文件
        checkpoint_path = Path(meta.file_path)
        if not checkpoint_path.exists():
            return RollbackResult(
                mode=mode,
                checkpoint_id=checkpoint_id,
                edges_affected=0,
                success=False,
                error=f"checkpoint file not found: {meta.file_path}",
            )

        try:
            with open(checkpoint_path, "r", encoding="utf-8") as f:
                data_dict = json.load(f)
        except (json.JSONDecodeError, OSError) as e:
            return RollbackResult(
                mode=mode,
                checkpoint_id=checkpoint_id,
                edges_affected=0,
                success=False,
                error=f"checkpoint file corrupted: {e}",
            )

        checkpoint_data = CheckpointData(
            version=data_dict.get("version", 1),
            checkpoint_id=data_dict.get("checkpoint_id", ""),
            created_at=data_dict.get("created_at", ""),
            tag=data_dict.get("tag", ""),
            source=data_dict.get("source", ""),
            edge_count=data_dict.get("edge_count", 0),
            edges=data_dict.get("edges", []),
        )

        # 4. 执行回滚
        with self._checkpoint_lock:
            try:
                if mode == "full":
                    affected = self._rollback_full(checkpoint_data)
                elif mode == "single":
                    if edges is None:
                        return RollbackResult(
                            mode=mode,
                            checkpoint_id=checkpoint_id,
                            edges_affected=0,
                            success=False,
                            error="edges parameter required for single rollback",
                        )
                    affected = self._rollback_single(checkpoint_data, edges)
                else:
                    return RollbackResult(
                        mode=mode,
                        checkpoint_id=checkpoint_id,
                        edges_affected=0,
                        success=False,
                        error=f"invalid rollback mode: {mode}",
                    )

                log.info(
                    "rollback completed: mode=%s, checkpoint_id=%s, edges_affected=%d",
                    mode,
                    checkpoint_id,
                    affected,
                )

                return RollbackResult(
                    mode=mode,
                    checkpoint_id=checkpoint_id,
                    edges_affected=affected,
                    success=True,
                )

            except Exception as e:
                log.error(
                    "rollback failed: mode=%s, checkpoint_id=%s, error=%s",
                    mode,
                    checkpoint_id,
                    e,
                )
                return RollbackResult(
                    mode=mode,
                    checkpoint_id=checkpoint_id,
                    edges_affected=0,
                    success=False,
                    error=str(e),
                )

    def _rollback_full(self, checkpoint_data: CheckpointData) -> int:
        """全量回滚

        在显式事务（BEGIN IMMEDIATE）内：
          1. DELETE FROM karma_edges（清空全局业力边）
          2. INSERT 从检查点恢复所有边

        不影响 karma_edges_personal 表。
        中途失败会 ROLLBACK，不会丢失数据。

        Args:
            checkpoint_data: 检查点数据

        Returns:
            恢复的边数量
        """
        edges = checkpoint_data.edges or []
        try:
            self._graph.conn.execute("BEGIN IMMEDIATE")

            # 清空全局业力边
            self._graph.conn.execute("DELETE FROM karma_edges")

            # 从检查点恢复
            for edge in edges:
                self._graph.conn.execute(
                    "INSERT INTO karma_edges (source, target, relation, weight, source_tag) "
                    "VALUES (?, ?, ?, ?, ?)",
                    (
                        edge["source"],
                        edge["target"],
                        edge["relation"],
                        edge["weight"],
                        edge.get("source_tag", ""),
                    ),
                )

            self._graph.conn.commit()
            return len(edges)
        except Exception:
            self._graph.conn.rollback()
            raise

    def _rollback_single(
        self, checkpoint_data: CheckpointData, edges: list[dict]
    ) -> int:
        """单边回滚

        仅恢复指定边的权重（UPSERT），检查点中不存在的边则 DELETE。
        不影响 karma_edges_personal 表。
        使用显式事务（BEGIN IMMEDIATE）保证原子性，中途失败会 ROLLBACK。

        Args:
            checkpoint_data: 检查点数据
            edges: 要回滚的边列表，格式 [{source, target, relation}, ...]

        Returns:
            受影响的边数量
        """
        # 构建检查点中边的索引：(source, target, relation) → edge_data
        checkpoint_edges: dict[tuple[str, str, str], dict] = {}
        for edge in checkpoint_data.edges or []:
            key = (edge["source"], edge["target"], edge["relation"])
            checkpoint_edges[key] = edge

        affected = 0

        try:
            self._graph.conn.execute("BEGIN IMMEDIATE")

            for edge_spec in edges:
                source = edge_spec["source"]
                target = edge_spec["target"]
                relation = edge_spec["relation"]
                key = (source, target, relation)

                if key in checkpoint_edges:
                    # 检查点中存在该边 → UPSERT 恢复权重
                    cp_edge = checkpoint_edges[key]
                    self._graph.conn.execute(
                        "INSERT INTO karma_edges (source, target, relation, weight, source_tag) "
                        "VALUES (?, ?, ?, ?, ?) "
                        "ON CONFLICT (source, target, relation) DO UPDATE "
                        "SET weight = ?, source_tag = ?",
                        (
                            source,
                            target,
                            relation,
                            cp_edge["weight"],
                            cp_edge.get("source_tag", ""),
                            cp_edge["weight"],
                            cp_edge.get("source_tag", ""),
                        ),
                    )
                else:
                    # 检查点中不存在该边 → DELETE
                    self._graph.conn.execute(
                        "DELETE FROM karma_edges "
                        "WHERE source = ? AND target = ? AND relation = ?",
                        (source, target, relation),
                    )

                affected += 1

            self._graph.conn.commit()
            return affected
        except Exception:
            self._graph.conn.rollback()
            raise

    # ───────────────────────────────────────────────────────
    #  守护线程
    # ───────────────────────────────────────────────────────

    def start_daemon(self) -> None:
        """启动守护线程

        守护线程每小时检查一次，到达 CHECKPOINT_CRON_HOUR 时创建自动检查点，
        同一天只创建一次。
        """
        if self._daemon_thread is not None and self._daemon_thread.is_alive():
            log.warning("checkpoint daemon already running")
            return

        self._shutdown_event.clear()
        self._daemon_thread = threading.Thread(
            target=self._daemon_loop,
            name="checkpoint-daemon",
            daemon=True,
        )
        self._daemon_thread.start()
        log.info(
            "checkpoint daemon started, cron_hour=%d", CHECKPOINT_CRON_HOUR
        )

    def stop_daemon(self) -> None:
        """停止守护线程"""
        if self._daemon_thread is None or not self._daemon_thread.is_alive():
            return

        self._shutdown_event.set()
        self._daemon_thread.join(timeout=10.0)

        if self._daemon_thread.is_alive():
            log.warning("checkpoint daemon did not stop within timeout")
        else:
            log.info("checkpoint daemon stopped")

        self._daemon_thread = None

    def _daemon_loop(self) -> None:
        """守护线程主循环

        每小时检查一次当前时间，若到达 CHECKPOINT_CRON_HOUR 且今天尚未
        创建过自动检查点，则创建一个 auto_cron 检查点。
        """
        while not self._shutdown_event.is_set():
            now = datetime.now(timezone.utc)
            current_date = now.strftime("%Y-%m-%d")
            current_hour = now.hour

            # 到达定时时间且今天尚未创建自动检查点
            if (
                current_hour == CHECKPOINT_CRON_HOUR
                and self._last_auto_checkpoint_date != current_date
            ):
                try:
                    self.create_checkpoint(
                        tag=f"auto_cron_{current_date}",
                        source=CheckpointSource.AUTO_CRON,
                    )
                    self._last_auto_checkpoint_date = current_date
                    log.info(
                        "auto cron checkpoint created for %s", current_date
                    )
                except Exception as e:
                    log.error(
                        "auto cron checkpoint failed for %s: %s",
                        current_date,
                        e,
                    )

            # 等待 1 小时或直到收到关闭信号
            self._shutdown_event.wait(timeout=3600.0)