"""
HebbianBinder — Hebbian 绑定器

检测时间窗口内感知元种子与概念种子的共同激活，
按 Hebbian 学习规则（一起激活的神经元连在一起）创建/增强绑定边。
"""

from __future__ import annotations

import logging
import threading
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Optional

from consciousness_sea.domain.graph_db import GraphDB
from .perception import PerceptActivationEvent, ConceptActivationEvent
from consciousness_sea.infrastructure.config import (
    HEBBIAN_TIME_WINDOW,
    HEBBIAN_LEARNING_RATE,
    HEBBIAN_NEGATIVE_DECAY_ENABLED,
    HEBBIAN_NEGATIVE_RATE,
    HEBBIAN_MAX_BINDINGS_PER_WINDOW,
    HEBBIAN_CHECK_INTERVAL,
    KARMA_MIN,
    KARMA_MAX,
)

log = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════
#  数据类
# ═══════════════════════════════════════════════════════════


@dataclass
class HebbianBinderStatus:
    """Hebbian 绑定器状态"""
    is_running: bool
    total_bindings: int = 0
    bindings_by_channel: dict[str, int] = field(default_factory=dict)
    top_bindings: list[dict] = field(default_factory=list)
    recent_co_activations: int = 0


# ═══════════════════════════════════════════════════════════
#  HebbianBinder
# ═══════════════════════════════════════════════════════════


class HebbianBinder:
    """Hebbian 绑定器 — 检测共同激活，按 Hebbian 规则创建/增强绑定边

    核心机制:
      - 维护感知激活事件队列和概念激活事件队列
      - 后台线程定期检查时间窗口内的共同激活
      - 共同激活 → 创建或增强 Hebbian 绑定边

    Hebbian 学习规则:
      weight_new = weight_old + HEBBIAN_LEARNING_RATE
      权重上界: KARMA_MAX (2.0)
      权重下界: KARMA_MIN (0.01)，低于此值自动删除

    时间窗口机制:
      - 感知元种子激活时间 T1，概念种子激活时间 T2
      - |T1 - T2| ≤ HEBBIAN_TIME_WINDOW (默认 1000ms) → 共同激活

    并发安全:
      - 事件队列使用 deque + _queue_lock 保护
      - 数据库操作由 SQLite WAL 模式保证并发安全
      - 与查询并发安全由 WAL 快照读保证

    Args:
        graph: 知识图谱连接
    """

    def __init__(self, graph: GraphDB) -> None:
        self._graph = graph
        self._queue_lock = threading.Lock()
        self._shutdown_event = threading.Event()
        self._daemon_thread: threading.Thread | None = None

        # 事件队列
        self._percept_queue: deque[PerceptActivationEvent] = deque(maxlen=1000)
        self._concept_queue: deque[ConceptActivationEvent] = deque(maxlen=1000)

        # 统计
        self._recent_co_activations = 0

    # ── 生命周期 ──────────────────────────────────────

    def start(self) -> None:
        """启动 Hebbian 绑定器后台线程"""
        self._shutdown_event.clear()
        self._daemon_thread = threading.Thread(
            target=self._check_loop,
            name="hebbian-binder",
            daemon=True,
        )
        self._daemon_thread.start()
        log.info("hebbian binder started")

    def stop(self) -> None:
        """停止 Hebbian 绑定器"""
        self._shutdown_event.set()
        if self._daemon_thread is not None:
            self._daemon_thread.join(timeout=5.0)
            self._daemon_thread = None
        log.info("hebbian binder stopped")

    # ── 事件接收 ──────────────────────────────────────

    def on_percept_activation(self, event: PerceptActivationEvent) -> None:
        """接收感知激活事件"""
        with self._queue_lock:
            self._percept_queue.append(event)

    def on_concept_activation(self, event: ConceptActivationEvent) -> None:
        """接收概念种子激活事件"""
        with self._queue_lock:
            self._concept_queue.append(event)

    # ── 共同激活检测 ──────────────────────────────────

    def _check_loop(self) -> None:
        """后台检测循环"""
        while not self._shutdown_event.is_set():
            try:
                self._check_co_activation()
            except Exception as e:
                log.error("hebbian binder check failed: %s", e)

            self._shutdown_event.wait(timeout=HEBBIAN_CHECK_INTERVAL / 1000.0)

    def _check_co_activation(self) -> None:
        """检测共同激活并更新绑定边

        算法:
          1. 取出所有待处理事件
          2. 对每个感知事件和概念事件，检查时间窗口
          3. 共同激活 → Hebbian 学习规则更新绑定边
          4. 可选负向衰减
          5. 提交数据库变更

        性能: 共同激活检测延迟 < 5ms
        """
        with self._queue_lock:
            percept_events = list(self._percept_queue)
            concept_events = list(self._concept_queue)
            self._percept_queue.clear()
            self._concept_queue.clear()

        if not percept_events or not concept_events:
            return

        co_activations: list[tuple[str, str]] = []

        for pe in percept_events:
            pe_ts = self._parse_timestamp(pe.timestamp)
            if pe_ts is None:
                continue

            for ce in concept_events:
                ce_ts = self._parse_timestamp(ce.timestamp)
                if ce_ts is None:
                    continue

                # 时间窗口检查
                delta_ms = abs((pe_ts - ce_ts).total_seconds() * 1000)
                if delta_ms <= HEBBIAN_TIME_WINDOW:
                    for seed_label in ce.activated_seeds:
                        co_activations.append((pe.perceptual_seed, seed_label))

        # 限制单次窗口绑定数
        co_activations = co_activations[:HEBBIAN_MAX_BINDINGS_PER_WINDOW]

        if not co_activations:
            return

        # Hebbian 学习规则
        for percept_label, concept_label in co_activations:
            try:
                # UPSERT: 创建或增强绑定边
                self._graph.conn.execute(
                    "INSERT INTO karma_edges (source, target, relation, weight, source_tag) "
                    "VALUES (?, ?, 'HEBBIAN_BIND', ?, 'hebbian_binding') "
                    "ON CONFLICT (source, target, relation) DO UPDATE "
                    "SET weight = MAX(?, MIN(?, weight + ?))",
                    (percept_label, concept_label,
                     max(KARMA_MIN, min(KARMA_MAX, HEBBIAN_LEARNING_RATE)),
                     KARMA_MIN, KARMA_MAX, HEBBIAN_LEARNING_RATE),
                )
            except Exception as e:
                log.warning("hebbian binding failed: %s → %s: %s", percept_label, concept_label, e)

        # 可选负向衰减
        if HEBBIAN_NEGATIVE_DECAY_ENABLED:
            self._apply_negative_decay(percept_events, concept_events)

        # 提交数据库变更
        try:
            self._graph.conn.commit()
        except Exception as e:
            log.warning("hebbian binder commit failed: %s", e)

        self._recent_co_activations += len(co_activations)

    def _apply_negative_decay(
        self,
        percept_events: list[PerceptActivationEvent],
        concept_events: list[ConceptActivationEvent],
    ) -> None:
        """负向衰减：感知激活但概念未在时间窗口内激活 → 减弱绑定边权重

        Args:
            percept_events: 感知激活事件列表
            concept_events: 概念激活事件列表
        """
        # 找出有共同激活的感知元种子
        co_activated_percepts: set[str] = set()
        for pe in percept_events:
            pe_ts = self._parse_timestamp(pe.timestamp)
            if pe_ts is None:
                continue
            for ce in concept_events:
                ce_ts = self._parse_timestamp(ce.timestamp)
                if ce_ts is None:
                    continue
                delta_ms = abs((pe_ts - ce_ts).total_seconds() * 1000)
                if delta_ms <= HEBBIAN_TIME_WINDOW:
                    co_activated_percepts.add(pe.perceptual_seed)

        # 对未共同激活的感知元种子，减弱其绑定边
        for pe in percept_events:
            if pe.perceptual_seed not in co_activated_percepts:
                try:
                    # 减弱该感知元种子的所有绑定边
                    self._graph.conn.execute(
                        "UPDATE karma_edges SET weight = weight - ? "
                        "WHERE source = ? AND source_tag = 'hebbian_binding' "
                        "AND weight > ?",
                        (HEBBIAN_NEGATIVE_RATE, pe.perceptual_seed, KARMA_MIN),
                    )
                    # 删除低于下界的边
                    self._graph.conn.execute(
                        "DELETE FROM karma_edges "
                        "WHERE source = ? AND source_tag = 'hebbian_binding' "
                        "AND weight < ?",
                        (pe.perceptual_seed, KARMA_MIN),
                    )
                except Exception as e:
                    log.warning("hebbian negative decay failed: %s", e)

    # ── 状态查询 ──────────────────────────────────────

    def get_status(self) -> HebbianBinderStatus:
        """查询绑定器状态"""
        total_bindings = 0
        bindings_by_channel: dict[str, int] = {}
        top_bindings: list[dict] = []

        try:
            # 绑定边总数
            row = self._graph.conn.execute(
                "SELECT COUNT(*) as cnt FROM karma_edges WHERE source_tag = 'hebbian_binding'"
            ).fetchone()
            total_bindings = row["cnt"] if row else 0

            # 各通道绑定边数量
            for channel in ("visual", "auditory", "somatic"):
                prefix = f"percept:{channel}:"
                ch_row = self._graph.conn.execute(
                    "SELECT COUNT(*) as cnt FROM karma_edges "
                    "WHERE source_tag = 'hebbian_binding' AND source LIKE ?",
                    (prefix + "%",),
                ).fetchone()
                bindings_by_channel[channel] = ch_row["cnt"] if ch_row else 0

            # 最强绑定边 Top-10
            rows = self._graph.conn.execute(
                "SELECT source, target, weight FROM karma_edges "
                "WHERE source_tag = 'hebbian_binding' "
                "ORDER BY weight DESC LIMIT 10"
            ).fetchall()
            top_bindings = [dict(r) for r in rows]
        except Exception as e:
            log.warning("hebbian binder status query failed: %s", e)

        return HebbianBinderStatus(
            is_running=self._daemon_thread is not None and self._daemon_thread.is_alive(),
            total_bindings=total_bindings,
            bindings_by_channel=bindings_by_channel,
            top_bindings=top_bindings,
            recent_co_activations=self._recent_co_activations,
        )

    # ── 时间戳解析 ──────────────────────────────────

    @staticmethod
    def _parse_timestamp(ts: str) -> datetime | None:
        """解析 ISO 8601 时间戳

        Args:
            ts: ISO 8601 格式时间戳字符串

        Returns:
            datetime 对象，解析失败返回 None
        """
        if not ts:
            return None
        try:
            # 尝试标准 ISO 格式
            return datetime.fromisoformat(ts)
        except (ValueError, TypeError):
            pass

        # 尝试去掉时区后缀
        try:
            clean = ts.rstrip("Z")
            if "+" in clean:
                clean = clean[:clean.index("+")]
            return datetime.fromisoformat(clean)
        except (ValueError, TypeError):
            log.debug("timestamp parse failed: %s", ts)
            return None