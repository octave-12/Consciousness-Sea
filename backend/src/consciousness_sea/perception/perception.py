"""
PerceptionManager — 感知管理器

统一管理所有感知通道的启停、状态查询、事件分发。
Phase 6 的入口组件，将感知激活事件分发给 Hebbian 绑定器。
"""

from __future__ import annotations

import logging
import threading
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import TYPE_CHECKING, Optional

from consciousness_sea.domain.graph_db import GraphDB
from consciousness_sea.infrastructure.config import (
    PERCEPTION_ENABLED,
    PERCEPTION_SHUTDOWN_TIMEOUT,
    PERCEPTION_CHANNEL_FAILURE_ALERT_THRESHOLD,
)

if TYPE_CHECKING:
    from .visual_anchor import VisualAnchor
    from .audio_anchor import AudioAnchor
    from .somatic_anchor import SomaticAnchor
    from .hebbian_binder import HebbianBinder
    from .multimodal_aligner import MultimodalAligner

log = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════
#  枚举与数据类
# ═══════════════════════════════════════════════════════════


class PerceptionChannel(str, Enum):
    """感知通道枚举"""
    VISUAL = "visual"
    AUDITORY = "auditory"
    SOMATIC = "somatic"


class PerceptualSeedStatus(str, Enum):
    """感知元种子状态枚举"""
    ACTIVE = "active"
    DISABLED = "disabled"
    RETIRED = "retired"


@dataclass
class PerceptActivationEvent:
    """感知激活事件"""
    perceptual_seed: str           # 感知元种子 label
    activation: float              # 激活值 [0.0, 1.0]
    timestamp: str                 # ISO 8601 毫秒精度
    channel: PerceptionChannel     # 感知通道


@dataclass
class ConceptActivationEvent:
    """概念种子激活事件（来自路由器）"""
    activated_seeds: list[str]     # 激活的概念种子 label 列表
    timestamp: str                 # ISO 8601 毫秒精度


@dataclass
class ChannelStatus:
    """单个感知通道状态"""
    running: bool
    last_activation: str | None = None
    mock_mode: bool = False
    consecutive_failures: int = 0


@dataclass
class PerceptionManagerStatus:
    """感知管理器整体状态"""
    enabled: bool
    channels: dict[str, ChannelStatus] = field(default_factory=dict)
    total_perceptual_seeds: int = 0
    total_hebbian_bindings: int = 0
    recent_activation_count: int = 0
    last_multimodal_alignment: str | None = None


# ═══════════════════════════════════════════════════════════
#  合法通道集合
# ═══════════════════════════════════════════════════════════

_VALID_CHANNELS: frozenset[str] = frozenset({"visual", "auditory", "somatic"})


# ═══════════════════════════════════════════════════════════
#  PerceptionManager
# ═══════════════════════════════════════════════════════════


class PerceptionManager:
    """感知管理器 — 统一管理所有感知通道的启停、状态查询、事件分发

    职责:
      - 管理视觉/听觉/本体感知锚定器的生命周期
      - 接收感知激活事件并分发给 Hebbian 绑定器
      - 接收概念种子激活事件（来自路由器）并转发给 Hebbian 绑定器
      - 提供感知通道状态查询
      - 生成预设感知元种子

    线程安全:
      - _event_lock 保护事件队列
      - 各感知通道在独立线程中运行
      - Hebbian 绑定器在独立线程中运行

    Args:
        graph: 知识图谱连接
    """

    def __init__(self, graph: GraphDB) -> None:
        self._graph = graph
        self._event_lock = threading.Lock()
        self._shutdown_event = threading.Event()

        # 感知通道（懒加载）
        self._visual_anchor: VisualAnchor | None = None
        self._auditory_anchor: AudioAnchor | None = None
        self._somatic_anchor: SomaticAnchor | None = None

        # Hebbian 绑定器（懒加载）
        self._hebbian_binder: HebbianBinder | None = None

        # 多模态对齐器（懒加载）
        self._multimodal_aligner: MultimodalAligner | None = None

        # 感知激活事件队列（最大容量 1000）
        self._percept_queue: deque[PerceptActivationEvent] = deque(maxlen=1000)

        # 通道状态
        self._channel_status: dict[str, ChannelStatus] = {
            "visual": ChannelStatus(running=False),
            "auditory": ChannelStatus(running=False),
            "somatic": ChannelStatus(running=False),
        }

    # ── 生命周期管理 ──────────────────────────────────

    def start(self) -> None:
        """启动感知管理器和所有感知通道"""
        if not PERCEPTION_ENABLED:
            log.info("perception disabled, skipping start")
            return

        # 1. 生成预设感知元种子
        self._generate_preset_perceptual_seeds()

        # 2. 启动 Hebbian 绑定器
        from .hebbian_binder import HebbianBinder
        self._hebbian_binder = HebbianBinder(self._graph)
        self._hebbian_binder.start()

        # 3. 启动多模态对齐器
        from .multimodal_aligner import MultimodalAligner
        self._multimodal_aligner = MultimodalAligner(self._graph)

        # 4. 启动各感知通道
        self._start_visual_channel()
        self._start_auditory_channel()
        self._start_somatic_channel()

        log.info("perception manager started")

    def stop(self) -> None:
        """优雅停止所有感知通道和 Hebbian 绑定器"""
        # 停止各感知通道
        if self._visual_anchor is not None:
            self._visual_anchor.stop()
        if self._auditory_anchor is not None:
            self._auditory_anchor.stop()
        if self._somatic_anchor is not None:
            self._somatic_anchor.stop()

        # 停止 Hebbian 绑定器
        if self._hebbian_binder is not None:
            self._hebbian_binder.stop()

        self._shutdown_event.set()
        log.info("perception manager stopped")

    # ── 通道启停 ──────────────────────────────────────

    def start_channel(self, channel: str) -> bool:
        """启动指定感知通道

        Args:
            channel: 通道名称（visual / auditory / somatic）

        Returns:
            True 启动成功, False 启动失败或通道不合法
        """
        if not PERCEPTION_ENABLED:
            return False

        if channel == "visual":
            return self._start_visual_channel()
        elif channel == "auditory":
            return self._start_auditory_channel()
        elif channel == "somatic":
            return self._start_somatic_channel()
        else:
            log.warning("unknown perception channel: %s", channel)
            return False

    def stop_channel(self, channel: str) -> bool:
        """停止指定感知通道

        Args:
            channel: 通道名称（visual / auditory / somatic）

        Returns:
            True 停止成功, False 通道不合法或未运行
        """
        if channel == "visual" and self._visual_anchor is not None:
            self._visual_anchor.stop()
            self._channel_status["visual"].running = False
            return True
        elif channel == "auditory" and self._auditory_anchor is not None:
            self._auditory_anchor.stop()
            self._channel_status["auditory"].running = False
            return True
        elif channel == "somatic" and self._somatic_anchor is not None:
            self._somatic_anchor.stop()
            self._channel_status["somatic"].running = False
            return True
        else:
            log.warning("cannot stop channel '%s': not running or unknown", channel)
            return False

    # ── 事件分发 ──────────────────────────────────────

    def on_percept_activation(self, event: PerceptActivationEvent) -> None:
        """接收感知激活事件，分发给 Hebbian 绑定器

        由各感知锚定器调用。
        线程安全: _event_lock 仅保护事件队列和通道状态，
        数据库操作在锁外执行以减少持锁时间。
        """
        with self._event_lock:
            self._percept_queue.append(event)
            ch = event.channel.value
            if ch in self._channel_status:
                self._channel_status[ch].last_activation = event.timestamp

        if self._hebbian_binder is not None:
            self._hebbian_binder.on_percept_activation(event)

        try:
            self._graph.conn.execute(
                "UPDATE seeds SET activation = ? WHERE label = ?",
                (event.activation, event.perceptual_seed),
            )
        except Exception as e:
            log.warning("感知元种子激活值更新失败: %s", e)

        try:
            self._graph.conn.execute(
                "INSERT INTO perception_events "
                "(perceptual_seed, activation, channel, timestamp) "
                "VALUES (?, ?, ?, ?)",
                (event.perceptual_seed, event.activation,
                 event.channel.value, event.timestamp),
            )
        except Exception as e:
            log.warning("感知事件写入失败: %s", e)

        try:
            self._graph.conn.execute(
                "UPDATE perceptual_seeds SET "
                "last_activation = ?, activation_count = activation_count + 1, "
                "updated_at = ? WHERE label = ?",
                (event.timestamp, datetime.now(timezone.utc).isoformat(),
                 event.perceptual_seed),
            )
        except Exception as e:
            log.warning("感知元种子状态更新失败: %s", e)

        try:
            self._graph.conn.commit()
        except Exception as e:
            log.warning("感知事件提交失败: %s", e)

    def on_concept_activation(self, event: ConceptActivationEvent) -> None:
        """接收概念种子激活事件，转发给 Hebbian 绑定器

        由路由器调用（在涟漪传播完成后）。
        """
        if self._hebbian_binder is not None:
            self._hebbian_binder.on_concept_activation(event)

    # ── 状态查询 ──────────────────────────────────────

    def get_status(self) -> PerceptionManagerStatus:
        """查询感知管理器整体状态"""
        # 感知元种子总数
        total_perceptual_seeds = 0
        try:
            row = self._graph.conn.execute(
                "SELECT COUNT(*) as cnt FROM perceptual_seeds WHERE status = 'active'"
            ).fetchone()
            total_perceptual_seeds = row["cnt"] if row else 0
        except Exception:
            pass

        # Hebbian 绑定边总数
        total_hebbian_bindings = 0
        try:
            row = self._graph.conn.execute(
                "SELECT COUNT(*) as cnt FROM karma_edges "
                "WHERE source_tag = 'hebbian_binding'"
            ).fetchone()
            total_hebbian_bindings = row["cnt"] if row else 0
        except Exception:
            pass

        # 最近激活事件数
        recent_activation_count = 0
        try:
            row = self._graph.conn.execute(
                "SELECT COUNT(*) as cnt FROM perception_events "
                "WHERE timestamp > datetime('now', '-1 hour')"
            ).fetchone()
            recent_activation_count = row["cnt"] if row else 0
        except Exception:
            pass

        # 多模态对齐最近运行时间
        last_multimodal_alignment = None
        if self._multimodal_aligner is not None:
            try:
                last_multimodal_alignment = self._multimodal_aligner.last_run_time
            except Exception:
                pass

        # 在锁内复制通道状态，避免与 on_percept_activation 的写入竞态
        with self._event_lock:
            channels_snapshot = {
                ch: ChannelStatus(
                    running=cs.running,
                    last_activation=cs.last_activation,
                    mock_mode=cs.mock_mode,
                    consecutive_failures=cs.consecutive_failures,
                )
                for ch, cs in self._channel_status.items()
            }

        return PerceptionManagerStatus(
            enabled=PERCEPTION_ENABLED,
            channels=channels_snapshot,
            total_perceptual_seeds=total_perceptual_seeds,
            total_hebbian_bindings=total_hebbian_bindings,
            recent_activation_count=recent_activation_count,
            last_multimodal_alignment=last_multimodal_alignment,
        )

    # ── 感知元种子管理 ──────────────────────────────

    def add_perceptual_seed(
        self,
        label: str,
        channel: str,
        feature_description: str,
        activation_threshold: float,
    ) -> bool:
        """创建自定义感知元种子

        Args:
            label: 感知元种子 label
            channel: 感知通道（visual / auditory / somatic）
            feature_description: 特征描述
            activation_threshold: 激活阈值 [0.0, 1.0]

        Returns:
            True 创建成功, False 创建失败
        """
        return self._create_perceptual_seed_record(
            label, channel, feature_description, activation_threshold
        )

    def get_perceptual_seed(self, label: str) -> dict | None:
        """查询单个感知元种子详情

        Args:
            label: 感知元种子 label

        Returns:
            感知元种子信息字典，不存在返回 None
        """
        try:
            row = self._graph.conn.execute(
                "SELECT * FROM perceptual_seeds WHERE label = ?", (label,)
            ).fetchone()
            if row:
                result = dict(row)
                # 附加 Hebbian 绑定边
                bindings = self._graph.conn.execute(
                    "SELECT target, weight, relation FROM karma_edges "
                    "WHERE source = ? AND source_tag = 'hebbian_binding'",
                    (label,),
                ).fetchall()
                result["hebbian_bindings"] = [dict(b) for b in bindings]
                return result
        except Exception as e:
            log.warning("查询感知元种子失败: %s", e)
        return None

    def list_perceptual_seeds(self, channel: str | None = None) -> list[dict]:
        """查询所有感知元种子

        Args:
            channel: 按通道过滤（可选）

        Returns:
            感知元种子信息字典列表
        """
        try:
            if channel:
                rows = self._graph.conn.execute(
                    "SELECT * FROM perceptual_seeds WHERE channel = ?",
                    (channel,),
                ).fetchall()
            else:
                rows = self._graph.conn.execute(
                    "SELECT * FROM perceptual_seeds"
                ).fetchall()
            return [dict(r) for r in rows]
        except Exception as e:
            log.warning("查询感知元种子列表失败: %s", e)
            return []

    def list_hebbian_bindings(self, channel: str | None = None) -> list[dict]:
        """查询所有 Hebbian 绑定边

        Args:
            channel: 按通道过滤（可选，过滤 source 前缀）

        Returns:
            绑定边信息字典列表
        """
        try:
            if channel:
                prefix = f"percept:{channel}:"
                rows = self._graph.conn.execute(
                    "SELECT source, target, relation, weight, source_tag "
                    "FROM karma_edges WHERE source_tag = 'hebbian_binding' "
                    "AND source LIKE ?",
                    (prefix + "%",),
                ).fetchall()
            else:
                rows = self._graph.conn.execute(
                    "SELECT source, target, relation, weight, source_tag "
                    "FROM karma_edges WHERE source_tag = 'hebbian_binding'"
                ).fetchall()
            return [dict(r) for r in rows]
        except Exception as e:
            log.warning("查询 Hebbian 绑定边失败: %s", e)
            return []

    def list_perception_events(self, limit: int = 20) -> list[dict]:
        """查询最近的感知激活事件

        Args:
            limit: 返回条数上限（最大 1000）

        Returns:
            事件信息字典列表
        """
        # 上限保护
        limit = max(1, min(limit, 1000))
        try:
            rows = self._graph.conn.execute(
                "SELECT * FROM perception_events "
                "ORDER BY timestamp DESC LIMIT ?",
                (limit,),
            ).fetchall()
            return [dict(r) for r in rows]
        except Exception as e:
            log.warning("查询感知激活事件失败: %s", e)
            return []

    # ── 多模态对齐 ──────────────────────────────────

    def run_multimodal_alignment(self) -> list[dict]:
        """执行一次多模态对齐

        Returns:
            对齐结果列表
        """
        if self._multimodal_aligner is None:
            return []
        return self._multimodal_aligner.run_alignment()

    @property
    def multimodal_aligner(self):
        """获取多模态对齐器实例"""
        return self._multimodal_aligner

    # ── 内部方法 ──────────────────────────────────────

    def _generate_preset_perceptual_seeds(self) -> int:
        """生成预设感知元种子（~16 个）

        视觉 6 个: percept:visual:red, green, blue, bright, dark, edge_dense
        听觉 5 个: percept:auditory:high_freq, low_freq, bright_sound, dark_sound, percussive
        本体 5 个: percept:somatic:high_temp, high_memory, slow_response, low_temp, low_memory

        Returns:
            新创建的感知元种子数量
        """
        presets: list[tuple[str, str, str, float]] = [
            # 视觉 6 个
            ("percept:visual:red", "visual", "红色通道占比", 0.3),
            ("percept:visual:green", "visual", "绿色通道占比", 0.3),
            ("percept:visual:blue", "visual", "蓝色通道占比", 0.3),
            ("percept:visual:bright", "visual", "亮度占比", 0.7),
            ("percept:visual:dark", "visual", "暗度占比", 0.3),
            ("percept:visual:edge_dense", "visual", "边缘密度", 0.4),
            # 听觉 5 个
            ("percept:auditory:high_freq", "auditory", "高频主频率", 500.0),
            ("percept:auditory:low_freq", "auditory", "低频主频率", 200.0),
            ("percept:auditory:bright_sound", "auditory", "频谱质心（明亮音色）", 3000.0),
            ("percept:auditory:dark_sound", "auditory", "频谱质心（暗淡音色）", 1500.0),
            ("percept:auditory:percussive", "auditory", "打击性声音", 0.5),
            # 本体 5 个
            ("percept:somatic:high_temp", "somatic", "CPU 高温", 70.0),
            ("percept:somatic:high_memory", "somatic", "内存占用过高", 80.0),
            ("percept:somatic:slow_response", "somatic", "响应延迟过高", 300.0),
            ("percept:somatic:low_temp", "somatic", "CPU 低温", 50.0),
            ("percept:somatic:low_memory", "somatic", "内存占用过低", 30.0),
        ]

        created = 0
        for label, channel, desc, threshold in presets:
            if self._create_perceptual_seed_record(label, channel, desc, threshold):
                created += 1

        log.info("预设感知元种子生成完成: %d/%d", created, len(presets))
        return created

    def _start_visual_channel(self) -> bool:
        """启动视觉通道"""
        try:
            from .visual_anchor import VisualAnchor
            self._visual_anchor = VisualAnchor(self)
            self._visual_anchor.start()
            self._channel_status["visual"].running = True
            self._channel_status["visual"].mock_mode = self._visual_anchor._mock_mode
            return True
        except Exception as e:
            log.warning("visual channel unavailable: %s", e)
            self._channel_status["visual"].running = False
            return False

    def _start_auditory_channel(self) -> bool:
        """启动听觉通道"""
        try:
            from .audio_anchor import AudioAnchor
            self._auditory_anchor = AudioAnchor(self)
            self._auditory_anchor.start()
            self._channel_status["auditory"].running = True
            self._channel_status["auditory"].mock_mode = self._auditory_anchor._mock_mode
            return True
        except Exception as e:
            log.warning("auditory channel unavailable: %s", e)
            self._channel_status["auditory"].running = False
            return False

    def _start_somatic_channel(self) -> bool:
        """启动本体感知通道"""
        try:
            from .somatic_anchor import SomaticAnchor
            self._somatic_anchor = SomaticAnchor(self)
            self._somatic_anchor.start()
            self._channel_status["somatic"].running = True
            return True
        except Exception as e:
            log.warning("somatic channel unavailable: %s", e)
            self._channel_status["somatic"].running = False
            return False

    def _create_perceptual_seed_record(
        self,
        label: str,
        channel: str,
        feature_description: str,
        activation_threshold: float,
    ) -> bool:
        """创建感知元种子记录（seeds 表 + perceptual_seeds 表，原子操作）

        流程:
          1. 检查 label 前缀（必须以 "percept:" 开头）
          2. 检查 channel 合法性（visual / auditory / somatic）
          3. 检查 perceptual_seeds 表是否已存在
          4. 使用显式事务确保 seeds + perceptual_seeds 写入原子性
          5. INSERT INTO seeds (label, type='PERCEPTUAL', domain='感知', activation=0.0, aliases='[]')
          6. INSERT INTO perceptual_seeds (label, channel, feature_description, activation_threshold, status='active', ...)
          7. 若步骤 6 失败 → 事务回滚自动清除步骤 5

        Args:
            label: 感知元种子 label（如 "percept:visual:red"）
            channel: 感知通道（visual / auditory / somatic）
            feature_description: 特征描述
            activation_threshold: 激活阈值 [0.0, 1.0]

        Returns:
            True 创建成功, False 已存在或创建失败
        """
        # 1. 确保前缀
        label = self._ensure_percept_prefix(label)

        # 2. 检查 channel 合法性
        if channel not in _VALID_CHANNELS:
            log.warning("invalid perception channel: %s", channel)
            return False

        # 3. 检查是否已存在
        try:
            existing = self._graph.conn.execute(
                "SELECT label FROM perceptual_seeds WHERE label = ?", (label,)
            ).fetchone()
            if existing:
                log.debug("感知元种子已存在: %s", label)
                return False
        except Exception:
            pass  # 表可能不存在，继续创建

        now = datetime.now(timezone.utc).isoformat()

        # 4-7. 使用显式事务确保原子性
        try:
            # 开始显式事务
            self._graph.conn.execute("BEGIN IMMEDIATE")
        except Exception:
            # 可能已经在事务中，继续
            pass

        try:
            # 5. 写入 seeds 表
            self._graph.conn.execute(
                "INSERT INTO seeds (label, type, domain, activation, aliases) "
                "VALUES (?, 'PERCEPTUAL', '感知', 0.0, '[]')",
                (label,),
            )

            # 6. 写入 perceptual_seeds 表
            self._graph.conn.execute(
                "INSERT INTO perceptual_seeds "
                "(label, channel, feature_description, activation_threshold, "
                "status, activation_count, created_at, updated_at) "
                "VALUES (?, ?, ?, ?, 'active', 0, ?, ?)",
                (label, channel, feature_description, activation_threshold, now, now),
            )

            # 提交事务
            self._graph.conn.commit()
            return True
        except Exception as e:
            # 7. 事务回滚（自动清除步骤 5 的记录）
            log.warning("感知元种子创建失败: %s，事务回滚", e)
            try:
                self._graph.conn.rollback()
            except Exception:
                pass
            return False

    @staticmethod
    def _ensure_percept_prefix(label: str) -> str:
        """确保 label 以 "percept:" 前缀开头"""
        if not label.startswith("percept:"):
            return f"percept:{label}"
        return label
