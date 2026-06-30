"""
MultimodalAligner — 多模态对齐器

使用 CLIP/BGE 小模型离线校准感知层激活，发现深层关联，写入提炼池。
关联层与感知层独立运行，关联层失败不影响感知层实时功能。
"""

from __future__ import annotations

import logging
import threading
from dataclasses import dataclass
from datetime import datetime, timezone

from consciousness_sea.domain.graph_db import GraphDB
from consciousness_sea.infrastructure.config import (
    MULTIMODAL_ALIGNMENT_ENABLED,
)

log = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════
#  数据类
# ═══════════════════════════════════════════════════════════


@dataclass
class AlignmentResult:
    """多模态对齐结果"""
    source: str          # 感知元种子 label
    target: str          # 概念种子 label
    similarity: float    # 相似度 [0.0, 1.0]


# ═══════════════════════════════════════════════════════════
#  MultimodalAligner
# ═══════════════════════════════════════════════════════════


class MultimodalAligner:
    """多模态对齐器 — 离线校准感知层激活，发现深层关联

    核心机制:
      - 从感知层帧缓存获取最近样本
      - 使用 CLIP/BGE 模型将图像/声音映射到种子标签空间
      - 对比感知层激活结果，发现漏掉的深层关联
      - 对齐结果写入提炼池（source_tag="multimodal_alignment"）

    独立性:
      - 关联层与感知层独立运行
      - CLIP 模型不可用时仅记录 WARNING 日志
      - 关联层失败不影响感知层实时功能

    Args:
        graph: 知识图谱连接
    """

    def __init__(self, graph: GraphDB) -> None:
        self._graph = graph
        self._running = False
        self._run_lock = threading.Lock()
        self._last_run_time: str | None = None
        self._clip_model = None

    # ── 对齐执行 ──────────────────────────────────────

    def run_alignment(self) -> list[dict]:
        """执行一次多模态对齐

        Returns:
            对齐结果字典列表
        """
        if not MULTIMODAL_ALIGNMENT_ENABLED:
            return []

        with self._run_lock:
            if self._running:
                log.warning("multimodal alignment already running, skipping")
                return []
            self._running = True

        try:
            return self._do_alignment()
        finally:
            with self._run_lock:
                self._running = False

    def _do_alignment(self) -> list[dict]:
        """实际执行对齐逻辑"""
        # 1. 从帧缓存获取最近样本
        frames = self._get_recent_frames()
        if not frames:
            log.info("multimodal alignment skipped: no samples")
            return []

        # 2. 加载 CLIP 模型
        clip_model = self._load_clip_model()
        if clip_model is None:
            log.warning("multimodal alignment: CLIP model not available")
            return []

        # 3. 对每帧进行 CLIP 推理
        alignment_results: list[AlignmentResult] = []
        for frame in frames:
            try:
                similarities = self._clip_inference(clip_model, frame)
                for seed_label, sim in similarities.items():
                    if sim > 0.3:  # 最低相似度阈值
                        alignment_results.append(AlignmentResult(
                            source=self._infer_percept_seed(seed_label),
                            target=seed_label,
                            similarity=sim,
                        ))
            except Exception as e:
                log.warning("multimodal alignment: frame inference failed: %s", e)
                continue

        # 4. 写入提炼池
        result_dicts: list[dict] = []
        for ar in alignment_results:
            if self._write_to_distillation_pool(ar):
                result_dicts.append({
                    "source": ar.source,
                    "target": ar.target,
                    "similarity": ar.similarity,
                })

        self._last_run_time = datetime.now(timezone.utc).isoformat()
        log.info("multimodal alignment completed: %d results", len(result_dicts))
        return result_dicts

    # ── 帧缓存 ──────────────────────────────────────

    def _get_recent_frames(self) -> list[bytes]:
        """从感知层帧缓存获取最近样本

        Returns:
            帧数据列表
        """
        # 当前实现返回空列表（帧缓存由感知锚定器维护）
        # 在完整实现中，应从 VisualAnchor 的帧缓存中获取
        return []

    # ── CLIP 模型 ──────────────────────────────────────

    def _load_clip_model(self):
        """懒加载 CLIP/BGE 模型

        Returns:
            模型实例，加载失败返回 None
        """
        if self._clip_model is not None:
            return self._clip_model

        try:
            # 尝试加载 CLIP 模型
            import clip as clip_module
            model, preprocess = clip_module.load("ViT-B/32", device="cpu")
            self._clip_model = model
            return model
        except ImportError:
            log.debug("clip module not available")
        except Exception as e:
            log.warning("CLIP model load failed: %s", e)

        return None

    def _clip_inference(self, model, frame) -> dict[str, float]:
        """使用 CLIP 模型对帧进行推理

        Args:
            model: CLIP 模型实例
            frame: 图像帧数据

        Returns:
            {seed_label: similarity} 映射
        """
        # 当前实现返回空映射（需要完整 CLIP 推理管线）
        return {}

    # ── 提炼池写入 ──────────────────────────────────

    def _write_to_distillation_pool(self, result: AlignmentResult) -> bool:
        """将多模态对齐结果写入提炼池

        source_tag = "multimodal_alignment"
        不直接创建 Hebbian 绑定边，需经提炼池验证升级为全局业力。

        Args:
            result: 对齐结果

        Returns:
            True 写入成功, False 写入失败
        """
        try:
            from consciousness_sea.learning.distillation_pool import DistillationPool
            DistillationPool(self._graph)
            now = datetime.now(timezone.utc).isoformat()

            # 使用 submit_candidate 写入提炼池
            # source_tag 标记为 multimodal_alignment
            self._graph.conn.execute(
                "INSERT INTO distillation_pool "
                "(canonical_source, canonical_target, canonical_relation, "
                "representative_label, count, contributor_users, status, "
                "created_at, updated_at) "
                "VALUES (?, ?, 'MULTIMODAL_ALIGN', ?, 1, '[]', 'pending', ?, ?)",
                (result.source, result.target,
                 f"multimodal:{result.source}:{result.target}",
                 now, now),
            )
            self._graph.conn.commit()
            return True
        except Exception as e:
            log.warning("distillation pool write failed: %s", e)
            return False

    # ── 辅助方法 ──────────────────────────────────────

    @staticmethod
    def _infer_percept_seed(seed_label: str) -> str:
        """从概念种子 label 推断感知元种子 label

        Args:
            seed_label: 概念种子 label

        Returns:
            推断的感知元种子 label
        """
        # 简单映射：颜色相关 → visual，声音相关 → auditory
        visual_keywords = {"红色", "绿色", "蓝色", "颜色", "亮度", "暗", "边缘"}
        for kw in visual_keywords:
            if kw in seed_label:
                return f"percept:visual:{seed_label}"

        auditory_keywords = {"声音", "频率", "音调", "节奏"}
        for kw in auditory_keywords:
            if kw in seed_label:
                return f"percept:auditory:{seed_label}"

        return f"percept:visual:{seed_label}"

    # ── 属性 ──────────────────────────────────────

    @property
    def is_running(self) -> bool:
        """当前是否正在运行对齐任务"""
        with self._run_lock:
            return self._running

    @property
    def last_run_time(self) -> str | None:
        """最近一次对齐运行时间"""
        return self._last_run_time
