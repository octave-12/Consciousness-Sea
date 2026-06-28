"""
VisualAnchor — 视觉锚定器

从摄像头/静态图像提取颜色直方图、边缘密度等轻量视觉特征，
根据阈值激活对应的视觉感知元种子。延迟 < 10ms。
"""

from __future__ import annotations

import logging
import struct
import threading
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Optional  # noqa: F401 — 保持与项目风格一致

from .config import (
    VISUAL_FRAME_INTERVAL,
    VISUAL_MOCK_MODE,
    VISUAL_RED_THRESHOLD,
    VISUAL_GREEN_THRESHOLD,
    VISUAL_BLUE_THRESHOLD,
    VISUAL_BRIGHT_THRESHOLD,
    VISUAL_DARK_THRESHOLD,
    VISUAL_EDGE_DENSE_THRESHOLD,
)

log = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════
#  数据类
# ═══════════════════════════════════════════════════════════


@dataclass
class VisualFeatures:
    """视觉特征数据"""
    red_ratio: float = 0.0       # 红色通道占比 [0.0, 1.0]
    green_ratio: float = 0.0     # 绿色通道占比 [0.0, 1.0]
    blue_ratio: float = 0.0      # 蓝色通道占比 [0.0, 1.0]
    brightness: float = 0.0      # 亮度 [0.0, 1.0]
    edge_density: float = 0.0    # 边缘密度 [0.0, 1.0]


# ═══════════════════════════════════════════════════════════
#  VisualAnchor
# ═══════════════════════════════════════════════════════════


class VisualAnchor:
    """视觉锚定器 — 从摄像头/静态图像提取视觉特征，激活感知元种子

    特征提取算法:
      - 颜色直方图: 统计 RGB 各通道像素值分布，计算各颜色占比
      - 边缘密度: 使用 Sobel 算子检测边缘像素比例
      - 亮度: 计算灰度均值 / 255

    性能要求: 640×480 图像特征提取延迟 < 10ms

    Mock 模式:
      - VISUAL_MOCK_MODE=True 时不访问摄像头
      - 通过 inject_mock_frame() 注入模拟帧数据

    线程安全:
      - 采集循环在独立线程中运行
      - _frame_lock 保护帧数据读写

    Args:
        perception_manager: 感知管理器（用于分发激活事件）
    """

    def __init__(self, perception_manager) -> None:
        self._pm = perception_manager
        self._shutdown_event = threading.Event()
        self._daemon_thread: threading.Thread | None = None
        self._frame_lock = threading.Lock()
        self._mock_mode = VISUAL_MOCK_MODE
        self._mock_frame: bytes | None = None
        self._video_capture = None  # 复用 cv2.VideoCapture 实例

        # 校验阈值合法性
        self._red_threshold = _clamp_threshold(VISUAL_RED_THRESHOLD, 0.3, "red")
        self._green_threshold = _clamp_threshold(VISUAL_GREEN_THRESHOLD, 0.3, "green")
        self._blue_threshold = _clamp_threshold(VISUAL_BLUE_THRESHOLD, 0.3, "blue")
        self._bright_threshold = _clamp_threshold(VISUAL_BRIGHT_THRESHOLD, 0.7, "bright")
        self._dark_threshold = _clamp_threshold(VISUAL_DARK_THRESHOLD, 0.3, "dark")
        self._edge_dense_threshold = _clamp_threshold(VISUAL_EDGE_DENSE_THRESHOLD, 0.4, "edge_dense")

    # ── 生命周期 ──────────────────────────────────────

    def start(self) -> None:
        """启动视觉锚定器"""
        if self._mock_mode:
            log.info("visual anchor started (mock mode)")
            return

        self._shutdown_event.clear()
        self._daemon_thread = threading.Thread(
            target=self._capture_loop,
            name="visual-anchor",
            daemon=True,
        )
        self._daemon_thread.start()
        log.info("visual anchor started")

    def stop(self) -> None:
        """停止视觉锚定器"""
        self._shutdown_event.set()
        if self._daemon_thread is not None:
            self._daemon_thread.join(timeout=5.0)
            self._daemon_thread = None
        # 释放复用的 VideoCapture 实例
        if self._video_capture is not None:
            try:
                self._video_capture.release()
            except Exception:
                pass
            self._video_capture = None
        log.info("visual anchor stopped")

    # ── 特征提取 ──────────────────────────────────────

    @staticmethod
    def extract_features(frame_data: bytes, width: int = 640, height: int = 480) -> VisualFeatures:
        """从图像帧提取视觉特征

        算法:
          1. 颜色直方图:
             - 将 RGB 像素按通道统计
             - red_ratio = count(R > 128 && R > G && R > B) / total_pixels
             - green_ratio = count(G > 128 && G > R && G > B) / total_pixels
             - blue_ratio = count(B > 128 && B > R && B > G) / total_pixels
             - brightness = mean((R+G+B)/3) / 255

          2. 边缘密度:
             - 将图像转为灰度
             - 使用 3×3 Sobel 算子计算梯度幅值（近似: abs(gx) + abs(gy)）
             - edge_density = count(gradient > 50) / total_pixels

        性能: 640×480 图像 < 10ms（使用 struct.unpack 批量解析像素）

        Args:
            frame_data: RGB 格式原始像素数据
            width: 图像宽度
            height: 图像高度

        Returns:
            VisualFeatures
        """
        total_pixels = width * height
        if total_pixels == 0 or len(frame_data) < total_pixels * 3:
            return VisualFeatures()

        # 批量解析 RGB 像素
        pixel_count = min(total_pixels, len(frame_data) // 3)
        fmt = f"<{pixel_count * 3}B"
        try:
            pixels = struct.unpack(fmt, frame_data[:pixel_count * 3])
        except struct.error:
            return VisualFeatures()

        # 颜色统计
        red_count = 0
        green_count = 0
        blue_count = 0
        brightness_sum = 0.0

        for i in range(pixel_count):
            r = pixels[i * 3]
            g = pixels[i * 3 + 1]
            b = pixels[i * 3 + 2]

            if r > 128 and r > g and r > b:
                red_count += 1
            if g > 128 and g > r and g > b:
                green_count += 1
            if b > 128 and b > r and b > g:
                blue_count += 1

            brightness_sum += (r + g + b) / 3.0

        red_ratio = red_count / pixel_count if pixel_count else 0.0
        green_ratio = green_count / pixel_count if pixel_count else 0.0
        blue_ratio = blue_count / pixel_count if pixel_count else 0.0
        brightness = (brightness_sum / pixel_count / 255.0) if pixel_count else 0.0

        # 边缘密度（Sobel 算子）
        edge_count = 0
        if width >= 3 and height >= 3:
            # 构建灰度图
            gray = []
            for i in range(pixel_count):
                gray.append(
                    int(pixels[i * 3] * 0.299
                        + pixels[i * 3 + 1] * 0.587
                        + pixels[i * 3 + 2] * 0.114)
                )

            # Sobel 边缘检测
            for y in range(1, height - 1):
                for x in range(1, width - 1):
                    idx = y * width + x
                    # Sobel X
                    gx = (
                        -gray[(y - 1) * width + (x - 1)]
                        + gray[(y - 1) * width + (x + 1)]
                        - 2 * gray[y * width + (x - 1)]
                        + 2 * gray[y * width + (x + 1)]
                        - gray[(y + 1) * width + (x - 1)]
                        + gray[(y + 1) * width + (x + 1)]
                    )
                    # Sobel Y
                    gy = (
                        -gray[(y - 1) * width + (x - 1)]
                        - 2 * gray[(y - 1) * width + x]
                        - gray[(y - 1) * width + (x + 1)]
                        + gray[(y + 1) * width + (x - 1)]
                        + 2 * gray[(y + 1) * width + x]
                        + gray[(y + 1) * width + (x + 1)]
                    )
                    gradient = abs(gx) + abs(gy)
                    if gradient > 50:
                        edge_count += 1

            interior_pixels = (width - 2) * (height - 2)
            edge_density = edge_count / interior_pixels if interior_pixels else 0.0
        else:
            edge_density = 0.0

        return VisualFeatures(
            red_ratio=red_ratio,
            green_ratio=green_ratio,
            blue_ratio=blue_ratio,
            brightness=brightness,
            edge_density=edge_density,
        )

    # ── 阈值判定与激活 ──────────────────────────────

    def check_and_activate(self, features: VisualFeatures) -> None:
        """根据特征值与阈值判定是否激活感知元种子

        规则:
          - red_ratio > threshold → 激活 percept:visual:red
          - green_ratio > threshold → 激活 percept:visual:green
          - blue_ratio > threshold → 激活 percept:visual:blue
          - brightness > threshold → 激活 percept:visual:bright
          - brightness < threshold → 激活 percept:visual:dark
          - edge_density > threshold → 激活 percept:visual:edge_dense

        激活值 = 特征值本身
        """
        from .perception import PerceptActivationEvent, PerceptionChannel

        now = datetime.now(timezone.utc).isoformat(timespec='milliseconds')

        if features.red_ratio > self._red_threshold:
            self._pm.on_percept_activation(PerceptActivationEvent(
                perceptual_seed="percept:visual:red",
                activation=features.red_ratio,
                timestamp=now,
                channel=PerceptionChannel.VISUAL,
            ))

        if features.green_ratio > self._green_threshold:
            self._pm.on_percept_activation(PerceptActivationEvent(
                perceptual_seed="percept:visual:green",
                activation=features.green_ratio,
                timestamp=now,
                channel=PerceptionChannel.VISUAL,
            ))

        if features.blue_ratio > self._blue_threshold:
            self._pm.on_percept_activation(PerceptActivationEvent(
                perceptual_seed="percept:visual:blue",
                activation=features.blue_ratio,
                timestamp=now,
                channel=PerceptionChannel.VISUAL,
            ))

        if features.brightness > self._bright_threshold:
            self._pm.on_percept_activation(PerceptActivationEvent(
                perceptual_seed="percept:visual:bright",
                activation=features.brightness,
                timestamp=now,
                channel=PerceptionChannel.VISUAL,
            ))

        if features.brightness < self._dark_threshold:
            self._pm.on_percept_activation(PerceptActivationEvent(
                perceptual_seed="percept:visual:dark",
                activation=1.0 - features.brightness,
                timestamp=now,
                channel=PerceptionChannel.VISUAL,
            ))

        if features.edge_density > self._edge_dense_threshold:
            self._pm.on_percept_activation(PerceptActivationEvent(
                perceptual_seed="percept:visual:edge_dense",
                activation=features.edge_density,
                timestamp=now,
                channel=PerceptionChannel.VISUAL,
            ))

    # ── 静态图像输入 ──────────────────────────────────

    def process_image(self, image_path: str) -> VisualFeatures | None:
        """从静态图像文件提取特征

        Args:
            image_path: 图像文件路径

        Returns:
            VisualFeatures 或 None（解码失败时）
        """
        try:
            from pathlib import Path
            path = Path(image_path)
            if not path.exists():
                log.warning("visual anchor: image not found: %s", image_path)
                return None

            # 尝试使用 PIL/Pillow 读取
            try:
                from PIL import Image
                img = Image.open(path).convert("RGB")
                img = img.resize((640, 480))
                frame_data = img.tobytes()
            except ImportError:
                # 无 PIL，尝试读取原始 RGB 数据
                data = path.read_bytes()
                if len(data) < 640 * 480 * 3:
                    log.warning("visual anchor: image too small or format unsupported: %s", image_path)
                    return None
                frame_data = data[:640 * 480 * 3]

            features = self.extract_features(frame_data)
            self.check_and_activate(features)
            return features
        except Exception as e:
            log.warning("visual anchor: image decode failed: %s", e)
            return None

    # ── Mock 模式 ──────────────────────────────────────

    def inject_mock_frame(self, frame_data: bytes) -> None:
        """注入模拟帧数据（mock 模式下使用）

        Args:
            frame_data: RGB 格式原始像素数据
        """
        with self._frame_lock:
            self._mock_frame = frame_data

        # 立即处理注入的帧
        features = self.extract_features(frame_data)
        self.check_and_activate(features)

    # ── 采集循环 ──────────────────────────────────────

    def _capture_loop(self) -> None:
        """摄像头采集循环"""
        while not self._shutdown_event.is_set():
            try:
                frame = self._capture_frame()
                if frame is not None:
                    features = self.extract_features(frame)
                    self.check_and_activate(features)
            except Exception as e:
                log.warning("visual anchor capture failed: %s", e)

            self._shutdown_event.wait(timeout=VISUAL_FRAME_INTERVAL / 1000.0)

    def _capture_frame(self) -> bytes | None:
        """从摄像头采集一帧图像

        使用标准库或可选的 cv2 库采集。
        复用 VideoCapture 实例避免频繁创建/销毁。
        不可用时返回 None。
        """
        try:
            import cv2
            if self._video_capture is None:
                self._video_capture = cv2.VideoCapture(0)
            if not self._video_capture.isOpened():
                self._video_capture.release()
                self._video_capture = None
                return None
            ret, frame = self._video_capture.read()
            if ret:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                return frame.tobytes()
        except ImportError:
            log.debug("cv2 not available, camera capture disabled")
        except Exception as e:
            log.debug("camera capture failed: %s", e)
            # 采集失败时释放实例，下次重新创建
            if self._video_capture is not None:
                try:
                    self._video_capture.release()
                except Exception:
                    pass
                self._video_capture = None

        return None


# ═══════════════════════════════════════════════════════════
#  辅助函数
# ═══════════════════════════════════════════════════════════


def _clamp_threshold(value: float, default: float, name: str) -> float:
    """校验并钳制阈值到合法范围 [0.0, 1.0]"""
    if value < 0.0 or value > 1.0:
        log.warning("invalid threshold value: %s=%s, using default: %s", name, value, default)
        return default
    return value