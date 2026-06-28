"""
AudioAnchor — 听觉锚定器

从麦克风音频提取频谱特征（主频率、频谱质心、频谱带宽、打击性检测），
根据频率模式激活对应的听觉感知元种子。延迟 < 10ms。
使用纯 Python math.cos / math.sin 实现 DFT，不强制依赖 NumPy。
"""

from __future__ import annotations

import logging
import math
import threading
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Optional  # noqa: F401 — 保持与项目风格一致

from .config import (
    AUDITORY_SAMPLE_RATE,
    AUDITORY_MOCK_MODE,
    AUDITORY_HIGH_FREQ_THRESHOLD,
    AUDITORY_LOW_FREQ_THRESHOLD,
    AUDITORY_BRIGHT_THRESHOLD,
)

log = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════
#  数据类
# ═══════════════════════════════════════════════════════════


@dataclass
class AudioFeatures:
    """听觉特征数据"""
    dominant_freq: float = 0.0     # 主频率 (Hz)
    spectral_centroid: float = 0.0 # 频谱质心 (Hz)
    spectral_bandwidth: float = 0.0  # 频谱带宽 (Hz)
    is_percussive: bool = False    # 是否为打击性声音


# ═══════════════════════════════════════════════════════════
#  AudioAnchor
# ═══════════════════════════════════════════════════════════


class AudioAnchor:
    """听觉锚定器 — 从麦克风音频提取频谱特征，激活感知元种子

    特征提取算法:
      - 主频率: 使用 DFT（离散傅里叶变换）计算频谱，取最大幅值对应频率
      - 频谱质心: 频谱幅值加权平均频率
      - 频谱带宽: 频谱幅值加权频率标准差
      - 打击性检测: 时域短时能量变化率

    性能要求: 1024 采样点频谱计算延迟 < 10ms

    DFT 实现说明:
      - 使用纯 Python math.cos / math.sin 实现 DFT
      - 仅计算前 N/2 个频率分量（奈奎斯特频率以下）
      - 采样率 16000Hz，1024 采样点 → 频率分辨率 ~15.6Hz

    Mock 模式:
      - AUDITORY_MOCK_MODE=True 时不访问麦克风
      - 通过 inject_mock_audio() 注入模拟音频数据

    Args:
        perception_manager: 感知管理器（用于分发激活事件）
    """

    def __init__(self, perception_manager) -> None:
        self._pm = perception_manager
        self._shutdown_event = threading.Event()
        self._daemon_thread: threading.Thread | None = None
        self._mock_mode = AUDITORY_MOCK_MODE
        self._mock_audio: list[float] | None = None
        self._pyaudio_instance = None  # 复用 PyAudio 实例
        self._audio_stream = None      # 复用音频流

        # 校验阈值合法性
        self._high_freq_threshold = _validate_freq_threshold(
            AUDITORY_HIGH_FREQ_THRESHOLD, 500.0, "high_freq"
        )
        self._low_freq_threshold = _validate_freq_threshold(
            AUDITORY_LOW_FREQ_THRESHOLD, 200.0, "low_freq"
        )
        self._bright_threshold = _validate_freq_threshold(
            AUDITORY_BRIGHT_THRESHOLD, 3000.0, "bright"
        )

    # ── 生命周期 ──────────────────────────────────────

    def start(self) -> None:
        """启动听觉锚定器"""
        if self._mock_mode:
            log.info("audio anchor started (mock mode)")
            return

        self._shutdown_event.clear()
        self._daemon_thread = threading.Thread(
            target=self._capture_loop,
            name="audio-anchor",
            daemon=True,
        )
        self._daemon_thread.start()
        log.info("audio anchor started")

    def stop(self) -> None:
        """停止听觉锚定器"""
        self._shutdown_event.set()
        if self._daemon_thread is not None:
            self._daemon_thread.join(timeout=5.0)
            self._daemon_thread = None
        # 释放复用的音频流和 PyAudio 实例
        if self._audio_stream is not None:
            try:
                self._audio_stream.stop_stream()
                self._audio_stream.close()
            except Exception:
                pass
            self._audio_stream = None
        if self._pyaudio_instance is not None:
            try:
                self._pyaudio_instance.terminate()
            except Exception:
                pass
            self._pyaudio_instance = None
        log.info("audio anchor stopped")

    # ── 特征提取 ──────────────────────────────────────

    @staticmethod
    def extract_features(
        audio_data: list[float],
        sample_rate: int = AUDITORY_SAMPLE_RATE,
    ) -> AudioFeatures:
        """从音频帧提取频谱特征

        算法:
          1. DFT 计算:
             N = len(audio_data)
             for k in range(N // 2):
               real = sum(x[n] * cos(2π * k * n / N) for n in range(N))
               imag = sum(x[n] * sin(2π * k * n / N) for n in range(N))
               magnitude[k] = sqrt(real² + imag²)
             freq[k] = k * sample_rate / N

          2. 主频率: argmax(magnitude) 对应的 freq

          3. 频谱质心:
             centroid = sum(magnitude[k] * freq[k]) / sum(magnitude[k])

          4. 频谱带宽:
             bandwidth = sqrt(sum(magnitude[k] * (freq[k] - centroid)²) / sum(magnitude[k]))

          5. 打击性检测:
             计算短时能量变化率，变化率 > 阈值 → percussive

        性能: 1024 采样点 < 10ms（纯 Python 实现）

        Args:
            audio_data: 音频采样数据（浮点数列表）
            sample_rate: 采样率 (Hz)

        Returns:
            AudioFeatures
        """
        N = len(audio_data)
        if N == 0:
            return AudioFeatures()

        # 检查无效数据（全零或包含 NaN）
        has_data = False
        for x in audio_data:
            if x != 0.0 and not math.isnan(x) and not math.isinf(x):
                has_data = True
                break
        if not has_data:
            return AudioFeatures()

        # DFT 计算（仅前 N/2 个频率分量）
        half_n = N // 2
        magnitudes: list[float] = []
        frequencies: list[float] = []

        for k in range(half_n):
            real = 0.0
            imag = 0.0
            for n in range(N):
                angle = 2.0 * math.pi * k * n / N
                real += audio_data[n] * math.cos(angle)
                imag -= audio_data[n] * math.sin(angle)
            mag = math.sqrt(real * real + imag * imag)
            magnitudes.append(mag)
            frequencies.append(k * sample_rate / N)

        if not magnitudes:
            return AudioFeatures()

        # 主频率
        max_idx = 0
        max_mag = magnitudes[0]
        for i in range(1, len(magnitudes)):
            if magnitudes[i] > max_mag:
                max_mag = magnitudes[i]
                max_idx = i
        dominant_freq = frequencies[max_idx]

        # 频谱质心
        mag_sum = sum(magnitudes)
        if mag_sum > 0:
            centroid = sum(m * f for m, f in zip(magnitudes, frequencies)) / mag_sum
        else:
            centroid = 0.0

        # 频谱带宽
        if mag_sum > 0:
            variance = sum(m * (f - centroid) ** 2 for m, f in zip(magnitudes, frequencies)) / mag_sum
            bandwidth = math.sqrt(variance)
        else:
            bandwidth = 0.0

        # 打击性检测：短时能量变化率
        is_percussive = False
        if N >= 4:
            quarter = N // 4
            energy_first = sum(x * x for x in audio_data[:quarter])
            energy_last = sum(x * x for x in audio_data[3 * quarter:])
            avg_energy = (energy_first + energy_last) / 2.0
            if avg_energy > 0:
                energy_change = abs(energy_last - energy_first) / avg_energy
                is_percussive = energy_change > 2.0

        return AudioFeatures(
            dominant_freq=dominant_freq,
            spectral_centroid=centroid,
            spectral_bandwidth=bandwidth,
            is_percussive=is_percussive,
        )

    # ── 阈值判定与激活 ──────────────────────────────

    def check_and_activate(self, features: AudioFeatures) -> None:
        """根据频谱特征与阈值判定是否激活感知元种子

        规则:
          - dominant_freq > high_freq_threshold → 激活 percept:auditory:high_freq
          - dominant_freq < low_freq_threshold → 激活 percept:auditory:low_freq
          - spectral_centroid > bright_threshold → 激活 percept:auditory:bright_sound
          - spectral_centroid < bright_threshold * 0.5 → 激活 percept:auditory:dark_sound
          - is_percussive → 激活 percept:auditory:percussive
        """
        from .perception import PerceptActivationEvent, PerceptionChannel

        now = datetime.now(timezone.utc).isoformat(timespec='milliseconds')

        if features.dominant_freq > self._high_freq_threshold:
            activation = min(1.0, features.dominant_freq / 1000.0)
            self._pm.on_percept_activation(PerceptActivationEvent(
                perceptual_seed="percept:auditory:high_freq",
                activation=activation,
                timestamp=now,
                channel=PerceptionChannel.AUDITORY,
            ))

        if features.dominant_freq < self._low_freq_threshold and features.dominant_freq > 0:
            activation = min(1.0, (self._low_freq_threshold - features.dominant_freq) / self._low_freq_threshold)
            self._pm.on_percept_activation(PerceptActivationEvent(
                perceptual_seed="percept:auditory:low_freq",
                activation=activation,
                timestamp=now,
                channel=PerceptionChannel.AUDITORY,
            ))

        if features.spectral_centroid > self._bright_threshold:
            activation = min(1.0, features.spectral_centroid / 10000.0)
            self._pm.on_percept_activation(PerceptActivationEvent(
                perceptual_seed="percept:auditory:bright_sound",
                activation=activation,
                timestamp=now,
                channel=PerceptionChannel.AUDITORY,
            ))

        if features.spectral_centroid < self._bright_threshold * 0.5 and features.spectral_centroid > 0:
            activation = min(1.0, features.spectral_centroid / self._bright_threshold)
            self._pm.on_percept_activation(PerceptActivationEvent(
                perceptual_seed="percept:auditory:dark_sound",
                activation=activation,
                timestamp=now,
                channel=PerceptionChannel.AUDITORY,
            ))

        if features.is_percussive:
            self._pm.on_percept_activation(PerceptActivationEvent(
                perceptual_seed="percept:auditory:percussive",
                activation=0.8,
                timestamp=now,
                channel=PerceptionChannel.AUDITORY,
            ))

    # ── Mock 模式 ──────────────────────────────────────

    def inject_mock_audio(self, audio_data: list[float]) -> None:
        """注入模拟音频数据（mock 模式下使用）"""
        self._mock_audio = audio_data

        # 检查无效数据
        has_valid = False
        for x in audio_data:
            if x != 0.0 and not math.isnan(x) and not math.isinf(x):
                has_valid = True
                break
        if not has_valid:
            log.warning("auditory anchor: invalid audio frame")
            return

        features = self.extract_features(audio_data)
        self.check_and_activate(features)

    # ── 采集循环 ──────────────────────────────────────

    def _capture_loop(self) -> None:
        """麦克风采集循环"""
        while not self._shutdown_event.is_set():
            try:
                audio_data = self._capture_audio()
                if audio_data is not None:
                    # 检查无效数据
                    has_valid = False
                    for x in audio_data:
                        if x != 0.0 and not math.isnan(x) and not math.isinf(x):
                            has_valid = True
                            break
                    if has_valid:
                        features = self.extract_features(audio_data)
                        self.check_and_activate(features)
                    else:
                        log.warning("auditory anchor: invalid audio frame")
            except Exception as e:
                log.warning("audio anchor capture failed: %s", e)

            self._shutdown_event.wait(timeout=0.1)

    def _capture_audio(self) -> list[float] | None:
        """从麦克风采集一帧音频

        使用标准库或可选的 pyaudio 库采集。
        复用 PyAudio 实例和音频流避免频繁创建/销毁。
        不可用时返回 None。
        """
        try:
            import pyaudio
            # 懒初始化 PyAudio 实例
            if self._pyaudio_instance is None:
                self._pyaudio_instance = pyaudio.PyAudio()
            # 懒初始化音频流
            if self._audio_stream is None:
                self._audio_stream = self._pyaudio_instance.open(
                    format=pyaudio.paInt16,
                    channels=1,
                    rate=AUDITORY_SAMPLE_RATE,
                    input=True,
                    frames_per_buffer=1024,
                )
            data = self._audio_stream.read(1024, exception_on_overflow=False)

            # 转换为浮点数
            import struct
            samples = struct.unpack(f"<{len(data) // 2}h", data)
            audio_data = [s / 32768.0 for s in samples]
            return audio_data
        except ImportError:
            log.debug("pyaudio not available, microphone capture disabled")
        except Exception as e:
            log.debug("microphone capture failed: %s", e)
            # 采集失败时释放流和实例，下次重新创建
            if self._audio_stream is not None:
                try:
                    self._audio_stream.stop_stream()
                    self._audio_stream.close()
                except Exception:
                    pass
                self._audio_stream = None
            if self._pyaudio_instance is not None:
                try:
                    self._pyaudio_instance.terminate()
                except Exception:
                    pass
                self._pyaudio_instance = None

        return None


# ═══════════════════════════════════════════════════════════
#  辅助函数
# ═══════════════════════════════════════════════════════════


def _validate_freq_threshold(value: float, default: float, name: str) -> float:
    """校验频率阈值合法性"""
    if value < 0:
        log.warning("invalid threshold value: %s=%s, using default: %s", name, value, default)
        return default
    return value