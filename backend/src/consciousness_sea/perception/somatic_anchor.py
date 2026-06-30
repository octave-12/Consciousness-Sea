"""
SomaticAnchor — 本体感知锚定器

从系统指标（CPU 温度、内存占用、响应延迟）中提取特征，
激活对应的本体感觉感知元种子。系统"感受"到自己变慢、变热了。
psutil 为可选依赖，不可用时优雅降级。
"""

from __future__ import annotations

import logging
import threading
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Optional  # noqa: F401 — 保持与项目风格一致

from consciousness_sea.infrastructure.config import (
    SOMATIC_HIGH_MEMORY_THRESHOLD,
    SOMATIC_HIGH_TEMP_THRESHOLD,
    SOMATIC_SAMPLE_INTERVAL,
    SOMATIC_SLOW_RESPONSE_THRESHOLD,
)

log = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════
#  数据类
# ═══════════════════════════════════════════════════════════


@dataclass
class SomaticFeatures:
    """本体感知特征数据"""
    cpu_temp: float | None = None       # CPU 温度 (°C)
    memory_percent: float | None = None # 内存占用率 (%)
    response_latency_ms: float | None = None  # 查询响应延迟 (ms)


# ═══════════════════════════════════════════════════════════
#  SomaticAnchor
# ═══════════════════════════════════════════════════════════


class SomaticAnchor:
    """本体感知锚定器 — 从系统指标提取特征，激活感知元种子

    数据源优先级:
      1. psutil 库（如已安装）→ CPU温度、内存占用
      2. /proc 文件系统（Linux）→ CPU温度、内存占用
      3. WMI（Windows）→ CPU温度、内存占用
      4. Observer 模块 → 响应延迟
      若均不可用，则仅采集响应延迟

    性能要求: 系统指标采集延迟 < 1ms

    线程安全:
      - 采集循环在独立线程中运行

    Args:
        perception_manager: 感知管理器（用于分发激活事件）
        observer: 可观测性模块（用于获取响应延迟）
    """

    def __init__(self, perception_manager, observer=None) -> None:
        self._pm = perception_manager
        self._observer = observer
        self._shutdown_event = threading.Event()
        self._daemon_thread: threading.Thread | None = None
        self._psutil_available: bool | None = None  # 懒检测

    # ── 生命周期 ──────────────────────────────────────

    def start(self) -> None:
        """启动本体感知锚定器"""
        self._shutdown_event.clear()
        self._daemon_thread = threading.Thread(
            target=self._capture_loop,
            name="somatic-anchor",
            daemon=True,
        )
        self._daemon_thread.start()
        log.info("somatic anchor started")

    def stop(self) -> None:
        """停止本体感知锚定器"""
        self._shutdown_event.set()
        if self._daemon_thread is not None:
            self._daemon_thread.join(timeout=5.0)
            self._daemon_thread = None
        log.info("somatic anchor stopped")

    # ── 系统指标采集 ──────────────────────────────────

    def collect_features(self) -> SomaticFeatures:
        """采集系统指标

        数据源策略:
          1. 尝试 import psutil
             - psutil.sensors_temperatures() → CPU 温度
             - psutil.virtual_memory().percent → 内存占用率
          2. psutil 不可用时:
             - Linux: 读取 /sys/class/thermal/thermal_zone0/temp
             - Windows: 使用 WMI 查询（subprocess 调用）
          3. 响应延迟:
             - 从 Observer 获取最近查询的平均延迟

        Returns:
            SomaticFeatures（部分字段可能为 None）
        """
        features = SomaticFeatures()

        # CPU 温度
        features.cpu_temp = self._read_cpu_temp()

        # 内存占用
        features.memory_percent = self._read_memory_percent()

        # 响应延迟（从 Observer 获取）
        if self._observer is not None:
            try:
                features.response_latency_ms = self._observer.get_avg_response_latency()
            except Exception:
                pass

        return features

    def _read_cpu_temp(self) -> float | None:
        """读取 CPU 温度

        优先级: psutil → /proc → WMI → None
        """
        # 1. psutil
        try:
            import psutil
            temps = psutil.sensors_temperatures()
            if temps:
                for name, entries in temps.items():
                    for entry in entries:
                        if entry.current is not None and 0 < entry.current < 150:
                            return float(entry.current)
        except ImportError:
            pass
        except Exception as e:
            log.debug("psutil 温度读取失败: %s", e)

        # 2. Linux /sys/class/thermal
        try:
            from pathlib import Path
            temp_path = Path("/sys/class/thermal/thermal_zone0/temp")
            if temp_path.exists():
                raw = temp_path.read_text().strip()
                temp = float(raw) / 1000.0  # 毫度→度
                if 0 < temp < 150:
                    return temp
        except Exception:
            pass

        # 3. Windows WMI (subprocess)
        #    使用二进制模式 + 手动解码，避免 Windows 本地编码（如 GBK）
        #    与 text=True 默认 UTF-8 解码冲突导致 UnicodeDecodeError
        try:
            import subprocess
            result = subprocess.run(
                [
                    "powershell", "-Command",
                    "Get-WmiObject MSAcpi_ThermalZoneTemperature -Namespace root/wmi "
                    "| Select-Object -First 1 -ExpandProperty CurrentTemperature"
                ],
                capture_output=True, timeout=5,
            )
            if result.returncode == 0 and result.stdout.strip():
                try:
                    stdout_text = result.stdout.decode("utf-8")
                except UnicodeDecodeError:
                    stdout_text = result.stdout.decode("gbk", errors="replace")
                # WMI 返回的是十分之一开尔文
                temp_k = float(stdout_text.strip()) / 10.0
                temp_c = temp_k - 273.15
                if 0 < temp_c < 150:
                    return temp_c
        except Exception:
            pass

        return None

    def _read_memory_percent(self) -> float | None:
        """读取内存占用率

        优先级: psutil → /proc/meminfo → None
        """
        # 1. psutil
        try:
            import psutil
            return psutil.virtual_memory().percent
        except ImportError:
            pass
        except Exception:
            pass

        # 2. Linux /proc/meminfo
        try:
            from pathlib import Path
            meminfo_path = Path("/proc/meminfo")
            if meminfo_path.exists():
                content = meminfo_path.read_text()
                mem_total = None
                mem_available = None
                for line in content.splitlines():
                    parts = line.split()
                    if len(parts) >= 2:
                        key = parts[0].rstrip(":")
                        value = int(parts[1])
                        if key == "MemTotal":
                            mem_total = value
                        elif key == "MemAvailable":
                            mem_available = value
                if mem_total and mem_available and mem_total > 0:
                    return round((1.0 - mem_available / mem_total) * 100.0, 1)
        except Exception:
            pass

        return None

    # ── 阈值判定与激活 ──────────────────────────────

    def check_and_activate(self, features: SomaticFeatures) -> None:
        """根据系统指标与阈值判定是否激活感知元种子

        规则:
          - cpu_temp > SOMATIC_HIGH_TEMP_THRESHOLD → 激活 percept:somatic:high_temp
          - cpu_temp < SOMATIC_HIGH_TEMP_THRESHOLD - 20 → 激活 percept:somatic:low_temp
          - memory_percent > SOMATIC_HIGH_MEMORY_THRESHOLD → 激活 percept:somatic:high_memory
          - memory_percent < SOMATIC_HIGH_MEMORY_THRESHOLD - 30 → 激活 percept:somatic:low_memory
          - response_latency_ms > SOMATIC_SLOW_RESPONSE_THRESHOLD → 激活 percept:somatic:slow_response

        异常值过滤:
          - CPU 温度 < 0 或 > 150 → 跳过，WARNING 日志
          - 内存占用 < 0 或 > 100 → 跳过，WARNING 日志
        """
        from .perception import PerceptActivationEvent, PerceptionChannel

        now = datetime.now(timezone.utc).isoformat(timespec='milliseconds')

        # CPU 温度
        if features.cpu_temp is not None:
            temp = features.cpu_temp
            if temp < 0 or temp > 150:
                log.warning("somatic anchor: invalid CPU temperature: %s", temp)
            else:
                if temp > SOMATIC_HIGH_TEMP_THRESHOLD:
                    activation = min(1.0, (temp - SOMATIC_HIGH_TEMP_THRESHOLD) / 30.0)
                    self._pm.on_percept_activation(PerceptActivationEvent(
                        perceptual_seed="percept:somatic:high_temp",
                        activation=activation,
                        timestamp=now,
                        channel=PerceptionChannel.SOMATIC,
                    ))

                if temp < SOMATIC_HIGH_TEMP_THRESHOLD - 20:
                    activation = min(1.0, (SOMATIC_HIGH_TEMP_THRESHOLD - 20 - temp) / 20.0)
                    self._pm.on_percept_activation(PerceptActivationEvent(
                        perceptual_seed="percept:somatic:low_temp",
                        activation=activation,
                        timestamp=now,
                        channel=PerceptionChannel.SOMATIC,
                    ))

        # 内存占用
        if features.memory_percent is not None:
            mem = features.memory_percent
            if mem < 0 or mem > 100:
                log.warning("somatic anchor: invalid memory percent: %s", mem)
            else:
                if mem > SOMATIC_HIGH_MEMORY_THRESHOLD:
                    activation = min(1.0, (mem - SOMATIC_HIGH_MEMORY_THRESHOLD) / 20.0)
                    self._pm.on_percept_activation(PerceptActivationEvent(
                        perceptual_seed="percept:somatic:high_memory",
                        activation=activation,
                        timestamp=now,
                        channel=PerceptionChannel.SOMATIC,
                    ))

                if mem < SOMATIC_HIGH_MEMORY_THRESHOLD - 30:
                    activation = min(1.0, (SOMATIC_HIGH_MEMORY_THRESHOLD - 30 - mem) / 30.0)
                    self._pm.on_percept_activation(PerceptActivationEvent(
                        perceptual_seed="percept:somatic:low_memory",
                        activation=activation,
                        timestamp=now,
                        channel=PerceptionChannel.SOMATIC,
                    ))

        # 响应延迟
        if features.response_latency_ms is not None:
            latency = features.response_latency_ms
            if latency > SOMATIC_SLOW_RESPONSE_THRESHOLD:
                activation = min(1.0, (latency - SOMATIC_SLOW_RESPONSE_THRESHOLD) / SOMATIC_SLOW_RESPONSE_THRESHOLD)
                self._pm.on_percept_activation(PerceptActivationEvent(
                    perceptual_seed="percept:somatic:slow_response",
                    activation=activation,
                    timestamp=now,
                    channel=PerceptionChannel.SOMATIC,
                ))

    # ── 采集循环 ──────────────────────────────────────

    def _capture_loop(self) -> None:
        """定时采集循环"""
        while not self._shutdown_event.is_set():
            try:
                features = self.collect_features()
                self.check_and_activate(features)
            except Exception as e:
                log.warning("somatic anchor capture failed: %s", e)

            self._shutdown_event.wait(timeout=SOMATIC_SAMPLE_INTERVAL / 1000.0)
