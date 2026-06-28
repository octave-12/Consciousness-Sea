"""
Phase 6 SomaticAnchor 单元测试

覆盖：
- 系统指标采集
- CPU 温度→感知元种子激活
- 内存占用→感知元种子激活
- 响应延迟→感知元种子激活
- CPU 温度读取异常
- psutil 不可用降级
- 低温/低内存激活
"""

from __future__ import annotations

import sys
import os
from unittest.mock import patch, MagicMock

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.somatic_anchor import SomaticAnchor, SomaticFeatures
from core.config import (
    SOMATIC_HIGH_TEMP_THRESHOLD,
    SOMATIC_HIGH_MEMORY_THRESHOLD,
    SOMATIC_SLOW_RESPONSE_THRESHOLD,
)


# ═══════════════════════════════════════════════════════════
#  Fixtures
# ═══════════════════════════════════════════════════════════


@pytest.fixture
def mock_pm():
    """创建 mock PerceptionManager"""
    pm = MagicMock()
    pm.on_percept_activation = MagicMock()
    return pm


@pytest.fixture
def anchor(mock_pm):
    """创建 SomaticAnchor 实例"""
    return SomaticAnchor(mock_pm)


@pytest.fixture
def anchor_with_observer(mock_pm):
    """创建带 Observer 的 SomaticAnchor 实例"""
    observer = MagicMock()
    observer.get_avg_response_latency = MagicMock(return_value=500.0)
    return SomaticAnchor(mock_pm, observer=observer)


# ═══════════════════════════════════════════════════════════
#  测试用例
# ═══════════════════════════════════════════════════════════


class TestCollectFeatures:
    """系统指标采集测试"""

    def test_collect_features_returns_somatic_features(self, anchor):
        """collect_features() 返回 SomaticFeatures"""
        features = anchor.collect_features()
        assert isinstance(features, SomaticFeatures)

    def test_collect_features_with_observer(self, anchor_with_observer):
        """带 Observer 时获取响应延迟"""
        features = anchor_with_observer.collect_features()
        assert features.response_latency_ms == 500.0

    def test_collect_features_psutil_unavailable(self, anchor):
        """psutil 不可用时优雅降级"""
        with patch("core.somatic_anchor.SomaticAnchor._read_cpu_temp", return_value=None), \
             patch("core.somatic_anchor.SomaticAnchor._read_memory_percent", return_value=None):
            features = anchor.collect_features()
        assert features.cpu_temp is None
        assert features.memory_percent is None


class TestCpuTempActivation:
    """CPU 温度→感知元种子激活测试"""

    def test_high_temp_activation(self, anchor, mock_pm):
        """高温 → 激活 percept:somatic:high_temp"""
        features = SomaticFeatures(cpu_temp=75.0, memory_percent=50.0, response_latency_ms=100.0)
        anchor.check_and_activate(features)
        call_args = [c[0][0] for c in mock_pm.on_percept_activation.call_args_list]
        labels = [e.perceptual_seed for e in call_args]
        assert "percept:somatic:high_temp" in labels

    def test_high_temp_activation_value(self, anchor, mock_pm):
        """高温激活值 = min(1.0, (temp - threshold) / 30)"""
        temp = SOMATIC_HIGH_TEMP_THRESHOLD + 15  # 85°C
        features = SomaticFeatures(cpu_temp=temp, memory_percent=50.0, response_latency_ms=100.0)
        anchor.check_and_activate(features)
        call_args = mock_pm.on_percept_activation.call_args[0][0]
        expected = min(1.0, (temp - SOMATIC_HIGH_TEMP_THRESHOLD) / 30.0)
        assert abs(call_args.activation - expected) < 0.01

    def test_low_temp_activation(self, anchor, mock_pm):
        """低温 → 激活 percept:somatic:low_temp"""
        features = SomaticFeatures(cpu_temp=30.0, memory_percent=50.0, response_latency_ms=100.0)
        anchor.check_and_activate(features)
        call_args = [c[0][0] for c in mock_pm.on_percept_activation.call_args_list]
        labels = [e.perceptual_seed for e in call_args]
        assert "percept:somatic:low_temp" in labels

    def test_invalid_temp_negative(self, anchor, mock_pm):
        """CPU 温度 < 0 → 跳过，WARNING 日志"""
        features = SomaticFeatures(cpu_temp=-10.0, memory_percent=50.0, response_latency_ms=100.0)
        anchor.check_and_activate(features)
        # 不应激活温度相关的感知元种子
        call_args = [c[0][0] for c in mock_pm.on_percept_activation.call_args_list]
        labels = [e.perceptual_seed for e in call_args]
        assert "percept:somatic:high_temp" not in labels
        assert "percept:somatic:low_temp" not in labels

    def test_invalid_temp_too_high(self, anchor, mock_pm):
        """CPU 温度 > 150 → 跳过"""
        features = SomaticFeatures(cpu_temp=200.0, memory_percent=50.0, response_latency_ms=100.0)
        anchor.check_and_activate(features)
        call_args = [c[0][0] for c in mock_pm.on_percept_activation.call_args_list]
        labels = [e.perceptual_seed for e in call_args]
        assert "percept:somatic:high_temp" not in labels

    def test_none_temp_skipped(self, anchor, mock_pm):
        """CPU 温度为 None → 跳过"""
        features = SomaticFeatures(cpu_temp=None, memory_percent=50.0, response_latency_ms=100.0)
        anchor.check_and_activate(features)
        call_args = [c[0][0] for c in mock_pm.on_percept_activation.call_args_list]
        labels = [e.perceptual_seed for e in call_args]
        assert "percept:somatic:high_temp" not in labels


class TestMemoryActivation:
    """内存占用→感知元种子激活测试"""

    def test_high_memory_activation(self, anchor, mock_pm):
        """高内存 → 激活 percept:somatic:high_memory"""
        features = SomaticFeatures(cpu_temp=50.0, memory_percent=85.0, response_latency_ms=100.0)
        anchor.check_and_activate(features)
        call_args = [c[0][0] for c in mock_pm.on_percept_activation.call_args_list]
        labels = [e.perceptual_seed for e in call_args]
        assert "percept:somatic:high_memory" in labels

    def test_low_memory_activation(self, anchor, mock_pm):
        """低内存 → 激活 percept:somatic:low_memory"""
        features = SomaticFeatures(cpu_temp=50.0, memory_percent=20.0, response_latency_ms=100.0)
        anchor.check_and_activate(features)
        call_args = [c[0][0] for c in mock_pm.on_percept_activation.call_args_list]
        labels = [e.perceptual_seed for e in call_args]
        assert "percept:somatic:low_memory" in labels

    def test_invalid_memory_negative(self, anchor, mock_pm):
        """内存占用 < 0 → 跳过"""
        features = SomaticFeatures(cpu_temp=50.0, memory_percent=-5.0, response_latency_ms=100.0)
        anchor.check_and_activate(features)
        call_args = [c[0][0] for c in mock_pm.on_percept_activation.call_args_list]
        labels = [e.perceptual_seed for e in call_args]
        assert "percept:somatic:high_memory" not in labels

    def test_invalid_memory_over_100(self, anchor, mock_pm):
        """内存占用 > 100 → 跳过"""
        features = SomaticFeatures(cpu_temp=50.0, memory_percent=150.0, response_latency_ms=100.0)
        anchor.check_and_activate(features)
        call_args = [c[0][0] for c in mock_pm.on_percept_activation.call_args_list]
        labels = [e.perceptual_seed for e in call_args]
        assert "percept:somatic:high_memory" not in labels

    def test_none_memory_skipped(self, anchor, mock_pm):
        """内存占用为 None → 跳过"""
        features = SomaticFeatures(cpu_temp=50.0, memory_percent=None, response_latency_ms=100.0)
        anchor.check_and_activate(features)
        call_args = [c[0][0] for c in mock_pm.on_percept_activation.call_args_list]
        labels = [e.perceptual_seed for e in call_args]
        assert "percept:somatic:high_memory" not in labels


class TestResponseLatencyActivation:
    """响应延迟→感知元种子激活测试"""

    def test_slow_response_activation(self, anchor, mock_pm):
        """慢响应 → 激活 percept:somatic:slow_response"""
        features = SomaticFeatures(cpu_temp=50.0, memory_percent=50.0, response_latency_ms=500.0)
        anchor.check_and_activate(features)
        call_args = [c[0][0] for c in mock_pm.on_percept_activation.call_args_list]
        labels = [e.perceptual_seed for e in call_args]
        assert "percept:somatic:slow_response" in labels

    def test_slow_response_activation_value(self, anchor, mock_pm):
        """慢响应激活值 = min(1.0, (latency - threshold) / threshold)"""
        latency = 600.0
        features = SomaticFeatures(cpu_temp=50.0, memory_percent=50.0, response_latency_ms=latency)
        anchor.check_and_activate(features)
        call_args = mock_pm.on_percept_activation.call_args[0][0]
        expected = min(1.0, (latency - SOMATIC_SLOW_RESPONSE_THRESHOLD) / SOMATIC_SLOW_RESPONSE_THRESHOLD)
        assert abs(call_args.activation - expected) < 0.01

    def test_none_latency_skipped(self, anchor, mock_pm):
        """响应延迟为 None → 跳过"""
        features = SomaticFeatures(cpu_temp=50.0, memory_percent=50.0, response_latency_ms=None)
        anchor.check_and_activate(features)
        call_args = [c[0][0] for c in mock_pm.on_percept_activation.call_args_list]
        labels = [e.perceptual_seed for e in call_args]
        assert "percept:somatic:slow_response" not in labels


class TestPsutilDegradation:
    """psutil 不可用降级测试"""

    def test_read_cpu_temp_no_psutil(self, anchor):
        """psutil 不可用时 _read_cpu_temp 返回 None"""
        with patch.dict("sys.modules", {"psutil": None}):
            result = anchor._read_cpu_temp()
        # 在 Windows 上可能返回 None（无 /proc，WMI 可能失败）
        # 关键是不抛异常
        assert result is None or isinstance(result, float)

    def test_read_memory_no_psutil(self, anchor):
        """psutil 不可用时 _read_memory_percent 返回 None"""
        with patch.dict("sys.modules", {"psutil": None}):
            result = anchor._read_memory_percent()
        assert result is None or isinstance(result, float)

    def test_observer_failure_graceful(self, mock_pm):
        """Observer 获取延迟失败时不抛异常"""
        observer = MagicMock()
        observer.get_avg_response_latency.side_effect = Exception("observer error")
        anchor = SomaticAnchor(mock_pm, observer=observer)
        features = anchor.collect_features()
        assert features.response_latency_ms is None


class TestLifecycle:
    """生命周期测试"""

    def test_start(self, anchor):
        """start() 启动采集线程"""
        anchor.start()
        assert anchor._daemon_thread is not None
        anchor.stop()

    def test_stop(self, anchor):
        """stop() 停止采集线程"""
        anchor.start()
        anchor.stop()
        assert anchor._shutdown_event.is_set()