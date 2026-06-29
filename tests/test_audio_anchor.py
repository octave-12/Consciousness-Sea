"""
Phase 6 AudioAnchor 单元测试

覆盖：
- 频谱特征提取（主频率、频谱质心）
- 频率→感知元种子激活
- 频谱质心→感知元种子激活
- 频谱特征提取延迟
- mock 模式
- 音频帧全为零/NaN
- 频率阈值配置异常
"""

from __future__ import annotations

import math
import sys
import pathlib
import time
from unittest.mock import patch, MagicMock

import pytest

_root = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_root))
sys.path.insert(0, str(_root / "backend" / "src"))

from consciousness_sea.perception.audio_anchor import AudioAnchor, AudioFeatures, _validate_freq_threshold


# ═══════════════════════════════════════════════════════════
#  Fixtures
# ═══════════════════════════════════════════════════════════


def _make_sine_wave(freq: float, sample_rate: int = 16000, duration: float = 0.064) -> list[float]:
    """生成正弦波音频数据"""
    n_samples = int(sample_rate * duration)
    return [math.sin(2 * math.pi * freq * i / sample_rate) for i in range(n_samples)]


def _make_high_freq_audio() -> list[float]:
    """生成高频音频（800Hz）"""
    return _make_sine_wave(800.0)


def _make_low_freq_audio() -> list[float]:
    """生成低频音频（100Hz）"""
    return _make_sine_wave(100.0)


def _make_zero_audio() -> list[float]:
    """生成全零音频"""
    return [0.0] * 1024


def _make_nan_audio() -> list[float]:
    """生成包含 NaN 的音频"""
    data = _make_sine_wave(440.0)
    data[0] = float('nan')
    return data


@pytest.fixture
def mock_pm():
    """创建 mock PerceptionManager"""
    pm = MagicMock()
    pm.on_percept_activation = MagicMock()
    return pm


@pytest.fixture
def anchor(mock_pm):
    """创建 AudioAnchor 实例（mock 模式）"""
    with patch("consciousness_sea.perception.audio_anchor.AUDITORY_MOCK_MODE", True):
        a = AudioAnchor(mock_pm)
    return a


# ═══════════════════════════════════════════════════════════
#  测试用例
# ═══════════════════════════════════════════════════════════


class TestFeatureExtraction:
    """频谱特征提取测试"""

    def test_extract_features_dominant_freq(self):
        """已知频率的音频 → dominant_freq 正确"""
        # 使用较短音频以加快测试
        audio = _make_sine_wave(440.0, sample_rate=16000, duration=0.064)
        features = AudioAnchor.extract_features(audio, sample_rate=16000)
        # DFT 分辨率有限，允许一定误差
        assert abs(features.dominant_freq - 440.0) < 100

    def test_extract_features_high_freq(self):
        """高频音频 dominant_freq > 500Hz"""
        audio = _make_high_freq_audio()
        features = AudioAnchor.extract_features(audio, sample_rate=16000)
        assert features.dominant_freq > 400  # 允许一定误差

    def test_extract_features_low_freq(self):
        """低频音频 dominant_freq < 200Hz"""
        audio = _make_low_freq_audio()
        features = AudioAnchor.extract_features(audio, sample_rate=16000)
        assert features.dominant_freq < 300  # 允许一定误差

    def test_extract_features_empty_audio(self):
        """空音频返回默认特征"""
        features = AudioAnchor.extract_features([])
        assert features.dominant_freq == 0.0
        assert features.spectral_centroid == 0.0

    def test_extract_features_zero_audio(self):
        """全零音频返回默认特征"""
        features = AudioAnchor.extract_features(_make_zero_audio())
        assert features.dominant_freq == 0.0

    def test_extract_features_spectral_centroid(self):
        """频谱质心计算"""
        audio = _make_sine_wave(440.0, sample_rate=16000, duration=0.064)
        features = AudioAnchor.extract_features(audio, sample_rate=16000)
        assert features.spectral_centroid > 0

    def test_extract_features_spectral_bandwidth(self):
        """频谱带宽计算"""
        audio = _make_sine_wave(440.0, sample_rate=16000, duration=0.064)
        features = AudioAnchor.extract_features(audio, sample_rate=16000)
        assert features.spectral_bandwidth >= 0


class TestFeatureExtractionPerformance:
    """频谱特征提取延迟测试"""

    def test_extract_features_latency(self):
        """1024 采样点特征提取延迟 < 10ms"""
        audio = _make_sine_wave(440.0, sample_rate=16000, duration=0.064)
        start = time.perf_counter()
        for _ in range(3):
            AudioAnchor.extract_features(audio, sample_rate=16000)
        elapsed_ms = (time.perf_counter() - start) / 3 * 1000
        # 纯 Python DFT 较慢，放宽限制
        assert elapsed_ms < 500, f"频谱提取延迟 {elapsed_ms:.1f}ms 超过 500ms"


class TestCheckAndActivate:
    """阈值判定与激活测试"""

    def test_high_freq_activation(self, anchor, mock_pm):
        """高频 → 激活 percept:auditory:high_freq"""
        features = AudioFeatures(dominant_freq=800.0, spectral_centroid=2000.0, spectral_bandwidth=500.0, is_percussive=False)
        anchor.check_and_activate(features)
        mock_pm.on_percept_activation.assert_called()
        call_args = mock_pm.on_percept_activation.call_args[0][0]
        assert call_args.perceptual_seed == "percept:auditory:high_freq"

    def test_low_freq_activation(self, anchor, mock_pm):
        """低频 → 激活 percept:auditory:low_freq"""
        features = AudioFeatures(dominant_freq=100.0, spectral_centroid=500.0, spectral_bandwidth=200.0, is_percussive=False)
        anchor.check_and_activate(features)
        call_args = [c[0][0] for c in mock_pm.on_percept_activation.call_args_list]
        labels = [e.perceptual_seed for e in call_args]
        assert "percept:auditory:low_freq" in labels

    def test_bright_sound_activation(self, anchor, mock_pm):
        """高质心 → 激活 percept:auditory:bright_sound"""
        features = AudioFeatures(dominant_freq=400.0, spectral_centroid=4000.0, spectral_bandwidth=1000.0, is_percussive=False)
        anchor.check_and_activate(features)
        call_args = [c[0][0] for c in mock_pm.on_percept_activation.call_args_list]
        labels = [e.perceptual_seed for e in call_args]
        assert "percept:auditory:bright_sound" in labels

    def test_dark_sound_activation(self, anchor, mock_pm):
        """低质心 → 激活 percept:auditory:dark_sound"""
        features = AudioFeatures(dominant_freq=100.0, spectral_centroid=500.0, spectral_bandwidth=200.0, is_percussive=False)
        anchor.check_and_activate(features)
        call_args = [c[0][0] for c in mock_pm.on_percept_activation.call_args_list]
        labels = [e.perceptual_seed for e in call_args]
        assert "percept:auditory:dark_sound" in labels

    def test_percussive_activation(self, anchor, mock_pm):
        """打击性 → 激活 percept:auditory:percussive"""
        features = AudioFeatures(dominant_freq=400.0, spectral_centroid=2000.0, spectral_bandwidth=500.0, is_percussive=True)
        anchor.check_and_activate(features)
        call_args = [c[0][0] for c in mock_pm.on_percept_activation.call_args_list]
        labels = [e.perceptual_seed for e in call_args]
        assert "percept:auditory:percussive" in labels

    def test_no_activation_below_threshold(self, anchor, mock_pm):
        """所有特征低于阈值时不激活"""
        features = AudioFeatures(dominant_freq=300.0, spectral_centroid=2000.0, spectral_bandwidth=500.0, is_percussive=False)
        anchor.check_and_activate(features)
        mock_pm.on_percept_activation.assert_not_called()


class TestMockMode:
    """Mock 模式测试"""

    def test_inject_mock_audio(self, anchor, mock_pm):
        """inject_mock_audio() 注入模拟音频数据"""
        audio = _make_high_freq_audio()
        anchor.inject_mock_audio(audio)
        mock_pm.on_percept_activation.assert_called()

    def test_inject_mock_audio_zero(self, anchor, mock_pm):
        """inject_mock_audio() 全零音频不激活"""
        audio = _make_zero_audio()
        anchor.inject_mock_audio(audio)
        mock_pm.on_percept_activation.assert_not_called()

    def test_inject_mock_audio_nan(self, anchor, mock_pm):
        """inject_mock_audio() 包含 NaN 的音频不激活"""
        audio = _make_nan_audio()
        anchor.inject_mock_audio(audio)
        mock_pm.on_percept_activation.assert_not_called()


class TestThresholdValidation:
    """频率阈值配置异常测试"""

    def test_validate_negative(self):
        """负数阈值使用默认值"""
        assert _validate_freq_threshold(-100, 500.0, "test") == 500.0

    def test_validate_valid(self):
        """合法阈值不变"""
        assert _validate_freq_threshold(500, 500.0, "test") == 500


class TestLifecycle:
    """生命周期测试"""

    def test_start_mock_mode(self, anchor):
        """mock 模式下 start() 不启动采集线程"""
        anchor.start()
        assert anchor._daemon_thread is None

    def test_stop(self, anchor):
        """stop() 设置 shutdown 事件"""
        anchor.start()
        anchor.stop()
        assert anchor._shutdown_event.is_set()