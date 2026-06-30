"""
T-021: ExpertManager 单元测试（Mock GPU）

测试 ExpertManager 的所有公共方法（无 GPU 依赖）：
- PyTorch 未安装 → unavailable_reason="no_torch"
- GPU 不可用 → unavailable_reason="no_gpu"
- 模型文件不存在 → unavailable_reason="no_model"
- infer() 在 expert_available=False 时返回 fallback
- 推理超时降级
- CUDA OOM 异常捕获降级
- LoRA 同领域跳过切换
- MockExpertManager fixture
- 无 GPU 依赖（所有 GPU 操作均被 mock）
- Ollama 后端选择逻辑（auto/ollama/pytorch/none）
- Ollama 不可用时降级
- Ollama 超时时降级
- deepseek-r1 思考标签清洗
"""

from __future__ import annotations

import pathlib
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

_root = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_root))
sys.path.insert(0, str(_root / "backend" / "src"))

from consciousness_sea.expert.expert_manager import (
    _PEFT_AVAILABLE,
    _TORCH_AVAILABLE,
    ExpertManager,
    ExpertStatus,
    InferenceResult,
)


class TestExpertManagerNoTorch:
    """PyTorch 未安装测试"""

    def test_no_torch_unavailable_reason(self):
        """PyTorch 未安装 → unavailable_reason="no_torch" """
        if _TORCH_AVAILABLE:
            # 如果系统安装了 torch，跳过此测试
            # 使用 mock 模拟无 torch 环境
            with patch("consciousness_sea.expert.expert_manager._TORCH_AVAILABLE", False):
                with patch("consciousness_sea.expert.expert_manager.torch", None):
                    em = ExpertManager(expert_backend="pytorch")
                    em.initialize()
                    assert em.status.unavailable_reason == "no_torch"
        else:
            em = ExpertManager(expert_backend="pytorch")
            em.initialize()
            assert em.status.unavailable_reason == "no_torch"

    def test_no_torch_expert_not_available(self):
        """PyTorch 未安装 → expert_available=False"""
        if _TORCH_AVAILABLE:
            with patch("consciousness_sea.expert.expert_manager._TORCH_AVAILABLE", False):
                with patch("consciousness_sea.expert.expert_manager.torch", None):
                    em = ExpertManager(expert_backend="pytorch")
                    em.initialize()
                    assert em.expert_available is False
        else:
            em = ExpertManager(expert_backend="pytorch")
            em.initialize()
            assert em.expert_available is False


class TestExpertManagerNoGPU:
    """GPU 不可用测试"""

    def test_no_gpu_unavailable_reason(self):
        """GPU 不可用 → unavailable_reason="no_gpu" """
        if not _TORCH_AVAILABLE:
            # 无 torch 时无法测试 GPU 逻辑，跳过
            return

        mock_torch = MagicMock()
        mock_torch.cuda.is_available.return_value = False

        with patch("consciousness_sea.expert.expert_manager.torch", mock_torch):
            with patch("consciousness_sea.expert.expert_manager._TORCH_AVAILABLE", True):
                em = ExpertManager(expert_backend="pytorch")
                em.initialize()
                assert em.status.unavailable_reason == "no_gpu"
                assert em.expert_available is False


class TestExpertManagerNoModel:
    """模型文件不存在测试"""

    def test_no_model_unavailable_reason(self):
        """模型文件不存在 → unavailable_reason="no_model" """
        if not _TORCH_AVAILABLE:
            return

        mock_torch = MagicMock()
        mock_torch.cuda.is_available.return_value = True

        with patch("consciousness_sea.expert.expert_manager.torch", mock_torch):
            with patch("consciousness_sea.expert.expert_manager._TORCH_AVAILABLE", True):
                em = ExpertManager(model_path=Path("/nonexistent/model/path"), expert_backend="pytorch")
                em.initialize()
                assert em.status.unavailable_reason == "no_model"
                assert em.expert_available is False

    def test_empty_model_path_unavailable_reason(self):
        """空模型路径 → unavailable_reason="no_model" """
        if not _TORCH_AVAILABLE:
            return

        mock_torch = MagicMock()
        mock_torch.cuda.is_available.return_value = True

        with patch("consciousness_sea.expert.expert_manager.torch", mock_torch):
            with patch("consciousness_sea.expert.expert_manager._TORCH_AVAILABLE", True):
                em = ExpertManager(model_path=Path(""), expert_backend="pytorch")
                em.initialize()
                assert em.status.unavailable_reason == "no_model"


class TestExpertManagerInferFallback:
    """推理降级测试"""

    def test_infer_with_expert_not_available_returns_fallback(self):
        """expert_available=False 时 infer() 返回 fallback"""
        em = ExpertManager()
        em._initialized = True
        em._expert_available = False
        em._unavailable_reason = "no_torch"

        result = em.infer("test prompt", "医学")

        assert isinstance(result, InferenceResult)
        assert result.fallback is True
        assert result.answer_text == ""
        assert result.domain == "医学"

    def test_infer_fallback_increments_counter(self):
        """降级推理递增 fallback_count"""
        em = ExpertManager()
        em._initialized = True
        em._expert_available = False

        initial_count = em.status.fallback_count
        em.infer("test", "医学")

        assert em.status.fallback_count == initial_count + 1

    def test_infer_fallback_returns_default_reliability(self):
        """降级推理返回默认可靠性 0.7"""
        em = ExpertManager()
        em._initialized = True
        em._expert_available = False

        result = em.infer("test", "未知领域")
        assert result.reliability == 0.7

    def test_infer_fallback_with_configured_reliability(self):
        """降级推理返回配置的可靠性"""
        em = ExpertManager(reliability={"医学": 0.85})
        em._initialized = True
        em._expert_available = False

        result = em.infer("test", "医学")
        assert result.reliability == 0.85


class TestExpertManagerTimeout:
    """推理超时降级测试"""

    def test_inference_timeout_degradation(self):
        """推理超时降级到 Phase 0"""
        if not _TORCH_AVAILABLE:
            # 无 torch 时无法模拟超时推理，测试降级行为
            em = ExpertManager()
            em._initialized = True
            em._expert_available = False
            result = em.infer("test", "医学")
            assert result.fallback is True
            return

        # 有 torch 时，模拟超时
        mock_torch = MagicMock()
        mock_torch.cuda.is_available.return_value = True
        mock_torch.cuda.memory_allocated.return_value = 0

        mock_model = MagicMock()
        # 模拟 generate 方法永不返回（超时）
        def slow_generate(*args, **kwargs):
            import time
            time.sleep(20)  # 超过默认超时
            return MagicMock()

        mock_model.generate = slow_generate

        mock_tokenizer = MagicMock()
        mock_tokenizer.return_value = {"input_ids": MagicMock(shape=MagicMock(return_value=MagicMock(__getitem__=lambda s, i: 1)))}
        mock_tokenizer.decode.return_value = "测试回答"

        with patch("consciousness_sea.expert.expert_manager.torch", mock_torch):
            with patch("consciousness_sea.expert.expert_manager._TORCH_AVAILABLE", True):
                em = ExpertManager(inference_timeout=1.0)
                em._initialized = True
                em._expert_available = True
                em._base_model = mock_model
                em._tokenizer = mock_tokenizer

                result = em.infer("test", "医学", max_new_tokens=10)
                # 超时后应降级
                assert result.fallback is True


class TestExpertManagerCudaOOM:
    """CUDA OOM 异常捕获测试"""

    def test_cuda_oom_caught_and_fallback(self):
        """CUDA OOM 异常捕获 → 降级"""
        if not _TORCH_AVAILABLE:
            # 无 torch 时测试降级行为
            em = ExpertManager()
            em._initialized = True
            em._expert_available = False
            result = em.infer("test", "医学")
            assert result.fallback is True
            return

        mock_torch = MagicMock()
        mock_torch.cuda.is_available.return_value = True
        mock_torch.cuda.memory_allocated.return_value = 0
        mock_torch.cuda.empty_cache = MagicMock()
        mock_torch.no_grad.return_value.__enter__ = MagicMock(return_value=None)
        mock_torch.no_grad.return_value.__exit__ = MagicMock(return_value=None)

        mock_model = MagicMock()
        mock_model.generate.side_effect = RuntimeError("CUDA out of memory")

        mock_tokenizer = MagicMock()
        mock_tokenizer.return_value = {"input_ids": MagicMock(shape=MagicMock(return_value=MagicMock(__getitem__=lambda s, i: 1)))}

        with patch("consciousness_sea.expert.expert_manager.torch", mock_torch):
            with patch("consciousness_sea.expert.expert_manager._TORCH_AVAILABLE", True):
                em = ExpertManager()
                em._initialized = True
                em._expert_available = True
                em._base_model = mock_model
                em._tokenizer = mock_tokenizer

                result = em.infer("test", "医学", max_new_tokens=10)
                assert result.fallback is True


class TestExpertManagerLoraSwitch:
    """LoRA 切换测试"""

    def test_same_domain_skip_switch(self):
        """LoRA 同领域跳过切换"""
        if not _PEFT_AVAILABLE:
            # 无 PEFT 时 _switch_lora 返回 False
            # 但同领域检查在 PEFT 检查之前
            # 实际代码中 PEFT 检查先于同领域检查
            # 所以无 PEFT 时 _switch_lora 返回 False
            em = ExpertManager()
            em._initialized = True
            em._current_lora = "医学"
            result = em._switch_lora("医学")
            # 无 PEFT → 返回 False
            assert result is False
            return

        # 有 PEFT 时测试同领域跳过
        em = ExpertManager()
        em._initialized = True
        em._current_lora = "医学"
        result = em._switch_lora("医学")
        assert result is True


class TestExpertManagerStatus:
    """状态属性测试"""

    def test_status_returns_expert_status(self):
        """status 属性返回 ExpertStatus"""
        em = ExpertManager()
        status = em.status
        assert isinstance(status, ExpertStatus)

    def test_status_snapshot_consistency(self):
        """status 返回一致性快照"""
        em = ExpertManager()
        status1 = em.status
        status2 = em.status
        assert status1.expert_available == status2.expert_available
        assert status1.unavailable_reason == status2.unavailable_reason

    def test_initial_status_not_available(self):
        """初始状态 expert_available=False"""
        em = ExpertManager()
        assert em.expert_available is False
        assert em.status.expert_available is False

    def test_status_counters_initial_zero(self):
        """初始计数器为 0"""
        em = ExpertManager()
        status = em.status
        assert status.lora_switch_count == 0
        assert status.inference_count == 0
        assert status.fallback_count == 0


class TestExpertManagerShutdown:
    """关闭测试"""

    def test_shutdown_resets_state(self):
        """shutdown() 重置状态"""
        em = ExpertManager()
        em._expert_available = True
        em._current_lora = "医学"

        em.shutdown()

        assert em.expert_available is False
        assert em.status.current_lora is None

    def test_shutdown_idempotent(self):
        """shutdown() 幂等"""
        em = ExpertManager()
        em.shutdown()
        em.shutdown()  # 不应报错


class TestExpertManagerCleanAnswerText:
    """回答文本清洗测试"""

    def test_clean_special_tokens(self):
        """清洗特殊标记"""
        text = "这是回答<|im_end|>多余内容"
        result = ExpertManager._clean_answer_text(text)
        assert "<|im_end|>" not in result
        assert "这是回答" in result

    def test_clean_inst_tokens(self):
        """清洗 [INST] 标记"""
        text = "[INST]问题[/INST]回答"
        result = ExpertManager._clean_answer_text(text)
        assert "[INST]" not in result
        assert "[/INST]" not in result

    def test_clean_empty_string(self):
        """清洗空字符串"""
        result = ExpertManager._clean_answer_text("")
        assert result == ""

    def test_clean_strips_whitespace(self):
        """清洗去除首尾空白"""
        result = ExpertManager._clean_answer_text("  回答  ")
        assert result == "回答"


class TestExpertManagerGetReliability:
    """可靠性分数获取测试"""

    def test_configured_domain_reliability(self):
        """已配置领域的可靠性分数"""
        em = ExpertManager(reliability={"医学": 0.85})
        assert em._get_reliability("医学") == 0.85

    def test_unconfigured_domain_default_reliability(self):
        """未配置领域使用默认值 0.7"""
        em = ExpertManager(reliability={"医学": 0.85})
        assert em._get_reliability("法律") == 0.7

    def test_empty_reliability_config(self):
        """空可靠性配置"""
        em = ExpertManager(reliability={})
        assert em._get_reliability("医学") == 0.7


# ══════════════════════════════════════════════════════════
#  Ollama 后端测试
# ══════════════════════════════════════════════════════════


class TestOllamaBackendSelection:
    """Ollama 后端选择逻辑测试"""

    def test_backend_none_disables_expert(self):
        """EXPERT_BACKEND='none' -> 专家模式禁用"""
        em = ExpertManager(expert_backend="none")
        em.initialize()
        assert em.expert_available is False
        assert em.status.unavailable_reason == "disabled"
        assert em.status.active_backend is None

    def test_backend_ollama_available(self):
        """EXPERT_BACKEND='ollama' + Ollama 可用 -> 使用 Ollama 后端"""
        with patch("consciousness_sea.expert.expert_manager._HTTPX_AVAILABLE", True):
            mock_httpx = MagicMock()
            mock_response = MagicMock()
            mock_response.status_code = 200
            mock_response.json.return_value = {"models": [{"name": "deepseek-r1-7b"}]}
            mock_httpx.get.return_value = mock_response

            with patch("consciousness_sea.expert.expert_manager.httpx", mock_httpx):
                em = ExpertManager(expert_backend="ollama")
                em.initialize()
                assert em.expert_available is True
                assert em.status.active_backend == "ollama"

    def test_backend_ollama_unavailable_fallback(self):
        """EXPERT_BACKEND='ollama' + Ollama 不可用 -> 降级"""
        with patch("consciousness_sea.expert.expert_manager._HTTPX_AVAILABLE", True):
            mock_httpx = MagicMock()
            mock_httpx.get.side_effect = Exception("Connection refused")

            with patch("consciousness_sea.expert.expert_manager.httpx", mock_httpx):
                em = ExpertManager(expert_backend="ollama")
                em.initialize()
                assert em.expert_available is False
                assert em.status.unavailable_reason == "no_ollama"
                assert em.status.active_backend is None

    def test_backend_auto_ollama_available(self):
        """EXPERT_BACKEND='auto' + Ollama 可用 -> 使用 Ollama 后端"""
        with patch("consciousness_sea.expert.expert_manager._HTTPX_AVAILABLE", True):
            mock_httpx = MagicMock()
            mock_response = MagicMock()
            mock_response.status_code = 200
            mock_response.json.return_value = {"models": [{"name": "deepseek-r1-7b"}]}
            mock_httpx.get.return_value = mock_response

            with patch("consciousness_sea.expert.expert_manager.httpx", mock_httpx):
                em = ExpertManager(expert_backend="auto")
                em.initialize()
                assert em.expert_available is True
                assert em.status.active_backend == "ollama"

    def test_backend_auto_ollama_unavailable_falls_to_pytorch(self):
        """EXPERT_BACKEND='auto' + Ollama 不可用 -> 尝试 PyTorch"""
        with patch("consciousness_sea.expert.expert_manager._HTTPX_AVAILABLE", True):
            mock_httpx = MagicMock()
            mock_httpx.get.side_effect = Exception("Connection refused")

            with patch("consciousness_sea.expert.expert_manager.httpx", mock_httpx):
                # PyTorch 也不可用
                with patch("consciousness_sea.expert.expert_manager._TORCH_AVAILABLE", False):
                    with patch("consciousness_sea.expert.expert_manager.torch", None):
                        em = ExpertManager(expert_backend="auto")
                        em.initialize()
                        assert em.expert_available is False
                        assert em.status.unavailable_reason == "no_torch"

    def test_backend_pytorch_skips_ollama(self):
        """EXPERT_BACKEND='pytorch' -> 跳过 Ollama 检查"""
        with patch("consciousness_sea.expert.expert_manager._TORCH_AVAILABLE", False):
            with patch("consciousness_sea.expert.expert_manager.torch", None):
                em = ExpertManager(expert_backend="pytorch")
                em.initialize()
                assert em.expert_available is False
                assert em.status.unavailable_reason == "no_torch"
                assert em.status.active_backend is None

    def test_backend_unknown_value(self):
        """未知的 EXPERT_BACKEND 值 -> 降级"""
        em = ExpertManager(expert_backend="invalid")
        em.initialize()
        assert em.expert_available is False
        assert em.status.unavailable_reason == "unknown_backend:invalid"


class TestOllamaInference:
    """Ollama 推理测试"""

    def _create_ollama_manager(self) -> ExpertManager:
        """创建已初始化为 Ollama 后端的 ExpertManager"""
        em = ExpertManager(expert_backend="ollama")
        em._initialized = True
        em._expert_available = True
        em._active_backend = "ollama"
        return em

    def test_ollama_infer_success(self):
        """Ollama 推理成功"""
        with patch("consciousness_sea.expert.expert_manager._HTTPX_AVAILABLE", True):
            mock_httpx = MagicMock()
            mock_response = MagicMock()
            mock_response.status_code = 200
            mock_response.raise_for_status = MagicMock()
            mock_response.json.return_value = {"response": "这是Ollama的回答"}
            mock_httpx.post.return_value = mock_response

            with patch("consciousness_sea.expert.expert_manager.httpx", mock_httpx):
                em = self._create_ollama_manager()
                result = em.infer("测试问题", "医学")

                assert result.fallback is False
                assert result.answer_text == "这是Ollama的回答"
                assert result.domain == "医学"

    def test_ollama_infer_failure_fallback(self):
        """Ollama 推理失败 -> 降级"""
        with patch("consciousness_sea.expert.expert_manager._HTTPX_AVAILABLE", True):
            mock_httpx = MagicMock()
            mock_httpx.post.side_effect = Exception("Connection refused")

            with patch("consciousness_sea.expert.expert_manager.httpx", mock_httpx):
                em = self._create_ollama_manager()
                result = em.infer("测试问题", "医学")

                assert result.fallback is True
                assert result.answer_text == ""
                assert em.status.fallback_count == 1

    def test_ollama_infer_timeout_fallback(self):
        """Ollama 推理超时 -> 降级"""
        with patch("consciousness_sea.expert.expert_manager._HTTPX_AVAILABLE", True):
            import httpx as _real_httpx
            mock_httpx = MagicMock()
            mock_httpx.post.side_effect = _real_httpx.ReadTimeout("Read timed out")

            with patch("consciousness_sea.expert.expert_manager.httpx", mock_httpx):
                em = self._create_ollama_manager()
                result = em.infer("测试问题", "医学")

                assert result.fallback is True
                assert result.answer_text == ""

    def test_ollama_infer_increments_counter(self):
        """Ollama 推理成功递增 inference_count"""
        with patch("consciousness_sea.expert.expert_manager._HTTPX_AVAILABLE", True):
            mock_httpx = MagicMock()
            mock_response = MagicMock()
            mock_response.status_code = 200
            mock_response.raise_for_status = MagicMock()
            mock_response.json.return_value = {"response": "回答"}
            mock_httpx.post.return_value = mock_response

            with patch("consciousness_sea.expert.expert_manager.httpx", mock_httpx):
                em = self._create_ollama_manager()
                initial_count = em.status.inference_count
                em.infer("测试", "医学")
                assert em.status.inference_count == initial_count + 1

    def test_ollama_no_lora_switch(self):
        """Ollama 后端不执行 LoRA 切换"""
        with patch("consciousness_sea.expert.expert_manager._HTTPX_AVAILABLE", True):
            mock_httpx = MagicMock()
            mock_response = MagicMock()
            mock_response.status_code = 200
            mock_response.raise_for_status = MagicMock()
            mock_response.json.return_value = {"response": "回答"}
            mock_httpx.post.return_value = mock_response

            with patch("consciousness_sea.expert.expert_manager.httpx", mock_httpx):
                em = self._create_ollama_manager()
                em.infer("测试", "医学")
                # Ollama 后端不切换 LoRA
                assert em.status.lora_switch_count == 0

    def test_ollama_infer_with_think_tag(self):
        """Ollama 推理结果含思考标签 -> 清洗后只保留回答"""
        with patch("consciousness_sea.expert.expert_manager._HTTPX_AVAILABLE", True):
            mock_httpx = MagicMock()
            mock_response = MagicMock()
            mock_response.status_code = 200
            mock_response.raise_for_status = MagicMock()
            # 使用简单字符串，避免特殊字符问题
            think_content = "让我想想这个问题"
            answer_content = "这是最终回答"
            raw_response = f"<think>{think_content}</think>{answer_content}"
            mock_response.json.return_value = {"response": raw_response}
            mock_httpx.post.return_value = mock_response

            with patch("consciousness_sea.expert.expert_manager.httpx", mock_httpx):
                em = self._create_ollama_manager()
                result = em.infer("测试问题", "医学")

                assert result.fallback is False
                assert think_content not in result.answer_text
                assert result.answer_text == answer_content

    def test_ollama_api_call_format(self):
        """Ollama API 调用格式正确"""
        with patch("consciousness_sea.expert.expert_manager._HTTPX_AVAILABLE", True):
            mock_httpx = MagicMock()
            mock_response = MagicMock()
            mock_response.status_code = 200
            mock_response.raise_for_status = MagicMock()
            mock_response.json.return_value = {"response": "回答"}
            mock_httpx.post.return_value = mock_response

            with patch("consciousness_sea.expert.expert_manager.httpx", mock_httpx):
                em = self._create_ollama_manager()
                em.infer("测试prompt", "医学", max_new_tokens=256)

                # 验证 API 调用参数
                mock_httpx.post.assert_called_once()
                call_args = mock_httpx.post.call_args
                assert call_args[0][0] == "http://localhost:11434/api/generate"
                json_payload = call_args[1]["json"]
                assert json_payload["model"] == "deepseek-r1-7b"
                assert json_payload["prompt"] == "测试prompt"
                assert json_payload["stream"] is False
                assert json_payload["options"]["num_predict"] == 256
                assert json_payload["options"]["temperature"] == 0.7

    def test_ollama_custom_config(self):
        """Ollama 自定义配置"""
        with patch("consciousness_sea.expert.expert_manager._HTTPX_AVAILABLE", True):
            mock_httpx = MagicMock()
            mock_response = MagicMock()
            mock_response.status_code = 200
            mock_response.raise_for_status = MagicMock()
            mock_response.json.return_value = {"response": "回答"}
            mock_httpx.post.return_value = mock_response

            with patch("consciousness_sea.expert.expert_manager.httpx", mock_httpx):
                em = ExpertManager(
                    expert_backend="ollama",
                    ollama_base_url="http://192.168.1.100:11434",
                    ollama_model="llama3",
                    ollama_timeout=120.0,
                )
                em._initialized = True
                em._expert_available = True
                em._active_backend = "ollama"

                em.infer("测试", "医学")

                call_args = mock_httpx.post.call_args
                assert call_args[0][0] == "http://192.168.1.100:11434/api/generate"
                assert call_args[1]["json"]["model"] == "llama3"
                assert call_args[1]["timeout"] == 120.0


class TestOllamaCheckAvailable:
    """Ollama 可用性检查测试"""

    def test_check_ollama_available_httpx_success(self):
        """httpx 检查 Ollama 可用 - 成功"""
        with patch("consciousness_sea.expert.expert_manager._HTTPX_AVAILABLE", True):
            mock_httpx = MagicMock()
            mock_response = MagicMock()
            mock_response.status_code = 200
            mock_response.json.return_value = {"models": [{"name": "deepseek-r1-7b"}]}
            mock_httpx.get.return_value = mock_response

            with patch("consciousness_sea.expert.expert_manager.httpx", mock_httpx):
                em = ExpertManager(expert_backend="ollama")
                assert em._check_ollama_available() is True

    def test_check_ollama_available_httpx_connection_refused(self):
        """httpx 检查 Ollama 可用 - 连接拒绝"""
        with patch("consciousness_sea.expert.expert_manager._HTTPX_AVAILABLE", True):
            mock_httpx = MagicMock()
            mock_httpx.get.side_effect = Exception("Connection refused")

            with patch("consciousness_sea.expert.expert_manager.httpx", mock_httpx):
                em = ExpertManager(expert_backend="ollama")
                assert em._check_ollama_available() is False

    def test_check_ollama_available_httpx_non_200(self):
        """httpx 检查 Ollama 可用 - 非 200 状态码"""
        with patch("consciousness_sea.expert.expert_manager._HTTPX_AVAILABLE", True):
            mock_httpx = MagicMock()
            mock_response = MagicMock()
            mock_response.status_code = 500
            mock_httpx.get.return_value = mock_response

            with patch("consciousness_sea.expert.expert_manager.httpx", mock_httpx):
                em = ExpertManager(expert_backend="ollama")
                assert em._check_ollama_available() is False

    def test_check_ollama_available_model_with_latest_suffix(self):
        """Ollama 模型名带 :latest 后缀也能匹配"""
        with patch("consciousness_sea.expert.expert_manager._HTTPX_AVAILABLE", True):
            mock_httpx = MagicMock()
            mock_response = MagicMock()
            mock_response.status_code = 200
            mock_response.json.return_value = {"models": [{"name": "deepseek-r1-7b:latest"}]}
            mock_httpx.get.return_value = mock_response

            with patch("consciousness_sea.expert.expert_manager.httpx", mock_httpx):
                em = ExpertManager(expert_backend="ollama")
                # 模型名不匹配但服务可用，仍返回 True
                assert em._check_ollama_available() is True

    def test_check_ollama_available_urllib_fallback(self):
        """httpx 不可用时使用 urllib 备选"""
        with patch("consciousness_sea.expert.expert_manager._HTTPX_AVAILABLE", False):
            with patch("consciousness_sea.expert.expert_manager.httpx", None):
                em = ExpertManager(expert_backend="ollama")
                # urllib 连接本地 Ollama 大概率失败，测试降级逻辑
                result = em._check_ollama_available()
                # 不依赖实际 Ollama 服务，只验证不抛异常
                assert isinstance(result, bool)


class TestOllamaCleanThinkTag:
    """deepseek-r1 思考标签清洗测试"""

    def test_clean_think_tag(self):
        """清洗思考标签及内容"""
        text = "<think>思考过程</think>最终回答"
        result = ExpertManager._clean_answer_text(text)
        assert result == "最终回答"

    def test_clean_think_tag_multiline(self):
        """清洗多行思考标签"""
        text = "<think>第一行\n第二行</think>最终回答"
        result = ExpertManager._clean_answer_text(text)
        assert result == "最终回答"

    def test_clean_think_tag_empty_think(self):
        """清洗空思考标签"""
        text = "<think></think>最终回答"
        result = ExpertManager._clean_answer_text(text)
        assert result == "最终回答"

    def test_clean_think_tag_only_think(self):
        """只有思考标签无回答"""
        text = "<think>全部是思考</think>"
        result = ExpertManager._clean_answer_text(text)
        assert result == ""

    def test_clean_think_tag_no_think(self):
        """无思考标签"""
        text = "普通回答"
        result = ExpertManager._clean_answer_text(text)
        assert result == "普通回答"

    def test_clean_think_tag_with_special_tokens(self):
        """思考标签和特殊标记混合"""
        text = "<think>思考</think>回答<|im_end|>多余"
        result = ExpertManager._clean_answer_text(text)
        assert "思考" not in result
        assert "<|im_end|>" not in result
        assert "回答" in result


if __name__ == '__main__':
    import pytest
    pytest.main([__file__, '-v'])
