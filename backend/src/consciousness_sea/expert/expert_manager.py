"""
ExpertManager — 专家模型生命周期管理器

职责:
  - 基座模型懒加载 / 卸载
  - LoRA 适配器热切换
  - 推理调用（含超时保护）
  - VRAM 预算监控
  - 降级决策
  - Ollama HTTP 后端支持

线程安全:
  - _inference_lock 保护模型推理串行化
  - _state_lock 保护状态变量读写
  - _inference_lock 可嵌套持有 _state_lock（推理锁内获取状态锁），反向禁止

PyTorch 可选依赖:
  - 模块级 Import Guard 确保 PyTorch 未安装时模块可被 import
  - 所有 GPU 操作在 _TORCH_AVAILABLE 检查保护下
  - 无 GPU 时自动降级到 Phase 0

Ollama 可选依赖:
  - httpx 为 Ollama 后端的唯一额外依赖
  - httpx 不可用时自动降级到 PyTorch 后端或 Phase 0
  - Ollama 后端不需要 LoRA 切换——模型本身已包含领域知识
  - Ollama 后端不需要 VRAM 检查——显存由 Ollama 管理
"""

from __future__ import annotations

import logging
import re
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from .expert_reliability import DEFAULT_RELIABILITY, ExpertReliabilityStore

log = logging.getLogger(__name__)

# ══════════════════════════════════════════════════════════
#  Import Guard — PyTorch / PEFT 可选依赖检测
# ══════════════════════════════════════════════════════════

_TORCH_AVAILABLE = False
_PEFT_AVAILABLE = False

try:
    import torch
    _TORCH_AVAILABLE = True
except ImportError:
    torch = None  # type: ignore[assignment]

try:
    import peft
    _PEFT_AVAILABLE = True
except ImportError:
    peft = None  # type: ignore[assignment]

_HTTPX_AVAILABLE = False

try:
    import httpx
    _HTTPX_AVAILABLE = True
except ImportError:
    httpx = None  # type: ignore[assignment]


# ══════════════════════════════════════════════════════════
#  数据类
# ══════════════════════════════════════════════════════════

@dataclass
class ExpertStatus:
    """专家状态快照"""
    expert_available: bool = False
    current_lora: Optional[str] = None
    vram_usage_mb: Optional[float] = None
    reliability_scores: dict[str, float] = field(default_factory=dict)
    lora_switch_count: int = 0
    inference_count: int = 0
    fallback_count: int = 0
    unavailable_reason: Optional[str] = None  # "no_gpu" | "no_model" | "no_torch" | "load_failed" | "no_ollama" | "no_httpx"
    active_backend: Optional[str] = None  # "ollama" | "pytorch" | None


@dataclass
class InferenceResult:
    """单次推理结果"""
    answer_text: str
    domain: str
    reliability: float
    inference_time_ms: float
    fallback: bool = False  # 是否降级到 Phase 0


# ══════════════════════════════════════════════════════════
#  ExpertManager
# ══════════════════════════════════════════════════════════

class ExpertManager:
    """专家模型生命周期管理器

    职责:
      - 基座模型懒加载 / 卸载
      - LoRA 适配器热切换
      - 推理调用（含超时保护）
      - VRAM 预算监控
      - 降级决策
      - Ollama HTTP 后端支持

    线程安全:
      - _inference_lock 保护模型推理串行化
      - _state_lock 保护状态变量读写
      - _inference_lock 可嵌套持有 _state_lock（推理锁内获取状态锁），反向禁止

    Args:
        model_path: 基座模型路径 (pathlib.Path)
        lora_adapters: 领域→LoRA路径映射
        reliability: 领域→可靠性分数映射
        default_lora: 默认LoRA领域名
        max_vram_gb: VRAM预算上限(GB)
        inference_timeout: 推理超时(秒)
        ollama_base_url: Ollama API 地址
        ollama_model: Ollama 模型名
        ollama_timeout: Ollama 请求超时(秒)
        expert_backend: 专家后端选择: "auto" | "ollama" | "pytorch" | "none"
    """

    def __init__(
        self,
        model_path: Path = Path(""),
        lora_adapters: dict[str, Path] | None = None,
        reliability: dict[str, float] | None = None,
        default_lora: str | None = None,
        max_vram_gb: float = 5.5,
        inference_timeout: float = 10.0,
        reliability_store: ExpertReliabilityStore | None = None,
        ollama_base_url: str = "http://localhost:11434",
        ollama_model: str = "deepseek-r1-7b",
        ollama_timeout: float = 60.0,
        expert_backend: str = "auto",
    ) -> None:
        self._model_path = model_path
        self._lora_adapters: dict[str, Path] = lora_adapters or {}
        self._reliability: dict[str, float] = reliability or {}
        self._default_lora = default_lora
        self._max_vram_gb = max_vram_gb
        self._inference_timeout = inference_timeout
        self._reliability_store = reliability_store
        self._ollama_base_url = ollama_base_url
        self._ollama_model = ollama_model
        self._ollama_timeout = ollama_timeout
        self._expert_backend = expert_backend

        # ── 状态变量（由 _state_lock 保护）──
        self._expert_available: bool = False
        self._unavailable_reason: Optional[str] = None
        self._base_model: Optional[object] = None
        self._tokenizer: Optional[object] = None
        self._current_lora: Optional[str] = None
        self._lora_model: Optional[object] = None
        self._initialized: bool = False
        self._active_backend: Optional[str] = None  # "ollama" | "pytorch" | None

        # ── 统计计数器（由 _state_lock 保护）──
        self._lora_switch_count: int = 0
        self._inference_count: int = 0
        self._fallback_count: int = 0

        # ── 超时冷却（由 _state_lock 保护）──
        self._timeout_cooldown_until: float = 0.0

        # ── 线程锁 ──
        self._inference_lock = threading.Lock()  # 推理串行化锁
        self._state_lock = threading.Lock()       # 状态读写锁

    # ══════════════════════════════════════════════════════════
    #  公共属性
    # ══════════════════════════════════════════════════════════

    @property
    def expert_available(self) -> bool:
        """专家模式是否可用（线程安全读取）"""
        with self._state_lock:
            return self._expert_available

    @property
    def status(self) -> ExpertStatus:
        """获取当前专家状态快照（线程安全）

        返回一致性快照，所有字段在同一时刻读取。
        """
        with self._state_lock:
            vram_mb: Optional[float] = None
            if _TORCH_AVAILABLE and torch is not None and self._expert_available:
                try:
                    vram_mb = torch.cuda.memory_allocated() / (1024 * 1024)
                except (RuntimeError, OSError):
                    vram_mb = None

            return ExpertStatus(
                expert_available=self._expert_available,
                current_lora=self._current_lora,
                vram_usage_mb=vram_mb,
                reliability_scores=dict(self._reliability),
                lora_switch_count=self._lora_switch_count,
                inference_count=self._inference_count,
                fallback_count=self._fallback_count,
                unavailable_reason=self._unavailable_reason,
                active_backend=self._active_backend,
            )

    # ══════════════════════════════════════════════════════════
    #  初始化（懒加载）
    # ══════════════════════════════════════════════════════════

    def initialize(self) -> None:
        """尝试加载基座模型（懒加载入口）

        仅在首次推理请求时调用，不阻塞启动。
        加载失败时标记 expert_available=False，不抛异常。
        """
        with self._state_lock:
            if self._initialized:
                return
            self._initialized = True

        # 在锁外执行耗时的模型加载
        self._load_base_model()

    # ══════════════════════════════════════════════════════════
    #  基座模型加载与卸载 (T-004)
    # ══════════════════════════════════════════════════════════

    def _load_base_model(self) -> bool:
        """加载基座模型到 VRAM（或连接 Ollama 后端）

        后端选择逻辑:
          EXPERT_BACKEND = "auto":
            1. 先检查 Ollama 是否可用（GET /api/tags）
            2. Ollama 可用 → 使用 Ollama 后端
            3. Ollama 不可用 → 检查 PyTorch + GPU
            4. PyTorch 可用 → 使用 PyTorch 后端
            5. 都不可用 → Phase 0 降级

          EXPERT_BACKEND = "ollama":
            只尝试 Ollama，不可用则降级

          EXPERT_BACKEND = "pytorch":
            只尝试 PyTorch（现有逻辑），不可用则降级

          EXPERT_BACKEND = "none":
            直接 Phase 0 降级

        Returns:
            True 加载成功, False 加载失败
        """
        backend = self._expert_backend

        # ── "none" → 直接降级 ──
        if backend == "none":
            with self._state_lock:
                self._expert_available = False
                self._unavailable_reason = "disabled"
                self._active_backend = None
            log.info("EXPERT_BACKEND='none'，专家模式已禁用")
            return False

        # ── "auto" 或 "ollama" → 先尝试 Ollama ──
        if backend in ("auto", "ollama"):
            ollama_ok = self._check_ollama_available()
            if ollama_ok:
                with self._state_lock:
                    self._expert_available = True
                    self._unavailable_reason = None
                    self._active_backend = "ollama"
                log.info(
                    "Ollama 后端已连接: base_url=%s, model=%s",
                    self._ollama_base_url, self._ollama_model,
                )
                return True
            else:
                if backend == "ollama":
                    # 只尝试 Ollama，不可用则降级
                    with self._state_lock:
                        self._expert_available = False
                        self._unavailable_reason = "no_ollama"
                        self._active_backend = None
                    log.warning("Ollama 不可用且 EXPERT_BACKEND='ollama'，专家模式不可用")
                    return False
                # "auto" → 继续尝试 PyTorch
                log.info("Ollama 不可用，尝试 PyTorch 后端...")

        # ── "auto" 或 "pytorch" → 尝试 PyTorch ──
        if backend in ("auto", "pytorch"):
            return self._load_pytorch_model()

        # ── 未知后端值 ──
        with self._state_lock:
            self._expert_available = False
            self._unavailable_reason = f"unknown_backend:{backend}"
            self._active_backend = None
        log.warning("未知的 EXPERT_BACKEND 值: %s", backend)
        return False

    def _check_ollama_available(self) -> bool:
        """检查 Ollama 服务是否可用

        通过 GET /api/tags 探测 Ollama 服务。
        优先使用 httpx，httpx 不可用时使用 urllib.request 作为备选。

        Returns:
            True Ollama 可用, False Ollama 不可用
        """
        # ── 检查 httpx 可用性 ──
        if _HTTPX_AVAILABLE and httpx is not None:
            try:
                response = httpx.get(
                    f"{self._ollama_base_url}/api/tags",
                    timeout=5.0,
                )
                if response.status_code == 200:
                    # 检查模型是否存在
                    data = response.json()
                    models = [m.get("name", "") for m in data.get("models", [])]
                    # Ollama 模型名可能带 :latest 后缀
                    model_found = any(
                        m == self._ollama_model or m.startswith(f"{self._ollama_model}:")
                        for m in models
                    )
                    if not model_found:
                        log.warning(
                            "Ollama 服务可用但模型 '%s' 不存在，可用模型: %s",
                            self._ollama_model, models,
                        )
                        # 模型不存在仍然返回 True，Ollama 会在推理时自动拉取
                    return True
                return False
            except Exception as e:
                log.debug("Ollama 连接检查失败 (httpx): %s", e)
                return False

        # ── httpx 不可用，使用 urllib.request 备选 ──
        try:
            import json as _json
            import urllib.request

            req = urllib.request.Request(
                f"{self._ollama_base_url}/api/tags",
                method="GET",
            )
            with urllib.request.urlopen(req, timeout=5.0) as resp:
                if resp.status == 200:
                    data = _json.loads(resp.read().decode("utf-8"))
                    models = [m.get("name", "") for m in data.get("models", [])]
                    model_found = any(
                        m == self._ollama_model or m.startswith(f"{self._ollama_model}:")
                        for m in models
                    )
                    if not model_found:
                        log.warning(
                            "Ollama 服务可用但模型 '%s' 不存在，可用模型: %s",
                            self._ollama_model, models,
                        )
                    return True
            return False
        except Exception as e:
            log.debug("Ollama 连接检查失败 (urllib): %s", e)
            return False

    def _load_pytorch_model(self) -> bool:
        """加载 PyTorch 基座模型到 VRAM

        Returns:
            True 加载成功, False 加载失败
        """
        # ── 检查 PyTorch 可用性 ──
        if not _TORCH_AVAILABLE or torch is None:
            with self._state_lock:
                self._expert_available = False
                self._unavailable_reason = "no_torch"
                self._active_backend = None
            log.warning("PyTorch 未安装，专家模式不可用")
            return False

        # ── 检查 GPU 可用性 ──
        try:
            if not torch.cuda.is_available():
                with self._state_lock:
                    self._expert_available = False
                    self._unavailable_reason = "no_gpu"
                    self._active_backend = None
                log.warning("GPU 不可用，专家模式不可用")
                return False
        except (RuntimeError, OSError) as e:
            with self._state_lock:
                self._expert_available = False
                self._unavailable_reason = "no_gpu"
                self._active_backend = None
            log.warning("GPU 检测异常: %s，专家模式不可用", e)
            return False

        # ── 检查模型文件存在 ──
        if not self._model_path or str(self._model_path) == "":
            with self._state_lock:
                self._expert_available = False
                self._unavailable_reason = "no_model"
                self._active_backend = None
            log.warning("基座模型路径为空，专家模式不可用")
            return False

        model_path = self._model_path
        if not model_path.exists():
            with self._state_lock:
                self._expert_available = False
                self._unavailable_reason = "no_model"
                self._active_backend = None
            log.warning("基座模型文件不存在: %s", model_path)
            return False

        # ── 加载模型 ──
        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer

            log.info("正在加载基座模型: %s ...", model_path)
            start_time = time.monotonic()

            tokenizer = AutoTokenizer.from_pretrained(
                str(model_path), trust_remote_code=True
            )
            model = AutoModelForCausalLM.from_pretrained(
                str(model_path),
                device_map="auto",
                torch_dtype=torch.float16,
                trust_remote_code=True,
            )

            duration = time.monotonic() - start_time
            log.info("基座模型加载完成，耗时 %.1fs", duration)

            with self._state_lock:
                self._base_model = model
                self._tokenizer = tokenizer
                self._expert_available = True
                self._unavailable_reason = None
                self._active_backend = "pytorch"

            return True

        except Exception as e:
            log.error("基座模型加载失败: %s", e, exc_info=True)
            with self._state_lock:
                self._expert_available = False
                self._unavailable_reason = "load_failed"
                self._base_model = None
                self._tokenizer = None
                self._active_backend = None
            return False

    def shutdown(self) -> None:
        """卸载基座模型和当前 LoRA，释放 VRAM"""
        with self._state_lock:
            base_model = self._base_model
            lora_model = self._lora_model
            tokenizer = self._tokenizer
            self._base_model = None
            self._tokenizer = None
            self._lora_model = None
            self._current_lora = None
            self._expert_available = False
            self._unavailable_reason = None
            self._active_backend = None

        with self._inference_lock:
            # 同时持有 _inference_lock 防止推理期间释放资源
            if lora_model is not None:
                try:
                    del lora_model
                except Exception:
                    pass

            if base_model is not None:
                try:
                    del base_model
                except Exception:
                    pass

            if tokenizer is not None:
                try:
                    del tokenizer
                except Exception:
                    pass

            # 释放 GPU 显存
            if _TORCH_AVAILABLE and torch is not None:
                try:
                    torch.cuda.empty_cache()
                    log.info("基座模型已卸载，VRAM 已释放")
                except (RuntimeError, OSError) as e:
                    log.warning("释放 VRAM 时异常: %s", e)

    # ══════════════════════════════════════════════════════════
    #  LoRA 热切换 (T-005)
    # ══════════════════════════════════════════════════════════

    def _switch_lora(self, target_domain: str) -> bool:
        """LoRA 热切换

        流程:
          1. 若 target_domain == current_lora → 跳过
          2. 目标 LoRA 路径不存在时回退到默认 LoRA
          3. 卸载当前 LoRA
          4. 加载目标 LoRA
          5. 更新 current_lora

        Args:
            target_domain: 目标领域名

        Returns:
            True 切换成功, False 切换失败（回退到旧 LoRA）
        """
        # ── 无 PEFT 时直接返回 False ──
        if not _PEFT_AVAILABLE or peft is None:
            log.warning("PEFT 未安装，无法切换 LoRA")
            return False

        # ── 同领域跳过 ──
        with self._state_lock:
            current_lora = self._current_lora

        if target_domain == current_lora:
            return True

        # ── 目标 LoRA 路径检查 ──
        if target_domain not in self._lora_adapters:
            log.warning("LoRA 路径未配置: %s", target_domain)
            # 尝试回退到默认 LoRA
            if (self._default_lora
                    and self._default_lora != current_lora
                    and self._default_lora in self._lora_adapters):
                log.info("回退到默认 LoRA: %s", self._default_lora)
                return self._switch_lora(self._default_lora)
            return False

        lora_path = self._lora_adapters[target_domain]
        if not lora_path.exists():
            log.warning("LoRA 文件不存在: %s → %s", target_domain, lora_path)
            # 尝试回退到默认 LoRA
            if (self._default_lora
                    and self._default_lora != current_lora
                    and self._default_lora in self._lora_adapters):
                log.info("回退到默认 LoRA: %s", self._default_lora)
                return self._switch_lora(self._default_lora)
            return False

        # ── 执行切换 ──
        old_lora = current_lora
        start_time = time.monotonic()

        try:
            with self._state_lock:
                lora_model = self._lora_model
                base_model = self._base_model

            if lora_model is None and base_model is not None:
                # 首次加载: 从基座模型创建 PeftModel
                lora_model = peft.PeftModel.from_pretrained(
                    base_model, str(lora_path), adapter_name=target_domain
                )
                with self._state_lock:
                    self._lora_model = lora_model
                    self._current_lora = target_domain
            elif lora_model is not None:
                # 后续切换: 加载新适配器并切换
                # 先禁用当前适配器
                try:
                    lora_model.disable_adapter()
                except Exception:
                    pass  # 首次可能没有活动的适配器

                # 加载新适配器
                try:
                    lora_model.load_adapter(str(lora_path), adapter_name=target_domain)
                except Exception:
                    # 适配器可能已加载，尝试直接切换
                    pass

                lora_model.set_adapter(target_domain)
                with self._state_lock:
                    self._current_lora = target_domain
            else:
                # 无基座模型
                log.error("LoRA 切换失败: 无基座模型")
                return False

            duration_ms = (time.monotonic() - start_time) * 1000
            log.info(
                "LoRA switched: %s → %s, duration=%.1fms",
                old_lora or "none", target_domain, duration_ms,
            )

            with self._state_lock:
                self._lora_switch_count += 1

            return True

        except Exception as e:
            log.error("LoRA 切换失败: %s", e, exc_info=True)

            # 回退: 尝试恢复旧 LoRA
            if old_lora and old_lora in self._lora_adapters:
                try:
                    with self._state_lock:
                        lora_model = self._lora_model
                    if lora_model is not None:
                        lora_model.set_adapter(old_lora)
                        with self._state_lock:
                            self._current_lora = old_lora
                        log.info("已回退到旧 LoRA: %s", old_lora)
                except Exception as fallback_err:
                    log.error("回退旧 LoRA 也失败: %s", fallback_err)
                    with self._state_lock:
                        self._lora_model = None
                        self._current_lora = None

            return False

    # ══════════════════════════════════════════════════════════
    #  推理调用 + 超时/异常保护 (T-006)
    # ══════════════════════════════════════════════════════════

    def infer(
        self,
        prompt: str,
        target_domain: str,
        max_new_tokens: int = 512,
    ) -> InferenceResult:
        """执行专家推理

        Args:
            prompt: 完整的推理 prompt（含 system + context + query）
            target_domain: 目标领域（决定使用哪个 LoRA）
            max_new_tokens: 最大生成 token 数

        Returns:
            InferenceResult 包含回答文本、领域、可靠性

        异常保护:
            - 推理超时 → 降级
            - CUDA OOM → 降级
            - Ollama 不可用 → 降级
            - 任意 RuntimeError → 降级
        """
        # ── 懒加载检查 ──
        with self._state_lock:
            initialized = self._initialized
            available = self._expert_available
            active_backend = self._active_backend

        if not initialized:
            self.initialize()

        with self._state_lock:
            available = self._expert_available
            active_backend = self._active_backend

        if not available:
            with self._state_lock:
                self._fallback_count += 1
            return InferenceResult(
                answer_text="",
                domain=target_domain,
                reliability=self._get_reliability(target_domain),
                inference_time_ms=0.0,
                fallback=True,
            )

        # ── Ollama 后端推理 ──
        if active_backend == "ollama":
            return self._infer_ollama(prompt, target_domain, max_new_tokens)

        # ── PyTorch 后端推理（以下为原有逻辑）──

        # ── VRAM 预算检查 ──
        if not self._check_vram_budget():
            log.warning("VRAM 超预算，降级到 Phase 0")
            with self._state_lock:
                self._fallback_count += 1
            return InferenceResult(
                answer_text="",
                domain=target_domain,
                reliability=self._get_reliability(target_domain),
                inference_time_ms=0.0,
                fallback=True,
            )

        # ── 超时冷却检查 ──
        with self._state_lock:
            cooldown_until = self._timeout_cooldown_until
        if time.monotonic() < cooldown_until:
            log.warning("推理冷却中（前次超时），降级到 Phase 0")
            with self._state_lock:
                self._fallback_count += 1
            return InferenceResult(
                answer_text="",
                domain=target_domain,
                reliability=self._get_reliability(target_domain),
                inference_time_ms=0.0,
                fallback=True,
            )

        # ── 持有推理锁执行推理 ──
        with self._inference_lock:
            start_time = time.monotonic()

            try:
                # LoRA 切换
                self._switch_lora(target_domain)

                with self._state_lock:
                    base_model = self._base_model
                    tokenizer = self._tokenizer
                    lora_model = self._lora_model

                # 选择推理模型（有 LoRA 用 LoRA，否则用基座）
                inference_model = lora_model if lora_model is not None else base_model

                if inference_model is None or tokenizer is None:
                    with self._state_lock:
                        self._fallback_count += 1
                    return InferenceResult(
                        answer_text="",
                        domain=target_domain,
                        reliability=self._get_reliability(target_domain),
                        inference_time_ms=0.0,
                        fallback=True,
                    )

                # 构造输入
                inputs = tokenizer(prompt, return_tensors="pt").to("cuda")

                # 推理（含超时保护）
                result_text = self._run_inference_with_timeout(
                    inference_model, inputs, max_new_tokens
                )

                # 清洗回答文本
                result_text = self._clean_answer_text(result_text)

                inference_time_ms = (time.monotonic() - start_time) * 1000

                with self._state_lock:
                    self._inference_count += 1

                return InferenceResult(
                    answer_text=result_text,
                    domain=target_domain,
                    reliability=self._get_reliability(target_domain),
                    inference_time_ms=inference_time_ms,
                    fallback=False,
                )

            except RuntimeError as e:
                # CUDA OOM 或其他 RuntimeError
                if "out of memory" in str(e).lower():
                    log.error("CUDA OOM，降级到 Phase 0: %s", e)
                else:
                    log.error("推理 RuntimeError，降级到 Phase 0: %s", e)

                # 尝试清理 CUDA 缓存
                if _TORCH_AVAILABLE and torch is not None:
                    try:
                        torch.cuda.empty_cache()
                    except (RuntimeError, OSError):
                        pass

                with self._state_lock:
                    self._fallback_count += 1

                return InferenceResult(
                    answer_text="",
                    domain=target_domain,
                    reliability=self._get_reliability(target_domain),
                    inference_time_ms=(time.monotonic() - start_time) * 1000,
                    fallback=True,
                )

            except Exception as e:
                log.error("推理异常，降级到 Phase 0: %s", e, exc_info=True)

                with self._state_lock:
                    self._fallback_count += 1

                return InferenceResult(
                    answer_text="",
                    domain=target_domain,
                    reliability=self._get_reliability(target_domain),
                    inference_time_ms=(time.monotonic() - start_time) * 1000,
                    fallback=True,
                )

    def infer_multi_domain(
        self,
        prompt: str,
        domains: list[str],
        max_new_tokens: int = 512,
    ) -> list[InferenceResult]:
        """多领域推理（用于交叉验证）

        按领域激活值降序依次推理，每个领域使用对应 LoRA。
        单个领域推理失败不影响其他领域。

        Args:
            prompt: 推理 prompt
            domains: 目标领域列表（按优先级排序）
            max_new_tokens: 最大生成 token 数

        Returns:
            成功推理的 InferenceResult 列表
        """
        results: list[InferenceResult] = []

        for domain in domains:
            result = self.infer(prompt, domain, max_new_tokens=max_new_tokens)
            # 只保留非降级的结果
            if not result.fallback and result.answer_text:
                results.append(result)

        return results

    # ══════════════════════════════════════════════════════════
    #  Ollama 后端推理
    # ══════════════════════════════════════════════════════════

    def _infer_ollama(
        self,
        prompt: str,
        target_domain: str,
        max_new_tokens: int = 512,
    ) -> InferenceResult:
        """通过 Ollama HTTP API 执行推理

        优先使用 httpx，httpx 不可用时使用 urllib.request 作为备选。
        Ollama 后端不需要 LoRA 切换——模型本身已包含领域知识。
        Ollama 后端不需要 VRAM 检查——显存由 Ollama 管理。

        Args:
            prompt: 完整的推理 prompt
            target_domain: 目标领域（Ollama 后端不使用 LoRA，仅用于结果标记）
            max_new_tokens: 最大生成 token 数

        Returns:
            InferenceResult 包含回答文本、领域、可靠性
        """
        start_time = time.monotonic()

        try:
            result_text = self._call_ollama_api(prompt, max_new_tokens)

            # 清洗回答文本（含 deepseek-r1 思考标签清洗）
            result_text = self._clean_answer_text(result_text)

            inference_time_ms = (time.monotonic() - start_time) * 1000

            with self._state_lock:
                self._inference_count += 1

            return InferenceResult(
                answer_text=result_text,
                domain=target_domain,
                reliability=self._get_reliability(target_domain),
                inference_time_ms=inference_time_ms,
                fallback=False,
            )

        except Exception as e:
            log.error("Ollama 推理异常，降级到 Phase 0: %s", e, exc_info=True)

            with self._state_lock:
                self._fallback_count += 1

            return InferenceResult(
                answer_text="",
                domain=target_domain,
                reliability=self._get_reliability(target_domain),
                inference_time_ms=(time.monotonic() - start_time) * 1000,
                fallback=True,
            )

    def _call_ollama_api(self, prompt: str, max_new_tokens: int) -> str:
        """调用 Ollama HTTP API 生成回答

        优先使用 httpx，httpx 不可用时使用 urllib.request 作为备选。
        所有调用均有超时保护。

        Args:
            prompt: 推理 prompt
            max_new_tokens: 最大生成 token 数

        Returns:
            生成的文本

        Raises:
            Exception: API 调用失败
        """
        # ── 优先使用 httpx ──
        if _HTTPX_AVAILABLE and httpx is not None:
            response = httpx.post(
                f"{self._ollama_base_url}/api/generate",
                json={
                    "model": self._ollama_model,
                    "prompt": prompt,
                    "stream": False,
                    "options": {
                        "num_predict": max_new_tokens,
                        "temperature": 0.7,
                    },
                },
                timeout=self._ollama_timeout,
            )
            response.raise_for_status()
            result = response.json()
            return result.get("response", "")

        # ── httpx 不可用，使用 urllib.request 备选 ──
        import json as _json
        import urllib.request

        payload = _json.dumps({
            "model": self._ollama_model,
            "prompt": prompt,
            "stream": False,
            "options": {
                "num_predict": max_new_tokens,
                "temperature": 0.7,
            },
        }).encode("utf-8")

        req = urllib.request.Request(
            f"{self._ollama_base_url}/api/generate",
            data=payload,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=self._ollama_timeout) as resp:
            data = _json.loads(resp.read().decode("utf-8"))
            return data.get("response", "")

    # ══════════════════════════════════════════════════════════
    #  辅助方法
    # ══════════════════════════════════════════════════════════

    def _check_vram_budget(self) -> bool:
        """检查 VRAM 占用是否在预算内

        Returns:
            True 占用正常, False 超预算
        """
        if not _TORCH_AVAILABLE or torch is None:
            return False

        try:
            allocated_bytes = torch.cuda.memory_allocated()
            allocated_gb = allocated_bytes / (1024 ** 3)
            return allocated_gb <= self._max_vram_gb
        except (RuntimeError, OSError):
            return False

    def _get_reliability(self, domain: str) -> float:
        """获取指定领域的可靠性分数

        优先从持久化存储读取，回退到配置。

        Args:
            domain: 领域名

        Returns:
            可靠性分数 [0.0, 1.0]，默认 0.7
        """
        # 优先从持久化存储读取
        if self._reliability_store is not None:
            return self._reliability_store.get_reliability(domain)
        # 回退到配置
        with self._state_lock:
            return self._reliability.get(domain, DEFAULT_RELIABILITY)

    def _run_inference_with_timeout(
        self,
        model: object,
        inputs: object,
        max_new_tokens: int,
    ) -> str:
        """带超时保护的模型推理

        使用 threading.Event + 超时线程实现推理超时保护。
        超时时中断推理并返回空字符串。

        Args:
            model: 模型实例
            inputs: tokenizer 输出
            max_new_tokens: 最大生成 token 数

        Returns:
            生成的文本
        """
        result_container: dict[str, str] = {"text": ""}
        error_container: dict[str, Exception | None] = {"error": None}
        done_event = threading.Event()

        def _do_inference() -> None:
            try:
                with torch.no_grad():
                    outputs = model.generate(
                        **inputs,
                        max_new_tokens=max_new_tokens,
                        do_sample=True,
                        temperature=0.7,
                        top_p=0.9,
                    )

                # 解码输出（跳过输入部分）
                with self._state_lock:
                    tokenizer = self._tokenizer

                if tokenizer is not None:
                    input_len = inputs["input_ids"].shape[1]
                    generated_ids = outputs[0][input_len:]
                    result_container["text"] = tokenizer.decode(
                        generated_ids, skip_special_tokens=True
                    )
            except Exception as e:
                error_container["error"] = e
            finally:
                done_event.set()

        # 启动推理线程
        inference_thread = threading.Thread(target=_do_inference, daemon=True)
        inference_thread.start()

        # 等待推理完成或超时
        if not done_event.wait(timeout=self._inference_timeout):
            # 超时
            log.error(
                "推理超时（%.1fs），降级到 Phase 0",
                self._inference_timeout,
            )
            # 设置冷却期，防止频繁超时导致 GPU 资源泄漏
            with self._state_lock:
                self._timeout_cooldown_until = time.monotonic() + 30.0  # 30s cooldown
            # 注意: 无法安全地中断 CUDA 推理线程
            # 线程会在后台继续运行，但结果被丢弃
            return ""

        # 检查推理是否出错
        if error_container["error"] is not None:
            raise error_container["error"]

        return result_container["text"]

    @staticmethod
    def _clean_answer_text(text: str) -> str:
        """清洗专家回答文本，移除特殊标记

        对 Ollama deepseek-r1 等思考型模型，会移除 <think>...</think> 标签
        及其中的思考内容，只保留最终回答。

        Args:
            text: 原始回答文本

        Returns:
            清洗后的文本
        """
        if not text:
            return text

        # 移除 deepseek-r1 思考标签及内容（<think>...</think>）
        # 使用 DOTALL 标志使 . 匹配换行符
        text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)

        # 移除常见的特殊标记
        patterns = [
            r"<\|im_end\|>",
            r"<\|im_start\|>",
            r"\[INST\]",
            r"\[/INST\]",
            r"<\|.*?\|>",  # 其他特殊标记
        ]
        for pattern in patterns:
            text = re.sub(pattern, "", text)

        # 移除首尾空白
        text = text.strip()

        return text
