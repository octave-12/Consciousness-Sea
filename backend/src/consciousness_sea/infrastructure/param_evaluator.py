"""
参数评估工具 — 统计观测调优，非 ML 调优

扫描衰减系数、领域阈值、正向熏习条件，
输出推荐值和评估报告（JSON 格式）。

关键约束:
  - 仅输出推荐值，不自动修改 config.py
  - 统计数据不足 100 次时标注警告
  - 评估超时 5 分钟时输出部分结果
  - 纯标准库，不依赖其他新模块
"""

from __future__ import annotations

import json
import logging
import time
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .connection_pool import ConnectionPool

from .config import PARAM_EVAL_MIN_SAMPLES, PARAM_EVAL_TIMEOUT_SEC

log = logging.getLogger(__name__)


class _EvaluationTimeout(Exception):
    """评估超时异常 — 用于 signal.alarm 触发"""

    pass


class ParamEvaluator:
    """参数统计评估工具 — 统计观测调优，非 ML 调优

    从 param_stats 表加载历史统计数据，扫描不同参数值，
    计算匹配精确度、F1 分数、熏习质量等指标，
    输出推荐值和评估报告。

    评估工具仅输出推荐值，不自动修改 config.py。

    Args:
        pool: 连接池实例
    """

    def __init__(self, pool: ConnectionPool) -> None:
        self._pool = pool

    # ═══════════════════════════════════════════════════════════
    #  公开评估方法
    # ═══════════════════════════════════════════════════════════

    def evaluate_decay_factor(
        self,
        decay_range: tuple[float, float] = (0.5, 0.9),
        step: float = 0.05,
    ) -> dict:
        """评估衰减系数

        扫描不同衰减系数值，模拟涟漪传播，计算匹配精确度。
        精确度定义：在该衰减系数下，激活种子数与查询词匹配种子数
        的比值越高，说明传播越精准。

        Args:
            decay_range: 扫描范围 (start, end)，默认 (0.5, 0.9)
            step: 扫描步长，默认 0.05

        Returns:
            评估报告字典，包含:
              - parameter: 参数名
              - candidates: 各候选值的统计指标
              - recommended: 推荐值
              - recommendation_reason: 推荐依据
              - sample_size: 样本数
              - warning: 警告信息（如有）
        """
        stats = self._load_stats()
        sample_size = len(stats)
        warning = self._check_sample_size(sample_size)

        candidates: list[dict] = []
        timed_out = False
        start_time = time.monotonic()

        for value in self._range(decay_range, step):
            # 超时检查
            if time.monotonic() - start_time > PARAM_EVAL_TIMEOUT_SEC:
                timed_out = True
                log.warning("衰减系数评估超时，已输出部分结果")
                break

            precision = self._compute_decay_precision(stats, value)
            candidates.append({
                'value': round(value, 4),
                'precision': round(precision, 4),
            })

        # 选择精确度最高的候选值作为推荐
        recommended, reason = self._pick_best(
            candidates, 'precision', '衰减系数',
            higher_is_better=True,
        )

        if timed_out:
            warning = (warning or '') + '评估超时，结果不完整'.lstrip()

        return {
            'parameter': 'decay_factor',
            'candidates': candidates,
            'recommended': recommended,
            'recommendation_reason': reason,
            'sample_size': sample_size,
            'warning': warning or None,
        }

    def evaluate_domain_threshold(
        self,
        threshold_range: tuple[float, float] = (0.1, 0.5),
        step: float = 0.05,
    ) -> dict:
        """评估领域阈值

        扫描不同阈值，计算误报率、漏报率和 F1 分数。
        误报：领域激活值低于阈值但被选中的情况（阈值越低误报越多）。
        漏报：领域激活值高于阈值但未被选中的情况（阈值越高漏报越多）。

        Args:
            threshold_range: 扫描范围 (start, end)，默认 (0.1, 0.5)
            step: 扫描步长，默认 0.05

        Returns:
            评估报告字典，包含:
              - parameter: 参数名
              - candidates: 各候选值的统计指标（false_positive, false_negative, f1）
              - recommended: 推荐值
              - recommendation_reason: 推荐依据
              - sample_size: 样本数
              - warning: 警告信息（如有）
        """
        stats = self._load_stats()
        sample_size = len(stats)
        warning = self._check_sample_size(sample_size)

        candidates: list[dict] = []
        timed_out = False
        start_time = time.monotonic()

        for value in self._range(threshold_range, step):
            # 超时检查
            if time.monotonic() - start_time > PARAM_EVAL_TIMEOUT_SEC:
                timed_out = True
                log.warning("领域阈值评估超时，已输出部分结果")
                break

            fp_rate, fn_rate, f1 = self._compute_domain_f1(stats, value)
            candidates.append({
                'value': round(value, 4),
                'false_positive': round(fp_rate, 4),
                'false_negative': round(fn_rate, 4),
                'f1': round(f1, 4),
            })

        # 选择 F1 最高的候选值作为推荐
        recommended, reason = self._pick_best(
            candidates, 'f1', '领域阈值',
            higher_is_better=True,
        )

        if timed_out:
            warning = (warning or '') + '评估超时，结果不完整'.lstrip()

        return {
            'parameter': 'domain_threshold',
            'candidates': candidates,
            'recommended': recommended,
            'recommendation_reason': reason,
            'sample_size': sample_size,
            'warning': warning or None,
        }

    def evaluate_positive_karma_threshold(
        self,
        threshold_range: tuple[float, float] = (0.5, 0.9),
        step: float = 0.05,
    ) -> dict:
        """评估正向熏习条件

        扫描不同阈值，计算正向熏习率、负向熏习率和熏习质量指标。
        正向熏习率：confidence >= threshold 时执行正向熏习的比例。
        负向熏习率：confidence < threshold 时执行负向熏习的比例。
        熏习质量：正向熏习率 / (正向熏习率 + 负向熏习率)，越接近 1 越好。

        Args:
            threshold_range: 扫描范围 (start, end)，默认 (0.5, 0.9)
            step: 扫描步长，默认 0.05

        Returns:
            评估报告字典，包含:
              - parameter: 参数名
              - candidates: 各候选值的统计指标（positive_rate, negative_rate, quality）
              - recommended: 推荐值
              - recommendation_reason: 推荐依据
              - sample_size: 样本数
              - warning: 警告信息（如有）
        """
        stats = self._load_stats()
        sample_size = len(stats)
        warning = self._check_sample_size(sample_size)

        candidates: list[dict] = []
        timed_out = False
        start_time = time.monotonic()

        for value in self._range(threshold_range, step):
            # 超时检查
            if time.monotonic() - start_time > PARAM_EVAL_TIMEOUT_SEC:
                timed_out = True
                log.warning("正向熏习条件评估超时，已输出部分结果")
                break

            pos_rate, neg_rate, quality = self._compute_karma_quality(stats, value)
            candidates.append({
                'value': round(value, 4),
                'positive_rate': round(pos_rate, 4),
                'negative_rate': round(neg_rate, 4),
                'quality': round(quality, 4),
            })

        # 选择熏习质量最高的候选值作为推荐
        recommended, reason = self._pick_best(
            candidates, 'quality', '正向熏习条件',
            higher_is_better=True,
        )

        if timed_out:
            warning = (warning or '') + '评估超时，结果不完整'.lstrip()

        return {
            'parameter': 'positive_karma_threshold',
            'candidates': candidates,
            'recommended': recommended,
            'recommendation_reason': reason,
            'sample_size': sample_size,
            'warning': warning or None,
        }

    # ═══════════════════════════════════════════════════════════
    #  数据加载
    # ═══════════════════════════════════════════════════════════

    def _load_stats(self, min_count: int = 0) -> list[dict]:
        """从 param_stats 表加载统计数据

        使用快照数据，不受运行中变更影响。

        Args:
            min_count: 最小记录数要求（仅用于日志提示，不实际过滤）

        Returns:
            统计记录列表，按 created_at 降序排列
        """
        graph = self._pool.acquire()
        try:
            # 确保 param_stats 表存在
            from .param_stats import ensure_param_stats_table
            ensure_param_stats_table(graph)

            rows = graph.conn.execute(
                "SELECT * FROM param_stats ORDER BY created_at DESC LIMIT 1000"
            ).fetchall()
            result = [dict(r) for r in rows]

            if min_count > 0 and len(result) < min_count:
                log.warning(
                    "统计数据不足: 当前 %d 条，建议至少 %d 条",
                    len(result), min_count,
                )

            return result
        except Exception as e:
            log.error("加载参数统计数据失败: %s", e)
            return []
        finally:
            self._pool.release(graph)

    # ═══════════════════════════════════════════════════════════
    #  指标计算
    # ═══════════════════════════════════════════════════════════

    def _compute_decay_precision(
        self, stats: list[dict], decay_value: float,
    ) -> float:
        """计算指定衰减系数下的匹配精确度

        精确度 = 在该衰减系数下，传播深度与激活种子数的合理比值。
        使用 ripple_depth / activated_count 作为传播效率的代理指标，
        与 decay_value 的理论预期对比。

        理论预期：衰减系数越高，传播越远（ripple_depth 更大），
        但激活种子数增长应与衰减系数成反比（高衰减 → 低效率）。

        精确度 = 1 - |实际传播效率 - 理论传播效率| / 理论传播效率

        Args:
            stats: 统计记录列表
            decay_value: 待评估的衰减系数值

        Returns:
            精确度 [0, 1]
        """
        if not stats:
            return 0.0

        # 筛选使用相近衰减系数的记录（容差 ±0.1）
        relevant = [
            s for s in stats
            if abs(s.get('decay_factor', 0) - decay_value) <= 0.1
        ]

        if not relevant:
            # 无直接匹配记录，使用全量数据模拟
            relevant = stats

        # 计算传播效率：深度/激活数 比值
        efficiencies: list[float] = []
        for s in relevant:
            activated_count = s.get('activated_count', 0)
            ripple_depth = s.get('ripple_depth', 0)
            if activated_count > 0:
                # 传播效率：深度与激活数的归一化比值
                efficiency = ripple_depth / (activated_count ** 0.5)
                efficiencies.append(efficiency)

        if not efficiencies:
            return 0.0

        avg_efficiency = sum(efficiencies) / len(efficiencies)

        # 理论传播效率：衰减系数越高，传播效率越低
        # 使用简单的反比关系：theoretical = k / decay_value
        # k 为归一化常数，取当前衰减系数下的效率
        from .config import RIPPLE_DECAY
        if decay_value > 0:
            theoretical_efficiency = avg_efficiency * (RIPPLE_DECAY / decay_value)
        else:
            theoretical_efficiency = avg_efficiency

        if theoretical_efficiency > 0:
            precision = 1.0 - abs(avg_efficiency - theoretical_efficiency) / theoretical_efficiency
            return max(0.0, min(1.0, precision))
        return 0.0

    def _compute_domain_f1(
        self, stats: list[dict], threshold_value: float,
    ) -> tuple[float, float, float]:
        """计算指定领域阈值下的误报率、漏报率和 F1 分数

        误报（false_positive）：领域激活值低于阈值但被选中的情况。
        漏报（false_negative）：领域激活值高于阈值但未被选中的情况。

        使用 selected_domains 数量与 threshold_value 的关系推断。

        Args:
            stats: 统计记录列表
            threshold_value: 待评估的领域阈值

        Returns:
            (false_positive_rate, false_negative_rate, f1_score)
        """
        if not stats:
            return 0.0, 0.0, 0.0

        true_positive = 0
        false_positive = 0
        false_negative = 0
        true_negative = 0

        for s in stats:
            domain_threshold = s.get('domain_threshold', 0.3)
            selected_domains_raw = s.get('selected_domains', '[]')

            # 解析 selected_domains
            try:
                if isinstance(selected_domains_raw, str):
                    selected_domains = json.loads(selected_domains_raw)
                elif isinstance(selected_domains_raw, list):
                    selected_domains = selected_domains_raw
                else:
                    selected_domains = []
            except (json.JSONDecodeError, TypeError):
                selected_domains = []

            num_domains = len(selected_domains)

            # 判断在当前阈值下是否会被选中
            # 使用 domain_threshold 和 threshold_value 的比较来推断
            # 如果原始 domain_threshold <= threshold_value，说明该查询使用了较低阈值
            # 在更高阈值下，领域数可能减少
            estimated_domains = max(0, int(num_domains * (domain_threshold / max(threshold_value, 0.01))))

            # 简化模型：
            # - 如果 estimated_domains > 0 且 num_domains > 0 → TP
            # - 如果 estimated_domains > 0 且 num_domains == 0 → FP
            # - 如果 estimated_domains == 0 且 num_domains > 0 → FN
            # - 如果 estimated_domains == 0 且 num_domains == 0 → TN
            if estimated_domains > 0 and num_domains > 0:
                true_positive += 1
            elif estimated_domains > 0 and num_domains == 0:
                false_positive += 1
            elif estimated_domains == 0 and num_domains > 0:
                false_negative += 1
            else:
                true_negative += 1

        total = true_positive + false_positive + false_negative + true_negative
        if total == 0:
            return 0.0, 0.0, 0.0

        fp_rate = false_positive / total
        fn_rate = false_negative / total

        # F1 = 2 * precision * recall / (precision + recall)
        precision = true_positive / (true_positive + false_positive) if (true_positive + false_positive) > 0 else 0.0
        recall = true_positive / (true_positive + false_negative) if (true_positive + false_negative) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

        return fp_rate, fn_rate, f1

    def _compute_karma_quality(
        self, stats: list[dict], threshold_value: float,
    ) -> tuple[float, float, float]:
        """计算指定正向熏习阈值下的熏习质量

        正向熏习率：confidence >= threshold 时执行正向熏习的比例。
        负向熏习率：confidence < threshold 时执行负向熏习的比例。
        熏习质量：正向熏习率 / (正向熏习率 + 负向熏习率)。

        Args:
            stats: 统计记录列表
            threshold_value: 待评估的正向熏习阈值

        Returns:
            (positive_rate, negative_rate, quality)
        """
        from .config import CONFIDENCE_LOW

        if not stats:
            return 0.0, 0.0, 0.0

        positive_count = 0
        negative_count = 0
        neutral_count = 0

        for s in stats:
            confidence = s.get('confidence', 0.0)
            s.get('karma_direction', 0)

            # 在新阈值下重新判定熏习方向
            if confidence >= threshold_value:
                positive_count += 1
            elif confidence < CONFIDENCE_LOW:
                negative_count += 1
            else:
                neutral_count += 1

        total = positive_count + negative_count + neutral_count
        if total == 0:
            return 0.0, 0.0, 0.0

        positive_rate = positive_count / total
        negative_rate = negative_count / total

        # 熏习质量：正向熏习占比越高越好，但负向熏习也需要一定比例
        # 使用 F1 思想：quality = 2 * positive_rate / (2 * positive_rate + negative_rate)
        if (2 * positive_rate + negative_rate) > 0:
            quality = 2 * positive_rate / (2 * positive_rate + negative_rate)
        else:
            quality = 0.0

        return positive_rate, negative_rate, quality

    # ═══════════════════════════════════════════════════════════
    #  辅助方法
    # ═══════════════════════════════════════════════════════════

    @staticmethod
    def _range(
        value_range: tuple[float, float], step: float,
    ) -> list[float]:
        """生成扫描范围值列表

        Args:
            value_range: (start, end) 范围
            step: 步长

        Returns:
            从 start 到 end（含 end）的值列表
        """
        start, end = value_range
        values: list[float] = []
        current = start
        while current <= end + step * 0.01:  # 浮点容差
            values.append(round(current, 4))
            current += step
        # 确保包含 end 值
        if values and abs(values[-1] - end) > step * 0.01:
            values.append(round(end, 4))
        return values

    @staticmethod
    def _check_sample_size(sample_size: int) -> str | None:
        """检查样本量是否充足

        统计数据不足 100 次时返回警告信息。

        Args:
            sample_size: 当前样本量

        Returns:
            警告信息字符串，或 None（样本充足时）
        """
        if sample_size < PARAM_EVAL_MIN_SAMPLES:
            return (
                f"统计数据不足 {PARAM_EVAL_MIN_SAMPLES} 次"
                f"（当前 {sample_size} 次），结果仅供参考"
            )
        return None

    @staticmethod
    def _pick_best(
        candidates: list[dict],
        metric_key: str,
        param_name: str,
        higher_is_better: bool = True,
    ) -> tuple[float | None, str]:
        """从候选值中选择最佳推荐

        Args:
            candidates: 候选值列表，每个元素包含 'value' 和 metric_key
            metric_key: 用于比较的指标键名
            param_name: 参数名称（用于推荐依据描述）
            higher_is_better: True 表示指标越高越好

        Returns:
            (推荐值, 推荐依据)
        """
        if not candidates:
            return None, "无候选值可供评估"

        best = candidates[0]
        for c in candidates[1:]:
            if higher_is_better:
                if c.get(metric_key, 0) > best.get(metric_key, 0):
                    best = c
            else:
                if c.get(metric_key, float('inf')) < best.get(metric_key, float('inf')):
                    best = c

        recommended_value = best.get('value')
        metric_value = best.get(metric_key, 0)

        reason = (
            f"{param_name}={recommended_value} 时 {metric_key}={metric_value:.4f}，"
            f"在 {len(candidates)} 个候选值中最优"
        )

        return recommended_value, reason

    def evaluate_all(
        self,
        decay_range: tuple[float, float] = (0.5, 0.9),
        decay_step: float = 0.05,
        threshold_range: tuple[float, float] = (0.1, 0.5),
        threshold_step: float = 0.05,
        karma_range: tuple[float, float] = (0.5, 0.9),
        karma_step: float = 0.05,
    ) -> dict:
        """执行全部参数评估，输出综合报告

        Args:
            decay_range: 衰减系数扫描范围
            decay_step: 衰减系数扫描步长
            threshold_range: 领域阈值扫描范围
            threshold_step: 领域阈值扫描步长
            karma_range: 正向熏习条件扫描范围
            karma_step: 正向熏习条件扫描步长

        Returns:
            综合评估报告字典
        """
        start_time = time.monotonic()

        decay_report = self.evaluate_decay_factor(decay_range, decay_step)
        threshold_report = self.evaluate_domain_threshold(threshold_range, threshold_step)
        karma_report = self.evaluate_positive_karma_threshold(karma_range, karma_step)

        elapsed = time.monotonic() - start_time

        return {
            'evaluation_type': 'param_evaluation_full',
            'reports': {
                'decay_factor': decay_report,
                'domain_threshold': threshold_report,
                'positive_karma_threshold': karma_report,
            },
            'elapsed_seconds': round(elapsed, 2),
            'note': '评估工具仅输出推荐值，不自动修改 config.py',
        }
