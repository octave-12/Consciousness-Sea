"""
T-020: ExpertReliabilityStore 单元测试

测试 ExpertReliabilityStore 的所有公共方法：
- initialize_table() 幂等创建表
- 初始分数从 config 写入数据库
- get_reliability() 返回默认值 0.7（未配置领域）
- update_reliability() 写入数据库 + 更新缓存
- score > 1.0 截断到 1.0 并记录 WARNING
- score < 0.0 截断到 0.0 并记录 WARNING
- get_all_scores() 返回所有领域分数
- 使用 sqlite3.connect(":memory:") 内存数据库
"""

from __future__ import annotations

import logging
import sqlite3
import sys
import pathlib

_root = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_root))
sys.path.insert(0, str(_root / "backend" / "src"))

from consciousness_sea.expert.expert_reliability import ExpertReliabilityStore, DEFAULT_RELIABILITY


class TestExpertReliabilityStoreInit:
    """表初始化测试"""

    def test_initialize_table_idempotent(self):
        """initialize_table() 幂等创建表"""
        conn = sqlite3.connect(':memory:')
        store = ExpertReliabilityStore()

        # 第一次调用
        store.initialize_table(conn)
        # 第二次调用不应报错
        store.initialize_table(conn)

        # 验证表存在
        rows = conn.execute("SELECT COUNT(*) FROM expert_reliability").fetchone()
        assert rows[0] == 0  # 无初始分数

    def test_initialize_table_with_initial_scores(self):
        """初始分数从 config 写入数据库"""
        conn = sqlite3.connect(':memory:')
        initial = {"医学": 0.85, "常识": 0.6}
        store = ExpertReliabilityStore(initial_scores=initial)
        store.initialize_table(conn)

        # 验证初始分数已写入
        rows = conn.execute(
            "SELECT domain, score FROM expert_reliability ORDER BY domain"
        ).fetchall()
        assert len(rows) == 2
        # 排序后验证（ORDER BY domain 可能按拼音排序）
        domains = {row[0]: row[1] for row in rows}
        assert abs(domains["常识"] - 0.6) < 0.001
        assert abs(domains["医学"] - 0.85) < 0.001

    def test_initialize_table_creates_index(self):
        """initialize_table() 创建索引"""
        conn = sqlite3.connect(':memory:')
        store = ExpertReliabilityStore()
        store.initialize_table(conn)

        # 验证索引存在
        indexes = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='index' AND name='idx_expert_reliability_domain'"
        ).fetchone()
        assert indexes is not None

    def test_initialize_table_twice_no_duplicate_data(self):
        """多次 initialize_table 不产生重复数据"""
        conn = sqlite3.connect(':memory:')
        initial = {"医学": 0.85}
        store = ExpertReliabilityStore(initial_scores=initial)

        store.initialize_table(conn)
        store.initialize_table(conn)  # 第二次

        rows = conn.execute("SELECT COUNT(*) FROM expert_reliability").fetchone()
        assert rows[0] == 1  # 不应重复


class TestExpertReliabilityStoreGet:
    """分数读取测试"""

    def test_get_reliability_from_cache(self):
        """从内存缓存读取分数"""
        conn = sqlite3.connect(':memory:')
        store = ExpertReliabilityStore(initial_scores={"医学": 0.85})
        store.initialize_table(conn)

        # 第一次读取（从数据库加载到缓存）
        score = store.get_reliability("医学")
        assert abs(score - 0.85) < 0.001

    def test_get_reliability_default_for_unconfigured_domain(self):
        """未配置领域返回默认值 0.7"""
        conn = sqlite3.connect(':memory:')
        store = ExpertReliabilityStore(initial_scores={"医学": 0.85})
        store.initialize_table(conn)

        score = store.get_reliability("法律")
        assert score == DEFAULT_RELIABILITY
        assert score == 0.7

    def test_get_reliability_no_connection_returns_default(self):
        """无数据库连接时返回默认值"""
        store = ExpertReliabilityStore(initial_scores={})
        # 不调用 initialize_table，_conn 为 None
        score = store.get_reliability("医学")
        assert score == DEFAULT_RELIABILITY

    def test_get_reliability_lookup_order(self):
        """查找顺序: 内存缓存 → 数据库 → 默认值"""
        conn = sqlite3.connect(':memory:')
        store = ExpertReliabilityStore(initial_scores={"医学": 0.85})
        store.initialize_table(conn)

        # 1. 内存缓存（初始化时已加载）
        assert store.get_reliability("医学") == 0.85

        # 2. 默认值（未配置领域）
        assert store.get_reliability("未知") == 0.7


class TestExpertReliabilityStoreUpdate:
    """分数更新测试"""

    def test_update_reliability_writes_to_db_and_cache(self):
        """update_reliability() 写入数据库 + 更新缓存"""
        conn = sqlite3.connect(':memory:')
        store = ExpertReliabilityStore(initial_scores={"医学": 0.85})
        store.initialize_table(conn)

        store.update_reliability("医学", 0.9)

        # 验证缓存更新
        assert abs(store.get_reliability("医学") - 0.9) < 0.001

        # 验证数据库更新
        row = conn.execute(
            "SELECT score FROM expert_reliability WHERE domain = ?", ("医学",)
        ).fetchone()
        assert abs(row[0] - 0.9) < 0.001

    def test_update_reliability_new_domain(self):
        """更新不存在的领域（新增）"""
        conn = sqlite3.connect(':memory:')
        store = ExpertReliabilityStore(initial_scores={"医学": 0.85})
        store.initialize_table(conn)

        store.update_reliability("法律", 0.8)

        assert abs(store.get_reliability("法律") - 0.8) < 0.001

        row = conn.execute(
            "SELECT score FROM expert_reliability WHERE domain = ?", ("法律",)
        ).fetchone()
        assert abs(row[0] - 0.8) < 0.001

    def test_update_reliability_score_above_1_clamped(self):
        """score > 1.0 截断到 1.0 并记录 WARNING"""
        conn = sqlite3.connect(':memory:')
        store = ExpertReliabilityStore()
        store.initialize_table(conn)

        with self._capture_logs() as log_records:
            store.update_reliability("测试", 1.5)

        assert store.get_reliability("测试") == 1.0

        # 验证 WARNING 日志
        warning_logs = [r for r in log_records if r.levelno == logging.WARNING]
        assert any("1.5" in r.message for r in warning_logs)

    def test_update_reliability_score_below_0_clamped(self):
        """score < 0.0 截断到 0.0 并记录 WARNING"""
        conn = sqlite3.connect(':memory:')
        store = ExpertReliabilityStore()
        store.initialize_table(conn)

        with self._capture_logs() as log_records:
            store.update_reliability("测试", -0.5)

        assert store.get_reliability("测试") == 0.0

        # 验证 WARNING 日志
        warning_logs = [r for r in log_records if r.levelno == logging.WARNING]
        assert any("-0.5" in r.message or "-0.5000" in r.message for r in warning_logs)

    def test_update_reliability_score_at_boundary(self):
        """score 恰好为 0.0 和 1.0 不截断"""
        conn = sqlite3.connect(':memory:')
        store = ExpertReliabilityStore()
        store.initialize_table(conn)

        store.update_reliability("零值", 0.0)
        assert store.get_reliability("零值") == 0.0

        store.update_reliability("满值", 1.0)
        assert store.get_reliability("满值") == 1.0

    @staticmethod
    def _capture_logs():
        """捕获日志记录的上下文管理器"""
        import contextlib

        @contextlib.contextmanager
        def _capture():
            handler = logging.Handler()
            records = []
            handler.emit = records.append
            logger = logging.getLogger("consciousness_sea.expert.expert_reliability")
            logger.addHandler(handler)
            try:
                yield records
            finally:
                logger.removeHandler(handler)

        return _capture()


class TestExpertReliabilityStoreGetAll:
    """get_all_scores() 测试"""

    def test_get_all_scores_returns_all_domains(self):
        """get_all_scores() 返回所有领域分数"""
        conn = sqlite3.connect(':memory:')
        initial = {"医学": 0.85, "常识": 0.6}
        store = ExpertReliabilityStore(initial_scores=initial)
        store.initialize_table(conn)

        scores = store.get_all_scores()
        assert "医学" in scores
        assert abs(scores["医学"] - 0.85) < 0.001
        assert "常识" in scores
        assert abs(scores["常识"] - 0.6) < 0.001

    def test_get_all_scores_includes_updated(self):
        """get_all_scores() 包含更新后的分数"""
        conn = sqlite3.connect(':memory:')
        store = ExpertReliabilityStore(initial_scores={"医学": 0.85})
        store.initialize_table(conn)

        store.update_reliability("法律", 0.8)

        scores = store.get_all_scores()
        assert "法律" in scores
        assert abs(scores["法律"] - 0.8) < 0.001

    def test_get_all_scores_empty_db(self):
        """空数据库返回空字典"""
        conn = sqlite3.connect(':memory:')
        store = ExpertReliabilityStore(initial_scores={})
        store.initialize_table(conn)

        scores = store.get_all_scores()
        assert isinstance(scores, dict)


class TestExpertReliabilityStorePersistence:
    """持久化测试"""

    def test_data_survives_reinitialize(self):
        """数据在重新初始化后保留（从数据库读取）"""
        conn = sqlite3.connect(':memory:')
        initial = {"医学": 0.85}
        store1 = ExpertReliabilityStore(initial_scores=initial)
        store1.initialize_table(conn)

        # 更新分数
        store1.update_reliability("医学", 0.9)

        # 创建新的 store 实例，使用同一个连接
        store2 = ExpertReliabilityStore(initial_scores={})
        store2.initialize_table(conn)

        # 数据库中已有数据，应从数据库读取
        score = store2.get_reliability("医学")
        assert abs(score - 0.9) < 0.001


class TestExpertReliabilityStoreDefaultReliability:
    """DEFAULT_RELIABILITY 常量测试"""

    def test_default_reliability_value(self):
        """DEFAULT_RELIABILITY = 0.7"""
        assert DEFAULT_RELIABILITY == 0.7

    def test_default_reliability_used_for_unknown_domain(self):
        """未配置领域使用默认值"""
        conn = sqlite3.connect(':memory:')
        store = ExpertReliabilityStore(initial_scores={"医学": 0.85})
        store.initialize_table(conn)

        assert store.get_reliability("完全不存在的领域") == 0.7


if __name__ == '__main__':
    import pytest
    pytest.main([__file__, '-v'])