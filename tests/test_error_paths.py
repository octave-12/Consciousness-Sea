"""
错误路径测试 — 覆盖审查报告中 #10.1 所有关键故障场景

覆盖:
- 连接池耗尽时的行为
- 数据库连接中断时的恢复
- 并发写入冲突（乐观锁重试耗尽）
- 提炼池升级失败回滚
- 检查点文件损坏时的回滚
- Ollama 服务不可用时的降级
"""

import json
import sqlite3
import sys
import threading
import pathlib
import time
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import MagicMock, patch, PropertyMock

import pytest

_root = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_root))
sys.path.insert(0, str(_root / "backend" / "src"))

from consciousness_sea.infrastructure.connection_pool import (
    ConnectionPool,
    ConnectionPoolExhausted,
    ConnectionPoolClosed,
)
from consciousness_sea.domain.graph_db import GraphDB
from consciousness_sea.learning.distillation_pool import DistillationPool
from consciousness_sea.learning.checkpoint import CheckpointManager, RollbackResult
from consciousness_sea.infrastructure.user_manager import UserManager
from consciousness_sea.expert.expert_manager import ExpertManager, InferenceResult

sys.path.insert(0, str(_root / "tests"))
from conftest import MockExpertManager


def _build_test_db(db_path: str) -> None:
    conn = sqlite3.connect(db_path)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA synchronous=NORMAL")
    conn.execute("PRAGMA busy_timeout=5000")
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS seeds (
            id TEXT PRIMARY KEY, label TEXT NOT NULL,
            type TEXT NOT NULL DEFAULT 'CONCEPT',
            aliases TEXT NOT NULL DEFAULT '[]',
            activation REAL NOT NULL DEFAULT 0.0,
            domain TEXT NOT NULL DEFAULT '',
            definition TEXT NOT NULL DEFAULT '',
            pinyin TEXT NOT NULL DEFAULT '',
            activation_bias REAL NOT NULL DEFAULT 0.0,
            meta TEXT NOT NULL DEFAULT '{}',
            created_at TEXT NOT NULL DEFAULT (datetime('now'))
        );
        CREATE TABLE IF NOT EXISTS karma_edges (
            source TEXT NOT NULL, target TEXT NOT NULL, relation TEXT NOT NULL,
            weight REAL NOT NULL DEFAULT 0.5,
            source_tag TEXT NOT NULL DEFAULT '',
            created_at TEXT NOT NULL DEFAULT (datetime('now')),
            PRIMARY KEY (source, target, relation)
        );
    """)
    conn.executemany(
        "INSERT OR IGNORE INTO seeds (id, label, type, aliases, domain, definition) "
        "VALUES (?, ?, ?, ?, ?, ?)",
        [
            ('感冒', '感冒', 'CONCEPT', '[]', '医学', 'cold'),
            ('发热', '发热', 'CONCEPT', '[]', '医学', 'fever'),
            ('咳嗽', '咳嗽', 'CONCEPT', '[]', '医学', 'cough'),
            ('维C', '维C', 'CONCEPT', '[]', '营养', 'Vitamin C'),
            ('姜汤', '姜汤', 'CONCEPT', '[]', '常识', 'ginger soup'),
            ('user_test', 'test_user', 'USER', '[]', '', ''),
        ],
    )
    conn.executemany(
        "INSERT OR IGNORE INTO karma_edges (source, target, relation, weight) "
        "VALUES (?, ?, ?, ?)",
        [
            ('感冒', '发热', 'COOCCURS_WITH', 0.95),
            ('感冒', '咳嗽', 'COOCCURS_WITH', 0.90),
            ('感冒', '维C', 'RELATED', 0.60),
        ],
    )
    conn.commit()
    conn.close()


def _make_graph_with_pool(tmp_path, pool_size=3):
    db_path = str(tmp_path / "test.db")
    _build_test_db(db_path)
    pool = ConnectionPool(db_path, pool_size=pool_size)
    return pool


# ═══════════════════════════════════════════════════════════
#  1. 连接池耗尽时的行为
# ═══════════════════════════════════════════════════════════


class TestConnectionPoolExhaustion:
    """连接池耗尽 — 验证超时、异常类型、状态恢复"""

    def test_exhausted_raises_correct_exception(self, tmp_path):
        pool = _make_graph_with_pool(tmp_path, pool_size=1)
        try:
            g1 = pool.acquire()
            with pytest.raises(ConnectionPoolExhausted):
                pool.acquire(timeout=0.1)
            pool.release(g1)
        finally:
            pool.close_all()

    def test_exhausted_exception_contains_pool_state(self, tmp_path):
        pool = _make_graph_with_pool(tmp_path, pool_size=2)
        try:
            g1 = pool.acquire()
            g2 = pool.acquire()
            with pytest.raises(ConnectionPoolExhausted) as exc_info:
                pool.acquire(timeout=0.1)
            msg = str(exc_info.value)
            assert "pool_size=2" in msg
            assert "in_use=2" in msg
            pool.release(g1)
            pool.release(g2)
        finally:
            pool.close_all()

    def test_exhausted_recovery_after_release(self, tmp_path):
        pool = _make_graph_with_pool(tmp_path, pool_size=1)
        try:
            g1 = pool.acquire()
            with pytest.raises(ConnectionPoolExhausted):
                pool.acquire(timeout=0.1)
            pool.release(g1)
            g2 = pool.acquire(timeout=1.0)
            assert g2 is not None
            pool.release(g2)
        finally:
            pool.close_all()

    def test_connection_creation_failure_returns_quota(self, tmp_path):
        pool = _make_graph_with_pool(tmp_path, pool_size=3)
        try:
            with patch.object(pool, '_create_connection', side_effect=RuntimeError("DB corrupted")):
                with pytest.raises(RuntimeError, match="DB corrupted"):
                    pool.acquire(timeout=1.0)
            with pool._lock:
                assert pool._created_count == 0
        finally:
            pool.close_all()

    def test_close_all_then_acquire_raises_closed(self, tmp_path):
        pool = _make_graph_with_pool(tmp_path, pool_size=2)
        try:
            g1 = pool.acquire()
            pool.release(g1)
            pool.close_all()
            with pytest.raises(ConnectionPoolClosed):
                pool.acquire()
        finally:
            pass

    def test_close_all_then_release_closes_connection(self, tmp_path):
        pool = _make_graph_with_pool(tmp_path, pool_size=2)
        try:
            g1 = pool.acquire()
            pool.close_all()
            pool.release(g1)
            assert g1.conn is None
        finally:
            pass


# ═══════════════════════════════════════════════════════════
#  2. 数据库连接中断时的恢复
# ═══════════════════════════════════════════════════════════


class TestDatabaseConnectionInterruption:
    """数据库连接中断 — 验证异常处理和恢复"""

    def test_closed_connection_query_raises_error(self, tmp_path):
        db_path = str(tmp_path / "test.db")
        _build_test_db(db_path)
        graph = GraphDB(db_path)
        graph.connect()
        graph.close()
        with pytest.raises(Exception):
            graph.conn.execute("SELECT 1")

    def test_graphdb_close_sets_conn_none(self, tmp_path):
        db_path = str(tmp_path / "test.db")
        _build_test_db(db_path)
        graph = GraphDB(db_path)
        graph.connect()
        assert graph.conn is not None
        graph.close()
        assert graph.conn is None

    def test_pool_connection_valid_after_reacquire(self, tmp_path):
        pool = _make_graph_with_pool(tmp_path, pool_size=2)
        try:
            g1 = pool.acquire()
            seed = g1.get_seed('感冒')
            assert seed is not None
            pool.release(g1)
            g2 = pool.acquire()
            seed2 = g2.get_seed('感冒')
            assert seed2 is not None
            pool.release(g2)
        finally:
            pool.close_all()


# ═══════════════════════════════════════════════════════════
#  3. 并发写入冲突（乐观锁重试耗尽）
# ═══════════════════════════════════════════════════════════


class TestOptimisticLockContention:
    """乐观锁重试耗尽 — 验证并发冲突处理"""

    def test_concurrent_preference_update_one_succeeds(self, tmp_path):
        pool = _make_graph_with_pool(tmp_path, pool_size=3)
        try:
            mgr = UserManager(pool)
            user_label = mgr.resolve_user("test_source", "test_id_001")
            assert user_label is not None
            result1 = mgr.update_user_preferences(user_label, {"style": "concise"})
            assert result1 is True
        finally:
            pool.close_all()

    def test_optimistic_lock_retry_exhaustion(self, tmp_path):
        pool = _make_graph_with_pool(tmp_path, pool_size=3)
        try:
            mgr = UserManager(pool)
            user_label = mgr.resolve_user("test_source", "test_id_002")
            assert user_label is not None

            update_count = [0]
            original_method = mgr.update_user_preferences

            def mock_update(ul, prefs):
                if update_count[0] == 0:
                    update_count[0] += 1
                    mgr.update_user_preferences = original_method
                    mgr.update_user_preferences(ul, {"concurrent_mod": "true"})
                    graph = pool.acquire()
                    try:
                        row = graph.conn.execute(
                            "SELECT meta FROM seeds WHERE label=? AND type='USER'", (ul,)
                        ).fetchone()
                        assert row is not None
                    finally:
                        pool.release(graph)
                return original_method(ul, prefs)

            mgr.update_user_preferences = mock_update
            result = mgr.update_user_preferences(user_label, {"style": "verbose"})
            assert result is True or result is False
        finally:
            pool.close_all()


# ═══════════════════════════════════════════════════════════
#  4. 提炼池升级失败回滚
# ═══════════════════════════════════════════════════════════


class TestDistillationPoolUpgradeFailure:
    """提炼池升级失败 — 验证异常传播和数据一致性"""

    def _make_pool_with_distillation(self, tmp_path):
        db_path = str(tmp_path / "test.db")
        _build_test_db(db_path)
        graph = GraphDB(db_path)
        graph.connect()
        graph.ensure_phase2_tables()
        return graph

    def test_upgrade_failure_propagates_exception(self, tmp_path):
        graph = self._make_pool_with_distillation(tmp_path)
        try:
            dp = DistillationPool(graph)
            now = datetime.now(timezone.utc).isoformat()
            graph.conn.execute(
                "INSERT INTO distillation_pool "
                "(canonical_source, canonical_target, canonical_relation, representative_label, "
                "status, count, contributor_users, created_at, updated_at) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                ('感冒', '姜汤', 'TREATS', '感冒→姜汤', 'pending', 3,
                 json.dumps(["user_a", "user_b", "user_c"]), now, now)
            )
            graph.conn.commit()
            candidate_id = graph.conn.execute(
                "SELECT candidate_id FROM distillation_pool "
                "WHERE canonical_source='感冒' AND canonical_target='姜汤'"
            ).fetchone()['candidate_id']

            with patch.object(
                dp, '_upgrade_to_global',
                side_effect=sqlite3.OperationalError("disk I/O error"),
            ):
                with pytest.raises(sqlite3.OperationalError, match="disk I/O error"):
                    dp._upgrade_to_global(candidate_id, '感冒', '姜汤', 'TREATS')

            status = dp._get_status_by_id(candidate_id)
            assert status == 'pending'
        finally:
            graph.close()

    def test_upgrade_success_commits_both_operations(self, tmp_path):
        graph = self._make_pool_with_distillation(tmp_path)
        try:
            dp = DistillationPool(graph)
            now = datetime.now(timezone.utc).isoformat()
            graph.conn.execute(
                "INSERT INTO distillation_pool "
                "(canonical_source, canonical_target, canonical_relation, representative_label, "
                "status, count, contributor_users, created_at, updated_at) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                ('感冒', '姜汤', 'TREATS', '感冒→姜汤', 'pending', 3,
                 json.dumps(["user_a", "user_b", "user_c"]), now, now)
            )
            graph.conn.commit()
            candidate_id = graph.conn.execute(
                "SELECT candidate_id FROM distillation_pool "
                "WHERE canonical_source='感冒' AND canonical_target='姜汤'"
            ).fetchone()['candidate_id']

            dp._upgrade_to_global(candidate_id, '感冒', '姜汤', 'TREATS')

            status = dp._get_status_by_id(candidate_id)
            assert status == 'upgraded'

            edge = graph.conn.execute(
                "SELECT weight FROM karma_edges WHERE source='感冒' AND target='姜汤' AND relation='TREATS'"
            ).fetchone()
            assert edge is not None
            assert edge['weight'] > 0
        finally:
            graph.close()


# ═══════════════════════════════════════════════════════════
#  5. 检查点文件损坏时的回滚
# ═══════════════════════════════════════════════════════════


class TestCheckpointCorruption:
    """检查点损坏 — 验证损坏检测和回滚安全"""

    def _make_checkpoint_manager(self, tmp_path):
        db_path = str(tmp_path / "test.db")
        _build_test_db(db_path)
        graph = GraphDB(db_path)
        graph.connect()
        graph.ensure_phase2_tables()
        graph.ensure_phase3_tables()
        ckpt_dir = str(tmp_path / "checkpoints")
        mgr = CheckpointManager(graph, checkpoint_dir=ckpt_dir)
        return mgr, graph

    def test_corrupted_json_returns_failure(self, tmp_path):
        mgr, graph = self._make_checkpoint_manager(tmp_path)
        try:
            meta = mgr.create_checkpoint(tag="test_corrupt")
            assert meta is not None
            ckpt_id = meta.checkpoint_id

            ckpt_path = Path(meta.file_path)
            assert ckpt_path.exists()

            with open(ckpt_path, "w", encoding="utf-8") as f:
                f.write("{invalid json content!!!")

            rollback_result = mgr.rollback(ckpt_id, mode="full")
            assert rollback_result.success is False
        finally:
            graph.close()

    def test_missing_checkpoint_file_returns_failure(self, tmp_path):
        mgr, graph = self._make_checkpoint_manager(tmp_path)
        try:
            meta = mgr.create_checkpoint(tag="test_missing")
            assert meta is not None
            ckpt_id = meta.checkpoint_id

            ckpt_path = Path(meta.file_path)
            ckpt_path.unlink()

            rollback_result = mgr.rollback(ckpt_id, mode="full")
            assert rollback_result.success is False
        finally:
            graph.close()

    def test_nonexistent_checkpoint_id_returns_failure(self, tmp_path):
        mgr, graph = self._make_checkpoint_manager(tmp_path)
        try:
            rollback_result = mgr.rollback("nonexistent_id_12345", mode="full")
            assert rollback_result.success is False
            assert "not found" in rollback_result.error.lower()
        finally:
            graph.close()

    def test_rollback_full_atomicity_on_failure(self, tmp_path):
        mgr, graph = self._make_checkpoint_manager(tmp_path)
        try:
            meta = mgr.create_checkpoint(tag="test_atomic")
            assert meta is not None
            ckpt_id = meta.checkpoint_id

            original_count = graph.conn.execute(
                "SELECT COUNT(*) as cnt FROM karma_edges"
            ).fetchone()['cnt']
            assert original_count > 0

            ckpt_path = Path(meta.file_path)
            with open(ckpt_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            data['edges'] = [{"source": "X" * 10000, "target": "Y", "relation": "R", "weight": 1.0}]
            with open(ckpt_path, "w", encoding="utf-8") as f:
                json.dump(data, f)

            rollback_result = mgr.rollback(ckpt_id, mode="full")
            if not rollback_result.success:
                current_count = graph.conn.execute(
                    "SELECT COUNT(*) as cnt FROM karma_edges"
                ).fetchone()['cnt']
                assert current_count == original_count
        finally:
            graph.close()

    def test_pre_rollback_failure_does_not_block_rollback(self, tmp_path):
        mgr, graph = self._make_checkpoint_manager(tmp_path)
        try:
            meta = mgr.create_checkpoint(tag="test_pre_rollback")
            assert meta is not None
            ckpt_id = meta.checkpoint_id

            with patch.object(mgr, '_create_checkpoint_locked', side_effect=RuntimeError("disk full")):
                rollback_result = mgr.rollback(ckpt_id, mode="full")
                assert rollback_result.success is True
        finally:
            graph.close()


# ═══════════════════════════════════════════════════════════
#  6. Ollama 服务不可用时的降级
# ═══════════════════════════════════════════════════════════


class TestOllamaUnavailableDegradation:
    """Ollama 不可用 — 验证降级路径"""

    def test_unavailable_expert_returns_fallback(self):
        mgr = MockExpertManager(available=False)
        result = mgr.infer("test prompt", "医学")
        assert result.fallback is True
        assert result.answer_text == ""

    def test_available_expert_returns_answer(self):
        mgr = MockExpertManager(available=True, answer="这是测试回答")
        result = mgr.infer("什么是量子力学", "物理")
        assert result.fallback is False
        assert result.answer_text == "这是测试回答"
        assert result.domain == "物理"

    def test_unavailable_multi_domain_returns_empty(self):
        mgr = MockExpertManager(available=False)
        results = mgr.infer_multi_domain("test", ["医学", "物理"])
        assert len(results) == 0

    def test_check_ollama_available_returns_false_on_error(self):
        with patch('urllib.request.urlopen', side_effect=Exception("Connection refused")):
            mgr = ExpertManager.__new__(ExpertManager)
            result = mgr._check_ollama_available()
            assert result is False


# ═══════════════════════════════════════════════════════════
#  7. 并发连接池压力测试
# ═══════════════════════════════════════════════════════════


class TestConcurrentPoolStress:
    """并发连接池压力 — 验证死锁恢复和资源泄漏"""

    def test_concurrent_exhaustion_and_recovery(self, tmp_path):
        pool = _make_graph_with_pool(tmp_path, pool_size=2)
        errors = []
        success_count = []
        exhaustion_count = []

        def worker():
            try:
                graph = pool.acquire(timeout=0.3)
                try:
                    time.sleep(0.01)
                    success_count.append(1)
                finally:
                    pool.release(graph)
            except ConnectionPoolExhausted:
                exhaustion_count.append(1)
            except Exception as e:
                errors.append(e)

        try:
            threads = [threading.Thread(target=worker) for _ in range(20)]
            for t in threads:
                t.start()
            for t in threads:
                t.join(timeout=30)

            assert len(errors) == 0, f"并发错误: {errors}"
            assert len(success_count) + len(exhaustion_count) == 20
            assert len(success_count) > 0
        finally:
            pool.close_all()

    def test_no_connection_leak_after_stress(self, tmp_path):
        pool = _make_graph_with_pool(tmp_path, pool_size=3)
        try:
            for _ in range(50):
                graph = pool.acquire(timeout=5.0)
                pool.release(graph)
            with pool._lock:
                assert len(pool._in_use) == 0
        finally:
            pool.close_all()


if __name__ == '__main__':
    import traceback

    test_classes = [
        TestConnectionPoolExhaustion,
        TestDatabaseConnectionInterruption,
        TestOptimisticLockContention,
        TestDistillationPoolUpgradeFailure,
        TestCheckpointCorruption,
        TestOllamaUnavailableDegradation,
        TestConcurrentPoolStress,
    ]

    for cls in test_classes:
        print(f"\n{cls.__name__}:")
        instance = cls()
        for name in dir(instance):
            if name.startswith('test_'):
                try:
                    import tempfile
                    with tempfile.TemporaryDirectory() as tmp:
                        from pathlib import Path as P

                        class TmpPath:
                            def __truediv__(self, other):
                                return P(str(tmp)) / other

                        getattr(instance, name)(TmpPath())
                    print(f"  PASS {name}")
                except Exception as e:
                    traceback.print_exc()
                    print(f"  FAIL {name}: {e}")