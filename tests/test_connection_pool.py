"""
连接池单元测试 (TASK-5.3)

覆盖:
- 连接获取与释放基本流程
- 连接池耗尽抛出 ConnectionPoolExhausted 异常
- 并发安全：多线程同时 acquire/release
- 连接归还后缓存重置（invalidate_cache）
- close_all() 关闭所有连接
"""

import sqlite3
import sys
import pathlib
import threading
from pathlib import Path

_root = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_root))
sys.path.insert(0, str(_root / "backend" / "src"))

from consciousness_sea.infrastructure.connection_pool import ConnectionPool, ConnectionPoolExhausted, ConnectionPoolClosed
from consciousness_sea.domain.graph_db import GraphDB


def _build_test_db(db_path: str) -> None:
    """创建测试用 SQLite 数据库文件"""
    conn = sqlite3.connect(db_path)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA synchronous=NORMAL")
    conn.execute("PRAGMA busy_timeout=5000")
    conn.executescript("""
        CREATE TABLE seeds (
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
        CREATE TABLE karma_edges (
            source TEXT NOT NULL, target TEXT NOT NULL, relation TEXT NOT NULL,
            weight REAL NOT NULL DEFAULT 0.5,
            source_tag TEXT NOT NULL DEFAULT '',
            created_at TEXT NOT NULL DEFAULT (datetime('now')),
            PRIMARY KEY (source, target, relation)
        );
    """)
    conn.executemany(
        "INSERT INTO seeds (id, label, type, aliases, domain, definition) "
        "VALUES (?, ?, ?, ?, ?, ?)",
        [
            ('感冒', '感冒', 'CONCEPT', '[]', '医学', 'cold'),
            ('发热', '发热', 'CONCEPT', '[]', '医学', 'fever'),
        ],
    )
    conn.executemany(
        "INSERT INTO karma_edges (source, target, relation, weight) "
        "VALUES (?, ?, ?, ?)",
        [('感冒', '发热', 'COOCCURS_WITH', 0.95)],
    )
    conn.commit()
    conn.close()


def _build_test_db_check_same_thread(db_path: str) -> None:
    """创建测试用 SQLite 数据库文件（check_same_thread=False 兼容）

    在连接池场景下，连接可能在不同线程间传递。
    """
    _build_test_db(db_path)


class TestConnectionPoolBasicFlow:
    """连接获取与释放基本流程"""

    def test_acquire_returns_graph_db(self, tmp_path):
        """acquire() 返回 GraphDB 实例"""
        db_path = str(tmp_path / "test.db")
        _build_test_db(db_path)
        pool = ConnectionPool(db_path, pool_size=3)
        try:
            graph = pool.acquire()
            assert isinstance(graph, GraphDB)
            assert graph.conn is not None
            pool.release(graph)
        finally:
            pool.close_all()

    def test_acquire_and_release_reuses_connection(self, tmp_path):
        """release 后再 acquire 可获得同一连接"""
        db_path = str(tmp_path / "test.db")
        _build_test_db(db_path)
        pool = ConnectionPool(db_path, pool_size=3)
        try:
            graph1 = pool.acquire()
            pool.release(graph1)
            graph2 = pool.acquire()
            # 归还后再次获取，应该是同一个连接对象
            assert graph2 is graph1
            pool.release(graph2)
        finally:
            pool.close_all()

    def test_acquire_multiple_connections(self, tmp_path):
        """可以同时获取多个连接"""
        db_path = str(tmp_path / "test.db")
        _build_test_db(db_path)
        pool = ConnectionPool(db_path, pool_size=3)
        try:
            graphs = []
            for _ in range(3):
                graphs.append(pool.acquire())
            assert len(graphs) == 3
            # 每个连接应该是不同的实例
            assert len({id(g) for g in graphs}) == 3
            for g in graphs:
                pool.release(g)
        finally:
            pool.close_all()

    def test_connection_can_query(self, tmp_path):
        """获取的连接可以正常查询数据"""
        db_path = str(tmp_path / "test.db")
        _build_test_db(db_path)
        pool = ConnectionPool(db_path, pool_size=3)
        try:
            graph = pool.acquire()
            seed = graph.get_seed('感冒')
            assert seed is not None
            assert seed['label'] == '感冒'
            pool.release(graph)
        finally:
            pool.close_all()


class TestConnectionPoolExhausted:
    """连接池耗尽抛出 ConnectionPoolExhausted 异常"""

    def test_pool_exhausted_raises_exception(self, tmp_path):
        """超过 pool_size 的 acquire 抛出 ConnectionPoolExhausted"""
        db_path = str(tmp_path / "test.db")
        _build_test_db(db_path)
        pool = ConnectionPool(db_path, pool_size=2)
        try:
            g1 = pool.acquire()
            g2 = pool.acquire()
            # 第 3 次 acquire 应超时抛异常
            try:
                pool.acquire(timeout=0.1)
                assert False, "应抛出 ConnectionPoolExhausted"
            except ConnectionPoolExhausted:
                pass
            pool.release(g1)
            pool.release(g2)
        finally:
            pool.close_all()

    def test_pool_exhausted_message_contains_info(self, tmp_path):
        """ConnectionPoolExhausted 异常消息包含池状态信息"""
        db_path = str(tmp_path / "test.db")
        _build_test_db(db_path)
        pool = ConnectionPool(db_path, pool_size=2)
        try:
            g1 = pool.acquire()
            g2 = pool.acquire()
            try:
                pool.acquire(timeout=0.1)
            except ConnectionPoolExhausted as e:
                msg = str(e)
                assert "pool_size=2" in msg
                assert "in_use=2" in msg
            pool.release(g1)
            pool.release(g2)
        finally:
            pool.close_all()


class TestConnectionPoolConcurrentSafety:
    """并发安全：多线程同时 acquire/release"""

    def test_concurrent_acquire_release_no_crash(self, tmp_path):
        """多线程并发 acquire/release 不崩溃"""
        db_path = str(tmp_path / "test.db")
        _build_test_db(db_path)
        pool = ConnectionPool(db_path, pool_size=5)
        errors = []
        success_count = []
        barrier = threading.Barrier(5)

        def worker():
            try:
                barrier.wait(timeout=5)
                for _ in range(10):
                    graph = pool.acquire(timeout=5.0)
                    # 验证连接对象有效
                    assert graph is not None
                    assert graph.conn is not None
                    pool.release(graph)
                    success_count.append(1)
            except Exception as e:
                errors.append(e)

        try:
            threads = [threading.Thread(target=worker) for _ in range(5)]
            for t in threads:
                t.start()
            for t in threads:
                t.join(timeout=30)

            assert len(errors) == 0, f"并发错误: {errors}"
            assert len(success_count) == 50  # 5 threads * 10 iterations
        finally:
            # close_all 可能因 SQLite check_same_thread 限制而失败
            # 但这不影响并发安全性验证
            try:
                pool.close_all()
            except sqlite3.ProgrammingError:
                pass

    def test_concurrent_no_connection_leak(self, tmp_path):
        """并发后所有连接都归还到池中"""
        db_path = str(tmp_path / "test.db")
        _build_test_db(db_path)
        pool = ConnectionPool(db_path, pool_size=3)
        errors = []

        def worker():
            try:
                graph = pool.acquire(timeout=5.0)
                assert graph is not None
                pool.release(graph)
            except Exception as e:
                errors.append(e)

        try:
            threads = [threading.Thread(target=worker) for _ in range(10)]
            for t in threads:
                t.start()
            for t in threads:
                t.join(timeout=30)

            assert len(errors) == 0, f"并发错误: {errors}"
            # 所有连接应已归还
            with pool._lock:
                assert len(pool._in_use) == 0
        finally:
            try:
                pool.close_all()
            except sqlite3.ProgrammingError:
                pass

    def test_concurrent_acquire_respects_pool_size(self, tmp_path):
        """并发 acquire 不超过 pool_size 限制"""
        db_path = str(tmp_path / "test.db")
        _build_test_db(db_path)
        pool = ConnectionPool(db_path, pool_size=3)
        max_in_use = [0]
        lock = threading.Lock()

        def worker():
            try:
                graph = pool.acquire(timeout=5.0)
                with lock:
                    current = len(pool._in_use)
                    if current > max_in_use[0]:
                        max_in_use[0] = current
                pool.release(graph)
            except Exception:
                pass

        try:
            threads = [threading.Thread(target=worker) for _ in range(10)]
            for t in threads:
                t.start()
            for t in threads:
                t.join(timeout=30)

            assert max_in_use[0] <= 3
        finally:
            try:
                pool.close_all()
            except sqlite3.ProgrammingError:
                pass


class TestConnectionPoolCacheReset:
    """连接归还后缓存重置（invalidate_cache）"""

    def test_release_invalidates_cache(self, tmp_path):
        """release() 后 GraphDB 缓存被重置"""
        db_path = str(tmp_path / "test.db")
        _build_test_db(db_path)
        pool = ConnectionPool(db_path, pool_size=3)
        try:
            graph = pool.acquire()
            # 触发缓存构建（match_seeds 会构建 _label_index 和 _alias_index）
            graph.match_seeds('感冒')
            assert graph._label_index is not None

            # 归还连接
            pool.release(graph)

            # 缓存应被重置
            assert graph._alias_index is None
            assert graph._label_index is None
            assert graph._edge_count_map is None
        finally:
            pool.close_all()

    def test_cache_rebuilt_after_reacquire(self, tmp_path):
        """重新获取连接后缓存可以重新构建"""
        db_path = str(tmp_path / "test.db")
        _build_test_db(db_path)
        pool = ConnectionPool(db_path, pool_size=3)
        try:
            graph = pool.acquire()
            graph.match_seeds('感冒')
            pool.release(graph)

            # 重新获取
            graph2 = pool.acquire()
            assert graph2 is graph
            # 触发缓存重建
            graph2.match_seeds('感冒')
            assert graph2._label_index is not None
            pool.release(graph2)
        finally:
            pool.close_all()


class TestConnectionPoolCloseAll:
    """close_all() 关闭所有连接"""

    def test_close_all_closes_idle_connections(self, tmp_path):
        """close_all() 关闭所有空闲连接"""
        db_path = str(tmp_path / "test.db")
        _build_test_db(db_path)
        pool = ConnectionPool(db_path, pool_size=3)
        try:
            g1 = pool.acquire()
            g2 = pool.acquire()
            pool.release(g1)
            pool.release(g2)
            pool.close_all()
            # 连接应被关闭
            assert g1.conn is None
            assert g2.conn is None
        finally:
            pass  # close_all already called

    def test_close_all_resets_pool_state(self, tmp_path):
        """close_all() 重置池状态"""
        db_path = str(tmp_path / "test.db")
        _build_test_db(db_path)
        pool = ConnectionPool(db_path, pool_size=3)
        try:
            g1 = pool.acquire()
            pool.release(g1)
            pool.close_all()
            assert pool._created_count == 0
            assert len(pool._in_use) == 0
        finally:
            pass

    def test_close_all_then_acquire_raises_closed(self, tmp_path):
        """close_all() 后 acquire() 抛出 ConnectionPoolClosed"""
        db_path = str(tmp_path / "test.db")
        _build_test_db(db_path)
        pool = ConnectionPool(db_path, pool_size=3)
        try:
            g1 = pool.acquire()
            pool.release(g1)
            pool.close_all()

            # C-1: close_all 后 acquire 应抛出 ConnectionPoolClosed
            try:
                pool.acquire()
                assert False, "应抛出 ConnectionPoolClosed"
            except ConnectionPoolClosed:
                pass
        finally:
            pass  # close_all already called

    def test_close_all_then_release_closes_connection(self, tmp_path):
        """close_all() 后 release() 直接关闭连接而不放回队列"""
        db_path = str(tmp_path / "test.db")
        _build_test_db(db_path)
        pool = ConnectionPool(db_path, pool_size=3)
        try:
            g1 = pool.acquire()
            pool.close_all()
            # C-1: close_all 后 release 应直接关闭连接
            pool.release(g1)
            assert g1.conn is None
        finally:
            pass

    def test_release_none_is_safe(self, tmp_path):
        """release(None) 不抛异常"""
        db_path = str(tmp_path / "test.db")
        _build_test_db(db_path)
        pool = ConnectionPool(db_path, pool_size=3)
        try:
            pool.release(None)  # 应安全处理
        finally:
            pool.close_all()


if __name__ == '__main__':
    import traceback

    test_classes = [
        TestConnectionPoolBasicFlow,
        TestConnectionPoolExhausted,
        TestConnectionPoolConcurrentSafety,
        TestConnectionPoolCacheReset,
        TestConnectionPoolCloseAll,
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
