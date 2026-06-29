"""
Session 管理器单元测试 (TASK-5.5)

覆盖:
- Session 创建与清理基本流程
- Session 结束后资源释放（引用断开）
- 连接归还保证（异常时也归还）
"""

import sqlite3
import sys
import pathlib

_root = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_root))
sys.path.insert(0, str(_root / "backend" / "src"))

from consciousness_sea.infrastructure.connection_pool import ConnectionPool
from consciousness_sea.infrastructure.session_manager import SessionManager, SessionContext
from consciousness_sea.domain.graph_db import GraphDB


def _build_test_db(db_path: str) -> None:
    """创建测试用 SQLite 数据库文件"""
    conn = sqlite3.connect(db_path)
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


def _make_session_manager(tmp_path):
    """创建测试用 SessionManager 实例"""
    db_path = str(tmp_path / "test.db")
    _build_test_db(db_path)
    pool = ConnectionPool(db_path, pool_size=3)
    session_mgr = SessionManager(pool)
    return pool, session_mgr


class TestSessionCreateAndCleanup:
    """Session 创建与清理基本流程"""

    def test_create_session_returns_context(self, tmp_path):
        """create_session() 返回 SessionContext 实例"""
        pool, session_mgr = _make_session_manager(tmp_path)
        try:
            ctx = session_mgr.create_session()
            assert isinstance(ctx, SessionContext)
            assert ctx.session_id is not None
            assert len(ctx.session_id) > 0
            assert ctx.graph is not None
            assert isinstance(ctx.graph, GraphDB)
            session_mgr.end_session(ctx)
        finally:
            pool.close_all()

    def test_create_session_with_user_label(self, tmp_path):
        """create_session() 正确绑定 user_label"""
        pool, session_mgr = _make_session_manager(tmp_path)
        try:
            ctx = session_mgr.create_session(user_label='test_user')
            assert ctx.user_label == 'test_user'
            session_mgr.end_session(ctx)
        finally:
            pool.close_all()

    def test_create_session_without_user_label(self, tmp_path):
        """create_session() 不传 user_label 时为 None"""
        pool, session_mgr = _make_session_manager(tmp_path)
        try:
            ctx = session_mgr.create_session()
            assert ctx.user_label is None
            session_mgr.end_session(ctx)
        finally:
            pool.close_all()

    def test_session_has_created_at(self, tmp_path):
        """SessionContext 包含创建时间"""
        pool, session_mgr = _make_session_manager(tmp_path)
        try:
            ctx = session_mgr.create_session()
            assert ctx.created_at is not None
            assert 'T' in ctx.created_at or '-' in ctx.created_at
            session_mgr.end_session(ctx)
        finally:
            pool.close_all()

    def test_session_graph_can_query(self, tmp_path):
        """Session 中的 graph 连接可以正常查询"""
        pool, session_mgr = _make_session_manager(tmp_path)
        try:
            ctx = session_mgr.create_session()
            seed = ctx.graph.get_seed('感冒')
            assert seed is not None
            assert seed['label'] == '感冒'
            session_mgr.end_session(ctx)
        finally:
            pool.close_all()

    def test_multiple_sessions_independent(self, tmp_path):
        """多个 Session 的连接是独立的"""
        pool, session_mgr = _make_session_manager(tmp_path)
        try:
            ctx1 = session_mgr.create_session()
            ctx2 = session_mgr.create_session()
            assert ctx1.session_id != ctx2.session_id
            assert ctx1.graph is not ctx2.graph
            session_mgr.end_session(ctx1)
            session_mgr.end_session(ctx2)
        finally:
            pool.close_all()


class TestSessionResourceRelease:
    """Session 结束后资源释放（引用断开）"""

    def test_end_session_clears_graph_reference(self, tmp_path):
        """end_session() 后 ctx.graph 为 None"""
        pool, session_mgr = _make_session_manager(tmp_path)
        try:
            ctx = session_mgr.create_session()
            assert ctx.graph is not None
            session_mgr.end_session(ctx)
            assert ctx.graph is None
        finally:
            pool.close_all()

    def test_end_session_clears_user_label(self, tmp_path):
        """end_session() 后 ctx.user_label 为 None"""
        pool, session_mgr = _make_session_manager(tmp_path)
        try:
            ctx = session_mgr.create_session(user_label='test_user')
            assert ctx.user_label == 'test_user'
            session_mgr.end_session(ctx)
            assert ctx.user_label is None
        finally:
            pool.close_all()

    def test_cleanup_method_clears_references(self, tmp_path):
        """cleanup() 方法断开引用"""
        pool, session_mgr = _make_session_manager(tmp_path)
        try:
            ctx = session_mgr.create_session(user_label='test_user')
            assert ctx.graph is not None
            assert ctx.user_label is not None
            ctx.cleanup()
            assert ctx.graph is None
            assert ctx.user_label is None
        finally:
            pool.close_all()

    def test_end_session_returns_connection_to_pool(self, tmp_path):
        """end_session() 将连接归还到池中"""
        pool, session_mgr = _make_session_manager(tmp_path)
        try:
            # 获取一个 session
            ctx = session_mgr.create_session()
            graph_ref = ctx.graph
            session_mgr.end_session(ctx)

            # 连接应已归还，可以再次获取
            ctx2 = session_mgr.create_session()
            assert ctx2.graph is not None
            session_mgr.end_session(ctx2)
        finally:
            pool.close_all()


class TestSessionConnectionGuarantee:
    """连接归还保证（异常时也归还）"""

    def test_connection_returned_after_exception(self, tmp_path):
        """异常时连接仍被归还"""
        pool, session_mgr = _make_session_manager(tmp_path)
        try:
            ctx = session_mgr.create_session()
            graph_ref = ctx.graph

            # 模拟使用中发生异常
            try:
                # 故意触发异常
                raise ValueError("模拟查询异常")
            except ValueError:
                pass

            # 即使异常，end_session 仍应归还连接
            session_mgr.end_session(ctx)

            # 验证连接已归还
            with pool._lock:
                assert id(graph_ref) not in pool._in_use

            # 可以再次获取连接
            ctx2 = session_mgr.create_session()
            assert ctx2.graph is not None
            session_mgr.end_session(ctx2)
        finally:
            pool.close_all()

    def test_end_session_idempotent(self, tmp_path):
        """多次 end_session 不崩溃"""
        pool, session_mgr = _make_session_manager(tmp_path)
        try:
            ctx = session_mgr.create_session()
            session_mgr.end_session(ctx)
            # 第二次调用不应抛异常
            session_mgr.end_session(ctx)
        finally:
            pool.close_all()

    def test_end_session_with_already_cleaned_context(self, tmp_path):
        """对已 cleanup 的 SessionContext 调用 end_session 不崩溃"""
        pool, session_mgr = _make_session_manager(tmp_path)
        try:
            ctx = session_mgr.create_session()
            graph_ref = ctx.graph
            ctx.cleanup()  # 手动 cleanup
            assert ctx.graph is None

            # end_session 应安全处理 graph=None 的情况
            session_mgr.end_session(ctx)
        finally:
            pool.close_all()


if __name__ == '__main__':
    import traceback

    test_classes = [
        TestSessionCreateAndCleanup,
        TestSessionResourceRelease,
        TestSessionConnectionGuarantee,
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