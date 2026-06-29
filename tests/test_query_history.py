"""
测试查询历史模块 (Query History)

覆盖:
- ensure_history_table 自动建表
- record_query 成功写入
- record_query 异常不阻塞
- get_history 分页查询
- query_text 截断
"""

import sqlite3
import sys
import pathlib
from unittest.mock import patch

_root = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_root))
sys.path.insert(0, str(_root / "backend" / "src"))

from consciousness_sea.domain.query_history import (
    ensure_history_table,
    record_query,
    get_history,
    MAX_QUERY_TEXT_LENGTH,
)


def _setup_db():
    """创建测试用内存数据库（不含 query_history 表）"""
    conn = sqlite3.connect(':memory:')
    conn.row_factory = sqlite3.Row
    return conn


class TestEnsureHistoryTable:
    """ensure_history_table 自动建表测试"""

    def setup_method(self):
        self.conn = _setup_db()

    def teardown_method(self):
        self.conn.close()

    def test_table_created(self):
        """首次调用自动创建 query_history 表"""
        ensure_history_table(self.conn)
        # 验证表存在
        result = self.conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='query_history'"
        ).fetchone()
        assert result is not None

    def test_index_created(self):
        """自动创建 idx_history_created_at 索引"""
        ensure_history_table(self.conn)
        result = self.conn.execute(
            "SELECT name FROM sqlite_master WHERE type='index' AND name='idx_history_created_at'"
        ).fetchone()
        assert result is not None

    def test_idempotent(self):
        """重复调用不报错（IF NOT EXISTS）"""
        ensure_history_table(self.conn)
        ensure_history_table(self.conn)  # 第二次调用不应抛异常
        result = self.conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='query_history'"
        ).fetchone()
        assert result is not None

    def test_table_columns(self):
        """表结构包含所有必要列"""
        ensure_history_table(self.conn)
        cursor = self.conn.execute("PRAGMA table_info(query_history)")
        columns = {row[1] for row in cursor.fetchall()}
        expected = {
            'query_id', 'query_text', 'matched_seeds_count',
            'selected_domains', 'confidence', 'karma_direction', 'created_at',
            'user_id',  # M-7: user_id 列
        }
        assert expected.issubset(columns)


class TestRecordQuery:
    """record_query 成功写入测试"""

    def setup_method(self):
        self.conn = _setup_db()
        ensure_history_table(self.conn)

    def teardown_method(self):
        self.conn.close()

    def test_record_success(self):
        """成功写入一条查询记录"""
        result = record_query(
            self.conn,
            query_text="感冒了怎么办",
            matched_seeds_count=3,
            selected_domains=["医学"],
            confidence=0.85,
            karma_direction=1,
        )
        assert result is True

        # 验证数据已写入
        row = self.conn.execute("SELECT * FROM query_history").fetchone()
        assert row is not None
        assert row['query_text'] == "感冒了怎么办"
        assert row['matched_seeds_count'] == 3
        assert row['confidence'] == 0.85
        assert row['karma_direction'] == 1

    def test_record_default_values(self):
        """默认值正确"""
        result = record_query(self.conn, query_text="测试查询")
        assert result is True

        row = self.conn.execute("SELECT * FROM query_history").fetchone()
        assert row['matched_seeds_count'] == 0
        assert row['confidence'] == 0.0
        assert row['karma_direction'] == 0

    def test_record_multiple(self):
        """连续写入多条记录"""
        for i in range(5):
            record_query(self.conn, query_text=f"查询{i}")

        count = self.conn.execute("SELECT COUNT(*) FROM query_history").fetchone()[0]
        assert count == 5

    def test_selected_domains_serialized(self):
        """selected_domains 序列化为 JSON"""
        record_query(
            self.conn,
            query_text="测试",
            selected_domains=["医学", "物理"],
        )
        row = self.conn.execute("SELECT selected_domains FROM query_history").fetchone()
        import json
        domains = json.loads(row['selected_domains'])
        assert domains == ["医学", "物理"]

    def test_created_at_format(self):
        """created_at 为 ISO 8601 格式"""
        record_query(self.conn, query_text="测试")
        row = self.conn.execute("SELECT created_at FROM query_history").fetchone()
        created_at = row['created_at']
        # ISO 8601 格式应包含 'T'
        assert 'T' in created_at or '-' in created_at


class TestRecordQueryException:
    """record_query 异常不阻塞测试"""

    def setup_method(self):
        self.conn = _setup_db()

    def teardown_method(self):
        self.conn.close()

    def test_closed_connection_returns_false(self):
        """连接关闭后写入返回 False"""
        ensure_history_table(self.conn)
        self.conn.close()
        # 使用已关闭的连接
        result = record_query(self.conn, query_text="测试")
        assert result is False

    def test_exception_does_not_raise(self):
        """异常不抛出，返回 False"""
        # 使用一个会导致 SQL 执行失败的场景：
        # 创建一个同名但结构不同的表，导致 INSERT 失败
        conn = sqlite3.connect(':memory:')
        # 创建一个只有单列的 query_history 表（缺少必要列）
        conn.execute("CREATE TABLE query_history (only_column TEXT)")
        conn.commit()

        # record_query 内部调用 ensure_history_table（IF NOT EXISTS 不重建）
        # 然后 INSERT 时会因列不匹配而失败
        result = record_query(conn, query_text="测试")
        assert result is False

        conn.close()


class TestGetHistory:
    """get_history 分页查询测试"""

    def setup_method(self):
        self.conn = _setup_db()
        ensure_history_table(self.conn)
        # 插入 25 条记录
        for i in range(25):
            record_query(
                self.conn,
                query_text=f"查询{i}",
                matched_seeds_count=i,
                confidence=0.5,
                karma_direction=1 if i % 2 == 0 else -1,
            )

    def teardown_method(self):
        self.conn.close()

    def test_default_pagination(self):
        """默认分页：limit=20, offset=0"""
        result = get_history(self.conn)
        assert result['total'] == 25
        assert len(result['records']) == 20
        assert result['limit'] == 20
        assert result['offset'] == 0

    def test_custom_limit(self):
        """自定义 limit"""
        result = get_history(self.conn, limit=5)
        assert len(result['records']) == 5
        assert result['limit'] == 5

    def test_offset_pagination(self):
        """offset 分页"""
        result = get_history(self.conn, limit=10, offset=20)
        assert len(result['records']) == 5  # 25 - 20 = 5
        assert result['offset'] == 20

    def test_order_desc(self):
        """按时间倒序排列"""
        result = get_history(self.conn, limit=5)
        # 最新的记录应该在前
        timestamps = [r['created_at'] for r in result['records']]
        assert timestamps == sorted(timestamps, reverse=True)

    def test_limit_capped_at_max(self):
        """limit 超过最大值时被裁剪"""
        result = get_history(self.conn, limit=200)
        assert result['limit'] == 100  # HISTORY_MAX_LIMIT

    def test_limit_minimum_1(self):
        """limit 最小为 1"""
        result = get_history(self.conn, limit=0)
        assert result['limit'] == 1

    def test_offset_minimum_0(self):
        """offset 最小为 0"""
        result = get_history(self.conn, offset=-5)
        assert result['offset'] == 0

    def test_empty_history(self):
        """空历史返回空列表"""
        conn = _setup_db()
        ensure_history_table(conn)
        result = get_history(conn)
        assert result['records'] == []
        assert result['total'] == 0
        conn.close()

    def test_record_fields(self):
        """返回记录包含所有字段"""
        result = get_history(self.conn, limit=1)
        record = result['records'][0]
        assert 'query_id' in record
        assert 'query_text' in record
        assert 'matched_seeds_count' in record
        assert 'selected_domains' in record
        assert 'confidence' in record
        assert 'karma_direction' in record
        assert 'created_at' in record
        assert 'user_id' in record  # M-7: user_id 字段


class TestQueryTextTruncation:
    """query_text 截断测试"""

    def setup_method(self):
        self.conn = _setup_db()
        ensure_history_table(self.conn)

    def teardown_method(self):
        self.conn.close()

    def test_short_text_not_truncated(self):
        """短文本不被截断"""
        short_text = "感冒"
        record_query(self.conn, query_text=short_text)
        row = self.conn.execute("SELECT query_text FROM query_history").fetchone()
        assert row['query_text'] == short_text

    def test_long_text_truncated(self):
        """超长文本被截断为 MAX_QUERY_TEXT_LENGTH 字符"""
        long_text = "测" * (MAX_QUERY_TEXT_LENGTH + 500)
        record_query(self.conn, query_text=long_text)
        row = self.conn.execute("SELECT query_text FROM query_history").fetchone()
        assert len(row['query_text']) == MAX_QUERY_TEXT_LENGTH

    def test_exact_length_not_truncated(self):
        """恰好 MAX_QUERY_TEXT_LENGTH 长度不截断"""
        exact_text = "测" * MAX_QUERY_TEXT_LENGTH
        record_query(self.conn, query_text=exact_text)
        row = self.conn.execute("SELECT query_text FROM query_history").fetchone()
        assert len(row['query_text']) == MAX_QUERY_TEXT_LENGTH

    def test_max_query_text_length_value(self):
        """MAX_QUERY_TEXT_LENGTH 为 1000"""
        assert MAX_QUERY_TEXT_LENGTH == 1000

    def test_empty_text_handled(self):
        """空字符串正常处理"""
        record_query(self.conn, query_text="")
        row = self.conn.execute("SELECT query_text FROM query_history").fetchone()
        assert row['query_text'] == ""


if __name__ == '__main__':
    import traceback
    classes = [
        TestEnsureHistoryTable, TestRecordQuery, TestRecordQueryException,
        TestGetHistory, TestQueryTextTruncation,
    ]
    for cls in classes:
        t = cls()
        for name in dir(t):
            if name.startswith('test_'):
                t.setup_method()
                try:
                    getattr(t, name)()
                    print(f"  [PASS] {cls.__name__}.{name}")
                except Exception as e:
                    print(f"  [FAIL] {cls.__name__}.{name}: {e}")
                    traceback.print_exc()
                t.teardown_method()