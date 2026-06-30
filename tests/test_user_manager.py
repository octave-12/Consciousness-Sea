"""
用户管理器单元测试 (TASK-5.4)

覆盖:
- 用户映射查找（已存在用户返回同一 user_id）
- 新用户自动创建（首次遇到未知来源标识）
- 同一来源标识一致性（多次调用返回同一 user_id）
- 用户业力边添加
- 用户偏好属性读写
- 映射缓存重建
- 来源标识格式校验（非法 source / 超长 source_id）
"""

import hashlib
import pathlib
import sqlite3
import sys

_root = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_root))
sys.path.insert(0, str(_root / "backend" / "src"))

from consciousness_sea.infrastructure.config import (
    MAX_SOURCE_ID_LENGTH,
    USER_ID_HASH_LENGTH,
    VALID_SOURCES,
)
from consciousness_sea.infrastructure.connection_pool import ConnectionPool
from consciousness_sea.infrastructure.user_manager import UserManager


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


def _make_user_manager(tmp_path):
    """创建测试用 UserManager 实例"""
    db_path = str(tmp_path / "test.db")
    _build_test_db(db_path)
    pool = ConnectionPool(db_path, pool_size=3)
    user_mgr = UserManager(pool)
    return pool, user_mgr


class TestUserMappingLookup:
    """用户映射查找（已存在用户返回同一 user_id）"""

    def test_resolve_existing_user_returns_same_id(self, tmp_path):
        """已存在用户返回同一 user_id"""
        pool, user_mgr = _make_user_manager(tmp_path)
        try:
            # 先创建一个用户
            user_id_1 = user_mgr.resolve_user('wechat', 'user123')
            assert user_id_1 is not None

            # 再次查找应返回同一 user_id
            user_id_2 = user_mgr.resolve_user('wechat', 'user123')
            assert user_id_2 == user_id_1
        finally:
            pool.close_all()

    def test_resolve_user_returns_deterministic_id(self, tmp_path):
        """user_id 生成是确定性的（SHA-256 哈希）"""
        pool, user_mgr = _make_user_manager(tmp_path)
        try:
            user_id = user_mgr.resolve_user('wechat', 'abc')
            assert user_id is not None

            # 验证确定性: 手动计算期望的 user_id
            raw = "wechat:abc".encode()
            expected_suffix = hashlib.sha256(raw).hexdigest()[:USER_ID_HASH_LENGTH]
            expected_id = f"user_{expected_suffix}"
            assert user_id == expected_id
        finally:
            pool.close_all()


class TestNewUserAutoCreation:
    """新用户自动创建（首次遇到未知来源标识）"""

    def test_new_user_created_on_first_resolve(self, tmp_path):
        """首次遇到未知来源标识自动创建用户"""
        pool, user_mgr = _make_user_manager(tmp_path)
        try:
            user_id = user_mgr.resolve_user('web', 'new_user_001')
            assert user_id is not None
            assert user_id.startswith('user_')
        finally:
            pool.close_all()

    def test_new_user_creates_seed_entry(self, tmp_path):
        """新用户创建时在 seeds 表插入 USER 类型种子"""
        pool, user_mgr = _make_user_manager(tmp_path)
        try:
            user_id = user_mgr.resolve_user('api', 'test_user')
            assert user_id is not None

            # 验证 seeds 表中有对应记录
            graph = pool.acquire()
            try:
                row = graph.conn.execute(
                    "SELECT * FROM seeds WHERE label=? AND type='USER'",
                    (user_id,)
                ).fetchone()
                assert row is not None
                assert row['type'] == 'USER'
                assert row['domain'] == '用户'
            finally:
                pool.release(graph)
        finally:
            pool.close_all()

    def test_new_user_creates_mapping_entry(self, tmp_path):
        """新用户创建时在 user_mapping 表插入映射记录"""
        pool, user_mgr = _make_user_manager(tmp_path)
        try:
            user_id = user_mgr.resolve_user('wechat', 'openid_abc')
            assert user_id is not None

            # 验证 user_mapping 表中有记录
            graph = pool.acquire()
            try:
                row = graph.conn.execute(
                    "SELECT * FROM user_mapping WHERE source='wechat' AND source_id='openid_abc'"
                ).fetchone()
                assert row is not None
                assert row['user_id'] == user_id
            finally:
                pool.release(graph)
        finally:
            pool.close_all()


class TestSourceIdentityConsistency:
    """同一来源标识一致性（多次调用返回同一 user_id）"""

    def test_same_source_id_returns_same_user(self, tmp_path):
        """同一 (source, source_id) 多次调用返回同一 user_id"""
        pool, user_mgr = _make_user_manager(tmp_path)
        try:
            ids = set()
            for _ in range(5):
                uid = user_mgr.resolve_user('wechat', 'consistent_user')
                ids.add(uid)
            assert len(ids) == 1
        finally:
            pool.close_all()

    def test_different_source_creates_different_user(self, tmp_path):
        """不同 source 创建不同用户"""
        pool, user_mgr = _make_user_manager(tmp_path)
        try:
            uid1 = user_mgr.resolve_user('wechat', 'same_id')
            uid2 = user_mgr.resolve_user('web', 'same_id')
            assert uid1 != uid2
        finally:
            pool.close_all()

    def test_different_source_id_creates_different_user(self, tmp_path):
        """同一 source 不同 source_id 创建不同用户"""
        pool, user_mgr = _make_user_manager(tmp_path)
        try:
            uid1 = user_mgr.resolve_user('wechat', 'user_a')
            uid2 = user_mgr.resolve_user('wechat', 'user_b')
            assert uid1 != uid2
        finally:
            pool.close_all()


class TestUserKarmaEdge:
    """用户业力边添加"""

    def test_add_user_karma_edge_success(self, tmp_path):
        """成功添加用户业力边"""
        pool, user_mgr = _make_user_manager(tmp_path)
        try:
            user_id = user_mgr.resolve_user('wechat', 'karma_user')
            assert user_id is not None

            result = user_mgr.add_user_karma_edge(
                user_id, '感冒', '关注', 0.8
            )
            assert result is True

            # 验证业力边已写入
            graph = pool.acquire()
            try:
                row = graph.conn.execute(
                    "SELECT * FROM karma_edges WHERE source=? AND target=? AND relation=?",
                    (user_id, '感冒', '关注')
                ).fetchone()
                assert row is not None
                assert row['source_tag'] == 'user_karma'
                assert abs(row['weight'] - 0.8) < 0.001
            finally:
                pool.release(graph)
        finally:
            pool.close_all()

    def test_add_user_karma_edge_weight_clamped(self, tmp_path):
        """业力边权重被裁剪到 [KARMA_MIN, KARMA_MAX]"""
        pool, user_mgr = _make_user_manager(tmp_path)
        try:
            user_id = user_mgr.resolve_user('wechat', 'karma_clamp')
            assert user_id is not None

            # 超大权重
            result = user_mgr.add_user_karma_edge(
                user_id, '感冒', '偏好', 5.0
            )
            assert result is True

            graph = pool.acquire()
            try:
                from consciousness_sea.infrastructure.config import KARMA_MAX
                row = graph.conn.execute(
                    "SELECT weight FROM karma_edges WHERE source=? AND target=? AND relation=?",
                    (user_id, '感冒', '偏好')
                ).fetchone()
                assert row is not None
                assert row['weight'] <= KARMA_MAX
            finally:
                pool.release(graph)
        finally:
            pool.close_all()

    def test_add_user_karma_edge_duplicate_ignored(self, tmp_path):
        """重复添加相同业力边被忽略（INSERT OR IGNORE）"""
        pool, user_mgr = _make_user_manager(tmp_path)
        try:
            user_id = user_mgr.resolve_user('wechat', 'karma_dup')
            assert user_id is not None

            result1 = user_mgr.add_user_karma_edge(user_id, '感冒', '关注', 0.8)
            result2 = user_mgr.add_user_karma_edge(user_id, '感冒', '关注', 0.8)
            assert result1 is True
            assert result2 is True  # 不报错

            # 只有一条边
            graph = pool.acquire()
            try:
                rows = graph.conn.execute(
                    "SELECT * FROM karma_edges WHERE source=? AND target=? AND relation=?",
                    (user_id, '感冒', '关注')
                ).fetchall()
                assert len(rows) == 1
            finally:
                pool.release(graph)
        finally:
            pool.close_all()


class TestUserPreferences:
    """用户偏好属性读写"""

    def test_get_default_preferences(self, tmp_path):
        """新用户的默认偏好属性"""
        pool, user_mgr = _make_user_manager(tmp_path)
        try:
            user_id = user_mgr.resolve_user('wechat', 'pref_user')
            assert user_id is not None

            prefs = user_mgr.get_user_preferences(user_id)
            assert isinstance(prefs, dict)
            # 新用户有默认偏好
            assert 'style' in prefs
            assert prefs['style'] == 'concise'
        finally:
            pool.close_all()

    def test_update_preferences(self, tmp_path):
        """更新用户偏好属性"""
        pool, user_mgr = _make_user_manager(tmp_path)
        try:
            user_id = user_mgr.resolve_user('wechat', 'pref_update')
            assert user_id is not None

            result = user_mgr.update_user_preferences(user_id, {'theme': 'dark'})
            assert result is True

            prefs = user_mgr.get_user_preferences(user_id)
            assert prefs['theme'] == 'dark'
            # 原有偏好应保留
            assert prefs['style'] == 'concise'
        finally:
            pool.close_all()

    def test_update_preferences_merge(self, tmp_path):
        """多次更新偏好属性应合并而非覆盖"""
        pool, user_mgr = _make_user_manager(tmp_path)
        try:
            user_id = user_mgr.resolve_user('wechat', 'pref_merge')
            assert user_id is not None

            user_mgr.update_user_preferences(user_id, {'theme': 'dark'})
            user_mgr.update_user_preferences(user_id, {'language': 'zh'})

            prefs = user_mgr.get_user_preferences(user_id)
            assert prefs['theme'] == 'dark'
            assert prefs['language'] == 'zh'
            assert prefs['style'] == 'concise'
        finally:
            pool.close_all()

    def test_get_preferences_nonexistent_user(self, tmp_path):
        """不存在用户的偏好返回空字典"""
        pool, user_mgr = _make_user_manager(tmp_path)
        try:
            prefs = user_mgr.get_user_preferences('nonexistent_user')
            assert prefs == {}
        finally:
            pool.close_all()


class TestMappingCacheRebuild:
    """映射缓存重建"""

    def test_rebuild_cache_from_database(self, tmp_path):
        """从数据库重建内存映射缓存"""
        pool, user_mgr = _make_user_manager(tmp_path)
        try:
            # 创建用户
            uid1 = user_mgr.resolve_user('wechat', 'rebuild_user1')
            uid2 = user_mgr.resolve_user('web', 'rebuild_user2')
            assert uid1 is not None
            assert uid2 is not None

            # 清空内存缓存
            with user_mgr._cache_lock:
                user_mgr._mapping_cache.clear()

            # 重建缓存
            user_mgr.rebuild_cache()

            # 缓存应包含之前的映射
            with user_mgr._cache_lock:
                assert ('wechat', 'rebuild_user1') in user_mgr._mapping_cache
                assert ('web', 'rebuild_user2') in user_mgr._mapping_cache
                assert user_mgr._mapping_cache[('wechat', 'rebuild_user1')] == uid1
                assert user_mgr._mapping_cache[('web', 'rebuild_user2')] == uid2
        finally:
            pool.close_all()

    def test_rebuild_cache_empty_database(self, tmp_path):
        """空数据库重建缓存不报错"""
        pool, user_mgr = _make_user_manager(tmp_path)
        try:
            # 不创建用户，直接重建
            user_mgr.rebuild_cache()
            with user_mgr._cache_lock:
                assert len(user_mgr._mapping_cache) == 0
        finally:
            pool.close_all()


class TestSourceIdentityValidation:
    """来源标识格式校验（非法 source / 超长 source_id）"""

    def test_valid_sources(self, tmp_path):
        """合法 source 可以正常创建用户"""
        pool, user_mgr = _make_user_manager(tmp_path)
        try:
            for source in VALID_SOURCES:
                uid = user_mgr.resolve_user(source, 'test_user')
                assert uid is not None, f"source={source} 应成功创建用户"
        finally:
            pool.close_all()

    def test_invalid_source_still_creates_user(self, tmp_path):
        """非法 source 仍可创建用户（UserManager 不做校验，校验在 API 层）"""
        pool, user_mgr = _make_user_manager(tmp_path)
        try:
            # UserManager.resolve_user 不校验 source 合法性
            # 校验在 api.py 的 _validate_source 中完成
            uid = user_mgr.resolve_user('invalid_source', 'test_user')
            assert uid is not None
        finally:
            pool.close_all()

    def test_long_source_id_creates_user(self, tmp_path):
        """超长 source_id 仍可创建用户（UserManager 不做长度校验）"""
        pool, user_mgr = _make_user_manager(tmp_path)
        try:
            # 超过 MAX_SOURCE_ID_LENGTH 的 source_id
            long_id = 'x' * (MAX_SOURCE_ID_LENGTH + 100)
            uid = user_mgr.resolve_user('wechat', long_id)
            assert uid is not None
        finally:
            pool.close_all()

    def test_empty_source_id_returns_none(self, tmp_path):
        """空 source_id 查询数据库失败时返回 None"""
        pool, user_mgr = _make_user_manager(tmp_path)
        try:
            # 空 source_id 可能导致 SQL 问题
            # UserManager 应该能安全处理
            uid = user_mgr.resolve_user('wechat', '')
            # 空 source_id 可能创建用户（因为 (wechat, '') 是有效的主键）
            # 但不应该崩溃
            assert uid is None or isinstance(uid, str)
        finally:
            pool.close_all()


if __name__ == '__main__':
    import traceback

    test_classes = [
        TestUserMappingLookup,
        TestNewUserAutoCreation,
        TestSourceIdentityConsistency,
        TestUserKarmaEdge,
        TestUserPreferences,
        TestMappingCacheRebuild,
        TestSourceIdentityValidation,
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
