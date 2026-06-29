"""
用户管理器 — 用户标识映射 + 用户种子管理

将外部来源标识（如微信 openid）映射为识海内部 user_id，
并管理用户种子节点和用户业力边。

核心功能:
  - resolve_user(): 将 (source, source_id) 映射为 user_label
  - _create_user(): 自动创建新用户（确定性 ID 生成）
  - add_user_karma_edge(): 为用户种子添加业力边
  - get_user_preferences() / update_user_preferences(): 用户偏好属性管理
  - rebuild_cache(): 从数据库重建内存映射缓存
"""

from __future__ import annotations

import hashlib
import json
import logging
import threading
from datetime import datetime, timezone
from typing import Optional

from .config import (
    USER_ID_HASH_LENGTH,
    KARMA_MIN,
    KARMA_MAX,
)
from .connection_pool import ConnectionPool

log = logging.getLogger(__name__)


class UserManager:
    """用户标识映射 + 用户种子管理

    使用内存缓存加速映射查找，线程安全。
    用户 ID 生成规则：user_{SHA-256(source:source_id)[:8]}，确定性且唯一。

    Args:
        pool: 连接池实例
    """

    def __init__(self, pool: ConnectionPool):
        self._pool = pool
        self._mapping_cache: dict[tuple[str, str], str] = {}  # (source, source_id) → user_id
        self._cache_lock = threading.Lock()
        self._table_ensured = False  # M-5: 首次执行后跳过建表

    def _ensure_user_mapping_table(self) -> None:
        """自动创建 user_mapping 表（含联合主键 + user_id 索引）

        M-5: 首次执行后设置 _table_ensured 标志，后续调用直接跳过。
        """
        if self._table_ensured:
            return

        graph = self._pool.acquire()
        try:
            graph.conn.execute("""
                CREATE TABLE IF NOT EXISTS user_mapping (
                    source      TEXT    NOT NULL,
                    source_id   TEXT    NOT NULL,
                    user_id     TEXT    NOT NULL,
                    created_at  TEXT    NOT NULL,
                    PRIMARY KEY (source, source_id)
                )
            """)
            graph.conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_user_mapping_user_id
                ON user_mapping (user_id)
            """)
            graph.conn.commit()
            self._table_ensured = True
        finally:
            self._pool.release(graph)

    def resolve_user(self, source: str, source_id: str) -> Optional[str]:
        """将来源标识映射为 user_label

        查找优先级:
          1. 先查内存缓存
          2. 再查数据库 user_mapping 表
          3. 不存在则自动创建

        Args:
            source: 来源平台（wechat/web/api）
            source_id: 来源平台用户标识

        Returns:
            user_label（如 "user_lzk"），失败返回 None
        """
        cache_key = (source, source_id)

        # 1. 查内存缓存
        with self._cache_lock:
            if cache_key in self._mapping_cache:
                return self._mapping_cache[cache_key]

        # 2. 查数据库
        graph = self._pool.acquire()
        try:
            self._ensure_user_mapping_table()
            row = graph.conn.execute(
                "SELECT user_id FROM user_mapping WHERE source=? AND source_id=?",
                (source, source_id)
            ).fetchone()
            if row:
                user_id = row['user_id']
                with self._cache_lock:
                    self._mapping_cache[cache_key] = user_id
                return user_id
        except Exception as e:
            log.warning("查询用户映射失败: %s", e)
            return None
        finally:
            self._pool.release(graph)

        # 3. 不存在则自动创建
        return self._create_user(source, source_id)

    def _create_user(self, source: str, source_id: str) -> Optional[str]:
        """创建新用户：插入 user_mapping + seeds 表

        流程:
          1. 生成 user_id（格式: user_{hash(source:source_id)[:8]}）
          2. 插入 user_mapping 表
          3. 插入 seeds 表（type=USER, domain='用户'）
          4. 更新内存缓存

        失败时回滚并返回 None。

        Args:
            source: 来源平台
            source_id: 来源平台用户标识

        Returns:
            新创建的 user_id，失败返回 None
        """
        # 生成确定性 user_id
        raw = f"{source}:{source_id}".encode()
        hash_suffix = hashlib.sha256(raw).hexdigest()[:USER_ID_HASH_LENGTH]
        user_id = f"user_{hash_suffix}"

        cache_key = (source, source_id)
        created_at = datetime.now(timezone.utc).isoformat()

        graph = self._pool.acquire()
        try:
            self._ensure_user_mapping_table()

            # 插入 user_mapping 表
            graph.conn.execute(
                "INSERT OR IGNORE INTO user_mapping (source, source_id, user_id, created_at) "
                "VALUES (?, ?, ?, ?)",
                (source, source_id, user_id, created_at)
            )

            # 插入 seeds 表（type=USER, domain='用户'）
            meta_json = json.dumps(
                {"style": "concise", "level": "expert"},
                ensure_ascii=False
            )
            graph.conn.execute(
                "INSERT OR IGNORE INTO seeds "
                "(id, label, type, domain, aliases, meta, activation, definition, pinyin, activation_bias) "
                "VALUES (?, ?, 'USER', '用户', '[]', ?, 0.0, ?, '', 0.0)",
                (user_id, user_id, meta_json, f"来源:{source}")
            )

            graph.conn.commit()

            # 更新内存缓存
            with self._cache_lock:
                self._mapping_cache[cache_key] = user_id

            log.info("新用户已创建: user_id=%s, source=%s", user_id, source)
            return user_id

        except Exception as e:
            log.warning("创建用户失败: %s", e)
            try:
                graph.conn.rollback()
            except Exception:
                pass
            return None
        finally:
            self._pool.release(graph)

    def add_user_karma_edge(self, user_label: str, target: str,
                            relation: str, weight: float) -> bool:
        """为用户种子添加业力边

        Args:
            user_label: 用户种子 label
            target: 目标概念种子 label
            relation: 关系类型（关注/偏好）
            weight: 权重 [0.005, 2.0]

        Returns:
            True 成功，False 失败
        """
        # 权重边界裁剪
        weight = max(KARMA_MIN, min(KARMA_MAX, weight))

        graph = self._pool.acquire()
        try:
            graph.conn.execute(
                "INSERT OR IGNORE INTO karma_edges "
                "(source, target, relation, weight, source_tag) VALUES (?, ?, ?, ?, 'user_karma')",
                (user_label, target, relation, weight)
            )
            graph.conn.commit()
            log.debug("用户业力边已添加: %s → %s (%s, %.3f)", user_label, target, relation, weight)
            return True
        except Exception as e:
            log.warning("添加用户业力边失败: %s", e)
            return False
        finally:
            self._pool.release(graph)

    def get_user_preferences(self, user_label: str) -> dict:
        """获取用户偏好属性（从 seeds 表 meta 字段读取）

        Args:
            user_label: 用户种子 label

        Returns:
            偏好属性字典，不存在或解析失败返回空字典
        """
        graph = self._pool.acquire()
        try:
            row = graph.conn.execute(
                "SELECT meta FROM seeds WHERE label=? AND type='USER'",
                (user_label,)
            ).fetchone()
            if row and row['meta']:
                try:
                    return json.loads(row['meta'])
                except (json.JSONDecodeError, TypeError):
                    return {}
            return {}
        except Exception as e:
            log.warning("获取用户偏好失败: %s", e)
            return {}
        finally:
            self._pool.release(graph)

    def update_user_preferences(self, user_label: str, preferences: dict) -> bool:
        """更新用户偏好属性（合并到 meta 字段）

        M-3: 使用乐观锁避免并发丢失更新。
        读取时记录旧 meta，更新时 WHERE 包含旧值，
        若 rowcount == 0 说明被并发修改，重试最多 3 次。

        Args:
            user_label: 用户种子 label
            preferences: 要合并的偏好属性

        Returns:
            True 成功，False 失败
        """
        graph = self._pool.acquire()
        try:
            max_retries = 3
            for attempt in range(max_retries):
                # 读取现有偏好
                row = graph.conn.execute(
                    "SELECT meta FROM seeds WHERE label=? AND type='USER'",
                    (user_label,)
                ).fetchone()

                existing = {}
                old_meta = row['meta'] if row and row['meta'] else None
                if old_meta:
                    try:
                        existing = json.loads(old_meta)
                    except (json.JSONDecodeError, TypeError):
                        existing = {}

                # 合并偏好
                existing.update(preferences)
                new_meta = json.dumps(existing, ensure_ascii=False)

                # 乐观锁：WHERE 包含旧 meta，若被并发修改则 rowcount == 0
                cursor = graph.conn.execute(
                    "UPDATE seeds SET meta=? WHERE label=? AND type='USER' AND meta=?",
                    (new_meta, user_label, old_meta)
                )
                if cursor.rowcount > 0:
                    graph.conn.commit()
                    return True

                # rowcount == 0，说明被并发修改，重试
                log.debug("用户偏好更新冲突 (第 %d 次)，重试中: user=%s", attempt + 1, user_label)

            # 重试耗尽
            log.warning("用户偏好更新重试 %d 次后仍冲突，放弃: user=%s", max_retries, user_label)
            return False
        except Exception as e:
            log.warning("更新用户偏好失败: %s", e)
            return False
        finally:
            self._pool.release(graph)

    def rebuild_cache(self) -> None:
        """从数据库重建内存映射缓存（映射表损坏时调用）

        全量加载 user_mapping 表到 _mapping_cache。
        """
        graph = self._pool.acquire()
        try:
            self._ensure_user_mapping_table()
            rows = graph.conn.execute(
                "SELECT source, source_id, user_id FROM user_mapping"
            ).fetchall()

            new_cache: dict[tuple[str, str], str] = {}
            for r in rows:
                new_cache[(r['source'], r['source_id'])] = r['user_id']

            with self._cache_lock:
                self._mapping_cache = new_cache

            log.info("用户映射缓存已重建: %d 条记录", len(new_cache))
        except Exception as e:
            log.warning("重建用户映射缓存失败: %s", e)
        finally:
            self._pool.release(graph)

    def post_query_increment(self, user_label: str | None) -> None:
        """查询后处理：递增冷启动计数"""
        if not user_label:
            return
        try:
            from consciousness_sea.learning.cold_start import ColdStartManager
            graph = self._pool.acquire()
            try:
                manager = ColdStartManager(graph)
                manager.increment_query_count(user_label)
            finally:
                self._pool.release(graph)
        except Exception as e:
            log.warning("冷启动计数递增失败: %s", e)

    def cleanup_user_data(self, user_label: str) -> bool:
        """删除用户的所有个人业力边，提炼池中该用户的贡献 count 减 1

        REQ-P2-019 异常场景 6: 用户删除后的清理。

        Args:
            user_label: 用户种子 label

        Returns:
            True 成功, False 失败
        """
        graph = self._pool.acquire()
        try:
            # 删除个人业力边
            graph.conn.execute(
                "DELETE FROM karma_edges_personal WHERE user_label=?",
                (user_label,)
            )

            # 提炼池中移除该用户的贡献者标记
            # 注意：count 代表历史总提交次数，不减 1（保留历史记录完整性）
            # contributor_users 代表当前活跃贡献者，移除该用户
            rows = graph.conn.execute(
                "SELECT candidate_id, contributor_users FROM distillation_pool "
                "WHERE status != 'upgraded'"
            ).fetchall()

            for r in rows:
                contributors: list[str] = json.loads(r['contributor_users'])
                if user_label in contributors:
                    contributors.remove(user_label)
                    graph.conn.execute(
                        "UPDATE distillation_pool SET contributor_users=?, updated_at=? "
                        "WHERE candidate_id=?",
                        (json.dumps(contributors, ensure_ascii=False),
                         datetime.now(timezone.utc).isoformat(),
                         r['candidate_id'])
                    )

            graph.conn.commit()
            log.info("用户数据已清理: user_label=%s", user_label)
            return True
        except Exception as e:
            log.warning("清理用户数据失败: %s", e)
            return False
        finally:
            self._pool.release(graph)