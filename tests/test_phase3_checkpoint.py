"""
Phase 3 检查点与回滚测试

测试 CheckpointManager 的所有公共方法：
- create_checkpoint: 创建业力检查点
- rollback: 执行回滚操作
- list_checkpoints: 查询检查点列表
- start_daemon / stop_daemon: 守护线程管理
"""

from __future__ import annotations

import json
import sqlite3
import sys
import os
import time
from pathlib import Path
from unittest.mock import patch

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.checkpoint import (
    CheckpointData,
    CheckpointManager,
    CheckpointMeta,
    CheckpointSource,
    RollbackResult,
)
from core.graph_db import GraphDB
from core.config import CHECKPOINT_RETAIN_COUNT


# ═══════════════════════════════════════════════════════════
#  Fixtures
# ═══════════════════════════════════════════════════════════


def _build_test_db() -> sqlite3.Connection:
    """创建含 Phase 3 表的内存测试数据库"""
    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row
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
        CREATE TABLE karma_edges_personal (
            user_label  TEXT    NOT NULL,
            source      TEXT    NOT NULL,
            target      TEXT    NOT NULL,
            relation    TEXT    NOT NULL,
            weight      REAL    NOT NULL,
            source_tag  TEXT    NOT NULL DEFAULT 'personal_karma',
            updated_at  TEXT    NOT NULL,
            PRIMARY KEY (user_label, source, target, relation)
        );
        CREATE TABLE checkpoint_meta (
            checkpoint_id    TEXT    PRIMARY KEY,
            tag              TEXT    NOT NULL DEFAULT '',
            edge_count       INTEGER NOT NULL DEFAULT 0,
            file_path        TEXT    NOT NULL,
            file_size_bytes  INTEGER NOT NULL DEFAULT 0,
            created_at       TEXT    NOT NULL,
            source           TEXT    NOT NULL DEFAULT 'manual'
        );
    """)

    # 插入测试种子
    seeds = [
        ("感冒", "感冒", "CONCEPT", "[]", "医学", "急性上呼吸道感染"),
        ("发热", "发热", "CONCEPT", "[]", "医学", "体温升高"),
        ("咳嗽", "咳嗽", "CONCEPT", "[]", "医学", "cough"),
    ]
    conn.executemany(
        "INSERT INTO seeds (id, label, type, aliases, domain, definition) VALUES (?, ?, ?, ?, ?, ?)",
        seeds,
    )

    # 插入测试业力边
    edges = [
        ("感冒", "发热", "RELATED", 0.8, "karma_delta"),
        ("感冒", "咳嗽", "RELATED", 0.6, "karma_delta"),
    ]
    conn.executemany(
        "INSERT INTO karma_edges (source, target, relation, weight, source_tag) VALUES (?, ?, ?, ?, ?)",
        edges,
    )
    conn.commit()
    return conn


def _make_graph_db(conn: sqlite3.Connection) -> GraphDB:
    """从已有连接创建 GraphDB 实例"""
    db = GraphDB(":memory:")
    db.conn = conn
    db.ensure_phase2_tables()
    db.ensure_phase3_tables()
    return db


def _create_manager(tmp_path: Path) -> tuple[CheckpointManager, GraphDB]:
    """创建独立的 CheckpointManager + GraphDB 实例"""
    conn = _build_test_db()
    g = _make_graph_db(conn)
    mgr = CheckpointManager(g, checkpoint_dir=str(tmp_path / "checkpoints"))
    return mgr, g


@pytest.fixture
def checkpoint_env(tmp_path):
    """创建独立的测试环境（每个测试使用独立的数据库和目录）"""
    mgr, g = _create_manager(tmp_path)
    yield mgr, g
    g.close()


# ═══════════════════════════════════════════════════════════
#  测试用例
# ═══════════════════════════════════════════════════════════


class TestCheckpointManager:
    """CheckpointManager 单元测试"""

    def test_create_checkpoint(self, checkpoint_env):
        """返回 CheckpointMeta，文件存在"""
        mgr, g = checkpoint_env
        meta = mgr.create_checkpoint(tag="test_checkpoint")

        assert isinstance(meta, CheckpointMeta)
        assert meta.checkpoint_id.startswith("cp_")
        assert meta.edge_count == 2  # 2条业力边
        assert meta.source == CheckpointSource.MANUAL
        assert meta.tag == "test_checkpoint"
        assert meta.file_size_bytes > 0

        # 验证文件存在
        checkpoint_path = Path(meta.file_path)
        assert checkpoint_path.exists()

    def test_checkpoint_content(self, checkpoint_env):
        """JSON 包含 version, checkpoint_id, edges 等字段"""
        mgr, g = checkpoint_env
        meta = mgr.create_checkpoint(tag="content_test")

        # 读取文件内容
        checkpoint_path = Path(meta.file_path)
        with open(checkpoint_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        assert "version" in data
        assert data["version"] == 1
        assert "checkpoint_id" in data
        assert data["checkpoint_id"] == meta.checkpoint_id
        assert "edges" in data
        assert len(data["edges"]) == 2
        assert "created_at" in data
        assert "tag" in data
        assert "source" in data
        assert "edge_count" in data

        # 验证边的字段
        edge = data["edges"][0]
        assert "source" in edge
        assert "target" in edge
        assert "relation" in edge
        assert "weight" in edge

    def test_atomic_write(self, checkpoint_env):
        """临时文件不存在，目标文件存在"""
        mgr, g = checkpoint_env
        meta = mgr.create_checkpoint(tag="atomic_test")

        # 临时文件不应存在
        checkpoint_dir = Path(meta.file_path).parent
        tmp_files = list(checkpoint_dir.glob("checkpoint_tmp_*.json"))
        assert len(tmp_files) == 0

        # 目标文件应存在
        assert Path(meta.file_path).exists()

    def test_retention_policy(self, tmp_path):
        """超出 RETAIN_COUNT 时删除最旧的"""
        metas = []
        for i in range(CHECKPOINT_RETAIN_COUNT + 3):
            # 每次创建独立的数据库避免 checkpoint_id 冲突
            mgr, g = _create_manager(tmp_path / f"cp_{i}")
            # 复用同一个数据库以测试保留策略
            conn = _build_test_db()
            g2 = _make_graph_db(conn)
            mgr2 = CheckpointManager(g2, checkpoint_dir=str(tmp_path / "shared_cp"))
            meta = mgr2.create_checkpoint(tag=f"retention_{i}")
            metas.append(meta)
            time.sleep(0.01)  # 确保 checkpoint_id 不重复
            g.close()
            g2.close()

        # 使用最后一个 manager 验证保留数量
        conn = _build_test_db()
        g_final = _make_graph_db(conn)
        mgr_final = CheckpointManager(g_final, checkpoint_dir=str(tmp_path / "shared_cp"))
        remaining = mgr_final.list_checkpoints(limit=100)
        assert len(remaining) <= CHECKPOINT_RETAIN_COUNT
        g_final.close()

    def test_full_rollback(self, tmp_path):
        """业力边恢复到检查点时刻的值"""
        mgr, g = _create_manager(tmp_path)

        # 创建检查点
        meta = mgr.create_checkpoint(tag="before_change")

        # 修改业力边
        g.conn.execute(
            "UPDATE karma_edges SET weight = 0.99 WHERE source = '感冒' AND target = '发热'"
        )
        g.conn.commit()

        # 添加新边
        g.conn.execute(
            "INSERT INTO karma_edges (source, target, relation, weight, source_tag) "
            "VALUES ('感冒', '咳嗽', 'COOCCURS_WITH', 0.7, 'test')"
        )
        g.conn.commit()

        # 验证修改已生效
        row = g.conn.execute(
            "SELECT weight FROM karma_edges WHERE source = '感冒' AND target = '发热' AND relation = 'RELATED'"
        ).fetchone()
        assert abs(row["weight"] - 0.99) < 0.01

        # 等待1秒确保 pre_rollback 检查点 ID 不冲突
        time.sleep(1.0)

        # 执行全量回滚
        result = mgr.rollback(meta.checkpoint_id, mode="full")

        assert result.success is True
        assert result.mode == "full"
        assert result.edges_affected == 2

        # 验证业力边已恢复
        row = g.conn.execute(
            "SELECT weight FROM karma_edges WHERE source = '感冒' AND target = '发热' AND relation = 'RELATED'"
        ).fetchone()
        assert abs(row["weight"] - 0.8) < 0.01

        # 验证新添加的边已被删除（检查点中不存在）
        row = g.conn.execute(
            "SELECT * FROM karma_edges WHERE source = '感冒' AND target = '咳嗽' AND relation = 'COOCCURS_WITH'"
        ).fetchone()
        assert row is None

    def test_single_rollback(self, tmp_path):
        """仅恢复指定边的权重"""
        mgr, g = _create_manager(tmp_path)

        # 创建检查点
        meta = mgr.create_checkpoint(tag="single_before")

        # 修改业力边
        g.conn.execute(
            "UPDATE karma_edges SET weight = 0.99 WHERE source = '感冒' AND target = '发热'"
        )
        g.conn.commit()

        # 等待1秒确保 pre_rollback 检查点 ID 不冲突
        time.sleep(1.0)

        # 执行单边回滚
        result = mgr.rollback(
            meta.checkpoint_id,
            mode="single",
            edges=[{"source": "感冒", "target": "发热", "relation": "RELATED"}],
        )

        assert result.success is True
        assert result.mode == "single"
        assert result.edges_affected == 1

        # 验证指定边已恢复
        row = g.conn.execute(
            "SELECT weight FROM karma_edges WHERE source = '感冒' AND target = '发热' AND relation = 'RELATED'"
        ).fetchone()
        assert abs(row["weight"] - 0.8) < 0.01

        # 验证其他边未受影响
        row = g.conn.execute(
            "SELECT weight FROM karma_edges WHERE source = '感冒' AND target = '咳嗽' AND relation = 'RELATED'"
        ).fetchone()
        assert row is not None  # 仍然存在

    def test_pre_rollback_checkpoint(self, tmp_path):
        """回滚前创建 pre_rollback 检查点"""
        mgr, g = _create_manager(tmp_path)

        # 创建初始检查点
        meta = mgr.create_checkpoint(tag="initial")

        # 等待1秒确保 pre_rollback 检查点 ID 不冲突
        time.sleep(1.0)

        # 执行回滚
        mgr.rollback(meta.checkpoint_id, mode="full")

        # 验证 pre_rollback 检查点已创建
        checkpoints = mgr.list_checkpoints(limit=100)
        pre_rollback_found = any(
            cp.source == CheckpointSource.PRE_ROLLBACK for cp in checkpoints
        )
        assert pre_rollback_found is True

    def test_rollback_no_personal(self, tmp_path):
        """回滚不影响个人业力"""
        mgr, g = _create_manager(tmp_path)

        # 插入个人业力边
        now = "2025-01-01T00:00:00+00:00"
        g.conn.execute(
            "INSERT INTO karma_edges_personal "
            "(user_label, source, target, relation, weight, source_tag, updated_at) "
            "VALUES (?, ?, ?, 'RELATED', 0.5, 'personal_karma', ?)",
            ("user_001", "感冒", "发热", now),
        )
        g.conn.commit()

        # 创建检查点
        meta = mgr.create_checkpoint(tag="personal_test")

        # 等待1秒确保 pre_rollback 检查点 ID 不冲突
        time.sleep(1.0)

        # 执行全量回滚
        result = mgr.rollback(meta.checkpoint_id, mode="full")
        assert result.success is True

        # 验证个人业力边未受影响
        row = g.conn.execute(
            "SELECT * FROM karma_edges_personal WHERE user_label = ?",
            ("user_001",),
        ).fetchone()
        assert row is not None

    def test_checkpoint_not_found(self, checkpoint_env):
        """返回错误提示"""
        mgr, g = checkpoint_env
        result = mgr.rollback("cp_nonexistent", mode="full")

        assert result.success is False
        assert "not found" in result.error

    def test_checkpoint_corrupted(self, tmp_path):
        """返回错误提示"""
        mgr, g = _create_manager(tmp_path)

        # 创建检查点
        meta = mgr.create_checkpoint(tag="corrupt_test")

        # 损坏检查点文件
        checkpoint_path = Path(meta.file_path)
        with open(checkpoint_path, "w", encoding="utf-8") as f:
            f.write("NOT VALID JSON {{{")

        # 等待1秒确保 pre_rollback 检查点 ID 不冲突
        time.sleep(1.0)

        # 执行回滚
        result = mgr.rollback(meta.checkpoint_id, mode="full")

        assert result.success is False
        assert "corrupted" in result.error

    def test_list_checkpoints(self, tmp_path):
        """按创建时间降序排列"""
        # 使用共享的数据库和目录
        conn = _build_test_db()
        g = _make_graph_db(conn)
        mgr = CheckpointManager(g, checkpoint_dir=str(tmp_path / "list_cp"))

        # 创建多个检查点
        metas = []
        for i in range(3):
            time.sleep(1.0)  # 确保 checkpoint_id 不重复
            meta = mgr.create_checkpoint(tag=f"list_{i}")
            metas.append(meta)

        checkpoints = mgr.list_checkpoints()

        # 验证按时间降序
        assert len(checkpoints) >= 3
        for i in range(len(checkpoints) - 1):
            assert checkpoints[i].created_at >= checkpoints[i + 1].created_at

        g.close()

    def test_daemon_start_stop(self, checkpoint_env):
        """守护线程启动和停止"""
        mgr, g = checkpoint_env
        # 启动守护线程
        mgr.start_daemon()
        assert mgr._daemon_thread is not None
        assert mgr._daemon_thread.is_alive()

        # 停止守护线程
        mgr.stop_daemon()
        # 等待线程结束
        time.sleep(0.5)

        # 验证线程已停止（_daemon_thread 被置为 None）
        assert mgr._daemon_thread is None

    def test_rollback_single_edge_not_in_checkpoint(self, tmp_path):
        """单边回滚时检查点中不存在的边应被删除"""
        mgr, g = _create_manager(tmp_path)

        # 创建检查点（只有2条边）
        meta = mgr.create_checkpoint(tag="before_new_edge")

        # 添加新边
        g.conn.execute(
            "INSERT INTO karma_edges (source, target, relation, weight, source_tag) "
            "VALUES ('发热', '咳嗽', 'RELATED', 0.5, 'test')"
        )
        g.conn.commit()

        # 验证新边存在
        row = g.conn.execute(
            "SELECT * FROM karma_edges WHERE source = '发热' AND target = '咳嗽' AND relation = 'RELATED'"
        ).fetchone()
        assert row is not None

        # 等待1秒确保 pre_rollback 检查点 ID 不冲突
        time.sleep(1.0)

        # 单边回滚该边（检查点中不存在）
        result = mgr.rollback(
            meta.checkpoint_id,
            mode="single",
            edges=[{"source": "发热", "target": "咳嗽", "relation": "RELATED"}],
        )

        assert result.success is True

        # 验证边已被删除
        row = g.conn.execute(
            "SELECT * FROM karma_edges WHERE source = '发热' AND target = '咳嗽' AND relation = 'RELATED'"
        ).fetchone()
        assert row is None

    def test_rollback_single_no_edges_param(self, checkpoint_env):
        """单边回滚未提供 edges 参数时报错"""
        mgr, g = checkpoint_env
        meta = mgr.create_checkpoint(tag="no_edges")

        result = mgr.rollback(meta.checkpoint_id, mode="single")

        assert result.success is False
        assert "edges parameter required" in result.error

    def test_rollback_invalid_mode(self, checkpoint_env):
        """无效回滚模式报错"""
        mgr, g = checkpoint_env
        meta = mgr.create_checkpoint(tag="invalid_mode")

        result = mgr.rollback(meta.checkpoint_id, mode="invalid")

        assert result.success is False
        assert "invalid rollback mode" in result.error
