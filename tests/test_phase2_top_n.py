"""
Phase 2 Top-N 熏习粒度测试 (T2.2)

覆盖:
- KARMA_FULL_SET=False 时仅选取 Top-N 种子
- 激活种子数少于 KARMA_TOP_N 时取全部种子
- 路径级筛选：仅 source/target 均在 Top-N 集合内的路径被熏习
- KARMA_MAX_PAIRS 上限保护
- KARMA_FULL_SET=True 时行为与 Phase 0/1 一致（向后兼容）
- 传播路径为空时返回 modified=0
"""

import sqlite3
import sys
import os
from unittest.mock import patch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.graph_db import GraphDB
from core.verifier import apply_karma
from core.router import RippleResult, ActivationNode


def _build_test_db(db_path: str) -> None:
    """创建测试用 SQLite 数据库文件（含 Phase 2 表）"""
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
        CREATE TABLE IF NOT EXISTS karma_edges_personal (
            user_label  TEXT    NOT NULL,
            source      TEXT    NOT NULL,
            target      TEXT    NOT NULL,
            relation    TEXT    NOT NULL,
            weight      REAL    NOT NULL,
            source_tag  TEXT    NOT NULL DEFAULT 'personal_karma',
            updated_at  TEXT    NOT NULL,
            PRIMARY KEY (user_label, source, target, relation)
        );
        CREATE TABLE IF NOT EXISTS distillation_pool (
            candidate_id        INTEGER PRIMARY KEY AUTOINCREMENT,
            canonical_source    TEXT    NOT NULL,
            canonical_target    TEXT    NOT NULL,
            canonical_relation  TEXT    NOT NULL,
            representative_label TEXT   NOT NULL,
            count               INTEGER NOT NULL DEFAULT 1,
            contributor_users   TEXT    NOT NULL DEFAULT '[]',
            status              TEXT    NOT NULL DEFAULT 'pending',
            upgraded_at         TEXT,
            created_at          TEXT    NOT NULL,
            updated_at          TEXT    NOT NULL
        );
        CREATE TABLE IF NOT EXISTS param_stats (
            stat_id          INTEGER PRIMARY KEY AUTOINCREMENT,
            query_text       TEXT    NOT NULL,
            decay_factor     REAL    NOT NULL,
            domain_threshold REAL    NOT NULL,
            confidence_high  REAL    NOT NULL,
            ripple_depth     INTEGER NOT NULL,
            activated_count  INTEGER NOT NULL,
            selected_domains TEXT    NOT NULL,
            confidence       REAL    NOT NULL,
            karma_direction  INTEGER NOT NULL,
            created_at       TEXT    NOT NULL
        );
    """)

    # 种子数据：30 个种子，用于测试 Top-N 筛选
    seeds = []
    for i in range(30):
        label = f"种子{i:02d}"
        seeds.append((label, label, 'CONCEPT', '[]', '测试', f'def {i}'))
    conn.executemany(
        "INSERT INTO seeds (id, label, type, aliases, domain, definition) "
        "VALUES (?, ?, ?, ?, ?, ?)",
        seeds,
    )

    # 边数据：每个种子连接到下一个
    edges = []
    for i in range(29):
        edges.append((f"种子{i:02d}", f"种子{i+1:02d}", 'RELATED', 0.5))
    conn.executemany(
        "INSERT INTO karma_edges (source, target, relation, weight) "
        "VALUES (?, ?, ?, ?)",
        edges,
    )
    conn.commit()
    conn.close()


def _make_ripple_result(num_seeds: int, num_paths: int | None = None) -> RippleResult:
    """构建测试用 RippleResult

    Args:
        num_seeds: 激活种子数量
        num_paths: 传播路径数量（默认为种子数-1，显式传 0 表示无路径）
    """
    result = RippleResult()
    result.query = "测试查询"

    for i in range(num_seeds):
        label = f"种子{i:02d}"
        # 激活值递减，便于验证 Top-N 排序
        activation = 1.0 - i * 0.01
        result.activated[label] = ActivationNode(
            label=label,
            activation=activation,
            domain='测试',
            definition=f'def {i}',
            depth=0 if i < 5 else 1,
        )

    if num_paths is None:
        num_paths = max(0, num_seeds - 1)

    for i in range(min(num_paths, num_seeds - 1) if num_seeds > 0 else 0):
        result.paths.append({
            'source': f"种子{i:02d}",
            'target': f"种子{i+1:02d}",
            'relation': 'RELATED',
            'weight': 0.5,
            'depth': 1,
            'ripple_activation': 0.3,
        })

    return result


class TestTopNSelection:
    """KARMA_FULL_SET=False 时仅选取 Top-N 种子"""

    def test_top_n_limits_activated_seeds(self, tmp_path):
        """KARMA_FULL_SET=False 时只熏习 Top-N 种子之间的边"""
        from pathlib import Path
        db_path = str(Path(tmp_path) / "test.db")
        _build_test_db(db_path)

        graph = GraphDB(db_path)
        graph.connect()

        result = _make_ripple_result(num_seeds=30)

        with patch('core.config.KARMA_FULL_SET', False), \
             patch('core.config.KARMA_TOP_N', 10):
            modified = apply_karma(result, graph, karma_direction=+1)

        # 只有 Top-10 种子之间的路径被熏习
        # 路径中 source 和 target 都必须在 Top-10 内
        assert modified <= 9  # 最多 9 条路径（种子0→1, 1→2, ..., 8→9）

        graph.close()

    def test_top_n_20_default(self, tmp_path):
        """默认 KARMA_TOP_N=20 时，30 个种子中只选前 20 个"""
        from pathlib import Path
        db_path = str(Path(tmp_path) / "test.db")
        _build_test_db(db_path)

        graph = GraphDB(db_path)
        graph.connect()

        result = _make_ripple_result(num_seeds=30)

        with patch('core.config.KARMA_FULL_SET', False), \
             patch('core.config.KARMA_TOP_N', 20):
            modified = apply_karma(result, graph, karma_direction=+1)

        # Top-20 种子之间的路径最多 19 条
        assert modified <= 19

        graph.close()


class TestFewerSeedsThanTopN:
    """激活种子数少于 KARMA_TOP_N 时取全部种子"""

    def test_fewer_seeds_than_top_n(self, tmp_path):
        """5 个激活种子，KARMA_TOP_N=20 时取全部 5 个"""
        from pathlib import Path
        db_path = str(Path(tmp_path) / "test.db")
        _build_test_db(db_path)

        graph = GraphDB(db_path)
        graph.connect()

        result = _make_ripple_result(num_seeds=5)

        with patch('core.config.KARMA_FULL_SET', False), \
             patch('core.config.KARMA_TOP_N', 20):
            modified = apply_karma(result, graph, karma_direction=+1)

        # 全部 5 个种子之间的路径（4 条）都被熏习
        assert modified == 4

        graph.close()

    def test_single_seed(self, tmp_path):
        """只有 1 个激活种子时，没有路径需要熏习"""
        from pathlib import Path
        db_path = str(Path(tmp_path) / "test.db")
        _build_test_db(db_path)

        graph = GraphDB(db_path)
        graph.connect()

        result = _make_ripple_result(num_seeds=1)

        with patch('core.config.KARMA_FULL_SET', False), \
             patch('core.config.KARMA_TOP_N', 20):
            modified = apply_karma(result, graph, karma_direction=+1)

        assert modified == 0

        graph.close()


class TestPathLevelFiltering:
    """路径级筛选：仅 source/target 均在 Top-N 集合内的路径被熏习"""

    def test_path_source_not_in_top_n_excluded(self, tmp_path):
        """source 不在 Top-N 集合中的路径被排除"""
        from pathlib import Path
        db_path = str(Path(tmp_path) / "test.db")
        _build_test_db(db_path)

        graph = GraphDB(db_path)
        graph.connect()

        result = _make_ripple_result(num_seeds=30)

        with patch('core.config.KARMA_FULL_SET', False), \
             patch('core.config.KARMA_TOP_N', 5):
            modified = apply_karma(result, graph, karma_direction=+1)

        # 只有 source 和 target 都在 Top-5 内的路径被熏习
        # 种子0→1, 1→2, 2→3, 3→4 共 4 条
        assert modified == 4

        graph.close()

    def test_path_target_not_in_top_n_excluded(self, tmp_path):
        """target 不在 Top-N 集合中的路径被排除"""
        from pathlib import Path
        db_path = str(Path(tmp_path) / "test.db")
        _build_test_db(db_path)

        graph = GraphDB(db_path)
        graph.connect()

        # 构造一条 source 在 Top-N 但 target 不在的路径
        result = _make_ripple_result(num_seeds=10)
        # 替换最后一条路径的 target 为 Top-N 之外的种子
        result.paths[-1] = {
            'source': '种子04',
            'target': '种子09',
            'relation': 'RELATED',
            'weight': 0.5,
            'depth': 1,
            'ripple_activation': 0.3,
        }

        with patch('core.config.KARMA_FULL_SET', False), \
             patch('core.config.KARMA_TOP_N', 5):
            modified = apply_karma(result, graph, karma_direction=+1)

        # 种子0→1, 1→2, 2→3, 3→4 共 4 条（4→9 被排除）
        assert modified == 4

        graph.close()


class TestKarmaMaxPairs:
    """KARMA_MAX_PAIRS 上限保护"""

    def test_max_pairs_limits_modified_count(self, tmp_path):
        """传播路径超过 KARMA_MAX_PAIRS 时截断"""
        from pathlib import Path
        db_path = str(Path(tmp_path) / "test.db")
        _build_test_db(db_path)

        graph = GraphDB(db_path)
        graph.connect()

        result = _make_ripple_result(num_seeds=30)

        with patch('core.config.KARMA_FULL_SET', True), \
             patch('core.config.KARMA_MAX_PAIRS', 5):
            modified = apply_karma(result, graph, karma_direction=+1)

        # 最多修改 5 条边
        assert modified == 5

        graph.close()

    def test_max_pairs_not_applied_when_below(self, tmp_path):
        """传播路径少于 KARMA_MAX_PAIRS 时不截断"""
        from pathlib import Path
        db_path = str(Path(tmp_path) / "test.db")
        _build_test_db(db_path)

        graph = GraphDB(db_path)
        graph.connect()

        result = _make_ripple_result(num_seeds=5)

        with patch('core.config.KARMA_FULL_SET', True), \
             patch('core.config.KARMA_MAX_PAIRS', 500):
            modified = apply_karma(result, graph, karma_direction=+1)

        # 4 条路径，不超过 500
        assert modified == 4

        graph.close()


class TestFullSetBackwardCompat:
    """KARMA_FULL_SET=True 时行为与 Phase 0/1 一致（向后兼容）"""

    def test_full_set_uses_all_seeds(self, tmp_path):
        """KARMA_FULL_SET=True 时所有激活种子的路径都被熏习"""
        from pathlib import Path
        db_path = str(Path(tmp_path) / "test.db")
        _build_test_db(db_path)

        graph = GraphDB(db_path)
        graph.connect()

        result = _make_ripple_result(num_seeds=10)

        with patch('core.config.KARMA_FULL_SET', True):
            modified = apply_karma(result, graph, karma_direction=+1)

        # 全部 9 条路径都被熏习
        assert modified == 9

        graph.close()

    def test_full_set_ignores_top_n(self, tmp_path):
        """KARMA_FULL_SET=True 时 KARMA_TOP_N 不生效"""
        from pathlib import Path
        db_path = str(Path(tmp_path) / "test.db")
        _build_test_db(db_path)

        graph = GraphDB(db_path)
        graph.connect()

        result = _make_ripple_result(num_seeds=10)

        with patch('core.config.KARMA_FULL_SET', True), \
             patch('core.config.KARMA_TOP_N', 3):
            modified = apply_karma(result, graph, karma_direction=+1)

        # 即使 TOP_N=3，FULL_SET=True 时仍熏习全部路径
        assert modified == 9

        graph.close()


class TestEmptyPaths:
    """传播路径为空时返回 modified=0"""

    def test_empty_paths_returns_zero(self, tmp_path):
        """没有传播路径时 modified=0"""
        from pathlib import Path
        db_path = str(Path(tmp_path) / "test.db")
        _build_test_db(db_path)

        graph = GraphDB(db_path)
        graph.connect()

        result = _make_ripple_result(num_seeds=3, num_paths=0)


        with patch('core.config.KARMA_FULL_SET', True):
            modified = apply_karma(result, graph, karma_direction=+1)

        assert modified == 0

        graph.close()

    def test_zero_karma_direction_returns_zero(self, tmp_path):
        """karma_direction=0 时返回 modified=0"""
        from pathlib import Path
        db_path = str(Path(tmp_path) / "test.db")
        _build_test_db(db_path)

        graph = GraphDB(db_path)
        graph.connect()

        result = _make_ripple_result(num_seeds=5)

        modified = apply_karma(result, graph, karma_direction=0)

        assert modified == 0

        graph.close()