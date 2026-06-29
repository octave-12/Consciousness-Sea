"""
可观测性模块单元测试 (TASK-5.6)

覆盖:
- 最热/最冷种子查询
- 最重业力边查询
- 最近查询记录
- 告警检测（权重超阈值）
- HTML 渲染输出包含关键内容
- get_status() 聚合数据完整性
"""

import json
import sqlite3
import sys
import pathlib

_root = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_root))
sys.path.insert(0, str(_root / "backend" / "src"))

from consciousness_sea.infrastructure.connection_pool import ConnectionPool
from consciousness_sea.infrastructure.observer import Observer, StatusData, SeedRankItem, KarmaRankItem, QueryRecord
from consciousness_sea.infrastructure.config import KARMA_ALERT_THRESHOLD


def _build_test_db(db_path: str) -> None:
    """创建测试用 SQLite 数据库文件（含丰富数据）"""
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
    # 种子数据：多个领域
    seeds = [
        ('感冒', '感冒', 'CONCEPT', '[]', '医学', 'cold'),
        ('发热', '发热', 'CONCEPT', '[]', '医学', 'fever'),
        ('咳嗽', '咳嗽', 'CONCEPT', '[]', '医学', 'cough'),
        ('量子力学', '量子力学', 'CONCEPT', '[]', '物理', 'quantum mechanics'),
        ('薛定谔方程', '薛定谔方程', 'CONCEPT', '[]', '物理', 'Schrodinger equation'),
        ('人工智能', '人工智能', 'CONCEPT', '["AI"]', '计算机', 'AI'),
    ]
    conn.executemany(
        "INSERT INTO seeds (id, label, type, aliases, domain, definition) "
        "VALUES (?, ?, ?, ?, ?, ?)",
        seeds,
    )
    # 边数据：感冒有最多出边（最热），咳嗽只有一条（最冷）
    edges = [
        ('感冒', '发热', 'COOCCURS_WITH', 0.95),
        ('感冒', '咳嗽', 'COOCCURS_WITH', 0.90),
        ('感冒', '量子力学', 'RELATED', 0.05),
        ('量子力学', '薛定谔方程', 'RELATED', 0.88),
        ('人工智能', '量子力学', 'RELATED', 1.90),  # 超阈值
    ]
    conn.executemany(
        "INSERT INTO karma_edges (source, target, relation, weight) "
        "VALUES (?, ?, ?, ?)",
        edges,
    )
    conn.commit()
    conn.close()


def _make_observer(tmp_path):
    """创建测试用 Observer 实例"""
    db_path = str(tmp_path / "test.db")
    _build_test_db(db_path)
    pool = ConnectionPool(db_path, pool_size=3)
    observer = Observer(pool)
    return pool, observer


def _insert_query_history(db_path: str) -> None:
    """向测试数据库插入查询历史记录"""
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    # 确保 query_history 表存在
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS query_history (
            query_id INTEGER PRIMARY KEY AUTOINCREMENT,
            query_text TEXT NOT NULL,
            matched_seeds_count INTEGER NOT NULL DEFAULT 0,
            selected_domains TEXT NOT NULL DEFAULT '[]',
            confidence REAL NOT NULL DEFAULT 0.0,
            karma_direction INTEGER NOT NULL DEFAULT 0,
            created_at TEXT NOT NULL DEFAULT (datetime('now'))
        );
        CREATE INDEX IF NOT EXISTS idx_history_created_at ON query_history (created_at);
    """)
    # 插入查询记录
    records = [
        ('感冒了怎么办', 3, json.dumps(['医学']), 0.85, 1),
        ('量子力学是什么', 2, json.dumps(['物理']), 0.90, 1),
        ('人工智能', 1, json.dumps(['计算机']), 0.75, 0),
    ]
    conn.executemany(
        "INSERT INTO query_history (query_text, matched_seeds_count, selected_domains, confidence, karma_direction) "
        "VALUES (?, ?, ?, ?, ?)",
        records,
    )
    conn.commit()
    conn.close()


class TestHottestSeeds:
    """最热种子查询"""

    def test_hottest_seeds_returns_list(self, tmp_path):
        """get_hottest_seeds() 返回列表"""
        pool, observer = _make_observer(tmp_path)
        try:
            result = observer.get_hottest_seeds()
            assert isinstance(result, list)
        finally:
            pool.close_all()

    def test_hottest_seeds_ordered_by_edge_count_desc(self, tmp_path):
        """最热种子按出边数降序排列"""
        pool, observer = _make_observer(tmp_path)
        try:
            result = observer.get_hottest_seeds()
            assert len(result) > 0
            # 感冒有3条出边，应排第一
            assert result[0].label == '感冒'
            assert result[0].edge_count == 3
            # 验证降序
            for i in range(len(result) - 1):
                assert result[i].edge_count >= result[i + 1].edge_count
        finally:
            pool.close_all()

    def test_hottest_seeds_limit(self, tmp_path):
        """limit 参数限制返回数量"""
        pool, observer = _make_observer(tmp_path)
        try:
            result = observer.get_hottest_seeds(limit=2)
            assert len(result) <= 2
        finally:
            pool.close_all()

    def test_hottest_seeds_item_type(self, tmp_path):
        """返回项为 SeedRankItem 类型"""
        pool, observer = _make_observer(tmp_path)
        try:
            result = observer.get_hottest_seeds()
            for item in result:
                assert isinstance(item, SeedRankItem)
                assert isinstance(item.label, str)
                assert isinstance(item.edge_count, int)
        finally:
            pool.close_all()


class TestColdestSeeds:
    """最冷种子查询"""

    def test_coldest_seeds_ordered_by_edge_count_asc(self, tmp_path):
        """最冷种子按出边数升序排列"""
        pool, observer = _make_observer(tmp_path)
        try:
            result = observer.get_coldest_seeds()
            assert len(result) > 0
            # 验证升序
            for i in range(len(result) - 1):
                assert result[i].edge_count <= result[i + 1].edge_count
        finally:
            pool.close_all()

    def test_coldest_seeds_limit(self, tmp_path):
        """limit 参数限制返回数量"""
        pool, observer = _make_observer(tmp_path)
        try:
            result = observer.get_coldest_seeds(limit=2)
            assert len(result) <= 2
        finally:
            pool.close_all()


class TestHeaviestKarma:
    """最重业力边查询"""

    def test_heaviest_karma_returns_list(self, tmp_path):
        """get_heaviest_karma() 返回列表"""
        pool, observer = _make_observer(tmp_path)
        try:
            result = observer.get_heaviest_karma()
            assert isinstance(result, list)
        finally:
            pool.close_all()

    def test_heaviest_karma_ordered_by_weight_desc(self, tmp_path):
        """最重业力边按权重降序排列"""
        pool, observer = _make_observer(tmp_path)
        try:
            result = observer.get_heaviest_karma()
            assert len(result) > 0
            # 验证降序
            for i in range(len(result) - 1):
                assert result[i].weight >= result[i + 1].weight
        finally:
            pool.close_all()

    def test_heaviest_karma_item_type(self, tmp_path):
        """返回项为 KarmaRankItem 类型"""
        pool, observer = _make_observer(tmp_path)
        try:
            result = observer.get_heaviest_karma()
            for item in result:
                assert isinstance(item, KarmaRankItem)
                assert isinstance(item.source, str)
                assert isinstance(item.target, str)
                assert isinstance(item.weight, float)
        finally:
            pool.close_all()

    def test_heaviest_karma_str_format(self, tmp_path):
        """KarmaRankItem.__str__() 格式化输出"""
        pool, observer = _make_observer(tmp_path)
        try:
            result = observer.get_heaviest_karma()
            if result:
                s = str(result[0])
                assert '↔' in s
        finally:
            pool.close_all()


class TestRecentQueries:
    """最近查询记录"""

    def test_recent_queries_returns_list(self, tmp_path):
        """get_recent_queries() 返回列表"""
        db_path = str(tmp_path / "test.db")
        _insert_query_history(db_path)
        pool, observer = _make_observer(tmp_path)
        try:
            result = observer.get_recent_queries()
            assert isinstance(result, list)
        finally:
            pool.close_all()

    def test_recent_queries_has_records(self, tmp_path):
        """有查询历史时返回记录"""
        db_path = str(tmp_path / "test.db")
        _insert_query_history(db_path)
        pool, observer = _make_observer(tmp_path)
        try:
            result = observer.get_recent_queries()
            assert len(result) > 0
        finally:
            pool.close_all()

    def test_recent_queries_item_type(self, tmp_path):
        """返回项为 QueryRecord 类型"""
        db_path = str(tmp_path / "test.db")
        _insert_query_history(db_path)
        pool, observer = _make_observer(tmp_path)
        try:
            result = observer.get_recent_queries()
            for item in result:
                assert isinstance(item, QueryRecord)
                assert isinstance(item.query_text, str)
                assert isinstance(item.selected_domains, list)
                assert isinstance(item.confidence, float)
        finally:
            pool.close_all()

    def test_recent_queries_no_history(self, tmp_path):
        """无查询历史时返回空列表"""
        pool, observer = _make_observer(tmp_path)
        try:
            result = observer.get_recent_queries()
            assert isinstance(result, list)
            # 可能返回空列表（无 query_history 表或无数据）
        finally:
            pool.close_all()


class TestAlertDetection:
    """告警检测（权重超阈值）"""

    def test_detect_alerts_returns_list(self, tmp_path):
        """detect_alerts() 返回列表"""
        pool, observer = _make_observer(tmp_path)
        try:
            result = observer.detect_alerts()
            assert isinstance(result, list)
        finally:
            pool.close_all()

    def test_detect_alerts_finds_high_weight(self, tmp_path):
        """检测到超阈值权重的业力边"""
        pool, observer = _make_observer(tmp_path)
        try:
            result = observer.detect_alerts()
            # 人工智能→量子力学 权重 1.90 > KARMA_ALERT_THRESHOLD(1.8)
            assert len(result) > 0
            # 告警信息应包含源和目标
            alert_text = ' '.join(result)
            assert '人工智能' in alert_text
            assert '量子力学' in alert_text
        finally:
            pool.close_all()

    def test_detect_alerts_format(self, tmp_path):
        """告警信息格式正确"""
        pool, observer = _make_observer(tmp_path)
        try:
            result = observer.detect_alerts()
            for alert in result:
                assert isinstance(alert, str)
                assert '权重异常高' in alert or '异常' in alert
        finally:
            pool.close_all()

    def test_detect_alerts_no_alerts_when_all_below_threshold(self, tmp_path):
        """所有边权重低于阈值时无告警"""
        db_path = str(tmp_path / "test.db")
        _build_test_db(db_path)
        # 删除超阈值边
        conn = sqlite3.connect(db_path)
        conn.execute("DELETE FROM karma_edges WHERE weight > ?", (KARMA_ALERT_THRESHOLD,))
        conn.commit()
        conn.close()

        pool = ConnectionPool(db_path, pool_size=3)
        observer = Observer(pool)
        try:
            result = observer.detect_alerts()
            assert len(result) == 0
        finally:
            pool.close_all()


class TestHtmlRender:
    """HTML 渲染输出包含关键内容"""

    def test_render_html_returns_string(self, tmp_path):
        """render_html() 返回字符串"""
        pool, observer = _make_observer(tmp_path)
        try:
            status = observer.get_status()
            html = observer.render_html(status)
            assert isinstance(html, str)
        finally:
            pool.close_all()

    def test_render_html_contains_title(self, tmp_path):
        """HTML 包含标题"""
        pool, observer = _make_observer(tmp_path)
        try:
            status = observer.get_status()
            html = observer.render_html(status)
            assert '识海监控面板' in html
        finally:
            pool.close_all()

    def test_render_html_contains_seed_count(self, tmp_path):
        """HTML 包含种子总数"""
        pool, observer = _make_observer(tmp_path)
        try:
            status = observer.get_status()
            html = observer.render_html(status)
            assert '种子总数' in html
        finally:
            pool.close_all()

    def test_render_html_contains_karma_edge_count(self, tmp_path):
        """HTML 包含业力边总数"""
        pool, observer = _make_observer(tmp_path)
        try:
            status = observer.get_status()
            html = observer.render_html(status)
            assert '业力边总数' in html
        finally:
            pool.close_all()

    def test_render_html_contains_alert_section(self, tmp_path):
        """HTML 包含告警区域"""
        pool, observer = _make_observer(tmp_path)
        try:
            status = observer.get_status()
            html = observer.render_html(status)
            assert '告警信息' in html
        finally:
            pool.close_all()

    def test_render_html_contains_hottest_seeds(self, tmp_path):
        """HTML 包含最热种子数据"""
        pool, observer = _make_observer(tmp_path)
        try:
            status = observer.get_status()
            html = observer.render_html(status)
            assert '最热种子' in html
            assert '感冒' in html
        finally:
            pool.close_all()

    def test_render_html_contains_heaviest_karma(self, tmp_path):
        """HTML 包含最重业力边数据"""
        pool, observer = _make_observer(tmp_path)
        try:
            status = observer.get_status()
            html = observer.render_html(status)
            assert '最重业力边' in html
        finally:
            pool.close_all()

    def test_render_html_is_valid_html(self, tmp_path):
        """HTML 是有效的 HTML 文档"""
        pool, observer = _make_observer(tmp_path)
        try:
            status = observer.get_status()
            html = observer.render_html(status)
            assert html.startswith('<!DOCTYPE html>')
            assert '</html>' in html
        finally:
            pool.close_all()

    def test_render_html_auto_refresh(self, tmp_path):
        """HTML 包含自动刷新 meta 标签"""
        pool, observer = _make_observer(tmp_path)
        try:
            status = observer.get_status()
            html = observer.render_html(status)
            assert 'http-equiv="refresh"' in html
        finally:
            pool.close_all()


class TestGetStatus:
    """get_status() 聚合数据完整性"""

    def test_get_status_returns_status_data(self, tmp_path):
        """get_status() 返回 StatusData 实例"""
        pool, observer = _make_observer(tmp_path)
        try:
            status = observer.get_status()
            assert isinstance(status, StatusData)
        finally:
            pool.close_all()

    def test_get_status_has_total_seeds(self, tmp_path):
        """StatusData 包含种子总数"""
        pool, observer = _make_observer(tmp_path)
        try:
            status = observer.get_status()
            assert status.total_seeds > 0
        finally:
            pool.close_all()

    def test_get_status_has_total_karma_edges(self, tmp_path):
        """StatusData 包含业力边总数"""
        pool, observer = _make_observer(tmp_path)
        try:
            status = observer.get_status()
            assert status.total_karma_edges > 0
        finally:
            pool.close_all()

    def test_get_status_has_hottest_seeds(self, tmp_path):
        """StatusData 包含最热种子列表"""
        pool, observer = _make_observer(tmp_path)
        try:
            status = observer.get_status()
            assert isinstance(status.hottest_seeds, list)
        finally:
            pool.close_all()

    def test_get_status_has_coldest_seeds(self, tmp_path):
        """StatusData 包含最冷种子列表"""
        pool, observer = _make_observer(tmp_path)
        try:
            status = observer.get_status()
            assert isinstance(status.coldest_seeds, list)
        finally:
            pool.close_all()

    def test_get_status_has_heaviest_karma(self, tmp_path):
        """StatusData 包含最重业力边列表"""
        pool, observer = _make_observer(tmp_path)
        try:
            status = observer.get_status()
            assert isinstance(status.heaviest_karma, list)
        finally:
            pool.close_all()

    def test_get_status_has_recent_queries(self, tmp_path):
        """StatusData 包含最近查询列表"""
        pool, observer = _make_observer(tmp_path)
        try:
            status = observer.get_status()
            assert isinstance(status.recent_queries, list)
        finally:
            pool.close_all()

    def test_get_status_has_alerts(self, tmp_path):
        """StatusData 包含告警列表"""
        pool, observer = _make_observer(tmp_path)
        try:
            status = observer.get_status()
            assert isinstance(status.alerts, list)
        finally:
            pool.close_all()

    def test_get_status_has_domain_distribution(self, tmp_path):
        """StatusData 包含领域分布"""
        pool, observer = _make_observer(tmp_path)
        try:
            status = observer.get_status()
            assert isinstance(status.domain_distribution, dict)
            assert len(status.domain_distribution) > 0
        finally:
            pool.close_all()

    def test_get_status_domain_distribution_values(self, tmp_path):
        """领域分布包含预期领域"""
        pool, observer = _make_observer(tmp_path)
        try:
            status = observer.get_status()
            domains = status.domain_distribution
            assert '医学' in domains
            assert '物理' in domains
            assert '计算机' in domains
        finally:
            pool.close_all()


if __name__ == '__main__':
    import traceback

    test_classes = [
        TestHottestSeeds,
        TestColdestSeeds,
        TestHeaviestKarma,
        TestRecentQueries,
        TestAlertDetection,
        TestHtmlRender,
        TestGetStatus,
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