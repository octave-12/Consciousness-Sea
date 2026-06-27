"""
GraphDB — SQLite 邻接表封装

识海的知识存储层。所有查询都走这里，不经过任何模型。

节点: seeds 表
边:   karma_edges 表
"""

import sqlite3
import json
import re
import logging
import threading
from typing import Optional
from .config import DEFAULT_DB_PATH, ENABLE_FUZZY, FUZZY_EDIT_DISTANCE, BUSY_TIMEOUT_MS

log = logging.getLogger(__name__)


class GraphDB:
    """知识图谱数据库封装"""

    def __init__(self, db_path: str = DEFAULT_DB_PATH):
        self.db_path = db_path
        self.conn: Optional[sqlite3.Connection] = None
        self._alias_index: Optional[dict[str, str]] = None  # 懒加载缓存
        self._label_index: Optional[set[str]] = None  # 懒加载缓存
        self._edge_count_map: Optional[dict[str, int]] = None  # 懒加载缓存
        self._cache_lock = threading.Lock()  # 保护懒加载的线程锁

    def connect(self, readonly: bool = False):
        """打开数据库连接"""
        uri = f'file:{self.db_path}?mode=ro' if readonly else self.db_path
        self.conn = sqlite3.connect(uri, uri=readonly)
        if not readonly:
            self.conn.execute("PRAGMA journal_mode=WAL")
            self.conn.execute("PRAGMA synchronous=NORMAL")
        self.conn.row_factory = sqlite3.Row

    def close(self):
        if self.conn:
            self.conn.close()
            self.conn = None

    # ═══════════════════════════════════════════════════════════
    # 节点查询
    # ═══════════════════════════════════════════════════════════

    def get_seed(self, label: str) -> Optional[dict]:
        """精确匹配单个种子（label 或 alias）"""
        row = self.conn.execute(
            "SELECT * FROM seeds WHERE label = ?", (label,)
        ).fetchone()
        if row:
            return dict(row)

        # 使用缓存的别名索引
        self._ensure_alias_index()
        if label in self._alias_index:
            return self.get_seed(self._alias_index[label])  # 递归查主 label
        return None

    def match_seeds(self, query: str) -> list[dict]:
        """
        从查询文本中匹配种子。

        使用升级版分词器 tokenizer.tokenize() 进行匹配：
        - 组合词优先匹配（最大正向匹配）
        - 同义词/别名扩展
        - 模糊匹配（编辑距离，可选）
        - 否定词识别（excluded 的种子跳过）
        """
        from .tokenizer import tokenize as tokenizer_tokenize

        self._ensure_label_index()
        self._ensure_alias_index()

        # 构建 edge_count_map 用于模糊匹配歧义消解
        edge_count_map = self._ensure_edge_count_map()

        # 构建长度分桶索引用于模糊匹配性能优化
        label_length_buckets = self._build_label_length_buckets() if ENABLE_FUZZY else None

        tokens = tokenizer_tokenize(
            query,
            self._label_index,
            self._alias_index,
            enable_fuzzy=ENABLE_FUZZY,
            max_edit_distance=FUZZY_EDIT_DISTANCE,
            edge_count_map=edge_count_map,
            label_length_buckets=label_length_buckets,
        )

        matched: dict[str, dict] = {}

        for tm in tokens:
            # 被否定词排除的种子跳过
            if tm.excluded:
                continue

            # 只处理有匹配结果的 token
            if tm.match_type not in ('exact', 'alias', 'fuzzy'):
                continue

            seed_label = tm.seed_label
            if not seed_label or seed_label in matched:
                continue

            row = self.conn.execute(
                "SELECT * FROM seeds WHERE label = ?", (seed_label,)
            ).fetchone()
            if row:
                d = dict(row)
                matched[d['label']] = d

        # 去重（按 label）
        seen = set()
        result = []
        for d in matched.values():
            if d['label'] not in seen:
                seen.add(d['label'])
                result.append(d)
        return result

    def _ensure_alias_index(self):
        """懒加载 alias → seed_label 映射（首次调用时构建，后续从缓存读）

        使用 double-checked locking 模式：先无锁检查，再加锁构建。
        缓存一旦构建完成，后续读取完全无锁（只读数据，线程安全）。
        """
        if self._alias_index is not None:  # 快速路径：无锁检查
            return
        with self._cache_lock:  # 慢路径：加锁构建
            if self._alias_index is not None:  # double-check
                return
            idx = {}
            rows = self.conn.execute(
                "SELECT label, aliases FROM seeds WHERE aliases != '[]' AND aliases != ''"
            ).fetchall()
            for r in rows:
                try:
                    for alias in json.loads(r['aliases']):
                        if alias and alias not in idx:
                            idx[alias] = r['label']
                except (json.JSONDecodeError, TypeError):
                    pass
            self._alias_index = idx

    def _ensure_label_index(self):
        """懒加载 label 索引（首次调用时一次性加载所有 label 到 set）

        使用 double-checked locking 模式：先无锁检查，再加锁构建。
        """
        if self._label_index is not None:  # 快速路径：无锁检查
            return
        with self._cache_lock:  # 慢路径：加锁构建
            if self._label_index is not None:  # double-check
                return
            rows = self.conn.execute("SELECT label FROM seeds").fetchall()
            self._label_index = {r['label'] for r in rows}
            log.debug("label 索引加载完成: %d 条", len(self._label_index))

    def _build_label_length_buckets(self) -> dict[int, set[str]]:
        """构建按长度分桶的 label 索引，用于模糊匹配性能优化。

        Returns:
            {label长度: {label1, label2, ...}} 的映射
        """
        self._ensure_label_index()
        buckets: dict[int, set[str]] = {}
        for label in self._label_index:
            length = len(label)
            if length not in buckets:
                buckets[length] = set()
            buckets[length].add(label)
        return buckets

    def invalidate_cache(self) -> None:
        """重置所有懒加载缓存，供连接归还时调用

        将 _alias_index、_label_index、_edge_count_map 置为 None，
        确保下次使用时重新加载最新数据。
        """
        with self._cache_lock:
            self._alias_index = None
            self._label_index = None
            self._edge_count_map = None

    def _ensure_edge_count_map(self) -> dict[str, int]:
        """懒加载 seed_label → 出边数 的映射，用于模糊匹配歧义消解。

        使用 double-checked locking 模式：先无锁检查，再加锁构建。
        当 enable_fuzzy=False 时跳过构建，返回空字典。
        """
        if self._edge_count_map is not None:  # 快速路径：无锁检查
            return self._edge_count_map
        with self._cache_lock:  # 慢路径：加锁构建
            if self._edge_count_map is not None:  # double-check
                return self._edge_count_map

            if not ENABLE_FUZZY:
                self._edge_count_map = {}
                return self._edge_count_map

            edge_count_map: dict[str, int] = {}
            rows = self.conn.execute(
                "SELECT source, COUNT(*) as cnt FROM karma_edges GROUP BY source"
            ).fetchall()
            for r in rows:
                edge_count_map[r['source']] = r['cnt']
            self._edge_count_map = edge_count_map
            return self._edge_count_map

    def _tokenize(self, query: str) -> list[str]:
        """
        [DEPRECATED] 简单分词：提取中文连续段、英文单词、数字。

        此方法已被 tokenizer.tokenize() 替代，保留仅为向后兼容。
        新代码请使用 tokenizer.tokenize()。
        """
        tokens = []
        # 中文连续段（粒度：单字到 4 字，以及更长）
        chinese_spans = re.findall(r'[\u4e00-\u9fff]+', query)
        for span in chinese_spans:
            # 先尝试整体
            tokens.append(span)
            # 再尝试 2-gram（用于组合词拆分时的后备）
            if len(span) >= 4:
                for i in range(0, len(span) - 1, 2):
                    tokens.append(span[i:i+2])
        # 英文/数字词
        for w in re.findall(r'[a-zA-Z0-9]+', query):
            tokens.append(w)
        return tokens

    # ═══════════════════════════════════════════════════════════
    # 边查询
    # ═══════════════════════════════════════════════════════════

    def outgoing_edges(self, source_label: str) -> list[dict]:
        """获取某个节点的所有出边"""
        rows = self.conn.execute(
            "SELECT * FROM karma_edges WHERE source = ?",
            (source_label,)
        ).fetchall()
        return [dict(r) for r in rows]

    def batch_get_seeds(self, labels: list[str]) -> dict[str, dict]:
        """批量获取种子信息（domain, definition）"""
        if not labels:
            return {}
        placeholders = ','.join('?' * len(labels))
        rows = self.conn.execute(
            f"SELECT label, domain, definition FROM seeds WHERE label IN ({placeholders})",
            labels
        ).fetchall()
        return {r['label']: {'domain': r['domain'], 'definition': r['definition']} for r in rows}

    def get_edge(self, source: str, target: str, relation: str) -> Optional[dict]:
        row = self.conn.execute(
            "SELECT * FROM karma_edges WHERE source=? AND target=? AND relation=?",
            (source, target, relation)
        ).fetchone()
        return dict(row) if row else None

    def adjust_karma(self, source: str, target: str,
                     relation: str = 'RELATED', delta: float = 0.01):
        """
        调整边权重（熏习）。

        新边自动创建，已有边累加 delta。
        边界会裁剪到 [KARMA_MIN, KARMA_MAX]。
        """
        from .config import KARMA_MIN, KARMA_MAX

        existing = self.conn.execute(
            "SELECT weight FROM karma_edges WHERE source=? AND target=? AND relation=?",
            (source, target, relation)
        ).fetchone()

        if existing:
            new_w = max(KARMA_MIN, min(KARMA_MAX, existing[0] + delta))
            self.conn.execute(
                "UPDATE karma_edges SET weight=? WHERE source=? AND target=? AND relation=?",
                (new_w, source, target, relation)
            )
        else:
            new_w = max(KARMA_MIN, min(KARMA_MAX, 0.5 + delta))
            self.conn.execute(
                "INSERT OR IGNORE INTO karma_edges "
                "(source,target,relation,weight,source_tag) VALUES (?,?,?,?,?)",
                (source, target, relation, new_w, 'karma_delta')
            )

    # ═══════════════════════════════════════════════════════════
    # 统计
    # ═══════════════════════════════════════════════════════════

    def stats(self) -> dict:
        nodes = self.conn.execute("SELECT COUNT(*) FROM seeds").fetchone()[0]
        edges = self.conn.execute("SELECT COUNT(*) FROM karma_edges").fetchone()[0]
        return {'nodes': nodes, 'edges': edges}

    def __enter__(self):
        self.connect()
        return self

    def __exit__(self, *args):
        self.close()
