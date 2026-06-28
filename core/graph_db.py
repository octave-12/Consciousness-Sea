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
from .config import DEFAULT_DB_PATH, ENABLE_FUZZY, FUZZY_EDIT_DISTANCE, BUSY_TIMEOUT_MS, META_SEED_ENABLED, COGNITIVE_GOAL_ENABLED, PERCEPTION_ENABLED

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
            self.conn.execute(f"PRAGMA busy_timeout={BUSY_TIMEOUT_MS}")
            # Phase 2: 自动创建新增表
            self.ensure_phase2_tables()
            # Phase 3: 自动创建新增表
            self.ensure_phase3_tables()
            # Phase 4: 自动创建新增表
            if META_SEED_ENABLED:
                self.ensure_phase4_tables()
            # Phase 5: 自动创建新增表
            if COGNITIVE_GOAL_ENABLED:
                self.ensure_phase5_tables()
            # Phase 6: 自动创建新增表
            if PERCEPTION_ENABLED:
                self.ensure_phase6_tables()
        self.conn.row_factory = sqlite3.Row

    def close(self):
        if self.conn:
            self.conn.close()
            self.conn = None

    def ensure_phase2_tables(self) -> None:
        """自动创建 Phase 2 新增表（karma_edges_personal, distillation_pool, param_stats）

        使用 CREATE TABLE IF NOT EXISTS 确保幂等，不破坏现有数据。
        使用事务确保原子性，避免频繁 commit 导致锁争用。
        """
        # 检查表是否已存在（避免不必要的 commit）
        existing = self.conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='karma_edges_personal'"
        ).fetchone()
        existing_expert = self.conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='expert_reliability'"
        ).fetchone()
        if existing and existing_expert:
            return  # 所有表已存在，无需重复创建

        # 个人业力层表
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS karma_edges_personal (
                user_label  TEXT    NOT NULL,
                source      TEXT    NOT NULL,
                target      TEXT    NOT NULL,
                relation    TEXT    NOT NULL,
                weight      REAL    NOT NULL,
                source_tag  TEXT    NOT NULL DEFAULT 'personal_karma',
                updated_at  TEXT    NOT NULL,
                PRIMARY KEY (user_label, source, target, relation)
            )
        """)
        self.conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_karma_personal_user
                ON karma_edges_personal (user_label)
        """)
        self.conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_karma_personal_source
                ON karma_edges_personal (source)
        """)

        # 提炼池表
        self.conn.execute("""
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
            )
        """)
        self.conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_distill_status
                ON distillation_pool (status)
        """)
        self.conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_distill_canonical
                ON distillation_pool (canonical_source, canonical_target, canonical_relation)
        """)

        # 参数统计表
        self.conn.execute("""
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
            )
        """)
        self.conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_param_stats_created
                ON param_stats (created_at)
        """)

        # 专家可靠性表（Phase 1 专家组新增）
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS expert_reliability (
                domain     TEXT PRIMARY KEY,
                score      REAL NOT NULL CHECK(score >= 0.0 AND score <= 1.0),
                updated_at TEXT NOT NULL
            )
        """)
        self.conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_expert_reliability_domain
                ON expert_reliability (domain)
        """)

        self.conn.commit()

    def ensure_phase3_tables(self) -> None:
        """自动创建 Phase 3 新增表（alias_backref_events, candidate_seeds,
        user_cold_start, checkpoint_meta）

        使用 CREATE TABLE IF NOT EXISTS 确保幂等，不破坏现有数据。
        使用事务确保原子性，避免频繁 commit 导致锁争用。
        """
        # 检查表是否已存在（避免不必要的 commit）
        existing = self.conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='alias_backref_events'"
        ).fetchone()
        existing_cold = self.conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='user_cold_start'"
        ).fetchone()
        if existing and existing_cold:
            return  # 所有表已存在，无需重复创建

        # 别名回指事件表
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS alias_backref_events (
                source_keyword  TEXT    NOT NULL,
                target_seed     TEXT    NOT NULL,
                ref_count       INTEGER NOT NULL DEFAULT 0,
                total_count     INTEGER NOT NULL DEFAULT 0,
                back_ref_rate   REAL    NOT NULL DEFAULT 0.0,
                status          TEXT    NOT NULL DEFAULT 'tracking',
                created_at      TEXT    NOT NULL,
                updated_at      TEXT    NOT NULL,
                PRIMARY KEY (source_keyword, target_seed)
            )
        """)
        self.conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_alias_backref_keyword
                ON alias_backref_events (source_keyword)
        """)
        self.conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_alias_backref_status
                ON alias_backref_events (status)
        """)

        # 候选种子表
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS candidate_seeds (
                label           TEXT    PRIMARY KEY,
                status          TEXT    NOT NULL DEFAULT 'candidate',
                count           INTEGER NOT NULL DEFAULT 1,
                domain          TEXT,
                co_occur_seeds  TEXT    NOT NULL DEFAULT '[]',
                candidate_since TEXT    NOT NULL,
                last_seen_at    TEXT    NOT NULL,
                promoted_at     TEXT,
                promoted_seed_id TEXT
            )
        """)
        self.conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_candidate_seeds_status
                ON candidate_seeds (status)
        """)
        self.conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_candidate_seeds_last_seen
                ON candidate_seeds (last_seen_at)
        """)

        # 用户冷启动表
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS user_cold_start (
                user_label  TEXT    PRIMARY KEY,
                query_count INTEGER NOT NULL DEFAULT 0,
                updated_at  TEXT    NOT NULL
            )
        """)
        self.conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_user_cold_start_label
                ON user_cold_start (user_label)
        """)

        # 检查点元数据表
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS checkpoint_meta (
                checkpoint_id    TEXT    PRIMARY KEY,
                tag              TEXT    NOT NULL DEFAULT '',
                edge_count       INTEGER NOT NULL DEFAULT 0,
                file_path        TEXT    NOT NULL,
                file_size_bytes  INTEGER NOT NULL DEFAULT 0,
                created_at       TEXT    NOT NULL,
                source           TEXT    NOT NULL DEFAULT 'manual'
            )
        """)
        self.conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_checkpoint_meta_created
                ON checkpoint_meta (created_at)
        """)

        self.conn.commit()

    def ensure_phase4_tables(self) -> None:
        """自动创建 Phase 4 新增表（meta_seeds, unmatched_queries）

        使用 CREATE TABLE IF NOT EXISTS 确保幂等，不破坏现有数据。
        """
        # 检查表是否已存在（避免不必要的 commit）
        existing = self.conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='meta_seeds'"
        ).fetchone()
        if existing:
            return  # 所有表已存在，无需重复创建

        # 元种子表
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS meta_seeds (
                label           TEXT    PRIMARY KEY NOT NULL,
                category        TEXT    NOT NULL,
                metrics_json    TEXT    NOT NULL DEFAULT '{}',
                status          TEXT    NOT NULL DEFAULT 'active',
                source_domain   TEXT,
                dormant_since   TEXT,
                unchanged_cycles INTEGER NOT NULL DEFAULT 0,
                previous_metrics_json TEXT NOT NULL DEFAULT '{}',
                created_at      TEXT    NOT NULL,
                updated_at      TEXT    NOT NULL
            )
        """)
        self.conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_meta_seeds_category
                ON meta_seeds (category)
        """)
        self.conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_meta_seeds_status
                ON meta_seeds (status)
        """)

        # 未匹配查询词表
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS unmatched_queries (
                query_text  TEXT    PRIMARY KEY NOT NULL,
                count       INTEGER NOT NULL DEFAULT 1,
                first_seen  TEXT    NOT NULL,
                last_seen   TEXT    NOT NULL
            )
        """)
        self.conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_unmatched_queries_count
                ON unmatched_queries (count DESC)
        """)

        self.conn.commit()

    def ensure_phase5_tables(self) -> None:
        """自动创建 Phase 5 新增表（cognitive_goals, goal_history）

        使用 CREATE TABLE IF NOT EXISTS 确保幂等，不破坏现有数据。
        """
        # 检查表是否已存在（避免不必要的 commit）
        existing = self.conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='cognitive_goals'"
        ).fetchone()
        if existing:
            return  # 所有表已存在，无需重复创建

        # 认知目标表
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS cognitive_goals (
                goal_id             TEXT    PRIMARY KEY NOT NULL,
                goal_type           TEXT    NOT NULL,
                trigger_condition   TEXT    NOT NULL,
                domain              TEXT    NOT NULL,
                priority_weight     REAL    NOT NULL DEFAULT 0.0,
                status              TEXT    NOT NULL DEFAULT 'pending',
                sub_goals           TEXT    NOT NULL DEFAULT '[]',
                execution_log       TEXT    NOT NULL DEFAULT '[]',
                associated_user     TEXT,
                decay_cycles_count  INTEGER NOT NULL DEFAULT 0,
                last_touched_at     TEXT    NOT NULL,
                created_at          TEXT    NOT NULL,
                updated_at          TEXT    NOT NULL
            )
        """)
        self.conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_cognitive_goals_status
                ON cognitive_goals (status)
        """)
        self.conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_cognitive_goals_domain_type
                ON cognitive_goals (domain, goal_type)
        """)
        self.conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_cognitive_goals_priority
                ON cognitive_goals (priority_weight DESC)
        """)
        self.conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_cognitive_goals_domain_type_status
                ON cognitive_goals (domain, goal_type, status)
        """)

        # 目标历史快照表
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS goal_history (
                history_id      INTEGER PRIMARY KEY AUTOINCREMENT,
                goal_id         TEXT    NOT NULL,
                old_status      TEXT    NOT NULL,
                new_status      TEXT    NOT NULL,
                old_weight      REAL    NOT NULL,
                new_weight      REAL    NOT NULL,
                reason          TEXT    NOT NULL,
                created_at      TEXT    NOT NULL
            )
        """)
        self.conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_goal_history_goal_id
                ON goal_history (goal_id)
        """)
        self.conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_goal_history_created_at
                ON goal_history (created_at)
        """)

        self.conn.commit()

    def ensure_phase6_tables(self) -> None:
        """自动创建 Phase 6 新增表（perceptual_seeds, perception_events）

        使用 CREATE TABLE IF NOT EXISTS 确保幂等，不破坏现有数据。
        """
        # 检查表是否已存在（避免不必要的 commit）
        existing = self.conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='perceptual_seeds'"
        ).fetchone()
        if existing:
            return  # 所有表已存在，无需重复创建

        # 感知元种子表
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS perceptual_seeds (
                label               TEXT    PRIMARY KEY NOT NULL,
                channel             TEXT    NOT NULL,
                feature_description TEXT    NOT NULL DEFAULT '',
                activation_threshold REAL   NOT NULL DEFAULT 0.3,
                status              TEXT    NOT NULL DEFAULT 'active',
                last_activation     TEXT,
                activation_count    INTEGER NOT NULL DEFAULT 0,
                created_at          TEXT    NOT NULL,
                updated_at          TEXT    NOT NULL
            )
        """)
        self.conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_perceptual_seeds_channel
                ON perceptual_seeds (channel)
        """)
        self.conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_perceptual_seeds_status
                ON perceptual_seeds (status)
        """)

        # 感知激活事件表
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS perception_events (
                event_id         INTEGER PRIMARY KEY AUTOINCREMENT,
                perceptual_seed  TEXT    NOT NULL,
                activation       REAL    NOT NULL,
                channel          TEXT    NOT NULL,
                timestamp        TEXT    NOT NULL,
                processed        INTEGER NOT NULL DEFAULT 0
            )
        """)
        self.conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_perception_events_timestamp
                ON perception_events (timestamp)
        """)
        self.conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_perception_events_processed
                ON perception_events (processed)
        """)

        self.conn.commit()

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
                "SELECT * FROM seeds WHERE label = ? AND type NOT IN ('META', 'PERCEPTUAL')", (seed_label,)
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

    def outgoing_edges(self, source_label: str, *, exclude_meta: bool = True) -> list[dict]:
        """获取某个节点的所有出边

        Args:
            source_label: 源节点 label
            exclude_meta: 是否排除 source 以 "meta:" 开头的边（默认 True）
        """
        if exclude_meta and source_label.startswith("meta:"):
            return []  # 元种子的出边不参与涟漪传播

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

    def adjust_karma_atomic(self, source: str, target: str,
                            relation: str = 'RELATED', delta: float = 0.01) -> bool:
        """原子化业力调整 — 使用 UPSERT 避免并发竞态 + 低权边自动删除

        使用 INSERT ... ON CONFLICT DO UPDATE (UPSERT) 替代原有的
        UPDATE → INSERT OR IGNORE 模式，消除并发时 delta 丢失的风险。

        Phase 2 变更:
          - UPSERT 后检查新权重，若 weight < KARMA_MIN 则自动删除该边
          - 返回值从隐式 None 变为 bool（True=边保留, False=边被删除）

        不在此方法内 commit，由调用方统一 commit。

        Args:
            source: 源节点 label
            target: 目标节点 label
            relation: 关系类型
            delta: 权重修改量（正数增强，负数减弱）

        Returns:
            True 边保留, False 边被删除（权重低于 KARMA_MIN）
        """
        from .config import KARMA_MIN, KARMA_MAX

        # C-3: 使用 UPSERT 消除并发竞态
        # INSERT 新边（weight = 0.5 + delta），若冲突则 UPDATE（weight = weight + delta）
        # karma_edges 表有 PRIMARY KEY (source, target, relation)，可直接 ON CONFLICT
        self.conn.execute(
            "INSERT INTO karma_edges (source, target, relation, weight, source_tag) "
            "VALUES (?, ?, ?, ?, 'karma_delta') "
            "ON CONFLICT (source, target, relation) DO UPDATE "
            "SET weight = MAX(?, MIN(?, weight + ?))",
            (source, target, relation, max(KARMA_MIN, min(KARMA_MAX, 0.5 + delta)),
             KARMA_MIN, KARMA_MAX, delta)
        )

        # Phase 2: 检查是否低于下界 → 自动删除
        row = self.conn.execute(
            "SELECT weight FROM karma_edges WHERE source=? AND target=? AND relation=?",
            (source, target, relation)
        ).fetchone()

        if row and row['weight'] < KARMA_MIN:
            self.conn.execute(
                "DELETE FROM karma_edges WHERE source=? AND target=? AND relation=?",
                (source, target, relation)
            )
            log.info(
                "karma edge deleted: %s → %s (%s), final_weight=%.4f",
                source, target, relation, row['weight']
            )
            return False

        return True

    # ═══════════════════════════════════════════════════════════
    # 个人业力层操作（Phase 2 新增）
    # ═══════════════════════════════════════════════════════════

    def adjust_karma_personal(self, user_label: str, source: str, target: str,
                              relation: str = 'RELATED', delta: float = 0.01) -> bool:
        """个人业力层原子化调整 — UPSERT + 低权边自动删除

        写入 karma_edges_personal 表，不影响全局 karma_edges 表。

        Args:
            user_label: 用户标识
            source: 源节点 label
            target: 目标节点 label
            relation: 关系类型
            delta: 权重修改量

        Returns:
            True 边保留, False 边被删除
        """
        from .config import KARMA_MIN, KARMA_MAX
        from datetime import datetime, timezone

        now = datetime.now(timezone.utc).isoformat()

        self.conn.execute(
            "INSERT INTO karma_edges_personal "
            "(user_label, source, target, relation, weight, source_tag, updated_at) "
            "VALUES (?, ?, ?, ?, ?, 'personal_karma', ?) "
            "ON CONFLICT (user_label, source, target, relation) DO UPDATE "
            "SET weight = MAX(?, MIN(?, weight + ?)), updated_at = ?",
            (user_label, source, target, relation,
             max(KARMA_MIN, min(KARMA_MAX, 0.5 + delta)), now,
             KARMA_MIN, KARMA_MAX, delta, now)
        )

        # 低权边检查
        row = self.conn.execute(
            "SELECT weight FROM karma_edges_personal "
            "WHERE user_label=? AND source=? AND target=? AND relation=?",
            (user_label, source, target, relation)
        ).fetchone()

        if row and row['weight'] < KARMA_MIN:
            self.conn.execute(
                "DELETE FROM karma_edges_personal "
                "WHERE user_label=? AND source=? AND target=? AND relation=?",
                (user_label, source, target, relation)
            )
            log.info(
                "personal karma edge deleted: %s → %s (%s), user=%s, final_weight=%.4f",
                source, target, relation, user_label, row['weight']
            )
            return False

        return True

    def get_personal_weight(self, user_label: str, source: str, target: str,
                            relation: str) -> float | None:
        """读取个人业力权重

        Args:
            user_label: 用户标识
            source: 源节点 label
            target: 目标节点 label
            relation: 关系类型

        Returns:
            权重值，不存在返回 None
        """
        row = self.conn.execute(
            "SELECT weight FROM karma_edges_personal "
            "WHERE user_label=? AND source=? AND target=? AND relation=?",
            (user_label, source, target, relation)
        ).fetchone()
        return row['weight'] if row else None

    def batch_get_personal_weights(self, user_label: str,
                                   source_labels: list[str]) -> dict[tuple[str, str, str], float]:
        """批量读取个人业力权重

        按 user_label 和 source_labels 批量查询，返回映射字典。

        Args:
            user_label: 用户标识
            source_labels: 源节点 label 列表

        Returns:
            {(source, target, relation): weight} 映射
        """
        if not user_label or not source_labels:
            return {}
        placeholders = ','.join('?' * len(source_labels))
        rows = self.conn.execute(
            f"SELECT source, target, relation, weight FROM karma_edges_personal "
            f"WHERE user_label=? AND source IN ({placeholders})",
            [user_label] + source_labels
        ).fetchall()
        return {(r['source'], r['target'], r['relation']): r['weight'] for r in rows}

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
