#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
识海 知识库导入脚本 — 四层混合导入
═════════════════════════════════════════════════════════

从龙珠四代积累的全部知识数据中提取 → 识海 SQLite 知识库。

四层:
  第一层: IS_A (骨架层级) + SYNONYM (别名)         → 300K 节点 + 700K 边
  第二层: RELATED + CAUSE + PART_OF + COOCCURS_WITH → 100K 节点 + 300K 边
  第三层: CC-CEDICT 释义 + 成语库                   →  60K 节点 (补 meta)
  第四层: Wikipedia 条目名 (低频种子)               →  50K 节点

总计: ~500K 节点 + ~1M 边

用法:
  python import_knowledge_base.py              # 全量导入
  python import_knowledge_base.py --dry-run    # 预览统计
  python import_knowledge_base.py --layer 1    # 只跑第一层
  python import_knowledge_base.py --data-dir /path/to/data  # 指定数据目录
"""

import sys
import json
import time
import sqlite3
import argparse
import logging
import pathlib
from collections import defaultdict

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
)
log = logging.getLogger('import_kb')

# ══ 路径配置 ══
# 确保 backend/src 在 sys.path 中
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
# 使用 pathlib.Path 替代 os.path，路径从 resolve_data_dir() + 配置文件名拼接
from core.config import (
    resolve_data_dir,
    CG_DB_FILENAME,
    CEDICT_FILENAME,
    ZHWIKI_FILENAME,
)

# ══ 关系类型配置 ══
# 第一层
LAYER1_RELATIONS = {'IS_A', 'SYNONYM'}
# 第二层
LAYER2_RELATIONS = {'RELATED', 'CAUSE', 'PART_OF', 'COOCCURS_WITH'}

# IS_A 权重
IS_A_WEIGHT = 0.8

# 第二层权重归一化映射
def map_weight(c: float) -> float:
    if c >= 0.9: return 0.7
    if c >= 0.7: return 0.5
    if c >= 0.5: return 0.3
    return 0.0  # 丢弃

# 批量大小
BATCH_SIZE = 50000


def check_data_files(data_dir: pathlib.Path) -> list[pathlib.Path]:
    """
    检查数据文件是否存在，返回缺失文件列表。

    Args:
        data_dir: 数据目录路径。

    Returns:
        缺失文件的路径列表。
    """
    required_files = [
        data_dir / CG_DB_FILENAME,
        data_dir / CEDICT_FILENAME,
    ]
    missing = [f for f in required_files if not f.exists()]
    return missing


def init_db(db_path: pathlib.Path):
    """创建识海 SQLite 知识库"""
    conn = sqlite3.connect(str(db_path))
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA synchronous=NORMAL")
    conn.execute("PRAGMA cache_size=-2000000")  # 2GB cache
    conn.execute("PRAGMA temp_store=MEMORY")

    conn.executescript("""
        CREATE TABLE IF NOT EXISTS seeds (
            id TEXT PRIMARY KEY,
            label TEXT NOT NULL,
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

        CREATE INDEX IF NOT EXISTS idx_seeds_label ON seeds(label);
        CREATE INDEX IF NOT EXISTS idx_seeds_type ON seeds(type);
        CREATE INDEX IF NOT EXISTS idx_seeds_domain ON seeds(domain);

        CREATE TABLE IF NOT EXISTS karma_edges (
            source TEXT NOT NULL,
            target TEXT NOT NULL,
            relation TEXT NOT NULL,
            weight REAL NOT NULL DEFAULT 0.5,
            source_tag TEXT NOT NULL DEFAULT '',
            created_at TEXT NOT NULL DEFAULT (datetime('now')),
            PRIMARY KEY (source, target, relation)
        );

        CREATE INDEX IF NOT EXISTS idx_edges_source ON karma_edges(source);
        CREATE INDEX IF NOT EXISTS idx_edges_target ON karma_edges(target);
        CREATE INDEX IF NOT EXISTS idx_edges_relation ON karma_edges(relation);
        CREATE INDEX IF NOT EXISTS idx_edges_weight ON karma_edges(weight);

        -- 临时批量导入表
        CREATE TEMP TABLE IF NOT EXISTS _batch_seeds (
            id TEXT, label TEXT, type TEXT, aliases TEXT,
            domain TEXT, definition TEXT, pinyin TEXT,
            activation_bias REAL, meta TEXT
        );

        CREATE TEMP TABLE IF NOT EXISTS _batch_edges (
            source TEXT, target TEXT, relation TEXT,
            weight REAL, source_tag TEXT
        );

        CREATE TEMP TABLE IF NOT EXISTS _batch_aliases (
            node_label TEXT, alias TEXT
        );
    """)

    conn.commit()
    return conn


def flush_batch_seeds(conn):
    """冲刷种子批次到主表（去重插入）"""
    conn.execute("""
        INSERT OR IGNORE INTO seeds (id, label, type, aliases, domain, definition, pinyin, activation_bias, meta)
        SELECT id, label, type, aliases, domain, definition, pinyin, activation_bias, meta
        FROM _batch_seeds
    """)
    conn.execute("DELETE FROM _batch_seeds")
    conn.commit()


def flush_batch_edges(conn):
    """冲刷边批次到主表（去重插入）"""
    conn.execute("""
        INSERT OR IGNORE INTO karma_edges (source, target, relation, weight, source_tag)
        SELECT source, target, relation, weight, source_tag
        FROM _batch_edges
    """)
    conn.execute("DELETE FROM _batch_edges")
    conn.commit()


def flush_batch_aliases(conn):
    """冲刷别名到种子 aliases 字段"""
    # 聚合别名
    conn.execute("""
        INSERT OR IGNORE INTO _batch_seeds (id, label, type, aliases)
        SELECT
            n.label AS id,
            n.label AS label,
            'CONCEPT' AS type,
            json_array(a.alias) AS aliases
        FROM _batch_aliases a
        JOIN seeds n ON n.label = a.node_label
    """)
    flush_batch_seeds(conn)

    # 对已有种子的别名更新: 用 JSON 操作追加
    # 先用临时表聚合
    conn.execute("""
        CREATE TEMP TABLE IF NOT EXISTS _alias_merge AS
        SELECT a.node_label, GROUP_CONCAT(DISTINCT json_quote(a.alias)) AS new_aliases_json
        FROM _batch_aliases a
        WHERE EXISTS (SELECT 1 FROM seeds s WHERE s.label = a.node_label)
        GROUP BY a.node_label
    """)

    cur = conn.execute("SELECT node_label, new_aliases_json FROM _alias_merge")
    for node_label, aliases_json in cur:
        # 合并已有 aliases 和新的
        conn.execute("""
            UPDATE seeds SET aliases = (
                SELECT json_group_array(DISTINCT value)
                FROM (
                    SELECT value FROM json_each(aliases)
                    UNION
                    SELECT value FROM json_each(?)
                )
            )
            WHERE label = ?
        """, (f'[{aliases_json}]', node_label))

    conn.execute("DROP TABLE IF EXISTS _alias_merge")
    conn.execute("DELETE FROM _batch_aliases")
    conn.commit()


# ═══════════════════════════════════════════════════════
#  第一层: IS_A 骨架 + SYNONYM 别名
# ═══════════════════════════════════════════════════════

def import_layer1(conn, cg_db_path: pathlib.Path, dry_run=False):
    """
    导入 IS_A 关系（边）和 SYNONYM（别名扩展）。
    源: concept_graph.db IS_A (713万) + SYNONYM (28.8万)
    """
    log.info("=" * 60)
    log.info("第一层: IS_A 骨架 + SYNONYM 别名")
    log.info("=" * 60)

    cg = sqlite3.connect(f'file:{cg_db_path}?mode=ro', uri=True)

    if dry_run:
        log.info("IS_A: 预计约 713 万条中的高质量部分 (c >= 0.5)")
        log.info("SYNONYM: 预计约 28.8 万条")
        cg.close()
        return {'isa_count': '~7.13M', 'synonym_count': '~288K', 'nodes': 0, 'edges': 0}

    # 批量导入 IS_A 边 + 节点
    cursor = cg.execute(
        "SELECT s, o, c FROM triples WHERE r = 'IS_A' AND c >= 0.5"
    )

    seed_buf = []
    edge_buf = []
    total_edges = 0
    total_seeds = 0
    seen_seeds = set()

    for s, o, c in cursor:
        weight = IS_A_WEIGHT  # 层级关系统一 0.8

        edge_buf.append((s, o, 'IS_A', weight, 'loong_cg_import'))

        if s not in seen_seeds:
            seen_seeds.add(s)
            seed_buf.append((s, s, 'CONCEPT', '[]', '', '', '', 0.0, '{}'))
        if o not in seen_seeds:
            seen_seeds.add(o)
            seed_buf.append((o, o, 'CONCEPT', '[]', '', '', '', 0.0, '{}'))

        if len(seed_buf) >= BATCH_SIZE:
            conn.executemany(
                "INSERT INTO _batch_seeds VALUES (?,?,?,?,?,?,?,?,?)",
                seed_buf
            )
            flush_batch_seeds(conn)
            total_seeds += len(seed_buf)
            seed_buf = []

        if len(edge_buf) >= BATCH_SIZE:
            conn.executemany(
                "INSERT INTO _batch_edges VALUES (?,?,?,?,?)",
                edge_buf
            )
            flush_batch_edges(conn)
            total_edges += len(edge_buf)
            edge_buf = []

    # 冲刷剩余
    if seed_buf:
        conn.executemany(
            "INSERT INTO _batch_seeds VALUES (?,?,?,?,?,?,?,?,?)",
            seed_buf
        )
        flush_batch_seeds(conn)
        total_seeds += len(seed_buf)
    if edge_buf:
        conn.executemany(
            "INSERT INTO _batch_edges VALUES (?,?,?,?,?)",
            edge_buf
        )
        flush_batch_edges(conn)
        total_edges += len(edge_buf)

    log.info(f"IS_A: 导入 {total_edges:,} 边, {len(seen_seeds):,} 唯一节点")

    # ── SYNONYM → 别名 ──
    log.info("SYNONYM: 预计约 28.8 万条，导入别名...")

    # SYNONYM: s = 标准名, o = 别名
    alias_cursor = cg.execute(
        "SELECT s, o FROM triples WHERE r = 'SYNONYM' AND c >= 0.5"
    )

    alias_buf = []
    alias_total = 0
    for s, o in alias_cursor:
        alias_buf.append((s, o))
        if len(alias_buf) >= BATCH_SIZE:
            conn.executemany(
                "INSERT INTO _batch_aliases VALUES (?,?)",
                alias_buf
            )
            flush_batch_aliases(conn)
            alias_total += len(alias_buf)
            alias_buf = []

    if alias_buf:
        conn.executemany(
            "INSERT INTO _batch_aliases VALUES (?,?)",
            alias_buf
        )
        flush_batch_aliases(conn)
        alias_total += len(alias_buf)

    log.info(f"SYNONYM: 导入 {alias_total:,} 别名对")

    cg.close()

    # 统计
    final_nodes = conn.execute("SELECT COUNT(*) FROM seeds").fetchone()[0]
    final_edges = conn.execute("SELECT COUNT(*) FROM karma_edges").fetchone()[0]
    log.info(f"第一层完成: {final_nodes:,} 节点, {final_edges:,} 边")

    return {
        'isa_edges': total_edges,
        'isa_nodes': len(seen_seeds),
        'synonym_aliases': alias_total,
        'nodes': final_nodes,
        'edges': final_edges,
    }


# ═══════════════════════════════════════════════════════
#  第二层: RELATED + CAUSE + PART_OF + COOCCURS_WITH
# ═══════════════════════════════════════════════════════

def import_layer2(conn, cg_db_path: pathlib.Path, dry_run=False):
    """
    导入横向关联边。
    权重归一化: c -> 识海权重
    """
    log.info("=" * 60)
    log.info("第二层: 横向关联 (RELATED/CAUSE/PART_OF/COOCCURS_WITH)")
    log.info("=" * 60)

    cg = sqlite3.connect(f'file:{cg_db_path}?mode=ro', uri=True)

    if dry_run:
        log.info("预计: RELATED 85.9万 + CAUSE 11.5万 + PART_OF 0.95万 + COOCCURS_WITH 15.1万")
        cg.close()
        return {'candidates': '~1.13M', 'imported_edges': 0, 'new_nodes': 0}

    rel_placeholders = ','.join(['?'] * len(LAYER2_RELATIONS))
    cursor = cg.execute(
        f"SELECT s, r, o, c FROM triples WHERE r IN ({rel_placeholders}) AND c >= 0.5",
        list(LAYER2_RELATIONS)
    )

    edge_buf = []
    seed_buf = []
    total_edges = 0
    total_new_nodes = 0
    skipped = 0

    # 获取已有节点集合（用于快速判断）
    existing = set()
    for row in conn.execute("SELECT label FROM seeds"):
        existing.add(row[0])

    for s, r, o, c in cursor:
        weight = map_weight(c)
        if weight == 0.0:
            skipped += 1
            continue

        edge_buf.append((s, o, r, weight, 'loong_cg_import'))

        # 新节点
        if s not in existing:
            existing.add(s)
            seed_buf.append((s, s, 'CONCEPT', '[]', '', '', '', 0.0, '{}'))
            total_new_nodes += 1
        if o not in existing:
            existing.add(o)
            seed_buf.append((o, o, 'CONCEPT', '[]', '', '', '', 0.0, '{}'))
            total_new_nodes += 1

        if len(seed_buf) >= BATCH_SIZE:
            conn.executemany(
                "INSERT INTO _batch_seeds VALUES (?,?,?,?,?,?,?,?,?)",
                seed_buf
            )
            flush_batch_seeds(conn)
            seed_buf = []

        if len(edge_buf) >= BATCH_SIZE:
            conn.executemany(
                "INSERT INTO _batch_edges VALUES (?,?,?,?,?)",
                edge_buf
            )
            flush_batch_edges(conn)
            total_edges += len(edge_buf)
            edge_buf = []

    if seed_buf:
        conn.executemany(
            "INSERT INTO _batch_seeds VALUES (?,?,?,?,?,?,?,?,?)",
            seed_buf
        )
        flush_batch_seeds(conn)
    if edge_buf:
        conn.executemany(
            "INSERT INTO _batch_edges VALUES (?,?,?,?,?)",
            edge_buf
        )
        flush_batch_edges(conn)
        total_edges += len(edge_buf)

    cg.close()

    log.info(f"导入 {total_edges:,} 边, {total_new_nodes:,} 新节点, 跳过 {skipped:,} 低权重")
    final_nodes = conn.execute("SELECT COUNT(*) FROM seeds").fetchone()[0]
    final_edges = conn.execute("SELECT COUNT(*) FROM karma_edges").fetchone()[0]
    log.info(f"第二层完成: {final_nodes:,} 节点, {final_edges:,} 边")

    return {
        'imported_edges': total_edges,
        'new_nodes': total_new_nodes,
        'skipped_low_weight': skipped,
    }


# ═══════════════════════════════════════════════════════
#  第三层: CC-CEDICT 释义 + 成语库
# ═══════════════════════════════════════════════════════

def import_layer3(conn, cedict_path: pathlib.Path, idioms_path: pathlib.Path, dry_run=False):
    """
    给已有种子补充 CC-CEDICT 释义 + 成语库创建新种子。
    """
    log.info("=" * 60)
    log.info("第三层: CC-CEDICT 释义 + 成语库")
    log.info("=" * 60)

    # ── CC-CEDICT ──
    log.info(f"加载 CC-CEDICT: {cedict_path}")
    with open(cedict_path, 'r', encoding='utf-8') as f:
        cedict = json.load(f)
    log.info(f"CC-CEDICT: {len(cedict):,} 词条")

    if dry_run:
        log.info(f"  成语: {len(json.load(open(idioms_path, encoding='utf-8'))):,} 条")
        return {'cedict_matched': 0, 'cedict_new': 0, 'idioms_new': 0}

    # 匹配已有种子 → 补释义
    matched = 0
    new_entries = 0
    seed_buf = []

    # 批量查询已有节点
    existing = set()
    for row in conn.execute("SELECT label FROM seeds"):
        existing.add(row[0])

    for word, info in cedict.items():
        if not isinstance(info, dict):
            continue
        definition = info.get('definition', '')
        if not definition:
            defs = info.get('definitions', [])
            if isinstance(defs, list) and defs:
                definition = '; '.join(defs)
        pinyin = info.get('pinyin', '')

        if word in existing:
            # 更新释义
            conn.execute(
                "UPDATE seeds SET definition = ?, pinyin = ? WHERE label = ?",
                (definition, pinyin, word)
            )
            matched += 1
        else:
            # 新种子（但只创建高频或有释义的）
            if definition:
                seed_buf.append((
                    word, word, 'CONCEPT', '[]',
                    '', definition, pinyin, 0.0, '{"source": "cedict"}'
                ))
                existing.add(word)
                new_entries += 1

        if len(seed_buf) >= BATCH_SIZE:
            conn.executemany(
                "INSERT INTO _batch_seeds VALUES (?,?,?,?,?,?,?,?,?)",
                seed_buf
            )
            flush_batch_seeds(conn)
            seed_buf = []

        if matched % 100000 == 0 and matched > 0:
            log.info(f"  释义匹配: {matched:,} / {len(cedict):,}")

    if seed_buf:
        conn.executemany(
            "INSERT INTO _batch_seeds VALUES (?,?,?,?,?,?,?,?,?)",
            seed_buf
        )
        flush_batch_seeds(conn)

    log.info(f"CC-CEDICT: 匹配 {matched:,} 释义, 创建 {new_entries:,} 新种子")

    # ── 成语库 ──
    log.info(f"加载成语库: {idioms_path}")
    with open(idioms_path, 'r', encoding='utf-8') as f:
        idioms = json.load(f)
    log.info(f"成语: {len(idioms):,} 条")

    idioms_new = 0
    seed_buf = []
    for item in idioms:
        if isinstance(item, str):
            word = item
        elif isinstance(item, dict):
            word = item.get('word', item.get('idiom', ''))
        else:
            continue

        if word and word not in existing:
            seed_buf.append((
                word, word, 'CONCEPT', '[]',
                '成语', '', '', 0.0, '{"source": "idioms"}'
            ))
            existing.add(word)
            idioms_new += 1

        if len(seed_buf) >= BATCH_SIZE:
            conn.executemany(
                "INSERT INTO _batch_seeds VALUES (?,?,?,?,?,?,?,?,?)",
                seed_buf
            )
            flush_batch_seeds(conn)
            seed_buf = []

    if seed_buf:
        conn.executemany(
            "INSERT INTO _batch_seeds VALUES (?,?,?,?,?,?,?,?,?)",
            seed_buf
        )
        flush_batch_seeds(conn)

    log.info(f"成语库: 创建 {idioms_new:,} 新种子")

    final_nodes = conn.execute("SELECT COUNT(*) FROM seeds").fetchone()[0]
    log.info(f"第三层完成: {final_nodes:,} 节点")

    return {
        'cedict_matched': matched,
        'cedict_new': new_entries,
        'idioms_new': idioms_new,
    }


# ═══════════════════════════════════════════════════════
#  第四层: Wikipedia 条目名 (低频种子)
# ═══════════════════════════════════════════════════════

def import_layer4(conn, zhwiki_db_path: pathlib.Path, dry_run=False):
    """
    从 Wikipedia 提取条目名 → 创建低频种子。
    只创建前两层没有的条目。
    """
    log.info("=" * 60)
    log.info("第四层: Wikipedia 条目名 (低频覆盖)")
    log.info("=" * 60)

    if not zhwiki_db_path.exists():
        log.warning(f"zhwiki.db 不存在: {zhwiki_db_path}，跳过第四层")
        return {'wiki_entries': 0, 'new_seeds': 0}

    if dry_run or conn is None:
        log.info("Wikipedia: 预计 430,000 中文条目，去重后约 50K 新种子")
        return {'wiki_entries': '~430K', 'new_seeds': '~50K'}

    wiki = sqlite3.connect(f'file:{zhwiki_db_path}?mode=ro', uri=True)

    # 尝试获取条目名
    try:
        # 先看表结构
        tables = wiki.execute(
            "SELECT name FROM sqlite_master WHERE type='table'"
        ).fetchall()
        table_names = [t[0] for t in tables]
        log.info(f"zhwiki 表: {table_names}")

        # 尝试不同可能的表名
        cursor = None
        if 'articles' in table_names:
            cursor = wiki.execute("SELECT title FROM articles")
        elif 'pages' in table_names:
            cursor = wiki.execute("SELECT title FROM pages")
        elif 'zhwiki' in table_names:
            cursor = wiki.execute("SELECT title FROM zhwiki")
        else:
            # 尝试查询第一个有 title 列的表
            for tname in table_names:
                cols = [c[1] for c in wiki.execute(f"PRAGMA table_info({tname})").fetchall()]
                if 'title' in cols:
                    cursor = wiki.execute(f"SELECT title FROM {tname}")
                    log.info(f"使用表 {tname}")
                    break
            else:
                log.warning("找不到 Wikipedia 条目表，跳过")
                wiki.close()
                return {'wiki_entries': 0, 'new_seeds': 0}
    except Exception as e:
        log.warning(f"Wikipedia 查询失败: {e}，跳过")
        wiki.close()
        return {'wiki_entries': 0, 'new_seeds': 0}

    # 已有节点
    existing = set()
    for row in conn.execute("SELECT label FROM seeds"):
        existing.add(row[0])

    new_seeds = 0
    seed_buf = []
    total = 0

    for (title,) in cursor:
        total += 1
        if title and title not in existing:
            # 过滤：跳过太短（单字）或包含过多标点的
            if len(title) < 2:
                continue
            if any(c in title for c in '{}[]<>|\\/"\'+=&%$#@!~`^*'):
                continue

            seed_buf.append((
                title, title, 'CONCEPT', '[]',
                '', '', '', 0.05,  # 低激活偏置
                '{"source": "wikipedia"}'
            ))
            existing.add(title)
            new_seeds += 1

        if len(seed_buf) >= BATCH_SIZE:
            conn.executemany(
                "INSERT INTO _batch_seeds VALUES (?,?,?,?,?,?,?,?,?)",
                seed_buf
            )
            flush_batch_seeds(conn)
            seed_buf = []

        if total % 50000 == 0:
            log.info(f"  扫描: {total:,} 条, 创建 {new_seeds:,} 新种子")

    if seed_buf:
        conn.executemany(
            "INSERT INTO _batch_seeds VALUES (?,?,?,?,?,?,?,?,?)",
            seed_buf
        )
        flush_batch_seeds(conn)

    wiki.close()

    log.info(f"Wikipedia: 扫描 {total:,} 条目, 创建 {new_seeds:,} 新种子")
    final_nodes = conn.execute("SELECT COUNT(*) FROM seeds").fetchone()[0]
    log.info(f"第四层完成: {final_nodes:,} 节点")

    return {
        'wiki_entries': total,
        'new_seeds': new_seeds,
    }


# ═══════════════════════════════════════════════════════
#  统计 & 验证
# ═══════════════════════════════════════════════════════

def show_stats(conn, output_db: pathlib.Path):
    """显示导入统计"""
    nodes = conn.execute("SELECT COUNT(*) FROM seeds").fetchone()[0]
    edges = conn.execute("SELECT COUNT(*) FROM karma_edges").fetchone()[0]
    edges_by_rel = conn.execute(
        "SELECT relation, COUNT(*) FROM karma_edges GROUP BY relation ORDER BY COUNT(*) DESC"
    ).fetchall()
    edges_by_tag = conn.execute(
        "SELECT source_tag, COUNT(*) FROM karma_edges GROUP BY source_tag"
    ).fetchall()
    nodes_with_def = conn.execute(
        "SELECT COUNT(*) FROM seeds WHERE definition != ''"
    ).fetchone()[0]
    nodes_with_domain = conn.execute(
        "SELECT COUNT(*) FROM seeds WHERE domain != ''"
    ).fetchone()[0]

    log.info("=" * 60)
    log.info("导入完成统计")
    log.info("=" * 60)
    log.info(f"节点总数:   {nodes:,}")
    log.info(f"  有释义:   {nodes_with_def:,}")
    log.info(f"  有领域:   {nodes_with_domain:,}")
    log.info(f"边总数:     {edges:,}")
    log.info(f"按关系类型:")
    for rel, cnt in edges_by_rel:
        log.info(f"  {rel}: {cnt:,}")
    log.info(f"按来源标签:")
    for tag, cnt in edges_by_tag:
        log.info(f"  {tag}: {cnt:,}")

    file_size = output_db.stat().st_size / 1024**2
    log.info(f"数据库文件: {file_size:.1f} MB")

    return {
        'nodes': nodes,
        'edges': edges,
        'relations': dict(edges_by_rel),
        'size_mb': file_size,
    }


def verify_queries(conn):
    """验证查询：10 个测试查询全部命中"""
    test_queries = [
        '感冒', '量子力学', '龙飞凤舞', '苏轼', '人工智能',
        '光合作用', '非典', '牛顿', '中国历史', '电脑',
    ]

    log.info("=" * 60)
    log.info("验证查询")
    log.info("=" * 60)

    results = {}
    for q in test_queries:
        # 精确匹配
        exact = conn.execute(
            "SELECT label, definition, domain FROM seeds WHERE label = ?", (q,)
        ).fetchone()

        if exact:
            results[q] = {'matched': 'exact', 'def': exact[1][:50] if exact[1] else '', 'domain': exact[2]}
            log.info(f"  ✅ {q}: 精确命中 (domain={exact[2]}, def={exact[1][:40] if exact[1] else 'N/A'})")
        else:
            # 模糊匹配（别名）
            fuzzy = conn.execute(
                "SELECT label, aliases FROM seeds WHERE aliases LIKE ? LIMIT 3",
                (f'%"{q}"%',)
            ).fetchall()
            if fuzzy:
                results[q] = {'matched': 'alias', 'found_in': [f[0] for f in fuzzy]}
                log.info(f"  ✅ {q}: 别名命中 → {[f[0] for f in fuzzy]}")
            else:
                results[q] = {'matched': 'miss'}
                log.info(f"  ❌ {q}: 未命中")

    hit_count = sum(1 for r in results.values() if r['matched'] != 'miss')
    log.info(f"命中率: {hit_count}/{len(test_queries)}")

    return results


# ═══════════════════════════════════════════════════════
#  main
# ═══════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description='识海知识库导入')
    parser.add_argument('--dry-run', action='store_true', help='只预览统计')
    parser.add_argument('--layer', type=int, choices=[1,2,3,4], help='只跑指定层')
    parser.add_argument('--db', type=str, default=None, help='输出数据库路径')
    parser.add_argument('--data-dir', type=str, default=None,
                        help='数据目录路径（优先级: 命令行 > 环境变量 CONSCIOUSNESS_SEA_DATA_DIR > 脚本相对 data/）')
    args = parser.parse_args()

    # 解析数据目录
    data_dir = resolve_data_dir(args.data_dir)

    # 构建数据文件路径
    output_db = pathlib.Path(args.db) if args.db else data_dir / 'consciousness_sea.db'
    cg_db_path = data_dir / CG_DB_FILENAME
    cedict_path = data_dir / CEDICT_FILENAME
    idioms_path = data_dir / 'idioms.json'  # 成语库文件名
    zhwiki_db_path = data_dir / ZHWIKI_FILENAME

    # 检查必需的数据文件
    if not args.dry_run:
        missing = check_data_files(data_dir)
        if missing:
            log.error("以下数据文件缺失，无法继续导入：")
            for f in missing:
                log.error(f"  - {f}")
            log.error("请确保数据文件存在于指定目录，或使用 --data-dir 指定正确的数据目录。")
            sys.exit(1)

    # 确保输出目录存在
    output_db.parent.mkdir(parents=True, exist_ok=True)

    t_start = time.time()

    if args.dry_run:
        log.info("DRY RUN — 仅预览统计")
        log.info(f"数据目录: {data_dir}")
        import_layer1(None, cg_db_path, dry_run=True)
        import_layer2(None, cg_db_path, dry_run=True)
        import_layer3(None, cedict_path, idioms_path, dry_run=True)
        import_layer4(None, zhwiki_db_path, dry_run=True)
        log.info(f"预览完成 ({time.time() - t_start:.1f}s)")
        return

    conn = init_db(output_db)

    all_layers = args.layer is None  # 默认跑全部

    stats = {}

    if all_layers or args.layer == 1:
        t1 = time.time()
        stats['layer1'] = import_layer1(conn, cg_db_path)
        log.info(f"第一层耗时: {time.time() - t1:.1f}s")

    if all_layers or args.layer == 2:
        t2 = time.time()
        stats['layer2'] = import_layer2(conn, cg_db_path)
        log.info(f"第二层耗时: {time.time() - t2:.1f}s")

    if all_layers or args.layer == 3:
        t3 = time.time()
        stats['layer3'] = import_layer3(conn, cedict_path, idioms_path)
        log.info(f"第三层耗时: {time.time() - t3:.1f}s")

    if all_layers or args.layer == 4:
        t4 = time.time()
        stats['layer4'] = import_layer4(conn, zhwiki_db_path)
        log.info(f"第四层耗时: {time.time() - t4:.1f}s")

    # 最终统计
    final = show_stats(conn, output_db)

    # 验证
    verify_results = verify_queries(conn)

    conn.close()

    elapsed = time.time() - t_start
    log.info(f"\n🎉 全部完成! 耗时: {elapsed:.0f}s ({elapsed/60:.1f}min)")

    # 保存统计摘要
    summary_path = output_db.parent / 'import_summary.json'
    summary = {
        'elapsed_s': elapsed,
        'layers': stats,
        'final': final,
        'verify': verify_results,
    }
    with open(summary_path, 'w', encoding='utf-8') as f:
        json.dump(summary, f, ensure_ascii=False, indent=2, default=str)
    log.info(f"摘要保存: {summary_path}")


if __name__ == '__main__':
    main()
