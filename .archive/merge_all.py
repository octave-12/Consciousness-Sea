#!/usr/bin/env python3
"""合并临时 DB → consciousness_sea.db — 修复版，避免 ATTACH 锁问题"""
import sqlite3, os, logging, json

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
log = logging.getLogger('merge_all')

DST = '/mnt/d/soso/projects/consciousnessSea/data/consciousness_sea.db'
TMP = '/mnt/d/soso/projects/consciousnessSea/data/tmp'
BATCH = 50000

TEMP_FILES = [
    'cg_COOCCURS_WITH-DEFINED_AS-HAS-IS_A-RELATE.db',
    'cg_BEFORE-CAUSE-HAS_SUBEVENT-LOCATED_IN-PAR.db',
    'cg_FOLLOWS-HAS_CAPABILITY-HAS_PROPERTY-MADE.db',
    'cg_COOCCURS_IN.db',
    'cg_SYNONYM.db',
]
CEDICT = f'{TMP}/cedict_defs.db'
IDIOMS = f'{TMP}/idioms.db'
WIKI = f'{TMP}/wiki.db'


def merge_temp_files():
    dst = sqlite3.connect(DST)
    dst.execute("PRAGMA journal_mode=WAL")
    dst.execute("PRAGMA synchronous=NORMAL")
    dst.execute("PRAGMA cache_size=-500000")

    # 建表
    dst.executescript("""
        CREATE TABLE IF NOT EXISTS seeds (
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
        CREATE TABLE IF NOT EXISTS karma_edges (
            source TEXT NOT NULL, target TEXT NOT NULL, relation TEXT NOT NULL,
            weight REAL NOT NULL DEFAULT 0.5,
            source_tag TEXT NOT NULL DEFAULT '',
            created_at TEXT NOT NULL DEFAULT (datetime('now')),
            PRIMARY KEY (source, target, relation)
        );
    """)

    all_nodes = set()
    total_edges = 0
    total_aliases = 0

    for tf_name in TEMP_FILES:
        tf = os.path.join(TMP, tf_name)
        if not os.path.exists(tf):
            log.info(f"Skip {tf_name} (not found)")
            continue

        log.info(f"Merging {tf_name}...")
        src = sqlite3.connect(f'file:{tf}?mode=ro', uri=True)

        # 收集节点
        node_count = 0
        for (lbl,) in src.execute("SELECT label FROM nodes"):
            if lbl not in all_nodes:
                all_nodes.add(lbl)
                node_count += 1
        log.info(f"  nodes: {node_count:,} new ({len(all_nodes):,} total)")

        # 边 — 分批读+写
        edge_count = 0
        batch = []
        for s, t, r, w, tag in src.execute("SELECT source, target, relation, weight, source_tag FROM edges"):
            batch.append((s, t, r, w, tag))
            if len(batch) >= BATCH:
                dst.executemany(
                    "INSERT OR IGNORE INTO karma_edges (source,target,relation,weight,source_tag) VALUES (?,?,?,?,?)",
                    batch
                )
                dst.commit()
                edge_count += len(batch)
                batch = []
        if batch:
            dst.executemany(
                "INSERT OR IGNORE INTO karma_edges (source,target,relation,weight,source_tag) VALUES (?,?,?,?,?)",
                batch
            )
            dst.commit()
            edge_count += len(batch)
        log.info(f"  edges: {edge_count:,}")
        total_edges += edge_count

        # 别名
        alias_count = 0
        try:
            alias_map = {}
            for node, alias in src.execute("SELECT node, alias FROM aliases"):
                if node not in alias_map:
                    alias_map[node] = []
                alias_map[node].append(alias)
                alias_count += 1
            if alias_map:
                alias_batch = []
                for node, aliases in alias_map.items():
                    alias_batch.append((json.dumps(aliases, ensure_ascii=False), node))
                    if len(alias_batch) >= BATCH:
                        dst.executemany("UPDATE seeds SET aliases=? WHERE label=?", alias_batch)
                        dst.commit()
                        alias_batch = []
                if alias_batch:
                    dst.executemany("UPDATE seeds SET aliases=? WHERE label=?", alias_batch)
                    dst.commit()
            log.info(f"  aliases: {alias_count:,}")
            total_aliases += alias_count
        except Exception as e:
            log.info(f"  aliases: none ({e})")

        src.close()

    # 写入所有概念节点
    log.info(f"Writing {len(all_nodes):,} concept nodes...")
    node_batch = []
    for i, n in enumerate(all_nodes):
        node_batch.append((n, n, 'CONCEPT', '[]'))
        if len(node_batch) >= BATCH:
            dst.executemany(
                "INSERT OR IGNORE INTO seeds (id, label, type, aliases) VALUES (?,?,?,?)",
                node_batch
            )
            dst.commit()
            if (i+1) % (BATCH*10) == 0:
                log.info(f"  {i+1:,}/{len(all_nodes):,}")
            node_batch = []
    if node_batch:
        dst.executemany(
            "INSERT OR IGNORE INTO seeds (id, label, type, aliases) VALUES (?,?,?,?)",
            node_batch
        )
        dst.commit()

    # Merge CC-CEDICT
    if os.path.exists(CEDICT):
        log.info("Merging CC-CEDICT...")
        src = sqlite3.connect(f'file:{CEDICT}?mode=ro', uri=True)
        batch = []
        for lbl, defn, py in src.execute("SELECT label, definition, pinyin FROM defs"):
            batch.append((defn, py, lbl))
            if len(batch) >= BATCH:
                dst.executemany("UPDATE seeds SET definition=?, pinyin=? WHERE label=?", batch)
                dst.commit()
                batch = []
        if batch:
            dst.executemany("UPDATE seeds SET definition=?, pinyin=? WHERE label=?", batch)
            dst.commit()
        src.close()

    # Merge idioms
    if os.path.exists(IDIOMS):
        log.info("Merging idioms...")
        src = sqlite3.connect(f'file:{IDIOMS}?mode=ro', uri=True)
        batch = []
        for lbl, domain in src.execute("SELECT label, domain FROM nodes"):
            batch.append((domain, lbl))
            if len(batch) >= BATCH:
                dst.executemany("UPDATE seeds SET domain=? WHERE label=?", batch)
                dst.commit()
                batch = []
        if batch:
            dst.executemany("UPDATE seeds SET domain=? WHERE label=?", batch)
            dst.commit()
        src.close()

    # Merge Wikipedia
    if os.path.exists(WIKI):
        log.info("Merging Wikipedia...")
        src = sqlite3.connect(f'file:{WIKI}?mode=ro', uri=True)
        batch = []
        for (lbl,) in src.execute("SELECT label FROM nodes"):
            batch.append(lbl)
            if len(batch) >= BATCH:
                dst.executemany(
                    "INSERT OR IGNORE INTO seeds (id,label,type,aliases,activation_bias,meta) VALUES (?,?,'CONCEPT','[]',0.05,'{\"source\":\"wikipedia\"}')",
                    [(l, l) for l in batch]
                )
                dst.commit()
                batch = []
        if batch:
            dst.executemany(
                "INSERT OR IGNORE INTO seeds (id,label,type,aliases,activation_bias,meta) VALUES (?,?,'CONCEPT','[]',0.05,'{\"source\":\"wikipedia\"}')",
                [(l, l) for l in batch]
            )
            dst.commit()
        src.close()

    # 最终统计
    nodes = dst.execute("SELECT COUNT(*) FROM seeds").fetchone()[0]
    edges = dst.execute("SELECT COUNT(*) FROM karma_edges").fetchone()[0]
    size_mb = os.path.getsize(DST) / (1024**2)

    log.info(f"\n{'='*60}")
    log.info(f"Nodes: {nodes:,}")
    log.info(f"Edges: {edges:,}")
    log.info(f"DB: {size_mb:.0f} MB")
    for rel, cnt in dst.execute("SELECT relation, COUNT(*) FROM karma_edges GROUP BY relation ORDER BY COUNT(*) DESC"):
        log.info(f"  {rel}: {cnt:,}")

    dst.close()


if __name__ == '__main__':
    merge_temp_files()
