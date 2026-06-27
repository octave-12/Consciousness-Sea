#!/usr/bin/env python3
"""识海全量并行导入 — 17种关系，多进程读+独立写+合并"""
import sqlite3, multiprocessing, time, logging, json, sys, shutil, argparse
import pathlib

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
log = logging.getLogger('import_all')

# 使用 pathlib.Path 和 resolve_data_dir() 替代硬编码路径
from core.config import (
    resolve_data_dir,
    CG_DB_FILENAME,
    CEDICT_FILENAME,
    ZHWIKI_FILENAME,
)

BATCH = 50000

# 需要导入的语义关系 → 权重策略
# weight_map: None=跳过, 'fixed:N'=固定权重, 'linear'=线性映射 c→[0.1,0.7]
RELATIONS = {
    'IS_A':              'fixed:0.8',      # 层级骨架
    'SYNONYM':           'alias',           # → aliases, 不进边表
    'HAS':               'fixed:0.5',       # 拥有关系
    'DEFINED_AS':        'fixed:0.5',       # 定义
    'RELATED':           'linear',          # 横向关联
    'COOCCURS_WITH':     'linear',          # 共现
    'CAUSE':             'linear',          # 因果
    'PART_OF':           'linear',          # 组成部分
    'BEFORE':            'fixed:0.4',       # 时间先后
    'LOCATED_IN':        'fixed:0.5',       # 位置
    'HAS_SUBEVENT':      'fixed:0.5',       # 子事件
    'HAS_CAPABILITY':    'fixed:0.5',       # 能力
    'MADE_OF':           'fixed:0.5',       # 材质
    'USED_FOR':          'fixed:0.5',       # 用途
    'HAS_PROPERTY':      'fixed:0.5',       # 属性
    'FOLLOWS':           'fixed:0.5',       # 因果链
    'COOCCURS_IN':       'fixed:0.4',       # 位置共现
}

# 排除的噪音
SKIP = {'POETIC_NEXT', 'POETIC_WITH', 'HAS_PINYIN'}

def weight_from(c, strategy):
    if strategy == 'linear':
        return round(0.1 + 0.6 * max(c, 0), 3)
    elif strategy.startswith('fixed:'):
        return float(strategy.split(':')[1])
    return 0.5

def worker_cg(rel_list, out_dir, src_db_path):
    """Worker: 从 concept_graph.db 读取指定关系，写入临时 SQLite"""
    name = '-'.join(sorted(rel_list))[:40]
    out_path = pathlib.Path(out_dir) / f'cg_{name}.db'
    if out_path.exists():
        out_path.unlink()

    out = sqlite3.connect(str(out_path))
    out.execute("PRAGMA journal_mode=OFF")
    out.execute("PRAGMA synchronous=OFF")
    out.execute("CREATE TABLE nodes (label TEXT PRIMARY KEY)")
    out.execute("CREATE TABLE edges (source TEXT, target TEXT, relation TEXT, weight REAL, source_tag TEXT)")
    out.execute("CREATE TABLE aliases (node TEXT, alias TEXT)")

    src = sqlite3.connect(f'file:{src_db_path}?mode=ro', uri=True)

    nodes_buf = set()
    edges_buf = []
    alias_buf = []
    total_edges = 0
    total_nodes = 0
    total_aliases = 0

    for rel in rel_list:
        strategy = RELATIONS[rel]
        cursor = src.execute("SELECT s, o, c FROM triples WHERE r = ?", (rel,))
        for s, o, c in cursor:
            if strategy == 'alias':
                alias_buf.append((s, o))
                nodes_buf.add(s)
                if len(alias_buf) >= BATCH:
                    out.executemany("INSERT INTO aliases VALUES (?,?)", alias_buf)
                    total_aliases += len(alias_buf)
                    alias_buf = []
            else:
                w = weight_from(c, strategy)
                edges_buf.append((s, o, rel, w, 'loong_cg_import'))
                nodes_buf.add(s)
                nodes_buf.add(o)

            if len(edges_buf) >= BATCH:
                out.executemany("INSERT INTO edges VALUES (?,?,?,?,?)", edges_buf)
                total_edges += len(edges_buf)
                edges_buf = []
            if len(nodes_buf) >= BATCH:
                out.executemany("INSERT OR IGNORE INTO nodes VALUES (?)",
                                [(n,) for n in nodes_buf])
                total_nodes += len(nodes_buf)
                nodes_buf = set()

    # flush
    if edges_buf:
        out.executemany("INSERT INTO edges VALUES (?,?,?,?,?)", edges_buf)
        total_edges += len(edges_buf)
    if nodes_buf:
        out.executemany("INSERT OR IGNORE INTO nodes VALUES (?)",
                        [(n,) for n in nodes_buf])
        total_nodes += len(nodes_buf)
    if alias_buf:
        out.executemany("INSERT INTO aliases VALUES (?,?)", alias_buf)
        total_aliases += len(alias_buf)

    src.close()
    out.execute("CREATE INDEX idx_edges_rel ON edges(relation)")
    out.commit()
    out.close()

    log.info(f"[{name}] {total_nodes:,} nodes, {total_edges:,} edges, {total_aliases:,} aliases → {out_path}")
    return str(out_path), total_nodes, total_edges, total_aliases

def worker_cedict(out_dir, cedict_path):
    """Worker: CC-CEDICT → 种子释义"""
    out_path = pathlib.Path(out_dir) / 'cedict_defs.db'
    if out_path.exists():
        out_path.unlink()
    out = sqlite3.connect(str(out_path))
    out.execute("CREATE TABLE defs (label TEXT PRIMARY KEY, definition TEXT, pinyin TEXT)")

    with open(cedict_path, 'r', encoding='utf-8') as f:
        cedict = json.load(f)

    buf = []
    total = 0
    for word, info in cedict.items():
        if not isinstance(info, dict):
            continue
        defs = info.get('definitions', [])
        definition = '; '.join(defs) if isinstance(defs, list) and defs else info.get('definition', '')
        pinyin = info.get('pinyin', '')
        if definition:
            buf.append((word, definition, pinyin))
            total += 1
        if len(buf) >= BATCH:
            out.executemany("INSERT INTO defs VALUES (?,?,?)", buf)
            buf = []
    if buf:
        out.executemany("INSERT INTO defs VALUES (?,?,?)", buf)

    out.commit()
    out.close()
    log.info(f"[cedict] {total:,} definitions → {out_path}")
    return str(out_path), total

def worker_idioms(out_dir, idioms_path):
    """Worker: 成语 → 新种子"""
    out_path = pathlib.Path(out_dir) / 'idioms.db'
    if out_path.exists():
        out_path.unlink()
    out = sqlite3.connect(str(out_path))
    out.execute("CREATE TABLE nodes (label TEXT PRIMARY KEY, domain TEXT)")

    with open(idioms_path, 'r', encoding='utf-8') as f:
        idioms = json.load(f)

    buf = []
    total = 0
    for item in idioms:
        word = item if isinstance(item, str) else item.get('word', item.get('idiom', ''))
        if word and len(word) >= 2:
            buf.append((word, '成语'))
            total += 1
        if len(buf) >= BATCH:
            out.executemany("INSERT OR IGNORE INTO nodes VALUES (?,?)", buf)
            buf = []
    if buf:
        out.executemany("INSERT OR IGNORE INTO nodes VALUES (?,?)", buf)

    out.commit()
    out.close()
    log.info(f"[idioms] {total:,} nodes → {out_path}")
    return str(out_path), total

def worker_wikipedia(out_dir, zhwiki_db_path):
    """Worker: Wikipedia 条目 → 低频种子"""
    out_path = pathlib.Path(out_dir) / 'wiki.db'
    if out_path.exists():
        out_path.unlink()
    out = sqlite3.connect(str(out_path))
    out.execute("CREATE TABLE nodes (label TEXT PRIMARY KEY)")

    zhwiki_path = pathlib.Path(zhwiki_db_path)
    if not zhwiki_path.exists():
        log.warning("zhwiki.db not found, skipping")
        out.close()
        return str(out_path), 0

    wiki = sqlite3.connect(f'file:{zhwiki_path}?mode=ro', uri=True)
    cursor = wiki.execute("SELECT title FROM articles")

    buf = []
    total = 0
    for (title,) in cursor:
        if title and len(title) >= 2:
            if not any(c in title for c in '{}[]<>|/"\'+=&%$#@!~`^*'):
                buf.append((title,))
                total += 1
        if len(buf) >= BATCH:
            out.executemany("INSERT OR IGNORE INTO nodes VALUES (?)", buf)
            buf = []
    if buf:
        out.executemany("INSERT OR IGNORE INTO nodes VALUES (?)", buf)

    wiki.close()
    out.commit()
    out.close()
    log.info(f"[wiki] {total:,} nodes → {out_path}")
    return str(out_path), total


def merge_all(db_path, temp_files, cedict_file, idioms_file, wiki_file):
    """合并所有临时 DB 到主 DB"""
    dst = sqlite3.connect(str(db_path))
    dst.execute("PRAGMA journal_mode=OFF")
    dst.execute("PRAGMA synchronous=OFF")
    dst.execute("PRAGMA cache_size=-2000000")

    # 创建主表
    dst.executescript("""
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

    # 1. 合并概念图 worker 的节点和边
    all_nodes = set()
    for tf in temp_files:
        tf_path = pathlib.Path(tf)
        if not tf_path.exists(): continue
        log.info(f"Merging {tf_path.name}...")
        t = sqlite3.connect(f'file:{tf}?mode=ro', uri=True)
        dst.execute("ATTACH ? AS tmp", (tf,))

        # 节点
        try:
            for (lbl,) in t.execute("SELECT DISTINCT label FROM nodes"):
                all_nodes.add(lbl)
        except Exception:
            pass

        # 边
        try:
            count = t.execute("SELECT COUNT(*) FROM edges").fetchone()[0]
            if count > 0:
                dst.execute("""
                    INSERT OR IGNORE INTO karma_edges (source,target,relation,weight,source_tag)
                    SELECT source,target,relation,weight,source_tag FROM tmp.edges
                """)
                log.info(f"  edges: {count:,}")
        except Exception:
            pass

        # 别名
        try:
            dst.execute("""
                INSERT OR IGNORE INTO karma_edges (source,target,relation,weight,source_tag)
                SELECT source,target,relation,weight,source_tag FROM tmp.edges
            """)
        except Exception:
            pass

        dst.execute("DETACH tmp")
        t.close()

    # 2. 合并 CC-CEDICT 释义
    cedict_path = pathlib.Path(cedict_file) if cedict_file else None
    if cedict_path and cedict_path.exists():
        log.info("Merging CC-CEDICT...")
        dst.execute("ATTACH ? AS cd", (cedict_file,))
        dst.execute("""
            INSERT OR IGNORE INTO seeds (id,label,type,aliases,definition,pinyin)
            SELECT label,label,'CONCEPT','[]',definition,pinyin FROM cd.defs
        """)
        dst.execute("""
            UPDATE seeds SET definition = (SELECT definition FROM cd.defs WHERE cd.defs.label = seeds.label),
                              pinyin = (SELECT pinyin FROM cd.defs WHERE cd.defs.label = seeds.label)
            WHERE EXISTS (SELECT 1 FROM cd.defs WHERE cd.defs.label = seeds.label)
        """)
        dst.execute("DETACH cd")

    # 3. 合并成语
    idioms_path = pathlib.Path(idioms_file) if idioms_file else None
    if idioms_path and idioms_path.exists():
        log.info("Merging idioms...")
        dst.execute("ATTACH ? AS idi", (idioms_file,))
        dst.execute("""
            INSERT OR IGNORE INTO seeds (id,label,type,aliases,domain)
            SELECT label,label,'CONCEPT','[]',domain FROM idi.nodes
        """)
        dst.execute("DETACH idi")

    # 4. 合并 Wikipedia
    wiki_path = pathlib.Path(wiki_file) if wiki_file else None
    if wiki_path and wiki_path.exists():
        log.info("Merging Wikipedia...")
        dst.execute("ATTACH ? AS wk", (wiki_file,))
        dst.execute("""
            INSERT OR IGNORE INTO seeds (id,label,type,aliases,activation_bias,meta)
            SELECT label,label,'CONCEPT','[]',0.05,'{"source":"wikipedia"}' FROM wk.nodes
        """)
        dst.execute("DETACH wk")

    # 5. 写入所有概念图节点
    log.info(f"Writing {len(all_nodes):,} nodes...")
    node_buf = [(n, n, 'CONCEPT', '[]') for n in all_nodes]
    for i in range(0, len(node_buf), BATCH):
        batch = node_buf[i:i+BATCH]
        dst.executemany(
            "INSERT OR IGNORE INTO seeds (id,label,type,aliases) VALUES (?,?,?,?)",
            batch
        )
        if i % (BATCH*5) == 0:
            dst.commit()

    dst.commit()

    # 6. 别名：SYNONYM → seeds.aliases 合并
    log.info("Merging aliases...")
    for tf in temp_files:
        tf_path = pathlib.Path(tf)
        if not tf_path.exists(): continue
        t = sqlite3.connect(f'file:{tf}?mode=ro', uri=True)
        try:
            for node, alias in t.execute("SELECT DISTINCT node, alias FROM aliases"):
                dst.execute("""
                    UPDATE seeds SET aliases = (
                        SELECT json_group_array(DISTINCT value) FROM (
                            SELECT value FROM json_each(aliases)
                            UNION SELECT ?
                        )
                    ) WHERE label = ?
                """, (alias, node))
        except Exception:
            pass
        t.close()

    dst.commit()

    # 7. 建索引
    log.info("Creating indexes...")
    dst.execute("CREATE INDEX idx_seeds_label ON seeds(label)")
    dst.execute("CREATE INDEX idx_edges_source ON karma_edges(source)")
    dst.execute("CREATE INDEX idx_edges_target ON karma_edges(target)")
    dst.execute("CREATE INDEX idx_edges_relation ON karma_edges(relation)")

    # 统计
    nodes = dst.execute("SELECT COUNT(*) FROM seeds").fetchone()[0]
    edges = dst.execute("SELECT COUNT(*) FROM karma_edges").fetchone()[0]
    defs = dst.execute("SELECT COUNT(*) FROM seeds WHERE definition != ''").fetchone()[0]

    db_path_obj = pathlib.Path(db_path)
    log.info(f"Final: {nodes:,} nodes, {edges:,} edges, {defs:,} with definitions")
    log.info(f"DB size: {db_path_obj.stat().st_size/1024**2:.0f} MB")

    dst.close()
    return nodes, edges, defs

def main():
    parser = argparse.ArgumentParser(description='识海全量并行导入')
    parser.add_argument('--data-dir', type=str, default=None,
                        help='数据目录路径（优先级: 命令行 > 环境变量 > 脚本相对 data/）')
    args = parser.parse_args()

    # 解析数据目录
    data_dir = resolve_data_dir(args.data_dir)

    # 构建数据文件路径
    src_db = data_dir / CG_DB_FILENAME
    dst_db = data_dir / 'consciousness_sea.db'
    tmp_dir = data_dir / 'tmp'
    cedict_path = data_dir / CEDICT_FILENAME
    idioms_path = data_dir / 'idioms.json'
    zhwiki_path = data_dir / ZHWIKI_FILENAME

    # 检查必需数据文件
    required_files = {
        src_db: '龙珠概念图数据库',
        cedict_path: 'CC-CEDICT 解析文件',
    }
    missing = [f for f in required_files if not f.exists()]
    if missing:
        log.error("以下数据文件缺失，无法继续导入：")
        for f in missing:
            log.error(f"  - {f} ({required_files[f]})")
        log.error("请确保数据文件存在于指定目录，或使用 --data-dir 指定正确的数据目录。")
        sys.exit(1)

    tmp_dir.mkdir(parents=True, exist_ok=True)

    # 删除旧 DB
    for suffix in ['', '-wal', '-shm']:
        old_db = pathlib.Path(str(dst_db) + suffix)
        if old_db.exists():
            old_db.unlink()

    t0 = time.time()

    # 分组：4 个 worker 各分一组关系
    all_rels = [r for r in RELATIONS if r not in SKIP and RELATIONS[r] != 'alias']
    alias_rels = [r for r in RELATIONS if RELATIONS[r] == 'alias']
    chunk = len(all_rels) // 4 + 1
    groups = [all_rels[i:i+chunk] for i in range(0, len(all_rels), chunk)]
    # SYNONYM 单独一个 worker
    groups.append(alias_rels)

    log.info(f"Workers: {len(groups)} groups: {[g[:3] for g in groups]}")

    pool = multiprocessing.Pool(len(groups))
    results = []

    # 启动概念图 worker
    for g in groups:
        if g:
            results.append(pool.apply_async(worker_cg, (g, str(tmp_dir), str(src_db))))

    # 启动 CC-CEDICT、成语、Wikipedia worker
    cedict_r = pool.apply_async(worker_cedict, (str(tmp_dir), str(cedict_path)))
    idioms_r = pool.apply_async(worker_idioms, (str(tmp_dir), str(idioms_path)))
    wiki_r = pool.apply_async(worker_wikipedia, (str(tmp_dir), str(zhwiki_path)))

    pool.close()

    # 收集结果
    temp_files = []
    for r in results:
        path, n, e, a = r.get()
        temp_files.append(path)

    cedict_path_result, cedict_count = cedict_r.get()
    idioms_path_result, idioms_count = idioms_r.get()
    wiki_path_result, wiki_count = wiki_r.get()

    pool.join()

    log.info(f"All workers done in {time.time()-t0:.0f}s, merging...")

    # 合并
    nodes, edges, defs = merge_all(str(dst_db), temp_files, cedict_path_result, idioms_path_result, wiki_path_result)

    elapsed = time.time() - t0
    log.info(f"\n🎉 Done! {elapsed:.0f}s ({elapsed/60:.1f}min)")
    log.info(f"   {nodes:,} nodes, {edges:,} edges, {defs:,} definitions")
    log.info(f"   DB: {dst_db.stat().st_size/1024**2:.0f} MB")

    # 清理临时文件
    shutil.rmtree(tmp_dir, ignore_errors=True)

    # 验证
    dst = sqlite3.connect(str(dst_db))
    for q in ['感冒','量子力学','人工智能','苏轼','龙','电脑']:
        r = dst.execute("SELECT label, definition FROM seeds WHERE label=?", (q,)).fetchone()
        if r:
            log.info(f"  ✅ {r[0]}: {r[1][:50] if r[1] else 'N/A'}")
        else:
            log.info(f"  ❌ {q}: miss")
    dst.close()

if __name__ == '__main__':
    main()
