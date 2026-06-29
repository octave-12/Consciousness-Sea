#!/usr/bin/env python3
"""补导入 RELATED — 并行读取 concept_graph.db，批量写入 consciousness_sea.db"""
import sqlite3, multiprocessing, time, logging, sys, argparse
import pathlib

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
log = logging.getLogger('import_related')

# 使用 pathlib.Path 和 resolve_data_dir() 替代硬编码路径
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent / "src"))
from consciousness_sea.infrastructure.config import (
    resolve_data_dir,
    CG_DB_FILENAME,
)

BATCH = 50000
WORKERS = 4

def count_total(src_db_path: pathlib.Path):
    src = sqlite3.connect(f'file:{src_db_path}?mode=ro', uri=True)
    n = src.execute("SELECT COUNT(*) FROM triples WHERE r='RELATED'").fetchone()[0]
    src.close()
    return n

def worker(offset, limit, result_queue, src_db_path):
    """读 concept_graph.db 的一段 RELATED 数据"""
    src = sqlite3.connect(f'file:{src_db_path}?mode=ro', uri=True)
    rows = src.execute(
        "SELECT s, o, c FROM triples WHERE r='RELATED' LIMIT ? OFFSET ?",
        (limit, offset)
    ).fetchall()
    src.close()
    result_queue.put(rows)
    log.info(f"Worker offset={offset}: read {len(rows)} rows")

def writer(result_queue, total, dst_db_path: pathlib.Path):
    """写入 consciousness_sea.db，边 + 新节点"""
    dst = sqlite3.connect(str(dst_db_path))
    dst.execute("PRAGMA journal_mode=WAL")
    dst.execute("PRAGMA synchronous=OFF")
    dst.execute("PRAGMA cache_size=-500000")

    # 获取已有节点
    existing = set()
    for (lbl,) in dst.execute("SELECT label FROM seeds"):
        existing.add(lbl)

    # 获取已有 RELATED 边（避免重复）
    existing_edges = set()
    for (s, t) in dst.execute("SELECT source, target FROM karma_edges WHERE relation='RELATED'"):
        existing_edges.add((s, t))

    edges_written = 0
    nodes_written = 0
    pending_edges = []
    pending_nodes = []
    done_workers = 0

    while done_workers < WORKERS:
        rows = result_queue.get()
        if rows is None:
            done_workers += 1
            continue

        for s, o, c in rows:
            # 权重映射：c ∈ [0,1] → [0.1, 0.7]
            weight = 0.1 + 0.6 * c if c > 0 else 0.1
            key = (s, o)
            if key not in existing_edges:
                existing_edges.add(key)
                pending_edges.append((s, o, 'RELATED', weight, 'loong_cg_import'))

                if s not in existing:
                    existing.add(s)
                    pending_nodes.append((s, s, 'CONCEPT', '[]', '', '', '', 0.0, '{}'))
                    nodes_written += 1
                if o not in existing:
                    existing.add(o)
                    pending_nodes.append((o, o, 'CONCEPT', '[]', '', '', '', 0.0, '{}'))
                    nodes_written += 1

        if len(pending_nodes) >= BATCH:
            dst.executemany(
                "INSERT OR IGNORE INTO seeds (id,label,type,aliases,domain,definition,pinyin,activation_bias,meta) VALUES (?,?,?,?,?,?,?,?,?)",
                pending_nodes
            )
            dst.commit()
            pending_nodes = []

        if len(pending_edges) >= BATCH:
            dst.executemany(
                "INSERT OR IGNORE INTO karma_edges (source,target,relation,weight,source_tag) VALUES (?,?,?,?,?)",
                pending_edges
            )
            edges_written += len(pending_edges)
            dst.commit()
            pending_edges = []
            log.info(f"Edges: {edges_written:,} / ~{total:,}")

    # flush
    if pending_nodes:
        dst.executemany(
            "INSERT OR IGNORE INTO seeds (id,label,type,aliases,domain,definition,pinyin,activation_bias,meta) VALUES (?,?,?,?,?,?,?,?,?)",
            pending_nodes
        )
    if pending_edges:
        dst.executemany(
            "INSERT OR IGNORE INTO karma_edges (source,target,relation,weight,source_tag) VALUES (?,?,?,?,?)",
            pending_edges
        )
        edges_written += len(pending_edges)
    dst.commit()

    log.info(f"Done: {edges_written:,} edges, {nodes_written:,} new nodes")
    dst.close()

def main():
    parser = argparse.ArgumentParser(description='补导入 RELATED 边')
    parser.add_argument('--data-dir', type=str, default=None,
                        help='数据目录路径（优先级: 命令行 > 环境变量 > 脚本相对 data/）')
    args = parser.parse_args()

    # 解析数据目录
    data_dir = resolve_data_dir(args.data_dir)

    # 构建数据文件路径
    src_db = data_dir / CG_DB_FILENAME
    dst_db = data_dir / 'consciousness_sea.db'

    # 检查必需数据文件
    if not src_db.exists():
        log.error(f"龙珠概念图数据库缺失: {src_db}")
        log.error("请确保数据文件存在于指定目录，或使用 --data-dir 指定正确的数据目录。")
        sys.exit(1)

    if not dst_db.exists():
        log.error(f"识海数据库缺失: {dst_db}")
        log.error("请先运行 import_knowledge_base.py 创建识海数据库。")
        sys.exit(1)

    total = count_total(src_db)
    log.info(f"RELATED total: {total:,}")

    chunk = total // WORKERS + 1
    result_queue = multiprocessing.Queue()

    # Start writer
    w = multiprocessing.Process(target=writer, args=(result_queue, total, dst_db))
    w.start()

    # Start readers
    readers = []
    for i in range(WORKERS):
        offset = i * chunk
        limit = min(chunk, total - offset)
        if limit <= 0:
            result_queue.put(None)
            continue
        p = multiprocessing.Process(target=worker, args=(offset, limit, result_queue, src_db))
        p.start()
        readers.append(p)

    for p in readers:
        p.join()
        result_queue.put(None)  # signal done

    w.join()
    log.info("All done")

if __name__ == '__main__':
    main()