#!/usr/bin/env python3
"""
导入龙珠 DB 数据到识海 — v2 (简化批量版)

策略:
  - 使用 ATTACH 直接跨库查询，避免内存加载全部数据
  - 批量 UPDATE (50K/批，BEGIN/COMMIT)
  - 批量 INSERT 新节点
  - 嵌入向量单独存 numpy 文件
"""

import sqlite3, re, time, os, json, sys
import numpy as np

CS_DB = '/mnt/d/soso/projects/consciousnessSea/data/consciousness_sea.db'
DD_DB = '/home/octave/agent_data/dragonball/dragonball.db'
EMB_OUT = '/mnt/d/soso/projects/consciousnessSea/data/dragonball_embeddings.npy'
IDX_OUT = '/mnt/d/soso/projects/consciousnessSea/data/dragonball_embedding_index.json'
BATCH = 50_000

EMOJI_RE = re.compile(r'^[^\w\u4e00-\u9fff]+')

def clean(s):
    return EMOJI_RE.sub('', s).strip()

def main():
    t0 = time.time()

    # ━━━ 步骤 1: 从龙珠清洗标题，写入识海临时表 ━━━
    print("Step 1: Cleaning titles from dragonball...", flush=True)
    dd = sqlite3.connect(f'file:{DD_DB}?mode=ro', uri=True)
    
    cs = sqlite3.connect(CS_DB)
    cs.execute("PRAGMA journal_mode=WAL")
    cs.execute("PRAGMA synchronous=NORMAL")
    cs.execute("DROP TABLE IF EXISTS _dd_import")
    cs.execute("CREATE TEMP TABLE _dd_import (clean_title TEXT PRIMARY KEY, content TEXT, has_emb INTEGER DEFAULT 0)")
    
    total = 0
    buf = []
    for title, content, emb_blob in dd.execute(
        "SELECT title, content, embedding FROM nodes WHERE content != ''"
    ):
        c = clean(title)
        if not c:
            continue
        has_emb = 1 if emb_blob else 0
        buf.append((c, content, has_emb))
        total += 1
        if len(buf) >= BATCH:
            cs.executemany("INSERT OR REPLACE INTO _dd_import VALUES (?,?,?)", buf)
            buf = []
            print(f"  Step 1: {total:,} / ~128K", flush=True)
    if buf:
        cs.executemany("INSERT OR REPLACE INTO _dd_import VALUES (?,?,?)", buf)
    cs.commit()
    dd.close()
    print(f"  Step 1 done: {total:,} unique cleaned titles", flush=True)
    
    # ━━━ 步骤 2: 统计匹配 ━━━
    matched = cs.execute("""
        SELECT COUNT(*) FROM seeds s INNER JOIN _dd_import d ON s.label = d.clean_title
    """).fetchone()[0]
    unmatched = total - matched
    print(f"Step 2: Matched={matched:,}  Unmatched={unmatched:,}", flush=True)
    
    # ━━━ 步骤 3: 批量 UPDATE definition ━━━
    # 策略: 只更新当前 definition 为空或比龙珠 content 短的
    print("Step 3: Updating definitions...", flush=True)
    
    # 找出需要更新的 label 列表
    to_update = cs.execute("""
        SELECT s.label, d.content
        FROM seeds s
        INNER JOIN _dd_import d ON s.label = d.clean_title
        WHERE s.definition = '' OR length(s.definition) < length(d.content)
    """).fetchall()
    print(f"  {len(to_update):,} seeds to update", flush=True)
    
    # 分批 UPDATE
    updated = 0
    for i in range(0, len(to_update), BATCH):
        batch = to_update[i:i+BATCH]
        cs.execute("BEGIN")
        cs.executemany(
            "UPDATE seeds SET definition = ? WHERE label = ?",
            [(content, label) for label, content in batch]
        )
        cs.execute("COMMIT")
        updated += len(batch)
        if updated % (BATCH * 3) == 0:
            print(f"  Updated {updated:,} / {len(to_update):,}", flush=True)
    print(f"  Step 3 done: {updated:,} definitions updated", flush=True)
    
    # ━━━ 步骤 4: 批量 INSERT 新节点 ━━━
    if unmatched > 0:
        print(f"Step 4: Inserting {unmatched:,} new seeds...", flush=True)
        
        new_seeds = cs.execute("""
            SELECT d.clean_title, d.content
            FROM _dd_import d
            WHERE d.clean_title NOT IN (SELECT label FROM seeds)
        """).fetchall()
        print(f"  {len(new_seeds):,} new seeds to insert", flush=True)
        
        inserted = 0
        for i in range(0, len(new_seeds), BATCH):
            batch = new_seeds[i:i+BATCH]
            cs.execute("BEGIN")
            cs.executemany(
                """INSERT OR IGNORE INTO seeds (id, label, type, aliases, definition, meta)
                   VALUES (?, ?, 'CONCEPT', '[]', ?, '{"source":"dragonball"}')""",
                [(f'dd_{label}', label, content) for label, content in batch]
            )
            cs.execute("COMMIT")
            inserted += len(batch)
            if inserted % (BATCH * 2) == 0:
                print(f"  Inserted {inserted:,} / {len(new_seeds):,}", flush=True)
        print(f"  Step 4 done: {inserted:,} new seeds", flush=True)
    
    # ━━━ 步骤 5: 保存嵌入向量 ━━━
    print("Step 5: Extracting embeddings...", flush=True)
    dd = sqlite3.connect(f'file:{DD_DB}?mode=ro', uri=True)
    
    vecs = []
    labels = []
    count = 0
    for title, emb_blob in dd.execute(
        "SELECT title, embedding FROM nodes WHERE embedding IS NOT NULL"
    ):
        c = clean(title)
        if not c or not emb_blob:
            continue
        try:
            n = len(emb_blob) // 4
            if n >= 384:
                offset = (n - 384) * 4
                vec = np.frombuffer(emb_blob[offset:offset+1536], dtype=np.float32).copy()
            else:
                vec = np.frombuffer(emb_blob, dtype=np.float32).copy()
                if len(vec) != 384:
                    continue
            labels.append(c)
            vecs.append(vec)
            count += 1
            if count % 10000 == 0:
                print(f"  Decoded {count:,} embeddings...", flush=True)
        except:
            continue
    dd.close()
    
    if vecs:
        matrix = np.stack(vecs, axis=0)
        np.save(EMB_OUT, matrix)
        with open(IDX_OUT, 'w', encoding='utf-8') as f:
            json.dump(labels, f, ensure_ascii=False)
        print(f"  Step 5 done: {len(labels):,} embeddings ({matrix.shape}) → {EMB_OUT}", flush=True)
    
    # ━━━ 清理 + 统计 ━━━
    cs.execute("DROP TABLE IF EXISTS _dd_import")
    cs.commit()
    
    total_nodes = cs.execute("SELECT COUNT(*) FROM seeds").fetchone()[0]
    with_defs = cs.execute("SELECT COUNT(*) FROM seeds WHERE definition != ''").fetchone()[0]
    cs.close()
    
    elapsed = time.time() - t0
    print(f"\nDone in {elapsed:.0f}s ({elapsed/60:.1f}min)", flush=True)
    print(f"  Nodes: {total_nodes:,} (+{inserted if unmatched > 0 else 0:,})", flush=True)
    print(f"  With definitions: {with_defs:,}", flush=True)
    print(f"  DB size: {os.path.getsize(CS_DB)/1024**3:.2f} GB", flush=True)
    print(f"  Embeddings: {len(labels):,} vectors saved", flush=True)

if __name__ == '__main__':
    main()
