#!/bin/bash
# merge_final.sh — 补全剩余数据（边、释义、别名）
set -e

MAIN="/mnt/d/soso/projects/consciousnessSea/data/consciousness_sea.db"
TMPDIR="/mnt/d/soso/projects/consciousnessSea/data/tmp"

echo "[$(date +%H:%M:%S)] 建索引..."
sqlite3 "$MAIN" "CREATE INDEX IF NOT EXISTS idx_seeds_label ON seeds(label);"

# ── 1. 补边 ─────────────────────────────────────────
for f in \
    "cg_BEFORE-CAUSE-HAS_SUBEVENT-LOCATED_IN-PAR.db" \
    "cg_FOLLOWS-HAS_CAPABILITY-HAS_PROPERTY-MADE.db" \
    "cg_COOCCURS_IN.db" \
; do
    tf="$TMPDIR/$f"
    [ -f "$tf" ] || continue
    echo -n "[$(date +%H:%M:%S)] 边 $f ... "
    sqlite3 "$MAIN" "ATTACH '$tf' AS s; INSERT OR IGNORE INTO karma_edges(source,target,relation,weight,source_tag) SELECT source,target,relation,weight,source_tag FROM s.edges; DETACH s;"
    echo "done"
done

# ── 2. 补节点（从还没写过的 temp） ──────────────────
for f in \
    "cg_BEFORE-CAUSE-HAS_SUBEVENT-LOCATED_IN-PAR.db" \
    "cg_FOLLOWS-HAS_CAPABILITY-HAS_PROPERTY-MADE.db" \
    "cg_SYNONYM.db" \
; do
    tf="$TMPDIR/$f"
    [ -f "$tf" ] || continue
    echo -n "[$(date +%H:%M:%S)] 节点 $f ... "
    sqlite3 "$MAIN" "ATTACH '$tf' AS s; INSERT OR IGNORE INTO seeds(id,label,type,aliases) SELECT DISTINCT label,label,'CONCEPT','[]' FROM s.nodes; DETACH s;"
    echo "done"
done

# ── 3. CC-CEDICT ────────────────────────────────────
tf="$TMPDIR/cedict_defs.db"
if [ -f "$tf" ]; then
    echo -n "[$(date +%H:%M:%S)] CC-CEDICT ... "
    sqlite3 "$MAIN" <<SQL
ATTACH '$tf' AS s;
INSERT OR IGNORE INTO seeds(id,label,type,aliases,definition,pinyin) SELECT label,label,'CONCEPT','[]',definition,pinyin FROM s.defs;
UPDATE seeds SET definition=(SELECT definition FROM s.defs WHERE s.defs.label=seeds.label), pinyin=(SELECT pinyin FROM s.defs WHERE s.defs.label=seeds.label) WHERE EXISTS(SELECT 1 FROM s.defs WHERE s.defs.label=seeds.label);
DETACH s;
SQL
    echo "done"
fi

# ── 4. 成语 ─────────────────────────────────────────
tf="$TMPDIR/idioms.db"
if [ -f "$tf" ]; then
    echo -n "[$(date +%H:%M:%S)] 成语 ... "
    sqlite3 "$MAIN" "ATTACH '$tf' AS s; INSERT OR IGNORE INTO seeds(id,label,type,aliases,domain) SELECT label,label,'CONCEPT','[]',domain FROM s.nodes; DETACH s;"
    echo "done"
fi

# ── 5. Wikipedia ────────────────────────────────────
tf="$TMPDIR/wiki.db"
if [ -f "$tf" ]; then
    echo -n "[$(date +%H:%M:%S)] Wikipedia ... "
    sqlite3 "$MAIN" "ATTACH '$tf' AS s; INSERT OR IGNORE INTO seeds(id,label,type,aliases,activation_bias,meta) SELECT label,label,'CONCEPT','[]',0.05,'{\"source\":\"wikipedia\"}' FROM s.nodes; DETACH s;"
    echo "done"
fi

# ── 6. 别名（事务包装，批量更新） ──────────────────
tf="$TMPDIR/cg_SYNONYM.db"
if [ -f "$tf" ]; then
    echo -n "[$(date +%H:%M:%S)] 别名 ... "
    python3 -c "
import sqlite3, json
src = sqlite3.connect('file:$tf?mode=ro', uri=True)
alias_map = {}
for node, alias in src.execute('SELECT node, alias FROM aliases'):
    alias_map.setdefault(node, []).append(alias)
src.close()
print(f'  read {sum(len(v) for v in alias_map.values())} aliases for {len(alias_map)} nodes')

dst = sqlite3.connect('$MAIN')
dst.execute('PRAGMA synchronous=NORMAL')
dst.execute('BEGIN')
count = 0
for node, aliases in alias_map.items():
    dst.execute('UPDATE seeds SET aliases=? WHERE label=?',
                (json.dumps(aliases, ensure_ascii=False), node))
    count += 1
    if count % 50000 == 0:
        dst.execute('COMMIT')
        dst.execute('BEGIN')
        print(f'  {count}/{len(alias_map)}')
dst.execute('COMMIT')
dst.close()
print(f'  done: {count} nodes updated')
" 2>&1
    echo "done"
fi

# ── 统计 ───────────────────────────────────────────
echo ""
echo "[$(date +%H:%M:%S)] ========== 最终统计 =========="
sqlite3 "$MAIN" <<SQL
.mode column
SELECT '节点' as 类型, COUNT(*) as 数量 FROM seeds
UNION ALL
SELECT '边', COUNT(*) FROM karma_edges
UNION ALL
SELECT '有释义', COUNT(*) FROM seeds WHERE definition != '';
SQL

echo ""
echo "边类型 Top-10:"
sqlite3 "$MAIN" "SELECT relation, COUNT(*) as cnt FROM karma_edges GROUP BY relation ORDER BY cnt DESC LIMIT 10;"

echo ""
echo "DB 大小: $(du -h "$MAIN" | cut -f1)"
echo "[$(date +%H:%M:%S)] 全部完成!"
