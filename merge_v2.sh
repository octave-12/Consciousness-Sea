#!/bin/bash
# merge_v2.sh — 用 SQLite CLI ATTACH+INSERT 合并，比 Python 逐行读快 10 倍+
set -e

MAIN="/mnt/d/soso/projects/consciousnessSea/data/consciousness_sea.db"
TMPDIR="/mnt/d/soso/projects/consciousnessSea/data/tmp"

echo "[$(date +%H:%M:%S)] 创建主库..."

# 删除旧库重建
rm -f "$MAIN"

sqlite3 "$MAIN" <<'SQL'
PRAGMA journal_mode=WAL;
PRAGMA synchronous=NORMAL;
PRAGMA cache_size=-500000;

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
SQL

echo "[$(date +%H:%M:%S)] 主库就绪, 开始合并..."

# 合并每个临时库的边
for f in \
    "cg_COOCCURS_WITH-DEFINED_AS-HAS-IS_A-RELATE.db" \
    "cg_BEFORE-CAUSE-HAS_SUBEVENT-LOCATED_IN-PAR.db" \
    "cg_FOLLOWS-HAS_CAPABILITY-HAS_PROPERTY-MADE.db" \
    "cg_COOCCURS_IN.db" \
    "cg_SYNONYM.db" \
; do
    tf="$TMPDIR/$f"
    if [ ! -f "$tf" ]; then
        echo "  skip $f (not found)"
        continue
    fi
    size=$(du -h "$tf" | cut -f1)
    echo -n "[$(date +%H:%M:%S)] $f ($size) ... "
    
    sqlite3 "$MAIN" <<SQL
ATTACH '$tf' AS src;
INSERT OR IGNORE INTO karma_edges (source,target,relation,weight,source_tag)
    SELECT source,target,relation,weight,source_tag FROM src.edges;
DETACH src;
SQL
    echo "done"
done

# 合并所有节点（从所有 temp 的 nodes 表去重）
echo -n "[$(date +%H:%M:%S)] 合并节点 ... "
for f in \
    "cg_COOCCURS_WITH-DEFINED_AS-HAS-IS_A-RELATE.db" \
    "cg_BEFORE-CAUSE-HAS_SUBEVENT-LOCATED_IN-PAR.db" \
    "cg_FOLLOWS-HAS_CAPABILITY-HAS_PROPERTY-MADE.db" \
    "cg_COOCCURS_IN.db" \
    "cg_SYNONYM.db" \
; do
    tf="$TMPDIR/$f"
    [ -f "$tf" ] || continue
    sqlite3 "$MAIN" <<SQL
ATTACH '$tf' AS src;
INSERT OR IGNORE INTO seeds (id,label,type,aliases)
    SELECT DISTINCT label,label,'CONCEPT','[]' FROM src.nodes;
DETACH src;
SQL
done
echo "done"

# CC-CEDICT
tf="$TMPDIR/cedict_defs.db"
if [ -f "$tf" ]; then
    echo -n "[$(date +%H:%M:%S)] CC-CEDICT ... "
    sqlite3 "$MAIN" <<SQL
ATTACH '$tf' AS src;
INSERT OR IGNORE INTO seeds (id,label,type,aliases,definition,pinyin)
    SELECT label,label,'CONCEPT','[]',definition,pinyin FROM src.defs;
UPDATE seeds SET 
    definition = (SELECT definition FROM src.defs WHERE src.defs.label = seeds.label),
    pinyin = (SELECT pinyin FROM src.defs WHERE src.defs.label = seeds.label)
WHERE EXISTS (SELECT 1 FROM src.defs WHERE src.defs.label = seeds.label);
DETACH src;
SQL
    echo "done"
fi

# 成语
tf="$TMPDIR/idioms.db"
if [ -f "$tf" ]; then
    echo -n "[$(date +%H:%M:%S)] 成语 ... "
    sqlite3 "$MAIN" <<SQL
ATTACH '$tf' AS src;
INSERT OR IGNORE INTO seeds (id,label,type,aliases,domain)
    SELECT label,label,'CONCEPT','[]',domain FROM src.nodes;
DETACH src;
SQL
    echo "done"
fi

# Wikipedia
tf="$TMPDIR/wiki.db"
if [ -f "$tf" ]; then
    echo -n "[$(date +%H:%M:%S)] Wikipedia ... "
    sqlite3 "$MAIN" <<SQL
ATTACH '$tf' AS src;
INSERT OR IGNORE INTO seeds (id,label,type,aliases,activation_bias,meta)
    SELECT label,label,'CONCEPT','[]',0.05,'{"source":"wikipedia"}' FROM src.nodes;
DETACH src;
SQL
    echo "done"
fi

# SYNONYM aliases
tf="$TMPDIR/cg_SYNONYM.db"
if [ -f "$tf" ]; then
    echo -n "[$(date +%H:%M:%S)] 别名 ... "
    # 读取别名并构建 JSON，这个必须用 Python
    python3 -c "
import sqlite3, json
src = sqlite3.connect('file:$tf?mode=ro', uri=True)
alias_map = {}
for node, alias in src.execute('SELECT node, alias FROM aliases'):
    alias_map.setdefault(node, []).append(alias)
src.close()

dst = sqlite3.connect('$MAIN')
for node, aliases in alias_map.items():
    dst.execute('UPDATE seeds SET aliases=? WHERE label=?', 
                (json.dumps(aliases, ensure_ascii=False), node))
dst.commit()
dst.close()
print(f'  {len(alias_map)} nodes with aliases')
" 2>&1
    echo "done"
fi

# 统计
echo ""
echo "[$(date +%H:%M:%S)] ========== 统计 =========="
sqlite3 "$MAIN" <<SQL
.mode column
SELECT '节点' as 类型, COUNT(*) as 数量 FROM seeds
UNION ALL
SELECT '边', COUNT(*) FROM karma_edges;
SQL

echo ""
echo "边类型分布:"
sqlite3 "$MAIN" <<SQL
.mode column
SELECT relation, COUNT(*) as cnt FROM karma_edges GROUP BY relation ORDER BY cnt DESC;
SQL

size=$(du -h "$MAIN" | cut -f1)
echo ""
echo "DB 大小: $size"
echo "[$(date +%H:%M:%S)] 完成!"
