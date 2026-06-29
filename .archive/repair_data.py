#!/usr/bin/env python3
"""
识海数据修复脚本 (Data Repair Script)
═══════════════════════════════════════

修复识海数据库中三个严重数据问题:
  1. RELATED 边仅 720 条（预期 85.9 万）    → Phase 1: 补齐
  2. IS_A 边 523 万条（预期 70 万）           → Phase 2: 裁剪
  3. 节点 domain 字段几乎全为空                → Phase 3: 推断
  4. 修复前后对比统计                          → Phase 4: 报告

用法:
  python repair_data.py --db data/consciousness_sea.db --cg /path/to/concept_graph.db
  python repair_data.py --phase 1       # 只跑 RELATED 补齐
  python repair_data.py --phase 3       # 只跑领域推断
  python repair_data.py --dry-run       # 预览模式
  python repair_data.py --data-dir /path # 指定数据目录
"""

from __future__ import annotations

import argparse
import json
import logging
import sqlite3
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

from core.config import (
    CG_DB_FILENAME,
    CEDICT_FILENAME,
    DEFAULT_DB_PATH,
    DEFAULT_DATA_DIR,
    DOMAIN_COVERAGE_TARGET,
    ISA_MAX_COUNT,
    ISA_PRUNE_THRESHOLD,
    RELATED_MIN_CONFIDENCE,
    REPAIR_BATCH_SIZE,
    ZHWIKI_FILENAME,
    resolve_data_dir,
)
from core.domain_inference import DomainInferenceReport, infer_domains

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
log = logging.getLogger("repair_data")


# ═══════════════════════════════════════════════════════════════
#  RepairProgress — 断点续传状态管理
# ═══════════════════════════════════════════════════════════════


class RepairProgress:
    """
    管理修复脚本的断点续传状态。

    状态保存到 data/repair_progress.json，记录每个 phase 的完成状态。
    中断后可从上次进度继续。
    """

    PHASES = (1, 2, 3, 4)

    def __init__(self, progress_file: Path) -> None:
        self._file = progress_file
        self._state: dict = self._load()

    # ── 持久化 ──────────────────────────────────────────────

    def _load(self) -> dict:
        """从磁盘加载进度文件"""
        if self._file.exists():
            try:
                with open(self._file, "r", encoding="utf-8") as f:
                    data = json.load(f)
                log.info("断点续传: 已加载进度文件 %s", self._file)
                return data
            except (json.JSONDecodeError, OSError) as exc:
                log.warning("进度文件损坏，从头开始: %s", exc)
        return self._default_state()

    @staticmethod
    def _default_state() -> dict:
        """默认进度状态"""
        return {
            "phase_1_completed": False,
            "phase_1_offset": 0,
            "phase_2_completed": False,
            "phase_2_offset": 0,
            "phase_3_completed": False,
            "phase_3_offset": 0,
            "phase_4_completed": False,
        }

    def save(self) -> None:
        """将当前状态持久化到磁盘"""
        self._file.parent.mkdir(parents=True, exist_ok=True)
        with open(self._file, "w", encoding="utf-8") as f:
            json.dump(self._state, f, indent=2, ensure_ascii=False)

    def cleanup(self) -> None:
        """全部完成后删除进度文件"""
        if self._file.exists():
            try:
                self._file.unlink()
                log.info("已清理进度文件: %s", self._file)
            except OSError:
                pass

    # ── Phase 状态读写 ─────────────────────────────────────

    def is_phase_completed(self, phase: int) -> bool:
        """检查某个 phase 是否已完成"""
        return self._state.get(f"phase_{phase}_completed", False)

    def mark_phase_completed(self, phase: int) -> None:
        """标记某个 phase 已完成"""
        self._state[f"phase_{phase}_completed"] = True
        self.save()
        log.info("Phase %d 已完成，进度已保存", phase)

    def get_offset(self, phase: int) -> int:
        """获取某个 phase 的处理偏移量"""
        return self._state.get(f"phase_{phase}_offset", 0)

    def set_offset(self, phase: int, offset: int) -> None:
        """设置某个 phase 的处理偏移量"""
        self._state[f"phase_{phase}_offset"] = offset
        self.save()


# ═══════════════════════════════════════════════════════════════
#  RepairReport — 修复报告数据类
# ═══════════════════════════════════════════════════════════════


@dataclass
class RepairReport:
    """修复报告"""

    started_at: str                          # 开始时间
    finished_at: str                         # 结束时间
    elapsed_s: float                         # 总耗时（秒）

    # Phase 1: RELATED 边补齐
    related_imported: int = 0                # 导入的 RELATED 边数量
    related_skipped: int = 0                 # 跳过的低置信度边数量

    # Phase 2: IS_A 边裁剪
    isa_before: int = 0                      # 裁剪前 IS_A 边数量
    isa_after: int = 0                       # 裁剪后 IS_A 边数量
    isa_pruned: int = 0                      # 被裁剪的 IS_A 边数量
    orphan_seeds: int = 0                    # 孤立种子数量

    # Phase 3: 领域推断
    domain_before_empty: int = 0             # 修复前 domain 为空的种子数
    domain_after_empty: int = 0              # 修复后 domain 为空的种子数
    domain_coverage_rate: float = 0.0        # 领域覆盖率
    domain_inference_report: dict | None = None  # DomainInferenceReport 转为 dict

    # Phase 4: 统计
    total_nodes: int = 0                     # 节点总数
    total_edges: int = 0                     # 边总数
    relation_distribution: dict = field(default_factory=dict)   # 关系类型分布
    domain_distribution: dict = field(default_factory=dict)     # 领域分布

    # 修复前后对比快照
    snapshot_before: dict = field(default_factory=dict)   # 修复前快照
    snapshot_after: dict = field(default_factory=dict)    # 修复后快照

    def to_dict(self) -> dict:
        """转换为字典（用于 JSON 序列化）"""
        return {
            "started_at": self.started_at,
            "finished_at": self.finished_at,
            "elapsed_s": round(self.elapsed_s, 2),
            "phase_1": {
                "related_imported": self.related_imported,
                "related_skipped": self.related_skipped,
            },
            "phase_2": {
                "isa_before": self.isa_before,
                "isa_after": self.isa_after,
                "isa_pruned": self.isa_pruned,
                "orphan_seeds": self.orphan_seeds,
            },
            "phase_3": {
                "domain_before_empty": self.domain_before_empty,
                "domain_after_empty": self.domain_after_empty,
                "domain_coverage_rate": round(self.domain_coverage_rate, 4),
                "domain_inference_report": self.domain_inference_report,
            },
            "phase_4": {
                "total_nodes": self.total_nodes,
                "total_edges": self.total_edges,
                "relation_distribution": self.relation_distribution,
                "domain_distribution": self.domain_distribution,
            },
            "snapshot_before": self.snapshot_before,
            "snapshot_after": self.snapshot_after,
        }


# ═══════════════════════════════════════════════════════════════
#  辅助函数
# ═══════════════════════════════════════════════════════════════


def _normalize_related_weight(confidence: float) -> float:
    """
    RELATED 边权重归一化。

    c >= 0.9 → 0.7
    c >= 0.7 → 0.5
    c >= 0.5 → 0.3
    c < 0.5  → 0.0 (丢弃)
    """
    if confidence >= 0.9:
        return 0.7
    if confidence >= 0.7:
        return 0.5
    if confidence >= 0.5:
        return 0.3
    return 0.0


def _apply_sqlite_pragma(conn: sqlite3.Connection) -> None:
    """对连接应用 SQLite PRAGMA 优化"""
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA synchronous=NORMAL")
    conn.execute("PRAGMA cache_size=-2000000")  # 2 GB 缓存
    conn.execute("PRAGMA temp_store=MEMORY")


def _take_snapshot(conn: sqlite3.Connection) -> dict:
    """采集数据库当前状态的快照"""
    total_nodes = conn.execute("SELECT COUNT(*) FROM seeds").fetchone()[0]
    total_edges = conn.execute("SELECT COUNT(*) FROM karma_edges").fetchone()[0]

    relation_dist = {}
    for rel, cnt in conn.execute(
        "SELECT relation, COUNT(*) FROM karma_edges GROUP BY relation ORDER BY COUNT(*) DESC"
    ).fetchall():
        relation_dist[rel] = cnt

    domain_dist = {}
    for domain, cnt in conn.execute(
        "SELECT domain, COUNT(*) FROM seeds GROUP BY domain ORDER BY COUNT(*) DESC"
    ).fetchall():
        domain_dist[domain if domain else "(空)"] = cnt

    related_count = conn.execute(
        "SELECT COUNT(*) FROM karma_edges WHERE relation = 'RELATED'"
    ).fetchone()[0]

    isa_count = conn.execute(
        "SELECT COUNT(*) FROM karma_edges WHERE relation = 'IS_A'"
    ).fetchone()[0]

    empty_domain = conn.execute(
        "SELECT COUNT(*) FROM seeds WHERE domain = '' OR domain IS NULL"
    ).fetchone()[0]

    return {
        "total_nodes": total_nodes,
        "total_edges": total_edges,
        "related_edges": related_count,
        "isa_edges": isa_count,
        "empty_domain_seeds": empty_domain,
        "relation_distribution": relation_dist,
        "domain_distribution": domain_dist,
    }


# ═══════════════════════════════════════════════════════════════
#  Phase 1: RELATED 边补齐
# ═══════════════════════════════════════════════════════════════


def phase1_repair_related(
    conn: sqlite3.Connection,
    cg_db_path: Path,
    progress: RepairProgress,
    dry_run: bool = False,
    batch_size: int = REPAIR_BATCH_SIZE,
) -> tuple[int, int]:
    """
    Phase 1: 从龙珠概念图补齐 RELATED 边。

    流程:
      1. 连接 concept_graph.db
      2. 提取 r='RELATED' AND c >= 0.5 的三元组
      3. 权重归一化: c≥0.9→0.7, c≥0.7→0.5, c≥0.5→0.3
      4. 去重: INSERT OR IGNORE (source, target, relation 主键)
      5. 断点续传: 记录已处理的 offset

    Args:
        conn: 识海数据库连接
        cg_db_path: 龙珠概念图数据库路径
        progress: 断点续传状态
        dry_run: 预览模式
        batch_size: 批量提交大小

    Returns:
        (imported_count, skipped_count) 导入数量和跳过数量
    """
    log.info("=" * 60)
    log.info("Phase 1: RELATED 边补齐")
    log.info("=" * 60)

    # 检查龙珠概念图是否可访问
    if not cg_db_path.exists():
        log.error("龙珠概念图数据库不可访问: %s", cg_db_path)
        log.error("Phase 1 终止，不修改识海数据库")
        raise SystemExit(1)

    # 连接龙珠概念图（只读）
    try:
        cg_conn = sqlite3.connect(f"file:{cg_db_path}?mode=ro", uri=True)
        # 验证 triples 表存在
        table_check = cg_conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='triples'"
        ).fetchone()
        if table_check is None:
            log.error("龙珠概念图中不存在 triples 表")
            cg_conn.close()
            raise SystemExit(1)
    except sqlite3.Error as exc:
        log.error("龙珠概念图数据库连接失败: %s", exc)
        log.error("Phase 1 终止，不修改识海数据库")
        raise SystemExit(1)

    # 统计源数据量
    total_candidates = cg_conn.execute(
        "SELECT COUNT(*) FROM triples WHERE r = 'RELATED' AND c >= ?",
        (RELATED_MIN_CONFIDENCE,),
    ).fetchone()[0]
    log.info("龙珠概念图中 RELATED (c >= %.1f) 候选边: %d", RELATED_MIN_CONFIDENCE, total_candidates)

    if dry_run:
        log.info("[dry-run] 预计导入约 %d 条 RELATED 边", total_candidates)
        cg_conn.close()
        return (0, 0)

    # 断点续传: 获取已处理偏移量
    offset = progress.get_offset(1)
    if offset > 0:
        log.info("断点续传: 从 offset=%d 继续", offset)

    # 预加载已有 RELATED 边（用于快速去重判断）
    existing_edges: set[tuple[str, str]] = set()
    for s, t in conn.execute(
        "SELECT source, target FROM karma_edges WHERE relation = 'RELATED'"
    ).fetchall():
        existing_edges.add((s, t))
    log.info("识海数据库中已有 RELATED 边: %d", len(existing_edges))

    # 预加载已有种子标签（用于判断是否需要创建新节点）
    existing_labels: set[str] = set()
    for (label,) in conn.execute("SELECT label FROM seeds").fetchall():
        existing_labels.add(label)

    # 分批读取龙珠概念图数据
    imported_count = 0
    skipped_count = 0
    batch_edges: list[tuple[str, str, str, float, str]] = []
    batch_seeds: list[tuple[str, str, str, str, str, str, str, float, str]] = []

    cursor = cg_conn.execute(
        "SELECT s, o, c FROM triples WHERE r = 'RELATED' AND c >= ? "
        "ORDER BY c DESC LIMIT ? OFFSET ?",
        (RELATED_MIN_CONFIDENCE, total_candidates, offset),
    )

    rows_processed = 0

    for source, target, confidence in cursor:
        rows_processed += 1

        # 权重归一化
        weight = _normalize_related_weight(confidence)
        if weight == 0.0:
            skipped_count += 1
            continue

        # 去重检查
        edge_key = (source, target)
        if edge_key in existing_edges:
            skipped_count += 1
            continue

        existing_edges.add(edge_key)

        # 添加边
        batch_edges.append((source, target, "RELATED", weight, "loong_cg_repair"))

        # 添加缺失节点
        if source not in existing_labels:
            existing_labels.add(source)
            batch_seeds.append(
                (source, source, "CONCEPT", "[]", "", "", "", 0.0, "{}")
            )
        if target not in existing_labels:
            existing_labels.add(target)
            batch_seeds.append(
                (target, target, "CONCEPT", "[]", "", "", "", 0.0, "{}")
            )

        # 批量提交
        if len(batch_edges) >= batch_size:
            conn.executemany(
                "INSERT OR IGNORE INTO seeds "
                "(id, label, type, aliases, domain, definition, pinyin, activation_bias, meta) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                batch_seeds,
            )
            conn.executemany(
                "INSERT OR IGNORE INTO karma_edges "
                "(source, target, relation, weight, source_tag) "
                "VALUES (?, ?, ?, ?, ?)",
                batch_edges,
            )
            conn.commit()
            imported_count += len(batch_edges)
            log.info(
                "Phase 1 进度: %d / %d (导入 %d, 跳过 %d)",
                rows_processed + offset,
                total_candidates + offset,
                imported_count,
                skipped_count,
            )

            # 保存断点
            progress.set_offset(1, offset + rows_processed)

            batch_edges = []
            batch_seeds = []

    # 冲刷剩余批次
    if batch_seeds:
        conn.executemany(
            "INSERT OR IGNORE INTO seeds "
            "(id, label, type, aliases, domain, definition, pinyin, activation_bias, meta) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
            batch_seeds,
        )
    if batch_edges:
        conn.executemany(
            "INSERT OR IGNORE INTO karma_edges "
            "(source, target, relation, weight, source_tag) "
            "VALUES (?, ?, ?, ?, ?)",
            batch_edges,
        )
        imported_count += len(batch_edges)
    conn.commit()

    cg_conn.close()

    log.info(
        "Phase 1 完成: 导入 %d 条 RELATED 边, 跳过 %d 条",
        imported_count,
        skipped_count,
    )

    progress.mark_phase_completed(1)
    return (imported_count, skipped_count)


# ═══════════════════════════════════════════════════════════════
#  Phase 2: IS_A 边裁剪
# ═══════════════════════════════════════════════════════════════


def phase2_prune_isa(
    conn: sqlite3.Connection,
    data_dir: Path,
    progress: RepairProgress,
    dry_run: bool = False,
    batch_size: int = REPAIR_BATCH_SIZE,
) -> tuple[int, int, int, int]:
    """
    Phase 2: 裁剪低质量 IS_A 边。

    流程:
      1. 备份: 将 weight < 0.3 的 IS_A 边写入 data/pruned_is_a_edges.jsonl
      2. 删除: DELETE FROM karma_edges WHERE relation='IS_A' AND weight < 0.3
      3. 数量约束: 若裁剪后仍 > 150 万条，按权重升序继续删除
      4. 孤立节点检测: 统计无 IS_A 入边和出边的种子数

    Args:
        conn: 识海数据库连接
        data_dir: 数据目录
        progress: 断点续传状态
        dry_run: 预览模式
        batch_size: 批量提交大小

    Returns:
        (isa_before, isa_after, pruned_count, orphan_seeds)
    """
    log.info("=" * 60)
    log.info("Phase 2: IS_A 边裁剪")
    log.info("=" * 60)

    # 裁剪前 IS_A 边数量
    isa_before = conn.execute(
        "SELECT COUNT(*) FROM karma_edges WHERE relation = 'IS_A'"
    ).fetchone()[0]
    log.info("裁剪前 IS_A 边数量: %d", isa_before)

    if dry_run:
        low_weight_count = conn.execute(
            "SELECT COUNT(*) FROM karma_edges WHERE relation = 'IS_A' AND weight < ?",
            (ISA_PRUNE_THRESHOLD,),
        ).fetchone()[0]
        log.info("[dry-run] 预计裁剪 weight < %.1f 的 IS_A 边: %d", ISA_PRUNE_THRESHOLD, low_weight_count)
        remaining = isa_before - low_weight_count
        if remaining > ISA_MAX_COUNT:
            extra = remaining - ISA_MAX_COUNT
            log.info("[dry-run] 裁剪后仍 > %d，需额外删除 %d 条", ISA_MAX_COUNT, extra)
        return (isa_before, isa_before, 0, 0)

    # ── 步骤 1: 备份低权重 IS_A 边 ──────────────────────────
    pruned_file = data_dir / "pruned_is_a_edges.jsonl"
    pruned_file.parent.mkdir(parents=True, exist_ok=True)

    log.info("备份低权重 IS_A 边到 %s ...", pruned_file)

    # 查询需要裁剪的边
    low_weight_cursor = conn.execute(
        "SELECT source, target, weight, source_tag "
        "FROM karma_edges WHERE relation = 'IS_A' AND weight < ?",
        (ISA_PRUNE_THRESHOLD,),
    )

    pruned_count = 0
    with open(pruned_file, "w", encoding="utf-8") as f:
        for source, target, weight, source_tag in low_weight_cursor:
            record = {
                "source": source,
                "target": target,
                "relation": "IS_A",
                "weight": weight,
                "source_tag": source_tag or "",
            }
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
            pruned_count += 1

    log.info("备份完成: %d 条 IS_A 边写入 %s", pruned_count, pruned_file)

    # 备份文件必须写入完成后才能执行删除
    # 验证备份文件
    if pruned_count > 0:
        if not pruned_file.exists() or pruned_file.stat().st_size == 0:
            log.error("备份文件写入失败，终止裁剪操作")
            return (isa_before, isa_before, 0, 0)

    # ── 步骤 2: 删除低权重 IS_A 边 ──────────────────────────
    log.info("删除 weight < %.1f 的 IS_A 边 ...", ISA_PRUNE_THRESHOLD)
    conn.execute(
        "DELETE FROM karma_edges WHERE relation = 'IS_A' AND weight < ?",
        (ISA_PRUNE_THRESHOLD,),
    )
    conn.commit()

    isa_after = conn.execute(
        "SELECT COUNT(*) FROM karma_edges WHERE relation = 'IS_A'"
    ).fetchone()[0]
    log.info("删除后 IS_A 边数量: %d (裁剪 %d 条)", isa_after, pruned_count)

    # ── 步骤 3: 数量约束 — 若仍 > ISA_MAX_COUNT 则继续裁剪 ──
    if isa_after > ISA_MAX_COUNT:
        extra_to_prune = isa_after - ISA_MAX_COUNT
        log.info(
            "IS_A 边仍 > %d 条，需额外裁剪 %d 条（按权重升序）",
            ISA_MAX_COUNT,
            extra_to_prune,
        )

        # 备份额外裁剪的边
        extra_cursor = conn.execute(
            "SELECT source, target, weight, source_tag "
            "FROM karma_edges WHERE relation = 'IS_A' "
            "ORDER BY weight ASC LIMIT ?",
            (extra_to_prune,),
        )

        with open(pruned_file, "a", encoding="utf-8") as f:
            for source, target, weight, source_tag in extra_cursor:
                record = {
                    "source": source,
                    "target": target,
                    "relation": "IS_A",
                    "weight": weight,
                    "source_tag": source_tag or "",
                    "extra_prune": True,
                }
                f.write(json.dumps(record, ensure_ascii=False) + "\n")

        # 删除额外裁剪的边
        # 使用子查询获取要删除的边的 rowid
        conn.execute(
            "DELETE FROM karma_edges WHERE rowid IN ("
            "  SELECT rowid FROM karma_edges "
            "  WHERE relation = 'IS_A' "
            "  ORDER BY weight ASC LIMIT ?"
            ")",
            (extra_to_prune,),
        )
        conn.commit()
        pruned_count += extra_to_prune

        isa_after = conn.execute(
            "SELECT COUNT(*) FROM karma_edges WHERE relation = 'IS_A'"
        ).fetchone()[0]
        log.info("额外裁剪后 IS_A 边数量: %d", isa_after)

    # ── 步骤 4: 孤立节点检测 ────────────────────────────────
    # 统计无 IS_A 入边和出边的种子数
    orphan_seeds = conn.execute(
        "SELECT COUNT(*) FROM seeds s "
        "WHERE NOT EXISTS ("
        "  SELECT 1 FROM karma_edges e "
        "  WHERE e.relation = 'IS_A' AND (e.source = s.label OR e.target = s.label)"
        ")"
    ).fetchone()[0]
    log.info("孤立种子数量（无 IS_A 入边和出边）: %d", orphan_seeds)

    log.info(
        "Phase 2 完成: IS_A 边 %d → %d (裁剪 %d), 孤立种子 %d",
        isa_before,
        isa_after,
        pruned_count,
        orphan_seeds,
    )

    progress.mark_phase_completed(2)
    return (isa_before, isa_after, pruned_count, orphan_seeds)


# ═══════════════════════════════════════════════════════════════
#  Phase 3: 领域推断
# ═══════════════════════════════════════════════════════════════


def phase3_infer_domains(
    conn: sqlite3.Connection,
    db_path: Path,
    data_dir: Path,
    progress: RepairProgress,
    dry_run: bool = False,
) -> tuple[int, int, float, dict | None]:
    """
    Phase 3: 领域推断。

    流程:
      1. 调用 domain_inference.infer_domains()
      2. 验证 domain 非空率 >= 95%
      3. 输出推断报告

    Args:
        conn: 识海数据库连接
        db_path: 识海数据库文件路径
        data_dir: 数据目录
        progress: 断点续传状态
        dry_run: 预览模式

    Returns:
        (before_empty, after_empty, coverage_rate, inference_report_dict)
    """
    log.info("=" * 60)
    log.info("Phase 3: 领域推断")
    log.info("=" * 60)

    # 修复前 domain 为空的种子数
    before_empty = conn.execute(
        "SELECT COUNT(*) FROM seeds WHERE domain = '' OR domain IS NULL"
    ).fetchone()[0]
    total_seeds = conn.execute("SELECT COUNT(*) FROM seeds").fetchone()[0]
    log.info(
        "修复前: domain 为空的种子 %d / %d (%.2f%%)",
        before_empty,
        total_seeds,
        (1 - before_empty / total_seeds) * 100 if total_seeds > 0 else 0,
    )

    if dry_run:
        log.info("[dry-run] 预计推断 %d 个种子的领域", before_empty)
        return (before_empty, before_empty, 0.0, None)

    # 构建外部数据源路径
    cedict_path = data_dir / CEDICT_FILENAME
    wiki_db_path = data_dir / ZHWIKI_FILENAME

    # 进度文件
    domain_progress_file = data_dir / "domain_inference_progress.json"

    # 调用领域推断引擎
    report: DomainInferenceReport = infer_domains(
        db_path=db_path,
        cedict_path=cedict_path if cedict_path.exists() else None,
        wiki_db_path=wiki_db_path if wiki_db_path.exists() else None,
        batch_size=REPAIR_BATCH_SIZE,
        progress_file=domain_progress_file,
    )

    # 修复后统计
    after_empty = conn.execute(

        "SELECT COUNT(*) FROM seeds WHERE domain = '' OR domain IS NULL"
    ).fetchone()[0]
    coverage_rate = report.coverage_rate

    log.info("领域推断报告:")
    log.info("  总空域种子: %d", report.total_empty)
    log.info("  成功推断: %d", report.inferred)
    log.info("  兜底(常识): %d", report.fallback_common)
    log.info("  循环检测: %d", report.cycles_detected)
    log.info("  覆盖率: %.2f%%", coverage_rate * 100)
    log.info("  耗时: %.1fs", report.elapsed_s)
    log.info("  按方法: %s", report.by_method)

    if coverage_rate < DOMAIN_COVERAGE_TARGET:
        log.warning(
            "领域覆盖率 %.2f%% 低于目标 %.2f%%",
            coverage_rate * 100,
            DOMAIN_COVERAGE_TARGET * 100,
        )

    progress.mark_phase_completed(3)
    return (before_empty, after_empty, coverage_rate, report.to_dict())


# ═══════════════════════════════════════════════════════════════
#  Phase 4: 统计报告
# ═══════════════════════════════════════════════════════════════


def phase4_generate_report(
    conn: sqlite3.Connection,
    data_dir: Path,
    report: RepairReport,
    progress: RepairProgress,
    dry_run: bool = False,
) -> None:
    """
    Phase 4: 生成统计报告。

    包含:
      - 节点/边总数
      - 关系类型分布
      - 领域分布
      - 修复前后对比
      - 输出 data/repair_report.json

    Args:
        conn: 识海数据库连接
        data_dir: 数据目录
        report: 修复报告数据
        progress: 断点续传状态
        dry_run: 预览模式
    """
    log.info("=" * 60)
    log.info("Phase 4: 统计报告")
    log.info("=" * 60)

    # 采集修复后快照
    snapshot_after = _take_snapshot(conn)

    # 填充报告
    report.total_nodes = snapshot_after["total_nodes"]
    report.total_edges = snapshot_after["total_edges"]
    report.relation_distribution = snapshot_after["relation_distribution"]
    report.domain_distribution = snapshot_after["domain_distribution"]
    report.snapshot_after = snapshot_after

    # 时间信息
    report.finished_at = time.strftime("%Y-%m-%d %H:%M:%S")

    # 输出报告
    report_path = data_dir / "repair_report.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)

    if not dry_run:
        with open(report_path, "w", encoding="utf-8") as f:
            json.dump(report.to_dict(), f, indent=2, ensure_ascii=False)
        log.info("修复报告已输出到: %s", report_path)
    else:
        log.info("[dry-run] 修复报告预览:")
        log.info(json.dumps(report.to_dict(), indent=2, ensure_ascii=False))

    # 打印汇总
    log.info("=" * 60)
    log.info("修复汇总")
    log.info("=" * 60)
    log.info("节点总数: %d", report.total_nodes)
    log.info("边总数:   %d", report.total_edges)
    log.info("关系类型分布:")
    for rel, cnt in sorted(report.relation_distribution.items(), key=lambda x: -x[1]):
        log.info("  %s: %d", rel, cnt)
    log.info("领域分布 (Top 10):")
    sorted_domains = sorted(report.domain_distribution.items(), key=lambda x: -x[1])
    for domain, cnt in sorted_domains[:10]:
        log.info("  %s: %d", domain, cnt)
    log.info("Phase 1 — RELATED 导入: %d", report.related_imported)
    log.info("Phase 2 — IS_A 裁剪: %d → %d (删除 %d)", report.isa_before, report.isa_after, report.isa_pruned)
    log.info("Phase 2 — 孤立种子: %d", report.orphan_seeds)
    log.info("Phase 3 — 领域覆盖率: %.2f%%", report.domain_coverage_rate * 100)
    log.info("总耗时: %.1fs", report.elapsed_s)

    progress.mark_phase_completed(4)


# ═══════════════════════════════════════════════════════════════
#  主修复流程
# ═══════════════════════════════════════════════════════════════


def run_repair(
    db_path: Path,
    cg_db_path: Path | None = None,
    data_dir: Path | None = None,
    phase: int | None = None,
    dry_run: bool = False,
) -> RepairReport:
    """
    执行数据修复流程。

    严格按 Phase 1 → 2 → 3 → 4 顺序执行。
    支持 --phase 参数只跑某个阶段。

    Args:
        db_path: 识海数据库路径
        cg_db_path: 龙珠概念图数据库路径（Phase 1 需要）
        data_dir: 数据目录
        phase: 只执行指定阶段 (1-4)，None 表示全量执行
        dry_run: 预览模式

    Returns:
        RepairReport 修复报告
    """
    start_time = time.monotonic()
    started_at = time.strftime("%Y-%m-%d %H:%M:%S")

    # 默认数据目录
    if data_dir is None:
        data_dir = db_path.parent

    # 进度文件
    progress_file = data_dir / "repair_progress.json"
    progress = RepairProgress(progress_file)

    # 连接识海数据库
    conn = sqlite3.connect(str(db_path))
    _apply_sqlite_pragma(conn)

    # 采集修复前快照
    snapshot_before = _take_snapshot(conn)
    log.info("修复前快照: %d 节点, %d 边, RELATED %d, IS_A %d, 空域 %d",
             snapshot_before["total_nodes"],
             snapshot_before["total_edges"],
             snapshot_before["related_edges"],
             snapshot_before["isa_edges"],
             snapshot_before["empty_domain_seeds"])

    # 初始化报告
    report = RepairReport(
        started_at=started_at,
        finished_at="",
        elapsed_s=0.0,
        snapshot_before=snapshot_before,
    )

    try:
        # ── Phase 1: RELATED 边补齐 ──────────────────────────
        if phase is None or phase == 1:
            if phase == 1 or not progress.is_phase_completed(1):
                if cg_db_path is None:
                    log.error("Phase 1 需要 --cg 参数指定龙珠概念图路径")
                    raise SystemExit(1)

                imported, skipped = phase1_repair_related(
                    conn=conn,
                    cg_db_path=cg_db_path,
                    progress=progress,
                    dry_run=dry_run,
                )
                report.related_imported = imported
                report.related_skipped = skipped
            else:
                log.info("Phase 1 已完成，跳过")

        # ── Phase 2: IS_A 边裁剪 ────────────────────────────
        if phase is None or phase == 2:
            if phase == 2 or not progress.is_phase_completed(2):
                isa_before, isa_after, pruned, orphans = phase2_prune_isa(
                    conn=conn,
                    data_dir=data_dir,
                    progress=progress,
                    dry_run=dry_run,
                )
                report.isa_before = isa_before
                report.isa_after = isa_after
                report.isa_pruned = pruned
                report.orphan_seeds = orphans
            else:
                log.info("Phase 2 已完成，跳过")

        # ── Phase 3: 领域推断 ───────────────────────────────
        if phase is None or phase == 3:
            if phase == 3 or not progress.is_phase_completed(3):
                before_empty, after_empty, coverage, inference_dict = phase3_infer_domains(
                    conn=conn,
                    db_path=db_path,
                    data_dir=data_dir,
                    progress=progress,
                    dry_run=dry_run,
                )
                report.domain_before_empty = before_empty
                report.domain_after_empty = after_empty
                report.domain_coverage_rate = coverage
                report.domain_inference_report = inference_dict
            else:
                log.info("Phase 3 已完成，跳过")

        # ── Phase 4: 统计报告 ───────────────────────────────
        if phase is None or phase == 4:
            phase4_generate_report(
                conn=conn,
                data_dir=data_dir,
                report=report,
                progress=progress,
                dry_run=dry_run,
            )

        # 计算总耗时
        report.elapsed_s = time.monotonic() - start_time

        # 全部完成后清理进度文件
        if phase is None:
            progress.cleanup()

    finally:
        conn.close()

    return report


# ═══════════════════════════════════════════════════════════════
#  CLI 接口
# ═══════════════════════════════════════════════════════════════


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """解析命令行参数"""
    parser = argparse.ArgumentParser(
        description="识海数据修复脚本",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  python repair_data.py --db data/consciousness_sea.db --cg /path/to/concept_graph.db
  python repair_data.py --phase 1       # 只跑 RELATED 补齐
  python repair_data.py --phase 3       # 只跑领域推断
  python repair_data.py --dry-run       # 预览模式
  python repair_data.py --data-dir /path # 指定数据目录
        """,
    )
    parser.add_argument(
        "--db",
        type=str,
        default=None,
        help="识海数据库路径 (默认: <data-dir>/consciousness_sea.db)",
    )
    parser.add_argument(
        "--cg",
        type=str,
        default=None,
        help="龙珠概念图数据库路径 (默认: <data-dir>/concept_graph.db)",
    )
    parser.add_argument(
        "--phase",
        type=int,
        choices=[1, 2, 3, 4],
        default=None,
        help="只执行指定阶段 (1=RELATED补齐, 2=IS_A裁剪, 3=领域推断, 4=统计报告)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        default=False,
        help="预览模式，不实际修改数据库",
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default=None,
        help="数据目录路径 (优先级: 命令行 > 环境变量 > 脚本相对 data/)",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    """主入口"""
    args = parse_args(argv)

    # 解析数据目录
    data_dir = resolve_data_dir(args.data_dir)

    # 解析数据库路径
    if args.db:
        db_path = Path(args.db).resolve()
    else:
        db_path = data_dir / "consciousness_sea.db"

    # 解析龙珠概念图路径
    if args.cg:
        cg_db_path = Path(args.cg).resolve()
    else:
        cg_db_path = data_dir / CG_DB_FILENAME

    # 验证数据库文件存在
    if not db_path.exists():
        log.error("识海数据库不存在: %s", db_path)
        log.error("请先运行 import_knowledge_base.py 创建识海数据库。")
        sys.exit(1)

    # Phase 1 需要龙珠概念图
    if (args.phase is None or args.phase == 1) and not cg_db_path.exists():
        log.error("龙珠概念图数据库不存在: %s", cg_db_path)
        log.error("Phase 1 需要龙珠概念图，请使用 --cg 参数指定路径。")
        sys.exit(1)

    log.info("识海数据库: %s", db_path)
    log.info("龙珠概念图: %s", cg_db_path)
    log.info("数据目录:   %s", data_dir)
    if args.phase:
        log.info("执行阶段:   Phase %d", args.phase)
    else:
        log.info("执行阶段:   全量 (Phase 1 → 2 → 3 → 4)")
    if args.dry_run:
        log.info("模式:       预览 (dry-run)")

    # 执行修复
    report = run_repair(
        db_path=db_path,
        cg_db_path=cg_db_path,
        data_dir=data_dir,
        phase=args.phase,
        dry_run=args.dry_run,
    )

    log.info("修复完成，总耗时 %.1fs", report.elapsed_s)


if __name__ == "__main__":
    main()