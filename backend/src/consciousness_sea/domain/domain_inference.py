"""
领域推断引擎 (Domain Inference Engine)

为识海数据库中 domain 为空的种子推断领域标签。
覆盖 REQ-P0-001 / REQ-P0-002 / REQ-P0-003。

推断优先级:
  1. IS_A 层级上溯 (BFS) → 继承祖先 domain
  2. CC-CEDICT 释义关键词映射
  3. Wikipedia 分类关键词映射
  4. 兜底 → "常识"
"""

from __future__ import annotations

import json
import logging
import sqlite3
import time
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path

from consciousness_sea.infrastructure.config import (
    DEFAULT_DB_PATH,
    REPAIR_BATCH_SIZE,
    DOMAIN_COVERAGE_TARGET,
)

log = logging.getLogger(__name__)

# ═══════════════════════════════════════════════════════════════
#  BFS 上溯最大深度限制
# ═══════════════════════════════════════════════════════════════
BFS_MAX_DEPTH = 20

# ═══════════════════════════════════════════════════════════════
#  CC-CEDICT 释义英文关键词 → 领域映射
# ═══════════════════════════════════════════════════════════════
CEDICT_DOMAIN_KEYWORDS: dict[str, str] = {
    # 医学
    "medicine": "医学",
    "disease": "医学",
    "drug": "医学",
    "symptom": "医学",
    "clinical": "医学",
    "therapy": "医学",
    "surgery": "医学",
    "virus": "医学",
    "bacteria": "医学",
    "anatomy": "医学",
    # 物理
    "physics": "物理",
    "quantum": "物理",
    "mechanics": "物理",
    "energy": "物理",
    "force": "物理",
    "wave": "物理",
    "particle": "物理",
    "electromagnetic": "物理",
    "gravity": "物理",
    "thermodynamic": "物理",
    # 数学
    "mathematics": "数学",
    "equation": "数学",
    "theorem": "数学",
    "proof": "数学",
    "algebra": "数学",
    "geometry": "数学",
    "calculus": "数学",
    "probability": "数学",
    "statistics": "数学",
    # 化学
    "chemistry": "化学",
    "reaction": "化学",
    "element": "化学",
    "molecule": "化学",
    "compound": "化学",
    "organic": "化学",
    # 生物
    "biology": "生物",
    "cell": "生物",
    "gene": "生物",
    "species": "生物",
    "evolution": "生物",
    "ecology": "生物",
    "photosynthesis": "生物",
    # 文学
    "literature": "文学",
    "poetry": "文学",
    "poem": "文学",
    "novel": "文学",
    "writer": "文学",
    "dynasty": "文学",
    "calligraphy": "文学",
    # 法律
    "law": "法律",
    "legal": "法律",
    "court": "法律",
    "statute": "法律",
    "contract": "法律",
    "crime": "法律",
    "justice": "法律",
    # 计算机
    "computer": "计算机",
    "software": "计算机",
    "algorithm": "计算机",
    "programming": "计算机",
    "database": "计算机",
    "network": "计算机",
    "artificial intelligence": "计算机",
    "machine learning": "计算机",
    "deep learning": "计算机",
    # 历史
    "history": "历史",
    "war": "历史",
    "emperor": "历史",
    "ancient": "历史",
    # 营养
    "nutrition": "营养",
    "vitamin": "营养",
    "diet": "营养",
    "food": "营养",
}

# CEDICT 关键词按长度降序预排序（优先匹配更具体的关键词）
_SORTED_CEDICT_KEYWORDS = sorted(CEDICT_DOMAIN_KEYWORDS.keys(), key=len, reverse=True)

# ═══════════════════════════════════════════════════════════════
#  Wikipedia 分类关键词 → 领域映射
# ═══════════════════════════════════════════════════════════════
WIKI_CATEGORY_DOMAIN: dict[str, str] = {
    # 医学
    "医学": "医学",
    "疾病": "医学",
    "药物": "医学",
    "治疗": "医学",
    "医院": "医学",
    "症状": "医学",
    "手术": "医学",
    "卫生": "医学",
    # 物理
    "物理": "物理",
    "力学": "物理",
    "量子": "物理",
    "相对论": "物理",
    "光学": "物理",
    "电磁学": "物理",
    "热力学": "物理",
    # 数学
    "数学": "数学",
    "几何": "数学",
    "代数": "数学",
    "统计": "数学",
    "概率": "数学",
    "数论": "数学",
    # 化学
    "化学": "化学",
    "有机化学": "化学",
    "无机化学": "化学",
    "元素": "化学",
    "化合物": "化学",
    # 生物
    "生物": "生物",
    "动物": "生物",
    "植物": "生物",
    "生态": "生物",
    "遗传": "生物",
    "细胞": "生物",
    "进化": "生物",
    # 文学
    "文学": "文学",
    "诗歌": "文学",
    "小说": "文学",
    "作家": "文学",
    "书法": "文学",
    "散文": "文学",
    "戏剧": "文学",
    # 法律
    "法律": "法律",
    "法规": "法律",
    "司法": "法律",
    "刑法": "法律",
    "民法": "法律",
    "宪法": "法律",
    # 计算机
    "计算机": "计算机",
    "软件": "计算机",
    "编程": "计算机",
    "互联网": "计算机",
    "人工智能": "计算机",
    "数据库": "计算机",
    "算法": "计算机",
    # 历史
    "历史": "历史",
    "朝代": "历史",
    "战争": "历史",
    "皇帝": "历史",
    "古代": "历史",
    "文明": "历史",
    # 营养
    "营养": "营养",
    "食物": "营养",
    "饮食": "营养",
    "维生素": "营养",
    "健康": "营养",
}


# ═══════════════════════════════════════════════════════════════
#  DomainInferenceReport 数据类
# ═══════════════════════════════════════════════════════════════

@dataclass
class DomainInferenceReport:
    """领域推断报告"""

    total_empty: int           # 修复前 domain 为空的种子数
    inferred: int              # 成功推断的种子数（含兜底）
    fallback_common: int       # 兜底为"常识"的种子数
    cycles_detected: int       # 检测到的 IS_A 循环链数
    coverage_rate: float       # domain 非空率
    elapsed_s: float           # 耗时秒数
    by_method: dict[str, int] = field(default_factory=dict)  # 按推断方法的统计

    def to_dict(self) -> dict:
        """转换为字典（用于 JSON 序列化）"""
        return {
            "total_empty": self.total_empty,
            "inferred": self.inferred,
            "fallback_common": self.fallback_common,
            "cycles_detected": self.cycles_detected,
            "coverage_rate": round(self.coverage_rate, 4),
            "elapsed_s": round(self.elapsed_s, 2),
            "by_method": dict(self.by_method),
        }


# ═══════════════════════════════════════════════════════════════
#  infer_single_domain — 单种子领域推断
# ═══════════════════════════════════════════════════════════════

def infer_single_domain(
    seed_label: str,
    isa_map: dict[str, list[str]],
    domain_map: dict[str, str],
    cedict_data: dict[str, dict] | None = None,
    wiki_categories: dict[str, list[str]] | None = None,
) -> tuple[str, str]:
    """
    推断单个种子的领域。

    推断优先级:
      1. IS_A 层级上溯 (BFS) → 继承祖先 domain
      2. CC-CEDICT 释义关键词映射
      3. Wikipedia 分类关键词映射
      4. 兜底 → "常识"

    Args:
        seed_label: 种子标签
        isa_map: IS_A 边映射 {source: [target1, target2, ...]}
                 source IS_A target 表示 source 是 target 的一种
        domain_map: 种子 domain 映射 {label: domain}
        cedict_data: CC-CEDICT 数据 {word: {"english": "...", ...}}（可选）
        wiki_categories: Wikipedia 分类 {title: [cat1, cat2, ...]}（可选）

    Returns:
        (domain, method) 二元组
        method ∈ {'isa_inherit', 'cedict', 'wikipedia', 'fallback_common', 'cycle_detected'}
    """

    # ── 路径 1: IS_A BFS 上溯 ──────────────────────────────
    visited: set[str] = set()
    cycle_detected = False
    # BFS 队列: (node_label, depth)
    queue: deque[tuple[str, int]] = deque()
    queue.append((seed_label, 0))
    visited.add(seed_label)

    while queue:
        current, depth = queue.popleft()

        # 超过最大深度则停止沿此路径继续
        if depth >= BFS_MAX_DEPTH:
            continue

        # 获取 current 的 IS_A 出边 → 父节点列表
        parents = isa_map.get(current, [])
        for parent in parents:
            if parent in visited:
                # 检测到循环链，跳过该分支，继续搜索其他分支
                log.debug("IS_A 循环检测: %s → ... → %s", seed_label, parent)
                cycle_detected = True
                continue

            visited.add(parent)

            # 检查父节点是否有 domain
            parent_domain = domain_map.get(parent, "")
            if parent_domain:
                return (parent_domain, "isa_inherit")

            # 继续上溯
            queue.append((parent, depth + 1))

    # 如果 BFS 过程中检测到循环且未找到有效领域，返回循环检测结果
    if cycle_detected:
        return ("常识", "cycle_detected")

    # ── 路径 2: CC-CEDICT 释义关键词映射 ──────────────────
    if cedict_data is not None:
        cedict_result = _infer_from_cedict(seed_label, cedict_data)
        if cedict_result is not None:
            return (cedict_result, "cedict")

    # ── 路径 3: Wikipedia 分类关键词映射 ──────────────────
    if wiki_categories is not None:
        wiki_result = _infer_from_wikipedia(seed_label, wiki_categories)
        if wiki_result is not None:
            return (wiki_result, "wikipedia")

    # ── 路径 4: 兜底 → "常识" ─────────────────────────────
    return ("常识", "fallback_common")


def _infer_from_cedict(
    label: str,
    cedict_data: dict[str, dict],
) -> str | None:
    """
    从 CC-CEDICT 释义中提取领域关键词映射。

    Args:
        label: 种子标签
        cedict_data: CC-CEDICT 数据 {word: {"english": "...", ...}}

    Returns:
        领域名称或 None
    """
    entry = cedict_data.get(label)
    if entry is None:
        return None

    english_text = entry.get("english", "")
    if not english_text:
        return None

    # 将英文释义转为小写，便于关键词匹配
    english_lower = english_text.lower()

    # 按关键词长度降序排列，优先匹配更具体的关键词
    # （如 "deep learning" 优先于 "learning"）
    for keyword in _SORTED_CEDICT_KEYWORDS:
        if keyword in english_lower:
            return CEDICT_DOMAIN_KEYWORDS[keyword]

    return None


def _infer_from_wikipedia(
    label: str,
    wiki_categories: dict[str, list[str]],
) -> str | None:
    """
    从 Wikipedia 分类中推断领域。

    Args:
        label: 种子标签
        wiki_categories: Wikipedia 分类 {title: [cat1, cat2, ...]}

    Returns:
        领域名称或 None
    """
    categories = wiki_categories.get(label)
    if categories is None or not categories:
        return None

    # 遍历所有分类，匹配关键词
    for category in categories:
        for keyword, domain in WIKI_CATEGORY_DOMAIN.items():
            if keyword in category:
                return domain

    return None


# ═══════════════════════════════════════════════════════════════
#  辅助函数: 加载外部数据源
# ═══════════════════════════════════════════════════════════════

def _load_isa_map(conn: sqlite3.Connection) -> dict[str, list[str]]:
    """
    预加载所有 IS_A 边到内存。

    查询: SELECT source, target FROM karma_edges WHERE relation='IS_A'
    映射: source → [target1, target2, ...]
          (source IS_A target 表示 source 是 target 的一种)

    Args:
        conn: SQLite 数据库连接

    Returns:
        IS_A 边映射字典
    """
    isa_map: dict[str, list[str]] = {}
    rows = conn.execute(
        "SELECT source, target FROM karma_edges WHERE relation = 'IS_A'"
    ).fetchall()
    for source, target in rows:
        if source not in isa_map:
            isa_map[source] = []
        isa_map[source].append(target)
    return isa_map


def _load_domain_map(conn: sqlite3.Connection) -> dict[str, str]:
    """
    预加载所有种子的 domain 字段到内存。

    查询: SELECT label, domain FROM seeds

    Args:
        conn: SQLite 数据库连接

    Returns:
        domain 映射字典 {label: domain}
    """
    domain_map: dict[str, str] = {}
    rows = conn.execute("SELECT label, domain FROM seeds").fetchall()
    for label, domain in rows:
        domain_map[label] = domain
    return domain_map


def _load_cedict_data(cedict_path: Path | None) -> dict[str, dict] | None:
    """
    加载 CC-CEDICT 解析后的 JSON 数据。

    Args:
        cedict_path: CC-CEDICT JSON 文件路径

    Returns:
        CC-CEDICT 数据字典，加载失败返回 None
    """
    if cedict_path is None:
        return None

    if not cedict_path.exists():
        log.warning("CC-CEDICT 数据文件不存在: %s，跳过 CEDICT 推断路径", cedict_path)
        return None

    try:
        with open(cedict_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        log.info("CC-CEDICT 数据加载成功: %d 条记录", len(data))
        return data
    except (json.JSONDecodeError, OSError) as e:
        log.warning("CC-CEDICT 数据加载失败: %s，跳过 CEDICT 推断路径", e)
        return None


def _load_wiki_categories(wiki_db_path: Path | None) -> dict[str, list[str]] | None:
    """
    加载 Wikipedia 分类数据。

    查询 zhwiki.db 的 categories 表:
      SELECT title, category FROM categories

    Args:
        wiki_db_path: Wikipedia 数据库路径

    Returns:
        Wikipedia 分类字典 {title: [cat1, cat2, ...]}，加载失败返回 None
    """
    if wiki_db_path is None:
        return None

    if not wiki_db_path.exists():
        log.warning("Wikipedia 数据库不存在: %s，跳过 Wikipedia 推断路径", wiki_db_path)
        return None

    try:
        wiki_conn = sqlite3.connect(str(wiki_db_path))
        wiki_conn.row_factory = sqlite3.Row

        # 检查 categories 表是否存在
        table_check = wiki_conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='categories'"
        ).fetchone()
        if table_check is None:
            log.warning("Wikipedia 数据库中不存在 categories 表，跳过 Wikipedia 推断路径")
            wiki_conn.close()
            return None

        rows = wiki_conn.execute(
            "SELECT title, category FROM categories"
        ).fetchall()

        wiki_categories: dict[str, list[str]] = {}
        for row in rows:
            title = row[0]
            category = row[1]
            if title not in wiki_categories:
                wiki_categories[title] = []
            wiki_categories[title].append(category)

        wiki_conn.close()
        log.info("Wikipedia 分类数据加载成功: %d 个条目", len(wiki_categories))
        return wiki_categories

    except sqlite3.Error as e:
        log.warning("Wikipedia 数据库加载失败: %s，跳过 Wikipedia 推断路径", e)
        return None


# ═══════════════════════════════════════════════════════════════
#  断点续传
# ═══════════════════════════════════════════════════════════════

def _load_progress(progress_file: Path) -> dict:
    """
    加载断点续传进度。

    Args:
        progress_file: 进度文件路径

    Returns:
        进度字典，格式: {"processed": int, "total": int, "updated": int}
    """
    if progress_file.exists():
        try:
            with open(progress_file, "r", encoding="utf-8") as f:
                return json.load(f)
        except (json.JSONDecodeError, OSError):
            log.warning("进度文件损坏，从头开始")
    return {"processed": 0, "total": 0, "updated": 0}


def _save_progress(progress_file: Path, progress: dict) -> None:
    """
    保存断点续传进度。

    Args:
        progress_file: 进度文件路径
        progress: 进度字典
    """
    progress_file.parent.mkdir(parents=True, exist_ok=True)
    with open(progress_file, "w", encoding="utf-8") as f:
        json.dump(progress, f, indent=2, ensure_ascii=False)


# ═══════════════════════════════════════════════════════════════
#  infer_domains — 主函数
# ═══════════════════════════════════════════════════════════════

def infer_domains(
    db_path: str | Path = DEFAULT_DB_PATH,
    cedict_path: str | Path | None = None,
    wiki_db_path: str | Path | None = None,
    batch_size: int = REPAIR_BATCH_SIZE,
    progress_file: str | Path | None = None,
) -> DomainInferenceReport:
    """
    为所有 domain 为空的种子推断领域。

    处理流程:
      1. 预加载 IS_A 边到内存 isa_map
      2. 预加载所有种子 domain 到内存 domain_map
      3. 收集 domain 为空的种子
      4. 逐个调用 infer_single_domain() 推断
      5. 每 batch_size 个种子批量 UPDATE
      6. 断点续传: 进度记录到 data/domain_inference_progress.json

    Args:
        db_path: 识海数据库路径
        cedict_path: CC-CEDICT 解析后 JSON 文件路径（可选）
        wiki_db_path: Wikipedia 中文数据库路径（可选）
        batch_size: 批量 UPDATE 大小，默认 50000
        progress_file: 断点续传进度文件路径（可选）

    Returns:
        DomainInferenceReport 推断报告
    """
    start_time = time.monotonic()

    db_path = Path(db_path)
    cedict_path_resolved = Path(cedict_path) if cedict_path else None
    wiki_db_path_resolved = Path(wiki_db_path) if wiki_db_path else None

    # 进度文件默认路径
    if progress_file is None:
        progress_file = db_path.parent / "domain_inference_progress.json"
    else:
        progress_file = Path(progress_file)

    # 加载断点续传进度
    progress = _load_progress(progress_file)

    # 连接数据库
    conn = sqlite3.connect(str(db_path))
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA synchronous=NORMAL")
    conn.execute("PRAGMA cache_size=-2000000")  # 2GB 缓存

    try:
        # ── 步骤 1: 预加载 IS_A 边 ──────────────────────────
        log.info("预加载 IS_A 边到内存...")
        isa_map = _load_isa_map(conn)
        log.info("IS_A 边加载完成: %d 个源节点", len(isa_map))

        # ── 步骤 2: 预加载 domain 映射 ──────────────────────
        log.info("预加载种子 domain 到内存...")
        domain_map = _load_domain_map(conn)
        log.info("Domain 映射加载完成: %d 个种子", len(domain_map))

        # ── 步骤 3: 加载外部数据源 ──────────────────────────
        log.info("加载 CC-CEDICT 数据...")
        cedict_data = _load_cedict_data(cedict_path_resolved)

        log.info("加载 Wikipedia 分类数据...")
        wiki_categories = _load_wiki_categories(wiki_db_path_resolved)

        # ── 步骤 4: 收集 domain 为空的种子 ──────────────────
        empty_seeds_rows = conn.execute(
            "SELECT label FROM seeds WHERE domain = '' OR domain IS NULL"
        ).fetchall()
        empty_labels = [row[0] for row in empty_seeds_rows]
        total_empty = len(empty_labels)

        log.info("domain 为空的种子数: %d / %d", total_empty, len(domain_map))

        # ── 步骤 5: 断点续传 — 跳过已处理的种子 ────────────
        processed_offset = progress.get("processed", 0)
        if processed_offset > 0:
            log.info("断点续传: 跳过前 %d 个已处理种子", processed_offset)
            # 重新计算: 从数据库中获取已更新的种子标签
            # 简化策略: 已处理的种子在 domain_map 中已有 domain，跳过
            remaining_labels = []
            for label in empty_labels:
                if domain_map.get(label, "") == "":
                    remaining_labels.append(label)
            empty_labels = remaining_labels
            total_empty = len(empty_labels)
            log.info("断点续传后剩余待处理种子: %d", total_empty)

        # ── 步骤 6: 逐个推断 + 批量 UPDATE ─────────────────
        by_method: dict[str, int] = {}
        fallback_common = 0
        cycles_detected = 0
        inferred = 0
        batch_updates: list[tuple[str, str]] = []  # (domain, label)

        for i, label in enumerate(empty_labels):
            domain, method = infer_single_domain(
                seed_label=label,
                isa_map=isa_map,
                domain_map=domain_map,
                cedict_data=cedict_data,
                wiki_categories=wiki_categories,
            )

            # 统计
            by_method[method] = by_method.get(method, 0) + 1
            if method == "fallback_common":
                fallback_common += 1
            if method == "cycle_detected":
                cycles_detected += 1
                fallback_common += 1  # 循环检测也归入兜底

            # 更新内存中的 domain_map（后续种子可能依赖此结果）
            domain_map[label] = domain
            inferred += 1

            # 加入批量更新队列
            batch_updates.append((domain, label))

            # 批量写入
            if len(batch_updates) >= batch_size:
                _batch_update_domains(conn, batch_updates)
                log.info(
                    "批量更新: %d 条 (进度: %d / %d)",
                    len(batch_updates), i + 1, total_empty,
                )
                # 保存进度
                progress["processed"] = i + 1
                progress["total"] = total_empty
                progress["updated"] = progress.get("updated", 0) + len(batch_updates)
                _save_progress(progress_file, progress)
                batch_updates = []

        # 处理剩余的批次
        if batch_updates:
            _batch_update_domains(conn, batch_updates)
            log.info("最终批量更新: %d 条", len(batch_updates))

        # ── 步骤 7: 计算覆盖率 ──────────────────────────────
        total_seeds = conn.execute("SELECT COUNT(*) FROM seeds").fetchone()[0]
        with_domain = conn.execute(
            "SELECT COUNT(*) FROM seeds WHERE domain != '' AND domain IS NOT NULL"
        ).fetchone()[0]
        coverage_rate = with_domain / total_seeds if total_seeds > 0 else 0.0

        elapsed_s = time.monotonic() - start_time

        # ── 步骤 8: 清理进度文件 ─────────────────────────────
        if progress_file.exists():
            try:
                progress_file.unlink()
            except OSError:
                pass

        report = DomainInferenceReport(
            total_empty=total_empty,
            inferred=inferred,
            fallback_common=fallback_common,
            cycles_detected=cycles_detected,
            coverage_rate=coverage_rate,
            elapsed_s=elapsed_s,
            by_method=by_method,
        )

        log.info(
            "领域推断完成: 推断 %d, 兜底 %d, 循环 %d, 覆盖率 %.2f%%, 耗时 %.1fs",
            inferred, fallback_common, cycles_detected,
            coverage_rate * 100, elapsed_s,
        )

        if coverage_rate < DOMAIN_COVERAGE_TARGET:
            log.warning(
                "领域覆盖率 %.2f%% 低于目标 %.2f%%",
                coverage_rate * 100, DOMAIN_COVERAGE_TARGET * 100,
            )

        return report

    finally:
        conn.close()


def _batch_update_domains(
    conn: sqlite3.Connection,
    updates: list[tuple[str, str]],
) -> None:
    """
    批量更新种子的 domain 字段。

    使用参数化查询防止 SQL 注入。

    Args:
        conn: SQLite 数据库连接
        updates: 更新列表 [(domain, label), ...]
    """
    conn.executemany(
        "UPDATE seeds SET domain = ? WHERE label = ?",
        updates,
    )
    conn.commit()