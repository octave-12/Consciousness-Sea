from __future__ import annotations

import argparse
import json
import logging
import shutil
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

from consciousness_sea.infrastructure.config import resolve_data_dir

log = logging.getLogger(__name__)


def create_backup(
    db_path: str | None = None,
    backup_dir: str | None = None,
) -> Path:
    if db_path is None:
        db_path = str(resolve_data_dir() / "consciousness_sea.db")
    if backup_dir is None:
        backup_dir = str(resolve_data_dir() / "backups")

    backup_path = Path(backup_dir)
    backup_path.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    backup_file = backup_path / f"consciousness_sea_{timestamp}.db"

    source = Path(db_path)
    if not source.exists():
        raise FileNotFoundError(f"数据库文件不存在: {db_path}")

    source_conn = sqlite3.connect(str(source))
    backup_conn = sqlite3.connect(str(backup_file))

    with backup_conn:
        source_conn.backup(backup_conn)

    source_conn.close()
    backup_conn.close()

    meta = {
        "backup_file": str(backup_file),
        "source_db": str(source),
        "timestamp": timestamp,
        "file_size_bytes": backup_file.stat().st_size,
    }
    meta_file = backup_path / f"consciousness_sea_{timestamp}.json"
    meta_file.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")

    log.info("备份完成: %s (%d bytes)", backup_file, meta["file_size_bytes"])
    return backup_file


def restore_backup(
    backup_file: str,
    db_path: str | None = None,
) -> None:
    if db_path is None:
        db_path = str(resolve_data_dir() / "consciousness_sea.db")

    source = Path(backup_file)
    if not source.exists():
        raise FileNotFoundError(f"备份文件不存在: {backup_file}")

    target = Path(db_path)
    if target.exists():
        pre_restore = target.parent / f"{target.stem}_pre_restore_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}.db"
        shutil.copy2(str(target), str(pre_restore))
        log.info("恢复前快照: %s", pre_restore)

    backup_conn = sqlite3.connect(str(source))
    target_conn = sqlite3.connect(str(target))

    with target_conn:
        backup_conn.backup(target_conn)

    backup_conn.close()
    target_conn.close()

    log.info("恢复完成: %s → %s", backup_file, db_path)


def list_backups(backup_dir: str | None = None) -> list[dict]:
    if backup_dir is None:
        backup_dir = str(resolve_data_dir() / "backups")

    backup_path = Path(backup_dir)
    if not backup_path.exists():
        return []

    backups = []
    for meta_file in sorted(backup_path.glob("*.json"), reverse=True):
        try:
            meta = json.loads(meta_file.read_text(encoding="utf-8"))
            backups.append(meta)
        except Exception:
            pass

    return backups


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser(description="识海数据备份与恢复")
    sub = parser.add_subparsers(dest="command")

    backup_parser = sub.add_parser("backup", help="创建备份")
    backup_parser.add_argument("--db-path", help="数据库路径")
    backup_parser.add_argument("--backup-dir", help="备份目录")

    restore_parser = sub.add_parser("restore", help="从备份恢复")
    restore_parser.add_argument("backup_file", help="备份文件路径")
    restore_parser.add_argument("--db-path", help="目标数据库路径")

    list_parser = sub.add_parser("list", help="列出备份")
    list_parser.add_argument("--backup-dir", help="备份目录")

    args = parser.parse_args()

    if args.command == "backup":
        create_backup(args.db_path, args.backup_dir)
    elif args.command == "restore":
        restore_backup(args.backup_file, args.db_path)
    elif args.command == "list":
        for b in list_backups(args.backup_dir):
            print(f"  {b['timestamp']}: {b['backup_file']} ({b['file_size_bytes']} bytes)")
    else:
        parser.print_help()