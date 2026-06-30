from __future__ import annotations

import json
import logging
import os
import sqlite3
import threading
from datetime import datetime, timezone
from typing import Any, Optional

from consciousness_sea.infrastructure.config import resolve_data_dir

log = logging.getLogger(__name__)

_AUDIT_DB_PATH = os.environ.get(
    "CONSCIOUSNESS_SEA_AUDIT_DB_PATH",
    str(resolve_data_dir() / "audit_log.db"),
)

_audit_lock = threading.Lock()
_audit_conn: Optional[sqlite3.Connection] = None


def _ensure_audit_db() -> sqlite3.Connection:
    global _audit_conn
    if _audit_conn is not None:
        return _audit_conn
    with _audit_lock:
        if _audit_conn is not None:
            return _audit_conn
        conn = sqlite3.connect(_AUDIT_DB_PATH, check_same_thread=False)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS audit_log (
                log_id        INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp     TEXT    NOT NULL,
                user_id       TEXT,
                action        TEXT    NOT NULL,
                resource      TEXT,
                method        TEXT,
                path          TEXT,
                status_code   INTEGER,
                ip_address    TEXT,
                user_agent    TEXT,
                request_body  TEXT,
                response_time_ms REAL,
                metadata_json TEXT
            )
        """)
        conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_audit_timestamp
                ON audit_log (timestamp DESC)
        """)
        conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_audit_user_id
                ON audit_log (user_id)
        """)
        conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_audit_action
                ON audit_log (action)
        """)
        conn.commit()
        _audit_conn = conn
        return _audit_conn


def record_audit(
    action: str,
    user_id: Optional[str] = None,
    resource: Optional[str] = None,
    method: Optional[str] = None,
    path: Optional[str] = None,
    status_code: Optional[int] = None,
    ip_address: Optional[str] = None,
    user_agent: Optional[str] = None,
    request_body: Optional[str] = None,
    response_time_ms: Optional[float] = None,
    metadata: Optional[dict[str, Any]] = None,
) -> None:
    try:
        conn = _ensure_audit_db()
        now = datetime.now(timezone.utc).isoformat()
        metadata_json = json.dumps(metadata, ensure_ascii=False) if metadata else None
        with _audit_lock:
            conn.execute(
                "INSERT INTO audit_log "
                "(timestamp, user_id, action, resource, method, path, status_code, "
                "ip_address, user_agent, request_body, response_time_ms, metadata_json) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (now, user_id, action, resource, method, path, status_code,
                 ip_address, user_agent, request_body, response_time_ms, metadata_json),
            )
            conn.commit()
    except Exception as e:
        log.warning("审计日志写入失败: %s", e)


def query_audit_log(
    user_id: Optional[str] = None,
    action: Optional[str] = None,
    start_time: Optional[str] = None,
    end_time: Optional[str] = None,
    limit: int = 100,
    offset: int = 0,
) -> list[dict[str, Any]]:
    try:
        conn = _ensure_audit_db()
        conditions: list[str] = []
        params: list[Any] = []

        if user_id is not None:
            conditions.append("user_id = ?")
            params.append(user_id)
        if action is not None:
            conditions.append("action = ?")
            params.append(action)
        if start_time is not None:
            conditions.append("timestamp >= ?")
            params.append(start_time)
        if end_time is not None:
            conditions.append("timestamp <= ?")
            params.append(end_time)

        where_clause = ""
        if conditions:
            where_clause = "WHERE " + " AND ".join(conditions)

        rows = conn.execute(
            f"SELECT * FROM audit_log {where_clause} ORDER BY timestamp DESC LIMIT ? OFFSET ?",
            params + [limit, offset],
        ).fetchall()

        return [dict(r) for r in rows]
    except Exception as e:
        log.warning("审计日志查询失败: %s", e)
        return []


def close_audit_db() -> None:
    global _audit_conn
    if _audit_conn is not None:
        with _audit_lock:
            if _audit_conn is not None:
                _audit_conn.close()
                _audit_conn = None
