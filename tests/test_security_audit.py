"""
安全审计与渗透测试脚本

验证:
  - SQL 注入防护（参数化查询）
  - XSS 防护（输入校验 + 输出转义）
  - 认证绕过防护
  - 限流器防暴力破解
  - JWT 令牌安全
"""

from __future__ import annotations

import sys
from pathlib import Path

_src = str(Path(__file__).resolve().parent.parent / "backend" / "src")
if _src not in sys.path:
    sys.path.insert(0, _src)

import sqlite3
from unittest.mock import patch

import pytest
from consciousness_sea.domain.graph_db import GraphDB
from consciousness_sea.infrastructure.rate_limiter import RateLimiter


class TestSQLInjectionPrevention:
    """SQL 注入防护测试"""

    @pytest.fixture
    def graph(self):
        conn = sqlite3.connect(":memory:", check_same_thread=False)
        conn.row_factory = sqlite3.Row
        conn.executescript("""
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
        conn.executemany(
            "INSERT INTO seeds (id, label, type, aliases, domain, definition) VALUES (?, ?, ?, ?, ?, ?)",
            [("感冒", "感冒", "CONCEPT", "[]", "医学", "test")],
        )
        conn.commit()
        db = GraphDB(":memory:")
        db.conn = conn
        db.ensure_phase2_tables()
        return db

    def test_match_seeds_sql_injection(self, graph):
        """match_seeds 使用参数化查询，SQL 注入无效"""
        malicious = "'; DROP TABLE seeds; --"
        result = graph.match_seeds(malicious)
        assert isinstance(result, list)
        rows = graph.conn.execute("SELECT COUNT(*) FROM seeds").fetchone()
        assert rows[0] > 0

    def test_get_seed_sql_injection(self, graph):
        """get_seed 使用参数化查询"""
        malicious = "'; DROP TABLE seeds; --"
        result = graph.get_seed(malicious)
        assert result is None
        rows = graph.conn.execute("SELECT COUNT(*) FROM seeds").fetchone()
        assert rows[0] > 0

    def test_outgoing_edges_sql_injection(self, graph):
        """outgoing_edges 使用参数化查询"""
        malicious = "'; DROP TABLE seeds; --"
        result = graph.outgoing_edges(malicious, exclude_meta=False)
        assert isinstance(result, list)
        rows = graph.conn.execute("SELECT COUNT(*) FROM seeds").fetchone()
        assert rows[0] > 0

    def test_adjust_karma_sql_injection(self, graph):
        """adjust_karma 使用参数化查询"""
        graph.adjust_karma("感冒", "'; DROP TABLE seeds; --", "RELATED", 0.1)
        rows = graph.conn.execute("SELECT COUNT(*) FROM seeds").fetchone()
        assert rows[0] > 0


class TestXSSPrevention:
    """XSS 防护测试"""

    @pytest.fixture
    def graph(self):
        conn = sqlite3.connect(":memory:", check_same_thread=False)
        conn.row_factory = sqlite3.Row
        conn.executescript("""
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
        db = GraphDB(":memory:")
        db.conn = conn
        db.ensure_phase2_tables()
        return db

    def test_seed_label_html_safe_storage(self, graph):
        """种子标签中的 HTML 特殊字符被安全存储"""
        xss_label = '<img onerror=alert(1)>'
        graph.conn.execute(
            "INSERT INTO seeds (id, label, type, aliases, domain, definition) VALUES (?, ?, 'CONCEPT', '[]', 'test', 'test')",
            (xss_label, xss_label),
        )
        graph.conn.commit()
        result = graph.get_seed(xss_label)
        assert result is not None
        assert result["label"] == xss_label

    def test_match_seeds_with_html_chars(self, graph):
        """match_seeds 安全处理 HTML 字符"""
        result = graph.match_seeds("<script>alert(1)</script>")
        assert isinstance(result, list)


class TestAuthSecurity:
    """认证安全测试"""

    def test_jwt_token_verification(self):
        """JWT 令牌验证正确"""
        from consciousness_sea.infrastructure import auth as auth_module
        with patch.object(auth_module, "_JWT_SECRET", "test-secret-key-for-security-audit"):
            token = auth_module.create_access_token("user1")
            payload = auth_module.decode_access_token(token)
            assert payload is not None
            assert payload["sub"] == "user1"

    def test_jwt_rejects_tampered_token(self):
        """JWT 拒绝被篡改的令牌"""
        from consciousness_sea.infrastructure import auth as auth_module
        with patch.object(auth_module, "_JWT_SECRET", "test-secret-key-for-security-audit"):
            token = auth_module.create_access_token("user1")
            tampered = token[:-5] + "XXXXX"
            payload = auth_module.decode_access_token(tampered)
            assert payload is None

    def test_jwt_rejects_expired_token(self):
        """JWT 拒绝过期令牌"""
        import base64
        import hashlib
        import json
        import time

        from consciousness_sea.infrastructure import auth as auth_module

        secret = "test-secret-key-for-security-audit"
        with patch.object(auth_module, "_JWT_SECRET", secret):
            header = {"alg": "HS256", "typ": "JWT"}
            payload = {"sub": "user1", "exp": time.time() - 3600}

            def _b64url(data: bytes) -> str:
                return base64.urlsafe_b64encode(data).rstrip(b"=").decode("ascii")

            header_b64 = _b64url(json.dumps(header, separators=(",", ":")).encode())
            payload_b64 = _b64url(json.dumps(payload, separators=(",", ":")).encode())
            signing_input = f"{header_b64}.{payload_b64}"
            sig = __import__("hmac").new(secret.encode(), signing_input.encode(), hashlib.sha256).digest()
            token = f"{signing_input}.{_b64url(sig)}"

            result = auth_module.decode_access_token(token)
            assert result is None

    def test_api_key_validation(self):
        """API Key 验证正确"""
        from consciousness_sea.infrastructure.auth import APIKeyAuth
        auth = APIKeyAuth(api_key="test-key-123", enabled=True)
        assert auth.validate("test-key-123") is True
        assert auth.validate("wrong-key") is False
        assert auth.validate(None) is False


class TestRateLimiterSecurity:
    """限流器安全测试"""

    def test_brute_force_protection(self):
        """限流器防止暴力破解"""
        limiter = RateLimiter(ip_limit=5, ip_window=60)
        ip = "192.168.1.100"

        for _ in range(5):
            assert limiter.check_ip(ip) is True

        assert limiter.check_ip(ip) is False

    def test_different_ips_independent(self):
        """不同 IP 的限流计数独立"""
        limiter = RateLimiter(ip_limit=3, ip_window=60)
        for _ in range(3):
            assert limiter.check_ip("1.1.1.1") is True
        assert limiter.check_ip("1.1.1.1") is False
        assert limiter.check_ip("2.2.2.2") is True

    def test_rate_limiter_reset(self):
        """限流器重置后允许请求"""
        limiter = RateLimiter(ip_limit=2, ip_window=60)
        ip = "3.3.3.3"
        limiter.check_ip(ip)
        limiter.check_ip(ip)
        assert limiter.check_ip(ip) is False
        limiter._ip_entries.clear()
        assert limiter.check_ip(ip) is True

    def test_user_rate_limiting(self):
        """用户级限流"""
        limiter = RateLimiter(user_limit=3, user_window=60)
        for _ in range(3):
            assert limiter.check_user("user1") is True
        assert limiter.check_user("user1") is False
        assert limiter.check_user("user2") is True


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
