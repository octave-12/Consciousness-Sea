from __future__ import annotations

import logging
import threading
import time
from collections import defaultdict
from dataclasses import dataclass

from fastapi import Request
from fastapi.responses import JSONResponse

log = logging.getLogger(__name__)

IP_RATE_LIMIT = 100
IP_RATE_WINDOW = 60
USER_RATE_LIMIT = 1000
USER_RATE_WINDOW = 3600
CIRCUIT_BREAKER_THRESHOLD = 5
CIRCUIT_BREAKER_RESET_SECONDS = 60


@dataclass
class RateLimitEntry:
    count: int = 0
    window_start: float = 0.0


@dataclass
class CircuitBreakerState:
    failure_count: int = 0
    last_failure_time: float = 0.0
    is_open: bool = False


class RateLimiter:
    def __init__(
        self,
        ip_limit: int = IP_RATE_LIMIT,
        ip_window: int = IP_RATE_WINDOW,
        user_limit: int = USER_RATE_LIMIT,
        user_window: int = USER_RATE_WINDOW,
    ) -> None:
        self._ip_limit = ip_limit
        self._ip_window = ip_window
        self._user_limit = user_limit
        self._user_window = user_window
        self._ip_entries: dict[str, RateLimitEntry] = defaultdict(RateLimitEntry)
        self._user_entries: dict[str, RateLimitEntry] = defaultdict(RateLimitEntry)
        self._lock = threading.Lock()

    def check_ip(self, ip: str) -> bool:
        now = time.time()
        with self._lock:
            entry = self._ip_entries[ip]
            if now - entry.window_start > self._ip_window:
                entry.count = 0
                entry.window_start = now
            entry.count += 1
            return entry.count <= self._ip_limit

    def check_user(self, user_id: str) -> bool:
        now = time.time()
        with self._lock:
            entry = self._user_entries[user_id]
            if now - entry.window_start > self._user_window:
                entry.count = 0
                entry.window_start = now
            entry.count += 1
            return entry.count <= self._user_limit

    def get_ip_remaining(self, ip: str) -> int:
        now = time.time()
        with self._lock:
            entry = self._ip_entries[ip]
            if now - entry.window_start > self._ip_window:
                return self._ip_limit
            return max(0, self._ip_limit - entry.count)

    def get_user_remaining(self, user_id: str) -> int:
        now = time.time()
        with self._lock:
            entry = self._user_entries[user_id]
            if now - entry.window_start > self._user_window:
                return self._user_limit
            return max(0, self._user_limit - entry.count)


class CircuitBreaker:
    def __init__(
        self,
        failure_threshold: int = CIRCUIT_BREAKER_THRESHOLD,
        reset_seconds: int = CIRCUIT_BREAKER_RESET_SECONDS,
    ) -> None:
        self._failure_threshold = failure_threshold
        self._reset_seconds = reset_seconds
        self._circuits: dict[str, CircuitBreakerState] = defaultdict(CircuitBreakerState)
        self._lock = threading.Lock()

    def is_open(self, service: str) -> bool:
        with self._lock:
            circuit = self._circuits[service]
            if circuit.is_open:
                if time.time() - circuit.last_failure_time > self._reset_seconds:
                    circuit.is_open = False
                    circuit.failure_count = 0
                    return False
                return True
            return False

    def record_failure(self, service: str) -> None:
        with self._lock:
            circuit = self._circuits[service]
            circuit.failure_count += 1
            circuit.last_failure_time = time.time()
            if circuit.failure_count >= self._failure_threshold:
                circuit.is_open = True
                log.warning("熔断器触发: service=%s, failures=%d", service, circuit.failure_count)

    def record_success(self, service: str) -> None:
        with self._lock:
            circuit = self._circuits[service]
            circuit.failure_count = 0
            circuit.is_open = False

    def get_state(self, service: str) -> dict:
        with self._lock:
            circuit = self._circuits[service]
            return {
                "is_open": circuit.is_open,
                "failure_count": circuit.failure_count,
                "last_failure_time": circuit.last_failure_time,
            }


rate_limiter = RateLimiter()
circuit_breaker = CircuitBreaker()


async def rate_limit_middleware(request: Request, call_next):
    client_ip = request.client.host if request.client else "unknown"

    if not rate_limiter.check_ip(client_ip):
        return JSONResponse(
            status_code=429,
            content={"error": "rate_limit_exceeded", "message": "IP rate limit exceeded"},
            headers={
                "X-RateLimit-Limit": str(IP_RATE_LIMIT),
                "X-RateLimit-Window": f"{IP_RATE_WINDOW}s",
                "Retry-After": str(IP_RATE_WINDOW),
            },
        )

    response = await call_next(request)

    ip_remaining = rate_limiter.get_ip_remaining(client_ip)
    response.headers["X-RateLimit-Limit"] = str(IP_RATE_LIMIT)
    response.headers["X-RateLimit-Remaining"] = str(ip_remaining)

    return response
