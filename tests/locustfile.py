"""
Locust 性能压测脚本

验收标准:
  - QPS > 100
  - P99 延迟 < 500ms
  - 错误率 < 1%

使用方法:
  locust -f tests/locustfile.py --host http://localhost:8000
"""

from __future__ import annotations

from locust import HttpUser, between, task


class ConsciousnessSeaUser(HttpUser):
    wait_time = between(0.5, 2.0)

    @task(5)
    def query(self):
        self.client.post(
            "/api/v1/query",
            json={"query": "感冒 发热"},
            name="/api/v1/query",
        )

    @task(3)
    def stats(self):
        self.client.get("/api/v1/stats", name="/api/v1/stats")

    @task(2)
    def health(self):
        self.client.get("/health", name="/health")

    @task(1)
    def status(self):
        self.client.get("/status", name="/status")

    @task(1)
    def meta_seeds(self):
        self.client.get("/api/v1/meta-seeds", name="/api/v1/meta-seeds")

    @task(1)
    def cognitive_goals(self):
        self.client.get("/api/v1/cognitive-goals", name="/api/v1/cognitive-goals")

    @task(1)
    def perception_status(self):
        self.client.get("/api/v1/perception/status", name="/api/v1/perception/status")
