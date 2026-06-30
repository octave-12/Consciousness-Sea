#!/usr/bin/env python3
"""
识海全链路测试脚本

用法:
  python scripts/test_e2e.py              # API 模式（启动服务后查询）
  python scripts/test_e2e.py --cli        # CLI 模式（直接查询，无需启动服务）
  python scripts/test_e2e.py --host 0.0.0.0 --port 9000  # 自定义地址
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
import urllib.request
import urllib.error
import argparse
import signal
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
BACKEND_SRC = PROJECT_ROOT / "backend" / "src"
PYTHON = sys.executable

QUERIES = [
    "感冒了吃什么",
    "量子力学是什么",
    "如何学习Python编程",
    "什么是人工智能",
    "太阳系有哪些行星",
    "高血压怎么预防",
    "什么是区块链",
    "如何提高记忆力",
    "什么是相对论",
    "怎样保持健康",
]

API_HOST = "127.0.0.1"
API_PORT = 8111
STARTUP_TIMEOUT = 120
QUERY_TIMEOUT = 60


def _check_health(host: str, port: int) -> bool:
    try:
        r = urllib.request.urlopen(f"http://{host}:{port}/health", timeout=3)
        return r.status == 200
    except Exception:
        return False


def _api_query(host: str, port: int, text: str) -> dict:
    data = json.dumps({"query": text}).encode("utf-8")
    req = urllib.request.Request(
        f"http://{host}:{port}/api/v1/query",
        data=data,
        headers={"Content-Type": "application/json"},
    )
    resp = urllib.request.urlopen(req, timeout=QUERY_TIMEOUT)
    return json.loads(resp.read().decode("utf-8"))


def run_api_mode(host: str, port: int) -> int:
    env = os.environ.copy()
    env["PYTHONPATH"] = str(BACKEND_SRC)

    proc = subprocess.Popen(
        [PYTHON, "-m", "consciousness_sea.interfaces.api"],
        cwd=str(PROJECT_ROOT),
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )

    def _cleanup(*_):
        proc.terminate()
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()
        sys.exit(1)

    signal.signal(signal.SIGINT, _cleanup)
    signal.signal(signal.SIGTERM, _cleanup)

    print(f"正在启动识海 API 服务 (PID={proc.pid})...")
    for i in range(STARTUP_TIMEOUT):
        time.sleep(1)
        if proc.poll() is not None:
            stderr = proc.stderr.read().decode("utf-8", errors="replace")[:3000]
            print(f"服务启动失败 (returncode={proc.returncode})")
            print(stderr)
            return 1
        if _check_health(host, port):
            print(f"服务已启动! (耗时 {i+1}s) http://{host}:{port}")
            break
    else:
        print(f"服务启动超时 ({STARTUP_TIMEOUT}s)")
        proc.terminate()
        return 1

    print("=" * 70)
    fail_count = 0

    for idx, q in enumerate(QUERIES, 1):
        try:
            result = _api_query(host, port, q)
            expert_answer = result.get("expert_answer")
            answer = expert_answer if expert_answer else "（检索式回答，无专家回答）"
            confidence = result.get("confidence", 0)
            domains = result.get("selected_domains", [])
            matched = result.get("matched_seeds", 0)
            activated = result.get("total_activated", 0)
            decision = result.get("decision", "")
            print(f"\n【{idx}】用户: {q}")
            print(f"    识海: {answer}")
            print(f"    置信度: {confidence:.2f} | 决策: {decision} | 匹配: {matched} | 激活: {activated} | 领域: {domains}")
        except urllib.error.HTTPError as e:
            body = e.read().decode("utf-8", errors="replace")[:200]
            print(f"\n【{idx}】用户: {q}")
            print(f"    识海: [HTTP {e.code}] {body}")
            fail_count += 1
        except Exception as e:
            print(f"\n【{idx}】用户: {q}")
            print(f"    识海: [错误] {e}")
            fail_count += 1

    proc.terminate()
    try:
        proc.wait(timeout=10)
    except subprocess.TimeoutExpired:
        proc.kill()

    print(f"\n{'=' * 70}")
    print(f"测试完成: {len(QUERIES) - fail_count}/{len(QUERIES)} 成功")
    return 1 if fail_count else 0


def run_cli_mode() -> int:
    sys.path.insert(0, str(BACKEND_SRC))

    from consciousness_sea import GraphDB, route, answer_from_activation, verify, apply_karma
    from consciousness_sea.infrastructure.config import DEFAULT_DB_PATH

    db = GraphDB(DEFAULT_DB_PATH)
    db.connect()
    fail_count = 0

    try:
        s = db.stats()
        print(f"数据库: 节点={s['nodes']:,}, 边={s['edges']:,}")
        print("=" * 70)

        for idx, q in enumerate(QUERIES, 1):
            try:
                result = route(q, db)
                answer = answer_from_activation(result, db)
                verdict = verify(answer, result, db)
                domains = result.selected_domains or ["常识"]
                n_karma = apply_karma(result, db, verdict["karma_direction"], dry_run=True)
                print(f"\n【{idx}】用户: {q}")
                print(f"    识海: {answer}")
                print(f"    置信度: {verdict['confidence']:.2f} | 决策: {verdict['decision']} | "
                      f"匹配: {len(result.seed_matches)} | 激活: {len(result.activated)} | 领域: {domains}")
            except Exception as e:
                print(f"\n【{idx}】用户: {q}")
                print(f"    识海: [错误] {e}")
                fail_count += 1
    finally:
        db.close()

    print(f"\n{'=' * 70}")
    print(f"测试完成: {len(QUERIES) - fail_count}/{len(QUERIES)} 成功")
    return 1 if fail_count else 0


def main() -> int:
    parser = argparse.ArgumentParser(description="识海全链路测试")
    parser.add_argument("--cli", action="store_true", help="使用 CLI 模式（不启动 HTTP 服务）")
    parser.add_argument("--host", default=API_HOST, help=f"API 监听地址 (默认: {API_HOST})")
    parser.add_argument("--port", type=int, default=API_PORT, help=f"API 监听端口 (默认: {API_PORT})")
    args = parser.parse_args()

    if args.cli:
        return run_cli_mode()
    return run_api_mode(args.host, args.port)


if __name__ == "__main__":
    sys.exit(main())