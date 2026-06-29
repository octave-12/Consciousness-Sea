#!/usr/bin/env python3
"""
识海 CLI — Phase 0 命令行入口

用法:
  python cli.py query "感冒了吃什么"
  python cli.py query "量子力学是什么" --user user_lzk
  python cli.py stats
  python cli.py server    # Phase 1: 启动 HTTP API
"""

import sys
import pathlib
import argparse
import logging

# 添加 backend/src 到 path
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent / "src"))

from consciousness_sea import GraphDB, route, answer_from_activation, answer_as_dict, verify, apply_karma
from consciousness_sea.infrastructure.config import DEFAULT_DB_PATH, API_HOST, API_PORT

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s'
)
log = logging.getLogger('cli')


def cmd_query(args):
    """执行一次查询"""
    db_path = args.db or DEFAULT_DB_PATH

    with GraphDB(db_path) as graph:
        # 1. 路由
        log.info(f"查询: {args.query}")
        result = route(args.query, graph, user_label=args.user)

        # 2. 回答
        if args.format == 'json':
            import json
            output = answer_as_dict(result)
            print(json.dumps(output, ensure_ascii=False, indent=2))
        else:
            answer = answer_from_activation(result, graph)
            print(answer)

        # 3. 校验
        answer_text = answer if args.format == 'text' else str(answer_as_dict(result))
        verdict = verify(answer_text, result, graph)
        log.info(f"校验: confidence={verdict['confidence']}, "
                 f"decision={verdict['decision']}, "
                 f"keywords={verdict['matched_keywords']}/{verdict['total_keywords']}")

        # 4. 熏习
        if not args.dry_run:
            n = apply_karma(result, graph, verdict['karma_direction'])
            log.info(f"熏习: {n} 条边, direction={verdict['karma_direction']:+d}")
        else:
            n = apply_karma(result, graph, verdict['karma_direction'], dry_run=True)
            log.info(f"熏习(dry-run): {n} 条边, direction={verdict['karma_direction']:+d}")

        # 5. 统计
        print(f"\n--- 激活了 {len(result.activated)} 个种子, "
              f"匹配 {len(result.seed_matches)} 个, "
              f"领域: {result.selected_domains or ['常识']}")


def cmd_stats(args):
    """查看数据库统计"""
    db_path = args.db or DEFAULT_DB_PATH
    with GraphDB(db_path) as graph:
        s = graph.stats()
        size_mb = pathlib.Path(db_path).stat().st_size / (1024**2) if pathlib.Path(db_path).exists() else 0.0
        print(f"数据库: {db_path}")
        print(f"大小:   {size_mb:.0f} MB")
        print(f"节点:   {s['nodes']:,}")
        print(f"边:     {s['edges']:,}")

        # 边的关系类型分布
        rows = graph.conn.execute(
            "SELECT relation, COUNT(*) as cnt FROM karma_edges "
            "GROUP BY relation ORDER BY cnt DESC"
        ).fetchall()
        if rows:
            print(f"\n关系类型分布:")
            for r in rows:
                print(f"  {r['relation']:25s}: {r['cnt']:>10,}")


def cmd_server(args):
    """启动 HTTP API 服务"""
    import uvicorn
    host = args.host
    port = args.port
    log.info(f"启动识海 API 服务: http://{host}:{port}")
    uvicorn.run("consciousness_sea.interfaces.api:app", host=host, port=port, workers=1)


def main():
    parser = argparse.ArgumentParser(description='识海 CLI')
    sub = parser.add_subparsers(dest='command')

    # query
    q = sub.add_parser('query', help='执行查询')
    q.add_argument('query', help='查询文本')
    q.add_argument('--user', '-u', help='用户种子 label')
    q.add_argument('--db', help='数据库路径')
    q.add_argument('--format', choices=['text', 'json'], default='text')
    q.add_argument('--dry-run', action='store_true', help='不写回业力')
    q.set_defaults(func=cmd_query)

    # stats
    s = sub.add_parser('stats', help='数据库统计')
    s.add_argument('--db', help='数据库路径')
    s.set_defaults(func=cmd_stats)

    # server
    sv = sub.add_parser('server', help='启动 HTTP API 服务')
    sv.add_argument('--host', default=API_HOST, help=f'监听地址 (默认: {API_HOST})')
    sv.add_argument('--port', type=int, default=API_PORT, help=f'监听端口 (默认: {API_PORT})')
    sv.set_defaults(func=cmd_server)

    args = parser.parse_args()
    if not args.command:
        parser.print_help()
        return
    args.func(args)


if __name__ == '__main__':
    main()