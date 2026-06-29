"""基础设施层：配置、连接池、会话管理、用户管理、观测、业力清理、参数统计、参数评估

使用 __getattr__ 实现延迟导入，避免与 consciousness_sea.domain.graph_db 的循环依赖。
"""

# config 是基础模块，无外部依赖，可以安全地顶层导入
from .config import DEFAULT_DB_PATH

__all__ = [
    'DEFAULT_DB_PATH',
    'ConnectionPool', 'ConnectionPoolExhausted',
    'SessionManager', 'SessionContext',
    'UserManager',
    'Observer', 'StatusData', 'SeedRankItem', 'KarmaRankItem', 'QueryRecord',
    'KarmaCleaner',
    'ensure_param_stats_table', 'record_param_stats',
    'ParamEvaluator',
]


def __getattr__(name: str):
    """延迟导入：避免 infrastructure 子模块加载时触发循环依赖"""
    _lazy = {
        'ConnectionPool': '.connection_pool',
        'ConnectionPoolExhausted': '.connection_pool',
        'SessionManager': '.session_manager',
        'SessionContext': '.session_manager',
        'UserManager': '.user_manager',
        'Observer': '.observer',
        'StatusData': '.observer',
        'SeedRankItem': '.observer',
        'KarmaRankItem': '.observer',
        'QueryRecord': '.observer',
        'KarmaCleaner': '.karma_cleaner',
        'ensure_param_stats_table': '.param_stats',
        'record_param_stats': '.param_stats',
        'ParamEvaluator': '.param_evaluator',
    }
    if name in _lazy:
        import importlib
        module = importlib.import_module(_lazy[name], __name__)
        return getattr(module, name)
    raise AttributeError(f'module {__name__!r} has no attribute {name!r}')
