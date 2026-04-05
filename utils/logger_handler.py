import logging
import time
import functools
from typing import Callable, Any

"""
中间件日志模块 (Middleware Logger)
提供统一的日志记录能力，用于监控 Agent 工具调用链路、耗时与异常。
"""

# 项目统一 Logger（单例）
_logger = None


def get_logger(name: str = "ai_chef") -> logging.Logger:
    """获取项目统一的 Logger 实例"""
    global _logger
    if _logger is not None and name == "ai_chef":
        return _logger

    logger = logging.getLogger(name)
    if not logger.handlers:
        logger.setLevel(logging.DEBUG)
        logger.propagate = False

        handler = logging.StreamHandler()
        handler.setLevel(logging.INFO)
        handler.setFormatter(logging.Formatter(
            "[%(asctime)s] %(levelname)-8s %(message)s",
            datefmt="%m/%d/%y %H:%M:%S"
        ))
        logger.addHandler(handler)

    if name == "ai_chef":
        _logger = logger
    return logger


def log_tool_call(func: Callable) -> Callable:
    """
    装饰器：自动记录工具调用的入参、出参与耗时。

    使用示例:
        @log_tool_call
        def get_fridge_inventory(user_id: str) -> str:
            ...
    """
    logger = get_logger("ai_chef.tools")

    @functools.wraps(func)
    def wrapper(*args, **kwargs) -> Any:
        func_name = func.__name__
        call_args = ", ".join(
            [repr(a) for a in args] + [f"{k}={v!r}" for k, v in kwargs.items()]
        )
        logger.info(f"CALL  -> {func_name}({call_args})")

        start_time = time.time()
        try:
            result = func(*args, **kwargs)
            elapsed = time.time() - start_time
            result_preview = str(result)[:200]
            logger.info(f"OK    <- {func_name} ({elapsed:.2f}s) => {result_preview}")
            return result
        except Exception as e:
            elapsed = time.time() - start_time
            logger.error(f"ERROR <- {func_name} ({elapsed:.2f}s) => {type(e).__name__}: {e}")
            raise

    return wrapper


def log_async_tool_call(func: Callable) -> Callable:
    """装饰器：异步版本的工具调用日志记录"""
    logger = get_logger("ai_chef.tools")

    @functools.wraps(func)
    async def wrapper(*args, **kwargs) -> Any:
        func_name = func.__name__
        call_args = ", ".join(
            [repr(a) for a in args] + [f"{k}={v!r}" for k, v in kwargs.items()]
        )
        logger.info(f"CALL  -> {func_name}({call_args})")

        start_time = time.time()
        try:
            result = await func(*args, **kwargs)
            elapsed = time.time() - start_time
            result_preview = str(result)[:200]
            logger.info(f"OK    <- {func_name} ({elapsed:.2f}s) => {result_preview}")
            return result
        except Exception as e:
            elapsed = time.time() - start_time
            logger.error(f"ERROR <- {func_name} ({elapsed:.2f}s) => {type(e).__name__}: {e}")
            raise

    return wrapper
