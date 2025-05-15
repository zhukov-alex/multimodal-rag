import asyncio
import functools
from typing import Callable
from multimodal_rag.log_config import logger

DEFAULT_RETRY_ATTEMPTS = 3


def backoff(
    exception: tuple[type[BaseException], ...],
    tries: int = DEFAULT_RETRY_ATTEMPTS,
    delay: float = 1.0,
    backoff: float = 2.0,
):
    def decorator(fn: Callable):
        @functools.wraps(fn)
        async def wrapper(*args, **kwargs):
            current_try = 0
            current_delay = delay
            while True:
                try:
                    return await fn(*args, **kwargs)
                except exception as e:
                    current_try += 1
                    if current_try >= tries:
                        cls_name = type(args[0]).__name__ if args else None
                        logger.debug("Max retries exceeded", extra={
                            "class": cls_name,
                            "function": fn.__name__,
                            "exception": str(e),
                            "tries": current_try
                        })
                        raise
                    logger.warning(f"Retry #{current_try} for {fn.__name__} in {current_delay:.2f}s: {type(e).__name__}")
                    await asyncio.sleep(current_delay)
                    current_delay *= backoff
        return wrapper
    return decorator
