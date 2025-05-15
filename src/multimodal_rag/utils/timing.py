import time
from contextlib import asynccontextmanager
from multimodal_rag.log_config import logger

@asynccontextmanager
async def log_duration(label: str, **extra):
    start = time.perf_counter()
    try:
        yield
    finally:
        duration = time.perf_counter() - start
        logger.info(f"{label} completed", extra={**extra, "duration": round(duration, 3), "step": label})
