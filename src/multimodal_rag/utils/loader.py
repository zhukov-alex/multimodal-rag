import io
import os
import base64
import asyncio
from PIL import Image
import aiohttp
from aiohttp import ClientError
from asyncio import TimeoutError
from multimodal_rag.utils.retry import backoff
from multimodal_rag.log_config import logger


@backoff(exception=(ClientError, TimeoutError), tries=3, delay=0.5, backoff=2)
async def load_file_from_url(url: str, timeout: float = 30.0) -> bytes:
    async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=timeout)) as session:
        async with session.get(url) as resp:
            resp.raise_for_status()
            return await resp.read()


async def load_file_from_path(path: str) -> bytes:
    def read_bytes() -> bytes:
        with open(path, "rb") as f:
            return f.read()
    return await asyncio.to_thread(read_bytes)


async def load_file_bytes(path: str, timeout: float = 30.0) -> bytes:
    if path.startswith(("http://", "https://")):
        return await load_file_from_url(path, timeout)
    elif os.path.exists(path):
        return await load_file_from_path(path)
    elif path.startswith("s3://"):
        raise NotImplementedError("S3 loading not implemented yet")
    else:
        raise ValueError(f"Unsupported file source: {path}")


async def load_image_from_source(path: str, timeout: float = 30.0) -> Image.Image:
    data = await load_file_bytes(path, timeout)
    return await asyncio.to_thread(lambda: Image.open(io.BytesIO(data)).convert("RGB"))


async def image_to_bytes(image: Image.Image, format: str = "PNG") -> bytes:
    buffer = io.BytesIO()
    image.save(buffer, format=format)
    return buffer.getvalue()


async def load_image_bytes_from_source(path: str, format: str = "PNG") -> bytes:
    image = await load_image_from_source(path)
    return await asyncio.to_thread(image_to_bytes, image, format)


async def load_image_base64(path: str, format: str = "PNG") -> str | None:
    try:
        image = await load_image_from_source(path)
        image_bytes = await asyncio.to_thread(image_to_bytes, image, format)
        return base64.b64encode(image_bytes).decode()
    except Exception as e:
        logger.warning(f"Failed to encode image from {path}: {e}")
        return None
