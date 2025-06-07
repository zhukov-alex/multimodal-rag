import io
import base64
import asyncio
from pathlib import Path
from PIL import Image


async def load_image_bytes(path: str, format: str = "PNG") -> bytes:
    image = await load_file(path)
    return await asyncio.to_thread(image_bytes_to_base64, image, format)


def image_bytes_to_base64(data: bytes, format: str = "PNG") -> str:
    image = Image.open(io.BytesIO(data)).convert("RGB")
    buffer = io.BytesIO()
    image.save(buffer, format=format)
    return base64.b64encode(buffer.getvalue()).decode()


async def load_file(path: str) -> bytes:
    path = Path(path.removeprefix("file://"))
    if not path.exists():
        raise FileNotFoundError(f"File not found at {path}")
    return await asyncio.to_thread(path.read_bytes)
