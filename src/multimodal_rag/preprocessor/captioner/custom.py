import os
import aiohttp
import asyncio
from typing import List
from multimodal_rag.preprocessor.captioner.types import ImageCaptioner
from multimodal_rag.utils.retry import backoff
from aiohttp import ClientError
from asyncio import TimeoutError


class CustomImageCaptioner(ImageCaptioner):
    """
    Caption generations for images using a local server API.
    """

    def __init__(self, model: str):
        self._model_name = model
        self.base_url = os.getenv("CUSTOM_CAPTIONER_BASE_URL", "http://localhost:5150")

    @property
    def model_name(self) -> str:
        return self._model_name

    async def generate_captions(self, images: list[bytes]) -> list[str]:
        async with aiohttp.ClientSession() as session:
            tasks = [self._caption_one(session, img) for img in images]
            return await asyncio.gather(*tasks)

    @backoff(exception=(ClientError, TimeoutError), tries=3, delay=0.5, backoff=2)
    async def _caption_one(self, session: aiohttp.ClientSession, img_bytes: bytes) -> str:
        url = f"{self.base_url}/caption"

        data = aiohttp.FormData()
        data.add_field("file", img_bytes, filename="image.png", content_type="image/png")
        data.add_field("model_name", self._model_name)

        async with session.post(url, data=data) as response:
            response.raise_for_status()
            result = await response.json()
            return result["caption"]
