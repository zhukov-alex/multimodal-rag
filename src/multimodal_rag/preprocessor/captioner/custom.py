import os
import aiohttp
import asyncio
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

    async def generate_captions(self, images: list[str]) -> list[str]:
        async with aiohttp.ClientSession() as session:
            tasks = [self._caption_one(session, img_b64) for img_b64 in images]
            return await asyncio.gather(*tasks)

    @backoff(exception=(ClientError, TimeoutError), tries=3, delay=0.5, backoff=2)
    async def _caption_one(self, session: aiohttp.ClientSession, img_b64: str) -> str:
        url = f"{self.base_url}/caption"

        if not img_b64.startswith("data:image/"):
            img_b64 = f"data:image/png;base64,{img_b64}"

        data = {
            "image_base64": img_b64,
            "model_name": self._model_name
        }

        async with session.post(url, json=data) as response:
            response.raise_for_status()
            result = await response.json()
            return result["caption"]
