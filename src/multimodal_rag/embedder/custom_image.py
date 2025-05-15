import os
import aiohttp
import asyncio
from typing import List
from multimodal_rag.embedder.types import ImageEmbedder
from multimodal_rag.utils.retry import backoff
from aiohttp import ClientError
from asyncio import TimeoutError


class CustomImageEmbedder(ImageEmbedder):
    """
    Embedding generator for images using a local server API.
    """

    def __init__(self, model: str):
        self._model_name = model
        self.base_url = os.getenv("CUSTOM_IMG_EMBEDDER_URL", "http://localhost:5600")

    @property
    def model_name(self) -> str:
        return self._model_name

    async def embed_images(self, images: List[bytes]) -> List[List[float]]:
        """
        Embed a list of images.
        Expects each image as raw bytes.
        """
        async with aiohttp.ClientSession() as session:
            tasks = [self._embed_one(session, img) for img in images]
            return await asyncio.gather(*tasks)

    @backoff(exception=(ClientError, TimeoutError), tries=3, delay=0.5, backoff=2)
    async def _embed_one(self, session: aiohttp.ClientSession, img_bytes: bytes) -> List[float]:
        """
        Sends an image embedding request.
        """
        url = f"{self.base_url}/embed"

        data = aiohttp.FormData()
        data.add_field("file", img_bytes, filename="image.png", content_type="image/png")
        data.add_field("model_name", self._model_name)

        async with session.post(url, data=data) as response:
            response.raise_for_status()
            result = await response.json()
            return result["embedding"]

    async def embed_texts(self, texts: List[str]) -> List[List[float]]:
        async with aiohttp.ClientSession() as session:
            tasks = [self._embed_text(session, text) for text in texts]
            return await asyncio.gather(*tasks)

    @backoff(exception=(ClientError, TimeoutError), tries=3, delay=0.5, backoff=2)
    async def _embed_text(self, session: aiohttp.ClientSession, text: str) -> List[float]:
        url = f"{self.base_url}/embed-text"

        data = {"text": text, "model_name": self._model_name}

        async with session.post(url, json=data) as response:
            response.raise_for_status()
            result = await response.json()
            return result["embedding"]
