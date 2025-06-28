import os
import aiohttp
import asyncio

from multimodal_rag.config.schema import ImageEmbeddingConfig
from multimodal_rag.embedder.types import ImageEmbedder
from multimodal_rag.utils.vector import l2_normalize
from multimodal_rag.utils.retry import backoff
from aiohttp import ClientError
from asyncio import TimeoutError


class CustomImageEmbedder(ImageEmbedder):
    """
    Embedding generator for images using a local server API.
    """

    def __init__(self, config: ImageEmbeddingConfig):
        self._config = config
        self.base_url = os.getenv("CUSTOM_IMG_EMBEDDER_URL", "http://localhost:5600")

    @property
    def model_name(self) -> str:
        return self._config.model

    async def embed_images(self, images: list[str]) -> list[list[float]]:
        """
        Embed a list of images.
        Expects each image as raw bytes.
        """
        async with aiohttp.ClientSession() as session:
            tasks = [self._embed_one(session, img) for img in images]
            return await asyncio.gather(*tasks)

    @backoff(exception=(ClientError, TimeoutError), tries=3, delay=0.5, backoff=2)
    async def _embed_one(self, session: aiohttp.ClientSession, img_b64: str) -> list[float]:
        url = f"{self.base_url}/embed"

        if not img_b64.startswith("data:image/"):
            img_b64 = f"data:image/png;base64,{img_b64}"

        data = {
            "image_base64": img_b64,
            "model_name": self._config.model
        }

        async with session.post(url, json=data) as response:
            response.raise_for_status()
            return (await response.json())["embedding"]

    async def embed_texts(self, texts: list[str]) -> list[list[float]]:
        async with aiohttp.ClientSession() as session:
            tasks = [self._embed_text(session, text) for text in texts]
            return await asyncio.gather(*tasks)

    @backoff(exception=(ClientError, TimeoutError), tries=3, delay=0.5, backoff=2)
    async def _embed_text(self, session: aiohttp.ClientSession, text: str) -> list[float]:
        url = f"{self.base_url}/embed-text"

        data = {"text": text, "model_name": self._config.model}

        async with session.post(url, json=data) as response:
            response.raise_for_status()
            result = await response.json()
            embedding = result["embedding"]
            if self._config.normalize:
                embedding = l2_normalize(embedding)
            return embedding
