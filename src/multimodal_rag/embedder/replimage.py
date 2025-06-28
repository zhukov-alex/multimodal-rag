import aiohttp
import asyncio

from multimodal_rag.config.schema import ImageEmbeddingConfig
from multimodal_rag.embedder.replmixin import ReplicateClientMixin
from multimodal_rag.embedder.types import ImageEmbedder
from multimodal_rag.utils.retry import backoff
from multimodal_rag.utils.vector import l2_normalize
from aiohttp import ClientError
from asyncio import TimeoutError


class ReplicateImageEmbedder(ReplicateClientMixin, ImageEmbedder):
    """
    Embedder for images using Replicate API (via data URI).
    """

    def __init__(self, config: ImageEmbeddingConfig):
        self._config = config

    @property
    def model_name(self) -> str:
        return self._config.model

    async def embed_images(self, images: list[str]) -> list[list[float]]:
        """
        Embed a list of images.
        """
        async with aiohttp.ClientSession() as session:
            tasks = [self._embed_one(session, img) for img in images]
            return await asyncio.gather(*tasks)


    @backoff(exception=(ClientError, TimeoutError))
    async def _embed_one(self, session: aiohttp.ClientSession, img_b64: str) -> list[float]:
        """
        Sends an image to Replicate and gets its embedding.
        """
        headers = {
            "Authorization": f"Token {self.replicate_token}",
            "Content-Type": "application/json"
        }

        if not img_b64.startswith("data:image/"):
            img_b64 = f"data:image/png;base64,{img_b64}"

        payload = {
            "version": self._config.model,
            "input": {
                "image": img_b64
            }
        }

        async with session.post(self.replicate_url, headers=headers, json=payload) as response:
            response.raise_for_status()
            data = await response.json()
            return data["output"]

    async def embed_texts(self, texts: list[str]) -> list[list[float]]:
        async with aiohttp.ClientSession() as session:
            tasks = [self._embed_text(session, text) for text in texts]
            return await asyncio.gather(*tasks)

    @backoff(exception=(ClientError, TimeoutError))
    async def _embed_text(self, session: aiohttp.ClientSession, text: str) -> list[float]:
        headers = {
            "Authorization": f"Token {self.replicate_token}",
            "Content-Type": "application/json"
        }

        payload = {
            "version": self._config.model,
            "input": {
                "text": text
            }
        }

        async with session.post(self.replicate_url, headers=headers, json=payload) as response:
            response.raise_for_status()
            data = await response.json()
            embedding = data["output"]
            if self._config.normalize:
                embedding = l2_normalize(embedding)
            return embedding
