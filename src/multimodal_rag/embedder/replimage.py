import aiohttp
import asyncio
import base64
from typing import List

from multimodal_rag.embedder.replmixin import ReplicateClientMixin
from multimodal_rag.embedder.types import ImageEmbedder
from multimodal_rag.utils.retry import backoff
from aiohttp import ClientError
from asyncio import TimeoutError


class ReplicateImageEmbedder(ReplicateClientMixin, ImageEmbedder):
    """
    Embedder for images using Replicate API (via data URI).
    """

    def __init__(self, model_name: str):
        self._model_name = model_name

    @property
    def model_name(self) -> str:
        return self._model_name

    async def embed_images(self, images: List[bytes]) -> List[List[float]]:
        """
        Embed a list of images.
        """
        async with aiohttp.ClientSession() as session:
            tasks = [self._embed_one(session, img) for img in images]
            return await asyncio.gather(*tasks)

    @backoff(exception=(ClientError, TimeoutError))
    async def _embed_one(self, session: aiohttp.ClientSession, img_bytes: bytes) -> List[float]:
        """
        Sends an image to Replicate and get its embedding.
        """
        headers = {
            "Authorization": f"Token {self.replicate_token}",
            "Content-Type": "application/json"
        }

        # Encode image as base64 data URI
        encoded = base64.b64encode(img_bytes).decode("utf-8")
        data_uri = f"data:image/png;base64,{encoded}"

        payload = {
            "version": self._model_name,
            "input": {
                "image": data_uri
            }
        }

        async with session.post(self.replicate_url, headers=headers, json=payload) as response:
            response.raise_for_status()
            data = await response.json()
            return data["output"]

    async def embed_texts(self, texts: List[str]) -> List[List[float]]:
        async with aiohttp.ClientSession() as session:
            tasks = [self._embed_text(session, text) for text in texts]
            return await asyncio.gather(*tasks)

    @backoff(exception=(ClientError, TimeoutError))
    async def _embed_text(self, session: aiohttp.ClientSession, text: str) -> List[float]:
        headers = {
            "Authorization": f"Token {self.replicate_token}",
            "Content-Type": "application/json"
        }

        payload = {
            "version": self._model_name,
            "input": {
                "text": text
            }
        }

        async with session.post(self.replicate_url, headers=headers, json=payload) as response:
            response.raise_for_status()
            data = await response.json()
            return data["output"]
