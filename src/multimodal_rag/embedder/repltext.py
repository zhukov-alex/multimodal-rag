import aiohttp
import asyncio
from typing import List

from multimodal_rag.embedder.replmixin import ReplicateClientMixin
from multimodal_rag.embedder.types import TextEmbedder
from multimodal_rag.utils.retry import backoff
from aiohttp import ClientError
from asyncio import TimeoutError


class ReplicateTextEmbedder(ReplicateClientMixin, TextEmbedder):
    """
    Embedder for texts using Replicate text embedding models.
    """

    def __init__(self, model_name: str):
        self._model_name = model_name

    @property
    def model_name(self) -> str:
        return self._model_name

    async def embed_texts(self, texts: List[str]) -> List[List[float]]:
        async with aiohttp.ClientSession() as session:
            tasks = [self._embed_one(session, text) for text in texts]
            return await asyncio.gather(*tasks)

    @backoff(exception=(ClientError, TimeoutError))
    async def _embed_one(self, session: aiohttp.ClientSession, text: str) -> List[float]:
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
