import aiohttp
import asyncio

from multimodal_rag.config.schema import TextEmbeddingConfig
from multimodal_rag.embedder.replmixin import ReplicateClientMixin
from multimodal_rag.embedder.types import TextEmbedder
from multimodal_rag.utils.vector import l2_normalize
from multimodal_rag.utils.retry import backoff
from aiohttp import ClientError
from asyncio import TimeoutError


class ReplicateTextEmbedder(ReplicateClientMixin, TextEmbedder):
    """
    Embedder for texts using Replicate text embedding models.
    """

    def __init__(self, config: TextEmbeddingConfig):
        self._config = config

    @property
    def model_name(self) -> str:
        return self._config.model

    async def embed_texts(self, texts: list[str]) -> list[list[float]]:
        async with aiohttp.ClientSession() as session:
            tasks = [self._embed_one(session, text) for text in texts]
            return await asyncio.gather(*tasks)

    @backoff(exception=(ClientError, TimeoutError))
    async def _embed_one(self, session: aiohttp.ClientSession, text: str) -> list[float]:
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
