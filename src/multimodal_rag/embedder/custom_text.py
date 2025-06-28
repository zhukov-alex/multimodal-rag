import os
import aiohttp
import asyncio
from aiohttp import ClientError
from asyncio import TimeoutError

from multimodal_rag.config.schema import TextEmbeddingConfig
from multimodal_rag.embedder.types import TextEmbedder
from multimodal_rag.utils.retry import backoff


class CustomTextEmbedder(TextEmbedder):
    """
    Embedding generator for texts using a local server API.
    """

    def __init__(self, config: TextEmbeddingConfig):
        self._config = config
        self.base_url = os.getenv("CUSTOM_TEXT_EMBEDDER_URL", "http://localhost:5500")

    @property
    def model_name(self) -> str:
        return self._config.model

    async def embed_texts(self, texts: list[str]) -> list[list[float]]:
        """
        Embed a list of texts.
        """
        async with aiohttp.ClientSession() as session:
            tasks = [self._embed_one(session, text) for text in texts]
            return await asyncio.gather(*tasks)

    @backoff(exception=(ClientError, TimeoutError), tries=3, delay=0.5, backoff=2)
    async def _embed_one(self, session: aiohttp.ClientSession, text: str) -> list[float]:
        """
        Sends a text to the embedding API.
        """
        payload = {
            "text": text,
            "model": self._config.model,
            "normalize": self._config.normalize,
        }

        async with session.post(f"{self.base_url}/embed", json=payload) as response:
            response.raise_for_status()
            result = await response.json()
            return result["embedding"]
