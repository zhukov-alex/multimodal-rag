import os
import aiohttp
import asyncio
from typing import List
from aiohttp import ClientError
from asyncio import TimeoutError

from multimodal_rag.config.schema import TextEmbeddingConfig
from multimodal_rag.utils.retry import backoff
from multimodal_rag.embedder.types import TextEmbedder
from multimodal_rag.utils.vector import l2_normalize


async def get_openai_models(embedding_only: bool = False) -> list[str]:
    """
    Returns a list of available OpenAI model IDs.
    """
    api_key = os.getenv("OPENAI_API_KEY")
    base_url = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")
    if not api_key:
        raise ValueError("Missing OPENAI_API_KEY in environment.")

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }

    async with aiohttp.ClientSession(headers=headers) as session:
        async with session.get(f"{base_url}/models") as response:
            response.raise_for_status()
            data = await response.json()
            model_ids = [model["id"] for model in data.get("data", [])]
            if embedding_only:
                model_ids = [m for m in model_ids if "embedding" in m]
            return model_ids


class OpenAIEmbedder(TextEmbedder):
    """
    Embedding generator using the OpenAI Embedding API.
    """

    def __init__(self, config: TextEmbeddingConfig):
        self._config = config
        self.api_key = os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError("Missing OPENAI_API_KEY in environment.")
        self.base_url = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")

    @property
    def model_name(self) -> str:
        return self._config.model

    async def embed_texts(self, texts: list[str]) -> list[list[float]]:
        """
        Embed a list of texts using OpenAI.
        """
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        async with aiohttp.ClientSession(headers=headers) as session:
            tasks = [self._embed_one(session, text) for text in texts]
            results = []
            for coro in asyncio.as_completed(tasks):
                result = await coro
                results.append(result)
            return results

    @backoff(exception=(ClientError, TimeoutError))
    async def _embed_one(self, session: aiohttp.ClientSession, text: str) -> list[float]:
        """
        Sends an embedding request to OpenAI.
        """
        payload = {"model": self._config.model, "input": text}
        async with session.post(f"{self.base_url}/embeddings", json=payload) as response:
            response.raise_for_status()
            data = await response.json()
            embedding = data["data"][0]["embedding"]
            if self._config.normalize:
                embedding = l2_normalize(embedding)
            return embedding
