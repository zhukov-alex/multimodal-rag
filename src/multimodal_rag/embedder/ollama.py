import os
import requests
import aiohttp
import asyncio
from typing import List
from aiohttp import ClientError
from asyncio import TimeoutError

from multimodal_rag.config.schema import TextEmbeddingConfig
from multimodal_rag.utils.retry import backoff
from multimodal_rag.embedder.types import TextEmbedder
from multimodal_rag.utils.vector import l2_normalize


def get_ollama_models() -> List[str]:
    """
    Returns a list of all available models in the local Ollama environment.
    """
    base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
    response = requests.get(f"{base_url}/api/tags")
    response.raise_for_status()
    data = response.json()
    return [model["name"] for model in data.get("models", [])]


class OllamaEmbedder(TextEmbedder):
    """
    Embedding generator that uses the Ollama local API.
    """

    def __init__(self, config: TextEmbeddingConfig):
        self._config = config
        self.base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
        self._ensure_model_available()

    @property
    def model_name(self) -> str:
        return self._config.model

    def _ensure_model_available(self):
        """
        Checks if the specified model is available in the local Ollama environment.
        """
        try:
            available_models = get_ollama_models()
            if self._config.model not in available_models:
                raise ValueError(f"Model '{self._config.model}' not found in available Ollama models: {available_models}")
        except Exception as e:
            raise RuntimeError(f"Failed to verify Ollama model availability: {e}")

    async def embed_texts(self, texts: List[str]) -> List[List[float]]:
        """
        Embed a list of texts using the Ollama model.
        """
        async with aiohttp.ClientSession() as session:
            tasks = [self._embed_one(session, text) for text in texts]
            results = []
            for coro in asyncio.as_completed(tasks):
                result = await coro
                results.append(result)
            return results

    @backoff(exception=(ClientError, TimeoutError))
    async def _embed_one(self, session: aiohttp.ClientSession, text: str) -> List[float]:
        """
        Sends an embedding request using aiohttp.
        """
        async with session.post(
            f"{self.base_url}/api/embeddings",
            json={"model": self._config.model, "prompt": text}
        ) as response:
            response.raise_for_status()
            data = await response.json()
            embedding = data["embedding"]
            if self._config.normalize:
                embedding = l2_normalize(embedding)
            return embedding
