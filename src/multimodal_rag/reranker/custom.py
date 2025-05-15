import os
import aiohttp
from aiohttp import ClientError
from asyncio import TimeoutError

from multimodal_rag.reranker.types import Reranker
from multimodal_rag.utils.retry import backoff
from multimodal_rag.log_config import logger
from multimodal_rag.document import ScoredItem


class CustomReranker(Reranker):
    """
    Reranks scored results using a local server API.
    """

    def __init__(self, model: str, supported_modes: set[str]) -> None:
        self._model_name = model
        self._supported_modes = supported_modes
        self.base_url = os.getenv("CUSTOM_RERANKER_BASE_URL", "http://localhost:5250")

    @property
    def model_name(self) -> str:
        return self._model_name

    def supports(self, mode: str) -> bool:
        return mode in self._supported_modes

    async def process(self, query: str, items: list[ScoredItem]) -> list[ScoredItem]:
        try:
            return await self._rerank(query, items)
        except Exception as e:
            logger.exception("Failed to rerank items", extra={"error": str(e), "url": self.base_url})
            return items

    @backoff(exception=(ClientError, TimeoutError), tries=3, delay=0.5, backoff=2)
    async def _rerank(self, query: str, items: list[ScoredItem]) -> list[ScoredItem]:
        payload_docs = [
            {
                "uuid": item.doc_uuid,
                "modality": item.modality,
                "text": item.content if item.modality == "text" else None,
                "caption": item.caption if item.modality == "image" else None,
                "image": item.image_base64 if item.modality == "image" else None,
            }
            for item in items
            if item.modality != "image" or item.image_base64 is not None
        ]

        payload = {
            "query": query,
            "model_name": self._model_name,
            "documents": payload_docs
        }

        async with aiohttp.ClientSession() as session:
            async with session.post(f"{self.base_url}/rerank", json=payload) as resp:
                resp.raise_for_status()
                result = await resp.json()

        score_map = {entry["uuid"]: entry["score"] for entry in result["results"]}
        for item in items:
            item.score = score_map.get(item.doc_uuid, 0.0)

        logger.debug("Reranked items", extra={"query": query, "scored": len(score_map)})
        return sorted(items, key=lambda i: i.score, reverse=True)
