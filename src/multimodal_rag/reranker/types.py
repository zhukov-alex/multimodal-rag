from typing import Protocol
from multimodal_rag.document import ScoredItem


class Reranker(Protocol):
    """
    Interface for documents reranking API.
    """

    def supports(self, mode: str) -> bool:
        """Check if the reranker supports the given modality."""
        ...

    def process(self, query: str, items: list[ScoredItem]) -> list[ScoredItem]:
        """Sort documents by relevance to the query."""
        ...
