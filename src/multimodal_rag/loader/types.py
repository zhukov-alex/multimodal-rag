from typing import Protocol, AsyncIterator, NamedTuple
from multimodal_rag.document import Document


class LoadResult(NamedTuple):
    documents: list[Document]
    next_sources: list[str] = []


class DocumentLoader(Protocol):
    """
    Interface for document loaders.
    """

    async def load(self, source: str, filter: str | None = None) -> LoadResult:
        """
        Load documents from the source and return them along with any additional sources to process.
        """
        ...
