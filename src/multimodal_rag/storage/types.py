from typing import Protocol, Any
from multimodal_rag.document import Document, ScoredChunk
from pydantic import BaseModel


class AggregateFilter(BaseModel):
    field: str
    value: str


class StorageClient(Protocol):
    async def get_connection(self) -> Any:
        ...

    async def close(self) -> None:
        ...

    async def create_document_collection(self, name: str) -> str:
        ...

    async def create_embedding_collection(
        self, name: str, embedding_model: str, dim: int, distance: str = "cosine"
    ) -> str:
        ...

    async def insert_documents(
        self, documents: list[Document], collection_name: str
    ) -> None:
        ...

    async def insert_chunks(
        self, documents: list[Document], collection_name: str
    ) -> None:
        ...

    async def delete_by_ids(
        self, collection_name: str, field: str, ids: list[str]
    ) -> None:
        ...

    async def aggregate_total_count(
        self, collection_name: str, filter_by: AggregateFilter
    ) -> int:
        ...

    async def query_by_filter(
        self, collection_name: str, filters: dict
    ) -> list[dict]:
        ...

    async def query_by_text(
        self, query: str, filters: dict | None = None
    ) -> list[dict]:
        ...

    async def query_by_vector(
            self, vector: list[float], collection_name: str, filters: dict | None = None, top_k: int = 10
    ) -> list[ScoredChunk]:
        ...

    async def hybrid_chunks(
            self, query: str, vector: list[float], collection_name: str, limit: int, filters: dict | None = None
    ) -> list[ScoredChunk]:
        ...
