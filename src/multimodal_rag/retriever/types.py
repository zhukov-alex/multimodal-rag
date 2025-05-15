from typing import Literal
from pydantic import BaseModel


class SearchByText(BaseModel):
    query: str
    search_type: Literal["embedding", "hybrid"] = "embedding"
    rerank: Literal["text", "images"] | None = None
    modality_top_k: dict[str, int]
    project_id: str
    filters: dict | None = None


class SearchByImage(BaseModel):
    blob: bytes
    caption: str | None = None
    rerank: Literal["text", "images"] | None = None
    top_k: int = 10
    project_id: str
    filters: dict | None = None
