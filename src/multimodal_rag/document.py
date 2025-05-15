from pydantic import BaseModel, Field
from typing import List, Optional


class Chunk(BaseModel):
    chunk_id: int
    content: str


class ScoredChunk(BaseModel):
    chunk: Chunk
    score: float


class ChunkGroup(BaseModel):
    chunks: List[Chunk]
    embedder_name: str
    modality: str  # text, image, etc.


class SourceConfig(BaseModel):
    source_path: str
    type: str
    loader: str


class MetaConfig(BaseModel):
    filename: Optional[str] = None
    size_bytes: Optional[int] = None
    last_modified: Optional[int] = None
    fingerprint: Optional[str] = None
    mime: Optional[str] = None


class ScoredItem(BaseModel):
    doc_uuid: str
    chunk_id: int
    content: str
    modality: str
    score: float
    source_path: str
    caption: Optional[str] = None
    image_base64: Optional[str] = None
    metadata: MetaConfig


class Document(BaseModel):
    uuid: str
    content: str
    lang: str
    tags: List[str] = Field(default_factory=list)
    source: SourceConfig
    metadata: MetaConfig
    chunk_groups: List[ChunkGroup] = Field(default_factory=list)

    def to_json(self) -> dict:
        return {
            "uuid": self.uuid,
            "tags": self.tags,
            "lang": self.lang,
            "content": self.content,
            "source": self.source.dict(),
            "metadata": self.metadata.dict(),
            "chunk_groups": [
                {
                    "chunks": [chunk.dict() for chunk in group.chunks],
                    "embedder_name": group.embedder_name,
                    "modality": group.modality,
                }
                for group in self.chunk_groups
            ],
        }

    @classmethod
    def from_json(cls, data: dict) -> "Document":
        return cls(
            uuid=data["uuid"],
            tags=data.get("tags", []),
            lang=data.get("lang", ""),
            content=data.get("content", ""),
            source=SourceConfig(**data["source"]),
            metadata=MetaConfig(**data["metadata"]),
            chunk_groups=[
                ChunkGroup(
                    chunks=[Chunk(**chunk_data) for chunk_data in group_data.get("chunks", [])],
                    embedder_name=group_data.get("embedder_name", ""),
                    modality=group_data.get("modality", "")
                )
                for group_data in data.get("chunk_groups", [])
            ],
        )
