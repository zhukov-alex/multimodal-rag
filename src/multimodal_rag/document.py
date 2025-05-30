from pydantic import BaseModel, Field


class Chunk(BaseModel):
    chunk_id: int
    content: str


class ScoredChunk(BaseModel):
    chunk: Chunk
    score: float


class ChunkGroup(BaseModel):
    chunks: list[Chunk]
    embedder_name: str
    modality: str  # text, image, etc.


class SourceConfig(BaseModel):
    source_path: str | None = None
    type: str
    loader: str
    tmp_path: str | None = Field(default=None, exclude=True, repr=False)


class MetaConfig(BaseModel):
    filename: str
    size_bytes: int
    last_modified: int
    fingerprint: str
    mime: str


class ScoredItem(BaseModel):
    doc_uuid: str
    chunk_id: int
    content: str
    modality: str
    score: float
    source_path: str
    caption: str | None = None
    image_base64: str | None = None
    metadata: MetaConfig


class Document(BaseModel):
    uuid: str
    content: str
    lang: str
    tags: list[str] = Field(default_factory=list)
    source: SourceConfig
    metadata: MetaConfig
    chunk_groups: list[ChunkGroup] = Field(default_factory=list)

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
