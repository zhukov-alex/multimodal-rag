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
    file_reader: str
    parsed_format: str  # text, markdown, json, code, image, blob, etc.
    storage_type: str | None = None  # s3, local, etc.
    asset_uri: str | None = None
    tmp_uri: str | None = Field(default=None, exclude=True, repr=False)

    def get_modality(self) -> str:
        if self.parsed_format == "image":
            return "image"
        elif self.parsed_format == "blob":
            return "blob"
        else:
            return "text"


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
    asset_storage: str | None = None
    asset_uri: str | None = None
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
            "source": self.source.model_dump(),
            "metadata": self.metadata.model_dump(),
            "chunk_groups": [
                {
                    "chunks": [chunk.model_dump() for chunk in group.chunks],
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
