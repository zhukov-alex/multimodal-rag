from pydantic import BaseModel
from typing import Literal, Any


class ChunkingConfig(BaseModel):
    token_chunker: dict[str, Any] | None = None
    sentence_chunker: dict[str, Any] | None = None
    markdown_chunker: dict[str, Any] | None = None
    json_chunker: dict[str, Any] | None = None
    code_chunker: dict[str, Any] | None = None
    recursive_chunker: dict[str, Any] | None = None
    content_type_to_chunker: dict[str, str]


class TextEmbeddingConfig(BaseModel):
    type: Literal["replicate", "custom"]
    model: str
    normalize: bool | None = True


class ImageEmbeddingConfig(BaseModel):
    type: Literal["replicate", "custom"]
    model: str
    input_size: int | None = 224


class EmbeddingConfig(BaseModel):
    text: TextEmbeddingConfig
    image: ImageEmbeddingConfig | None = None
    batch_size: int | None = 100


class WeaviateConnectionConfig(BaseModel):
    deployment: Literal["cloud", "local", "embedded"]
    url: str | None = None
    api_key: str | None = None
    port: int | None = 8080
    secure: bool | None = True
    dimension: int | None = None
    distance_metric: str | None = "cosine"


class StoragingConfig(BaseModel):
    type: Literal["weaviate", "faiss"]
    weaviate: WeaviateConnectionConfig | None = None


class S3AssetConfig(BaseModel):
    bucket: str
    region: str
    overwrite: bool = False
    endpoint_url: str | None = None


class LocalAssetConfig(BaseModel):
    root_dir: str
    overwrite: bool = False


class AssetStoreConfig(BaseModel):
    type: Literal["s3", "local"]
    s3: S3AssetConfig | None = None
    local: LocalAssetConfig | None = None


class TranscribingConfig(BaseModel):
    type: Literal["replicate", "custom"]
    model: str


class CaptioningConfig(BaseModel):
    type: Literal["custom"]
    model: str


class IndexingConfig(BaseModel):
    chunking: ChunkingConfig
    embedding: EmbeddingConfig
    transcribing: TranscribingConfig | None = None
    captioning: CaptioningConfig | None = None
    storaging: StoragingConfig
    asset_store: AssetStoreConfig | None = None


# --- RAG (retrieve + generate) config ---


class RerankerConfig(BaseModel):
    type: Literal["custom", "openai"]
    model: str
    supported_modes: set[str]


class GenerationConfig(BaseModel):
    type: Literal["openai", "ollama"]
    model: str
    context_limit: int | None = None


class RAGConfig(BaseModel):
    embedding: EmbeddingConfig
    storaging: StoragingConfig
    generation: GenerationConfig
    reranking: RerankerConfig | None = None
