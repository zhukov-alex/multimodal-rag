from multimodal_rag.config.schema import (
    TextEmbeddingConfig,
    ImageEmbeddingConfig,
    TranscribingConfig,
    CaptioningConfig,
    StoragingConfig,
    RerankerConfig,
    GenerationConfig,
)
from multimodal_rag.embedder.repltext import ReplicateTextEmbedder
from multimodal_rag.embedder.openai import OpenAIEmbedder
from multimodal_rag.embedder.ollama import OllamaEmbedder
from multimodal_rag.embedder.replimage import ReplicateImageEmbedder
from multimodal_rag.embedder.custom_text import CustomTextEmbedder
from multimodal_rag.embedder.custom_image import CustomImageEmbedder
from multimodal_rag.embedder.types import ImageEmbedder, TextEmbedder
from multimodal_rag.generator.llamacpp import LlamaCppGenerator
from multimodal_rag.generator.ollama import OllamaGenerator
from multimodal_rag.generator.openai import OpenAIGenerator
from multimodal_rag.generator.params.llamacpp import LlamaCppParams
from multimodal_rag.generator.params.ollama import OllamaParams
from multimodal_rag.generator.params.openai import OpenAIParams
from multimodal_rag.generator.types import Generator, LLMQueryParams

from multimodal_rag.preprocessor.transcriber.replicate import ReplicateTranscriber
from multimodal_rag.preprocessor.transcriber.custom import CustomAudioTranscriber
from multimodal_rag.preprocessor.transcriber.types import AudioTranscriber

from multimodal_rag.preprocessor.captioner.custom import CustomImageCaptioner
from multimodal_rag.preprocessor.captioner.types import ImageCaptioner

from multimodal_rag.reranker.custom import CustomReranker
from multimodal_rag.reranker.types import Reranker

from multimodal_rag.storage.weaviate import WeaviateClient
from multimodal_rag.storage.types import StorageClient


TEXT_EMBEDDER_MAPPING = {
    "replicate": ReplicateTextEmbedder,
    "openai": OpenAIEmbedder,
    "ollama": OllamaEmbedder,
    "custom": CustomTextEmbedder,
}

IMAGE_EMBEDDER_MAPPING = {
    "replicate": ReplicateImageEmbedder,
    "custom": CustomImageEmbedder,
}

TRANSCRIBER_MAPPING = {
    "replicate": ReplicateTranscriber,
    "custom": CustomAudioTranscriber,
}

CAPTIONER_MAPPING = {
    "custom": CustomImageCaptioner,
}

STORAGE_CLIENTS = {
    "weaviate": WeaviateClient,
}

GENERATOR_PARAMS_MAPPING = {
    "openai": OpenAIParams,
    "ollama": OllamaParams,
    "llamacpp": LlamaCppParams,
}

GENERATOR_MAPPING = {
    "openai": OpenAIGenerator,
    "ollama": OllamaGenerator,
    "llamacpp": LlamaCppGenerator,
}

RERANKER_MAPPING = {
    "custom": CustomReranker,
}


def create_transcriber(config: TranscribingConfig) -> AudioTranscriber:
    transcriber_class = TRANSCRIBER_MAPPING.get(config.type)
    if not transcriber_class:
        raise ValueError(f"Unknown transcriber type: {config.type}")
    return transcriber_class(model=config.model)


def create_captioner(config: CaptioningConfig) -> ImageCaptioner:
    captioner_class = CAPTIONER_MAPPING.get(config.type)
    if not captioner_class:
        raise ValueError(f"Unknown captioner type: {config.type}")
    return captioner_class(model=config.model)


def create_text_embedder(config: TextEmbeddingConfig) -> TextEmbedder:
    embedder_class = TEXT_EMBEDDER_MAPPING.get(config.type)
    if not embedder_class:
        raise ValueError(f"Unknown text embedder type: {config.type}")
    return embedder_class(model=config.model)


def create_image_embedder(config: ImageEmbeddingConfig) -> ImageEmbedder:
    embedder_class = IMAGE_EMBEDDER_MAPPING.get(config.type)
    if not embedder_class:
        raise ValueError(f"Unknown image embedder type: {config.type}")
    return embedder_class(model=config.model)


def create_storage_client(config: StoragingConfig) -> StorageClient:
    client_class = STORAGE_CLIENTS.get(config.type)
    if not client_class:
        raise ValueError(f"Unknown storage type: {config.type}")
    return client_class(config.get_client_config())


def create_reranker(config: RerankerConfig) -> Reranker:
    reranker_class = RERANKER_MAPPING.get(config.type)
    if not reranker_class:
        raise ValueError(f"Unknown reranker type: {config.type}")
    return reranker_class(model=config.model, supported_modes=config.supported_modes)


def create_generator(config: GenerationConfig) -> Generator:
    generator_class = GENERATOR_MAPPING.get(config.type)
    if not generator_class:
        raise ValueError(f"Unknown generator type: {config.type}")
    return generator_class(model=config.model, context_limit=config.context_limit)


def parse_llm_params(generator_type: str, payload: dict) -> LLMQueryParams:
    cls = GENERATOR_PARAMS_MAPPING.get(generator_type)
    if not cls:
        raise ValueError(f"Unknown generator type: {generator_type}")
    return cls(**payload)
