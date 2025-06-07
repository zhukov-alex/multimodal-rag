import importlib.util
import sys

from typing import Callable, Any

from multimodal_rag.config.schema import (
    TextEmbeddingConfig,
    ImageEmbeddingConfig,
    TranscribingConfig,
    CaptioningConfig,
    StoragingConfig,
    AssetStoreConfig,
    RerankerConfig,
    GenerationConfig,
)

from multimodal_rag.embedder.types import ImageEmbedder, TextEmbedder
from multimodal_rag.generator.types import Generator, LLMQueryParams
from multimodal_rag.preprocessor.transcriber.types import AudioTranscriber
from multimodal_rag.preprocessor.captioner.types import ImageCaptioner
from multimodal_rag.reranker.types import Reranker
from multimodal_rag.storage.types import StorageClient
from multimodal_rag.asset_store.types import AssetStore


class Registry:
    def __init__(self, lazy: bool = True):
        self.lazy = lazy

    def factory(self, module_path: str, class_name: str) -> Callable[[], Any]:
        if self.lazy:
            return lambda: self._import_lazy(module_path, class_name)
        else:
            cls = self._import_eager(module_path, class_name)
            return lambda: cls

    def _import_lazy(self, module_path: str, class_name: str) -> Any:
        if module_path in sys.modules:
            module = sys.modules[module_path]
        else:
            spec = importlib.util.find_spec(module_path)
            if spec is None:
                raise ImportError(f"Module '{module_path}' not found")
            loader = importlib.util.LazyLoader(spec.loader)
            spec.loader = loader
            module = importlib.util.module_from_spec(spec)
            sys.modules[module_path] = module
            loader.exec_module(module)
        return getattr(module, class_name)

    def _import_eager(self, module_path: str, class_name: str) -> Any:
        module = importlib.import_module(module_path)
        return getattr(module, class_name)


USE_LAZY = False
registry = Registry(lazy=USE_LAZY)

TEXT_EMBEDDER_MAPPING = {
    "replicate": registry.factory("multimodal_rag.embedder.repltext", "ReplicateTextEmbedder"),
    "openai": registry.factory("multimodal_rag.embedder.openai", "OpenAIEmbedder"),
    "ollama": registry.factory("multimodal_rag.embedder.ollama", "OllamaEmbedder"),
    "custom": registry.factory("multimodal_rag.embedder.custom_text", "CustomTextEmbedder"),
}

IMAGE_EMBEDDER_MAPPING = {
    "replicate": registry.factory("multimodal_rag.embedder.replimage", "ReplicateImageEmbedder"),
    "custom": registry.factory("multimodal_rag.embedder.custom_image", "CustomImageEmbedder"),
}

TRANSCRIBER_MAPPING = {
    "replicate": registry.factory("multimodal_rag.preprocessor.transcriber.replicate", "ReplicateTranscriber"),
    "custom": registry.factory("multimodal_rag.preprocessor.transcriber.custom", "CustomAudioTranscriber"),
}

CAPTIONER_MAPPING = {
    "custom": registry.factory("multimodal_rag.preprocessor.captioner.custom", "CustomImageCaptioner"),
}

STORAGE_CLIENTS = {
    "weaviate": registry.factory("multimodal_rag.storage.weaviate", "WeaviateClient"),
}

GENERATOR_PARAMS_MAPPING = {
    "openai": registry.factory("multimodal_rag.generator.params.openai", "OpenAIParams"),
    "ollama": registry.factory("multimodal_rag.generator.params.ollama", "OllamaParams"),
    "llamacpp": registry.factory("multimodal_rag.generator.params.llamacpp", "LlamaCppParams"),
}

GENERATOR_MAPPING = {
    "openai": registry.factory("multimodal_rag.generator.openai", "OpenAIGenerator"),
    "ollama": registry.factory("multimodal_rag.generator.ollama", "OllamaGenerator"),
    "llamacpp": registry.factory("multimodal_rag.generator.llamacpp", "LlamaCppGenerator"),
}

RERANKER_MAPPING = {
    "custom": registry.factory("multimodal_rag.reranker.custom", "CustomReranker"),
}

ASSET_STORE_CLIENTS = {
    "local": registry.factory("multimodal_rag.asset_store.local", "LocalAssetStore"),
    "s3": registry.factory("multimodal_rag.asset_store.s3", "S3AssetStore"),
}


def create_transcriber(config: TranscribingConfig) -> AudioTranscriber:
    cls = TRANSCRIBER_MAPPING.get(config.type)
    if not cls:
        raise ValueError(f"Unknown transcriber type: {config.type}")
    return cls()(model=config.model)


def create_captioner(config: CaptioningConfig) -> ImageCaptioner:
    cls = CAPTIONER_MAPPING.get(config.type)
    if not cls:
        raise ValueError(f"Unknown captioner type: {config.type}")
    return cls()(model=config.model)


def create_text_embedder(config: TextEmbeddingConfig) -> TextEmbedder:
    cls = TEXT_EMBEDDER_MAPPING.get(config.type)
    if not cls:
        raise ValueError(f"Unknown text embedder type: {config.type}")
    return cls()(model=config.model)


def create_image_embedder(config: ImageEmbeddingConfig) -> ImageEmbedder:
    cls = IMAGE_EMBEDDER_MAPPING.get(config.type)
    if not cls:
        raise ValueError(f"Unknown image embedder type: {config.type}")
    return cls()(model=config.model)


def create_storage_client(config: StoragingConfig) -> StorageClient:
    cls = STORAGE_CLIENTS.get(config.type)
    if not cls:
        raise ValueError(f"Unknown storage type: {config.type}")
    return cls()(getattr(config, config.type))


def create_asset_store(config: AssetStoreConfig) -> AssetStore:
    cls = ASSET_STORE_CLIENTS.get(config.type)
    if not cls:
        raise ValueError(f"Unknown asset store type: {config.type}")
    return cls()(getattr(config, config.type))


def create_reranker(config: RerankerConfig) -> Reranker:
    cls = RERANKER_MAPPING.get(config.type)
    if not cls:
        raise ValueError(f"Unknown reranker type: {config.type}")
    return cls()(model=config.model, supported_modes=config.supported_modes)


def create_generator(config: GenerationConfig) -> Generator:
    cls = GENERATOR_MAPPING.get(config.type)
    if not cls:
        raise ValueError(f"Unknown generator type: {config.type}")
    return cls()(model=config.model, context_limit=config.context_limit)


def parse_llm_params(generator_type: str, payload: dict) -> LLMQueryParams:
    factory = GENERATOR_PARAMS_MAPPING.get(generator_type)
    if not factory:
        raise ValueError(f"Unknown generator type: {generator_type}")
    return factory()(**payload)
