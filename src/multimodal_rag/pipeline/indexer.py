from multimodal_rag.config.schema import IndexingConfig
from multimodal_rag.config.factory import (
    create_transcriber,
    create_captioner,
    create_text_embedder,
    create_image_embedder,
    create_storage_client,
)
from multimodal_rag.loader.reader.extension_based import ExtensionBasedReader
from multimodal_rag.loader.reader.registry import ReaderRegistry
from multimodal_rag.loader.service import LoaderService
from multimodal_rag.chunker.registry import SplitterRegistry
from multimodal_rag.chunker.service import ChunkerService
from multimodal_rag.embedder.service import EmbedderService
from multimodal_rag.storage.service import StorageIndexerService
from multimodal_rag.utils.timing import log_duration


async def run_index_pipeline(source: str, config: IndexingConfig, project_id: str) -> None:
    transcriber = create_transcriber(config.transcribing) if config.transcribing else None
    captioner = create_captioner(config.captioning) if config.captioning else None
    default_reader = ExtensionBasedReader(transcriber=transcriber, captioner=captioner)

    registry = ReaderRegistry()
    registry.register(extensions=None, reader=default_reader)

    loader_service = LoaderService(source=source, registry=registry)

    splitter_registry = SplitterRegistry(config.chunking)
    chunker_service = ChunkerService(registry=splitter_registry)

    text_embedder = create_text_embedder(config.embedding.text)
    image_embedder = create_image_embedder(config.embedding.image) if config.embedding.image else None
    embedder_service = EmbedderService(
        text_embedder,
        image_embedder,
        config.embedding.batch_size,
    )

    storage = create_storage_client(config.storaging)
    indexer = StorageIndexerService(storage, config, project_id)

    async with log_duration("load_documents"):
        docs = await loader_service.load()

    async with log_duration("chunk_documents", count=len(docs)):
        await chunker_service.chunk_documents(docs)

    async with log_duration("embed_documents"):
        await embedder_service.embed_documents(docs)

    async with log_duration("ensure_collections"):
        collections = await indexer.ensure_collections_exist(docs)

    async with log_duration("import_documents", collections=len(collections), count=len(docs)):
        await indexer.import_documents(docs, collections)
