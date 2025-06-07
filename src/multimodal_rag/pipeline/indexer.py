from multimodal_rag.config.schema import IndexingConfig
from multimodal_rag.config.factory import (
    create_transcriber,
    create_captioner,
    create_text_embedder,
    create_image_embedder,
    create_asset_store,
    create_storage_client,
)
from multimodal_rag.loader.reader.extension_based import ExtensionBasedReader
from multimodal_rag.loader.reader.registry import ReaderRegistry
from multimodal_rag.loader.resolver import SourceResolver
from multimodal_rag.loader.service import RecursiveLoaderService
from multimodal_rag.asset_store.writer import AssetWriterService
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

    resolver = SourceResolver(registry=registry)
    recursive_loader = RecursiveLoaderService(resolver)

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

    asset_store = create_asset_store(config.asset_store) if config.asset_store else None
    asset_storage_service = (
        AssetWriterService(store=asset_store) if asset_store else None
    )

    docs = []
    try:
        async with log_duration("load_documents"):
            docs = await recursive_loader.load(source)

        if asset_storage_service:
            async with log_duration("store_documents"):
                await asset_storage_service.store_documents(project_id, docs)

        async with log_duration("chunk_documents", count=len(docs)):
            await chunker_service.chunk_documents(docs)

        async with log_duration("embed_documents"):
            await embedder_service.embed_documents(docs)

        async with log_duration("ensure_collections"):
            collections = await indexer.ensure_collections_exist(docs)

        async with log_duration("import_documents", collections=len(collections), count=len(docs)):
            await indexer.import_documents(docs, collections)

    finally:
        if asset_storage_service and docs:
            await asset_storage_service.cleanup_tmp_files(docs)
