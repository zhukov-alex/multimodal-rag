import asyncio
from pathlib import Path

from multimodal_rag.asset_store.types import AssetStore
from multimodal_rag.document import Document
from multimodal_rag.log_config import logger
from multimodal_rag.utils.temp_dirs import cleanup_tmp_dirs

DEFAULT_MAX_CONCURRENCY = 8


class AssetWriterService:
    """
    Handles storage of ingestion files in a persistent backend.
    """

    def __init__(self, store: AssetStore):
        self.store = store
        self.semaphore = asyncio.Semaphore(DEFAULT_MAX_CONCURRENCY)

    async def store_documents(self, project_id: str, documents: list[Document]) -> None:
        logger.info("Storing documents", extra={"project_id": project_id, "count": len(documents)})

        await self.store.ensure_storage(project_id)

        async def store_one(doc: Document):
            if not doc.source.tmp_uri:
                raise ValueError("Missing tmp_path in document source")

            tmp_path = Path(doc.source.tmp_uri)
            tmp_path_str = str(tmp_path)
            if not tmp_path.exists():
                raise FileNotFoundError(f"Tmp file does not exist: {tmp_path_str}")

            logger.debug("Storing document", extra={"tmp_path": tmp_path_str})

            try:
                async with self.semaphore:
                    uri = await self.store.store(
                        project_id=project_id,
                        tmp_path=tmp_path,
                        meta=doc.metadata
                    )
                    doc.source.asset_uri = uri
            except Exception as e:
                logger.exception("Failed to store document", extra={"tmp_path": tmp_path_str})
                raise

        await asyncio.gather(*(store_one(doc) for doc in documents))
        logger.info("Documents stored successfully")

    async def cleanup_tmp_files(self, documents: list[Document]) -> None:
        logger.info("Cleaning up tmp files", extra={"count": len(documents)})
        cleanup_tmp_dirs()
        logger.info("Cleanup complete")
