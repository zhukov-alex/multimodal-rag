from typing import List, Union
from multimodal_rag.document import Document
from multimodal_rag.config.schema import IndexingConfig
from multimodal_rag.storage.types import AggregateFilter, StorageClient
from multimodal_rag.log_config import logger


class StorageIndexerService:
    def __init__(
        self,
        storage: StorageClient,
        config: IndexingConfig,
        project_id: str,
    ):
        self.storage = storage
        self.config = config
        self.project_id = project_id

    async def ensure_collections_exist(
        self, docs: List[Document]
    ) -> dict[str, Union[str, List[str]]]:
        used_models = extract_used_model_dims(self.config, docs)

        doc_collection = await self.storage.create_document_collection(self.project_id)
        logger.info("Ensured document collection", extra={"collection": doc_collection})

        emb_collections = []
        for model_name, dim in used_models.items():
            collection_name = await self.storage.create_embedding_collection(
                name=self.project_id,
                embedding_model=model_name,
                dim=dim,
            )
            emb_collections.append(collection_name)
            logger.info("Ensured embedding collection", extra={"collection": collection_name, "model": model_name, "dim": dim})

        return {
            "document": doc_collection,
            "embeddings": emb_collections,
        }

    async def import_documents(
        self,
        docs: List[Document],
        collection_map: dict[str, Union[str, List[str]]],
    ) -> None:
        uuids = [doc.uuid for doc in docs]

        try:
            await self.storage.insert_documents(
                docs, collection_name=collection_map["document"]
            )
            logger.info("Imported documents", extra={"count": len(docs)})

            for chunk_collection in collection_map["embeddings"]:
                await self.storage.insert_chunks(
                    docs, collection_name=chunk_collection
                )
                logger.info("Imported chunks", extra={"collection": chunk_collection})
                await self._validate_chunks(docs, chunk_collection)

            logger.info("Completed chunk validation", extra={"collections": collection_map["embeddings"]})

        except Exception:
            logger.exception("Import failed, starting rollback", extra={"doc_uuids": uuids})
            await self._rollback(collection_map, uuids)
            raise

    async def _validate_chunks(
        self, docs: List[Document], chunk_collection: str
    ) -> None:
        for doc in docs:
            expected_count = sum(len(group.chunks) for group in doc.chunk_groups)
            if expected_count == 0:
                continue

            actual_count = await self.storage.aggregate_total_count(
                collection_name=chunk_collection,
                filter_by=AggregateFilter(field="doc_uuid", value=doc.uuid)
            )

            if actual_count != expected_count:
                logger.warning("Chunk count mismatch", extra={
                    "doc_id": doc.uuid,
                    "expected": expected_count,
                    "actual": actual_count,
                    "collection": chunk_collection
                })
                raise ValueError(
                    f"Chunk count mismatch for doc {doc.uuid} in {chunk_collection}: "
                    f"expected {expected_count}, got {actual_count}"
                )

    async def _rollback(
        self, collection_map: dict[str, Union[str, List[str]]], uuids: List[str]
    ) -> None:
        await self.storage.delete_by_ids(
            collection_name=collection_map["document"],
            field="uuid",
            ids=uuids,
        )
        logger.info("Rolled back documents", extra={"count": len(uuids)})

        for chunk_collection in collection_map["embeddings"]:
            await self.storage.delete_by_ids(
                collection_name=chunk_collection,
                field="doc_uuid",
                ids=uuids,
            )
            logger.info("Rolled back chunks", extra={"collection": chunk_collection, "count": len(uuids)})


def get_embedding_dim(doc: Document, modality: str) -> int | None:
    chunks = (
        chunk
        for group in doc.chunk_groups
        if group.modality == modality
        for chunk in group.chunks
        if chunk.embedding
    )
    first = next(chunks, None)
    return len(first.embedding) if first else None


def extract_used_model_dims(config: IndexingConfig, docs: list[Document]) -> dict[str, int]:
    """
    Extracts embedding dimensions for configured models (text/image),
    based on the actual embeddings present in the document chunks.
    """

    used: dict[str, int] = {}
    txt_model = config.embedding.text.model
    img_model = config.embedding.image.model

    for doc in docs:
        modality = doc.source.get_modality()

        if modality == "text" and txt_model not in used:
            dim = get_embedding_dim(doc, "text")
            if dim:
                used[txt_model] = dim

        elif modality == "image" and img_model not in used:
            dim = get_embedding_dim(doc, "image")
            if dim:
                used[img_model] = dim

        if len(used) >= len({txt_model, img_model}):
            break

    return used
