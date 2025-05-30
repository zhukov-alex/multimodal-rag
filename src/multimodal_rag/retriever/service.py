from typing import Optional, Any
import asyncio

from multimodal_rag.embedder.service import EmbedderService
from multimodal_rag.storage.types import StorageClient
from multimodal_rag.document import MetaConfig, ScoredChunk, ScoredItem
from multimodal_rag.retriever.types import SearchByText, SearchByImage
from multimodal_rag.utils.loader import load_image_base64
from multimodal_rag.log_config import logger


class MultiModalRetriever:
    def __init__(
        self,
        embedder: EmbedderService,
        storage: StorageClient,
        reranker: Optional[Any] = None,
    ) -> None:
        self.embedder = embedder
        self.storage = storage
        self.reranker = reranker

    async def retrieve_by_text(self, request: SearchByText) -> list[ScoredItem]:
        text_vec = await self.embedder.text_embedder.embed_text_query(request.query)
        image_vec = await self.embedder.embed_text_as_image(request.query)
        text_model = self.embedder.text_model_name
        image_model = self.embedder.image_model_name

        top_k_text = request.modality_top_k.get("text", 0) * 3
        top_k_image = request.modality_top_k.get("image", 0) * 3
        chunk_results: list[ScoredChunk] = []

        if top_k_text > 0:
            text_collection = f"{request.project_id}_embedding_{text_model}"
            if request.search_type == "embedding":
                retrieval = await self.storage.query_by_vector(
                    vector=text_vec,
                    collection_name=text_collection,
                    top_k=top_k_text,
                    filters=request.filters,
                )
            else:
                retrieval = await self.storage.hybrid_chunks(
                    query=request.query,
                    vector=text_vec,
                    collection_name=text_collection,
                    limit=top_k_text,
                    filters=request.filters,
                )
            chunk_results.extend(retrieval)

        if top_k_image > 0 and image_model:
            image_collection = f"{request.project_id}_embedding_{image_model}"
            if request.search_type == "embedding":
                retrieval = await self.storage.query_by_vector(
                    vector=image_vec,
                    collection_name=image_collection,
                    top_k=top_k_image,
                    filters=request.filters,
                )
            else:
                retrieval = await self.storage.hybrid_chunks(
                    query=request.query,
                    vector=image_vec,
                    collection_name=image_collection,
                    limit=top_k_image,
                    filters=request.filters,
                )
            chunk_results.extend(retrieval)

        doc_ids = list({sc.chunk.doc_uuid for sc in chunk_results})
        doc_records = await self.storage.query_by_filter(
            collection_name=f"{request.project_id}_documents",
            filters={"and": [{"field": "uuid", "operator": "contains_any", "value": doc_ids}]},
        )
        doc_map = {d["uuid"]: d for d in doc_records}

        results: list[ScoredItem] = [
            ScoredItem(
                doc_uuid=sc.chunk.doc_uuid,
                chunk_id=sc.chunk.chunk_id,
                content=sc.chunk.content,
                modality=doc.get("modality", "text"),
                score=sc.score,
                source_path=doc.get("source", {}).get("source_path", ""),
                caption=doc.get("metadata", {}).get("caption"),
                metadata=MetaConfig(**doc.get("metadata", {})),
            )
            for sc in chunk_results
            if (doc := doc_map.get(sc.chunk.doc_uuid)) is not None
        ]

        await self._load_images(results)

        if request.rerank and self.reranker and self.reranker.supports(request.rerank):
            results = await self.reranker.process(request.query, results)

        return self._top_k_results(results, request.modality_top_k)

    async def retrieve_by_image(self, request: SearchByImage) -> list[ScoredItem]:
        image_vec = (await self.embedder.image_embedder.embed_images([request.blob]))[0]
        image_model = self.embedder.image_model_name()
        collection = f"{request.project_id}_embedding_{image_model}"
        top_k = request.top_k * 3

        if request.caption:
            chunk_results = await self.storage.hybrid_chunks(
                query=request.caption,
                vector=image_vec,
                collection_name=collection,
                limit=top_k,
                filters=request.filters,
            )
        else:
            chunk_results = await self.storage.query_by_vector(
                vector=image_vec,
                collection_name=collection,
                top_k=top_k,
                filters=request.filters,
            )

        doc_ids = list({sc.chunk.doc_uuid for sc in chunk_results})
        document_results = await self.storage.query_by_filter(
            collection_name=f"{request.project_id}_documents",
            filters={"and": [{"field": "uuid", "operator": "contains_any", "value": doc_ids}]},
        )
        doc_map = {d["uuid"]: d for d in document_results}

        results: list[ScoredItem] = [
            ScoredItem(
                doc_uuid=sc.chunk.doc_uuid,
                chunk_id=sc.chunk.chunk_id,
                content=sc.chunk.content,
                modality=doc.get("modality", "image"),
                score=sc.score,
                source_path=doc.get("source", {}).get("source_path", ""),
                caption=doc.get("metadata", {}).get("caption"),
                metadata=MetaConfig(**doc.get("metadata", {})),
            )
            for sc in chunk_results
            if (doc := doc_map.get(sc.chunk.doc_uuid)) is not None
        ]

        await self._load_images(results)

        if request.rerank and self.reranker and self.reranker.supports("image"):
            results = await self.reranker.process(request.caption or "", results)

        results.sort(key=lambda i: i.score, reverse=True)
        return results[:request.top_k]

    async def _load_images(self, items: list[ScoredItem]) -> None:
        image_items = [
            item for item in items
            if item.modality == "image" and not item.image_base64 and item.source_path
        ]
        image_tasks = {item.doc_uuid: load_image_base64(item.source_path) for item in image_items}

        if not image_tasks:
            return

        try:
            image_data = await asyncio.gather(*image_tasks.values())
            image_map = dict(zip(image_tasks.keys(), image_data))
            for item in image_items:
                item.image_base64 = image_map.get(item.doc_uuid)
        except Exception as e:
            logger.warning(f"Failed to load image base64 in batch: {e}")

    @staticmethod
    def _top_k_results(results: list[ScoredItem], modality_top_k: dict[str, int]) -> list[ScoredItem]:
        by_modality = {"text": [], "image": []}
        for item in results:
            if item.modality in by_modality:
                by_modality[item.modality].append(item)

        final_results = []
        for mod, limit in modality_top_k.items():
            top_items = sorted(by_modality.get(mod, []), key=lambda i: i.score, reverse=True)[:limit]
            final_results.extend(top_items)

        return final_results
