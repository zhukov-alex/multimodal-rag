import asyncio

from multimodal_rag.document import Document, Chunk, ChunkGroup
from multimodal_rag.embedder.types import TextEmbedder, ImageEmbedder
from multimodal_rag.log_config import logger
from multimodal_rag.utils.loader import load_image_bytes_from_source

DEFAULT_MAX_CONCURRENCY = 8


class EmbedderService:
    DEFAULT_BATCH_SIZE = 100

    def __init__(
        self,
        text_embedder: TextEmbedder,
        image_embedder: ImageEmbedder | None = None,
        batch_size: int = DEFAULT_BATCH_SIZE,
    ):
        self.text_embedder = text_embedder
        self.image_embedder = image_embedder
        self.semaphore = asyncio.Semaphore(DEFAULT_MAX_CONCURRENCY)
        self.batch_size = batch_size

    @property
    def text_model_name(self) -> str:
        return self.text_embedder.model_name

    @property
    def image_model_name(self) -> str | None:
        return self.image_embedder.model_name if self.image_embedder else None

    async def embed_documents(self, docs: list[Document]) -> None:
        logger.info("Embedding documents", extra={"count": len(docs)})
        await asyncio.gather(*(self._embed_one_document(doc) for doc in docs))
        logger.info("Finished embedding documents")

    async def embed_text_query(self, query: str) -> list[float]:
        logger.debug("Embedding text query", extra={"query_length": len(query)})
        embeddings = await self.text_embedder.embed_texts([query])
        return embeddings[0]

    async def embed_text_as_image(self, text: str) -> list[float]:
        if not self.image_embedder:
            raise RuntimeError("Image embedder is not configured.")

        logger.debug("Embedding text using image embedder", extra={"text_length": len(text)})
        embeddings = await self.image_embedder.embed_texts([text])
        return embeddings[0]

    async def embed_image_query(self, path: str) -> list[float]:
        if not self.image_embedder:
            raise RuntimeError("Image embedder is not configured.")

        logger.debug("Embedding image query", extra={"path": path})
        image_bytes = await load_image_bytes_from_source(path)
        embeddings = await self.image_embedder.embed_images([image_bytes])
        return embeddings[0]

    async def _embed_one_document(self, doc: Document) -> None:
        match doc.source.type:
            case "image":
                await self._embed_image(doc)
            case "text":
                await self._embed_text(doc)
            case _:
                raise RuntimeError(f"Unsupported modality type: {doc.source.type} for doc {doc.uuid}")

    async def _embed_image(self, doc: Document) -> None:
        if not self.image_embedder:
            raise RuntimeError("No image embedder provided")

        path = doc.source.tmp_path
        logger.debug("Embedding image document", extra={"path": path})

        image_bytes = await load_image_bytes_from_source(path)
        embedding = (await self.image_embedder.embed_images([image_bytes]))[0]

        caption = doc.content or ""

        for group in doc.chunk_groups:
            if group.modality == "image":
                raise RuntimeError(f"Duplicate image chunk group in doc {doc.uuid}")

        doc.chunk_groups.append(
            ChunkGroup(
                embedder_name=self.image_model_name,
                modality="image",
                chunks=[
                    Chunk(
                        chunk_id=0,
                        content=caption,
                        embedding=embedding
                    )
                ]
            )
        )

    async def _embed_text(self, doc: Document) -> None:
        text_groups = [g for g in doc.chunk_groups if g.modality == "text"]

        for group in text_groups:
            chunks = group.chunks
            contents = [chunk.content for chunk in chunks]
            if not contents:
                logger.warning("No text content to embed", extra={"doc_id": doc.uuid})
                continue

            embeddings = await self._batch_embed_texts(contents)

            if len(embeddings) != len(chunks):
                logger.error("Embedding count mismatch", extra={"doc_id": doc.uuid})
                raise RuntimeError("Mismatch in embedding and chunk count")

            group.embedder_name = self.text_model_name

            for chunk, emb in zip(chunks, embeddings):
                chunk.embedding = emb

    async def _batch_embed_texts(self, contents: list[str]) -> list[list[float]]:
        batches = [
            contents[i: i + self.batch_size]
            for i in range(0, len(contents), self.batch_size)
        ]

        logger.debug("Batching text chunks", extra={"total_chunks": len(contents), "batches": len(batches)})

        async def embed_with_limit(batch):
            async with self.semaphore:
                return await self.text_embedder.embed_texts(batch)

        tasks = [asyncio.create_task(embed_with_limit(batch)) for batch in batches]
        results = await asyncio.gather(*tasks)

        embeddings = []
        for result in results:
            embeddings.extend(result)

        logger.debug("Completed embedding batches", extra={"total_embeddings": len(embeddings)})
        return embeddings
