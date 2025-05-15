from typing import List
import asyncio

from multimodal_rag.document import Document, Chunk, ChunkGroup
from multimodal_rag.chunker.registry import SplitterRegistry
from multimodal_rag.log_config import logger


class ChunkerService:
    """
    Service for splitting documents into smaller chunks.
    """

    DEFAULT_BUFFER_SIZE = 250_000  # Default max buffer size for splitting

    def __init__(self, registry: SplitterRegistry, buffer_size: int = DEFAULT_BUFFER_SIZE):
        """
        Args:
            registry: Instance of SplitterRegistry to resolve the splitter per document.
            buffer_size: Maximum size of buffer for splitting (default 250,000 characters).
        """
        self.registry = registry
        self.buffer_size = buffer_size

    async def chunk_documents(self, docs: List[Document]) -> None:
        """
        Split all documents into chunks.
        """
        logger.info("Chunking documents", extra={"count": len(docs)})
        await asyncio.gather(*(self._chunk_one(doc) for doc in docs))
        logger.info("Finished chunking documents")

    async def _chunk_one(self, doc: Document) -> None:
        """
        Chunk one document.
        """
        splitter = self.registry.get_splitter(doc)
        chunks = self._buffered_split(doc, splitter)

        doc.chunk_groups = [
            ChunkGroup(
                chunks=chunks,
                embedder_name="",
                modality="text"
            )
        ]

        logger.debug("Document chunked", extra={"doc_id": doc.uuid, "chunks": len(chunks)})

    def _buffered_split(self, doc: Document, splitter) -> List[Chunk]:
        text_chunks = []
        buffer_tail = ""

        for start in range(0, len(doc.content), self.buffer_size):
            part = buffer_tail + doc.content[start: start + self.buffer_size]
            part_chunks = splitter.split_text(part)

            if part_chunks:
                buffer_tail = part_chunks.pop()
                text_chunks.extend(chunk.strip() for chunk in part_chunks if chunk.strip())

        if buffer_tail.strip():
            text_chunks.append(buffer_tail.strip())

        return [
            Chunk(chunk_id=i, content=c)
            for i, c in enumerate(text_chunks)
        ]
