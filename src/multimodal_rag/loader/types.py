from typing import Protocol, AsyncIterator
from multimodal_rag.document import Document


class DocumentLoader(Protocol):
    """
    Interface for document loaders.
    """

    async def load(self) -> list[Document]:
        """
        Load all documents as a batch.
        """
        ...

    async def iter_documents(self) -> AsyncIterator[Document]:
        """
        Yield documents one by one.
        """
        ...
