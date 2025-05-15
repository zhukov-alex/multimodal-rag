from typing import Protocol
from multimodal_rag.document import Document


class FileReader(Protocol):
    """
    Interface for file-level content readers.
    """

    async def load(self, path: str) -> list[Document]:
        ...
