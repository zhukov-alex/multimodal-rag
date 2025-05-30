from pathlib import Path
from typing import Protocol
from multimodal_rag.document import Document


class FileReader(Protocol):
    """
    Interface for file-level content readers.
    """

    async def load(self, path: Path) -> list[Document]:
        ...
