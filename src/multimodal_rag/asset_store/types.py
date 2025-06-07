from pathlib import Path
from typing import Protocol

from multimodal_rag.document import MetaConfig


class AssetStore(Protocol):
    """
    Interface for persistent storage backends.
    """

    async def ensure_storage(self, project_id: str) -> None:
        """Ensure project-level storage exists (bucket, collection, etc.)"""
        ...

    async def store(self, project_id: str, tmp_path: Path, meta: MetaConfig) -> str:
        """
        Persist the given file to the store and return its storage URI.

        Args:
            project_id: Identifier for the project namespace.
            tmp_path: Path to the local temporary file to store.
            meta: Metadata describing the file.

        Returns:
            URI string pointing to the stored file.
        """
        ...

    async def read(self, uri: str) -> bytes:
        ...
