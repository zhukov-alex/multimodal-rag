import asyncio
import tempfile
from pathlib import Path
from typing import AsyncIterator

from multimodal_rag.document import Document
from multimodal_rag.constants import KNOWN_BUT_UNSUPPORTED
from multimodal_rag.log_config import logger
from multimodal_rag.loader.reader.registry import ReaderRegistry
from multimodal_rag.loader.types import DocumentLoader
from multimodal_rag.loader.directory import DirectoryLoader


class ArchiveLoader(DocumentLoader):
    """
    Archive loader that extracts the archive into a temporary directory,
    then delegates loading to DirectoryLoader.
    """

    def __init__(
        self,
        path: str,
        registry: ReaderRegistry,
        glob: str = "**/*",
        show_progress: bool = False,
    ):
        self.path = Path(path)
        self.glob = glob
        self.registry = registry
        self.show_progress = show_progress

    async def load(self) -> list[Document]:
        return [doc async for doc in self.iter_documents()]

    async def iter_documents(self) -> AsyncIterator[Document]:
        with tempfile.TemporaryDirectory() as tmp:
            extracted_path = Path(tmp)
            logger.info("Extracting archive", extra={"path": str(self.path), "destination": str(extracted_path)})
            await asyncio.to_thread(self._extract, extracted_path)

            directory_loader = DirectoryLoader(
                path=str(extracted_path),
                registry=self.registry,
                glob=self.glob,
                show_progress=self.show_progress,
            )

            async for doc in directory_loader.iter_documents():
                yield doc

    def _extract(self, target_path: Path) -> None:
        suffix = self.path.suffix.lower()
        suffixes = tuple(s.lower() for s in self.path.suffixes)

        if suffix == ".zip":
            import zipfile
            with zipfile.ZipFile(self.path, "r") as zf:
                zf.extractall(target_path)
        elif suffixes[-2:] == (".tar", ".gz"):
            import tarfile
            with tarfile.open(self.path, "r:gz") as tf:
                tf.extractall(target_path)
        elif suffixes in KNOWN_BUT_UNSUPPORTED:
            raise NotImplementedError(f"Archive type {suffixes} is known but not supported yet")
        else:
            raise ValueError(f"Unsupported archive format: {self.path}")
