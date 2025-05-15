import asyncio
from pathlib import Path
from typing import AsyncIterator

from multimodal_rag.document import Document
from multimodal_rag.loader.reader.registry import ReaderRegistry
from multimodal_rag.log_config import logger
from multimodal_rag.loader.types import DocumentLoader

try:
    from tqdm.asyncio import tqdm_asyncio as tqdm
except ImportError:
    from tqdm import tqdm

DEFAULT_MAX_CONCURRENCY = 8


class DirectoryLoader(DocumentLoader):
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
        if not self.path.exists():
            raise FileNotFoundError(f"Path does not exist: {self.path}")
        if not self.path.is_dir():
            raise ValueError(f"Path is not a directory: {self.path}")

        files = [f for f in self.path.glob(self.glob) if f.is_file()]
        semaphore = asyncio.Semaphore(DEFAULT_MAX_CONCURRENCY)

        async def process(path: Path) -> list[Document]:
            async with semaphore:
                try:
                    loader = self.registry(path)
                    logger.debug("Loading file", extra={"path": str(path)})
                    return await loader.load(str(path))
                except Exception as e:
                    logger.exception("Failed to load file", extra={"path": str(path), "error": str(e)})
                    raise e

        tasks = [process(p) for p in files]
        iterator = asyncio.as_completed(tasks)

        if self.show_progress:
            iterator = tqdm(iterator, total=len(tasks), desc="Async loading files")

        for coro in iterator:
            docs = await coro
            for doc in docs:
                yield doc
