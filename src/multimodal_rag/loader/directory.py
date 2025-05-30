import asyncio
from pathlib import Path

from multimodal_rag.document import Document
from multimodal_rag.loader.reader.registry import ReaderRegistry
from multimodal_rag.log_config import logger
from multimodal_rag.loader.types import DocumentLoader, LoadResult
from multimodal_rag.loader.utils import is_archive

try:
    from tqdm.asyncio import tqdm_asyncio as tqdm
except ImportError:
    from tqdm import tqdm

DEFAULT_MAX_CONCURRENCY = 8


class DirectoryLoader(DocumentLoader):
    """
    Loads documents from a directory using registry-based content readers.
    """

    def __init__(
        self,
        registry: ReaderRegistry,
        show_progress: bool = False,
    ):
        self.registry = registry
        self.show_progress = show_progress
        self.semaphore = asyncio.Semaphore(DEFAULT_MAX_CONCURRENCY)

    async def load(self, source: str, filter: str = "**/*") -> LoadResult:
        spath = Path(source)
        if not spath.exists():
            raise FileNotFoundError(f"Path does not exist: {source}")
        if not spath.is_dir():
            raise ValueError(f"Path is not a directory: {source}")

        files = [f for f in spath.glob(filter) if f.is_file()]
        next_sources = [str(f) for f in files if is_archive(f)]
        to_process = [f for f in files if not is_archive(f)]

        async def process(path: Path) -> list[Document]:
            async with self.semaphore:
                try:
                    loader = self.registry(path)
                    logger.debug("Loading file", extra={"path": str(path)})
                    return await loader.load(path)
                except Exception as e:
                    logger.exception("Failed to load file", extra={"path": str(path), "error": str(e)})
                    raise e

        tasks = [process(p) for p in to_process]
        iterator = asyncio.as_completed(tasks)

        if self.show_progress:
            iterator = tqdm(iterator, total=len(tasks), desc="Async loading files")

        all_documents = []
        for coro in iterator:
            docs = await coro
            all_documents.extend(docs)

        return LoadResult(documents=all_documents, next_sources=next_sources)
