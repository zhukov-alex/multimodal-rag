from multimodal_rag.document import Document
from multimodal_rag.loader.types import LoadResult
from multimodal_rag.loader.resolver import SourceResolver
from multimodal_rag.log_config import logger


class RecursiveLoaderService:
    """
    Loads documents from a given source recursively using source-specific loaders.
    """

    def __init__(self, resolver: SourceResolver, max_depth: int = 10):
        self.resolver = resolver
        self.max_depth = max_depth

    async def load(self, source: str, filter: str = "**/*") -> list[Document]:
        return await self._load_recursive(source, filter, depth=0)

    async def _load_recursive(self, source: str, filter: str, depth: int) -> list[Document]:
        if depth > self.max_depth:
            raise RuntimeError(f"Max recursion depth exceeded: {source}")

        loader, kind = self.resolver.resolve_loader(source)
        logger.info("Using loader", extra={"loader": type(loader).__name__, "source": source})

        result: LoadResult = await loader.load(source, filter)

        documents = list(result.documents)
        for sub in result.next_sources:
            documents.extend(await self._load_recursive(str(sub), filter, depth + 1))

        return documents
