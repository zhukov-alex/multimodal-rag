from pathlib import Path
from multimodal_rag.document import Document
from multimodal_rag.loader.github import GitHubRepoLoader
from multimodal_rag.loader.archive import ArchiveLoader
from multimodal_rag.loader.directory import DirectoryLoader
from multimodal_rag.loader.types import DocumentLoader
from multimodal_rag.loader.utils import is_archive, is_github_url
from multimodal_rag.loader.reader.registry import ReaderRegistry
from multimodal_rag.log_config import logger


class LoaderService:
    """
    Service responsible for resolving the source type (local, archive, GitHub)
    and loading documents using the appropriate loader.
    """

    def __init__(self, source: str, registry: ReaderRegistry, glob: str = "**/*"):
        """
        Args:
            source: Path or GitHub URL to load from
            registry: ReaderRegistry instance
            glob: File matching pattern
        """
        self.source = source
        self.glob = glob
        self.registry = registry
        self.loader = self._resolve_loader()

    def _detect_source_type(self):
        p = Path(self.source)
        if is_github_url(self.source):
            detected = "github"
        elif p.is_dir():
            detected = "directory"
        elif p.is_file():
            detected = "archive" if is_archive(p) else "file"
        else:
            raise ValueError(f"Unsupported source type: {self.source}")

        logger.info("Detected source type", extra={"source": self.source, "type": detected})
        return detected

    def _resolve_loader(self) -> DocumentLoader:
        kind = self._detect_source_type()

        if kind == "github":
            loader = GitHubRepoLoader(self.source, self.registry, self.glob)
        elif kind == "archive":
            loader = ArchiveLoader(self.source, self.registry, self.glob)
        elif kind == "directory":
            loader = DirectoryLoader(self.source, self.registry, self.glob)
        elif kind == "file":
            parent = str(Path(self.source).parent)
            file_glob = Path(self.source).name
            loader = DirectoryLoader(parent, self.registry, file_glob)
        else:
            raise RuntimeError(f"Unreachable state in _resolve_loader (kind: {kind})")

        logger.info("Resolved loader", extra={"loader": type(loader).__name__, "source": self.source})
        return loader

    async def load(self) -> list[Document]:
        logger.info("Loading documents", extra={"source": self.source})
        return await self.loader.load()
