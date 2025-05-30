from pathlib import Path
from multimodal_rag.loader.github import GitHubRepoLoader
from multimodal_rag.loader.archive import ArchiveLoader
from multimodal_rag.loader.directory import DirectoryLoader
from multimodal_rag.loader.types import DocumentLoader
from multimodal_rag.loader.utils import is_archive, is_github_url
from multimodal_rag.loader.reader.registry import ReaderRegistry
from multimodal_rag.log_config import logger


class SourceResolver:
    """
    Resolves the appropriate loader for the given source path or URL.
    """

    def __init__(self, registry: ReaderRegistry):
        self.registry = registry
        self.loader_cache: dict[str, DocumentLoader] = {}

    def resolve_loader(self, source: str) -> tuple[DocumentLoader, str]:
        kind = self._detect_source_type(source)

        if kind in self.loader_cache:
            return self.loader_cache[kind], kind

        match kind:
            case "github":
                loader = GitHubRepoLoader(self.registry)
            case "archive":
                loader = ArchiveLoader(self.registry)
            case "directory" | "file":
                loader = DirectoryLoader(self.registry)
            case _:
                raise RuntimeError(f"Unreachable state in resolve_loader (kind: {kind})")

        self.loader_cache[kind] = loader
        logger.info("Resolved loader", extra={"loader": type(loader).__name__, "source": source})
        return loader, kind

    def _detect_source_type(self, source: str):
        p = Path(source)
        if is_github_url(source):
            detected = "github"
        elif p.is_dir():
            detected = "directory"
        elif p.is_file():
            detected = "archive" if is_archive(p) else "file"
        else:
            raise ValueError(f"Unsupported source type: {source}")

        logger.info("Detected source type", extra={"source": source, "type": detected})
        return detected
