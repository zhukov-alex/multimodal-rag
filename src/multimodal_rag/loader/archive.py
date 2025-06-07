import asyncio
from pathlib import Path

from multimodal_rag.constants import KNOWN_BUT_UNSUPPORTED
from multimodal_rag.log_config import logger
from multimodal_rag.loader.reader.registry import ReaderRegistry
from multimodal_rag.loader.types import DocumentLoader, LoadResult
from multimodal_rag.utils.temp_dirs import make_tmp_dir


class ArchiveLoader(DocumentLoader):
    """
    Extracts the archive into a temp folder and returns for further processing.
    """

    def __init__(
        self,
        registry: ReaderRegistry,
        show_progress: bool = False,
    ):
        self.registry = registry
        self.show_progress = show_progress

    async def load(self, source: str, _: str | None = None) -> LoadResult:
        tmp_path = make_tmp_dir()
        tmp_path_str = str(tmp_path)
        logger.info("Extracting archive", extra={"path": source, "destination": tmp_path_str})
        await asyncio.to_thread(self._extract, Path(source), tmp_path)

        return LoadResult(documents=[], next_sources=[tmp_path_str])

    def _extract(self, arc_path: Path, target_path: Path) -> None:
        suffix = arc_path.suffix.lower()
        suffixes = tuple(s.lower() for s in arc_path.suffixes)

        if suffix == ".zip":
            import zipfile
            with zipfile.ZipFile(arc_path, "r") as zf:
                zf.extractall(target_path)
        elif suffixes[-2:] == (".tar", ".gz"):
            import tarfile
            with tarfile.open(arc_path, "r:gz") as tf:
                tf.extractall(target_path)
        elif suffixes in KNOWN_BUT_UNSUPPORTED:
            raise NotImplementedError(f"Archive type {suffixes} is known but not supported yet")
        else:
            raise ValueError(f"Unsupported archive format: {str(arc_path)}")
