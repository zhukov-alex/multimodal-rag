from pathlib import Path

from multimodal_rag.loader.reader.types import FileReader


class ReaderRegistry:
    """
    Registry that maps file extensions to specific reader instances.
    """

    def __init__(self):
        self._handlers: dict[str, FileReader] = {}
        self._default: FileReader | None = None

    def register(self, extensions: list[str] | None, reader: FileReader):
        """
        Register a reader for specific extensions, or set it as the default reader
        if extensions is None or an empty list.

        Args:
            extensions: list of file extensions (e.g. ['.txt', '.md']) or None/empty for default.
            reader: Reader instance that handles those extensions.
        """
        if not extensions:
            self._default = reader
        else:
            for ext in extensions:
                self._handlers[ext.lower()] = reader

    def get(self, path: Path) -> FileReader:
        """
        Return the appropriate reader for the given file path/uri.
        """
        reader = self._handlers.get(path.suffix.lower())
        if reader:
            return reader
        if self._default:
            return self._default
        raise ValueError(f"No reader registered for extension '{path.suffix}' and no default reader set.")

    def __call__(self, path: Path) -> FileReader:
        return self.get(path)
