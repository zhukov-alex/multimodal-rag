from multimodal_rag.document import Document
from multimodal_rag.config.schema import ChunkingConfig
from multimodal_rag.chunker.factory import create_splitter


class SplitterRegistry:
    def __init__(self, chunking_config: ChunkingConfig):
        self.chunking_config = chunking_config
        self._cache: dict[str, object] = {}

    def get_splitter(self, doc: Document):
        doc_type = doc.config.source.type
        content_type_to_chunker = self.chunking_config.content_type_to_chunker

        if doc_type not in content_type_to_chunker:
            raise ValueError(f"No chunker defined for document type: {doc_type}")

        chunker_name = content_type_to_chunker[doc_type]

        if chunker_name in self._cache:
            return self._cache[chunker_name]

        kwargs = getattr(self.chunking_config, chunker_name)
        if chunker_name == "code_chunker":
            kwargs = dict(kwargs)
            kwargs["language"] = doc_type.removeprefix("code_").lower()

        splitter = create_splitter(chunker_name, **kwargs)
        self._cache[chunker_name] = splitter
        return splitter
