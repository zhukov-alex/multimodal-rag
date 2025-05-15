from langchain_text_splitters import (
    RecursiveCharacterTextSplitter,
    MarkdownHeaderTextSplitter,
    RecursiveJsonSplitter,
)
from langchain_text_splitters.base import Language


def create_splitter(name: str, **kwargs):
    if name == "markdown_chunker":
        return MarkdownHeaderTextSplitter(**kwargs)

    elif name == "json_chunker":
        return RecursiveJsonSplitter(**kwargs)

    elif name == "recursive_chunker":
        return RecursiveCharacterTextSplitter(**kwargs)

    elif name == "code_chunker":
        lang = kwargs.pop("language")
        try:
            lang_enum = Language(lang.lower())
        except ValueError:
            raise ValueError(f"Unsupported language for code chunking: {lang}")

        return RecursiveCharacterTextSplitter.from_language(
            language=lang_enum,
            **kwargs
        )

    else:
        raise ValueError(f"Unknown splitter type: {name}")
