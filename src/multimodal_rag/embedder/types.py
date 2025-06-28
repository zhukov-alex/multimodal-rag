from typing import Protocol


class TextEmbedder(Protocol):
    """
    Interface for text embedding API.
    """

    async def embed_texts(self, texts: list[str]) -> list[list[float]]:
        """
        Embed a list of texts.
        """
        ...

    @property
    def model_name(self) -> str:
        raise NotImplementedError


class ImageEmbedder(Protocol):
    """
    Interface for image embedding API.
    """

    async def embed_images(self, images: list[str]) -> list[list[float]]:
        """
        Embed a list of images.
        """
        ...

    async def embed_texts(self, texts: list[str]) -> list[list[float]]:
        """
        Embed texts.
        """
        ...

    @property
    def model_name(self) -> str:
        raise NotImplementedError
