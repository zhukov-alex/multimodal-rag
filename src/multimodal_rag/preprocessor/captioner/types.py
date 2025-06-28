from typing import Protocol


class ImageCaptioner(Protocol):
    """
    Interface for image captioning API.
    """

    async def generate_captions(self, images: list[str]) -> list[str]:
        """
        Generate captions for a list of images.
        """
        ...

    def model_name(self) -> str:
        ...
