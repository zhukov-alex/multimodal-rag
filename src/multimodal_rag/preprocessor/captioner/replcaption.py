import aiohttp
import asyncio
from aiohttp import ClientError
from asyncio import TimeoutError

from multimodal_rag.embedder.replmixin import ReplicateClientMixin
from multimodal_rag.preprocessor.captioner.types import ImageCaptioner
from multimodal_rag.utils.retry import backoff


class ReplicateImageCaptioner(ReplicateClientMixin, ImageCaptioner):
    """
    Captioner for images using Replicate API (via data URI).
    """

    def __init__(self, model_name: str):
        self._model_name = model_name

    @property
    def model_name(self) -> str:
        return self._model_name

    async def generate_captions(self, images: list[str]) -> list[str]:
        """
        Generate captions for a list of images.
        """
        async with aiohttp.ClientSession() as session:
            tasks = [self._caption_one(session, img_b64) for img_b64 in images]
            return await asyncio.gather(*tasks)

    @backoff(exception=(ClientError, TimeoutError))
    async def _caption_one(self, session: aiohttp.ClientSession, img_b64: str) -> str:
        """
        Sends an image to Replicate and gets a caption.
        """
        headers = {
            "Authorization": f"Token {self.replicate_token}",
            "Content-Type": "application/json"
        }

        if not img_b64.startswith("data:image/"):
            img_b64 = f"data:image/png;base64,{img_b64}"

        payload = {
            "version": self._model_name,
            "input": {
                "image": img_b64
            }
        }

        async with session.post(self.replicate_url, headers=headers, json=payload) as response:
            response.raise_for_status()
            data = await response.json()
            return data["output"]
