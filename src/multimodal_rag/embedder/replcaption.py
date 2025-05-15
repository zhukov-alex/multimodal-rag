import aiohttp
import asyncio
import base64
from typing import List
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

    async def generate_captions(self, images: List[bytes]) -> List[str]:
        """
        Generate captions for a list of images.
        """
        async with aiohttp.ClientSession() as session:
            tasks = [self._caption_one(session, img) for img in images]
            return await asyncio.gather(*tasks)

    @backoff(exception=(ClientError, TimeoutError))
    async def _caption_one(self, session: aiohttp.ClientSession, img_bytes: bytes) -> str:
        """
        Sends an image to Replicate and gets a caption.
        """
        headers = {
            "Authorization": f"Token {self.replicate_token}",
            "Content-Type": "application/json"
        }

        # Encode image as base64 data URI
        encoded = base64.b64encode(img_bytes).decode("utf-8")
        data_uri = f"data:image/png;base64,{encoded}"

        payload = {
            "version": self._model_name,
            "input": {
                "image": data_uri
            }
        }

        async with session.post(self.replicate_url, headers=headers, json=payload) as response:
            response.raise_for_status()
            data = await response.json()
            return data["output"]
