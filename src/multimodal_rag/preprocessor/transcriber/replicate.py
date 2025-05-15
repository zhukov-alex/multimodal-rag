import os
import aiohttp
from multimodal_rag.preprocessor.transcriber.types import AudioTranscriber
from multimodal_rag.log_config import logger
from multimodal_rag.utils.retry import backoff
from aiohttp import ClientError
from asyncio import TimeoutError


class ReplicateTranscriber(AudioTranscriber):
    def __init__(self, model: str):
        self._model_name = model
        self.token = os.getenv("REPLICATE_API_TOKEN")
        if not self.token:
            raise ValueError("Missing REPLICATE_API_TOKEN environment variable.")
        self.base_url = os.getenv("REPLICATE_BASE_URL", "https://api.replicate.com/v1")

    @property
    def model_name(self) -> str:
        return self._model_name

    async def transcribe(self, audio_bytes: bytes, mime: str) -> str:
        try:
            return await self._transcribe(audio_bytes, mime)
        except Exception as e:
            logger.exception("Failed to transcribe audio (replicate)", extra={"error": str(e)})
            return ""

    @backoff(exception=(ClientError, TimeoutError), tries=3, delay=0.5, backoff=2)
    async def _transcribe(self, audio_bytes: bytes, mime: str) -> str:
        headers = {
            "Authorization": f"Token {self.token}"
        }

        form = aiohttp.FormData()
        form.add_field("file", audio_bytes, filename="audio.wav", content_type=mime)
        form.add_field("version", self._model_name)

        async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=60)) as session:
            async with session.post(f"{self.base_url}/predictions", headers=headers, data=form) as resp:
                resp.raise_for_status()
                json_data = await resp.json()
                text = json_data.get("transcription", "")
                logger.debug("Audio transcribed (replicate)", extra={"length": len(text)})
                return text
