import os
import aiohttp
from multimodal_rag.preprocessor.transcriber.types import AudioTranscriber
from multimodal_rag.log_config import logger
from multimodal_rag.utils.retry import backoff
from aiohttp import ClientError
from asyncio import TimeoutError


class CustomAudioTranscriber(AudioTranscriber):
    """
    Audio transcribing using a local server API.
    """

    def __init__(self, model: str):
        self._model_name = model
        self.base_url = os.getenv("CUSTOM_TRANSCRIBER_BASE_URL", "http://localhost:5100")

    @property
    def model_name(self) -> str:
        return self._model_name

    async def transcribe(self, audio_bytes: bytes, mime: str) -> str:
        try:
            return await self._transcribe(audio_bytes, mime)
        except Exception as e:
            logger.exception("Failed to transcribe audio", extra={"url": self.base_url, "error": str(e)})
            return ""

    @backoff(exception=(ClientError, TimeoutError), tries=3, delay=0.5, backoff=2)
    async def _transcribe(self, audio_bytes: bytes, mime: str) -> str:
        form = aiohttp.FormData()
        form.add_field("file", audio_bytes, filename="audio.wav", content_type=mime)
        form.add_field("model_name", self._model_name)

        async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=60)) as session:
            async with session.post(f"{self.base_url}/transcribe", data=form) as resp:
                resp.raise_for_status()
                json_data = await resp.json()
                text = json_data.get("text", "")
                logger.debug("Audio transcribed", extra={"length": len(text)})
                return text
