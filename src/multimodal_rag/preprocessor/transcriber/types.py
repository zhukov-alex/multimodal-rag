from typing import Protocol


class AudioTranscriber(Protocol):
    """
    Interface for audio transcribing API.
    """
    async def transcribe(self, audio_bytes: bytes, mime: str) -> str:
        """
        Transcribe audio and return the resulting text.
        """
        ...

    def model_name(self) -> str:
        ...
