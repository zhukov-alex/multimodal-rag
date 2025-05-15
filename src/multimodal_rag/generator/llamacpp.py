import os
import aiohttp
import json
from typing import AsyncGenerator
from asyncio import TimeoutError
from aiohttp.client_exceptions import ClientError, ClientResponseError

from multimodal_rag.generator.types import Generator, GenerateRequest
from multimodal_rag.generator.params.llamacpp import LlamaCppParams
from multimodal_rag.utils.retry import backoff


class LlamaCppGenerator(Generator):
    def __init__(self, model: str, context_limit: int | None = None):
        self.model = model
        self.context_limit = context_limit
        self.base_url = os.getenv("LLAMACPP_BASE_URL", "http://localhost:8080/v1")

    async def generate(self, request) -> str:
        if not isinstance(request.params, LlamaCppParams):
            raise TypeError("Expected llama.cpp-compatible generation params")

        payload = request.prompt_builder.build(request, self.model)

        async with aiohttp.ClientSession() as session:
            return await self._call_api(session, payload)

    async def generate_stream(self, request: GenerateRequest) -> AsyncGenerator[str, None]:
        if not isinstance(request.params, LlamaCppParams):
            raise TypeError("Expected llama.cpp-compatible generation params")

        payload = request.prompt_builder.build(request, self.model)
        payload["stream"] = True

        async with aiohttp.ClientSession() as session:
            async with session.post(f"{self.base_url}/chat/completions", json=payload) as response:
                await self._handle_response_errors(response)

                buffer = b""
                async for chunk in response.content.iter_chunked(1024):
                    buffer += chunk
                    for line in buffer.split(b"\n"):
                        if not line.startswith(b"data: "):
                            continue
                        raw = line.removeprefix(b"data: ").strip()
                        if raw == b"[DONE]":
                            break
                        try:
                            data = json.loads(raw)
                            yield data["choices"][0]["message"]["content"]
                        except Exception:
                            continue

    @backoff(exception=(ClientError, TimeoutError))
    async def _call_api(self, session: aiohttp.ClientSession, payload: dict) -> str:
        async with session.post(f"{self.base_url}/chat/completions", json=payload) as response:
            await self._handle_response_errors(response)
            data = await response.json()
            return data["choices"][0]["message"]["content"]

    async def _handle_response_errors(self, response: aiohttp.ClientResponse):
        try:
            response.raise_for_status()
        except ClientResponseError as e:
            text = await response.text()
            lowered = text.lower()
            if (
                "context" in lowered
                or "too many tokens" in lowered
                or "max context length" in lowered
                or "maximum context" in lowered
            ):
                raise ValueError("Token limit exceeded") from e
            raise
