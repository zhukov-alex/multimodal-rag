import os
import aiohttp
from typing import AsyncGenerator
from asyncio import TimeoutError
from aiohttp.client_exceptions import ClientError, ClientResponseError
import json

from multimodal_rag.generator.params.openai import OpenAIParams
from multimodal_rag.generator.types import Generator, GenerateRequest
from multimodal_rag.utils.retry import backoff
from multimodal_rag.utils.token_limit import validate_token_limit


class OpenAIGenerator(Generator):
    def __init__(self, model: str, context_limit: int | None = None):
        self.model = model
        self.context_limit = context_limit
        self.api_key = os.getenv("OPENAI_API_KEY")
        self.base_url = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")
        if not self.api_key:
            raise ValueError("Missing OPENAI_API_KEY")

    async def generate(self, request) -> str:
        if not isinstance(request.params, OpenAIParams):
            raise TypeError("Expected OpenAI-compatible generation params")

        headers = self._headers()
        payload = request.prompt_builder.build(request, self.model)
        validate_token_limit(payload, self.model, self.context_limit)
        payload["stream"] = False

        async with aiohttp.ClientSession(headers=headers) as session:
            return await self._call_api(session, payload)

    async def generate_stream(self, request: GenerateRequest) -> AsyncGenerator[str, None]:
        if not isinstance(request.params, OpenAIParams):
            raise TypeError("Expected OpenAI-compatible generation params")

        headers = self._headers()
        payload = request.prompt_builder.build(request, self.model)
        validate_token_limit(payload, self.model)
        payload["stream"] = True

        async with aiohttp.ClientSession(headers=headers) as session:
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
                            delta = data["choices"][0].get("delta", {})
                            if "content" in delta:
                                yield delta["content"]
                        except Exception:
                            continue

    def _headers(self) -> dict:
        return {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

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
            try:
                data = json.loads(text)
                if data.get("error", {}).get("code") == "context_length_exceeded":
                    raise ValueError("Token limit exceeded") from e
            except json.JSONDecodeError:
                raise
            raise
