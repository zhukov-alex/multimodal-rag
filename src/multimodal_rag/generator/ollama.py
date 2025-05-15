import os
import aiohttp
import json
from typing import AsyncGenerator
from asyncio import TimeoutError
from aiohttp.client_exceptions import ClientError, ClientResponseError

from multimodal_rag.generator.params.ollama import OllamaParams
from multimodal_rag.generator.types import Generator, GenerateRequest
from multimodal_rag.utils.retry import backoff
from multimodal_rag.log_config import logger


class OllamaGenerator(Generator):
    def __init__(self, model: str, context_limit: int | None = None):
        self.model = model
        self.context_limit = context_limit
        self.base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")

    async def generate(self, request) -> str:
        if not isinstance(request.params, OllamaParams):
            raise TypeError("Expected Ollama-compatible generation params")

        payload = request.prompt_builder.build(request, self.model)
        payload["stream"] = False

        async with aiohttp.ClientSession() as session:
            return await self._call_api(session, payload)

    async def generate_stream(self, request: GenerateRequest) -> AsyncGenerator[str, None]:
        if not isinstance(request.params, OllamaParams):
            raise TypeError("Expected Ollama-compatible generation params")

        payload = request.prompt_builder.build(request, self.model)
        payload["stream"] = True
        query_preview = request.query[:100]

        async with aiohttp.ClientSession() as session:
            async with session.post(f"{self.base_url}/api/generate", json=payload) as response:
                await self._handle_response_errors(response)
                async for line in response.content:
                    if not line.strip():
                        continue
                    try:
                        data = json.loads(line)
                        if "response" in data:
                            yield data["response"]
                        if data.get("done"):
                            logger.debug("Ollama client [stream-done]", extra={
                                "query": query_preview,
                                "total_duration": data.get("total_duration")
                            })
                            break
                    except json.JSONDecodeError:
                        continue

    @backoff(exception=(ClientError, TimeoutError))
    async def _call_api(self, session: aiohttp.ClientSession, payload: dict) -> str:
        async with session.post(f"{self.base_url}/api/generate", json=payload) as response:
            await self._handle_response_errors(response)
            data = await response.json()
            if "total_duration" in data:
                logger.debug("Ollama client [generate-done]", extra={
                    "query": payload.get("prompt", "")[:100],
                    "total_duration": data.get("total_duration")
                })
            return data["response"]

    async def _handle_response_errors(self, response: aiohttp.ClientResponse):
        try:
            response.raise_for_status()
        except ClientResponseError as e:
            text = await response.text()
            try:
                data = json.loads(text)
                error_msg = data.get("error", "").lower()
                if "too many tokens" in error_msg or "token limit" in error_msg:
                    raise ValueError("Token limit exceeded") from e
            except json.JSONDecodeError:
                pass
            raise
