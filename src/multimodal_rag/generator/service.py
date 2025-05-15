from asyncio import TimeoutError
from typing import AsyncGenerator
from aiohttp import ClientError

from multimodal_rag.generator.types import GenerateRequest
from multimodal_rag.generator.types import Generator
from multimodal_rag.log_config import logger


class GeneratorService:
    def __init__(self, generator: Generator):
        self.generator = generator

    async def generate(self, request: GenerateRequest) -> str:
        try:
            result = await self.generator.generate(request)
        except (ClientError, TimeoutError, ValueError) as e:
            logger.exception("Failed to generate response", extra={
                "error": str(e),
                "query": request.query[:100]
            })
            raise
        except Exception as e:
            logger.exception("Unexpected generation error", extra={
                "error": str(e),
                "query": request.query[:100]
            })
            raise

        return result

    async def generate_stream(self, request: GenerateRequest) -> AsyncGenerator[str, None]:
        try:
            stream = await self.generator.generate_stream(request)
            async for chunk in stream:
                yield chunk

        except (ClientError, TimeoutError, ValueError) as e:
            logger.exception("Failed to stream response", extra={
                "error": str(e),
                "query": request.query[:100]
            })
            raise
        except Exception as e:
            logger.exception("Unexpected streaming error", extra={
                "error": str(e),
                "query": request.query[:100]
            })
            raise
